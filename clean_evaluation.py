import torch
import numpy as np
import os
from tqdm import tqdm
from inference import model_factory, load_model_and_config, calculate_lsd_unified
from model_paths import MULTI_MODEL_PATHS
from fm_utils import ATF3DSampler, EulerSimulator

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_reference_results(results_dir, method_name):
    """Load pre-computed reference results from RESULTS folder."""
    pt_path = f'{results_dir}/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt'
    
    if not os.path.exists(pt_path):
        print(f"Warning: {method_name} results not found at {pt_path}")
        return None
    
    atf_mag_est = torch.load(pt_path, weights_only=False)
    print(f"{method_name} results loaded: {atf_mag_est.shape} (Mic, Freq, Src)")
    return atf_mag_est

def load_ground_truth():
    """Load ground truth data using the same approach as unified_evaluation.py."""
    import sys
    sys.path.append('AUTOENCODER/src')
    import AUTOENCODER.src.dataset as autoencoder_dataset
    from AUTOENCODER.src.configs import config_FSMPAE_10026
    
    config = config_FSMPAE_10026.copy()
    
    # Change to AUTOENCODER directory
    original_cwd = os.getcwd()
    os.chdir('AUTOENCODER')
    
    # Load dataset
    idataset = autoencoder_dataset.ATFdataset(config=config)
    data = idataset.Data
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    # Get ground truth
    dataset_name = config['dataset'][0]
    atf_mag_gt = data['test']['atf_mag'][dataset_name]
    
    print(f"Ground truth loaded: {atf_mag_gt.shape} (Mic, Freq, Src)")
    return atf_mag_gt, config

def evaluate_reference_method(atf_mag_est, atf_mag_gt, method_name, freq_up_to, num_sources=102):
    """Evaluate a reference method (FSMPAE or KRR) using M=5 microphone selection strategy.
    
    Note: Reference files only contain M=5 results, so we only evaluate for M=5.
    """
    print(f"Evaluating {method_name}...")
    
    # Load microphone selection matrix
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    if not os.path.exists(idx_mes_pos_path):
        print(f"Warning: Microphone selection matrix not found at {idx_mes_pos_path}")
        return {}
    
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    print(f"Loaded microphone selection matrix: {idx_mes_pos_mat.shape}")
    
    # Reference methods only have M=5 case
    M = 5
    print(f"  Evaluating with M={M} microphones (reference only supports M=5)...")
    
    lsd_scores = []
    mse_scores = []
    nmse_scores = []
    
    for src_idx in tqdm(range(num_sources), desc=f"{method_name} M={M}"):
        # Evaluate full reconstruction quality (all 1331 mics)
        # The reference method used M=5 observations to generate these full predictions
        # Use frequency range matching your model (truncated)
        pred_full = atf_mag_est[:, :freq_up_to, src_idx]  # [1331, freq]
        gt_full = atf_mag_gt[:, :freq_up_to, src_idx]     # [1331, freq]
        
        # Calculate LSD (frequency is dim=1)
        lsd_val = calculate_lsd_unified(pred_full, gt_full, freq_dim=1)
        lsd_scores.append(lsd_val.item())
        
        # Calculate MSE
        mse_val = torch.mean((pred_full - gt_full) ** 2).item()
        mse_scores.append(mse_val)
        
        # Calculate NMSE (in dB)
        gt_var = torch.var(gt_full).item()
        nmse_linear = mse_val / gt_var if gt_var > 0 else float('inf')
        nmse_val = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')
        nmse_scores.append(nmse_val)
    
    # Return results only for M=5
    results = {
        5: {
            'lsd_mean': np.mean(lsd_scores),
            'lsd_std': np.std(lsd_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'nmse_mean': np.mean(nmse_scores),
            'nmse_std': np.std(nmse_scores),
            'num_sources': num_sources
        }
    }
    
    return results

def evaluate_your_model(set_encoder, ode_3d, config, M_values, device, model_name, num_sources=102, guidance_scale=1.0):
    """Evaluate your 3D model."""
    print(f"Evaluating {model_name}...")
    
    data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    
    # Load data
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    simulator.ode.guidance_scale = guidance_scale
    
    # Load microphone selection matrix
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    
    results = {}
    
    for M in M_values:
        print(f"  Evaluating with M={M} microphones...")
        
        lsd_scores = []
        mse_scores = []
        nmse_scores = []
        
        for src_idx in tqdm(range(num_sources), desc=f"{model_name} M={M}"):
            with torch.no_grad():
                z_true = test_sampler.cubes[src_idx].unsqueeze(0).to(device)
                src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)
                
                # Use source-specific microphones
                source_specific_indices = idx_mes_pos_mat[:M, src_idx]  # First M mics for this source
                obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)
                
                obs_xyz_abs = grid_xyz[obs_indices]
                obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)
                
                z_flat = z_true.view(z_true.shape[1], -1)
                obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
                obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
                
                # Inference
                x0 = torch.randn_like(z_true)
                y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)
                
                ts = torch.linspace(0, 1, 11, device=device)
                ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)
                
                z_est = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                         obs_mask=obs_mask, pooled_context=pooled_context,
                                         paste_observations=True, obs_indices=obs_indices)
                
                # Denormalize
                z_est_denorm = z_est * spec_std + train_sampler.mean.item()
                z_true_denorm = z_true * spec_std + train_sampler.mean.item()
                
                # Calculate LSD directly on denormalized data
                lsd_val = calculate_lsd_unified(z_est_denorm.squeeze(0), z_true_denorm.squeeze(0), freq_dim=0)
                lsd_scores.append(lsd_val.item())
                
                # Calculate MSE
                mse_val = torch.mean((z_est_denorm - z_true_denorm) ** 2).item()
                mse_scores.append(mse_val)
                
                # Calculate NMSE (in dB)
                z_true_var = torch.var(z_true_denorm).item()
                nmse_linear = mse_val / z_true_var if z_true_var > 0 else float('inf')
                nmse_val = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')
                nmse_scores.append(nmse_val)
        
        results[M] = {
            'lsd_mean': np.mean(lsd_scores),
            'lsd_std': np.std(lsd_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'nmse_mean': np.mean(nmse_scores),
            'nmse_std': np.std(nmse_scores),
            'num_sources': num_sources
        }
    
    return results

def get_model_name_from_path(model_path):
    """Extract model name from path."""
    dir_name = model_path.split("artifacts/")[1].split("/")[0]
    filename = os.path.basename(model_path).replace('.pt', '')
    
    if filename == "model":
        return dir_name
    else:
        return f"{dir_name}_{filename}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M_values = [5]  # Default evaluation with 5 microphones (can be extended to [5, 10, 20, 50, 100] etc.)
    guidance_scale = 1.0  # Default guidance scale
    num_sources = 102  # All test sources
    
    print("=== CLEAN EVALUATION SCRIPT ===")
    print(f"Device: {device}")
    print(f"M_values: {M_values}")
    print(f"Sources: {num_sources}")
    print()
    
    # Load ground truth
    print("1. Loading ground truth data...")
    atf_mag_gt, ref_config = load_ground_truth()
    
    # Load reference methods
    print("\n2. Loading reference methods...")
    fsmpae_results = load_reference_results("RESULTS/out_20250323_FSMPAE_10026", "FSMPAE")
    krr_results = load_reference_results("RESULTS/out_20250324_KRR_10004", "KRR")
    
    # Load your models
    print("\n3. Loading your models...")
    all_results = {}
    freq_up_to = None
    
    # Evaluate reference methods first to get frequency range
    if len(MULTI_MODEL_PATHS) > 0:
        # Load first model to get frequency range
        checkpoint, config, model_states_cfg = load_model_and_config(MULTI_MODEL_PATHS[0], device)
        freq_up_to = config['model'].get('freq_up_to')
        print(f"Using frequency range: {freq_up_to} bins")
    
    # Evaluate reference methods
    if fsmpae_results is not None:
        fsmpae_eval = evaluate_reference_method(fsmpae_results, atf_mag_gt, "FSMPAE", freq_up_to, num_sources)
        all_results["FSMPAE"] = fsmpae_eval
    
    if krr_results is not None:
        krr_eval = evaluate_reference_method(krr_results, atf_mag_gt, "KRR", freq_up_to, num_sources)
        all_results["KRR"] = krr_eval
    
    # Evaluate your models
    for i, model_path in enumerate(MULTI_MODEL_PATHS):
        model_name = get_model_name_from_path(model_path)
        print(f"\nLoading model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")

        checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)
        set_encoder, unet_3d, ode_3d, is_new_model = model_factory(config, model_states_cfg, device)

        model_results = evaluate_your_model(set_encoder, ode_3d, config, M_values, device, model_name, num_sources, guidance_scale)
        all_results[model_name] = model_results
    
    # Print results table
    print("\n" + "="*100)
    print("=== EVALUATION RESULTS ===")
    print("="*100)
    print(f"Frequency range: 0-{freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz ({freq_up_to} bins)")
    print(f"Sources evaluated: {num_sources}")
    print(f"Evaluation: Full reconstruction quality (all 1331 mics) from M={M_values[0]} observations")
    print("-"*100)
    print(f"{'Method':<40} | {'M':<3} | {'LSD (dB)':<12} | {'MSE':<12} | {'NMSE (dB)':<12}")
    print("-"*100)
    
    for method_name, method_results in all_results.items():
        for M in M_values:
            if M in method_results:
                results = method_results[M]
                lsd_str = f"{results['lsd_mean']:.4f} ± {results['lsd_std']:.4f}"
                mse_str = f"{results['mse_mean']:.4f} ± {results['mse_std']:.4f}"
                nmse_str = f"{results['nmse_mean']:.4f} ± {results['nmse_std']:.4f}"
                
                # Truncate long method names
                display_name = method_name[:37] + "..." if len(method_name) > 40 else method_name
                
                print(f"{display_name:<40} | {M:<3} | {lsd_str:<12} | {mse_str:<12} | {nmse_str:<12}")
            else:
                # N/A result for mismatched M_values (e.g., reference methods don't have other M values)
                display_name = method_name[:37] + "..." if len(method_name) > 40 else method_name
                print(f"{display_name:<40} | {M:<3} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")
    
    print("="*100)
    print("Note: All methods use M=5 sparse observations to reconstruct full 1331-mic sound fields.")
    print("      Evaluation measures reconstruction quality across all microphone positions.")
    print("      Frequency range truncated to match your model for fair comparison.")

if __name__ == "__main__":
    main()
