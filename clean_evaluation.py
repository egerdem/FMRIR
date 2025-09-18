import torch
import numpy as np
import os
from tqdm import tqdm
from inference import model_factory, load_model_and_config, calculate_lsd_unified
from model_paths import MULTI_MODEL_PATHS
from fm_utils import ATF3DSampler, EulerSimulator

# ===== CONFIGURATION CONSTANTS =====
SEED = 42

# Data paths
DATA_DIR = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
AUTOENCODER_PATH = "AUTOENCODER/src"
MIC_SELECTION_PATH = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"

# Reference model result paths
FSMPAE_RESULTS_PATH = "RESULTS/out_20250916_EEAE_10001/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200_freq20.pt"
KRR_RESULTS_PATH = "RESULTS/out_20250324_KRR_10004/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt"

# Evaluation parameters
M_VALUES = [5, 20, 50]  # Number of microphones to evaluate
GUIDANCE_SCALE = 1.0  # Guidance scale for your model
NUM_TIMESTEPS = 10  # Number of timesteps for generation
NUM_SOURCES = 102  # Number of test sources to evaluate (None for all)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_reference_results():
    """Load pre-computed reference results."""
    fsmpae_results = None
    krr_results = None
    
    if os.path.exists(FSMPAE_RESULTS_PATH):
        fsmpae_results = torch.load(FSMPAE_RESULTS_PATH, weights_only=False)
        print(f"FSMPAE results loaded: {fsmpae_results.shape}")
    else:
        assert False, f"FSMPAE results not found at {FSMPAE_RESULTS_PATH}"
        
    if os.path.exists(KRR_RESULTS_PATH):
        krr_results = torch.load(KRR_RESULTS_PATH, weights_only=False)
        print(f"KRR results loaded: {krr_results.shape}")
    else:
        assert False, f"KRR results not found at {KRR_RESULTS_PATH}"
    
    return fsmpae_results, krr_results

def load_ground_truth():
    """Load ground truth data using the same approach as unified_evaluation.py."""
    import sys
    sys.path.append(AUTOENCODER_PATH)
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

def evaluate_reference_method(atf_mag_est, atf_mag_gt, method_name, freq_up_to):
    """Evaluate a reference method (FSMPAE or KRR) using M=5 microphone selection strategy.
    
    Note: Reference files only contain M=5 results, so we only evaluate for M=5.
    """
    print(f"Evaluating {method_name}...")
    
    # Load microphone selection matrix
    if not os.path.exists(MIC_SELECTION_PATH):
        assert False, f"Microphone selection matrix not found at {MIC_SELECTION_PATH}"
    
    idx_mes_pos_mat = np.load(MIC_SELECTION_PATH)
    print(f"Loaded microphone selection matrix: {idx_mes_pos_mat.shape}")
    
    # Reference methods only have M=5 case
    M = M_VALUES[0]  # Use first M value from config
    print(f"  Evaluating with M={M} microphones...")
    num_sources = NUM_SOURCES
    
    # Check if we need to match frequency dimensions
    pred_freq_bins = atf_mag_est.shape[1]
    print(f"  Original Model frequency bins: {pred_freq_bins}")
    print(f"  Using frequency bins up to: {freq_up_to}")
    use_freq_bins = freq_up_to
    
    lsd_scores = []
    mse_scores = []
    nmse_scores = []
    
    for src_idx in tqdm(range(num_sources), desc=f"{method_name} M={M}"):
        # Get predictions and ground truth, properly sliced
        pred_full = atf_mag_est[:, :use_freq_bins, src_idx]  # [1331, freq]
        gt_full = atf_mag_gt[:, :use_freq_bins, src_idx]    # [1331, freq]
        
        # Calculate LSD per position first (don't average across positions)
        squared_error = (pred_full - gt_full) ** 2
        lsd_per_position = torch.sqrt(torch.mean(squared_error, dim=1))  # [1331] positions
        
        # Store all position-wise LSD values
        if src_idx == 0:
            all_lsd_values = lsd_per_position  # [1331]
        else:
            all_lsd_values = torch.cat([all_lsd_values, lsd_per_position])  # [src_idx * 1331]
        
        # Calculate mean for this source (for progress tracking)
        lsd_val = torch.mean(lsd_per_position).item()
        lsd_scores.append(lsd_val)
        
        # Calculate MSE
        mse_val = torch.mean((pred_full - gt_full) ** 2).item()
        mse_scores.append(mse_val)
        
        # Calculate NMSE (in dB)
        gt_var = torch.var(gt_full).item()
        nmse_linear = mse_val / gt_var if gt_var > 0 else float('inf')
        nmse_val = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')
        nmse_scores.append(nmse_val)
    
    # Calculate final statistics across all positions and sources
    final_lsd_mean = torch.mean(all_lsd_values).item()
    final_lsd_std = torch.std(all_lsd_values).item()
    
    print(f"\nFinal LSD stats across all positions and sources:")
    print(f"  Mean: {final_lsd_mean:.4f}")
    print(f"  Std: {final_lsd_std:.4f}")
    
    # Return results only for M=5
    results = {
        5: {
            'lsd_mean': final_lsd_mean,
            'lsd_std': final_lsd_std,
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'nmse_mean': np.mean(nmse_scores),
            'nmse_std': np.std(nmse_scores),
            'num_sources': num_sources
        }
    }
    
    return results

def evaluate_your_model(set_encoder, ode_3d, config, device, model_name):
    """Evaluate your 3D model."""
    print(f"Evaluating {model_name}...")
    
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    num_sources = NUM_SOURCES
    # Load data
    train_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    simulator.ode.guidance_scale = GUIDANCE_SCALE
    
    # Load microphone selection matrix
    idx_mes_pos_mat = np.load(MIC_SELECTION_PATH)
    
    results = {}
    
    for M in M_VALUES:
        print(f"  Evaluating with M={M} microphones...")
        
        mse_scores = []
        nmse_scores = []

        all_lsd_values_list = []

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
                
                # Calculate LSD using the unified function
                # Reshape to [positions, freq] for consistency
                C = z_est_denorm.shape[1]
                z_est_squeezed = z_est_denorm.squeeze(0)  # Shape: [C, D, H, W]
                z_est_flat = z_est_squeezed.permute(1, 2, 3, 0).reshape(-1, C)  # Shape: [1331, 20]

                z_true_squeezed = z_true_denorm.squeeze(0)
                z_true_flat = z_true_squeezed.permute(1, 2, 3, 0).reshape(-1, C)
                
                _, lsd_per_position = calculate_lsd_unified(
                    z_est_flat, z_true_flat, 
                    freq_dim=1, return_mean_only=False
                )

                # Store position-wise LSD values for std calculation
                all_lsd_values_list.append(lsd_per_position)

                # Store the mean for this source
                # Calculate MSE
                mse_val = torch.mean((z_est_denorm - z_true_denorm) ** 2).item()
                mse_scores.append(mse_val)

                # Calculate MSE
                mse_val = torch.mean((z_est_denorm - z_true_denorm) ** 2).item()
                mse_scores.append(mse_val)
                
                # Calculate NMSE (in dB)
                z_true_var = torch.var(z_true_denorm).item()
                nmse_linear = mse_val / z_true_var if z_true_var > 0 else float('inf')
                nmse_val = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')
                nmse_scores.append(nmse_val)

        # ===== CHANGE 2: CONSISTENT MEAN/STD CALCULATION =====
        # Concatenate all position-wise LSDs from the list into a single tensor
        all_lsd_values = torch.cat(all_lsd_values_list)

        # Calculate final statistics from the complete tensor of position-wise LSDs
        final_lsd_mean = torch.mean(all_lsd_values).item()
        final_lsd_std = torch.std(all_lsd_values).item()

        print(f"\nFinal LSD stats (from all positions):")
        print(f"  Mean: {final_lsd_mean:.4f}")
        print(f"  Std: {final_lsd_std:.4f}")
        
        results[M] = {
            'lsd_mean': final_lsd_mean,
            'lsd_std': final_lsd_std,
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
    
    print("=== CLEAN EVALUATION SCRIPT ===")
    print(f"Device: {device}")
    print(f"M values: {M_VALUES}")
    print(f"Sources: {NUM_SOURCES}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print()
    
    # Load ground truth
    print("1. Loading ground truth data...")
    atf_mag_gt, ref_config = load_ground_truth()
    
    # Load reference methods
    print("\n2. Loading reference methods...")
    fsmpae_results, krr_results = load_reference_results()
    
    # Load your models
    print("\n3. Loading your models...")
    all_results = {}

    # Evaluate reference methods first to get frequency range
    freq_up_to = 20
    
    # Evaluate reference methods
    fsmpae_eval = evaluate_reference_method(fsmpae_results, atf_mag_gt, "FSMPAE", freq_up_to)
    all_results["FSMPAE"] = fsmpae_eval
    
    krr_eval = evaluate_reference_method(krr_results, atf_mag_gt, "KRR", freq_up_to)
    all_results["KRR"] = krr_eval
    
    # Evaluate your models
    for i, model_path in enumerate(MULTI_MODEL_PATHS):
        model_name = get_model_name_from_path(model_path)
        print(f"\nLoading model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")

        checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)
        set_encoder, unet_3d, ode_3d, is_new_model = model_factory(config, model_states_cfg, device)

        model_results = evaluate_your_model(set_encoder, ode_3d, config, device, model_name)
        all_results[model_name] = model_results
    
    # Print results table
    print("\n" + "="*100)
    print("=== EVALUATION RESULTS ===")
    print("="*100)
    print(f"Frequency range: 0-{freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz ({freq_up_to} bins)")
    print(f"Sources evaluated: {NUM_SOURCES}")
    print(f"Evaluation: Full reconstruction quality (all 1331 mics) from M={M_VALUES[0]} observations")
    print("-"*100)
    print(f"{'Method':<40} | {'M':<3} | {'LSD (dB)':<12} | {'MSE':<12} | {'NMSE (dB)':<12}")
    print("-"*100)
    
    for method_name, method_results in all_results.items():
        for M in M_VALUES:
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
