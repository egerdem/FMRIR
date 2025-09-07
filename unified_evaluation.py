import torch
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # Same as eval_AUTOENCODER.py
from matplotlib import pyplot as plt

# Your model imports
from fm_utils import (
    ATF3DSampler, SetEncoder, 
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, 
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator
)

# Reference model imports
import sys
sys.path.append('AUTOENCODER/src')
import AUTOENCODER.src.dataset as autoencoder_dataset
from AUTOENCODER.src.configs import config_FSMPAE_10026
import AUTOENCODER.src.utils as autoencoder_utils

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def calculate_lsd_unified(estimation, ground_truth, freq_dim=1):
    """
    Unified LSD calculation that works for both 3D spatial and microphone-based data.
    
    Args:
        estimation: Model prediction
        ground_truth: Ground truth
        freq_dim: Dimension along which frequency is stored
    
    Returns:
        LSD value in dB
    """
    squared_error = (estimation - ground_truth) ** 2
    lsd_per_position = torch.sqrt(torch.mean(squared_error, dim=freq_dim))
    return torch.mean(lsd_per_position)


def load_your_model(model_path, device):
    """Load your 3D Flow Matching model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']
    
    model_cfg = config['model']
    architecture = model_cfg.get('architecture_version')
    
    # Load models
    set_encoder = SetEncoder(
        num_freqs=model_cfg['freq_up_to'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)
    
    if architecture == "v2_residual_context":
        unet_3d = CrossAttentionUNet3D_RED3d(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D_V2(unet=unet_3d, set_encoder=set_encoder)
    else:
        unet_3d = CrossAttentionUNet3D(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)
    
    # Load weights
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()
    
    return set_encoder, unet_3d, ode_3d, config


def load_reference_model(device, freq_up_to):
    """Load the reference AUTOENCODER model data and predictions."""
    # Use the exact same config as in eval_AUTOENCODER.py (no modifications!)
    config = config_FSMPAE_10026.copy()
    
    try:
        # Change to AUTOENCODER directory so dataset loading works correctly
        import os
        original_cwd = os.getcwd()
        os.chdir('AUTOENCODER')
        
        # Load dataset (this will use ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/ in AUTOENCODER dir)
        idataset = autoencoder_dataset.ATFdataset(config=config)
        data = idataset.Data
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        # Load model predictions
        pt_dir = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026'
        dataset_name = config['dataset'][0]
        pt_path = f'{pt_dir}/atf_mag/atf_mag_test_{dataset_name}.pt'
        
        if not os.path.exists(pt_path):
            print(f"Warning: Reference model predictions not found at {pt_path}")
            return None, None, None, None
        
        atf_mag_est = torch.load(pt_path, weights_only=False)
        atf_mag_gt = data['test']['atf_mag'][dataset_name]
        
        # Truncate to match your model's frequency range for fair comparison
        atf_mag_est_truncated = atf_mag_est[:, :freq_up_to, :]
        atf_mag_gt_truncated = atf_mag_gt[:, :freq_up_to, :]
        
        print(f"Reference data loaded: {atf_mag_gt_truncated.shape} (Mic, Freq, Src)")
        print(f"Using first {freq_up_to} frequency bins for comparison")
        
        return atf_mag_est_truncated, atf_mag_gt_truncated, config, data
        
    except Exception as e:
        print(f"Error loading reference model: {e}")
        print("Using pre-computed reference results instead...")
        return None, None, None, None


def evaluate_your_model(set_encoder, unet_3d, ode_3d, config, M_values, device, num_sources_eval=None):
    """Evaluate your 3D model."""
    data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    
    # Load data
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    # Limit evaluation to first N sources if specified
    total_sources = len(test_sampler)
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    results = {}
    
    for M in M_values:
        print(f"Evaluating your model with M={M} microphones...")
        lsd_scores = []
        
        # Load the SAME microphone selection strategy as reference model
        idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
        if os.path.exists(idx_mes_pos_path):
            idx_mes_pos_mat = np.load(idx_mes_pos_path)
            print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
            print("Using source-specific microphone selection (different M=5 mics per source)")

        for i in tqdm(range(eval_sources), desc=f"Your Model M={M}"):
            with torch.no_grad():
                z_true = test_sampler.cubes[i].unsqueeze(0).to(device)
                src_xyz = test_sampler.source_coords[i].unsqueeze(0).to(device)
                
                # Use source-specific microphones (different M=5 for each source)
                source_specific_indices = idx_mes_pos_mat[:M, i]  # First M mics for this source
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
                
                simulator.ode.guidance_scale = 1.0
                z_est = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                         obs_mask=obs_mask, pooled_context=pooled_context,
                                         paste_observations=True, obs_indices=obs_indices)
                
                # Calculate LSD
                lsd_normalized = calculate_lsd_unified(z_est.squeeze(0), z_true.squeeze(0), freq_dim=0)
                lsd_db = lsd_normalized.item() * spec_std
                lsd_scores.append(lsd_db)
        
        results[M] = {
            'mean': np.mean(lsd_scores), 
            'std': np.std(lsd_scores),
            'num_sources_eval': eval_sources
        }
    
    return results, idx_mes_pos_mat


def evaluate_reference_model(atf_mag_est, atf_mag_gt, ref_config, num_sources_eval=None, your_freq_up_to=None):
    """Evaluate the reference AUTOENCODER model using the loaded data."""
    print("Evaluating reference AUTOENCODER model...")
    
    # Get the M value used by reference model
    dataset_name = ref_config['dataset'][0]
    ref_M = ref_config['num_mes_test']  # This is 5 for FSMPAE_10026
    print(f"Reference model uses M={ref_M} microphones")
    
    # Limit evaluation to first N sources if specified
    total_sources = atf_mag_gt.shape[2]
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
        print(f"Evaluating on all {eval_sources} sources")
    
    # Calculate LSD for full frequency range (64 bins)
    lsd_per_sample_full = []
    # Calculate LSD for same frequency range as your model (first 20 bins)
    lsd_per_sample_matched = []
    
    for src_idx in tqdm(range(eval_sources), desc="Reference Model"):
        # Full frequency range LSD
        lsd_val_full = calculate_lsd_unified(
            atf_mag_est[:, :, src_idx], 
            atf_mag_gt[:, :, src_idx], 
            freq_dim=1
        )
        lsd_per_sample_full.append(lsd_val_full.item())
        
        # Matched frequency range LSD (first your_freq_up_to bins)
        if your_freq_up_to is not None:
            lsd_val_matched = calculate_lsd_unified(
                atf_mag_est[:, :your_freq_up_to, src_idx], 
                atf_mag_gt[:, :your_freq_up_to, src_idx], 
                freq_dim=1
            )
            lsd_per_sample_matched.append(lsd_val_matched.item())
    
    result = {
        'mean': np.mean(lsd_per_sample_full), 
        'std': np.std(lsd_per_sample_full),
        'num_mics': ref_M,
        'num_sources_eval': eval_sources
    }
    
    # Add matched frequency range results if available
    if your_freq_up_to is not None and lsd_per_sample_matched:
        result['mean_matched_freq'] = np.mean(lsd_per_sample_matched)
        result['std_matched_freq'] = np.std(lsd_per_sample_matched)
        print(f"Reference LSD (full 64 bins): {result['mean']:.4f} ± {result['std']:.4f} dB")
        print(f"Reference LSD (first {your_freq_up_to} bins): {result['mean_matched_freq']:.4f} ± {result['std_matched_freq']:.4f} dB")
    
    return result


def plot_atf_comparisons(atf_mag_est_ref, atf_mag_est_yours, atf_mag_gt, ref_config, your_freq_up_to, num_sources_eval):
    """Plot ATF comparisons with 3 methods: True, Reference, Your Model for multiple combinations"""
    dataset_name = ref_config['dataset'][0]
    
    # Create frequency axes for both models
    ref_freq_bins = ref_config['num_freq']  # 64 bins
    fs = ref_config['fs']  # 2000 Hz
    
    # Reference frequency axis (0 to 1000 Hz, 64 bins)
    freq_ref = np.arange(1, ref_freq_bins + 1) / ref_freq_bins * fs / 2
    
    # Your model frequency axis (0 to ~312 Hz, 20 bins)  
    freq_yours = np.arange(1, your_freq_up_to + 1) / your_freq_up_to * fs / 2
    
    print(f"Reference freq range: 0-{freq_ref[-1]:.0f} Hz ({ref_freq_bins} bins)")
    print(f"Your model freq range: 0-{freq_yours[-1]:.0f} Hz ({your_freq_up_to} bins)")
    
    plt.rcParams["font.size"] = 18  # Same as eval_AUTOENCODER.py
    
    # Create output directory (same structure as inference_1d_atf.py)
    output_dir = "artifacts/eval/atf_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    if atf_mag_est_yours is not None:
        # Multiple source and microphone combinations (similar to inference_1d_atf.py)
        total_sources_for_plots = num_sources_eval if num_sources_eval is not None else atf_mag_gt.shape[2]
        source_indices = list(range(min(10, total_sources_for_plots)))  # Limit to 10 for plotting (can be adjusted)
        
        # Use the CORRECT microphone indices that match the PDF coordinates:
        # (-0.5,-0.5,-0.5), (0.30,-0.30,-0.30), (0.00,0.00,0.00), (-0.30,0.30,0.20), (0.50,0.50,0.50)
        mic_indices = [0, 272, 665, 937, 1330]  # Correct indices for PDF coordinates
        
        plot_count = 0
        total_plots = len(source_indices)  # One PDF per source (each with 5 subplots)
        
        print(f"Generating {total_plots} ATF comparison PDFs (5 microphones per PDF)...")
        
        # Get microphone coordinates for titles
        data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
        train_sampler = ATF3DSampler(data_path=data_path, mode='train', src_splits={'train': range(0, 820)}, 
                                   normalize=True, freq_up_to=your_freq_up_to)
        grid_xyz = train_sampler.grid_xyz
        
        for src_idx in source_indices:
            # Create one PDF with 5 subplots (like reference AUTOENCODER PDFs)
            fig, axes = plt.subplots(5, 1, figsize=(12, 6*5))
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            
            for i, mic_idx in enumerate(mic_indices):
                ax = axes[i]
                
                # Plot all three methods with correct frequency axes
                # All models plot the same frequency range for comparison (0-312 Hz)
                ax.plot(freq_yours, atf_mag_gt[mic_idx, :your_freq_up_to, src_idx], 'k--', label="True", linewidth=2)
                ax.plot(freq_yours, atf_mag_est_ref[mic_idx, :your_freq_up_to, src_idx], 'r-', label="Reference", linewidth=1.5)
                ax.plot(freq_yours, atf_mag_est_yours[mic_idx, :, src_idx], 'b-', label="Your Model", linewidth=1.5)
                
                ax.set_xscale('log')
                ax.grid(True)
                ax.legend()
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude (dB)")
                # ax.set_ylim([-50, 30])  # Same y-limits as reference
                
                # Set x-limits to show meaningful frequency range (avoid white space)
                # ax.set_xlim([freq_yours[0], freq_yours[-1]])  # From ~31 Hz to ~312 Hz
                
                # Get microphone coordinates for title (same format as reference)
                mic_coord = grid_xyz[mic_idx].numpy()
                ax.set_title(f"ATF ({mic_coord[0]:.2f} m, {mic_coord[1]:.2f} m, {mic_coord[2]:.2f} m)")
            
            plt.tight_layout()
            
            # Save with source-specific filename (like reference: ATF_Mag_..._src-XX_test.pdf)
            filename = f"ATF_Comparison_src{src_idx+922:04d}_test.pdf"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close to save memory
            
            plot_count += 1
            print(f"Saved {plot_count}/{len(source_indices)} plots: {filename}")
        
        print(f"All {len(source_indices)} ATF comparison PDFs saved to {output_dir}/")
    else:
        print("Your model predictions not available - skipping ATF plots")


def get_your_model_atf_predictions(set_encoder, ode_3d, config, device, atf_mag_gt, ref_config, your_freq_up_to, num_sources_eval):
    """
    Extract ATF predictions from your model in the same format as reference model.
    Based on inference_1d_atf.py approach.
    """
    print("Generating ATF predictions from your 3D model...")
    
    # Load your data (same as in inference_1d_atf.py)
    data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    
    # Load normalized data
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=your_freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=your_freq_up_to
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    # Create simulator
    simulator = EulerSimulator(ode=ode_3d)
    
    # Initialize output array matching reference format [Mic, Freq, Source]
    total_mics = atf_mag_gt.shape[0]
    total_sources = min(num_sources_eval or atf_mag_gt.shape[2], len(test_sampler))
    your_atf_predictions = torch.zeros(total_mics, your_freq_up_to, total_sources)
    
    # Fixed M and parameters (from inference_1d_atf.py)
    M = ref_config['num_mes_test']  # Use same M as reference (5)
    guidance_scale = 1.0
    num_timesteps = 10
    
    # Load the SAME microphone selection strategy as reference model
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    if os.path.exists(idx_mes_pos_path):
        idx_mes_pos_mat = np.load(idx_mes_pos_path)
        print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
        print("Using source-specific microphone selection for ATF generation")
        use_ref_strategy = True
    else:
        print("Warning: Could not load reference microphone selection, using random sampling")
        use_ref_strategy = False
    
    print(f"Generating predictions for {total_sources} sources with M={M} microphones...")
    
    # Generate predictions for each source
    for src_idx in tqdm(range(total_sources), desc="Your Model ATF"):
        with torch.no_grad():
            # Get source data (same as inference_1d_atf.py)
            z_true = test_sampler.cubes[src_idx].unsqueeze(0).to(device)
            src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)
            
            # Create sparse observations - use SAME strategy as reference for fair comparison
            if use_ref_strategy:
                # Use source-specific microphones (different M=5 for each source)
                source_specific_indices = idx_mes_pos_mat[:M, src_idx]  # First M mics for this source
                obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)
            else:
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]  # Fallback to random
            
            obs_xyz_abs = grid_xyz[obs_indices]
            obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)
            
            z_flat = z_true.view(z_true.shape[1], -1)
            obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
            obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
            
            # Get conditioning tokens
            y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)
            
            # Generate prediction (same as inference_1d_atf.py)
            x0 = torch.randn_like(z_true)
            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)
            
            simulator.ode.guidance_scale = guidance_scale
            x1_recon = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                        obs_mask=obs_mask, pooled_context=pooled_context,
                                        paste_observations=False, obs_indices=obs_indices)
            
            # De-normalize (same as inference_1d_atf.py)
            gen_cube_denorm = (x1_recon * std + mean)
            
            # Convert 3D grid to microphone format
            # Extract ATF values at all microphone positions
            nx, ny, nz = 11, 11, 11  # Grid dimensions
            for mic_idx in range(total_mics):
                # Convert flat microphone index to 3D coordinates (same as inference_1d_atf.py)
                iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
                
                # Extract frequency response at this microphone position
                if iz < gen_cube_denorm.shape[2] and iy < gen_cube_denorm.shape[3] and ix < gen_cube_denorm.shape[4]:
                    your_atf_predictions[mic_idx, :, src_idx] = gen_cube_denorm[0, :, iz, iy, ix].cpu()
    
    print(f"Generated ATF predictions: {your_atf_predictions.shape} (Mic, Freq, Source)")
    return your_atf_predictions

# def get_fallback_reference_results():
#     """Get pre-computed reference results as fallback."""
#     reference_results = {
#         100: {'mean': 3.7072, 'std': 0.8607},
#         50: {'mean': 3.9413, 'std': 0.8662},
#         20: {'mean': 4.1927, 'std': 0.8633},
#         10: {'mean': 4.3775, 'std': 0.8779},
#         5: {'mean': 4.4037, 'std': 0.8952}
#     }
#     return reference_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths - support for multiple models (like inference_1d_atf.py)
MODEL_LOAD_PATHS = [

    # "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/modelCONVoldcheckpoint.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-225845_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-222356_iter300000/model.pt", # 3.026 dB
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head4_sigma0ZERO_lr1e4to_e7_unet3_20250904-214817_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head4_sigma0ZERO_lr1e3to_e7_unet3_20250904-211357_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigmaE3_lr1e3to_e7_unet3_20250904-195716_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to10_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-140802_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer4_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-154234_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lrWARM5k_e4_toe7_unet3_20250905-165351_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lrWARM5k_e4_toe7_unet3_20250905-173800_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lrWARM5k_e4_toe5_unet3_20250905-182733_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma1e3_lrWARM5k_e4_toe6_unet3_20250905-193258_iter300000/model.pt", # 2.9339 dB

    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer6_20250906-191258_iter300000/model.pt",
    "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer3_20250906-215002_iter300000/model.pt", # 2.8967 dB
    # "/Users/ege/Projects/FMRIR/artifacts/M5to150_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-223838_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to10_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-140802_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-204240_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-204124_iter500000/model.pt",
    # #
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer4_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-154234_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lrWARM5k_e4_toe7_unet3_20250905-165351_iter300000/model.pt",

    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_V2_layer_20250906-173025_iter300000/model.pt",
    "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer3_20250906-191114_iter300000/model.pt",
    # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer6_20250906-191258_iter300000/model.pt",
]


M_values = [5]
num_sources_eval = None  # Set to None to evaluate all 102 sources, or e.g. 30 for faster testing

def get_model_name(model_path):
    """Extract model name from path (same as inference_1d_atf.py)"""
    return model_path.split("artifacts/")[1].split("/")[0]

# Get model names
MODEL_NAMES = [get_model_name(path) for path in MODEL_LOAD_PATHS]
MULTI_MODEL_MODE = len(MODEL_LOAD_PATHS) > 1

print(f"{'=== MULTI-MODEL EVALUATION ===' if MULTI_MODEL_MODE else '=== SINGLE MODEL EVALUATION ==='}")
print(f"Device: {device}")
for i, (path, name) in enumerate(zip(MODEL_LOAD_PATHS, MODEL_NAMES)):
    print(f"  Model {i+1}: {name}")
print()

# Load and evaluate all your models
print("\n1. Loading your 3D Flow Matching models...")
all_your_results = {}
all_your_predictions = {}  # Store predictions to avoid reloading best model
your_freq_up_to = None

for i, (model_path, model_name) in enumerate(zip(MODEL_LOAD_PATHS, MODEL_NAMES)):
    print(f"Loading model {i+1}/{len(MODEL_LOAD_PATHS)}: {model_name}")
    set_encoder, unet_3d, ode_3d, your_config = load_your_model(model_path, device)
    
    if your_freq_up_to is None:
        your_freq_up_to = your_config['model']['freq_up_to']
        print(f"Model frequency range: {your_freq_up_to}")
    
    # Evaluate this model
    model_results, idx_mes_pos_mat = evaluate_your_model(set_encoder, unet_3d, ode_3d, your_config, M_values, device, num_sources_eval)
    all_your_results[model_name] = model_results
    
    # Store model components for later plotting (avoid reloading best model)
    all_your_predictions[model_name] = (set_encoder, unet_3d, ode_3d, your_config)

# Load and evaluate reference model
print("\n2. Loading reference AUTOENCODER model...")
atf_mag_est, atf_mag_gt, ref_config, ref_data = load_reference_model(device, your_freq_up_to)

ref_results = evaluate_reference_model(atf_mag_est, atf_mag_gt, ref_config, num_sources_eval, your_freq_up_to)

# Print results
print("\n" + "="*80)
print("=== COMPARISON RESULTS ===")
print("="*80)
print(f"Your model freq range: 0-{your_freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz ({your_freq_up_to} bins)")
print(f"Reference freq range: 0-{ref_config['fs']//2} Hz ({ref_config['num_freq']} bins)")
print(f"Sources evaluated: {ref_results['num_sources_eval']} (out of 102 total)")
print("-"*100)
print(f"{'Method':<40} | {'LSD Same Freq':<15} | {'LSD Full Range':<15} | {'Std Dev':<10}")
print("-"*100)

# Reference model - show both frequency ranges
ref_same_freq = ref_results.get('mean_matched_freq', ref_results['mean'])
ref_full_range = ref_results['mean']
print(f"{'Reference (M=' + str(ref_results['num_mics']) + ' mics)':<40} | {ref_same_freq:.4f}        | {ref_full_range:.4f}        | {ref_results['std']:.4f}")

# All your models
for model_name, model_results in all_your_results.items():
    for M in M_values:
        # Truncate long model names for better display
        display_name = model_name[60:-5] + "..." if len(model_name) > 45 else model_name
        your_lsd = model_results[M]['mean']
        print(f"{display_name + f' (M={M})':<40} | {your_lsd:.4f}        | {your_lsd:.4f}        | {model_results[M]['std']:.4f}")
        
        # Show improvements for both comparisons
        improvement_same = ref_same_freq - your_lsd
        improvement_full = ref_full_range - your_lsd
        print(f"{'→ vs Ref (same freq)':<40} | {improvement_same:+.4f}        | {'N/A':<15} | {'N/A':<10}")
        print(f"{'→ vs Ref (full range)':<40} | {'N/A':<15} | {improvement_full:+.4f}        | {'N/A':<10}")
        print("-"*100)

# Find best model
best_model = None
best_lsd = float('inf')
for model_name, model_results in all_your_results.items():
    for M in M_values:
        if model_results[M]['mean'] < best_lsd:
            best_lsd = model_results[M]['mean']
            best_model = model_name

print("="*80)
print(f"🏆 BEST MODEL: {best_model}")
print(f"   Best LSD: {best_lsd:.4f} dB")
print(f"   Improvement over Reference: {ref_results['mean'] - best_lsd:+.4f} dB")
print("="*80)
print(f"Note: All models use M={ref_results['num_mics']} observation microphones")
print(f"      Reference uses source-specific microphone selection")
print(f"      Your models use SAME source-specific microphone selection")
print(f"      (Different M=5 microphones for each source, as per reference)")
print("="*80)

# Plot ATF comparisons using the best model (no reloading needed!)
print("\n3. Generating ATF comparison plots...")
print(f"Using best model for plots: {best_model}")

if best_model and best_model in all_your_predictions:
    # Use already loaded model components (efficient!)
    set_encoder_best, unet_3d_best, ode_3d_best, config_best = all_your_predictions[best_model]
    
    # Get your model's ATF predictions for plotting
    your_atf_predictions = get_your_model_atf_predictions(
        set_encoder_best, ode_3d_best, config_best, device,
        atf_mag_gt, ref_config, your_freq_up_to, num_sources_eval
    )
    
    plot_atf_comparisons(atf_mag_est, your_atf_predictions, atf_mag_gt, ref_config, your_freq_up_to, num_sources_eval)
else:
    print("Could not find best model for plotting")


