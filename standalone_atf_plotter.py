#!/usr/bin/env python3
"""
Standalone ATF Comparison Plotter

This script allows you to generate ATF comparison plots using a specified model path.
It extracts the plotting functionality from unified_evaluation.py and makes it standalone.

Usage:
    python standalone_atf_plotter.py --model_path <path_to_model> [options]

Example:
    python standalone_atf_plotter.py --model_path "find64_holeloss_ATFUNet_M20_20250811-102034_iter60000/model.pt" --guidance 3.0 --num_sources 5
"""

import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # Same as eval_AUTOENCODER.py
from matplotlib import pyplot as plt
from inference import model_factory, load_model_and_config
from model_paths import MODEL_LOAD_PATH

# Your model imports
from fm_utils import (
    ATF3DSampler, SetEncoder, 
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, 
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator,
    get_model_info, print_model_info
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
        
        print(f"Reference data loaded: {atf_mag_gt.shape} (Mic, Freq, Src)")
        print(f"Full reference data: {atf_mag_gt.shape[1]} frequency bins")
        print(f"Your model uses: {freq_up_to} frequency bins")
        
        # Return FULL reference data (don't truncate here - let evaluation function handle it)
        return atf_mag_est, atf_mag_gt, config, data
        
    except Exception as e:
        print(f"Error loading reference model: {e}")
        print("Using pre-computed reference results instead...")
        return None, None, None, None


def get_your_model_atf_predictions(set_encoder, ode_3d, config, device, atf_mag_gt, ref_config, freq_up_to, num_sources_eval, single_guidance=3.0):
    """
    Extract ATF predictions from your model in the same format as reference model.
    """
    print(f"Generating ATF predictions from your 3D model with guidance scale {single_guidance}...")
    
    # Load your data (same as in inference_1d_atf.py)
    data_path = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    
    # Load normalized data (exactly like unified_evaluation.py)
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    # Create simulator
    simulator = EulerSimulator(ode=ode_3d)
    
    # Initialize output array matching reference format [Guidance, Mic, Freq, Source]
    total_mics = atf_mag_gt.shape[0]
    total_sources = min(num_sources_eval or atf_mag_gt.shape[2], len(test_sampler))
    your_atf_predictions = {single_guidance: torch.zeros(total_mics, freq_up_to, total_sources)}
    
    # Fixed M and parameters (from inference_1d_atf.py)
    M = ref_config['num_mes_test']  # Use same M as reference (5)
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
    
    # Generate predictions for each source (exactly like unified_evaluation.py)
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
            
            # Run inference for this guidance scale
            simulator.ode.guidance_scale = single_guidance
            x1_recon = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                       obs_mask=obs_mask, pooled_context=pooled_context,
                                       paste_observations=True, obs_indices=obs_indices)
            
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
                    your_atf_predictions[single_guidance][mic_idx, :, src_idx] = gen_cube_denorm[0, :, iz, iy, ix].cpu()
    
    # print(f"Generated ATF predictions: {your_atf_predictions.shape} (Mic, Freq, Source)")
    predictions = your_atf_predictions
    
    return predictions


def plot_atf_comparisons(atf_mag_est_ref, atf_mag_est_yours, atf_mag_gt, ref_config, freq_up_to, num_sources_eval, best_guidance=None):
    """
    Plot ATF comparisons with 3 methods: True, Reference, Your Model
    """
    # Get the correct number of sources to evaluate
    total_sources = atf_mag_gt.shape[2]  # Total available sources
    eval_sources = min(num_sources_eval, total_sources) if num_sources_eval is not None else total_sources
    print(f"Evaluating ATF plots for first {eval_sources} sources (out of {total_sources})")

    # Use provided best_guidance or use the only available guidance
    if best_guidance is None:
        best_guidance = list(atf_mag_est_yours.keys())[0]
        print(f"Using guidance scale w={best_guidance}")
    else:
        print(f"Using provided best guidance scale w={best_guidance}")
    
    atf_mag_est_yours_best = atf_mag_est_yours[best_guidance]
    dataset_name = ref_config['dataset'][0]
    
    # Create frequency axis for your model (0 to freq_up_to bins)
    freq_yours = np.linspace(0, freq_up_to * ref_config['fs'] // 2 // ref_config['num_freq'], freq_up_to)
    
    print(f"Plotting frequency range: 0-{freq_yours[-1]:.0f} Hz ({freq_up_to} bins)")
    print(f"Reference model has {ref_config['num_freq']} frequency bins (0-{ref_config['fs']//2} Hz)")
    
    # Select specific combinations to plot (5 combinations as in original)
    combinations = [
        (0, 0),   # Mic 0, Src 0
        (1, 1),   # Mic 1, Src 1  
        (2, 2),   # Mic 2, Src 2
        (3, 3),   # Mic 3, Src 3
        (4, 4),   # Mic 4, Src 4
    ]
    
    # Ensure we don't exceed available data
    max_mic = min(4, atf_mag_gt.shape[0] - 1)
    max_src = min(4, eval_sources - 1)
    combinations = [(min(m, max_mic), min(s, max_src)) for m, s in combinations]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    for i, (mic_idx, src_idx) in enumerate(combinations):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Ensure indices are within bounds
        if mic_idx >= atf_mag_gt.shape[0] or src_idx >= eval_sources:
            ax.set_title(f"Mic {mic_idx}, Src {src_idx+922} (Out of bounds)")
            continue
        
        try:
            # Plot all three methods with correct frequency axes
            # All models plot the same frequency range for comparison (0-312 Hz)
            ax.plot(freq_yours, atf_mag_gt[mic_idx, :freq_up_to, src_idx], 'k--', label="True", linewidth=2)
            ax.plot(freq_yours, atf_mag_est_ref[mic_idx, :freq_up_to, src_idx], 'r-', label="Reference", linewidth=1.5)
            print(f"Plotting Source {src_idx+922}, Mic {mic_idx} (index {i+1}/5)")
            ax.plot(freq_yours, atf_mag_est_yours_best[mic_idx, :, src_idx], 'b-', 
                   label=f"Your Model (w={best_guidance})", linewidth=1.5)
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('ATF Magnitude')
            ax.set_title(f'Mic {mic_idx}, Src {src_idx+922}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            print(f"Error plotting combination {i}: {e}")
            ax.set_title(f"Mic {mic_idx}, Src {src_idx+922} (Error)")
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'standalone_atf_comparison.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"ATF comparison plot saved to: {plot_path}")
    
    plt.show()


def main():


    parser = argparse.ArgumentParser(description='Standalone ATF Comparison Plotter')
    parser.add_argument('--model_path', default=MODEL_LOAD_PATH,
                       help='Path to the model file (e.g., "find64_holeloss_ATFUNet_M20_20250811-102034_iter60000/model.pt")')
    parser.add_argument('--guidance', type=float, default=1,
                       help='Guidance scale for generation (default: 3.0)')
    parser.add_argument('--num_sources', type=int, default=10,
                       help='Number of sources to evaluate (default: 5)')
    parser.add_argument('--freq_up_to', type=int, default=20,
                       help='Number of frequency bins to use (default: 20)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your model
    print(f"Loading your model from: {args.model_path}")
    try:
        checkpoint, config, model_states_cfg = load_model_and_config(args.model_path, device)
        set_encoder, unet_3d, ode_3d, is_new_model = model_factory(config, model_states_cfg, device)
        print("Your model loaded successfully!")
    except Exception as e:
        print(f"Error loading your model: {e}")
        return
    
    # Load reference model and data
    print("Loading reference model and data...")
    atf_mag_est, atf_mag_gt, ref_config, ref_data = load_reference_model(device, args.freq_up_to)
    
    if atf_mag_est is None:
        print("Failed to load reference model. Exiting.")
        return
    
    print(f"Ground truth ATF shape: {atf_mag_gt.shape}")
    print(f"Reference predictions shape: {atf_mag_est.shape}")
    
    # Generate your model's ATF predictions
    print("Generating ATF predictions from your model...")
    your_atf_predictions = get_your_model_atf_predictions(
        set_encoder, ode_3d, config, device,
        atf_mag_gt, ref_config, args.freq_up_to, args.num_sources,
        single_guidance=args.guidance
    )
    
    # Generate ATF comparison plots
    print("Generating ATF comparison plots...")
    plot_atf_comparisons(
        atf_mag_est, your_atf_predictions, atf_mag_gt, ref_config,
        args.freq_up_to, args.num_sources, best_guidance=args.guidance
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
