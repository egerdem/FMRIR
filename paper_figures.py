import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import json
import time
from datetime import datetime

from fm_utils import (ATF3DSampler, CFGVectorFieldODE_3D, EulerSimulator, EulerMaruyamaSimulator,
                      CFGVectorFieldODE_3D_V2, DDPMScheduler, SetEncoder,
                      SetEncoder_v12, CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d,
                      CrossAttentionUNet3D_v3, DDPM_ODE_Sampler,
                      get_model_info, print_model_info)

from model_paths import MODEL_LOAD_PATH

# Import the exact same functions from inference.py (no math changes)
from inference import (calculate_lsd_unified, calculate_slice_metrics, model_factory, 
                      load_model_and_config, get_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def freq_bin_to_hz(freq_bin, freq_up_to, fs=2000, total_fft_bins=64):
    """
    Convert frequency bin index to Hz.
    
    The data is generated with fs=2000Hz creating 64 frequency bins total (0 to 1000Hz).
    The model only uses the first freq_up_to bins (e.g., 20 out of 64).
    
    Args:
        freq_bin: Frequency bin index (0-based)
        freq_up_to: Number of frequency bins used by the model (e.g., 20)
        fs: Sampling frequency in Hz (default 2000)
        total_fft_bins: Total FFT bins in the original data (default 64)
    
    Returns:
        Frequency in Hz
    """
    return (freq_bin + 1) * fs / (2 * total_fft_bins)

def generate_paper_figures(model_path, freq_idx_to_plot=[5, 10, 15, 20], 
                          z_slice_idx=5, guidance_scale=1.0, 
                          M_range=None, num_timesteps=300,
                          save_dir="paper_figures"):
    """
    Generate paper figures for multiple frequency bins.
    
    Args:
        model_path: Path to the model checkpoint
        freq_idx_to_plot: List of frequency bin indices to visualize
        z_slice_idx: Z-slice index to extract (default 5)
        guidance_scale: Guidance scale for generation (default 1.0)
        M_range: Range for number of microphones [min, max] (default from config)
        num_timesteps: Number of timesteps for generation (default 300)
        save_dir: Directory to save figures (default "paper_figures")
    """
    
    print(f"=== PAPER FIGURES GENERATION ===")
    print(f"Model: {get_model_name(model_path)}")
    print(f"Frequencies to plot: {freq_idx_to_plot}")
    print(f"Z-slice index: {z_slice_idx}")
    print(f"Guidance scale: {guidance_scale}")
    
    # Load model and config
    checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)
    
    # Extract configuration
    data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    
    # Set M_range from config if not provided
    if M_range is None:
        M_range = config['training'].get('M_range', [5, 20])
    
    print(f"M_range: {M_range}")
    print(f"freq_up_to: {freq_up_to}")
    
    # Load data configuration for frequency conversion
    data_config_path = os.path.join(data_dir, "config.json")
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    fs = data_config.get('fs', 2000)  # Sampling frequency
    
    # Convert frequency bins to Hz for labeling
    # The original data has 64 frequency bins from 0 to fs/2 Hz, we use only freq_up_to of them
    freq_hz_labels = [freq_bin_to_hz(f_idx, freq_up_to, fs, total_fft_bins=64) for f_idx in freq_idx_to_plot]
    print(f"Frequency bins {freq_idx_to_plot} correspond to {[f'{f:.1f} Hz' for f in freq_hz_labels]}")
    
    # Validate frequency indices
    if max(freq_idx_to_plot) >= freq_up_to:
        raise ValueError(f"Max frequency index {max(freq_idx_to_plot)} exceeds model's freq_up_to {freq_up_to}")
    
    # Load and create models
    set_encoder, unet_3d, ode_3d, architecture = model_factory(config, model_states_cfg, device)
    
    # Print model info
    model_name = get_model_name(model_path)
    print(f"\n--- Model Architecture: {model_name} ---")
    set_encoder_info = get_model_info(set_encoder, "SetEncoder")
    unet_info = get_model_info(unet_3d, "UNet3D")
    total_params = set_encoder_info['total_params'] + unet_info['total_params']
    total_size_mb = set_encoder_info['model_size_mb'] + unet_info['model_size_mb']
    print(f"Total parameters: {total_params:,} | Size: {total_size_mb:.2f} MB")
    
    # Load data samplers
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    
    # Normalize test data using training stats
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    print(f"Data stats - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Sample one random M value that will be shared across all frequencies
    M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
    print(f"Using M={M} microphones for all frequencies")
    
    # Get a random ground truth sample
    z_true, src_xyz, srcind = test_sampler.sample(1)
    z_true, src_xyz = z_true.to(device), src_xyz.to(device)
    
    print(f"Source index: {srcind[0]}")
    
    # Create sparse observation set (same for all frequencies)
    obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
    obs_xyz_abs = grid_xyz[obs_indices]
    obs_coords_rel = obs_xyz_abs - src_xyz
    
    z_flat = z_true.view(z_true.shape[1], -1)
    obs_values = z_flat[:, obs_indices].transpose(0, 1)
    
    # Batchify for the set encoder
    obs_coords_rel = obs_coords_rel.unsqueeze(0)
    obs_values = obs_values.unsqueeze(0)
    obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
    
    # Get conditioning tokens (same for all frequencies)
    y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)
    
    # Setup figure - 4 columns (frequencies) x 3 rows (input mics, true, generated)
    num_cols = len(freq_idx_to_plot)
    num_rows = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 4.5 * num_rows))
    
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Main title
    model_name_parts = model_name.split('_')
    title_line1 = '_'.join(model_name_parts[:4])
    title_line2 = '_'.join(model_name_parts[4:]) if len(model_name_parts) > 4 else ""

    title_line1 = ""
    title = f"(Z-Slice={z_slice_idx}, w={guidance_scale}, M={M})\n{title_line1}"
    if title_line2:
        # title += f"\n{title_line2}"
        pass
    
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Add model info
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    best_iteration = checkpoint.get('best_iteration', 'N/A')
    fig.text(0.5, 0.90, f"Best Val Loss: {best_val_loss} at Iteration {best_iteration}",
             ha='center', fontsize=10)
    
    # Process each frequency
    for col_idx, freq_idx in enumerate(freq_idx_to_plot):
        freq_hz = freq_hz_labels[col_idx]
        print(f"\nProcessing frequency bin {freq_idx} ({freq_hz:.1f} Hz)...")
        
        # Row 0: Input microphone configuration (2D scatter plot)
        ax_scatter = axes[0, col_idx]
        obs_xyz_plot = obs_xyz_abs.cpu().numpy()
        sc = ax_scatter.scatter(obs_xyz_plot[:, 0], obs_xyz_plot[:, 1], 
                               c=obs_xyz_plot[:, 2], cmap='coolwarm', s=20,
                               vmin=-0.5, vmax=0.5)
        ax_scatter.set_title(f"{freq_hz:.1f} Hz\nInput Mics (M={M})")
        ax_scatter.set_aspect('equal', adjustable='box')
        ax_scatter.set_xlim(-0.6, 0.6)
        ax_scatter.set_ylim(-0.6, 0.6)
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        
        # Store scatter plot for colorbar (will be added after all plots are created)
        if col_idx == num_cols - 1:  # Only on the last column
            last_scatter_plot = sc
            last_scatter_ax = ax_scatter
        
        # Row 1: True field
        z_true_denorm = (z_true * std + mean)
        gt_cube_raw = z_true_denorm[0, freq_idx].cpu().numpy()
        gt_slice = gt_cube_raw[z_slice_idx, :, :]
        
        axes[1, col_idx].imshow(gt_slice, origin='lower', cmap='viridis', 
                               vmin=gt_slice.min(), vmax=gt_slice.max())
        axes[1, col_idx].set_title("True Field")
        axes[1, col_idx].axis('off')
        
        # Row 2: Generated field
        print(f"  Generating field for {freq_hz:.1f} Hz...")
        
        # Set up simulator
        fm_vs_diff = config['model'].get('FM_vs_Diff', 'flow_matching')
        if fm_vs_diff == 'flow_matching' or fm_vs_diff is None:
            simulator = EulerSimulator(ode=ode_3d)
            simulator.ode.guidance_scale = guidance_scale
            
            # Start from pure noise
            x0 = torch.randn_like(z_true)
            xt = x0.clone()
            
            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)
            
            # Generate
            x1_recon = simulator.simulate(xt, ts, x0=x0, z_true=z_true, 
                                        y_tokens=y_tokens, obs_mask=obs_mask,
                                        pooled_context=pooled_context,
                                        paste_observations=True, obs_indices=obs_indices)
        
        elif fm_vs_diff == 'score_matching':
            print("  Using DDIM Sampler")
            ddpm_scheduler = DDPMScheduler(num_timesteps=num_timesteps)
            ddpm_ode = DDPM_ODE_Sampler(
                noise_predictor_network=unet_3d,
                set_encoder=set_encoder,
                scheduler=ddpm_scheduler,
                config=config
            )
            
            simulator = EulerSimulator(ode=ddpm_ode)
            simulator.ode.guidance_scale = guidance_scale
            
            xt = torch.randn_like(z_true)
            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)
            
            simulation_kwargs = {
                "obs_coords_rel": obs_coords_rel,
                "obs_values": obs_values,
                "obs_mask": obs_mask,
                "pooled_context": pooled_context
            }
            
            x1_recon = simulator.simulate(xt, ts, **simulation_kwargs)
        
        # Plot generated field
        x1_recon_denorm = (x1_recon * std + mean)
        recon_cube_to_plot = x1_recon_denorm[0, freq_idx].detach().cpu().numpy()
        recon_slice = recon_cube_to_plot[z_slice_idx, :, :]
        
        axes[2, col_idx].imshow(recon_slice, origin='lower', cmap='viridis',
                               vmin=gt_slice.min(), vmax=gt_slice.max())
        axes[2, col_idx].set_title("Generated Field")
        axes[2, col_idx].axis('off')
        
        # Calculate and display metrics
        slice_metrics = calculate_slice_metrics(x1_recon_denorm[0], z_true_denorm[0], 
                                              freq_idx, z_slice_idx)
        
        # Add metrics text under the generated field
        metric_text = f"MSE: {slice_metrics['mse']:.3f}\nLSD: {slice_metrics['lsd']:.3f} dB"
        axes[2, col_idx].text(0.5, -0.15, metric_text,
                             transform=axes[2, col_idx].transAxes,
                             ha='center', va='top', fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.8, 
                                     edgecolor='none', pad=2))
        
        print(f"  Metrics - MSE: {slice_metrics['mse']:.4f}, LSD: {slice_metrics['lsd']:.4f} dB")
    
    # Add Z-height colorbar for input microphones (attach to all scatter plots for consistent sizing)
    scatter_axes = [axes[0, col_idx] for col_idx in range(num_cols)]
    cbar_z = fig.colorbar(last_scatter_plot, ax=scatter_axes, fraction=0.015, pad=0.1, shrink=0.6)
    cbar_z.set_label('Z-height (m)', size=8)
    cbar_z.ax.tick_params(labelsize=6)
    
    # Add shared colorbar for true/generated fields with same dimensions as Z-height colorbar
    true_gen_axes = []
    for col_idx in range(num_cols):
        true_gen_axes.extend([axes[1, col_idx], axes[2, col_idx]])
    
    # Use the range from the first frequency for consistent scaling
    z_true_denorm = (z_true * std + mean)
    gt_cube_sample = z_true_denorm[0, freq_idx_to_plot[0]].cpu().numpy()
    gt_slice_sample = gt_cube_sample[z_slice_idx, :, :]
    
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=gt_slice_sample.min(), 
                                       vmax=gt_slice_sample.max()), 
        cmap='viridis'
    )
    # Use same dimensions as Z-height colorbar to maintain alignment
    cbar_mag = fig.colorbar(mappable, ax=true_gen_axes, fraction=0.015, pad=0.1, shrink=0.6)
    cbar_mag.set_label('Magnitude (dB)', size=8)
    cbar_mag.ax.tick_params(labelsize=6)
    
    # Row labels
    row_labels = ["", "True Field", "Generated Field"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, rotation=90, va='center', fontsize=12, fontweight='bold')
    
    # Adjust layout to provide space for colorbars on the right
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, left=0.08, right=0.85, hspace=0.3, wspace=0.2)
    
    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_str = "_".join([f"{int(f)}Hz" for f in freq_hz_labels])
    architecture_short = architecture if architecture else "unknown"
    
    filename = f"paper_fig_{architecture_short}_freqs_{freq_str}_z{z_slice_idx}_w{guidance_scale}_M{M}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return save_path

if __name__ == '__main__':
    # Configuration
    model_path = MODEL_LOAD_PATH
    freq_idx_to_plot = [5, 10, 15, 19]  # Frequency bin indices
    z_slice_idx = 5                     # Z-slice index
    guidance_scale = 1.0                # Guidance scale
    M_range = [5, 6]                   # Range for number of microphones
    num_timesteps = 10                 # Number of generation timesteps
    
    print("Starting paper figures generation...")
    
    save_path = generate_paper_figures(
        model_path=model_path,
        freq_idx_to_plot=freq_idx_to_plot,
        z_slice_idx=z_slice_idx,
        guidance_scale=guidance_scale,
        M_range=M_range,
        num_timesteps=num_timesteps,
        save_dir="paper_figures"
    )
    
    print(f"Paper figures generation completed!")
    print(f"Saved to: {save_path}")
