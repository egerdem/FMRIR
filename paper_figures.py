import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import json
import time
from datetime import datetime

from fm_utils import (ATF3DSampler, EulerSimulator,
                       DDPM_ODE_Sampler,
                      get_model_info, print_model_info)

from model_paths import MODEL_LOAD_PATH

# Import the exact same functions from inference.py (no math changes)
from inference import (calculate_lsd_unified, calculate_slice_metrics, model_factory, 
                      load_model_and_config, get_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== CONFIGURATION CONSTANTS =====
SEED = 36 # 42

# Data paths
DATA_DIR = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
AUTOENCODER_PATH = "AUTOENCODER/src"
MIC_SELECTION_PATH = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"

# Reference model result paths
# FSMPAE_RESULTS_PATH = "RESULTS/out_20250323_FSMPAE_10026/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt"
FSMPAE_RESULTS_PATH = "RESULTS/out_20250916_EEAE_10001/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200_freq20.pt"
KRR_RESULTS_PATH = "RESULTS/out_20250324_KRR_10004/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt"

# ATF plotting constants
ATF_M = 5  # Number of microphones used by reference methods
ATF_FFTLEN_ALGN = 128  # FFT length for frequency axis

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

def generate_SFfigures_FM(model_path, freq_idx_to_plot,
                          z_slice_idx, guidance_scale=1.0,
                          M_range=None, num_timesteps=10,
                          save_dir="paper_figures", random_mics_per_freq=False, 
                          random_M_per_freq=False, M_seed=None):
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
        random_mics_per_freq: If True, use different random mic indices for each frequency; 
                             if False, use same mic indices for all frequencies (default False)
        random_M_per_freq: If True, sample different M values for each frequency;
                          if False, use same M value for all frequencies (default False)
        M_seed: Separate seed for M value sampling (default None, uses main seed)
    """
    
    print(f"=== PAPER FIGURES GENERATION ===")
    print(f"Model: {get_model_name(model_path)}")
    print(f"Frequencies to plot: {freq_idx_to_plot}")
    print(f"Z-slice index: {z_slice_idx}")
    print(f"Guidance scale: {guidance_scale}")
    
    # Load model and config
    checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)
    
    # Extract configuration
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    
    # Set M_range from config if not provided
    if M_range is None:
        M_range = config['training'].get('M_range')
    
    print(f"M_range: {M_range}")
    print(f"freq_up_to: {freq_up_to}")
    
    # Load data configuration for frequency conversion
    data_config_path = os.path.join(DATA_DIR, "config.json")
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
        data_path=DATA_DIR, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    
    # Normalize test data using training stats
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    print(f"Data stats - Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"Random microphones per frequency: {random_mics_per_freq}")
    print(f"Random M per frequency: {random_M_per_freq}")
    print(f"M sampling seed: {M_seed if M_seed is not None else 'using main seed'}")
    
    # Set separate seed for M sampling if provided
    if M_seed is not None:
        torch.manual_seed(M_seed)
        np.random.seed(M_seed)
        random.seed(M_seed)
    
    # Sample M values based on the flag
    if random_M_per_freq:
        # Different M values for each frequency
        M_values = [torch.randint(M_range[0], M_range[1] + 1, (1,)).item() for _ in freq_idx_to_plot]
        print(f"Using different M values: {M_values} for frequencies {freq_idx_to_plot}")
    else:
        # Same M value for all frequencies
        M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
        M_values = [M] * len(freq_idx_to_plot)
        print(f"Using M={M} microphones for all frequencies")
    
    # Restore main seed after M sampling
    if M_seed is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
    
    # Pre-generate source indices for each frequency
    source_indices = []
    z_trues = []
    src_xyzs = []
    for _ in freq_idx_to_plot:
        z_true, src_xyz, srcind = test_sampler.sample(1)
        z_true, src_xyz = z_true.to(device), src_xyz.to(device)
        source_indices.append(srcind[0])
        z_trues.append(z_true)
        src_xyzs.append(src_xyz)
    
    print(f"Source indices: {source_indices}")
    
    # Pre-generate microphone indices for all frequencies
    all_obs_indices = []
    for i, freq_idx in enumerate(freq_idx_to_plot):
        M_current = M_values[i]
        
        if random_mics_per_freq or i == 0:  # Always generate new indices for first freq or if random_mics_per_freq is True
            obs_indices = torch.randperm(grid_xyz.shape[0])[:M_current]
        else:
            # Reuse indices from first frequency, but adjust for different M if needed
            if M_current != M_values[0]:
                # If M is different, we need new indices
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M_current]
            else:
                # Same M, reuse same indices
                obs_indices = all_obs_indices[0]
        
        all_obs_indices.append(obs_indices)

    print(" len(obs_indices), len(all_obs_indices) ", len(obs_indices), len(all_obs_indices))

    if random_mics_per_freq:
        print(f"Using different random microphone sets for each frequency")
    else:
        print(f"Using same microphone set for all frequencies (when M values match)")
    
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
    # Create M display string based on mode
    if random_M_per_freq:
        M_display = f"M={M_values}"
    else:
        M_display = f"M={M_values[0]}"
    
    title = f"(Z-Slice={z_slice_idx}, w={guidance_scale}, {M_display})\n{title_line1}"
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
        
        # Get source and microphone indices for this frequency
        obs_indices = all_obs_indices[col_idx]
        M_current = M_values[col_idx]
        z_true = z_trues[col_idx]
        src_xyz = src_xyzs[col_idx]
        src_idx = source_indices[col_idx]
        
        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = obs_xyz_abs - src_xyz
        
        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1)
        
        # Batchify for the set encoder
        obs_coords_rel = obs_coords_rel.unsqueeze(0)
        obs_values = obs_values.unsqueeze(0)
        obs_mask = torch.ones(1, M_current, dtype=torch.bool, device=device)
        
        # Get conditioning tokens for this frequency
        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

        # Row 0: Input microphone configuration (2D scatter plot)
        ax_scatter = axes[col_idx, 0]
        obs_xyz_plot = obs_xyz_abs.cpu().numpy()
        sc = ax_scatter.scatter(obs_xyz_plot[:, 0], obs_xyz_plot[:, 1], 
                               c=obs_xyz_plot[:, 2], cmap='coolwarm', s=20,
                               vmin=-0.5, vmax=0.5)
        ax_scatter.set_title(f"{freq_hz:.1f} Hz\nM={M_current}, Src: {src_idx}")
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

        axes[1, col_idx].imshow(gt_slice, origin='lower', cmap='pink',
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
                                        freq_contexts=freq_contexts,
                                        paste_observations=True, obs_indices=obs_indices)

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
    
    # Make axis borders almost invisible by reducing line weight
    for freq_row_idx in range(len(freq_idx_to_plot)):
        for method_idx in range(num_methods):
            ax = axes[freq_row_idx, method_idx]
            if not (freq_row_idx != len(freq_idx_to_plot) // 2 and method_idx == 0):  # Skip hidden scatter plots
                if method_idx == 0:  # Keep normal borders for scatter plot
                    # Keep default border styling for scatter plot
                    pass
                else:  # Apply thin borders to sound field plots only
                    # Set very thin border lines
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.2)
                        spine.set_color('gray')
                        spine.set_alpha(0.3)
                    # Make tick marks smaller and lighter
                    ax.tick_params(axis='both', which='major', labelsize=8, width=0.2, length=2, color='gray', labelcolor='gray')
    # Add Z-height colorbar for input microphones (attach to all scatter plots for consistent sizing)
    middle_scatter_ax = axes[len(freq_idx_to_plot) // 2, 0]
    cbar_z = fig.colorbar(last_scatter_plot, ax=middle_scatter_ax, fraction=0.046, pad=0.08, shrink=1.0)
    cbar_z.set_label('Z-height (m)', size=8)
    cbar_z.ax.tick_params(labelsize=6)
    
    # Add shared colorbar for true/generated fields with same dimensions as Z-height colorbar
    true_gen_axes = []
    for col_idx in range(len(freq_idx_to_plot)):
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
    # Make SF colorbar span across all 3 rows
    all_axes = []
    for row_idx in range(num_rows):
        for col_idx in range(len(freq_idx_to_plot)):
            all_axes.append(axes[row_idx, col_idx])

    cbar_mag = fig.colorbar(mappable, ax=all_axes, fraction=0.046, pad=0.04, shrink=1.0)
    cbar_mag.set_label('Magnitude (dB)', size=8)
    cbar_mag.ax.tick_params(labelsize=6)
    
    # Row labels
    row_labels = ["Input Mics", "True Field", "Generated Field"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, rotation=90, va='center', fontsize=12, fontweight='bold')
    
    # Adjust layout to provide space for colorbars on the right
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, left=0.08, right=0.85, hspace=0.3, wspace=0.2)
    
    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_str = "_".join([f"{int(f)}Hz" for f in freq_hz_labels])
    architecture_short = architecture if architecture else "unknown"
    
    filename = f"paper_fig_{architecture_short}_freqs_{freq_str}_z{z_slice_idx}_w{guidance_scale}__{timestamp}.pdf"
    save_path = os.path.join(save_dir, filename)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return save_path


def generate_SFfigures_FM_V2(model_path, srcind, freq_idx_to_plot=[5, 10, 15, 20],
                          z_slice_idx=5, guidance_scale=1.0,
                          num_timesteps=10,
                          save_dir="paper_figures",
                          idx_mes_pos_mat=None):
    import sys

    sys.path.append(AUTOENCODER_PATH)
    import AUTOENCODER.src.dataset as autoencoder_dataset
    from AUTOENCODER.src.configs import config_FSMPAE_10026

    config = config_FSMPAE_10026.copy()
    original_cwd = os.getcwd()
    os.chdir('AUTOENCODER')
    idataset = autoencoder_dataset.ATFdataset(config=config)
    data = idataset.Data
    # dataset_name = config['dataset'][0]
    mic_position = data['test']['mic_position']['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200']
    os.chdir(original_cwd)

    """
    Generate paper figures for multiple frequency bins.

    Args:
        model_path: Path to the model checkpoint
        freq_idx_to_plot: List of frequency bin indices to visualize
        z_slice_idx: Z-slice index to extract (default 5)
        guidance_scale: Guidance scale for generation (default 1.0)
        num_timesteps: Number of timesteps for generation (default 300)
        save_dir: Directory to save figures (default "paper_figures")
        M_seed: Separate seed for M value sampling (default None, uses main seed)
    """

    print(f"=== PAPER FIGURES GENERATION ===")
    print(f"Model: {get_model_name(model_path)}")
    print(f"Frequencies to plot: {freq_idx_to_plot}")
    print(f"Z-slice index: {z_slice_idx}")
    print(f"Guidance scale: {guidance_scale}")

    # Load model and config
    checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)

    # Extract configuration
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')

    print(f"freq_up_to: {freq_up_to}")

    # Load data configuration for frequency conversion
    data_config_path = os.path.join(DATA_DIR, "config.json")
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
        data_path=DATA_DIR, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=freq_up_to
    )

    # Normalize test data using training stats
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    grid_xyz = train_sampler.grid_xyz.to(device)
    print("GRID XYZ", len(grid_xyz))

    mean = train_sampler.mean.item()
    std = train_sampler.std.item()

    print(f"Data stats - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Print Z-slice height information
    grid_xyz_np = grid_xyz.cpu().numpy()
    unique_z = np.unique(grid_xyz_np[:, 2])  # Z coordinates are in column 2
    print(f"Available Z heights: {unique_z}")
    print(f"Z-slice index {z_slice_idx} corresponds to Z height: {unique_z[z_slice_idx]:.3f} m")

    # Set separate seed for M sampling if provided
    M = SPARSE_M
    M_values = [M] * len(freq_idx_to_plot)
    print(f"Using M={M} microphones for all frequencies")


    # Pre-generate source indices for each frequency
    source_indices = []
    z_trues = []
    src_xyzs = []
    for _ in freq_idx_to_plot:
        # z_true, src_xyz, srcind = test_sampler.sample(1)
        # z_true, src_xyz = z_true.to(device), src_xyz.to(device)

        z_true = test_sampler.cubes[srcind[0]].unsqueeze(0).to(device)
        src_idx = srcind[0] if isinstance(srcind, (list, tuple, np.ndarray)) else int(srcind)


        src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)

        source_indices.append(src_idx)
        z_trues.append(z_true)
        src_xyzs.append(src_xyz)

    print(f"Source indices: {source_indices}")

    # Pre-generate microphone indices for all frequencies
    all_obs_indices = []
    for i, freq_idx in enumerate(freq_idx_to_plot):
        M_current = M_values[i]
        # source_specific_mic_indices = idx_mes_pos_mat[:M_current, srcind]
        source_specific_mic_indices = idx_mes_pos_mat[:M_current, src_idx].reshape(-1)

        print("source_specific_mic_indices ", source_specific_mic_indices)
        for i in source_specific_mic_indices:
            print("Mic positions:", grid_xyz[i])

        # obs_indices = torch.randperm(grid_xyz.shape[0])[:M_current]
        # obs_indices = torch.tensor(source_specific_mic_indices, dtype=torch.long, device=device)
        obs_indices = torch.as_tensor(source_specific_mic_indices, dtype=torch.long, device=device).view(-1)

        all_obs_indices.append(obs_indices)

    # Setup figure - 5 rows x 3 columns (transposed: 3 rows x 5 columns)
    # Original: 5 methods × 3 frequencies = 15 plots
    # New: 3 frequencies × 5 methods = 15 plots
    num_frequencies = len(freq_idx_to_plot)
    num_methods = 5  # Input Mics, True Field, SF-Flow, AE Field, KRR Field
    
    fig, axes = plt.subplots(num_frequencies, num_methods, figsize=(4.5 * num_methods, 4.5 * num_frequencies))
    plt.subplots_adjust(right=0.84)  # leave more room for colorbars
    
    if num_frequencies == 1:
        axes = axes.reshape(1, -1)
    elif num_methods == 1:
        axes = axes.reshape(-1, 1)

    # Main title
    model_name_parts = model_name.split('_')
    title_line1 = '_'.join(model_name_parts[:4])
    title_line2 = '_'.join(model_name_parts[4:]) if len(model_name_parts) > 4 else ""

    # title_line1 = ""
    # Create M display string based on mode

    # title = f"(Z-Slice={z_slice_idx}, w={guidance_scale})\n{title_line1}"
    # if title_line2:
        # title += f"\n{title_line2}"
        # pass

    # fig.suptitle(title, fontsize=14, y=0.95)

    # Add model info
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    best_iteration = checkpoint.get('best_iteration', 'N/A')
    # fig.text(0.5, 0.90, f"Best Val Loss: {best_val_loss} at Iteration {best_iteration}",
            #  ha='center', fontsize=10)

    # Plot Ref figures
    fsmpae_results = torch.load(FSMPAE_RESULTS_PATH)
    print(f"  FSMPAE loaded: {fsmpae_results.shape}")

    krr_results = torch.load(KRR_RESULTS_PATH)
    print(f"  KRR loaded: {krr_results.shape}")

    # Process each frequency
    for freq_row_idx, freq_idx in enumerate(freq_idx_to_plot):
        freq_hz = freq_hz_labels[freq_row_idx]
        print(f"\nProcessing frequency bin {freq_idx} ({freq_hz:.1f} Hz)...")

        # Get source and microphone indices for this frequency
        obs_indices = all_obs_indices[freq_row_idx]
        M_current = M_values[freq_row_idx]
        z_true = z_trues[freq_row_idx]
        src_xyz = src_xyzs[freq_row_idx]
        src_idx = source_indices[freq_row_idx]

        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = obs_xyz_abs - src_xyz

        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1)

        # Batchify for the set encoder
        obs_coords_rel = obs_coords_rel.unsqueeze(0)
        obs_values = obs_values.unsqueeze(0)
        obs_mask = torch.ones(1, M_current, dtype=torch.bool, device=device)

        # Get conditioning tokens for this frequency
        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

        # Method 0: Input microphone configuration (2D scatter plot) - only show in middle row
        ax_scatter = axes[freq_row_idx, 0]
        
        if freq_row_idx == len(freq_idx_to_plot) // 2:  # Only show scatter plot in middle row
            obs_xyz_plot = obs_xyz_abs.cpu().numpy()
            sc = ax_scatter.scatter(obs_xyz_plot[:, 0], obs_xyz_plot[:, 1],
                                    c=obs_xyz_plot[:, 2], cmap='coolwarm', s=20,
                                    vmin=-0.5, vmax=0.5)
            ax_scatter.set_aspect('equal', adjustable='box')
            ax_scatter.set_xlim(-0.6, 0.6)
            ax_scatter.set_ylim(-0.6, 0.6)
            
            # Add axis labels and ticks for scatter plot
            ax_scatter.set_xlabel('x (m)', fontsize=10)
            ax_scatter.set_ylabel('  y (m)', fontsize=10)
            ax_scatter.set_xticks([-0.5, 0.0, 0.5])
            ax_scatter.set_xticklabels(['-0.5', '0.0', '0.5'])
            ax_scatter.set_yticks([-0.5, 0.0, 0.5])
            ax_scatter.set_yticklabels(['-0.5', '0.0', '0.5'])
            
            # Store scatter plot for colorbar
            last_scatter_plot = sc
            last_scatter_ax = ax_scatter
        else:
            # Hide other scatter plot positions
            ax_scatter.axis('off')

        # Row 1: True field
        z_true_denorm = (z_true * std + mean)
        gt_cube_raw = z_true_denorm[0, freq_idx].cpu().numpy()
        gt_slice = gt_cube_raw[z_slice_idx, :, :]

        axes[freq_row_idx, 1].imshow(gt_slice, origin='lower', cmap='viridis',
                                     vmin=gt_slice.min(), vmax=gt_slice.max())
        axes[freq_row_idx, 1].set_title(f"True: {freq_hz:.1f} Hz", fontsize=12, fontweight='bold', pad=10)
        
        # Add axis labels and ticks for True Field
        axes[freq_row_idx, 1].set_xlabel('x (m)', fontsize=10)
        axes[freq_row_idx, 1].set_ylabel('y (m)', fontsize=10)
        axes[freq_row_idx, 1].yaxis.set_label_position('right')
        axes[freq_row_idx, 1].yaxis.tick_right()
        axes[freq_row_idx, 1].set_xticks([0, 5, 10])
        axes[freq_row_idx, 1].set_xticklabels(['-0.5', '0.0', '0.5'])
        axes[freq_row_idx, 1].set_yticks([0, 5, 10])
        axes[freq_row_idx, 1].set_yticklabels(['-0.5', '0.0', '0.5'])

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
                                          freq_contexts=freq_contexts,
                                          paste_observations=True, obs_indices=obs_indices)

        # Plot generated field
        x1_recon_denorm = (x1_recon * std + mean)
        recon_cube_to_plot = x1_recon_denorm[0, freq_idx].detach().cpu().numpy()
        recon_slice = recon_cube_to_plot[z_slice_idx, :, :]

        grid_xyz_np = grid_xyz.cpu().numpy()
        print("grid_xyz shape: ", grid_xyz_np.shape)
        # Prepare reference method slices for this frequency and source
        srcind = src_idx
        # GLOBAL
        idx_dist = np.where(np.abs(mic_position[:, 2, srcind]) < 1e-6)

        if fsmpae_results is not None:
            atf_mag_est = fsmpae_results[: , :freq_up_to, :]
            fsmpae_slice = atf_mag_est[idx_dist, freq_idx, srcind].reshape(11, 11)
            print("shape fsmpae_cube_3d: ", fsmpae_slice)
            # fsmpae_slice = fsmpae_cube_3d[freq_idx, z_slice_idx, :, :]  # [y, x]

        if krr_results is not None:
            krr_raw_mag_est = krr_results[:, :freq_up_to, :]
            krr_slice = krr_raw_mag_est[idx_dist, freq_idx, srcind].reshape(11, 11)  # [y, x]
            print("shape krr_slice: ", krr_slice.shape)

        axes[freq_row_idx, 2].imshow(recon_slice, origin='lower', cmap='viridis',
                                     vmin=gt_slice.min(), vmax=gt_slice.max())
        # axes[freq_row_idx, 2].set_title("SF-Flow")
        
        # Add axis labels and ticks for SF-Flow
        axes[freq_row_idx, 2].set_xlabel('x (m)', fontsize=10)
        axes[freq_row_idx, 2].set_ylabel('y (m)', fontsize=10)
        axes[freq_row_idx, 2].yaxis.set_label_position('right')
        axes[freq_row_idx, 2].yaxis.tick_right()
        axes[freq_row_idx, 2].set_xticks([0, 5, 10])
        axes[freq_row_idx, 2].set_xticklabels(['-0.5', '0.0', '0.5'])
        axes[freq_row_idx, 2].set_yticks([0, 5, 10])
        axes[freq_row_idx, 2].set_yticklabels(['-0.5', '0.0', '0.5'])

        # Method 3: AE Field
        axes[freq_row_idx, 3].imshow(fsmpae_slice, origin='lower', cmap='viridis',
                                     vmin=gt_slice.min(), vmax=gt_slice.max())
        # axes[freq_row_idx, 3].set_title("AE Field")
        
        # Add axis labels and ticks for AE Field
        axes[freq_row_idx, 3].set_xlabel('x (m)', fontsize=10)
        axes[freq_row_idx, 3].set_ylabel('y (m)', fontsize=10)
        axes[freq_row_idx, 3].yaxis.set_label_position('right')
        axes[freq_row_idx, 3].yaxis.tick_right()
        axes[freq_row_idx, 3].set_xticks([0, 5, 10])
        axes[freq_row_idx, 3].set_xticklabels(['-0.5', '0.0', '0.5'])
        axes[freq_row_idx, 3].set_yticks([0, 5, 10])
        axes[freq_row_idx, 3].set_yticklabels(['-0.5', '0.0', '0.5'])

        # Method 4: KRR Field
        axes[freq_row_idx, 4].imshow(krr_slice, origin='lower', cmap='viridis',
                                     vmin=gt_slice.min(), vmax=gt_slice.max())
        # axes[freq_row_idx, 4].set_title("KRR Field")
        
        # Add axis labels and ticks for KRR Field
        axes[freq_row_idx, 4].set_xlabel('x (m)', fontsize=10)
        axes[freq_row_idx, 4].set_ylabel('y (m)', fontsize=10)
        axes[freq_row_idx, 4].yaxis.set_label_position('right')
        axes[freq_row_idx, 4].yaxis.tick_right()
        axes[freq_row_idx, 4].set_xticks([0, 5, 10])
        axes[freq_row_idx, 4].set_xticklabels(['-0.5', '0.0', '0.5'])
        axes[freq_row_idx, 4].set_yticks([0, 5, 10])
        axes[freq_row_idx, 4].set_yticklabels(['-0.5', '0.0', '0.5'])

        # Calculate and display metrics
        slice_metrics = calculate_slice_metrics(x1_recon_denorm[0], z_true_denorm[0],
                                                freq_idx, z_slice_idx)

        # Add metrics text under the generated field
        # metric_text = f"MSE: {slice_metrics['mse']:.3f}\nLSD: {slice_metrics['lsd']:.3f} dB"
        # axes[2, col_idx].text(0.5, -0.15, metric_text,
        #                       transform=axes[2, col_idx].transAxes,
        #                       ha='center', va='top', fontsize=9,
        #                       bbox=dict(facecolor='white', alpha=0.8,
        #                                 edgecolor='none', pad=2))

        print(f"  Metrics - MSE: {slice_metrics['mse']:.4f}, LSD: {slice_metrics['lsd']:.4f} dB")
    
    # Make axis borders almost invisible by reducing line weight
    for freq_row_idx in range(len(freq_idx_to_plot)):
        for method_idx in range(num_methods):
            ax = axes[freq_row_idx, method_idx]
            if not (freq_row_idx != len(freq_idx_to_plot) // 2 and method_idx == 0):  # Skip hidden scatter plots
                if method_idx == 0:  # Keep normal borders for scatter plot
                    # Keep default border styling for scatter plot
                    pass
                else:  # Apply thin borders to sound field plots only
                    # Set very thin border lines
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.2)
                        spine.set_color('gray')
                        spine.set_alpha(0.3)
                    # Make tick marks smaller and lighter
                    ax.tick_params(axis='both', which='major', labelsize=8, width=0.2, length=2, color='gray', labelcolor='gray')
    # Add Z-height colorbar for input microphones (attach to all scatter plots for consistent sizing)
    middle_scatter_ax = axes[len(freq_idx_to_plot) // 2, 0]
    cbar_z = fig.colorbar(last_scatter_plot, ax=middle_scatter_ax, fraction=0.046, pad=0.08, shrink=1.0)
    cbar_z.set_label('Z-height (m)', size=8)
    cbar_z.ax.tick_params(labelsize=6)

    # ---- PER-ROW NORMALIZE (own scale per frequency row) ----
    for row, freq_idx in enumerate(freq_idx_to_plot):
        # Build a Normalize from the row's ground-truth slice
        z_true_denorm_row = (z_trues[row] * std + mean)
        gt_slice_row = z_true_denorm_row[0, freq_idx].cpu().numpy()[z_slice_idx, :, :]

        row_norm = matplotlib.colors.Normalize(vmin=gt_slice_row.min(),
                                               vmax=gt_slice_row.max())
        row_mappable = matplotlib.cm.ScalarMappable(norm=row_norm, cmap='viridis')
        row_mappable.set_array([])  # This is crucial for PDF rendering
        row_mappable._A = []  # Additional fix for some matplotlib versions

        # last_ax = axes[row, -1]
        # cax = inset_axes(last_ax,
        #                  width="3%",
        #                  height="100%",
        #                  loc='lower left',
        #                  bbox_to_anchor=(1.02, 0., 1, 1),
        #                  bbox_transform=last_ax.transAxes,
        #                  borderpad=0)
        # cb = fig.colorbar(row_mappable, cax=cax)


        last_ax = axes[row, -1]
        cax = inset_axes(last_ax,
                         width="3%",
                         height="100%",
                         loc='lower left',
                         bbox_to_anchor=(1.4, 0., 1, 1),
                         bbox_transform=last_ax.transAxes,
                         borderpad=0)
        im_handle = axes[row, 2].images[0]  # or axes[row, 1].images[0] if you want GT scaling
        cb = fig.colorbar(im_handle, cax=cax)

        cb.set_label('Magnitude (dB)', size=8)
        cb.ax.tick_params(labelsize=6)

    # Column labels for methods (skip True Field since it shows frequency titles)
    method_labels = ["Input Mics", "", "SF-Flow", "AE", "KRR"]
    for method_idx, label in enumerate(method_labels):
        if label:  # Only set title if label is not empty
            axes[0, method_idx].set_title(label, fontsize=12, fontweight='bold', pad=10)
    
    # No row labels needed since frequencies are shown as titles on True Field plots

    # Adjust layout to provide space for colorbars on the right and prevent overflow
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, left=0.12, right=0.84, hspace=0.3, wspace=0.4)

    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_str = "_".join([f"{int(f)}Hz" for f in freq_hz_labels])
    architecture_short = architecture if architecture else "unknown"

    filename = f"paper_fig_{architecture_short}_freqs_{freq_str}_z{z_slice_idx}_w{guidance_scale}__{timestamp}.pdf"
    save_path = os.path.join(save_dir, filename)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save figure with improved PDF compatibility

    save_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    from matplotlib.backends.backend_pdf import PdfPages

    # ... after layout adjustments
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    print(f"\nFigure saved to: {save_path}")

    # fig.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    
    # print(f"\nFigure saved to: {save_path}")

    plt.show()

    return save_path

def generate_atf_plots(model_path, source_indices=[0], mic_indices=[665], 
                      save_dir="paper_figures", guidance_scale=1.0, num_timesteps=10):
    """
    Generate clean ATF comparison plots: Reference (FSMPAE), Reference (KRR), Your Model.
    Efficient version that only generates what's needed for specific source/mic combinations.
    
    Args:
        model_path: Path to your model checkpoint
        source_indices: List of source indices to plot (e.g., [0, 11, 22])  
        mic_indices: List of microphone indices to plot (e.g., [665, 156, 423])
        save_dir: Directory to save ATF plots
        guidance_scale: Guidance scale to use (default 1.0)
        num_timesteps: Number of timesteps for generation (default 10)
    
    Returns:
        str: Path to the ATF plots directory
    """
    print(f"\n=== ATF COMPARISON PLOTS ===")
    print(f"Your Model: {get_model_name(model_path)}")
    print(f"Sources: {source_indices} (test indices: {[s+922 for s in source_indices]})")
    print(f"Microphones: {mic_indices}")
    print(f"Guidance scale: {guidance_scale}")
    
    # 1. Load reference results (lightweight - just the prediction tensors)
    print("\n1. Loading reference results...")
    fsmpae_results = None
    krr_results = None
    
    if os.path.exists(FSMPAE_RESULTS_PATH):
        fsmpae_results = torch.load(FSMPAE_RESULTS_PATH, weights_only=False)
        print(f"  FSMPAE loaded: {fsmpae_results.shape}")
    else:
    #assert error
        assert "WRONG PATH"

        
    if os.path.exists(KRR_RESULTS_PATH):
        krr_results = torch.load(KRR_RESULTS_PATH, weights_only=False) 
        print(f"  KRR loaded: {krr_results.shape}")
    else:
        assert "WRONG PATH"

    
    # 2. Load ground truth (lightweight - reuse existing approach)
    print("\n2. Loading ground truth...")
    import sys
    sys.path.append(AUTOENCODER_PATH)
    import AUTOENCODER.src.dataset as autoencoder_dataset
    from AUTOENCODER.src.configs import config_FSMPAE_10026
    
    ref_config = config_FSMPAE_10026.copy()
    original_cwd = os.getcwd()
    os.chdir('AUTOENCODER')
    idataset = autoencoder_dataset.ATFdataset(config=ref_config)
    data = idataset.Data
    os.chdir(original_cwd)
    
    dataset_name = ref_config['dataset'][0]
    atf_mag_gt = data['test']['atf_mag'][dataset_name]
    print(f"  Ground truth loaded: {atf_mag_gt.shape}")
    
    # 3. Load your model (only what we need)
    print("\n3. Loading your model...")
    checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)
    set_encoder, unet_3d, ode_3d, architecture = model_factory(config, model_states_cfg, device)
    freq_up_to = config['model'].get('freq_up_to')
    print(f"  Your model loaded, freq_up_to: {freq_up_to}")
    
    # 4. Load your data (minimal setup)
    src_split = config['data']['src_splits']
    
    train_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    # 5. Setup simulator and microphone selection
    simulator = EulerSimulator(ode=ode_3d)
    simulator.ode.guidance_scale = guidance_scale
    
    # Load microphone selection matrix (same as references)
    idx_mes_pos_mat = np.load(MIC_SELECTION_PATH)
    
    # 6. Create frequency axis
    fs = ref_config['fs']  # 2000 Hz
    freq_axis = np.arange(1, ATF_FFTLEN_ALGN // 2 + 1) / ATF_FFTLEN_ALGN * fs
    freq_axis = freq_axis[:freq_up_to]  # Match your model's frequency range
    
    # 7. Create output directory
    atf_output_dir = os.path.join(save_dir, "atf_comparisons")
    os.makedirs(atf_output_dir, exist_ok=True)
    
    # 8. Generate ATF plots for each source
    for src_idx in source_indices:
        print(f"\n4. Processing source {src_idx} (test index {src_idx + 922})...")
        
        # Generate your model's prediction for this source only
        print("  Generating your model's prediction...")
        z_true = test_sampler.cubes[src_idx].unsqueeze(0).to(device)
        src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)
        
        # Use same microphone selection as references
        source_specific_indices = idx_mes_pos_mat[:ATF_M, src_idx]
        obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)
        
        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)
        
        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
        obs_mask = torch.ones(1, ATF_M, dtype=torch.bool, device=device)
        
        # Generate prediction
        x0 = torch.randn_like(z_true)
        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

        ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
        ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

        x1_recon = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                     obs_mask=obs_mask, pooled_context=pooled_context,
                                     freq_contexts=freq_contexts,
                                     paste_observations=True, obs_indices=obs_indices)
        
        # Denormalize your model's prediction
        gen_cube_denorm = (x1_recon * std + mean)
        
        # Create ATF plot for this source (all mics in one figure)
        print("  Creating ATF comparison plot...")
        fig, axes = plt.subplots(len(mic_indices), 1, figsize=(18, 5 * len(mic_indices)))
        if len(mic_indices) == 1:
            axes = [axes]
        plt.subplots_adjust(hspace=0.5)
        
        for i, mic_idx in enumerate(mic_indices):
            ax = axes[i]
            
            # Convert flat mic index to 3D coordinates  
            nx, ny, nz = 11, 11, 11
            iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
            
            # Extract ATF values for this microphone
            gt_atf = atf_mag_gt[mic_idx, :freq_up_to, src_idx].cpu().numpy()
            your_atf = gen_cube_denorm[0, :, iz, iy, ix].cpu().numpy()

            print("SECOND GRound truth", z_true.shape)
            
            # Plot ground truth and your model
            ax.plot(freq_axis, gt_atf, 'k--', label="Ground Truth", linewidth=2)
            ax.plot(freq_axis, your_atf, 'r-', label="SF-Flow", linewidth=1.5)
            
            # Add reference methods if available
            if fsmpae_results is not None:
                fsmpae_atf = fsmpae_results[mic_idx, :freq_up_to, src_idx].cpu().numpy()
                ax.plot(freq_axis, fsmpae_atf, 'b-', label="AE", linewidth=1.5)
                
            if krr_results is not None:
                krr_atf = krr_results[mic_idx, :freq_up_to, src_idx].cpu().numpy()
                ax.plot(freq_axis, krr_atf, 'g-', label="KRR", linewidth=1.5)
            
            # Format plot
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=16)
            ax.set_xlabel("Frequency (Hz)", fontsize=16)
            ax.set_ylabel("Magnitude (dB)", fontsize=16)
            
            # Set better x-axis ticks for paper clarity
            # ax.set_xlim([12, 315])  # Set limits to your actual frequency range
            ax.set_xticks([20,30,40,50, 100, 200, 312.5])  # More informative tick positions
            ax.set_xticklabels(['', '', '', '50', '100', '200', '312.5'], fontsize=12)  # Clear labels
            ax.tick_params(axis='both', which='major', labelsize=16)  # Increase both x and y tick label sizes
            
            # Get microphone coordinates for title
            mic_coord = grid_xyz[mic_idx].cpu().numpy()
            # ax.set_title(f"ATF Mic {mic_idx}: ({mic_coord[0]:.2f}, {mic_coord[1]:.2f}, {mic_coord[2]:.2f}) m")
        
        # plt.suptitle(f"ATF Comparison - Source {src_idx+922}", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save plot
        filename = f"ATF_comparison_src{src_idx+922:04d}.pdf"
        filepath = os.path.join(atf_output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {filename}")
    
    print(f"\nATF comparison plots saved to: {atf_output_dir}")
    return atf_output_dir

if __name__ == '__main__':



    # SF FOR MULTI PLOTTING ONLY FM
    model_path = MODEL_LOAD_PATH
    GENERATE_SF_PLOTS = False
    GENERATE_SF_PLOTS_V2 = True
    GENERATE_ATF_PLOTS = False  # Set to True to generate ATF comparison plots


    print("Starting paper figures generation...")
    
    # 1. Generate sound field figures
    if GENERATE_SF_PLOTS:

        freq_idx_to_plot = [15,15,15]  # Frequency bin indices
        z_slice_idx = 5  # Z-slice index
        guidance_scale = 1  # Guidance scale
        # Range for number of microphones
        num_timesteps = 10  # Number of generation timesteps
        random_mics_per_freq = True  # Different mic indices per frequency
        random_M_per_freq = True  # Different M values for each frequency
        M_seed = 120  # Separate seed for M sampling
        M_range = [5, 5]

        save_path = generate_SFfigures_FM(
            model_path=model_path,
            freq_idx_to_plot=freq_idx_to_plot,
            z_slice_idx=z_slice_idx,
            guidance_scale=guidance_scale,
            M_range=M_range,
            num_timesteps=num_timesteps,
            save_dir="paper_figures",
            random_mics_per_freq=random_mics_per_freq,
            random_M_per_freq=random_M_per_freq,
            M_seed=M_seed
        )

        print(f"Sound field figures saved to: {save_path}")

    if GENERATE_SF_PLOTS_V2:
        # SF FOR ALL METHODS
        SPARSE_M = 5
        srcind = [0] # 88, 66, 0, 12,
        guidance_scale = 1  # Guidance scale
        freq_idx_to_plot = [19]  # 4, 15
        z_slice_idx = 10
        num_timesteps = 10

        MIC_SELECTION_PATH = "./idx_mes_pos_s1024_m1331.npy"
        idx_mes_pos_mat = np.load(MIC_SELECTION_PATH)
        generate_SFfigures_FM_V2(model_path=model_path,
                                 srcind = srcind,
                freq_idx_to_plot=freq_idx_to_plot,
                z_slice_idx=z_slice_idx,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                save_dir="paper_figures",
                idx_mes_pos_mat=idx_mes_pos_mat)


    #
    # def generate_SFfigures_baselines():
    #     fsmpae_results = torch.load(FSMPAE_RESULTS_PATH, weights_only=False)
    #     krr_results = torch.load(KRR_RESULTS_PATH, weights_only=False)
    #
    #


    # 2. Generate ATF comparison plots (optional)
    if GENERATE_ATF_PLOTS:
        # ATF comparison configuration
        atf_model_path = model_path  # Can use different model for ATF plots
        atf_source_indices = [0]  # Specific source indices to plot
        atf_mic_indices = [665]  # Specific microphone indices to plot
        atf_guidance_scale = 1.0  # Guidance scale for ATF plots
        atf_num_timesteps = 10  # Timesteps for ATF generation

        print("\n" + "="*50)
        atf_output_dir = generate_atf_plots(
            model_path=atf_model_path,
            source_indices=atf_source_indices,
            mic_indices=atf_mic_indices,
            save_dir="paper_figures",
            guidance_scale=atf_guidance_scale,
            num_timesteps=atf_num_timesteps
        )

        print(f"📁 ATF comparisons saved to: {atf_output_dir}")

