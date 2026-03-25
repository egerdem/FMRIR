import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import random
from tqdm import tqdm

from fm_utils import (ATF3DSampler, CFGVectorFieldODE_3D, EulerSimulator, EulerMaruyamaSimulator,
                      CFGVectorFieldODE_3D_V2, SetEncoder, # DDPMScheduler
                      SetEncoder_v12, CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d,
                      CrossAttentionUNet3D_v3, DDPM_ODE_Sampler,
                      get_model_info, print_model_info)

from model_paths import MULTI_MODEL_PATHS, MODEL_LOAD_PATH


def _remap_red3d_keys(state_dict):
    """Remap pre-parametric RED3d checkpoint keys to ModuleList naming if needed.
    Detects old-style keys (enc1_res, enc2_res, ...) and remaps them.
    New checkpoints already have enc_res.0, enc_res.1, ... and are returned unchanged.
    """
    # Only remap if old-style keys are present
    if not any(k.startswith('enc1_res.') for k in state_dict):
        return state_dict
    remap = {
        'enc1_res.': 'enc_res.0.', 'enc2_res.': 'enc_res.1.',
        'enc1_attn.': 'enc_attn.0.', 'enc2_attn.': 'enc_attn.1.',
        'up1.': 'dec_up.0.', 'up2.': 'dec_up.1.',
        'dec1_res.': 'dec_res.0.', 'dec2_res.': 'dec_res.1.',
        'dec1_attn.': 'dec_attn.0.', 'dec2_attn.': 'dec_attn.1.',
    }
    new_sd = {}
    remapped = 0
    for k, v in state_dict.items():
        for old, new in remap.items():
            if k.startswith(old):
                k = new + k[len(old):]
                remapped += 1
                break
        new_sd[k] = v
    print(f"  [compat] Remapped {remapped} RED3d keys to ModuleList naming")
    return new_sd


# class DDIMSampler:
#     """
#     A deterministic sampler for a trained DDPM using the DDIM update rule.
#     """
#
#     def __init__(self, model, set_encoder, scheduler):
#         self.model = model
#         self.set_encoder = set_encoder
#         self.scheduler = scheduler
#         self.num_timesteps = scheduler.num_timesteps
#
#     @torch.no_grad()
#     def _get_predicted_original_sample(self, noisy_sample, t, predicted_noise):
#         """ Calculates the predicted clean sample (x0) from the predicted noise. """
#         sqrt_alpha_t = self.scheduler.sqrt_alphas_cumprod.to(noisy_sample.device)[t].view(-1, 1, 1, 1, 1)
#         sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod.to(noisy_sample.device)[t].view(-1, 1, 1,
#                                                                                                               1, 1)
#
#         # Formula: x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
#         pred_original_sample = (noisy_sample - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
#         return pred_original_sample
#
#     @torch.no_grad()
#     def step(self, xt, t, guidance_scale=1.0, **kwargs):
#         """
#         Performs a single DDIM denoising step from t to t-1.
#         """
#         device = xt.device
#
#         # Prepare discrete and continuous time tensors
#         t_discrete = torch.full((xt.shape[0],), t, dtype=torch.long, device=device)
#         t_continuous = (t_discrete.float() / self.num_timesteps).view(-1, 1, 1, 1, 1)
#
#         # --- Classifier-Free Guidance ---
#         # Get guided prediction
#         guided_predicted_noise = self.model(xt, t_continuous, **kwargs)
#
#         # Get unguided prediction
#         null_context = self.set_encoder.y_null_token.squeeze(1).expand(xt.shape[0], -1)
#         kwargs_unguided = kwargs.copy()
#         if "pooled_context" in kwargs_unguided:
#             kwargs_unguided["pooled_context"] = null_context
#
#         if "context" in kwargs_unguided:
#             y_tokens = kwargs_unguided["context"]
#             null_tokens = self.set_encoder.y_null_token.expand(xt.shape[0], y_tokens.shape[1], -1)
#             kwargs_unguided["context"] = null_tokens
#
#
#         unguided_predicted_noise = self.model(xt, t_continuous, **kwargs_unguided)
#
#         # Combine predictions
#         predicted_noise = (1 - guidance_scale) * unguided_predicted_noise + guidance_scale * guided_predicted_noise
#
#         # --- DDIM Update Rule ---
#         # 1. Get alphas for current and previous timesteps
#         alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
#         alpha_bar_t_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
#
#         # 2. Predict the original sample (x0) using the final predicted noise
#         pred_x0 = self._get_predicted_original_sample(xt, t_discrete, predicted_noise)
#
#         # 3. Calculate the direction pointing to x0
#         # This term uses sqrt(1 - alpha_bar_{t-1})
#         pred_dir_xt = torch.sqrt(1. - alpha_bar_t_prev) * predicted_noise
#
#         # 4. Calculate the final sample x_{t-1}
#         xt_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
#
#         return xt_prev

# Import unified LSD function for consistency with unified_evaluation.py
def calculate_lsd_unified(estimation, ground_truth, freq_dim=1, return_mean_only=True):
    """
    Unified LSD calculation that works for both 3D spatial and microphone-based data.
    Same as unified_evaluation.py for consistency.

    Args:
        estimation: Model prediction (should be in dB domain)
        ground_truth: Ground truth (should be in dB domain)
        freq_dim: Dimension along which frequency is stored (1 for [B,F,Z,Y,X], 0 for [F,Z,Y,X])
        return_mean_only: If True, returns only the mean LSD. If False, returns (mean, per_position_lsd) tuple
                         where per_position_lsd contains LSD values for each position.

    Returns:
        If return_mean_only=True: Mean LSD value in dB
        If return_mean_only=False: Tuple of (mean LSD, per-position LSD values)
    """
    squared_error = (estimation - ground_truth) ** 2
    lsd_per_position = torch.sqrt(torch.mean(squared_error, dim=freq_dim))  # [positions]
    mean_lsd = torch.mean(lsd_per_position)
    
    if return_mean_only:
        return mean_lsd
    else:
        # Return both mean and per-position values
        return mean_lsd, lsd_per_position.flatten()  # Ensure 1D tensor for concatenation

def calculate_slice_metrics(pred_cube, gt_cube, freq_idx, z_slice_idx):
    """
    Calculate MSE and LSD for a specific slice.
    
    Args:
        pred_cube: Predicted cube [freq, z, y, x] (denormalized, dB domain)
        gt_cube: Ground truth cube [freq, z, y, x] (denormalized, dB domain)
        freq_idx: Frequency index to extract
        z_slice_idx: Z-slice index to extract
        
    Returns:
        dict: {'mse': float, 'lsd': float}
    """
    # Extract the specific slice
    pred_slice = pred_cube[freq_idx, z_slice_idx, :, :]  # [y, x]
    gt_slice = gt_cube[freq_idx, z_slice_idx, :, :]      # [y, x]
    
    # MSE on the slice (in dB domain)
    slice_mse = torch.mean((pred_slice - gt_slice) ** 2).item()
    
    # LSD for the slice - we need frequency dimension, so add it back
    pred_slice_freq = pred_cube[:, z_slice_idx, :, :].unsqueeze(0)  # [1, freq, y, x]
    gt_slice_freq = gt_cube[:, z_slice_idx, :, :].unsqueeze(0)      # [1, freq, y, x]

    print(pred_slice_freq.shape, gt_slice_freq.shape)
    
    # LSD calculation (already in dB domain)
    slice_lsd = calculate_lsd_unified(pred_slice_freq, gt_slice_freq, freq_dim=1).item()
    
    return {'mse': slice_mse, 'lsd': slice_lsd}
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note: Reference AUTOENCODER model cannot generate full 3D cubes for MSE comparison
# It only predicts at sparse microphone positions, not full spatial grids
# For reference comparison, use unified_evaluation.py instead

SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for GPU
np.random.seed(SEED)
random.seed(SEED)

# --- Multi-Model Configuration ---
VISUALIZE_slice = True  # Set to True to visualize a single model's slice
COMPARE = False     # line errors but only for few num_example sources

# Custom model names for legends (optional, will auto-generate if None)
MODEL_NAMES = [
]

M_range = None
# SIGMA_SDE = 0.
M_range = [5, 20]
num_examples = 5
num_timesteps = 10
guidance_scales = [1]
freq_idx_to_plot = 10  # Pick a frequency channel to visualize
z_slice_idx_to_plot = 5
# SRC_IND = [0]
SRC_IND = None

# Option to exclude outermost boundary positions from MSE/LSD calculation
EXCLUDE_BOUNDARY = False  # Set to True to exclude outermost positions
BOUNDARY_THICKNESS = 1   # Number of boundary layers to exclude (1 = outermost layer only)

data_path = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"


def create_interior_mask(cube_shape, boundary_thickness=1):
    """
    Create a boolean mask to select interior positions, excluding boundary layers.

    Args:
        cube_shape: Shape of the cube (freq, z, y, x) or (z, y, x)
        boundary_thickness: Number of boundary layers to exclude

    Returns:
        mask: Boolean mask where True = interior position, False = boundary position
        interior_indices: Flat indices of interior positions
    """
    if len(cube_shape) == 4:  # (freq, z, y, x)
        _, nz, ny, nx = cube_shape
    else:  # (z, y, x)
        _, _, nz, ny, nx = cube_shape

    # Create 3D mask for interior positions
    mask_3d = torch.zeros(nz, ny, nx, dtype=torch.bool)

    # Set interior region to True
    z_start, z_end = boundary_thickness, nz - boundary_thickness
    y_start, y_end = boundary_thickness, ny - boundary_thickness
    x_start, x_end = boundary_thickness, nx - boundary_thickness

    mask_3d[z_start:z_end, y_start:y_end, x_start:x_end] = True

    # Convert to flat indices
    interior_indices = torch.nonzero(mask_3d.flatten()).squeeze(-1)

    total_positions = nz * ny * nx
    interior_positions = len(interior_indices)
    boundary_positions = total_positions - interior_positions

    print(f"Interior mask: {interior_positions}/{total_positions} positions "
          f"({boundary_positions} boundary positions excluded)")

    return mask_3d, interior_indices

def model_factory(config, model_states_cfg, device, SIGMA_SDE=0.1):
    """
    Reads the config and returns the correctly instantiated and loaded models
    and their corresponding inference wrapper.
    """
    model_cfg = config['model']
    architecture = model_cfg.get('architecture_version')
    setversion = model_cfg.get('setencoder_version')
    fm_or_diff = model_cfg.get('FM_vs_Diff', 'flow_matching')

    # Derive the actual channel count from the subband.  freq_from defaults to 0
    # for all pre-existing checkpoints that don't have this key in their config.
    freq_from = model_cfg.get('freq_from', 0)
    num_freqs = model_cfg['freq_up_to'] - freq_from

    # --- 1. Instantiate SetEncoder ---
    freq_ctx = model_cfg.get('freq_ctx', False)
    if setversion == "v3":
        print("--- Creating set encoder v3 ---")
        coord_dim = model_cfg.get('coord_dim', 3)
        num_ff_coord = model_cfg.get('num_ff_coord', 0)
        set_encoder = SetEncoder(
            num_freqs=num_freqs, d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'], num_layers=model_cfg['num_encoder_layers'],
            coord_dim=coord_dim, num_ff_coord=num_ff_coord, use_freq_ctx=freq_ctx
        ).to(device)
    else: # Fallback to v12
        print("--- Creating set encoder v12 ---")
        coord_dim = model_cfg.get('coord_dim', 3)
        num_ff_coord = model_cfg.get('num_ff_coord', 0)
        set_encoder = SetEncoder_v12(
            num_freqs=num_freqs, d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'], num_layers=model_cfg['num_encoder_layers'],
            coord_dim=coord_dim, num_ff_coord=num_ff_coord, use_freq_ctx=freq_ctx
        ).to(device)

    # --- 2. Instantiate Main Model (U-Net/DiT) ---
    freq_channel_bias = model_cfg.get('freq_channel_bias', False)
    freq_film = model_cfg.get('freq_film', False)

    if architecture == "v3_attention":
        print("--- Creating (v3) architecture with Self-Attention ---")
        main_model = CrossAttentionUNet3D_v3(
            in_channels=num_freqs, out_channels=num_freqs,
            channels=model_cfg['channels'], d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'], input_size=11,
            freq_channel_bias=freq_channel_bias, freq_film=freq_film, freq_ctx=freq_ctx
        ).to(device)
    elif architecture == "v2_residual_context":
        print("--- Creating (v2) architecture ---")
        main_model = CrossAttentionUNet3D_RED3d(
            in_channels=num_freqs, out_channels=num_freqs,
            channels=model_cfg['channels'], d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            freq_channel_bias=freq_channel_bias, freq_film=freq_film, freq_ctx=freq_ctx
        ).to(device)
    # Add other architectures like v1_legacy as needed...
    elif architecture == "v1_legacy" or architecture is None:
        print("--- Creating v1 architecture: standard 3d unet ---")
        main_model = CrossAttentionUNet3D(
            in_channels=num_freqs, out_channels=num_freqs,
            channels=model_cfg['channels'], d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            freq_channel_bias=freq_channel_bias, freq_film=freq_film, freq_ctx=freq_ctx
        ).to(device)

    elif architecture == "v4_DiT":
        # Instantiate the old U-Net and ODE wrapper for old checkpoints
        main_model = DiffusionTransformer3D(
            in_channels=num_freqs,
            out_channels=num_freqs,
            patch_size=model_cfg['patch_size'],
            depth=model_cfg.get('dit_depth', 12),
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_sde_wrapper = CFGVectorFieldODE_DiT_3D(unet=main_model, set_encoder=set_encoder)

    # --- 3. Instantiate the Correct Inference Wrapper ---
    if fm_or_diff == 'score_matching':
        print("--- Using Denoising Diffusion (SDE) Wrapper ---")
        # ode_sde_wrapper = GenerativeSDE(
        #     noise_predictor_network=main_model, set_encoder=set_encoder,
        #     config=config, sigma_sde=SIGMA_SDE
        # )
        ode_sde_wrapper = None

    elif fm_or_diff == 'flow_matching' and architecture != "v4_DiT":
        print("--- Using Flow Matching ODE Wrapper ---")
        print("inference.py")
        # CFGVectorFieldODE_3D_V2 is compatible with both v2 and v3 U-Nets

        if architecture == "v1_legacy" or architecture == None:
            print("v1 legacy")
            ode_sde_wrapper = CFGVectorFieldODE_3D(unet=main_model, set_encoder=set_encoder)
        else:
            ode_sde_wrapper = CFGVectorFieldODE_3D_V2(unet=main_model, set_encoder=set_encoder)

    # --- Load weights ---
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])

    if architecture == "v4_DiT":
        main_model.load_state_dict(model_states_cfg['dit'])
    else:
        main_model.load_state_dict(_remap_red3d_keys(model_states_cfg['unet']))

    set_encoder.eval()
    main_model.eval()

    return set_encoder, main_model, ode_sde_wrapper, architecture

# Helper function to plot a 3D box
def plot_room_box(ax, dimensions):
    w, d, h = dimensions  # width, depth, height
    # Define the 8 corners of the box
    corners = [
        [0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
        [0, 0, h], [w, 0, h], [w, d, h], [0, d, h]
    ]
    corners = np.array(corners)
    # Define the 6 faces of the box
    faces = [
        [corners[0], corners[1], corners[5], corners[4]],  # Front
        [corners[2], corners[3], corners[7], corners[6]],  # Back
        [corners[0], corners[3], corners[7], corners[4]],  # Left
        [corners[1], corners[2], corners[6], corners[5]],  # Right
        [corners[0], corners[1], corners[2], corners[3]],  # Bottom
        [corners[4], corners[5], corners[6], corners[7]]  # Top
    ]
    # Create and add the 3D polygon collection
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors='cyan', linewidths=1, edgecolors='darkblue', alpha=0.05
    ))
    # Set axis limits
    ax.set_xlim(0, w);
    ax.set_ylim(0, d);
    ax.set_zlim(0, h)


def run_single_inference(set_encoder, unet_3d, ode_3d, z_true, src_xyz, grid_xyz, M_range,
                         guidance_scales, num_timesteps, mean, std, device, exclude_boundary=False, boundary_thickness=1):
    """Run inference for a single example and return MSE results"""
    # Create a sparse observation set
    M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
    obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
    obs_xyz_abs = grid_xyz[obs_indices]
    obs_coords_rel = obs_xyz_abs - src_xyz

    z_flat = z_true.view(z_true.shape[1], -1)
    obs_values = z_flat[:, obs_indices].transpose(0, 1)

    # Batchify for the set encoder
    obs_coords_rel = obs_coords_rel.unsqueeze(0)
    obs_values = obs_values.unsqueeze(0)
    obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

    # Ground truth for comparison
    z_true_denorm = (z_true * std + mean)

    # Use the provided ODE instance
    simulator = EulerSimulator(ode=ode_3d)

    mse_results = []

    for w in guidance_scales:
        # Start from pure noise
        x0 = torch.randn_like(z_true)
        xt = x0.clone()

        # Get conditioning tokens
        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

        ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
        ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

        # Set the guidance scale
        simulator.ode.guidance_scale = w

        # Run simulation - pass pooled_context for v2 models
        x1_recon = simulator.simulate(xt, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                      obs_mask=obs_mask, pooled_context=pooled_context,
                                      freq_contexts=freq_contexts,
                                      paste_observations=True, obs_indices=obs_indices)

        # Calculate BOTH MSE and LSD for comprehensive comparison
        x1_recon_denorm = (x1_recon * std + mean)
        z_true_denorm = (z_true * std + mean)

        if exclude_boundary:
            # Create interior mask for this cube
            mask_3d, interior_indices = create_interior_mask(z_true.shape, boundary_thickness)

            # Flatten cubes and select interior positions only
            x1_flat = x1_recon.view(x1_recon.shape[1], -1)  # [freq, 1331]
            z_flat = z_true.view(z_true.shape[1], -1)       # [freq, 1331]
            x1_flat_denorm = x1_recon_denorm.view(x1_recon_denorm.shape[1], -1)
            z_flat_denorm = z_true_denorm.view(z_true_denorm.shape[1], -1)

            # Select interior positions
            x1_interior = x1_flat[:, interior_indices]      # [freq, interior_count]
            z_interior = z_flat[:, interior_indices]        # [freq, interior_count]
            x1_interior_denorm = x1_flat_denorm[:, interior_indices]
            z_interior_denorm = z_flat_denorm[:, interior_indices]

            # Calculate metrics on interior only
            mse = torch.mean((x1_interior_denorm - z_interior_denorm) ** 2).item()
            lsd_db = calculate_lsd_unified(x1_interior_denorm.T, z_interior_denorm.T, freq_dim=1).item()
        else:
            # Full cube calculation
            mse = torch.mean((x1_recon_denorm - z_true_denorm) ** 2).item()
            lsd_db = calculate_lsd_unified(x1_recon_denorm, z_true_denorm, freq_dim=1).item()

        # Return both metrics
        mse_results.append({'mse': mse, 'lsd': lsd_db})

    return mse_results, M


def plot_dual_metric_comparison(all_results, model_names, guidance_scales, save_path=None, block=True):
    """Create dual MSE+LSD comparison plots across multiple models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # Store results for printing under plots
    results_text_lines = []

    for i, (model_name, results_data) in enumerate(zip(model_names, all_results)):
        # Extract MSE and LSD data
        mse_data = [[result['mse'] for result in example] for example in results_data]
        lsd_data = [[result['lsd'] for result in example] for example in results_data]

        mse_array = np.array(mse_data)
        lsd_array = np.array(lsd_data)

        mean_mse = np.mean(mse_array, axis=0)
        std_mse = np.std(mse_array, axis=0)
        mean_lsd = np.mean(lsd_array, axis=0)
        std_lsd = np.std(lsd_array, axis=0)

        # MSE plot (only add label to first plot for shared legend)
        ax1.plot(guidance_scales, mean_mse, 'o-', label=model_name, color=colors[i], linewidth=2, markersize=6)
        ax1.fill_between(guidance_scales, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color=colors[i])

        # LSD plot (no label to avoid duplicate legend)
        ax2.plot(guidance_scales, mean_lsd, 'o-', color=colors[i], linewidth=2, markersize=6)
        ax2.fill_between(guidance_scales, mean_lsd - std_lsd, mean_lsd + std_lsd, alpha=0.2, color=colors[i])

        # Prepare results text for this model
        mse_results_str = ", ".join([f"w={w}: {mse:.3f}" for w, mse in zip(guidance_scales, mean_mse)])
        lsd_results_str = ", ".join([f"w={w}: {lsd:.3f}" for w, lsd in zip(guidance_scales, mean_lsd)])
        results_text_lines.append(f"{model_name}:")
        results_text_lines.append(f"  MSE: {mse_results_str}")
        results_text_lines.append(f"  LSD: {lsd_results_str}")
        results_text_lines.append("")  # Empty line between models

    # Configure MSE plot
    cube_label = "Interior Only" if EXCLUDE_BOUNDARY else "Full Cube"
    ax1.set_xlabel('Guidance Scale (w)', fontsize=12)
    ax1.set_ylabel(f'MSE ({cube_label})', fontsize=12)
    ax1.set_title('MSE: ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Configure LSD plot
    ax2.set_xlabel('Guidance Scale (w)', fontsize=12)
    ax2.set_ylabel(f'LSD ({cube_label}) [dB]', fontsize=12)
    ax2.set_title('LSD: ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Add single shared legend at the top center of the figure
    fig.legend(ax1.get_legend_handles_labels()[0], ax1.get_legend_handles_labels()[1],
              loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(len(model_names), 2),
              fontsize=9, frameon=True, fancybox=True, shadow=True)

    # Add text box with statistics
    stats_str = f'Averaged over {len(all_results[0])} examples\nShaded areas show ±1 std deviation'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, stats_str, ha='center', fontsize=10, bbox=props)

    # Add results text under the plots
    results_text = "\n".join(results_text_lines[:-1])  # Remove last empty line
    fig.text(0.5, 0.08, results_text, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.25)  # Make room for legend and results text

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Dual metric comparison plot saved to: {save_path}")

    plt.show(block=block)

def get_model_name(model_path):
    """Extract model name from path"""
    return model_path.split("artifacts/")[1].split("/")[0]

def load_model_and_config(model_path, device):
    """Load model checkpoint and extract configuration"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']
    return checkpoint, config, model_states_cfg


if __name__ == '__main__':

    # --- Main Execution Logic ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load minimal config for data samplers (needed for both modes) ---
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
    config = checkpoint.get('config', {})
    data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"  # Override with local
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    freq_from  = config['model'].get('freq_from', 0)  # 0 for all pre-subband checkpoints

    if VISUALIZE_slice:
        MODEL_NAME = get_model_name(MODEL_LOAD_PATH)

        print(f"=== SINGLE MODEL MODE ===")
        print(f"Model artifact: {MODEL_NAME}")

        training_params = config.get('training', {})
        sigma_train = training_params.get('sigma')
        model_mode = config["training"].get('model_mode', "spatial")

        print("\n--- Automatically Configured from Loaded Model ---")
        print(f"  Training Sigma: {sigma_train:.4f}")
        print("--------------------------------------------------\n")

    model_states_cfg = checkpoint['model_states']
    # --- KEY CHANGE: Detect Model Type ---
    # Check if the checkpoint has keys associated with the 3D model
    # is_3d_model = 'set_encoder' in model_states_cfg

    # if is_3d_model:
    print("--- Detected 3D Conditional Generation Model ---")

    # --- CONFIGURATION FOR VISUALIZATION ---
    PLOT_3D = True  # Set to True to generate a separate 3D scatter plot
    data_config_path = os.path.join(data_dir, "config.json")
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    room_dim = data_config.get('room_dim')
    center = data_config.get('center')

    # --- 1. Data Loading (Shared for both modes) ---
    # Create train sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, normalize=True,
        freq_up_to=freq_up_to, freq_from=freq_from,
    model_name = MODEL_NAME)
    # Create test sampler with raw data
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, normalize=False,
        freq_up_to=freq_up_to, freq_from=freq_from,
    model_name = MODEL_NAME)
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()

    print(f"Loaded Stats from 3D Training Set: Mean={mean:.4f}, Std={std:.4f}")

    if VISUALIZE_slice:
        # --- Set M_range ---
        if M_range is None:
            M_range = config['training'].get('M_range')

        # --- 2. Use the factory to get the correct models ---
        set_encoder, unet_3d, ode_3d, _ = model_factory(config, model_states_cfg, device)

        # --- 3. Print Model Information ---
        print(f"\n--- Model Architecture: {MODEL_NAME} ---")
        print_model_info(set_encoder, "SetEncoder")
        print_model_info(unet_3d, "UNet3D")

        # Combined model info
        set_encoder_info = get_model_info(set_encoder, "SetEncoder")
        unet_info = get_model_info(unet_3d, "UNet3D")
        total_params = set_encoder_info['total_params'] + unet_info['total_params']
        total_size_mb = set_encoder_info['model_size_mb'] + unet_info['model_size_mb']

        fm_vs_diff = config['model'].get('FM_vs_Diff')

        print(f"=== Combined Model ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Total size: {total_size_mb:.2f} MB")
        print("=" * 20)

        # --- 4. Original Inference & Visualization ---
        # if config['model'].get('FM_vs_Diff') == 'flow_matching' or config['model'].get('FM_vs_Diff') is None:
        #     print("--- Using Flow Matching ODE Simulator ---")
        #     simulator = EulerSimulator(ode=ode_3d)
        # elif config['model'].get('FM_vs_Diff') == 'score_matching':
        #     print("--- Using Denoising Diffusion (SDE) Simulator ---")
        #     simulator = EulerMaruyamaSimulator(sde=ode_3d)
            # simulator = None


        # --- SETUP THE PLOT GRID ---
        # Add an extra column at the far left for a 3D snapshot view
        num_cols = 3 + len(guidance_scales)
        # Increase figure height to accommodate title and metrics
        fig, axes = plt.subplots(num_examples, num_cols, figsize=(4.5 * num_cols, 4.5 * num_examples), squeeze=False)

        # Split long model name for better title formatting
        model_name_parts = MODEL_NAME.split('_')
        title_line1 = '_'.join(model_name_parts[:4])  # First 4 parts
        title_line2 = '_'.join(model_name_parts[4:])  # Rest of the parts

        # Two-line title with smaller font
        fig.suptitle(
            f"3D Conditional Generation (Freq Idx={freq_idx_to_plot}, Z-Slice={z_slice_idx_to_plot})\n{title_line1}\n{title_line2}",
            fontsize=12, y=0.98)

        # Move validation loss info closer to title
        best_val_loss = checkpoint.get('best_val_loss', {})
        best_iteration = checkpoint.get('best_iteration', {})
        fig.text(0.5, 0.90, f"Best Val Loss: {best_val_loss:.4f} at Iteration {best_iteration}",
                ha='center', fontsize=10)

        center_np = np.array(center)

        for row in range(num_examples):
        # Get a random ground truth sample

            if SRC_IND is not None:
             srcind = SRC_IND
             z_true, src_xyz = test_sampler.cubes[srcind[0]].unsqueeze(0).to(device), test_sampler.source_coords[
                 srcind[0]].unsqueeze(0).to(device)
            else:
                z_true, src_xyz, srcind = test_sampler.sample(1)
                z_true, src_xyz = z_true.to(device), src_xyz.to(device)

            # --- Create a sparse observation set on the fly ---
            M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
            obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
            obs_xyz_abs = grid_xyz[obs_indices]
            obs_coords_rel = obs_xyz_abs - src_xyz

            z_flat = z_true.view(z_true.shape[1], -1)
            obs_values = z_flat[:, obs_indices].transpose(0, 1)

            # Batchify for the set encoder
            obs_coords_rel = obs_coords_rel.unsqueeze(0)
            obs_values = obs_values.unsqueeze(0)
            obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

            # --- Plot Ground Truth and Sparse Input ---
            z_true_denorm = (z_true * std + mean)
            gt_cube_raw = z_true_denorm[0, freq_idx_to_plot].cpu().numpy()  # This is the (11, 11, 11) cube

            gt_slice = gt_cube_raw[z_slice_idx_to_plot, :, :]  # Select the specific slice

            axes[row, 1].imshow(gt_slice, origin='lower', cmap='viridis', vmin=gt_slice.min(), vmax=gt_slice.max())
            axes[row, 1].set_title("True (Z-projection)" if row == 0 else "")
            axes[row, 1].axis('off')

            # --- NEW: Plot Sparse Input (2D Scatter Projection) ---
            ax_scatter = axes[row, 2]
            obs_xyz_plot = obs_xyz_abs.cpu().numpy()
            # Plot X vs Y, and use Z for the color
            sc = ax_scatter.scatter(obs_xyz_plot[:, 0], obs_xyz_plot[:, 1], c=obs_xyz_plot[:, 2], cmap='coolwarm', s=20,
                                    vmin=-0.5, vmax=0.5)
            ax_scatter.set_title(f"Input Mics" if row == 0 else "")
            ax_scatter.set_aspect('equal', adjustable='box')
            ax_scatter.set_xlim(-0.6, 0.6);
            ax_scatter.set_ylim(-0.6, 0.6)  # Example limits
            ax_scatter.set_xticks([]);
            ax_scatter.set_yticks([])

            cbar_z = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.04)
            cbar_z.set_label('Z-height (m)', size=8)
            cbar_z.ax.tick_params(labelsize=7)

            # Show per-row microphone count BETWEEN GT and scatter columns
            pos_gt = axes[row, 1].get_position(fig)
            pos_sc = axes[row, 2].get_position(fig)
            x_mid_M = (pos_gt.x1 + pos_sc.x0) * 0.5
            y_mid_row = (pos_gt.y0 + pos_gt.y1) * 0.5
            fig.text(x_mid_M - 0.03, y_mid_row, f"M={M} \nsrc ind ={srcind[0]}", ha='left', va='center', fontsize=9)

            gs = axes[row, 0].get_gridspec()
            axes[row, 0].remove()
            ax3d_inline = fig.add_subplot(gs[row, 0], projection='3d')

            # Room box
            plot_room_box(ax3d_inline, room_dim)

            # Global positions for plotting
            obs_xyz_global = obs_xyz_abs.cpu().numpy() + center_np
            src_xyz_global = src_xyz.cpu().numpy() + center_np

            # Scatter microphones and source
            ax3d_inline.scatter(
                obs_xyz_global[:, 0], obs_xyz_global[:, 1], obs_xyz_global[:, 2],
                s=20, c='b'
            )
            ax3d_inline.scatter(
                src_xyz_global[0, 0], src_xyz_global[0, 1], src_xyz_global[0, 2],
                s=60, c='r', marker='*'
            )

            # Labels and limits
            ax3d_inline.set_xlabel('X (m)');
            ax3d_inline.set_ylabel('Y (m)');
            ax3d_inline.set_zlabel('Z (m)')
            ax3d_inline.set_title('Room (3D)' if row == 0 else '')

            # --- Generate for each guidance scale ---

            for g_idx, w in enumerate(guidance_scales):

                # Set the guidance scale on the ODE object
                if fm_vs_diff == 'flow_matching' or fm_vs_diff == None:
                    print("--> Using Flow Matching ODE Simulator")
                    simulator = EulerSimulator(ode=ode_3d)
                    simulator.ode.guidance_scale = w

                    # Start from pure noise
                    x0 = torch.randn_like(z_true)
                    xt = x0.clone()  # The simulation starts from x0
                    # Get conditioning tokens
                    y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

                    ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
                    ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

                    # Simulation loop
                    x1_recon = simulator.simulate(xt,
                                                  ts,
                                                  x0=x0,
                                                  z_true=z_true,
                                                  y_tokens=y_tokens,
                                                  obs_mask=obs_mask,
                                                  obs_coords_rel=obs_coords_rel,
                                                  obs_values=obs_values,
                                                  pooled_context=pooled_context,
                                                  freq_contexts=freq_contexts,
                                                  paste_observations=True,
                                                  obs_indices=obs_indices
                                                  )

                elif fm_vs_diff == 'score_matching':
                    # --- DDPM (DDIM) Inference ---
                    print("--> Using DDIM Sampler")
                    ddpm_scheduler = DDPMScheduler(num_timesteps=num_timesteps)

                    # Create the new ODE wrapper with the DDPM-trained model
                    ddpm_ode = DDPM_ODE_Sampler(
                        noise_predictor_network=unet_3d,
                        set_encoder=set_encoder,
                        scheduler=ddpm_scheduler,
                        config=config
                    )

                    simulator = EulerSimulator(ode=ddpm_ode)
                    simulator.ode.guidance_scale = w

                    # ddpm_scheduler = DDPMScheduler(num_timesteps=1000)
                    # sampler = DDIMSampler(model=unet_3d, set_encoder=set_encoder, scheduler=ddpm_scheduler)
                    # num_inference_steps = 50  # Or another value, e.g., 200

                    # 2. Prepare initial noise and conditioning
                    xt = torch.randn_like(z_true)  # Start with pure noise (at timestep T)
                    xt = xt.float()
                    y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)
                    # model_kwargs = {
                    #     "context": y_tokens,
                    #     "context_mask": obs_mask,
                    # }

                    # Create timesteps for simulation
                    ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
                    ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

                    simulation_kwargs = {
                        "obs_coords_rel": obs_coords_rel,
                        "obs_values": obs_values,
                        "obs_mask": obs_mask,
                        "pooled_context": pooled_context
                    }

                    # 2. Create the discrete timestep schedule for backward sampling
                    # inference_timesteps = torch.linspace(ddpm_scheduler.num_timesteps - 1, 0, num_timesteps,
                    #                                      dtype=torch.long, device=device)

                    # 4. The backward denoising loop
                    # for t_step in tqdm(inference_timesteps, desc=f"DDIM Sampling w={w}"):
                    #     xt = sampler.step(xt, t_step.item(), guidance_scale=w, **model_kwargs)
                    #     xt = xt.float()

                    x1_recon = simulator.simulate(xt, ts, **simulation_kwargs)


                # De-normalize and plot
                x1_recon_denorm = (x1_recon * std + mean)
                recon_cube_to_plot = x1_recon_denorm[0, freq_idx_to_plot].detach().cpu().numpy()

                # Calculate BOTH metrics for comprehensive evaluation
                if EXCLUDE_BOUNDARY:
                    # Create interior mask for this cube
                    mask_3d, interior_indices = create_interior_mask(z_true.shape, BOUNDARY_THICKNESS)

                    # Flatten cubes and select interior positions only
                    x1_flat = x1_recon.view(x1_recon.shape[1], -1)  # [freq, 1331]
                    z_flat = z_true.view(z_true.shape[1], -1)       # [freq, 1331]
                    x1_flat_denorm = x1_recon_denorm.view(x1_recon_denorm.shape[1], -1)
                    z_flat_denorm = z_true_denorm.view(z_true_denorm.shape[1], -1)

                    # Select interior positions
                    x1_interior = x1_flat[:, interior_indices]      # [freq, interior_count]
                    z_interior = z_flat[:, interior_indices]        # [freq, interior_count]
                    x1_interior_denorm = x1_flat_denorm[:, interior_indices]
                    z_interior_denorm = z_flat_denorm[:, interior_indices]

                    # Calculate metrics on interior only
                    lsd_db = calculate_lsd_unified(x1_interior_denorm.T, z_interior_denorm.T, freq_dim=1).item()
                    mse = torch.mean((x1_interior_denorm - z_interior_denorm) ** 2).item()

                    metric_label = f"Interior ({len(interior_indices)}/1331 pos)"
                else:
                    # Full cube calculation
                    lsd_db = calculate_lsd_unified(x1_recon_denorm, z_true_denorm, freq_dim=1).item()
                    mse = torch.mean((x1_recon_denorm - z_true_denorm) ** 2).item()

                    metric_label = "Full Cube (1331 pos)"
                
                # ALSO calculate slice-specific metrics for comparison
                slice_metrics = calculate_slice_metrics(x1_recon_denorm[0], z_true_denorm[0], 
                                                      freq_idx_to_plot, z_slice_idx_to_plot)

                print(f"Row {row}, w={w}:")
                print(f"  Full Cube - MSE: {mse:.4f}, LSD: {lsd_db:.4f} dB ({metric_label})")
                print(f"  Slice Only - MSE: {slice_metrics['mse']:.4f}, LSD: {slice_metrics['lsd']:.4f} dB")

                col_idx = g_idx + 3

                recon_slice = recon_cube_to_plot[z_slice_idx_to_plot, :, :]  # Select the same slice
                im = axes[row, col_idx].imshow(recon_slice, origin='lower', cmap='viridis', vmin=gt_slice.min(),
                                               vmax=gt_slice.max())

                # Set title and add MSE text under the plot
                axes[row, col_idx].set_title(f"w={w}" if row == 0 else "")
                axes[row, col_idx].axis('off')

                # Add both full cube and slice metrics under the plot
                metric_suffix = "Int" if EXCLUDE_BOUNDARY else "Full"
                metric_text = (f"Full: MSE={mse:.3f}, LSD={lsd_db:.3f}dB ({metric_suffix})\n"
                             f"Slice: MSE={slice_metrics['mse']:.3f}, LSD={slice_metrics['lsd']:.3f}dB")
                text_box = axes[row, col_idx].text(0.5, -0.15, metric_text,
                                                 transform=axes[row, col_idx].transAxes,
                                                 ha='center', va='top', fontsize=8,
                                                 bbox=dict(facecolor='white', alpha=0.8,
                                                         edgecolor='none', pad=2))

            # Shared colorbar for GT and generated columns (exclude scatter input)
            ax_list = [axes[row, 1]] + [axes[row, i + 3] for i in range(len(guidance_scales))]
            mappable = matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=gt_slice.min(), vmax=gt_slice.max()), cmap='viridis'
            )
            cbar_mag = fig.colorbar(mappable, ax=ax_list, fraction=0.046, pad=0.04)
            cbar_mag.set_label('Magnitude (dB)', size=8)
            cbar_mag.ax.tick_params(labelsize=7)

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.4)  # More space for title and between rows
        plt.show()
        plt.savefig("test.png", dpi=200, bbox_inches='tight')

        model_dir = os.path.dirname(MODEL_LOAD_PATH)
        outfile_name = f"{model_mode}_finf{freq_idx_to_plot}_z{z_slice_idx_to_plot}_{M_range[0]}to{M_range[1]}.png"

        if os.path.exists(os.path.join("artifacts", outfile_name)):
             rand = random.randint(0, 1000)
             outfile_name = f"{model_mode}_finf{freq_idx_to_plot}_z{z_slice_idx_to_plot}_{M_range[0]}to{M_range[1]}_{rand}.png"

        save_path = os.path.join(model_dir, outfile_name)
        print(f"Saving figure to: {save_path}")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    if COMPARE:
        # --- Set M_range for comparison mode ---
        if M_range is None:
            M_range = config['training'].get('M_range')

        print(f"=== MULTI-MODEL MSE COMPARISON ===")
        MODEL_NAME = get_model_name(MODEL_LOAD_PATH)
        print(f"Comparing {len(MULTI_MODEL_PATHS)} models:")
        for i, path in enumerate(MULTI_MODEL_PATHS):
            model_name = MODEL_NAMES[i] if MODEL_NAMES and i < len(MODEL_NAMES) else get_model_name(path)
            print(f"  {i + 1}. {model_name}")
        print()

        # --- MULTI-MODEL MSE COMPARISON MODE ---
        print("Running MSE comparison across multiple models...")

        # Prepare to collect MSE results for all models
        all_mse_results = []
        model_names_used = []

        # Process each model
        for model_idx, model_path in enumerate(MULTI_MODEL_PATHS):
            print(f"\nProcessing model {model_idx + 1}/{len(MULTI_MODEL_PATHS)}: {get_model_name(model_path)}")

            # Load model
            checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)

            # Create and load models
            set_encoder, unet_3d, ode_3d, _ = model_factory(config, model_states_cfg, device)

            # Get model name and print model info
            model_name = MODEL_NAMES[model_idx] if MODEL_NAMES and model_idx < len(MODEL_NAMES) else get_model_name(model_path)
            model_names_used.append(model_name)

            # Print model information for each model in comparison
            print(f"\n--- Model {model_idx + 1} Architecture: {model_name} ---")
            set_encoder_info = get_model_info(set_encoder, "SetEncoder")
            unet_info = get_model_info(unet_3d, "UNet3D")
            total_params = set_encoder_info['total_params'] + unet_info['total_params']
            total_size_mb = set_encoder_info['model_size_mb'] + unet_info['model_size_mb']
            print(f"Total parameters: {total_params:,} | Size: {total_size_mb:.2f} MB")

            # Run inference for multiple examples
            model_mse_results = []
            for example_idx in range(num_examples):
                print(f"  Example {example_idx + 1}/{num_examples}")

                # Get a random ground truth sample (using same seed for consistency)
                torch.manual_seed(SEED + example_idx)  # Consistent samples across models
                z_true, src_xyz = test_sampler.sample(1)

                z_true, src_xyz = z_true.to(device), src_xyz.to(device)

                # Run inference for this example
                mse_results, M = run_single_inference(
                    set_encoder, unet_3d, ode_3d, z_true, src_xyz, grid_xyz, M_range,
                    guidance_scales, num_timesteps, mean, std, device, EXCLUDE_BOUNDARY, BOUNDARY_THICKNESS
                )
                model_mse_results.append(mse_results)
                print(f"\n    M={M}")
                for j, result in enumerate(mse_results):
                    print(f"      w={guidance_scales[j]}: MSE={result['mse']:.4f}, LSD={result['lsd']:.4f} dB")

            all_mse_results.append(model_mse_results)

        # Create dual metric comparison plot
        print(f"\nCreating dual metric comparison plot...")
        plot_dual_metric_comparison(all_mse_results, model_names_used, guidance_scales,
                           save_path=os.path.join("artifacts", "dual_metric_comparison.png"),
                           )

