import torch
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)  # Non-interactive backend for saving plots
from matplotlib import pyplot as plt
import os
import json
import random

# Import your necessary classes
from fm_utils import (
    ATF3DSampler, LSD,
    SetEncoder,
    CrossAttentionUNet3D, CFGVectorFieldODE_3D, EulerSimulator
)

# Set seed for reproducible results
SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for GPU
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic behavior for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_1d_atf_comparison_multi_guidance(ax, freqs, gt_atf, gen_atf_dict, title):
    """Helper function to plot Ground Truth vs. Generated 1D ATF for multiple guidance levels."""
    # Plot ground truth
    ax.plot(freqs, gt_atf, label='Ground Truth', linewidth=2, color='black')
    
    # Plot generated ATFs for different guidance levels
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (guidance, gen_atf) in enumerate(gen_atf_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(freqs, gen_atf, label=f'Generated (w={guidance})', 
                linestyle='--', linewidth=1.5, color=color)
    
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()


def main():
    # --- Configuration ---
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET128_LRmin_e4_7_20250826-212533_iter100000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_sigmaE5_UNET128_LRmin_e6dot6e4toe7_d128_20250827-185835_iter400000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/modelCONVoldcheckpoint.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_UNET256_20250826-192413_iter200000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts//ATF3D-CrossAttn-v1-freq20_M40to50_sigmaE5_enclayer3_UNET128_LRmin_e6dot6e4toe7_d256_20250827-213218_iter500000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/model_CONVoldcheckpoint.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_sigmaE5_UNET128_LRmin_e6dot6e4toe7_d128_20250827-185835_iter400000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to100_sigmaE3_lr1e3to_e7_unet3_layer3_head3_d256_20250828-190043_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_lr1e3to_e7_unet3_layer3_head8_d256_20250828-233343_iter50000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to6_freq20_layer3_d256_head8_sigmaE3_lr1e3to_e7_unet3_20250903-144259_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/Loss169e3_M5to50_freq20_layer3_d256_head8_sigmaE3_lr1e3to_e7_unet3_20250904-191320_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e3to_e7_unet3_20250904-203305_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e3to_e7_unet3_20250904-203305_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head4_sigma0ZERO_lr1e4to_e7_unet3_20250904-214817_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-222356_iter300000/model.pt"

    MODEL_NAME = MODEL_LOAD_PATH.split("artifacts/")[1].split("/")[0]

    data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model and Config ---
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
    config = checkpoint.get('config', {})  # Use .get for safety
    model_states_cfg = checkpoint['model_states']

    freq_up_to = config['model'].get('freq_up_to')

    # --- Evaluation Configuration ---
    # 10 different sources from 922 to 1024 (indices 0 to 102 in test set)
    # source_indices = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]  # 10 sources
    source_indices = [0, 11, 22]  # 10 sources
    # 5 different random microphone positions from 0 to 1330
    mic_indices = [156, 423, 789, 1045, 1287, 665]  # 5 microphones
    
    M = 5  # Number of conditioning mics
    guidance = [1.0, 2.0]
    num_timesteps = 10
    lsd = LSD()

    # Create output directory
    output_dir = "artifacts/eval"
    output_dir = os.path.join(output_dir, MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data Sampler (to get data and metadata) ---

    # --- 1. Data Loading ---
    # Create train sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=config['data']['src_splits'], normalize=True, freq_up_to=freq_up_to
    )

    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()

    # Load the full test set for evaluation
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test',
        src_splits=config['data']['src_splits'],
        freq_up_to=config['model']['freq_up_to'],
        normalize=False  # Load raw, un-normalized data for plotting
    )
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    print(f"Loaded Stats from 3D Training Set: Mean={mean:.4f}, Std={std:.4f}")
    print(f"Test set contains {len(test_sampler.cubes)} samples")

    # --- Recreate and Load Models ---
    model_cfg = config['model']
    set_encoder = SetEncoder(
        num_freqs=train_sampler.cubes.shape[1],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)

    unet_3d = CrossAttentionUNet3D(
        in_channels=train_sampler.cubes.shape[1],
        out_channels=train_sampler.cubes.shape[1],
        channels=model_cfg['channels'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead']
    ).to(device)

    # --- 3. Load Weights ---
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()

    ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)
    simulator = EulerSimulator(ode=ode_3d)

    # Get the frequency axis values from your data generation config
    data_gen_config_path = os.path.join(data_path, "config.json")
    with open(data_gen_config_path, 'r') as f:
        data_gen_config = json.load(f)
    fftlen_algn = data_gen_config['fftlen_algn']
    fs = data_gen_config['fs']
    freq_axis = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
    freq_axis = freq_axis[:freq_up_to]  # Ensure it matches model's frequency count

    # Fixed conditioning microphones for all evaluations
    # obs_indices = torch.tensor([763, 654, 398, 823, 947])  # Use only M=5 microphones
    obs_indices = torch.randperm(grid_xyz.shape[0])[:M]

    plot_count = 0
    total_plots = len(source_indices) * len(mic_indices)
    
    print(f"Generating {total_plots} plots...")
    
    # Loop through all source and microphone combinations
    for source_idx in source_indices:
        # Get the ground truth cube and source position for this source
        z_true = test_sampler.cubes[source_idx].unsqueeze(0).to(device)
        src_xyz = test_sampler.source_coords[source_idx].unsqueeze(0).to(device)
        
        # z_true is already normalized from test_sampler.cubes
        # For plotting comparison later, we need the denormalized version
        z_true_denorm = (z_true * std + mean)
        
        # Prepare conditioning observations
        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = obs_xyz_abs - src_xyz
        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1)
        
        # Batchify for the set encoder
        obs_coords_rel_batch = obs_coords_rel.unsqueeze(0)
        obs_values_batch = obs_values.unsqueeze(0)
        obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
        
        # Get conditioning tokens (only need to compute once)
        y_tokens, _ = set_encoder(obs_coords_rel_batch, obs_values_batch, obs_mask)
        
        ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
        ts = ts.view(1, -1, 1, 1, 1, 1).expand(1, -1, -1, -1, -1, -1)
        
        # Generate ATFs for all guidance levels
        # Use the same initial noise for all guidance levels for fair comparison
        x0 = torch.randn_like(z_true)
        
        gen_cubes_denorm = {}
        for guid in guidance:
            xt = x0.clone()  # Start from the same initial noise
            
            # Set the guidance scale on the ODE object
            simulator.ode.guidance_scale = guid
            
            # Simulation loop
            x1_recon = simulator.simulate(xt,
                                          ts,
                                          x0=x0,
                                          z_true=z_true,
                                          y_tokens=y_tokens,
                                          obs_mask=obs_mask,
                                          paste_observations=False,
                                          obs_indices=obs_indices
                                          )
            
            # De-normalize generated result
            gen_cubes_denorm[guid] = (x1_recon * std + mean)
        
        # Loop through all microphones for this source
        for mic_idx in mic_indices:
            # Convert flat microphone index to 3D coordinates
            nx, ny, nz = 11, 11, 11
            iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
            
            # Extract the 1D vector of frequencies for the chosen mic
            # Ground truth (same for all guidance levels)
            gt_atf_1d = z_true_denorm[0, :, iz, iy, ix].cpu().numpy()
            
            # Generated ATFs for all guidance levels
            gen_atf_dict = {}
            for guid in guidance:
                gen_atf_dict[guid] = gen_cubes_denorm[guid][0, :, iz, iy, ix].cpu().numpy()
                print(f"\n len gt_atf_1d {len(gt_atf_1d)}, len gen_atf_dict[guid] {len(gen_atf_dict[guid])}")
                lsd_val = lsd(gt_atf_1d, gen_atf_dict[guid], dim=1, mean=False)
                lsd_mean = lsd_val.mean()
                lsd_std = lsd_val.std()
                print(f'LSD: {lsd_mean:.4f} +- {lsd_std:.4f} dB')

            if True:
                # Create and save plot
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                plot_1d_atf_comparison_multi_guidance(ax, freq_axis, gt_atf_1d, gen_atf_dict,
                                                      title=f"ATF Comparison: Source {source_idx+922}, Mic {mic_idx}")
                plt.tight_layout()

                # Save plot
                filename = f"src{source_idx+922:04d}_mic{mic_idx:04d}_multi_guidance.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)  # Close to save memory

                plot_count += 1
                print(f"Saved plot {plot_count}/{total_plots}: {filename}")
    
    print(f"All {total_plots} plots saved to {output_dir}/")


if __name__ == '__main__':
    main()