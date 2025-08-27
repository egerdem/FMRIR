import torch
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
import os
import json
import random

# Import your necessary classes
from fm_utils import (
    ATF3DSampler,
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
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def plot_1d_atf_comparison(ax, freqs, gt_atf, gen_atf, title):
    """Helper function to plot Ground Truth vs. Generated 1D ATF."""
    ax.plot(freqs, gt_atf, label='Ground Truth')
    ax.plot(freqs, gen_atf, label='Generated', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()


def main():
    # --- Configuration ---
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET128_LRmin_e4_7_20250826-212533_iter100000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_sigmaE5_UNET128_LRmin_e6dot6e4toe7_d128_20250827-185835_iter400000/model.pt"
    data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model and Config ---
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
    config = checkpoint.get('config', {})  # Use .get for safety
    model_states_cfg = checkpoint['model_states']

    # processed_file = os.path.join(data_path, f'processed_atf3d_test_freqsNone.pt')
    # dataset = torch.load(processed_file)

    freq_up_to = config['model'].get('freq_up_to')

    # --- Select which sample to analyze ---
    SOURCE_ID_TO_PLOT = 922
    SOURCE_ID_TO_PLOT = SOURCE_ID_TO_PLOT - 922
    MIC_ID_TO_PLOT = 665
    M = 50  # Or any number of conditioning mics

    guidance = 1.0
    num_timesteps = 10

    # --- Load Data Sampler (to get data and metadata) ---

    # --- 1. Data Loading ---
    # Create train sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=config['data']['src_splits'], normalize=True, freq_up_to=freq_up_to
    )

    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()

    # We load the full test set to find our specific source ID
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test',
        src_splits=config['data']['src_splits'],
        freq_up_to=config['model']['freq_up_to'],
        normalize=False  # Load raw, un-normalized data for plotting
    )
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    print(f"Loaded Stats from 3D Training Set: Mean={mean:.4f}, Std={std:.4f}")

    # Get the ground truth cube and source position
    z_true = test_sampler.cubes[SOURCE_ID_TO_PLOT].unsqueeze(0).to(device)
    src_xyz = test_sampler.source_coords[SOURCE_ID_TO_PLOT].unsqueeze(0).to(device)
    # xm, ym, zm = dataset.get("grid_xyz")[MIC_ID_TO_PLOT]

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

    # --- Generate the Full 3D Cube (Inference) ---
    obs_indices = torch.randperm(grid_xyz.shape[0])[:M]

    obs_xyz_abs = grid_xyz[obs_indices]
    obs_coords_rel = obs_xyz_abs - src_xyz

    z_flat = z_true.view(z_true.shape[1], -1)
    obs_values = z_flat[:, obs_indices].transpose(0, 1)

    # Batchify for the set encoder
    obs_coords_rel = obs_coords_rel.unsqueeze(0)
    obs_values = obs_values.unsqueeze(0)
    obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

    # z_true is already normalized from test_sampler.cubes
    # For plotting comparison later, we need the denormalized version
    z_true_denorm = (z_true * std + mean)

    x0 = torch.randn_like(z_true)
    xt = x0.clone()
    # Get conditioning tokens
    y_tokens, _ = set_encoder(obs_coords_rel, obs_values, obs_mask)

    ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
    ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

    # Set the guidance scale on the ODE object
    simulator.ode.guidance_scale = guidance

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

    # De-normalize
    x1_recon_denorm = (x1_recon * std + mean)

    # --- Extract and Plot the 1D ATFs ---
    # iz, iy, ix = xm, ym, zm
    nx, ny, nz = 11, 11, 11
    iz, iy, ix = np.unravel_index(MIC_ID_TO_PLOT, (nz, ny, nx))

    # Extract the 1D vector of frequencies for the chosen mic
    # For comparison, both should be in the same scale (denormalized)
    gt_atf_1d = z_true_denorm[0, :, iz, iy, ix].cpu().numpy()  # Use denormalized ground truth
    gen_atf_1d_denorm = x1_recon_denorm[0, :, iz, iy, ix].cpu().numpy()  # Denormalized generated

    # Get the frequency axis values from your data generation config
    data_gen_config_path = os.path.join(data_path, "config.json")
    with open(data_gen_config_path, 'r') as f:
        data_gen_config = json.load(f)
    fftlen_algn = data_gen_config['fftlen_algn']
    fs = data_gen_config['fs']
    freq_axis = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
    freq_axis = freq_axis[:freq_up_to]  # Ensure it matches model's frequency count

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_1d_atf_comparison(ax, freq_axis, gt_atf_1d, gen_atf_1d_denorm,
                           title=f"ATF Comparison at Mic {MIC_ID_TO_PLOT} (Source {SOURCE_ID_TO_PLOT})")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()