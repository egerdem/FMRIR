import torch
import numpy as np
import os
import json
from tqdm import tqdm
from fm_utils import (
    ATF3DSampler, SetEncoder, 
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, 
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator
)

# Set a seed for reproducibility of the random microphone selection
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def model_factory(config, model_states_cfg, device):
    """
    Reads the config and returns the correctly instantiated and loaded models.
    """
    model_cfg = config['model']
    print(f"Model Config: {model_cfg}")

    # Use the presence of the version key to decide which architecture to build
    architecture = model_cfg.get('architecture_version')

    # --- Instantiate models based on version ---
    set_encoder = SetEncoder(
        num_freqs=model_cfg['freq_up_to'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)

    if architecture == "v2_residual_context":
        print("--- Creating (v2) architecture ---")
        unet_3d = CrossAttentionUNet3D_RED3d(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D_V2(unet=unet_3d, set_encoder=set_encoder)

    else:
        print("--- Creating v1 architecture: standard 3d unet ---")
        # Instantiate the old U-Net and ODE wrapper for old checkpoints
        unet_3d = CrossAttentionUNet3D(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)

    # --- Load weights ---
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()

    return set_encoder, unet_3d, ode_3d, architecture


def calculate_lsd(estimation_norm, ground_truth_norm):
    """
    Calculates the Log-Spectral Distortion (LSD) in the normalized domain.
    The paper's formula is the average of the RMSE over the frequency dimension.

    Args:
        estimation_norm (Tensor): The model's generated 3D cube (normalized).
        ground_truth_norm (Tensor): The ground truth 3D cube (normalized).

    Returns:
        Tensor: A scalar tensor with the calculated LSD in the normalized domain.
    """
    # The ATF data is structured as [Batch, Freq, Z, Y, X]
    # We calculate the squared error between the estimation and ground truth.
    squared_error = (estimation_norm - ground_truth_norm) ** 2

    # We take the mean over the frequency dimension (dim=1) and then the square root.
    # This results in the RMSE for each microphone position in the 3D grid.
    lsd_per_mic = torch.sqrt(torch.mean(squared_error, dim=1))

    # Finally, we average the LSD over all microphone positions to get a single value for the sample.
    return torch.mean(lsd_per_mic)


def main():
    """
    Main function to run the quantitative evaluation.
    """
    # --- 1. SETUP: Model and Data Loading ---
    #
    # This section is adapted from your inference.py script.
    #
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE4_20250825-214233_iter200000/model.pt"
    MODEL_LOAD_PATH =  "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_V2_layer_20250906-173025_iter300000/model.pt"

    MODEL_NAME = os.path.basename(os.path.dirname(MODEL_LOAD_PATH))
    print(f"Loading model: {MODEL_NAME}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']

    data_dir = config['data']['data_dir']
    # Override data_dir with your local path if necessary
    data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')

    # Load training sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, normalize=True, freq_up_to=freq_up_to
    )
    # Create test sampler with raw data
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, normalize=False, freq_up_to=freq_up_to
    )
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_mean = train_sampler.mean.item()
    spec_std = train_sampler.std.item()

    print(f"Loaded Stats from 3D Training Set: Mean={spec_mean:.4f}, Std={spec_std:.4f}")

    # --- Use the factory to get the correct models ---
    set_encoder, unet_3d, ode_3d, architecture = model_factory(config, model_states_cfg, device)
    simulator = EulerSimulator(ode=ode_3d)

    # --- 2. EVALUATION PARAMETERS ---
    #
    # Define the M values and inference settings.
    #
    M_values_to_test = [5, 40]
    guidance_scale = 1.0  # Use w=1.0 for standard conditional generation, matching the paper's method
    num_timesteps = 10  # Increased for better accuracy during evaluation
    all_results = {}

    print("\n--- Starting Quantitative Evaluation ---")

    # --- 3. EVALUATION LOOP ---
    #
    # Loop over each M value, then over the entire test set.
    #
    for M in M_values_to_test:
        print(f"\n--- Evaluating for M = {M} microphones ---")
        lsd_scores_db = []

        # tqdm provides a helpful progress bar
        for i in tqdm(range(len(test_sampler)), desc=f"M={M}"):
            with torch.no_grad():
                # Get the i-th ground truth sample from the test set
                z_true = test_sampler.cubes[i].unsqueeze(0).to(device)
                src_xyz = test_sampler.source_coords[i].unsqueeze(0).to(device)

                # --- Create a sparse observation set ---
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
                obs_xyz_abs = grid_xyz[obs_indices]
                obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)

                z_flat = z_true.view(z_true.shape[1], -1)
                obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
                obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

                # --- Perform Inference ---
                x0 = torch.randn_like(z_true)  # Start from pure noise
                xt = x0.clone()
                
                # Get conditioning tokens
                y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)

                # Set up time steps
                ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
                ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

                # Set the guidance scale
                simulator.ode.guidance_scale = guidance_scale

                # Run simulation
                z_est = simulator.simulate(xt, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                         obs_mask=obs_mask, pooled_context=pooled_context,
                                         paste_observations=False, obs_indices=obs_indices)

                # --- Calculate and Store LSD ---
                lsd_normalized = calculate_lsd(z_est.squeeze(0), z_true.squeeze(0))

                # IMPORTANT: De-normalize the LSD to get the final value in dB
                # The error was calculated on data scaled by std, so we multiply by std to restore the scale.
                lsd_db = lsd_normalized.item() * spec_std
                lsd_scores_db.append(lsd_db)

        # --- Report results for the current M value ---
        avg_lsd = np.mean(lsd_scores_db)
        std_lsd = np.std(lsd_scores_db)
        all_results[M] = {'mean': avg_lsd, 'std': std_lsd}

        print(f"--> Results for M={M}: Average LSD = {avg_lsd:.4f} dB (Std Dev = {std_lsd:.4f} dB)")

    # --- 4. FINAL SUMMARY ---
    #
    # Print a clean table of all results.
    #
    print("\n" + "=" * 45)
    print("--- Final Evaluation Summary ---")
    print("=" * 45)
    print(f"{'# Mics (M)':<20} | {'Average LSD (dB)':<20}")
    print("-" * 45)
    for M, stats in all_results.items():
        print(f"{M:<20} | {stats['mean']:.4f} (+/- {stats['std']:.4f})")
    print("=" * 45)


if __name__ == '__main__':
    main()