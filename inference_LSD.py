import torch
import numpy as np
import os
import json
from tqdm import tqdm
from fm_utils import ATF3DSampler, SetEncoder, CrossAttentionUNet3D

# Set a seed for reproducibility of the random microphone selection
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_20250825-184335_iter200000/model.pt"

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

    # Re-create and load models
    model_cfg = config['model']
    set_encoder = SetEncoder(
        num_freqs=train_sampler.cubes.shape[1],
        d_model=model_cfg['d_model'], nhead=model_cfg['nhead'], num_layers=model_cfg['num_encoder_layers']
    ).to(device)
    unet_3d = CrossAttentionUNet3D(
        in_channels=train_sampler.cubes.shape[1], out_channels=train_sampler.cubes.shape[1],
        channels=model_cfg['channels'], d_model=model_cfg['d_model'], nhead=model_cfg['nhead']
    ).to(device)
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()

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
                xt = torch.randn_like(z_true)  # Start from pure noise
                y_tokens, _ = set_encoder(obs_coords_rel, obs_values, obs_mask)
                null_tokens = set_encoder.y_null_token.expand(1, y_tokens.shape[1], -1)

                for t_step in range(num_timesteps):
                    t = torch.tensor([t_step / num_timesteps], device=device)
                    guided_drift = unet_3d(xt, t, context=y_tokens, context_mask=obs_mask)
                    # For w=1.0, we don't need the unguided drift, simplifying the calculation.
                    # drift = (1 - guidance_scale) * unguided_drift + guidance_scale * guided_drift
                    drift = guided_drift
                    xt = xt + (1 / num_timesteps) * drift

                z_est = xt

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