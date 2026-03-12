"""
diag_lsd_per_bin.py
====================
Diagnostic: LSD broken down per frequency bin for an existing trained model.

Tells you whether reconstruction error degrades gradually across bins or
has a sharp cliff — key information before committing to the freq-conditioned
architecture.

Usage:
    python diag_lsd_per_bin.py
    
Edit MODEL_PATH and DATA_DIR below.
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless — saves to file, no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm

from inference import model_factory, load_model_and_config
from fm_utils import ATF3DSampler, EulerSimulator

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "artifacts/KCL_RNG_Val102src10step_Mval5_r1_M5_10_20_50_100_200_freq20_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay100_unet4v1_setv3_300k_20260227-015304_iter300000/model_288299_lsd2.6896.pt"   # ← change this
DATA_DIR     = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
MIC_SEL_PATH = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"

M            = 5          # microphones per source (use same as eval)
NUM_SOURCES  = 102        # how many test sources to evaluate
NUM_STEPS    = 10         # ODE steps
GUIDANCE     = 1.0

# Physical constants for freq-axis labelling
FS           = 2000       # sample rate (Hz) — read from data dir name "fs2000"
F_TOTAL      = 64         # total bins in one full ATF cube (NOT the model's freq_up_to)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────────────


def freq_hz(bin_idx: np.ndarray, fs: int = FS, f_total: int = F_TOTAL) -> np.ndarray:
    """Convert 0-based bin index to Hz.  bin 0 → Δf, bin F_TOTAL-1 → fs/2."""
    delta_f = (fs / 2) / f_total    # Hz per bin
    return (bin_idx + 1) * delta_f  # 1-indexed convention matches irdata_gen


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint, config, model_states_cfg = load_model_and_config(MODEL_PATH, device)
    set_encoder, unet_3d, ode_3d, _ = model_factory(config, model_states_cfg, device)

    freq_up_to = config['model'].get('freq_up_to')
    freq_from  = config['model'].get('freq_from', 0)
    num_freqs  = freq_up_to - freq_from
    src_split  = config['data']['src_splits']
    model_name = config['model'].get('name', os.path.basename(os.path.dirname(MODEL_PATH)))

    print(f"Model freq range: bins {freq_from}..{freq_up_to-1}  ({num_freqs} bins)")
    print(f"  = {freq_hz(np.array([freq_from]))[0]:.1f} Hz .. {freq_hz(np.array([freq_up_to-1]))[0]:.1f} Hz")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=freq_up_to, freq_from=freq_from,
        model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=DATA_DIR, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=freq_up_to, freq_from=freq_from,
        model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    grid_xyz = train_sampler.grid_xyz.to(device)
    mean_val = train_sampler.mean.item()
    std_val  = train_sampler.std.item()

    # Microphone selection: same matrix used by reference model for fair comparison
    if os.path.exists(MIC_SEL_PATH):
        idx_mes = np.load(MIC_SEL_PATH)
        print(f"Using fixed mic selection matrix: {idx_mes.shape}")
        random_mics = False
    else:
        print(f"Mic selection file not found at {MIC_SEL_PATH}, using random selection.")
        random_mics = True

    n_eval = min(NUM_SOURCES, len(test_sampler))

    # ── Accumulate squared errors per bin ─────────────────────────────────────
    # shape: [num_freqs, num_positions]  — we'll accumulate over sources
    sum_sq_err  = torch.zeros(num_freqs)   # Σ MSE per bin
    sum_sq_err2 = torch.zeros(num_freqs)   # for std — Σ MSE² per bin
    n_samples   = 0

    simulator = EulerSimulator(ode=ode_3d)
    simulator.ode.guidance_scale = GUIDANCE

    for src_idx in tqdm(range(n_eval), desc="Sources"):
        with torch.no_grad():
            z_true  = test_sampler.cubes[src_idx].unsqueeze(0).to(device)   # [1, F, 11, 11, 11]
            src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)

            if random_mics:
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
            else:
                obs_indices = torch.tensor(idx_mes[:M, src_idx], dtype=torch.long, device=device)

            obs_xyz_abs   = grid_xyz[obs_indices]
            obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)  # [1, M, 3]
            z_flat         = z_true.view(z_true.shape[1], -1)      # [F, 1331]
            obs_values     = z_flat[:, obs_indices].T.unsqueeze(0)  # [1, M, F]
            obs_mask       = torch.ones(1, M, dtype=torch.bool, device=device)

            # ODE solve
            x0   = torch.randn_like(z_true)
            ts   = torch.linspace(0, 1, NUM_STEPS + 1, device=device)
            ts   = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

            y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)

            z_est = simulator.simulate(
                x0, ts, x0=x0, z_true=z_true,
                y_tokens=y_tokens, obs_mask=obs_mask,
                pooled_context=pooled_context,
                paste_observations=True, obs_indices=obs_indices
            )

            # Denormalize → dB domain
            z_est_db  = z_est  * std_val + mean_val   # [1, F, 11, 11, 11]
            z_true_db = z_true * std_val + mean_val

            # Per-bin LSD: sqrt(mean_over_positions( (est-gt)² ))
            # Flatten spatial dims: [1, F, 1331]
            est_flat  = z_est_db.view(1, num_freqs, -1)
            gt_flat   = z_true_db.view(1, num_freqs, -1)

            mse_per_bin = torch.mean((est_flat - gt_flat) ** 2, dim=2).squeeze(0)  # [F]
            lsd_per_bin = torch.sqrt(mse_per_bin)                                   # [F]

            sum_sq_err  += lsd_per_bin.cpu()
            sum_sq_err2 += (lsd_per_bin ** 2).cpu()
            n_samples   += 1

    mean_lsd_per_bin = sum_sq_err / n_samples                                    # [F]
    # std across sources via E[X²] - E[X]²  (approximation)
    std_lsd_per_bin  = torch.sqrt(
        torch.clamp(sum_sq_err2 / n_samples - mean_lsd_per_bin ** 2, min=0)
    )

    # ── Print table ───────────────────────────────────────────────────────────
    bin_indices = np.arange(freq_from, freq_up_to)
    hz_values   = freq_hz(bin_indices)

    print(f"\n{'Bin':>4}  {'Freq (Hz)':>10}  {'LSD (dB)':>10}  {'±std':>8}")
    print("-" * 40)
    for i, (b, hz, lsd, std) in enumerate(zip(
            bin_indices, hz_values,
            mean_lsd_per_bin.numpy(), std_lsd_per_bin.numpy())):
        print(f"{b:>4}  {hz:>10.1f}  {lsd:>10.4f}  {std:>8.4f}")

    overall_lsd = mean_lsd_per_bin.mean().item()
    print(f"\nOverall mean LSD across bins: {overall_lsd:.4f} dB")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LSD vs bin index
    ax = axes[0]
    ax.plot(bin_indices, mean_lsd_per_bin.numpy(), 'b-o', markersize=4, linewidth=1.5, label='Mean LSD')
    ax.fill_between(
        bin_indices,
        (mean_lsd_per_bin - std_lsd_per_bin).numpy(),
        (mean_lsd_per_bin + std_lsd_per_bin).numpy(),
        alpha=0.25, color='blue', label='±1 std (across sources)'
    )
    ax.axhline(overall_lsd, color='red', linestyle='--', linewidth=1, label=f'Mean = {overall_lsd:.3f} dB')
    ax.set_xlabel('Frequency bin index')
    ax.set_ylabel('LSD (dB)')
    ax.set_title('LSD per frequency bin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: LSD vs actual Hz (log x-axis)
    ax = axes[1]
    ax.plot(hz_values, mean_lsd_per_bin.numpy(), 'b-o', markersize=4, linewidth=1.5, label='Mean LSD')
    ax.fill_between(
        hz_values,
        (mean_lsd_per_bin - std_lsd_per_bin).numpy(),
        (mean_lsd_per_bin + std_lsd_per_bin).numpy(),
        alpha=0.25, color='blue'
    )
    ax.axhline(overall_lsd, color='red', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('LSD (dB)')
    ax.set_title('LSD per frequency bin (log-Hz axis)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    model_tag = os.path.basename(os.path.dirname(MODEL_PATH))
    fig.suptitle(f'Per-bin LSD  |  {n_eval} sources, M={M}\n{model_tag}', fontsize=10)
    plt.tight_layout()

    out_path = f"diag_lsd_per_bin_{model_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out_path}")


if __name__ == '__main__':
    main()
