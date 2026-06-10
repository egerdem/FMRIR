"""
Grid figure generator for visual inspection.

Generates figures with ROWS_PER_FIG rows, each row = one source.
All rows in a figure share the same frequency bin.
Run once per frequency; change FREQ_IDX to switch frequency.

Output: RESULTS/grid_f{FREQ_IDX}/
"""

import os
import sys
import numpy as np
import random
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
FREQ_IDX      = 30          # target frequency bin (change to 40, 50, 60 etc.)
N_FIGURES     = 10          # how many figures to generate
ROWS_PER_FIG  = 5           # rows (= sources) per figure
SPARSE_M      = 5           # number of microphones
GUIDANCE_SCALE = 1.0
NUM_TIMESTEPS  = 10

# Max test source index — adjust if you know the exact test set size
MAX_SRC_IDX   = 101         # exclusive upper bound (i.e. indices 0..100)

# Z-slice choices (0–10 for an 11-level grid; middle-ish slices look best)
Z_CHOICES     = [0,1,4,5,8]

SEED          = 42
MODEL_PATH    = (
    "/Volumes/T7 Shield/SFlow/FMRIR_experiments/"
    "KCL_ros10008val200_fctx_UnetV2res_freq64_Val102_Mval5_r1_M5_10_20_50_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay100_setv3_unet3_256to1024v2_110k_20260421-091511_iter100000/"
    "model_89199_lsd5.5800.pt"
)
MIC_SELECTION_PATH = "./idx_mes_pos_s1024_m1331.npy"
SAVE_DIR = f"RESULTS/grid_f{FREQ_IDX}"
# ──────────────────────────────────────────────────────────────────────────────

# Make SPARSE_M visible as a module-level global so generate_SFfigures_FM_V2
# can reference it (the function does `M = SPARSE_M`).
import builtins
builtins.SPARSE_M = SPARSE_M  # fallback; the real one is set below

# Import after builtins patch so the function can find SPARSE_M
sys.path.insert(0, os.path.dirname(__file__))

# paper_figures imports matplotlib with Qt5Agg which may fail on headless
# machines — that's fine, we only need the generation function.
import paper_figures as pf

# Override SPARSE_M in the paper_figures module namespace
pf.SPARSE_M = SPARSE_M  # type: ignore[attr-defined]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    idx_mes_pos_mat = np.load(MIC_SELECTION_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Build a shuffled pool of (src_idx, z_slice) pairs
    all_pairs = [
        (src, z)
        for src in range(MAX_SRC_IDX)
        for z in Z_CHOICES
    ]
    random.shuffle(all_pairs)

    # We need N_FIGURES * ROWS_PER_FIG pairs; tile if pool is too small
    needed = N_FIGURES * ROWS_PER_FIG
    while len(all_pairs) < needed:
        all_pairs = all_pairs + all_pairs
    all_pairs = all_pairs[:needed]

    print(f"Generating {N_FIGURES} figures × {ROWS_PER_FIG} rows "
          f"for freq_idx={FREQ_IDX}, saving to {SAVE_DIR}/")

    for fig_i in range(N_FIGURES):
        batch = all_pairs[fig_i * ROWS_PER_FIG : (fig_i + 1) * ROWS_PER_FIG]
        srcind_batch   = [p[0] for p in batch]
        z_slice_batch  = [p[1] for p in batch]
        freq_batch     = [FREQ_IDX] * ROWS_PER_FIG

        print(f"\n[Fig {fig_i+1}/{N_FIGURES}] "
              f"srcs={srcind_batch}  z={z_slice_batch}")

        try:
            pf.generate_SFfigures_FM_V2(
                model_path      = MODEL_PATH,
                srcind          = srcind_batch,
                freq_idx_to_plot= freq_batch,
                z_slice_idx     = z_slice_batch,
                guidance_scale  = GUIDANCE_SCALE,
                num_timesteps   = NUM_TIMESTEPS,
                save_dir        = SAVE_DIR,
                idx_mes_pos_mat = idx_mes_pos_mat,
            )
        except Exception as e:
            print(f"  ERROR on fig {fig_i+1}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\nDone. Files in: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
