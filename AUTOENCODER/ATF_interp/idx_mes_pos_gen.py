import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate (or extend) a mic-permutation matrix.")
    parser.add_argument('--n_sources', type=int, default=8192,
                        help='Total number of sources (columns). Default 8192 covers r4.')
    parser.add_argument('--n_mics', type=int, default=1331,
                        help='Total number of mics (rows). Default 1331.')
    parser.add_argument('--extend_from', type=str, default=None,
                        help='Path to an existing .npy matrix to extend instead of regenerating from scratch. '
                             'Columns 0..old_N-1 will be preserved exactly, new columns appended.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for new columns (does NOT affect preserved columns).')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Output directory. Default: current directory.')
    args = parser.parse_args()

    M = args.n_mics
    N = args.n_sources

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    if args.extend_from is not None and os.path.exists(args.extend_from):
        old = np.load(args.extend_from)
        old_M, old_N = old.shape
        assert old_M == M, f"Row count mismatch: existing={old_M}, requested={M}"
        assert old_N <= N, f"Cannot shrink: existing has {old_N} cols, requested {N}"
        print(f"Extending existing matrix {old.shape} → ({M}, {N})")
        idx_mes_pos = np.zeros((M, N), dtype=np.int16)
        idx_mes_pos[:, :old_N] = old          # preserve existing columns exactly
        for i in range(old_N, N):
            idx_mes_pos[:, i] = np.random.permutation(M)
        print(f"  Preserved columns 0..{old_N - 1} unchanged.")
        print(f"  Generated new columns {old_N}..{N - 1}.")
    else:
        print(f"Generating new matrix ({M}, {N}) from scratch.")
        idx_mes_pos = np.zeros((M, N), dtype=np.int16)
        for i in range(N):
            idx_mes_pos[:, i] = np.random.permutation(M)

    out_path = os.path.join(args.out_dir, f"idx_mes_pos_s{N}_m{M}.npy")
    np.save(out_path, idx_mes_pos)
    print(f"Saved → {out_path}  shape={idx_mes_pos.shape}  dtype={idx_mes_pos.dtype}")
