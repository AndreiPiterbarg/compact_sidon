"""Export test parents to .mat format for Octave benchmark.

Exports sampled parents from each level's checkpoint as continuous weights
(parent_int / m), readable by Octave/MATLAB.  Uses the same seed (42) and
sampling strategy as run_benchmark.py for consistency.

Estimated time: ~5-10 seconds (loads 147M-row checkpoint via mmap).

Usage:
    python -m baseline.export_test_parents
    python -m baseline.export_test_parents --n_sample 50
"""
import argparse
import os
import sys

import numpy as np
import scipy.io

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_root, 'data')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

M = 20
C_TARGET = 1.4
SEED = 42

# Only L4 is valid for the MATLAB/Octave comparison: L3 survivors have
# integer bins summing to m=20 (continuous weights sum to 1.0).
# Earlier checkpoints (L0-L2) are from test runs with sum=m/2, which
# breaks the MATLAB algorithm's normalization assumptions.
LEVELS = [
    {'name': 'L4', 'checkpoint': 'checkpoint_L3_survivors.npy', 'd_parent': 32},
]


def main():
    parser = argparse.ArgumentParser(description='Export test parents to .mat')
    parser.add_argument('--n_sample', type=int, default=50,
                        help='Parents to sample per level (default: 50)')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    print(f"Exporting test parents (n_sample={args.n_sample}, seed={args.seed})")

    for level in LEVELS:
        path = os.path.join(DATA_DIR, level['checkpoint'])
        if not os.path.exists(path):
            print(f"  {level['name']}: {path} not found, skipping")
            continue

        parents = np.load(path, mmap_mode='r')
        n_total = len(parents)
        n_sample = min(args.n_sample, n_total)

        rng = np.random.default_rng(args.seed)
        # Over-sample then filter out zero/invalid rows
        oversample = min(n_sample * 2, n_total)
        indices = rng.choice(n_total, size=oversample, replace=False)
        indices.sort()
        candidates = np.array(parents[indices])
        del parents
        # Keep only rows that sum to M (valid parents)
        row_sums = np.sum(candidates, axis=1)
        valid_mask = row_sums == M
        candidates = candidates[valid_mask]
        if len(candidates) > n_sample:
            candidates = candidates[:n_sample]
        sample_int = candidates
        n_sample = len(sample_int)

        # Convert integer bin weights to continuous weights: f_i = bin_i / m
        sample_continuous = sample_int.astype(np.float64) / M

        outpath = os.path.join(OUTPUT_DIR, f'test_parents_{level["name"]}.mat')
        scipy.io.savemat(outpath, {
            'parents': sample_continuous,
            'parents_int': sample_int.astype(np.int32),
            'm': np.float64(M),
            'c_target': np.float64(C_TARGET),
            'gridSpace': np.float64(1.0 / M),
            'd_parent': np.float64(level['d_parent']),
            'n_sample': np.float64(n_sample),
            'n_total': np.float64(n_total),
            'seed': np.float64(args.seed),
        })
        print(f"  {level['name']}: exported {n_sample} of {n_total:,} parents -> {outpath}")

    print("Done.")


if __name__ == '__main__':
    main()
