"""Direct timing comparison: odometer (with subtree pruning) vs Gray code (no subtree pruning).

Runs both kernels on the same sample of L3 parents and compares wall-clock time.
"""
import argparse
import os
import struct
import sys
import time

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

print("Loading modules + JIT compilation...", flush=True)
t_load = time.time()
from cpu.run_cascade import (
    _fused_generate_and_prune,
    _fused_generate_and_prune_gray,
    _compute_bin_ranges,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
N_HALF_CHILD = N_HALF * (2 ** 4)

CHECKPOINT_PATH = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')


def load_partial_npy(path, n_cols=32, dtype=np.int32):
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        magic = f.read(6)
        major, minor = struct.unpack('BB', f.read(2))
        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]
        _ = f.read(header_len)
        header_offset = f.tell()
    data_bytes = fsize - header_offset
    row_bytes = n_cols * np.dtype(dtype).itemsize
    n_rows = data_bytes // row_bytes
    data = np.memmap(path, dtype=dtype, mode='r',
                     offset=header_offset, shape=(n_rows, n_cols))
    return np.array(data)


def load_sample(n_sample, seed=42):
    print(f"Loading partial L3 survivors...")
    parents = load_partial_npy(CHECKPOINT_PATH, n_cols=D_PARENT, dtype=np.int32)
    n_total = len(parents)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    indices.sort()
    sample = parents[indices]
    del parents
    print(f"  {len(sample)} parents sampled from {n_total:,} readable rows")
    return sample


def warmup_jit():
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
    _fused_generate_and_prune_gray(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)


def run_kernel(name, kernel_fn, sample):
    """Run kernel on all parents, return (total_time, total_surv, total_cart, per_parent_times)."""
    times = []
    total_surv = 0
    total_cart = 0

    t0 = time.time()
    for i in range(len(sample)):
        parent = sample[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, D_CHILD,
                                     n_half_child=N_HALF_CHILD)
        if result is None:
            continue
        lo_arr, hi_arr, total_children = result
        if total_children == 0:
            continue

        total_cart += total_children
        max_buf = min(total_children, 500_000)
        out_buf = np.empty((max_buf, D_CHILD), dtype=np.int32)

        t_p = time.time()
        stats = kernel_fn(parent, N_HALF_CHILD, M, C_TARGET, lo_arr, hi_arr, out_buf)
        dt_p = time.time() - t_p
        times.append(dt_p)
        total_surv += stats[0]

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{name}] {i+1}/{len(sample)}, elapsed={elapsed:.1f}s", flush=True)

    total_time = time.time() - t0
    return total_time, total_surv, total_cart, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sample', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_rounds', type=int, default=3,
                        help='Extra warmup rounds to stabilize JIT')
    args = parser.parse_args()

    sample = load_sample(args.n_sample, args.seed)

    print("Warming up JIT...", flush=True)
    warmup_jit()
    # Extra warmup: run a few real parents to ensure JIT is fully optimized
    for _ in range(args.warmup_rounds):
        for i in range(min(3, len(sample))):
            result = _compute_bin_ranges(sample[i], M, C_TARGET, D_CHILD,
                                         n_half_child=N_HALF_CHILD)
            if result is None:
                continue
            lo, hi, tc = result
            if tc == 0:
                continue
            buf = np.empty((min(tc, 1000), D_CHILD), dtype=np.int32)
            _fused_generate_and_prune(sample[i], N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
            _fused_generate_and_prune_gray(sample[i], N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
    print("  JIT ready.", flush=True)

    # --- Run odometer ---
    print(f"\n  Running ODOMETER (with subtree pruning) on {len(sample)} parents...", flush=True)
    odo_time, odo_surv, odo_cart, odo_times = run_kernel(
        "ODO", _fused_generate_and_prune, sample)

    # --- Run Gray code ---
    print(f"\n  Running GRAY CODE (no subtree pruning) on {len(sample)} parents...", flush=True)
    gray_time, gray_surv, gray_cart, gray_times = run_kernel(
        "GRAY", _fused_generate_and_prune_gray, sample)

    # --- Report ---
    print(f"\n{'='*72}")
    print(f"  TIMING COMPARISON ({len(sample)} parents)")
    print(f"{'='*72}")
    print(f"  Total Cartesian product:   {odo_cart:>14,}")
    print(f"")
    print(f"  {'Metric':<30} {'Odometer':>12} {'Gray Code':>12} {'Ratio':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Total wall-clock time':<30} {odo_time:>11.2f}s {gray_time:>11.2f}s {gray_time/max(0.001,odo_time):>7.2f}x")
    print(f"  {'Mean per parent':<30} {np.mean(odo_times):>11.4f}s {np.mean(gray_times):>11.4f}s {np.mean(gray_times)/max(0.0001,np.mean(odo_times)):>7.2f}x")
    print(f"  {'Median per parent':<30} {np.median(odo_times):>11.4f}s {np.median(gray_times):>11.4f}s {np.median(gray_times)/max(0.0001,np.median(odo_times)):>7.2f}x")
    print(f"  {'Max per parent':<30} {np.max(odo_times):>11.4f}s {np.max(gray_times):>11.4f}s")
    print(f"  {'Survivors':<30} {odo_surv:>12} {gray_surv:>12}")
    print(f"")

    # Percentile analysis
    odo_arr = np.array(odo_times)
    gray_arr = np.array(gray_times)
    for pct in [25, 50, 75, 90, 95, 99]:
        ov = np.percentile(odo_arr, pct)
        gv = np.percentile(gray_arr, pct)
        print(f"  P{pct:<3} per parent:              {ov:>11.4f}s {gv:>11.4f}s {gv/max(0.0001,ov):>7.2f}x")

    print(f"\n  Interpretation:")
    ratio = gray_time / max(0.001, odo_time)
    if ratio > 1.0:
        print(f"    Gray code is {ratio:.2f}x SLOWER (no subtree pruning)")
        savings_pct = 100 * (1 - 1/ratio)
        print(f"    Subtree pruning saves ~{savings_pct:.0f}% of odometer wall-clock time")
        print(f"    Adding subtree pruning to Gray code could recover this gap")
    else:
        print(f"    Gray code is {1/ratio:.2f}x FASTER despite visiting more children")
        print(f"    O(d) per-step advantage outweighs subtree pruning benefit")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
