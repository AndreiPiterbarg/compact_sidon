"""Instrumented benchmark on partial L3 data.

The L3 checkpoint is truncated (~13M of 147M rows readable).
This script loads the readable portion and runs the instrumented
odometer kernel to measure carry patterns and subtree pruning stats.

Usage:
    python -m baseline.run_instrumented_partial --n_sample 5   # calibration
    python -m baseline.run_instrumented_partial --n_sample 100
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
    _fused_generate_and_prune_instrumented,
    _fused_generate_and_prune,
    _compute_bin_ranges,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
N_HALF_CHILD = N_HALF * (2 ** 4)  # = 32

CHECKPOINT_PATH = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')


def load_partial_npy(path, n_cols=32, dtype=np.int32):
    """Load as many complete rows as possible from a truncated .npy file."""
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        magic = f.read(6)
        major, minor = struct.unpack('BB', f.read(2))
        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]
        _ = f.read(header_len)  # skip header string
        header_offset = f.tell()

    data_bytes = fsize - header_offset
    row_bytes = n_cols * np.dtype(dtype).itemsize
    n_rows = data_bytes // row_bytes

    data = np.memmap(path, dtype=dtype, mode='r',
                     offset=header_offset, shape=(n_rows, n_cols))
    return np.array(data)  # copy into RAM so memmap can be closed


def load_sample(n_sample, seed=42):
    print(f"Loading partial L3 survivors from {CHECKPOINT_PATH}...")
    parents = load_partial_npy(CHECKPOINT_PATH, n_cols=D_PARENT, dtype=np.int32)
    n_total = len(parents)
    print(f"  {n_total:,} readable rows (of 147,279,894 expected)")

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    indices.sort()
    sample = parents[indices]
    del parents
    print(f"  Sampled {len(sample)} parents (seed={seed})")
    return sample


def warmup_jit():
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
    _fused_generate_and_prune_instrumented(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)


def main():
    parser = argparse.ArgumentParser(description='Instrumented L4 benchmark (partial L3)')
    parser.add_argument('--n_sample', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: L3 checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    sample = load_sample(args.n_sample, args.seed)

    print("Warming up JIT...", flush=True)
    warmup_jit()
    print("  JIT ready.", flush=True)

    # --- Per-parent detail tracking ---
    per_parent = []

    totals = {
        'n_surv': 0,
        'n_fast': 0, 'n_short': 0, 'n_deep': 0,
        'n_subtree_success': 0, 'n_subtree_children_skipped': 0,
        'n_qc_hit': 0, 'n_full_scan': 0, 'n_visited': 0,
        'n_cartesian': 0,
        'n_asym_skipped': 0,
        'n_no_range': 0,
    }

    print(f"\nProcessing {len(sample)} parents (instrumented odometer)...", flush=True)
    t0 = time.time()

    for i in range(len(sample)):
        parent = sample[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, D_CHILD,
                                     n_half_child=N_HALF_CHILD)
        if result is None:
            totals['n_no_range'] += 1
            continue

        lo_arr, hi_arr, total_children = result
        if total_children == 0:
            totals['n_no_range'] += 1
            continue

        totals['n_cartesian'] += total_children
        max_buf = min(total_children, 500_000)
        out_buf = np.empty((max_buf, D_CHILD), dtype=np.int32)

        t_parent = time.time()
        stats = _fused_generate_and_prune_instrumented(
            parent, N_HALF_CHILD, M, C_TARGET, lo_arr, hi_arr, out_buf)
        dt_parent = time.time() - t_parent

        n_surv = stats[0]
        n_fast, n_short, n_deep = int(stats[1]), int(stats[2]), int(stats[3])
        n_sub_s, n_sub_skip = int(stats[4]), int(stats[5])
        n_qc, n_fs, n_vis = int(stats[6]), int(stats[7]), int(stats[8])

        if n_vis == 0:
            totals['n_asym_skipped'] += 1

        totals['n_surv'] += n_surv
        totals['n_fast'] += n_fast
        totals['n_short'] += n_short
        totals['n_deep'] += n_deep
        totals['n_subtree_success'] += n_sub_s
        totals['n_subtree_children_skipped'] += n_sub_skip
        totals['n_qc_hit'] += n_qc
        totals['n_full_scan'] += n_fs
        totals['n_visited'] += n_vis

        per_parent.append({
            'idx': i, 'cartesian': total_children, 'visited': n_vis,
            'fast': n_fast, 'short': n_short, 'deep': n_deep,
            'sub_success': n_sub_s, 'sub_skipped': n_sub_skip,
            'qc_hit': n_qc, 'full_scan': n_fs,
            'survivors': n_surv, 'time': dt_parent,
        })

        elapsed = time.time() - t0
        if (i + 1) <= 5 or (i + 1) % 10 == 0:
            rate = (i + 1) / elapsed
            eta = (len(sample) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(sample)}] {dt_parent:.2f}s this parent, "
                  f"cart={total_children:,}, vis={n_vis:,}, "
                  f"sub_skip={n_sub_skip:,}, surv={n_surv}, "
                  f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    # --- Report ---
    T = totals
    cart = T['n_cartesian']
    vis = T['n_visited']
    n_advances = T['n_fast'] + T['n_short'] + T['n_deep']

    print(f"{'='*72}")
    print(f"  INSTRUMENTATION REPORT ({len(sample)} parents, {len(per_parent)} processed)")
    print(f"{'='*72}")
    print(f"  No-range / empty parents:  {T['n_no_range']}")
    print(f"  Asymmetry-skipped parents: {T['n_asym_skipped']}")
    print(f"  Total Cartesian product:   {cart:>18,}")
    print(f"  Children visited:          {vis:>18,}")
    print(f"  Advance steps:             {n_advances:>18,}")
    print(f"  Survivors:                 {T['n_surv']:>18,}")

    print(f"\n  Path distribution ({n_advances:,} advance steps):")
    for name, val in [('Fast path (n_changed=1)', T['n_fast']),
                      ('Short carry (2<=n<=thr)', T['n_short']),
                      ('Deep carry (n>thr)', T['n_deep'])]:
        pct = 100 * val / max(1, n_advances)
        print(f"    {name:<28} {val:>14,}  ({pct:>5.1f}%)")

    print(f"\n  Subtree pruning (odometer deep-carry variant):")
    print(f"    Deep carries attempted:    {T['n_deep']:>14,}")
    print(f"    Subtree prunes succeeded:  {T['n_subtree_success']:>14,}"
          f"  ({100*T['n_subtree_success']/max(1,T['n_deep']):>5.1f}% of deep)")
    print(f"    Children skipped:          {T['n_subtree_children_skipped']:>14,}"
          f"  ({100*T['n_subtree_children_skipped']/max(1,cart):>5.1f}% of Cartesian)")
    print(f"    Effective skip rate:       "
          f"{100*T['n_subtree_children_skipped']/max(1,T['n_subtree_children_skipped']+vis):>5.1f}%"
          f"  (skipped / (skipped + visited))")

    print(f"\n  Quick-check:")
    print(f"    QC hits:                   {T['n_qc_hit']:>14,}"
          f"  ({100*T['n_qc_hit']/max(1,vis):>5.1f}% of visited)")
    print(f"    Full scans:                {T['n_full_scan']:>14,}"
          f"  ({100*T['n_full_scan']/max(1,vis):>5.1f}% of visited)")

    print(f"\n  Gray code projection:")
    print(f"    Gray code would visit:     {cart:>14,}  (100% of Cartesian)")
    additional = cart - vis
    print(f"    Odometer subtree-skipped:  {T['n_subtree_children_skipped']:>14,}")
    print(f"    Gray code extra children:  {additional:>14,}"
          f"  ({100*additional/max(1,vis):>5.1f}% more than odometer visited)")

    # Compute per-parent statistics
    if per_parent:
        times = [p['time'] for p in per_parent]
        carts = [p['cartesian'] for p in per_parent]
        sub_rates = [100 * p['sub_skipped'] / max(1, p['cartesian'])
                     for p in per_parent if p['cartesian'] > 0]

        print(f"\n  Per-parent timing:")
        print(f"    Mean:   {np.mean(times):.2f}s")
        print(f"    Median: {np.median(times):.2f}s")
        print(f"    Min:    {np.min(times):.2f}s")
        print(f"    Max:    {np.max(times):.2f}s")
        print(f"    Total:  {np.sum(times):.1f}s")

        print(f"\n  Per-parent Cartesian product:")
        print(f"    Mean:   {np.mean(carts):,.0f}")
        print(f"    Median: {np.median(carts):,.0f}")
        print(f"    Min:    {np.min(carts):,}")
        print(f"    Max:    {np.max(carts):,}")

        if sub_rates:
            print(f"\n  Per-parent subtree skip rate (% of Cartesian):")
            print(f"    Mean:   {np.mean(sub_rates):.1f}%")
            print(f"    Median: {np.median(sub_rates):.1f}%")
            print(f"    Min:    {np.min(sub_rates):.1f}%")
            print(f"    Max:    {np.max(sub_rates):.1f}%")

            # Distribution
            bins = [0, 1, 5, 10, 20, 50, 100.01]
            labels = ['0-1%', '1-5%', '5-10%', '10-20%', '20-50%', '50-100%']
            hist, _ = np.histogram(sub_rates, bins=bins)
            print(f"\n    Distribution:")
            for label, count in zip(labels, hist):
                print(f"      {label:<10} {count:>4} parents "
                      f"({100*count/len(sub_rates):.0f}%)")

    print(f"{'='*72}")


if __name__ == '__main__':
    main()
