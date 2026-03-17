"""Instrumented benchmark — measure carry patterns and subtree pruning effectiveness.

Runs the instrumented variant of _fused_generate_and_prune on the same 100
L3 parents (seed=42) as run_benchmark.py and reports detailed statistics.

Usage:
    python -m baseline.run_instrumented
    python -m baseline.run_instrumented --n_sample 20   # quick check
"""
import argparse
import os
import sys
import time

import numpy as np

# -- Path setup --
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

print("Loading modules + JIT warmup...", flush=True)
t_load = time.time()
from cpu.run_cascade import (
    _fused_generate_and_prune_instrumented,
    _fused_generate_and_prune,
    _compute_bin_ranges,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

# Constants — must match run_benchmark.py
N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
N_HALF_CHILD = N_HALF * (2 ** 4)  # = 32

CHECKPOINT_PATH = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')


def load_sample(n_sample, seed=42):
    print(f"Loading L3 survivors from {CHECKPOINT_PATH}...")
    parents = np.load(CHECKPOINT_PATH)
    n_total = len(parents)
    print(f"  {n_total:,} parents, shape={parents.shape}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    indices.sort()
    sample = parents[indices]
    del parents
    print(f"  Sampled {len(sample)} parents (seed={seed})")
    return sample


def warmup_jit():
    """Warm up JIT for both original and instrumented functions."""
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
    _fused_generate_and_prune_instrumented(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)


def main():
    parser = argparse.ArgumentParser(description='Instrumented L4 benchmark')
    parser.add_argument('--n_sample', type=int, default=100,
                        help='Number of L3 parents to sample (default: 100)')
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: L3 checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    sample = load_sample(args.n_sample)

    print("Warming up JIT...", flush=True)
    warmup_jit()

    # --- Run instrumented version ---
    print(f"\nProcessing {len(sample)} parents (instrumented)...", flush=True)

    totals = {
        'n_surv': 0,
        'n_fast': 0, 'n_short': 0, 'n_deep': 0,
        'n_subtree_success': 0, 'n_subtree_children_skipped': 0,
        'n_qc_hit': 0, 'n_full_scan': 0, 'n_visited': 0,
        'n_cartesian': 0,
        'n_asym_skipped': 0,
    }

    t0 = time.time()
    for i in range(len(sample)):
        parent = sample[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, D_CHILD)
        if result is None:
            continue

        lo_arr, hi_arr, total_children = result
        if total_children == 0:
            continue

        totals['n_cartesian'] += total_children
        max_buf = min(total_children, 500_000)
        out_buf = np.empty((max_buf, D_CHILD), dtype=np.int32)

        stats = _fused_generate_and_prune_instrumented(
            parent, N_HALF_CHILD, M, C_TARGET, lo_arr, hi_arr, out_buf)

        # Unpack 9-tuple
        n_surv, n_fast, n_short, n_deep = stats[0], stats[1], stats[2], stats[3]
        n_sub_s, n_sub_skip = stats[4], stats[5]
        n_qc, n_fs, n_vis = stats[6], stats[7], stats[8]

        if n_vis == 0:
            totals['n_asym_skipped'] += 1

        totals['n_surv'] += n_surv
        totals['n_fast'] += int(n_fast)
        totals['n_short'] += int(n_short)
        totals['n_deep'] += int(n_deep)
        totals['n_subtree_success'] += int(n_sub_s)
        totals['n_subtree_children_skipped'] += int(n_sub_skip)
        totals['n_qc_hit'] += int(n_qc)
        totals['n_full_scan'] += int(n_fs)
        totals['n_visited'] += int(n_vis)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample)} parents, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    # --- Report ---
    T = totals
    cart = T['n_cartesian']
    vis = T['n_visited']
    n_advances = T['n_fast'] + T['n_short'] + T['n_deep']

    print(f"{'='*65}")
    print(f"  INSTRUMENTATION REPORT ({len(sample)} parents)")
    print(f"{'='*65}")
    print(f"  Asymmetry-skipped parents: {T['n_asym_skipped']}")
    print(f"  Total Cartesian product:   {cart:>14,}")
    print(f"  Children visited:          {vis:>14,}")
    print(f"  Advance steps:             {n_advances:>14,}")
    print(f"  Survivors:                 {T['n_surv']:>14,}")

    print(f"\n  Path distribution ({n_advances:,} advance steps):")
    for name, val in [('Fast path (n_changed=1)', T['n_fast']),
                      ('Short carry (2<=n<=thr)', T['n_short']),
                      ('Deep carry (n>thr)', T['n_deep'])]:
        pct = 100 * val / max(1, n_advances)
        print(f"    {name:<28} {val:>12,}  ({pct:>5.1f}%)")

    print(f"\n  Subtree pruning:")
    print(f"    Deep carries attempted:    {T['n_deep']:>12,}")
    print(f"    Subtree prunes succeeded:  {T['n_subtree_success']:>12,}"
          f"  ({100*T['n_subtree_success']/max(1,T['n_deep']):>5.1f}% of deep)")
    print(f"    Children skipped:          {T['n_subtree_children_skipped']:>12,}"
          f"  ({100*T['n_subtree_children_skipped']/max(1,cart):>5.1f}% of Cartesian)")
    print(f"    Effective skip rate:       "
          f"{100*T['n_subtree_children_skipped']/max(1,T['n_subtree_children_skipped']+vis):>5.1f}%"
          f"  (skipped / (skipped + visited))")

    print(f"\n  Quick-check:")
    print(f"    QC hits:                   {T['n_qc_hit']:>12,}"
          f"  ({100*T['n_qc_hit']/max(1,vis):>5.1f}% of visited)")
    print(f"    Full scans:                {T['n_full_scan']:>12,}"
          f"  ({100*T['n_full_scan']/max(1,vis):>5.1f}% of visited)")

    print(f"\n  Gray code projection:")
    print(f"    Gray code would visit:     {cart:>12,}  (100% of Cartesian)")
    additional = cart - vis
    print(f"    Additional children:       {additional:>12,}"
          f"  ({100*additional/max(1,vis):>5.1f}% more work)")

    print(f"{'='*65}")


if __name__ == '__main__':
    main()
