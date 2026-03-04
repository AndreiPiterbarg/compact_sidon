"""Experiment: Does checking window sizes ell > d prune additional children?

For L3 survivors (d_parent=32), children have d_child=64.
The autoconvolution has conv_len = 2*64-1 = 127 values.
Window size ell checks convolution sums of ell-1 consecutive values.

Question: If we check ell=2..64 (old bound) vs ell=2..128 (new/current bound),
do the extra windows ell=65..128 actually prune additional children?

We sample parents from L3, generate all children for each,
and compute convolution once, then scan windows in two ranges.

EFFICIENT: single autoconvolution, two window passes.
"""
import sys
import os
import time
import math
import numpy as np

# Path setup
_proj = r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon"
_cs_dir = os.path.join(_proj, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)

from pruning import correction
import numba
from numba import njit, prange
import itertools


@njit(parallel=True, cache=False)
def _compare_window_bounds(batch_int, n_half, inv_m, d_child, prune_threshold):
    """For each child, compute TV with ell<=d and ell<=2d.

    Returns (B, 3) array:
      col 0: TV with ell=2..d  (old bound)
      col 1: TV with ell=2..2d (new bound)
      col 2: best ell that achieves the new-bound max
    """
    B = batch_int.shape[0]
    d = d_child
    conv_len = 2 * d - 1
    scale = 4.0 * n_half * inv_m
    result = np.empty((B, 3), dtype=np.float64)

    for b in prange(B):
        # Autoconvolution (compute once)
        conv = np.zeros(conv_len, dtype=np.float64)
        for i in range(d):
            ai = batch_int[b, i] * scale
            for j in range(d):
                conv[i + j] += ai * batch_int[b, j] * scale

        # Prefix sums (in-place)
        for k in range(1, conv_len):
            conv[k] += conv[k - 1]

        # Pass 1: window max for ell=2..d (old bound)
        best_old = 0.0
        for ell in range(2, d + 1):
            n_cv = ell - 1
            inv_norm = 1.0 / (4.0 * n_half * ell)
            for s_lo in range(conv_len - n_cv + 1):
                s_hi = s_lo + n_cv - 1
                ws = conv[s_hi]
                if s_lo > 0:
                    ws -= conv[s_lo - 1]
                tv = ws * inv_norm
                if tv > best_old:
                    best_old = tv

        # Pass 2: window max for ell=2..2d (new bound) — includes old range
        best_new = best_old  # can only be >= old
        best_ell = 0
        for ell in range(d + 1, 2 * d + 1):
            n_cv = ell - 1
            inv_norm = 1.0 / (4.0 * n_half * ell)
            n_windows = conv_len - n_cv + 1
            if n_windows <= 0:
                break
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                ws = conv[s_hi]
                if s_lo > 0:
                    ws -= conv[s_lo - 1]
                tv = ws * inv_norm
                if tv > best_new:
                    best_new = tv
                    best_ell = ell

        result[b, 0] = best_old
        result[b, 1] = best_new
        result[b, 2] = best_ell  # 0 means old bound was sufficient

    return result


def generate_children_for_parent(parent_int, m, c_target):
    """Generate all child compositions from a parent via uniform 2-split."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)

    per_bin_choices = []
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            return np.empty((0, d_child), dtype=np.int32)
        per_bin_choices.append(list(range(lo, hi + 1)))

    total = 1
    for choices in per_bin_choices:
        total *= len(choices)

    if total == 0:
        return np.empty((0, d_child), dtype=np.int32)

    # Cap for sanity — skip parents with too many children
    if total > 500_000:
        return None  # signal "too many"

    children = np.empty((total, d_child), dtype=np.int32)
    idx = 0
    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            children[idx, 2 * i] = combo[i]
            children[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1

    return children


def main():
    # Parameters from checkpoint
    n_half = 2
    m = 20
    c_target = 1.4
    d_parent = 32
    d_child = 64
    inv_m = 1.0 / m
    corr = correction(m)
    prune_threshold = c_target + corr

    print(f"Parameters: n_half={n_half}, m={m}, c_target={c_target}")
    print(f"d_parent={d_parent}, d_child={d_child}")
    print(f"correction={corr:.6f}, prune_threshold={prune_threshold:.6f}")
    print(f"Old bound: ell = 2..{d_child}")
    print(f"New bound: ell = 2..{2*d_child}")
    print(flush=True)

    # Load L3 survivors
    survivors_path = os.path.join(_proj, "data", "checkpoint_L3_survivors.npy")
    print(f"Loading L3 survivors from {survivors_path}...")
    sys.stdout.flush()
    survivors = np.load(survivors_path)
    print(f"Loaded {len(survivors)} L3 survivors, shape={survivors.shape}")
    print(flush=True)

    # Sample parents
    rng = np.random.default_rng(42)
    n_sample = 100
    sample_idx = rng.choice(len(survivors), size=n_sample, replace=False)
    parents = survivors[sample_idx]
    print(f"Sampled {n_sample} parents (seed=42)")
    print(flush=True)

    # JIT warmup with a tiny batch (d=4)
    print("JIT warmup...", flush=True)
    tiny = np.array([[5, 5, 5, 5]], dtype=np.int32)
    _ = _compare_window_bounds(tiny, 2, 0.05, 4, 1.5)
    print("JIT warmup done.", flush=True)
    print(flush=True)

    # First pass: check how many children each parent generates
    print("Pre-scanning parent child counts...", flush=True)
    child_counts = []
    for p_idx in range(n_sample):
        parent = parents[p_idx]
        d_p = len(parent)
        thresh = c_target + corr + 1e-9
        x_cap = int(math.floor(m * math.sqrt(thresh / (2 * d_p))))
        x_cap = min(x_cap, m)
        total = 1
        for i in range(d_p):
            b_i = int(parent[i])
            lo = max(0, b_i - x_cap)
            hi = min(b_i, x_cap)
            if lo > hi:
                total = 0
                break
            total *= (hi - lo + 1)
        child_counts.append(total)

    child_counts = np.array(child_counts)
    print(f"Child count stats: min={child_counts.min()}, median={np.median(child_counts):.0f}, "
          f"max={child_counts.max()}, mean={child_counts.mean():.0f}")
    print(f"Parents with <= 500K children: {np.sum(child_counts <= 500000)}/{n_sample}")
    print(flush=True)

    # Process parents with manageable child counts
    total_children_tested = 0
    total_pruned_both = 0
    total_pruned_new_only = 0
    total_survived_both = 0
    parents_with_extra_pruning = 0
    parents_processed = 0
    parents_skipped = 0

    # Track largest TV difference
    max_tv_diff = 0.0

    t0 = time.time()
    for p_idx in range(n_sample):
        parent = parents[p_idx]
        children = generate_children_for_parent(parent, m, c_target)

        if children is None:
            parents_skipped += 1
            continue
        if len(children) == 0:
            continue

        n_children = len(children)
        total_children_tested += n_children
        parents_processed += 1

        # Compute both window bounds in one pass
        results = _compare_window_bounds(children, n_half, inv_m, d_child, prune_threshold)

        tv_old = results[:, 0]
        tv_new = results[:, 1]

        # Classify
        pruned_old = tv_old >= prune_threshold
        pruned_new = tv_new >= prune_threshold

        n_pruned_both = int(np.sum(pruned_old & pruned_new))
        n_pruned_new_only = int(np.sum(~pruned_old & pruned_new))
        n_survived_both = int(np.sum(~pruned_old & ~pruned_new))

        total_pruned_both += n_pruned_both
        total_pruned_new_only += n_pruned_new_only
        total_survived_both += n_survived_both

        if n_pruned_new_only > 0:
            parents_with_extra_pruning += 1

        # Track largest TV increase from larger windows
        diffs = tv_new - tv_old
        if len(diffs) > 0:
            this_max = float(np.max(diffs))
            if this_max > max_tv_diff:
                max_tv_diff = this_max

        if parents_processed % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {parents_processed} parents ({p_idx+1}/{n_sample}): "
                  f"{n_children} children, extra_prune={n_pruned_new_only}, "
                  f"elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Parents sampled: {n_sample}")
    print(f"Parents processed (<=500K children): {parents_processed}")
    print(f"Parents skipped (>500K children):    {parents_skipped}")
    print(f"Total children tested: {total_children_tested:,}")
    print(f"Time: {elapsed:.1f}s")
    print()
    print(f"Pruned by BOTH bounds (ell<=d and ell<=2d):  {total_pruned_both:>12,}")
    print(f"Pruned ONLY by new bound (ell>d matters):    {total_pruned_new_only:>12,}")
    print(f"Survived with BOTH bounds:                   {total_survived_both:>12,}")
    print()
    print(f"Max TV increase from ell>d windows: {max_tv_diff:.8f}")

    if total_children_tested > 0:
        pct_extra = 100.0 * total_pruned_new_only / total_children_tested
        pct_pruned_old = 100.0 * total_pruned_both / total_children_tested
        pct_survived = 100.0 * total_survived_both / total_children_tested
        print(f"\n% pruned by old bound alone:          {pct_pruned_old:.4f}%")
        print(f"% extra pruning from ell>d:            {pct_extra:.4f}%")
        print(f"% survived (neither bound):            {pct_survived:.4f}%")

        if total_survived_both + total_pruned_new_only > 0:
            extra_reduction_pct = 100.0 * total_pruned_new_only / (total_survived_both + total_pruned_new_only)
            print(f"% of old-survivors killed by ell>d:    {extra_reduction_pct:.4f}%")

    print(f"\nParents with any extra pruning from ell>d: {parents_with_extra_pruning}/{parents_processed}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
