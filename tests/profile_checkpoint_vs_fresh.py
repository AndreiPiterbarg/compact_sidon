#!/usr/bin/env python
"""Quick check: are L3 checkpoint survivors really the hardest parents?

Compare pruning stats for L3-checkpoint parents vs fresh L3 parents
generated from L2 cascade.
"""

import os, sys, math, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)

from pruning import correction

M = 20
C_TARGET = 1.4
DATA_DIR = os.path.join(_project_dir, "data")


def compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child):
    d_parent = len(parent_int)
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
    x_cap = min(x_cap, x_cap_cs, M)
    x_cap = max(x_cap, 0)
    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_children = 1
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_children *= (hi - lo + 1)
    return lo_arr, hi_arr, total_children


def analyze_parent_batch(parents, label):
    """Compute statistics about a batch of parents as L4 parents."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    total_children_list = []
    x_cap = None

    for parent in parents:
        result = compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            total_children_list.append(0)
        else:
            lo_arr, hi_arr, tc = result
            total_children_list.append(tc)
            if x_cap is None:
                corr = correction(M, n_half_child)
                thresh = C_TARGET + corr + 1e-9
                x_cap = int(math.floor(M * math.sqrt(thresh / d_child)))
                x_cap_cs = int(math.floor(M * math.sqrt(C_TARGET / d_child)))
                x_cap = min(x_cap, x_cap_cs, M)

    tc_arr = np.array(total_children_list)
    nz_counts = np.array([np.count_nonzero(p) for p in parents])
    max_vals = np.array([np.max(p) for p in parents])

    # Key metric: how many parent bins have mass > x_cap?
    # These bins have range=1 (cursor is fixed), contributing no branching
    bins_above_xcap = np.array([np.sum(p > x_cap) for p in parents])
    bins_at_xcap = np.array([np.sum(p == x_cap) for p in parents])

    print(f"  {label} ({len(parents)} parents, d_parent={d_parent}):")
    print(f"    x_cap = {x_cap}")
    print(f"    Nonzero bins:    mean={nz_counts.mean():.1f}, "
          f"min={nz_counts.min()}, max={nz_counts.max()}")
    print(f"    Max bin value:   mean={max_vals.mean():.1f}, "
          f"min={max_vals.min()}, max={max_vals.max()}")
    print(f"    Bins > x_cap:    mean={bins_above_xcap.mean():.1f}, "
          f"min={bins_above_xcap.min()}, max={bins_above_xcap.max()}")
    print(f"    Children/parent: mean={tc_arr.mean():.0f}, "
          f"min={tc_arr.min()}, max={tc_arr.max()}")

    # Mass distribution
    all_masses = parents.flatten()
    mass_hist = np.bincount(all_masses, minlength=M+1)
    print(f"    Mass distribution across all bins:")
    for v in range(min(M+1, 10)):
        if mass_hist[v] > 0:
            print(f"      mass={v}: {mass_hist[v]:,} bins ({100*mass_hist[v]/all_masses.size:.1f}%)")
    if np.any(all_masses >= 10):
        print(f"      mass>=10: {np.sum(all_masses >= 10):,} bins")

    # Conv value analysis: for a few parents, compute max conv entry
    print(f"    Autoconvolution peak analysis (first 5 parents):")
    for i in range(min(5, len(parents))):
        parent = parents[i]
        result = compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, tc = result

        # Build first child
        child = np.empty(d_child, dtype=np.int32)
        for j in range(d_parent):
            child[2*j] = lo_arr[j]
            child[2*j+1] = parent[j] - lo_arr[j]

        # Compute conv
        conv_len = 2*d_child - 1
        conv = np.zeros(conv_len, dtype=np.int64)
        for ii in range(d_child):
            if child[ii] != 0:
                conv[2*ii] += child[ii]**2
                for jj in range(ii+1, d_child):
                    if child[jj] != 0:
                        conv[ii+jj] += 2 * child[ii] * child[jj]

        # Best window for ell=33 (d_child/2+1 for d_child=64)
        inv_4n = 1.0 / (4.0 * n_half_child)
        ell = 33
        n_cv = ell - 1
        base = C_TARGET * M * M * ell * inv_4n
        max_ratio = 0
        for s_lo in range(conv_len - n_cv + 1):
            ws = int(np.sum(conv[s_lo:s_lo+n_cv]))
            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)
            prefix = np.zeros(d_child + 1, dtype=np.int64)
            for k in range(d_child):
                prefix[k+1] = prefix[k] + child[k]
            W_int = prefix[hi_bin+1] - prefix[lo_bin]
            thresh_val = base + 1.0 + 1e-9*M*M + 2.0*float(W_int)
            ratio = ws / max(thresh_val, 1)
            if ratio > max_ratio:
                max_ratio = ratio

        print(f"      parent {i}: max_conv={int(conv.max())}, "
              f"best ratio(ell=33)={max_ratio:.4f}, "
              f"nz_child={np.count_nonzero(child)}/{d_child}")

    print()


def main():
    print("=" * 80)
    print("CHECKPOINT vs FRESH PARENT COMPARISON")
    print("=" * 80)
    print()

    # Load L3 checkpoint parents
    l3_path = os.path.join(DATA_DIR, "checkpoint_L3_survivors.npy")
    l3_data = np.load(l3_path, mmap_mode='r')

    rng = np.random.RandomState(42)
    idx_ckpt = rng.choice(l3_data.shape[0], size=100, replace=False)
    checkpoint_parents = np.array([l3_data[i] for i in idx_ckpt], dtype=np.int32)
    del l3_data

    analyze_parent_batch(checkpoint_parents, "L3 CHECKPOINT parents (hardest)")

    # Generate fresh L3 parents from L0 -> L1 -> L2 cascade
    # Import the actual kernel
    sys.path.insert(0, os.path.join(_cs_dir, "cpu"))
    from run_cascade import _fused_generate_and_prune_gray, _compute_bin_ranges

    l0 = np.load(os.path.join(DATA_DIR, "checkpoint_L0_survivors.npy"))
    print("Running L1 cascade...")
    l1_survivors = []
    for parent in l0:
        d_c = 8; nhc = 4
        result = _compute_bin_ranges(parent, M, C_TARGET, d_c, nhc)
        if result is None: continue
        lo, hi, tc = result
        buf = np.empty((min(tc+1, 100000), d_c), dtype=np.int32)
        ns, _ = _fused_generate_and_prune_gray(parent, nhc, M, C_TARGET, lo, hi, buf)
        if ns > 0 and ns <= buf.shape[0]:
            l1_survivors.append(buf[:ns])
    l1 = np.vstack(l1_survivors)
    # Dedup
    sort_idx = np.lexsort(l1.T[::-1])
    l1 = l1[sort_idx]
    mask = np.ones(len(l1), dtype=bool)
    for i in range(1, len(l1)):
        if np.array_equal(l1[i], l1[i-1]):
            mask[i] = False
    l1 = l1[mask]
    print(f"  L1: {len(l1)} unique survivors")

    print("Running L2 cascade (sample 200 L1 parents)...")
    idx_l1 = rng.choice(len(l1), size=min(200, len(l1)), replace=False)
    l2_survivors = []
    for i in idx_l1:
        parent = l1[i]
        d_c = 16; nhc = 8
        result = _compute_bin_ranges(parent, M, C_TARGET, d_c, nhc)
        if result is None: continue
        lo, hi, tc = result
        buf = np.empty((min(tc+1, 500000), d_c), dtype=np.int32)
        ns, _ = _fused_generate_and_prune_gray(parent, nhc, M, C_TARGET, lo, hi, buf)
        if ns > 0 and ns <= buf.shape[0]:
            l2_survivors.append(buf[:ns])
    if l2_survivors:
        l2 = np.vstack(l2_survivors)
        sort_idx = np.lexsort(l2.T[::-1])
        l2 = l2[sort_idx]
        mask = np.ones(len(l2), dtype=bool)
        for i in range(1, len(l2)):
            if np.array_equal(l2[i], l2[i-1]):
                mask[i] = False
        l2 = l2[mask]
    else:
        l2 = np.empty((0, 16), dtype=np.int32)
    print(f"  L2: {len(l2)} unique survivors")

    print("Running L3 cascade (sample 50 L2 parents)...")
    idx_l2 = rng.choice(len(l2), size=min(50, len(l2)), replace=False)
    l3_survivors = []
    for i in idx_l2:
        parent = l2[i]
        d_c = 32; nhc = 16
        result = _compute_bin_ranges(parent, M, C_TARGET, d_c, nhc)
        if result is None: continue
        lo, hi, tc = result
        buf = np.empty((min(tc+1, 2000000), d_c), dtype=np.int32)
        ns, _ = _fused_generate_and_prune_gray(parent, nhc, M, C_TARGET, lo, hi, buf)
        if ns > 0 and ns <= buf.shape[0]:
            l3_survivors.append(buf[:ns])
    if l3_survivors:
        l3_fresh = np.vstack(l3_survivors)
        sort_idx = np.lexsort(l3_fresh.T[::-1])
        l3_fresh = l3_fresh[sort_idx]
        mask = np.ones(len(l3_fresh), dtype=bool)
        for i in range(1, len(l3_fresh)):
            if np.array_equal(l3_fresh[i], l3_fresh[i-1]):
                mask[i] = False
        l3_fresh = l3_fresh[mask]
    else:
        l3_fresh = np.empty((0, 32), dtype=np.int32)
    print(f"  L3 fresh: {len(l3_fresh)} unique survivors")
    print()

    if len(l3_fresh) > 0:
        idx_fresh = rng.choice(len(l3_fresh), size=min(100, len(l3_fresh)), replace=False)
        fresh_parents = l3_fresh[idx_fresh]
        analyze_parent_batch(fresh_parents, "FRESH L3 parents (from cascade)")


if __name__ == "__main__":
    main()
