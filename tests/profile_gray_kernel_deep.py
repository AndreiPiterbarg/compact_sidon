#!/usr/bin/env python
"""Deep investigation: why does L4 (d=64) have 100% survival rate?

Hypothesis: at d_child=64 with m=20, the pruning threshold is too loose.
The test-value bound is: ws > floor(c_target * m^2 * ell / (4n) + 1 + eps + 2*W_int)

At d_child=64, n_half_child=32, so 4n = 128.
For ell near d_child/2=32: threshold_base = 1.4 * 400 * 32/128 = 140.0
And W_int can be up to m=20, so +1+40 = 181.

Meanwhile, the maximum possible convolution window sum depends on the child
composition. With only 20 units of mass spread over 64 bins, the convolutions
are very dilute.

Let's verify by computing actual test values for some children and comparing
to thresholds.
"""

import os
import sys
import math
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)
sys.path.insert(0, os.path.join(_cs_dir, "cpu"))

from numba import njit
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
    x_cap = min(x_cap, x_cap_cs)
    x_cap = min(x_cap, m)
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


def main():
    print("=" * 80)
    print("DEEP INVESTIGATION: WHY DOES L4 HAVE 100% SURVIVAL?")
    print("=" * 80)
    print()

    # Load one parent
    l3_path = os.path.join(DATA_DIR, "checkpoint_L3_survivors.npy")
    l3_data = np.load(l3_path, mmap_mode='r')

    rng = np.random.RandomState(42)
    idx = rng.choice(l3_data.shape[0])
    parent = np.array(l3_data[idx], dtype=np.int32)
    del l3_data

    d_parent = len(parent)
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    print(f"Sample parent (d={d_parent}): {parent}")
    print(f"  Parent sum: {parent.sum()}")
    print(f"  Nonzero bins: {np.count_nonzero(parent)}/{d_parent}")
    print()

    # Correction and thresholds
    corr = correction(M, n_half_child)
    print(f"  correction(m={M}, n_half_child={n_half_child}) = {corr:.6f}")
    print(f"  c_target + correction = {C_TARGET + corr:.6f}")
    print()

    result = compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
    if result is None:
        print("  EMPTY range!")
        return
    lo_arr, hi_arr, total_children = result
    ranges = hi_arr - lo_arr + 1

    thresh_cs = C_TARGET + corr + 1e-9
    x_cap = int(math.floor(M * math.sqrt(thresh_cs / d_child)))
    x_cap_cs = int(math.floor(M * math.sqrt(C_TARGET / d_child)))
    x_cap = min(x_cap, x_cap_cs, M)
    print(f"  x_cap = {x_cap}")
    print(f"  Cursor ranges: {ranges}")
    print(f"  Total children: {total_children:,}")
    print()

    # -----------------------------------------------------------------------
    # Compute pruning thresholds for all ell values
    # -----------------------------------------------------------------------
    print("--- PRUNING THRESHOLDS ---")
    print()
    inv_4n = 1.0 / (4.0 * n_half_child)
    m_d = float(M)
    eps_margin = 1e-9 * m_d * m_d
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    print(f"  {'ell':>4s} {'base_thresh':>12s} {'+1+2*m':>10s} {'int_thresh':>12s} "
          f"{'max_ws_theory':>14s} {'max_ws/thresh':>14s}")
    print(f"  {'-'*4} {'-'*12} {'-'*10} {'-'*12} {'-'*14} {'-'*14}")

    for ell in [2, 4, 8, 16, 32, 33, 34, 48, 64, 96, 127]:
        if ell > 2 * d_child:
            continue
        base = C_TARGET * m_d * m_d * ell * inv_4n
        # Worst case W_int = m (all mass in window)
        dyn_x = base + 1.0 + eps_margin + 2.0 * m_d
        dyn_it = int(dyn_x * one_minus_4eps)

        # Maximum possible ws for a window of size ell-1 in convolution
        # The convolution has length 2*d_child - 1 = 127
        # Maximum sum in any window: upper bound is m^2 = 400 (all mass concentrated)
        # But with m=20 spread over 64 bins, actual values are much lower.
        # Theoretical max: if all mass m in 2 adjacent bins (child[k]=m, child[k+1]=0),
        # then conv[2k] = m^2 = 400, conv[2k+1] = 0.
        # For a window of size ell-1, max is m^2 = 400.
        max_ws_theory = M * M  # absolute max

        ratio = max_ws_theory / max(dyn_it, 1)
        print(f"  {ell:>4d} {base:>12.1f} {dyn_x:>10.1f} {dyn_it:>12d} "
              f"{max_ws_theory:>14d} {ratio:>14.4f}")

    print()

    # -----------------------------------------------------------------------
    # Compute actual test values for first child and some random children
    # -----------------------------------------------------------------------
    print("--- ACTUAL TEST VALUES FOR SAMPLE CHILDREN ---")
    print()

    # Build first child (all cursors at lo)
    child0 = np.empty(d_child, dtype=np.int32)
    for i in range(d_parent):
        child0[2 * i] = lo_arr[i]
        child0[2 * i + 1] = parent[i] - lo_arr[i]

    # Build a "concentrated" child (try to maximize mass concentration)
    child_conc = np.empty(d_child, dtype=np.int32)
    for i in range(d_parent):
        child_conc[2 * i] = hi_arr[i]  # max into even bins
        child_conc[2 * i + 1] = parent[i] - hi_arr[i]

    for label, child in [("First child (lo cursors)", child0),
                         ("Concentrated child (hi cursors)", child_conc)]:
        print(f"  {label}:")
        print(f"    child = {child}")
        print(f"    nonzero: {np.count_nonzero(child)}/{d_child}, sum={child.sum()}")

        # Compute full convolution
        conv_len = 2 * d_child - 1
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            if child[i] != 0:
                conv[2 * i] += child[i] * child[i]
                for j in range(i + 1, d_child):
                    if child[j] != 0:
                        conv[i + j] += 2 * child[i] * child[j]

        print(f"    conv max: {conv.max()}")
        print(f"    conv sum: {conv.sum()}")
        print(f"    conv nonzero: {np.count_nonzero(conv)}/{conv_len}")

        # Prefix sum for child masses
        prefix_c = np.zeros(d_child + 1, dtype=np.int64)
        for i in range(d_child):
            prefix_c[i + 1] = prefix_c[i] + child[i]

        # Find maximum window sum for each ell
        print(f"    {'ell':>6s} {'max_window_sum':>14s} {'threshold':>12s} {'pruned?':>8s} {'ratio':>8s}")
        for ell in [2, 4, 8, 16, 32, 33, 34, 48, 64, 96, 127]:
            if ell > 2 * d_child:
                continue
            n_cv = ell - 1
            base = C_TARGET * m_d * m_d * ell * inv_4n

            max_ws = 0
            best_s = -1
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                ws = int(np.sum(conv[s_lo:s_lo + n_cv]))
                # Compute W_int
                lo_bin = max(0, s_lo - (d_child - 1))
                hi_bin = min(d_child - 1, s_lo + ell - 2)
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]

                dyn_x = base + 1.0 + eps_margin + 2.0 * float(W_int)
                dyn_it = int(dyn_x * one_minus_4eps)

                # Check if this window would prune
                if ws > dyn_it:
                    if best_s == -1:
                        best_s = s_lo

                if ws > max_ws:
                    max_ws = ws
                    best_W = int(W_int)

            # Threshold for the best W_int
            dyn_x_best = base + 1.0 + eps_margin + 2.0 * float(best_W)
            dyn_it_best = int(dyn_x_best * one_minus_4eps)
            pruned = "YES" if max_ws > dyn_it_best else "no"
            ratio = max_ws / max(dyn_it_best, 1)

            print(f"    {ell:>6d} {max_ws:>14d} {dyn_it_best:>12d} {pruned:>8s} {ratio:>8.4f}")

        print()

    # -----------------------------------------------------------------------
    # Compare L3 parameters vs L4
    # -----------------------------------------------------------------------
    print("--- COMPARISON: L3 vs L4 PRUNING REGIME ---")
    print()

    for level, d_ch, n_half_ch in [(3, 32, 16), (4, 64, 32)]:
        corr_lev = correction(M, n_half_ch)
        thresh_lev = C_TARGET + corr_lev
        x_cap_lev = int(math.floor(M * math.sqrt(thresh_lev / d_ch)))
        x_cap_cs_lev = int(math.floor(M * math.sqrt(C_TARGET / d_ch)))
        x_cap_lev = min(x_cap_lev, x_cap_cs_lev, M)
        inv_4n_lev = 1.0 / (4.0 * n_half_ch)

        # At ell ~ d_ch: threshold ~ c_target * m^2 * d_ch / (4*n_half_ch)
        # = c_target * m^2 * 2 = 1.4 * 400 * 2 = 1120 (independent of d!)
        # But max conv value ~ (m/sqrt(d_ch))^2 * d_ch = m^2 = 400
        # So threshold ~ 2.8 * max_possible_ws at ell=d_ch

        ell_half = d_ch // 2
        base_half = C_TARGET * M * M * ell_half * inv_4n_lev
        base_full = C_TARGET * M * M * d_ch * inv_4n_lev

        print(f"  Level L{level} (d_child={d_ch}, n_half_child={n_half_ch}):")
        print(f"    correction = {corr_lev:.4f}")
        print(f"    x_cap = {x_cap_lev}")
        print(f"    inv_4n = {inv_4n_lev:.6f}")
        print(f"    At ell={ell_half}: base_thresh = {base_half:.1f}")
        print(f"    At ell={d_ch}: base_thresh = {base_full:.1f}")
        print(f"    m^2 (max possible ws entry) = {M*M}")
        print(f"    Ratio base_thresh(ell=d/2) / m^2 = {base_half / (M*M):.3f}")
        print(f"    Effective: child mass spread over {d_ch} bins -> avg mass/bin = {M/d_ch:.3f}")
        print(f"    Cauchy-Schwarz max mass per bin: x_cap = {x_cap_lev}")
        print()

    # -----------------------------------------------------------------------
    # Scaling analysis: as d grows, does pruning ever work?
    # -----------------------------------------------------------------------
    print("--- SCALING ANALYSIS: PRUNING EFFECTIVENESS vs d ---")
    print()
    print(f"  {'d_child':>8s} {'n_half':>6s} {'x_cap':>6s} {'ranges/bin':>12s} "
          f"{'base_thresh(d/2)':>18s} {'max_conv_sum':>14s} {'ratio':>8s}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*18} {'-'*14} {'-'*8}")

    for d_ch in [4, 8, 16, 32, 64, 128]:
        nh = d_ch // 2
        corr_d = correction(M, nh)
        thresh_d = C_TARGET + corr_d
        xc = int(math.floor(M * math.sqrt(thresh_d / d_ch)))
        xc_cs = int(math.floor(M * math.sqrt(C_TARGET / d_ch)))
        xc = min(xc, xc_cs, M)
        inv_4n_d = 1.0 / (4.0 * nh)
        ell_half = d_ch // 2
        base_half = C_TARGET * M * M * ell_half * inv_4n_d

        # Max conv sum in a window of ell_half-1 terms:
        # If mass m is spread evenly, conv peaks ~ m^2/d. Window of d/2
        # terms sums to ~ m^2/2 (roughly).
        # More precisely: uniform composition a_i=m/d gives
        # conv[k] = sum_{i+j=k} 2*a_i*a_j ~ 2*(m/d)^2 * min(k+1, d, 2d-1-k)
        # Peak at k=d-1: conv[d-1] ~ 2*(m/d)^2 * d = 2*m^2/d
        # Window sum of d/2 terms around peak: ~ 2*m^2/d * d/2 = m^2
        # This is very rough. Let's compute it.

        # Uniform child for this d_ch:
        if d_ch <= 64:
            # Compute exact conv for a "typical" child
            # Use a spread composition: 20 mass over d_ch bins, ~0 or 1 each
            child_test = np.zeros(d_ch, dtype=np.int32)
            # Spread M mass evenly across d_ch bins
            step = max(d_ch // M, 1)
            placed = 0
            for i in range(d_ch):
                if placed < M and (i % step == 0 or d_ch - i <= M - placed):
                    child_test[i] += 1
                    placed += 1
            # If still deficit, add to first bins
            while child_test.sum() < M:
                for i in range(d_ch):
                    if child_test.sum() >= M:
                        break
                    if child_test[i] < xc:
                        child_test[i] += 1

            conv_test_len = 2 * d_ch - 1
            conv_test = np.zeros(conv_test_len, dtype=np.int64)
            for i in range(d_ch):
                if child_test[i] != 0:
                    conv_test[2 * i] += child_test[i] ** 2
                    for j in range(i + 1, d_ch):
                        if child_test[j] != 0:
                            conv_test[i + j] += 2 * child_test[i] * child_test[j]

            # Window of ell_half-1 terms
            n_cv = max(ell_half - 1, 1)
            max_ws_test = 0
            for s in range(conv_test_len - n_cv + 1):
                ws = int(np.sum(conv_test[s:s + n_cv]))
                if ws > max_ws_test:
                    max_ws_test = ws
        else:
            max_ws_test = -1

        ratio = max_ws_test / max(base_half, 1) if max_ws_test > 0 else -1
        # ranges per bin: each parent bin has range up to 2*x_cap+1
        rng_per = min(2 * xc + 1, M + 1)
        print(f"  {d_ch:>8d} {nh:>6d} {xc:>6d} {rng_per:>12d} "
              f"{base_half:>18.1f} {max_ws_test:>14d} {ratio:>8.4f}")

    print()
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("  As d_child grows, the base threshold scales as:")
    print("    base_thresh(ell=d/2) = c_target * m^2 * (d/2) / (4 * d/2)")
    print("                        = c_target * m^2 / 4")
    print(f"                        = {C_TARGET} * {M**2} / 4 = {C_TARGET * M**2 / 4}")
    print()
    print("  But the actual window sums for spread-out compositions also scale")
    print("  with the number of cross-terms in the window.")
    print()
    print("  The critical ratio is: max_window_sum / threshold")
    print("  If this ratio < 1 for ALL windows and ALL children, then")
    print("  NO child can ever be pruned, and the level has 100% survival.")
    print()
    print("  This means L4 with (m=20, c_target=1.4) is fundamentally")
    print("  unable to prune anything -- the discretization is too coarse.")
    print("  Need higher m or lower c_target to make L4 effective.")


if __name__ == "__main__":
    main()
