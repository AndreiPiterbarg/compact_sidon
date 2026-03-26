"""A/B benchmark: baseline Gray code kernel vs quadratic range skipping.

Loads real L2 checkpoint parents (d=16 -> d=32 refinement) and runs both
kernels on identical inputs.  Verifies correctness (same survivor count)
and measures wall-clock speedup.

Usage:
    python bench_quadratic_skip.py [--n_parents 200] [--warmup 3]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import numba
from numba import njit

# ---------------------------------------------------------------------------
# Import project utilities
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.join(_this_dir, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)

from pruning import correction

# ---------------------------------------------------------------------------
# _compute_bin_ranges (copied verbatim from run_cascade.py)
# ---------------------------------------------------------------------------
def _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child=None):
    d_parent = len(parent_int)
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
    x_cap = min(x_cap, x_cap_cs, m)
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


# ===================================================================
# KERNEL A — BASELINE  (exact copy of _fused_generate_and_prune_gray)
# ===================================================================
@njit(cache=False)
def kernel_baseline(parent_int, n_half_child, m, c_target,
                    lo_arr, hi_arr, out_buf):
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    # --- Asymmetry filter ---
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    # --- Dynamic pruning constants ---
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    # --- Subtree pruning constants ---
    J_MIN = 7
    n_subtree_pruned = 0
    partial_conv = np.empty(conv_len, dtype=np.int32)

    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    # --- Allocate arrays ---
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    # --- Precompute per-ell constants ---
    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx] = 2.0 * np.float64(ell) * inv_4n

    # --- Optimized ell scan order ---
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell)
        ell_used[ell - 2] = np.int32(1)
        oi += 1
    for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                d_child * 2, d_child + d_child // 2, d_child // 2):
        if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    # --- Compute full raw_conv for initial child ---
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    # --- Gray code setup ---
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i
            radix[n_active] = r
            n_active += 1

    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    # --- Main loop ---
    while True:
        # === TEST current child ===
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + two_ell_arr[ell_idx_qc] * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True

        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            pruned = False
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                dyn_base_ell = dyn_base_ell_arr[ell_idx]
                two_ell_inv_4n = two_ell_arr[ell_idx]
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                    dyn_it = np.int64(dyn_x * one_minus_4eps)
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                use_rev = False
                for i in range(d_child):
                    jj = d_child - 1 - i
                    if child[jj] < child[i]:
                        use_rev = True
                        break
                    elif child[jj] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0

        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        if gc_a[j] == 0 or gc_a[j] == radix[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # === INCREMENTAL UPDATE ===
        k1 = 2 * pos
        k2 = k1 + 1
        old1 = np.int32(child[k1])
        old2 = np.int32(child[k2])
        child[k1] = cursor[pos]
        child[k2] = parent_int[pos] - cursor[pos]
        new1 = np.int32(child[k1])
        new2 = np.int32(child[k2])
        delta1 = new1 - old1
        delta2 = new2 - old2

        raw_conv[2 * k1] += new1 * new1 - old1 * old1
        raw_conv[2 * k2] += new2 * new2 - old2 * old2
        raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
        for jj in range(k1):
            cj = np.int32(child[jj])
            if cj != 0:
                raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        for jj in range(k2 + 1, d_child):
            cj = np.int32(child[jj])
            if cj != 0:
                raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                raw_conv[k2 + jj] += np.int32(2) * delta2 * cj

        if qc_ell > 0:
            qc_lo = qc_s - (d_child - 1)
            if qc_lo < 0:
                qc_lo = 0
            qc_hi = qc_s + qc_ell - 2
            if qc_hi > d_child - 1:
                qc_hi = d_child - 1
            if qc_lo <= k1 and k1 <= qc_hi:
                qc_W_int += np.int64(delta1)
            if qc_lo <= k2 and k2 <= qc_hi:
                qc_W_int += np.int64(delta2)

        # === SUBTREE PRUNING CHECK ===
        if j == J_MIN and n_active > J_MIN:
            fixed_parent_boundary = active_pos[J_MIN - 1]
            fixed_len = 2 * fixed_parent_boundary

            if fixed_len >= 4:
                partial_conv_len = 2 * fixed_len - 1
                for kk in range(partial_conv_len):
                    partial_conv[kk] = np.int32(0)
                for ii in range(fixed_len):
                    ci = np.int32(child[ii])
                    if ci != 0:
                        partial_conv[2 * ii] += ci * ci
                        for jj2 in range(ii + 1, fixed_len):
                            cj2 = np.int32(child[jj2])
                            if cj2 != 0:
                                partial_conv[ii + jj2] += np.int32(2) * ci * cj2
                for kk in range(1, partial_conv_len):
                    partial_conv[kk] += partial_conv[kk - 1]

                prefix_c[0] = 0
                for ii in range(fixed_len):
                    prefix_c[ii + 1] = prefix_c[ii] + np.int64(child[ii])

                first_unfixed_parent = fixed_parent_boundary
                subtree_pruned = False

                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    dyn_base_ell = dyn_base_ell_arr[ell_idx]
                    two_ell_inv_4n = two_ell_arr[ell_idx]

                    n_windows_partial = partial_conv_len - n_cv + 1
                    if n_windows_partial <= 0:
                        continue

                    for s_lo in range(n_windows_partial):
                        s_hi = s_lo + n_cv - 1
                        ws = np.int64(partial_conv[s_hi])
                        if s_lo > 0:
                            ws -= np.int64(partial_conv[s_lo - 1])

                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

                        fixed_hi = hi_bin
                        if fixed_hi > fixed_len - 1:
                            fixed_hi = fixed_len - 1
                        if fixed_hi >= lo_bin:
                            lo_clamp = lo_bin
                            if lo_clamp < 0:
                                lo_clamp = 0
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)

                        unfixed_lo_bin = lo_bin
                        if unfixed_lo_bin < fixed_len:
                            unfixed_lo_bin = fixed_len
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = unfixed_lo_bin // 2
                            p_hi = hi_bin // 2
                            if p_lo < first_unfixed_parent:
                                p_lo = first_unfixed_parent
                            if p_hi >= d_parent:
                                p_hi = d_parent - 1
                            if p_lo <= p_hi:
                                W_int_unfixed = parent_prefix[p_hi + 1] - parent_prefix[p_lo]
                            else:
                                W_int_unfixed = np.int64(0)
                        else:
                            W_int_unfixed = np.int64(0)

                        W_int_max = W_int_fixed + W_int_unfixed
                        dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int_max)
                        dyn_it = np.int64(dyn_x * one_minus_4eps)
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    next_focus = gc_focus[J_MIN]
                    for kk in range(J_MIN):
                        gc_a[kk] = 0
                        gc_dir[kk] = 1
                        gc_focus[kk] = kk
                    gc_focus[0] = next_focus
                    gc_focus[J_MIN] = J_MIN
                    for kk in range(J_MIN):
                        p = active_pos[kk]
                        cursor[p] = lo_arr[p]
                        child[2 * p] = lo_arr[p]
                        child[2 * p + 1] = parent_int[p] - lo_arr[p]
                    for kk in range(conv_len):
                        raw_conv[kk] = np.int32(0)
                    for ii in range(d_child):
                        ci = np.int32(child[ii])
                        if ci != 0:
                            raw_conv[2 * ii] += ci * ci
                            for jj2 in range(ii + 1, d_child):
                                cj2 = np.int32(child[jj2])
                                if cj2 != 0:
                                    raw_conv[ii + jj2] += np.int32(2) * ci * cj2
                    if qc_ell > 0:
                        qc_lo2 = qc_s - (d_child - 1)
                        if qc_lo2 < 0:
                            qc_lo2 = 0
                        qc_hi2 = qc_s + qc_ell - 2
                        if qc_hi2 > d_child - 1:
                            qc_hi2 = d_child - 1
                        qc_W_int = np.int64(0)
                        for ii in range(qc_lo2, qc_hi2 + 1):
                            qc_W_int += np.int64(child[ii])
                    continue

    return n_surv, n_subtree_pruned


# ===================================================================
# KERNEL B — WITH QUADRATIC RANGE SKIPPING
# ===================================================================
@njit(cache=False)
def kernel_quadskip(parent_int, n_half_child, m, c_target,
                    lo_arr, hi_arr, out_buf):
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    # --- Asymmetry filter ---
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0, 0, 0  # n_surv, n_subtree, n_qskip_checks, n_qskip_fired

    # --- Dynamic pruning constants ---
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    # --- Subtree pruning constants ---
    J_MIN = 7
    n_subtree_pruned = 0
    partial_conv = np.empty(conv_len, dtype=np.int32)

    # --- Quadratic skip stats ---
    n_qskip_checks = 0
    n_qskip_fired = 0

    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx] = 2.0 * np.float64(ell) * inv_4n

    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell)
        ell_used[ell - 2] = np.int32(1)
        oi += 1
    for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                d_child * 2, d_child + d_child // 2, d_child // 2):
        if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i
            radix[n_active] = r
            n_active += 1

    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    # --- Main loop ---
    while True:
        # === TEST current child ===
        pruned_by_window = False   # did any window prune this child?
        kill_ell = np.int32(0)
        kill_s = np.int32(0)

        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + two_ell_arr[ell_idx_qc] * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                pruned_by_window = True
                kill_ell = qc_ell
                kill_s = qc_s

        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            pruned = False
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                dyn_base_ell = dyn_base_ell_arr[ell_idx]
                two_ell_inv_4n = two_ell_arr[ell_idx]
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                    dyn_it = np.int64(dyn_x * one_minus_4eps)
                    if ws > dyn_it:
                        pruned = True
                        pruned_by_window = True
                        kill_ell = np.int32(ell)
                        kill_s = np.int32(s_lo)
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                use_rev = False
                for i in range(d_child):
                    jj = d_child - 1 - i
                    if child[jj] < child[i]:
                        use_rev = True
                        break
                    elif child[jj] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0

        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        at_boundary = (gc_a[j] == 0 or gc_a[j] == radix[j] - 1)
        if at_boundary:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # === INCREMENTAL UPDATE ===
        k1 = 2 * pos
        k2 = k1 + 1
        old1 = np.int32(child[k1])
        old2 = np.int32(child[k2])
        child[k1] = cursor[pos]
        child[k2] = parent_int[pos] - cursor[pos]
        new1 = np.int32(child[k1])
        new2 = np.int32(child[k2])
        delta1 = new1 - old1
        delta2 = new2 - old2

        raw_conv[2 * k1] += new1 * new1 - old1 * old1
        raw_conv[2 * k2] += new2 * new2 - old2 * old2
        raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
        for jj in range(k1):
            cj = np.int32(child[jj])
            if cj != 0:
                raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        for jj in range(k2 + 1, d_child):
            cj = np.int32(child[jj])
            if cj != 0:
                raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                raw_conv[k2 + jj] += np.int32(2) * delta2 * cj

        if qc_ell > 0:
            qc_lo = qc_s - (d_child - 1)
            if qc_lo < 0:
                qc_lo = 0
            qc_hi = qc_s + qc_ell - 2
            if qc_hi > d_child - 1:
                qc_hi = d_child - 1
            if qc_lo <= k1 and k1 <= qc_hi:
                qc_W_int += np.int64(delta1)
            if qc_lo <= k2 and k2 <= qc_hi:
                qc_W_int += np.int64(delta2)

        # =============================================================
        # === QUADRATIC RANGE SKIPPING (the proposed optimization) ===
        # =============================================================
        # After advancing position j and updating conv, if the PREVIOUS
        # child was pruned by window (kill_ell, kill_s), check whether
        # ALL remaining values in j's current sweep direction are also
        # pruned by that same window.
        #
        # S_w(x) = a*x^2 + b*x + c  is the window sum as function of
        # cursor[pos] = x, with child[k1]=x, child[k2]=parent[pos]-x.
        # T_w(x) = alpha + beta*x is the threshold (linear in x via W_int).
        # D(x) = S_w(x) - T_w(x); if D(x) > 0 for all remaining x, skip.
        # =============================================================
        if pruned_by_window and not at_boundary and j == 0:
            # Determine remaining sweep range for position j
            # Current value: gc_a[j] (just advanced)
            # Sweep direction: gc_dir[j] BEFORE boundary reversal
            # (boundary reversal already happened if at_boundary, but we
            #  excluded that case above)
            # Direction we're heading: gc_dir[j] (current direction)
            cur_a = gc_a[j]
            direction = gc_dir[j]
            # Remaining values in sweep: from cur_a toward boundary
            if direction > 0:
                sweep_hi = radix[j] - 1
                remaining = sweep_hi - cur_a
            else:
                sweep_lo = 0
                remaining = cur_a - sweep_lo

            if remaining >= 3:
                n_qskip_checks += 1

                # Map gc_a values to actual cursor values
                # cursor[pos] = lo_arr[pos] + gc_a[j]
                # Current x = lo_arr[pos] + cur_a (already set)
                # Range of x to check: [x_current, x_boundary]
                lo_p = lo_arr[pos]
                if direction > 0:
                    x_lo = lo_p + cur_a       # current (already tested above after advance)
                    x_hi = lo_p + radix[j] - 1  # boundary
                else:
                    x_lo = lo_p               # boundary
                    x_hi = lo_p + cur_a       # current

                # The window that killed the previous child
                w_ell = kill_ell
                w_s = kill_s
                w_ncv = w_ell - 1
                w_s_hi = w_s + w_ncv - 1   # last conv index in window

                # -- Compute quadratic coefficients of S_w(x) --
                # S_w = sum of raw_conv[k] for k in [w_s, w_s_hi]
                # raw_conv[k] depends on x through:
                #   1) Self/mutual terms at indices 2*k1, k1+k2, 2*k2
                #   2) Cross-terms at k1+j, k2+j for other bins j

                b_val = parent_int[pos]  # parent mass at this position

                # Quadratic coefficient: which of {2*k1, k1+k2, 2*k2} are in window?
                idx_self1 = 2 * k1       # = 4*pos
                idx_mutual = k1 + k2     # = 4*pos + 1
                idx_self2 = 2 * k2       # = 4*pos + 2

                coeff_a = np.float64(0)
                coeff_b_from_self = np.float64(0)
                # coeff_a and linear from self/mutual:
                # raw_conv[idx_self1] has x^2, no linear from self
                # raw_conv[idx_mutual] has -2x^2 + 2*b_val*x
                # raw_conv[idx_self2] has x^2 - 2*b_val*x + b_val^2

                b_f = np.float64(b_val)

                if w_s <= idx_self1 <= w_s_hi:
                    coeff_a += 1.0
                if w_s <= idx_mutual <= w_s_hi:
                    coeff_a += -2.0
                    coeff_b_from_self += 2.0 * b_f
                if w_s <= idx_self2 <= w_s_hi:
                    coeff_a += 1.0
                    coeff_b_from_self += -2.0 * b_f

                # Linear coefficient from cross-terms:
                # For each other bin jj (jj != k1, jj != k2):
                #   raw_conv[k1+jj] gets +2*x*child[jj]  (contrib to linear: +2*child[jj] if k1+jj in window)
                #   raw_conv[k2+jj] gets +2*(b-x)*child[jj] = -2*x*child[jj] + ... (contrib: -2*child[jj] if k2+jj in window)
                coeff_b_from_cross = np.float64(0)
                for jj in range(d_child):
                    if jj == k1 or jj == k2:
                        continue
                    cj_f = np.float64(child[jj])
                    if cj_f == 0.0:
                        continue
                    idx1 = k1 + jj
                    idx2 = k2 + jj
                    if w_s <= idx1 <= w_s_hi:
                        coeff_b_from_cross += 2.0 * cj_f
                    if w_s <= idx2 <= w_s_hi:
                        coeff_b_from_cross -= 2.0 * cj_f

                coeff_b_total = coeff_b_from_self + coeff_b_from_cross

                # Constant term c: S_w at x=0
                # We can compute c = S_w(current_x) - coeff_a*x^2 - coeff_b*x
                # But we need S_w at the current child state.
                # Current window sum:
                ws_current = np.int64(0)
                for k in range(w_s, w_s + w_ncv):
                    ws_current += np.int64(raw_conv[k])

                x_cur_f = np.float64(child[k1])  # = current cursor value
                coeff_c = np.float64(ws_current) - coeff_a * x_cur_f * x_cur_f - coeff_b_total * x_cur_f

                # -- Threshold T_w(x) = alpha + beta * x --
                # T_w = dyn_base_ell + two_ell_inv_4n * W_int(x)
                # W_int(x) depends on whether k1, k2 are in bin range [lo_bin, hi_bin]
                w_ell_idx = w_ell - 2
                dyn_base_ell_w = dyn_base_ell_arr[w_ell_idx]
                two_ell_inv_4n_w = two_ell_arr[w_ell_idx]

                lo_bin_w = w_s - (d_child - 1)
                if lo_bin_w < 0:
                    lo_bin_w = 0
                hi_bin_w = w_s + w_ell - 2
                if hi_bin_w > d_child - 1:
                    hi_bin_w = d_child - 1

                # W_int = W_base + gamma * x where gamma depends on which
                # of k1,k2 are in [lo_bin_w, hi_bin_w]
                # child[k1] = x contributes +x if k1 in range
                # child[k2] = b - x contributes +(b-x) if k2 in range
                gamma_w = np.float64(0)
                w_int_const = np.float64(0)
                k1_in_range = (lo_bin_w <= k1 and k1 <= hi_bin_w)
                k2_in_range = (lo_bin_w <= k2 and k2 <= hi_bin_w)
                if k1_in_range:
                    gamma_w += 1.0
                if k2_in_range:
                    gamma_w -= 1.0
                    w_int_const += b_f

                # W_int(x) = W_int_fixed_others + w_int_const + gamma_w * x
                # T_w(x) = dyn_base_ell_w + two_ell_inv_4n_w * (W_int_fixed_others + w_int_const + gamma_w * x)
                # But we know the CURRENT W_int value at current x:
                # We need to compute W_int at current x first.
                # Simplification: compute T at current x, derive slope.
                # T_w(x) = T_w(x_cur) + two_ell_inv_4n_w * gamma_w * (x - x_cur)

                # Compute current W_int for this window
                w_int_cur = np.int64(0)
                for ii in range(lo_bin_w, hi_bin_w + 1):
                    w_int_cur += np.int64(child[ii])

                T_cur = dyn_base_ell_w + two_ell_inv_4n_w * np.float64(w_int_cur)
                # Apply floor and rounding guard for the threshold
                T_cur_int = np.float64(np.int64(T_cur * one_minus_4eps))

                # T_w(x) as continuous: alpha_T + beta_T * x
                # beta_T = two_ell_inv_4n_w * gamma_w
                beta_T = two_ell_inv_4n_w * gamma_w

                # D(x) = S_w(x) - T_w_continuous(x)
                # D(x) = (coeff_a)*x^2 + (coeff_b_total - beta_T)*x + (coeff_c - T_alpha)
                # where T_alpha = T_cur - beta_T * x_cur (the intercept)
                T_alpha = T_cur - beta_T * x_cur_f

                Da = coeff_a
                Db = coeff_b_total - beta_T
                # Use the FLOORED threshold for conservative check
                # D_conservative(x) = S_w(x) - T_smooth(x)
                # If D_conservative > 0 for all x, then S_w > floor(T*(1-4eps)) too
                Dc = coeff_c - T_alpha

                # Check min D(x) > 0 over [x_lo, x_hi] (integers in remaining sweep)
                x_lo_f = np.float64(x_lo)
                x_hi_f = np.float64(x_hi)

                skip_all = False
                if Da > 1e-12:
                    # Convex: minimum at vertex
                    x_star = -Db / (2.0 * Da)
                    if x_star < x_lo_f:
                        x_star = x_lo_f
                    elif x_star > x_hi_f:
                        x_star = x_hi_f
                    D_min = Da * x_star * x_star + Db * x_star + Dc
                    if D_min > 0.5:  # guard: D must exceed 0 by margin for integer S_w
                        skip_all = True
                elif Da < -1e-12:
                    # Concave: minimum at endpoints
                    D_lo = Da * x_lo_f * x_lo_f + Db * x_lo_f + Dc
                    D_hi = Da * x_hi_f * x_hi_f + Db * x_hi_f + Dc
                    D_min = D_lo if D_lo < D_hi else D_hi
                    if D_min > 0.5:
                        skip_all = True
                else:
                    # Linear: check endpoints
                    D_lo = Db * x_lo_f + Dc
                    D_hi = Db * x_hi_f + Dc
                    D_min = D_lo if D_lo < D_hi else D_hi
                    if D_min > 0.5:
                        skip_all = True

                if skip_all:
                    n_qskip_fired += 1
                    # Skip to the boundary of position j's current sweep.
                    # Set gc_a[j] to boundary, reverse direction, fix focus.
                    if direction > 0:
                        target_a = radix[j] - 1
                    else:
                        target_a = 0

                    target_x = lo_p + target_a
                    # Update child and conv to the target position
                    old1b = np.int32(child[k1])
                    old2b = np.int32(child[k2])
                    child[k1] = np.int32(target_x)
                    child[k2] = np.int32(b_val - target_x)
                    new1b = np.int32(child[k1])
                    new2b = np.int32(child[k2])
                    delta1b = new1b - old1b
                    delta2b = new2b - old2b

                    if delta1b != 0 or delta2b != 0:
                        raw_conv[2 * k1] += new1b * new1b - old1b * old1b
                        raw_conv[2 * k2] += new2b * new2b - old2b * old2b
                        raw_conv[k1 + k2] += np.int32(2) * (new1b * new2b - old1b * old2b)
                        for jj in range(k1):
                            cj = np.int32(child[jj])
                            if cj != 0:
                                raw_conv[k1 + jj] += np.int32(2) * delta1b * cj
                                raw_conv[k2 + jj] += np.int32(2) * delta2b * cj
                        for jj in range(k2 + 1, d_child):
                            cj = np.int32(child[jj])
                            if cj != 0:
                                raw_conv[k1 + jj] += np.int32(2) * delta1b * cj
                                raw_conv[k2 + jj] += np.int32(2) * delta2b * cj

                    # Update cursor and gc state
                    gc_a[j] = target_a
                    cursor[pos] = np.int32(target_x)

                    # This position is now at boundary: trigger direction reversal
                    gc_dir[j] = -gc_dir[j]
                    gc_focus[j] = gc_focus[j + 1]
                    gc_focus[j + 1] = j + 1

                    # Update qc_W_int
                    if qc_ell > 0:
                        qc_lo3 = qc_s - (d_child - 1)
                        if qc_lo3 < 0:
                            qc_lo3 = 0
                        qc_hi3 = qc_s + qc_ell - 2
                        if qc_hi3 > d_child - 1:
                            qc_hi3 = d_child - 1
                        if qc_lo3 <= k1 and k1 <= qc_hi3:
                            qc_W_int += np.int64(delta1b)
                        if qc_lo3 <= k2 and k2 <= qc_hi3:
                            qc_W_int += np.int64(delta2b)

        # === SUBTREE PRUNING CHECK ===
        if j == J_MIN and n_active > J_MIN:
            fixed_parent_boundary = active_pos[J_MIN - 1]
            fixed_len = 2 * fixed_parent_boundary

            if fixed_len >= 4:
                partial_conv_len = 2 * fixed_len - 1
                for kk in range(partial_conv_len):
                    partial_conv[kk] = np.int32(0)
                for ii in range(fixed_len):
                    ci = np.int32(child[ii])
                    if ci != 0:
                        partial_conv[2 * ii] += ci * ci
                        for jj2 in range(ii + 1, fixed_len):
                            cj2 = np.int32(child[jj2])
                            if cj2 != 0:
                                partial_conv[ii + jj2] += np.int32(2) * ci * cj2
                for kk in range(1, partial_conv_len):
                    partial_conv[kk] += partial_conv[kk - 1]

                prefix_c[0] = 0
                for ii in range(fixed_len):
                    prefix_c[ii + 1] = prefix_c[ii] + np.int64(child[ii])

                first_unfixed_parent = fixed_parent_boundary
                subtree_pruned = False

                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    dyn_base_ell = dyn_base_ell_arr[ell_idx]
                    two_ell_inv_4n = two_ell_arr[ell_idx]

                    n_windows_partial = partial_conv_len - n_cv + 1
                    if n_windows_partial <= 0:
                        continue

                    for s_lo in range(n_windows_partial):
                        s_hi = s_lo + n_cv - 1
                        ws = np.int64(partial_conv[s_hi])
                        if s_lo > 0:
                            ws -= np.int64(partial_conv[s_lo - 1])

                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

                        fixed_hi = hi_bin
                        if fixed_hi > fixed_len - 1:
                            fixed_hi = fixed_len - 1
                        if fixed_hi >= lo_bin:
                            lo_clamp = lo_bin
                            if lo_clamp < 0:
                                lo_clamp = 0
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)

                        unfixed_lo_bin = lo_bin
                        if unfixed_lo_bin < fixed_len:
                            unfixed_lo_bin = fixed_len
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = unfixed_lo_bin // 2
                            p_hi = hi_bin // 2
                            if p_lo < first_unfixed_parent:
                                p_lo = first_unfixed_parent
                            if p_hi >= d_parent:
                                p_hi = d_parent - 1
                            if p_lo <= p_hi:
                                W_int_unfixed = parent_prefix[p_hi + 1] - parent_prefix[p_lo]
                            else:
                                W_int_unfixed = np.int64(0)
                        else:
                            W_int_unfixed = np.int64(0)

                        W_int_max = W_int_fixed + W_int_unfixed
                        dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int_max)
                        dyn_it = np.int64(dyn_x * one_minus_4eps)
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    next_focus = gc_focus[J_MIN]
                    for kk in range(J_MIN):
                        gc_a[kk] = 0
                        gc_dir[kk] = 1
                        gc_focus[kk] = kk
                    gc_focus[0] = next_focus
                    gc_focus[J_MIN] = J_MIN
                    for kk in range(J_MIN):
                        p = active_pos[kk]
                        cursor[p] = lo_arr[p]
                        child[2 * p] = lo_arr[p]
                        child[2 * p + 1] = parent_int[p] - lo_arr[p]
                    for kk in range(conv_len):
                        raw_conv[kk] = np.int32(0)
                    for ii in range(d_child):
                        ci = np.int32(child[ii])
                        if ci != 0:
                            raw_conv[2 * ii] += ci * ci
                            for jj2 in range(ii + 1, d_child):
                                cj2 = np.int32(child[jj2])
                                if cj2 != 0:
                                    raw_conv[ii + jj2] += np.int32(2) * ci * cj2
                    if qc_ell > 0:
                        qc_lo2 = qc_s - (d_child - 1)
                        if qc_lo2 < 0:
                            qc_lo2 = 0
                        qc_hi2 = qc_s + qc_ell - 2
                        if qc_hi2 > d_child - 1:
                            qc_hi2 = d_child - 1
                        qc_W_int = np.int64(0)
                        for ii in range(qc_lo2, qc_hi2 + 1):
                            qc_W_int += np.int64(child[ii])
                    continue

    return n_surv, n_subtree_pruned, n_qskip_checks, n_qskip_fired


# ===================================================================
# Benchmark harness
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="A/B benchmark: quadratic range skipping")
    parser.add_argument("--n_parents", type=int, default=200,
                        help="Number of L2 parents to benchmark (default 200)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="JIT warmup parents (default 3)")
    parser.add_argument("--level", type=int, default=2,
                        help="Checkpoint level to load parents from (default 2: L2->L3)")
    args = parser.parse_args()

    m = 20
    c_target = 1.3

    # Load checkpoint
    ckpt_path = os.path.join(_this_dir, "data",
                             f"checkpoint_L{args.level}_survivors.npy")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    parents = np.load(ckpt_path)
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_parent  # n_half for child = d_parent (since d_child = 2*d_parent = 2*n_half_child)

    print(f"Loaded {parents.shape[0]} parents from L{args.level} checkpoint")
    print(f"  d_parent={d_parent}, d_child={d_child}, m={m}, c_target={c_target}")
    print(f"  Testing {args.n_parents} parents + {args.warmup} warmup\n")

    # Select parents — pick a representative spread (not just first N)
    n_total = min(args.n_parents + args.warmup, parents.shape[0])
    # Use evenly spaced indices for diversity
    if n_total < parents.shape[0]:
        indices = np.linspace(0, parents.shape[0] - 1, n_total, dtype=int)
    else:
        indices = np.arange(n_total)

    # Precompute bin ranges
    ranges_list = []
    for idx in indices:
        parent = parents[idx].copy()
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        ranges_list.append((parent, result))

    buf_cap = 5_000_000 if d_child <= 32 else 100_000
    out_buf_a = np.empty((buf_cap, d_child), dtype=np.int32)
    out_buf_b = np.empty((buf_cap, d_child), dtype=np.int32)

    # === JIT warmup ===
    print("JIT warmup...")
    for wi in range(min(args.warmup, len(ranges_list))):
        parent, result = ranges_list[wi]
        if result is None:
            continue
        lo_arr, hi_arr, tc = result
        kernel_baseline(parent, n_half_child, m, c_target, lo_arr, hi_arr, out_buf_a)
        kernel_quadskip(parent, n_half_child, m, c_target, lo_arr, hi_arr, out_buf_b)
    print("Warmup done.\n")

    # === Timed runs ===
    test_parents = ranges_list[args.warmup:]
    n_test = len(test_parents)

    times_a = np.zeros(n_test)
    times_b = np.zeros(n_test)
    survs_a = np.zeros(n_test, dtype=np.int64)
    survs_b = np.zeros(n_test, dtype=np.int64)
    children_arr = np.zeros(n_test, dtype=np.int64)
    qskip_checks_arr = np.zeros(n_test, dtype=np.int64)
    qskip_fired_arr = np.zeros(n_test, dtype=np.int64)
    radix_info = []  # (n_active, max_radix, mean_radix)

    n_mismatch = 0
    n_skipped = 0

    for pi in range(n_test):
        parent, result = test_parents[pi]
        if result is None:
            n_skipped += 1
            continue
        lo_arr, hi_arr, total_children = result
        children_arr[pi] = total_children

        # Collect radix info
        n_act = 0
        max_r = 0
        sum_r = 0
        for ii in range(d_parent):
            r = hi_arr[ii] - lo_arr[ii] + 1
            if r > 1:
                n_act += 1
                if r > max_r:
                    max_r = r
                sum_r += r
        mean_r = sum_r / max(n_act, 1)
        radix_info.append((n_act, max_r, mean_r))

        # Run baseline
        t0 = time.perf_counter()
        sa, _ = kernel_baseline(parent, n_half_child, m, c_target,
                                lo_arr, hi_arr, out_buf_a)
        t1 = time.perf_counter()
        times_a[pi] = t1 - t0
        survs_a[pi] = sa

        # Run quadskip
        t0 = time.perf_counter()
        sb, _, qc, qf = kernel_quadskip(parent, n_half_child, m, c_target,
                                         lo_arr, hi_arr, out_buf_b)
        t1 = time.perf_counter()
        times_b[pi] = t1 - t0
        survs_b[pi] = sb
        qskip_checks_arr[pi] = qc
        qskip_fired_arr[pi] = qf

        # Correctness check
        if sa != sb:
            n_mismatch += 1
            print(f"  *** MISMATCH parent {pi}: baseline={sa}, quadskip={sb} ***")
        elif sa > 0:
            # Also verify identical survivors
            surv_a_sorted = np.sort(out_buf_a[:sa].copy(), axis=0)
            surv_b_sorted = np.sort(out_buf_b[:sb].copy(), axis=0)
            # Simple row-by-row comparison (both should be canonical)
            if not np.array_equal(out_buf_a[:sa], out_buf_b[:sb]):
                # Could be ordering difference — check sorted
                match = True
                for ri in range(sa):
                    for ci in range(d_child):
                        if surv_a_sorted[ri, ci] != surv_b_sorted[ri, ci]:
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    n_mismatch += 1
                    print(f"  *** CONTENT MISMATCH parent {pi}: same count {sa} but different survivors ***")

    # === Report ===
    active_mask = children_arr > 0
    n_active_parents = int(np.sum(active_mask))

    print("=" * 72)
    print("BENCHMARK RESULTS: Baseline vs Quadratic Range Skipping")
    print("=" * 72)
    print(f"Parents tested:  {n_test} ({n_skipped} skipped, {n_active_parents} active)")
    print(f"Correctness:     {'ALL MATCH' if n_mismatch == 0 else f'{n_mismatch} MISMATCHES'}")
    print()

    if n_active_parents == 0:
        print("No active parents — nothing to benchmark.")
        return

    ta = times_a[active_mask]
    tb = times_b[active_mask]

    total_a = np.sum(ta)
    total_b = np.sum(tb)
    speedup = total_a / total_b if total_b > 0 else float('inf')

    print(f"Total wall time baseline:   {total_a:.4f}s")
    print(f"Total wall time quadskip:   {total_b:.4f}s")
    print(f"Aggregate speedup:          {speedup:.4f}x")
    print()

    # Per-parent stats
    speedups = ta / np.maximum(tb, 1e-12)
    print(f"Per-parent speedup:  median={np.median(speedups):.3f}x  "
          f"mean={np.mean(speedups):.3f}x  "
          f"p10={np.percentile(speedups,10):.3f}x  "
          f"p90={np.percentile(speedups,90):.3f}x")
    print()

    # Quadratic skip statistics
    total_qc = int(np.sum(qskip_checks_arr))
    total_qf = int(np.sum(qskip_fired_arr))
    total_children = int(np.sum(children_arr[active_mask]))
    fire_rate = total_qf / max(total_qc, 1)
    print(f"Total children enumerated:  {total_children:,}")
    print(f"Quadratic skip checks:      {total_qc:,}")
    print(f"Quadratic skips fired:      {total_qf:,}")
    print(f"Fire rate (fired/checks):   {fire_rate:.2%}")
    if total_children > 0:
        print(f"Checks per child:           {total_qc / total_children:.4f}")
        print(f"Skips per child:            {total_qf / total_children:.6f}")
    print()

    # Radix distribution
    if radix_info:
        max_radixes = [r[1] for r in radix_info]
        mean_radixes = [r[2] for r in radix_info]
        n_actives = [r[0] for r in radix_info]
        print(f"Active positions per parent: mean={np.mean(n_actives):.1f}  "
              f"min={min(n_actives)}  max={max(n_actives)}")
        print(f"Max radix per parent:        mean={np.mean(max_radixes):.1f}  "
              f"min={min(max_radixes)}  max={max(max_radixes)}")
        print(f"Mean radix per parent:       mean={np.mean(mean_radixes):.2f}")
    print()

    # Breakdown by parent size
    print("--- Breakdown by total_children ---")
    children_active = children_arr[active_mask]
    for lo_c, hi_c, label in [(0, 100, "<100"), (100, 10000, "100-10K"),
                               (10000, 1000000, "10K-1M"),
                               (1000000, 10**12, ">1M")]:
        mask = (children_active >= lo_c) & (children_active < hi_c)
        if np.sum(mask) == 0:
            continue
        ta_sub = ta[mask]
        tb_sub = tb[mask]
        sp = np.sum(ta_sub) / max(np.sum(tb_sub), 1e-12)
        print(f"  {label:>8s}: {int(np.sum(mask)):>4d} parents, "
              f"baseline={np.sum(ta_sub):.4f}s, quadskip={np.sum(tb_sub):.4f}s, "
              f"speedup={sp:.3f}x")
    print()

    # Verdict
    print("=" * 72)
    if n_mismatch > 0:
        print("VERDICT: CORRECTNESS FAILURE — optimization produces wrong results.")
    elif speedup > 1.05:
        print(f"VERDICT: Optimization provides {speedup:.2f}x speedup.")
    elif speedup > 0.98:
        print(f"VERDICT: No significant speedup ({speedup:.3f}x). Overhead ~= savings.")
    else:
        print(f"VERDICT: Optimization is a SLOWDOWN ({speedup:.3f}x). Overhead exceeds savings.")
    print("=" * 72)


if __name__ == "__main__":
    main()
