"""Profile the Gray code kernel to find REAL computational bottlenecks.

Creates an instrumented version of _fused_generate_and_prune_gray that counts:
- Total children visited
- Quick-check kills
- Full window scan entries
- Ell values tried before pruning (average)
- Windows scanned per ell (average)
- Subtree prunes
- Survivors
- Cross-term update ops vs quick-check vs full scan vs canonicalization

Uses 10 sampled L2 parents (d_child=32) with M=20, C_TARGET=1.4.
"""

import sys
import os
import math
import time

import numpy as np
import numba
from numba import njit

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from pruning import correction

# Import the real kernel for warm-up and timing comparison
from cpu.run_cascade import (
    _fused_generate_and_prune_gray,
    _compute_bin_ranges,
)


# =====================================================================
# Instrumented kernel: returns detailed counters
# =====================================================================
@njit(cache=False)
def _instrumented_gray_kernel(parent_int, n_half_child, m, c_target,
                               lo_arr, hi_arr, out_buf):
    """Instrumented version: same logic, but returns 20-element counter array.

    counters[0]  = total children visited
    counters[1]  = killed by quick-check
    counters[2]  = entered full window scan
    counters[3]  = total ell values tried across all full scans
    counters[4]  = total windows scanned across all full scans (all ells)
    counters[5]  = subtree prunes fired
    counters[6]  = survivors (not pruned)
    counters[7]  = total cross-term update iterations (sparse nz_count per step)
    counters[8]  = total quick-check conv sum iterations (qc_ell-1 per qc)
    counters[9]  = total prefix_c build iterations (d_child per full scan entry)
    counters[10] = total canonicalization comparisons
    counters[11] = total initial conv build ops (for first child)
    counters[12] = total self+mutual term updates (3 per step)
    counters[13] = total sparse nz_list maintenance ops
    counters[14] = total full scan window sum init iterations
    counters[15] = total subtree recompute ops (O(d^2) per subtree prune)
    counters[16] = killed by full scan (not quick-check)
    counters[17] = (unused)
    counters[18] = total gray code advance ops
    counters[19] = total qc_W_int update ops
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200

    counters = np.zeros(20, dtype=np.int64)

    # --- Ell kill histogram (ell 2..2*d_child) ---
    ell_kill_hist = np.zeros(2 * d_child + 1, dtype=np.int64)

    # --- Asymmetry filter ---
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, counters, ell_kill_hist

    # --- Dynamic pruning constants ---
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d

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

    # --- Sparse cross-term ---
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    # --- Build initial child ---
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    # --- Precompute per-ell constants ---
    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = c_target * m_d * m_d * np.float64(ell) * inv_4n

    # --- Optimized ell scan order ---
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                    d_child * 2, d_child + d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = np.int32(1)
                oi += 1
    else:
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
    init_ops = np.int64(0)
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            init_ops += 1
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj
                    init_ops += 1
    counters[11] = init_ops

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
        counters[0] += 1  # total children visited

        # === TEST current child ===
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            counters[8] += n_cv_qc  # quick-check conv sum iterations
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + 1.0 + eps_margin + 2.0 * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                counters[1] += 1  # killed by quick-check

        if not quick_killed:
            counters[2] += 1  # entered full window scan

            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])
            counters[9] += d_child  # prefix_c build iterations

            pruned = False
            ells_tried = 0
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                dyn_base_ell = dyn_base_ell_arr[ell_idx]
                n_windows = conv_len - n_cv + 1
                ells_tried += 1

                # Sliding window: initialize sum for s_lo=0
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                counters[14] += n_cv  # window sum init

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
                    dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
                    dyn_it = np.int64(dyn_x * one_minus_4eps)
                    counters[4] += 1  # windows scanned
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        ell_kill_hist[ell] += 1
                        counters[16] += 1  # killed by full scan
                        break

            counters[3] += ells_tried

            if not pruned:
                # Canonicalization
                use_rev = False
                half_d = d_child // 2
                canon_comps = 0
                for i in range(half_d):
                    j = d_child - 1 - i
                    canon_comps += 1
                    if child[j] < child[i]:
                        use_rev = True
                        break
                    elif child[j] > child[i]:
                        break
                counters[10] += canon_comps

                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1
                counters[6] += 1  # survivors

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0
        counters[18] += 1  # gray code advance ops

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

        # Self-terms + mutual
        raw_conv[2 * k1] += new1 * new1 - old1 * old1
        raw_conv[2 * k2] += new2 * new2 - old2 * old2
        raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
        counters[12] += 3  # self+mutual ops

        # Cross-terms
        if use_sparse:
            # nz_list maintenance
            maint_ops = 0
            if old1 != 0 and new1 == 0:
                p = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k1] = -1
                maint_ops += 1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
                maint_ops += 1
            if old2 != 0 and new2 == 0:
                p = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k2] = -1
                maint_ops += 1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
                maint_ops += 1
            counters[13] += maint_ops

            cross_ops = 0
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
                    cross_ops += 1
            counters[7] += cross_ops
        else:
            cross_ops = 0
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
                    cross_ops += 1
            for jj in range(k2 + 1, d_child):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
                    cross_ops += 1
            counters[7] += cross_ops

        # qc_W_int update
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
            counters[19] += 1

        # === SUBTREE PRUNING CHECK ===
        if j == J_MIN and n_active > J_MIN:
            fixed_parent_boundary = active_pos[J_MIN - 1]
            fixed_len = 2 * fixed_parent_boundary

            if fixed_len >= 4:
                partial_conv_len = 2 * fixed_len - 1
                for kk in range(partial_conv_len):
                    partial_conv[kk] = np.int32(0)
                st_ops = 0
                for ii in range(fixed_len):
                    ci = np.int32(child[ii])
                    if ci != 0:
                        partial_conv[2 * ii] += ci * ci
                        st_ops += 1
                        for jj2 in range(ii + 1, fixed_len):
                            cj2 = np.int32(child[jj2])
                            if cj2 != 0:
                                partial_conv[ii + jj2] += np.int32(2) * ci * cj2
                                st_ops += 1
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
                        dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int_max)
                        dyn_it = np.int64(dyn_x * one_minus_4eps)
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    counters[5] += 1
                    counters[15] += st_ops  # subtree recompute ops

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
                    recomp_ops = 0
                    for ii in range(d_child):
                        ci = np.int32(child[ii])
                        if ci != 0:
                            raw_conv[2 * ii] += ci * ci
                            recomp_ops += 1
                            for jj2 in range(ii + 1, d_child):
                                cj2 = np.int32(child[jj2])
                                if cj2 != 0:
                                    raw_conv[ii + jj2] += np.int32(2) * ci * cj2
                                    recomp_ops += 1
                    counters[15] += recomp_ops

                    if use_sparse:
                        nz_count = 0
                        for ii in range(d_child):
                            if child[ii] != 0:
                                nz_list[nz_count] = ii
                                nz_pos[ii] = nz_count
                                nz_count += 1
                            else:
                                nz_pos[ii] = -1

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

    return n_surv, counters, ell_kill_hist


# =====================================================================
# Timing-only kernel: isolate cost of cross-term updates
# =====================================================================

@njit(cache=False)
def _time_crossterm_only(parent_int, n_half_child, m, c_target,
                          lo_arr, hi_arr):
    """Run Gray code traversal doing ONLY cross-term updates, no pruning.
    Returns number of children visited."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)

    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]
    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j_inner in range(i + 1, d_child):
                cj = np.int32(child[j_inner])
                if cj != 0:
                    raw_conv[i + j_inner] += np.int32(2) * ci * cj

    # Gray code setup
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

    n_visited = np.int64(0)
    while True:
        n_visited += 1
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

        if use_sparse:
            if old1 != 0 and new1 == 0:
                p = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k1] = -1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
            if old2 != 0 and new2 == 0:
                p = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        else:
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

    return n_visited


# =====================================================================
# Main profiling harness
# =====================================================================
def main():
    M = 20
    C_TARGET = 1.4

    # Load BOTH diverse and high-prune parents for complete picture
    diverse_path = os.path.join(_root, 'data', 'test_fix', 'profile_parents.npy')
    high_prune_path = os.path.join(_root, 'data', 'test_fix', 'high_prune_parents.npy')
    parts = []
    if os.path.exists(diverse_path):
        parts.append(np.load(diverse_path))
    if os.path.exists(high_prune_path):
        parts.append(np.load(high_prune_path))
    if not parts:
        raise FileNotFoundError("No parent files found")
    parents = np.concatenate(parts, axis=0)

    d_parent = parents.shape[1]  # 16
    d_child = 2 * d_parent       # 32
    n_half_child = d_child // 2  # 16

    print(f"Profiling Gray code kernel: d_parent={d_parent}, d_child={d_child}, m={M}, c_target={C_TARGET}")
    print(f"Number of parents: {parents.shape[0]}")
    print()

    # Compute bin ranges for each parent
    parent_data = []
    for i in range(parents.shape[0]):
        p = parents[i].astype(np.int32)
        result = _compute_bin_ranges(p, M, C_TARGET, d_child, n_half_child)
        if result is not None:
            lo, hi, total = result
            parent_data.append((p, lo, hi, total))
            print(f"  Parent {i}: total_children={total:,}, active_bins={np.sum(hi - lo > 0)}")
        else:
            print(f"  Parent {i}: SKIPPED (empty range)")
    print()

    # --- Phase 1: JIT warm-up ---
    print("Warming up JIT (this takes ~30s for 3 kernels)...")
    sys.stdout.flush()
    p0, lo0, hi0, _ = parent_data[0]
    buf = np.empty((100_000, d_child), dtype=np.int32)

    t_warmup = time.perf_counter()
    # Warm up instrumented kernel
    _instrumented_gray_kernel(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)
    print(f"  Instrumented kernel warmed up ({time.perf_counter()-t_warmup:.1f}s)")
    sys.stdout.flush()
    t2 = time.perf_counter()
    # Warm up crossterm-only kernel
    _time_crossterm_only(p0, n_half_child, M, C_TARGET, lo0, hi0)
    print(f"  Crossterm-only kernel warmed up ({time.perf_counter()-t2:.1f}s)")
    sys.stdout.flush()
    t3 = time.perf_counter()
    # Warm up real kernel
    _fused_generate_and_prune_gray(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)
    print(f"  Real kernel warmed up ({time.perf_counter()-t3:.1f}s)")
    sys.stdout.flush()
    print(f"JIT warm-up complete ({time.perf_counter()-t_warmup:.1f}s total).")
    print()

    # --- Phase 2: Run instrumented kernel on all parents ---
    print("=" * 80)
    print("PHASE 2: Instrumented kernel - operation counts")
    print("=" * 80)

    total_counters = np.zeros(20, dtype=np.int64)
    total_ell_hist = np.zeros(2 * d_child + 1, dtype=np.int64)

    for idx, (p, lo, hi, total_ch) in enumerate(parent_data):
        buf_cap = min(total_ch, 5_000_000)
        buf = np.empty((max(buf_cap, 1), d_child), dtype=np.int32)

        t0 = time.perf_counter()
        n_surv, counters, ell_hist = _instrumented_gray_kernel(
            p, n_half_child, M, C_TARGET, lo, hi, buf)
        elapsed = time.perf_counter() - t0

        total_counters += counters
        total_ell_hist += ell_hist

        visited = counters[0]
        qc_kills = counters[1]
        full_scans = counters[2]
        survivors = counters[6]
        qc_rate = qc_kills / max(visited - 1, 1) * 100

        print(f"  Parent {idx}: visited={visited:>10,}  qc_kills={qc_kills:>10,} ({qc_rate:.1f}%)  "
              f"full_scans={full_scans:>8,}  survivors={survivors:>6,}  "
              f"subtree={counters[5]:>4}  time={elapsed:.2f}s")
        sys.stdout.flush()

    print()
    print("-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)
    T = total_counters
    visited = T[0]
    qc_kills = T[1]
    full_scans = T[2]
    ells_tried = T[3]
    windows_scanned = T[4]
    subtree_prunes = T[5]
    survivors = T[6]
    cross_ops = T[7]
    qc_conv_iters = T[8]
    prefix_build_iters = T[9]
    canon_comps = T[10]
    self_mutual_ops = T[12]
    sparse_maint = T[13]
    window_init_iters = T[14]
    subtree_recomp = T[15]
    full_scan_kills = T[16]
    gc_advances = T[18]
    qc_wint_updates = T[19]

    print(f"Total children visited:     {visited:>15,}")
    print(f"Killed by quick-check:      {qc_kills:>15,}  ({qc_kills/visited*100:.2f}%)")
    print(f"Entered full window scan:   {full_scans:>15,}  ({full_scans/visited*100:.2f}%)")
    print(f"  Killed by full scan:      {full_scan_kills:>15,}  ({full_scan_kills/visited*100:.2f}%)")
    print(f"Survivors:                  {survivors:>15,}  ({survivors/visited*100:.4f}%)")
    print(f"Subtree prunes:             {subtree_prunes:>15,}")
    print()
    print(f"Avg ells tried per full scan:     {ells_tried/max(full_scans,1):.2f}")
    print(f"Avg windows per full scan:        {windows_scanned/max(full_scans,1):.1f}")
    print(f"Avg windows per ell tried:        {windows_scanned/max(ells_tried,1):.1f}")
    print()

    # Ell kill distribution
    print("Ell kill distribution (which ell values kill children in full scan):")
    for ell in range(2, 2 * d_child + 1):
        if total_ell_hist[ell] > 0:
            pct = total_ell_hist[ell] / max(full_scan_kills, 1) * 100
            print(f"  ell={ell:>3}: {total_ell_hist[ell]:>10,} kills ({pct:>5.1f}%)")

    print()
    print("-" * 80)
    print("OPERATION COST MODEL (weighted by approximate cycle cost)")
    print("-" * 80)

    # Each "op" ~ 1 int32 multiply-add or equivalent memory access
    # Cross-term: 2 multiply-adds per nz bin (delta1*cj, delta2*cj, + 2 store)
    cross_cost = cross_ops * 4  # 2 mul + 2 add-store
    # Self+mutual: 3 mul-add-store
    self_cost = self_mutual_ops * 2  # cheaper: no loop, register ops
    # Quick-check: ~2 ops per conv sum iteration + 1 threshold comparison
    qc_cost = qc_conv_iters * 2 + qc_kills
    # Full scan: window W_int lookup + threshold comparison per window
    full_scan_cost = windows_scanned * 4 + window_init_iters * 2 + prefix_build_iters * 2
    # Canonicalization: 2 loads + compare per step
    canon_cost = canon_comps * 3 + survivors * d_child  # copy cost
    # Subtree recompute
    subtree_cost = subtree_recomp * 3  # mul+add+store per nonzero pair
    # Gray code advance: ~8 ops (focus lookup, digit update, cursor write, boundary check)
    gc_cost = gc_advances * 8
    # Sparse nz_list maintenance
    sparse_cost = sparse_maint * 5
    # qc_W_int update: ~4 ops (bounds check + conditional add)
    qc_wint_cost = qc_wint_updates * 4

    total_cost = (cross_cost + self_cost + qc_cost + full_scan_cost +
                  canon_cost + subtree_cost + gc_cost + sparse_cost + qc_wint_cost)

    print(f"Cross-term updates:    {cross_cost:>15,} ops  ({cross_cost/total_cost*100:>5.1f}%)")
    print(f"Self+mutual terms:     {self_cost:>15,} ops  ({self_cost/total_cost*100:>5.1f}%)")
    print(f"Quick-check eval:      {qc_cost:>15,} ops  ({qc_cost/total_cost*100:>5.1f}%)")
    print(f"Full window scan:      {full_scan_cost:>15,} ops  ({full_scan_cost/total_cost*100:>5.1f}%)")
    print(f"Canonicalization:      {canon_cost:>15,} ops  ({canon_cost/total_cost*100:>5.1f}%)")
    print(f"Subtree recompute:     {subtree_cost:>15,} ops  ({subtree_cost/total_cost*100:>5.1f}%)")
    print(f"Gray code advance:     {gc_cost:>15,} ops  ({gc_cost/total_cost*100:>5.1f}%)")
    print(f"Sparse nz maintenance: {sparse_cost:>15,} ops  ({sparse_cost/total_cost*100:>5.1f}%)")
    print(f"QC W_int update:       {qc_wint_cost:>15,} ops  ({qc_wint_cost/total_cost*100:>5.1f}%)")
    print(f"TOTAL estimated ops:   {total_cost:>15,}")

    # Per-child averages
    print()
    print("-" * 80)
    print("PER-CHILD AVERAGES")
    print("-" * 80)
    print(f"Avg cross-term nz iters per step:  {cross_ops/max(gc_advances,1):.1f}  (of d_child={d_child})")
    print(f"Avg nz_count (approx):             {cross_ops/max(gc_advances,1) + 2:.1f}")
    print(f"Avg self+mutual per step:          {self_mutual_ops/max(gc_advances,1):.1f}  (always 3)")
    print(f"Avg qc conv sum iters per child:   {qc_conv_iters/visited:.1f}")
    print(f"Avg full scan windows per child:   {(windows_scanned)/visited:.1f}")
    print(f"Avg full scan iters per child:     {(windows_scanned+window_init_iters)/visited:.1f}")
    print(f"Avg prefix builds per child:       {prefix_build_iters/visited:.1f}")
    qc_attempted = visited - 1  # first child can't have qc
    print(f"Quick-check hit rate:              {qc_kills/max(qc_attempted,1)*100:.2f}% of attempts")

    # --- Phase 3: Wall-clock timing comparison ---
    print()
    print("=" * 80)
    print("PHASE 3: Wall-clock timing - cross-term fraction")
    print("=" * 80)

    # Pick 3 parents with most children for timing
    parent_data_sorted = sorted(parent_data, key=lambda x: x[3], reverse=True)
    timing_parents = parent_data_sorted[:3]

    for tidx, (p, lo, hi, total_ch) in enumerate(timing_parents):
        buf_cap = min(total_ch, 5_000_000)
        buf = np.empty((max(buf_cap, 1), d_child), dtype=np.int32)

        # Time real kernel (best of 3)
        times_real = []
        for _ in range(3):
            t0 = time.perf_counter()
            n_s, _ = _fused_generate_and_prune_gray(p, n_half_child, M, C_TARGET, lo, hi, buf)
            t1 = time.perf_counter()
            times_real.append(t1 - t0)
        real_time = min(times_real)

        # Time crossterm-only traversal (best of 3)
        times_cross = []
        for _ in range(3):
            t0 = time.perf_counter()
            n_v = _time_crossterm_only(p, n_half_child, M, C_TARGET, lo, hi)
            t1 = time.perf_counter()
            times_cross.append(t1 - t0)
        cross_time = min(times_cross)

        rate = total_ch / real_time if real_time > 0 else 0
        print(f"  Parent (total_ch={total_ch:>10,}):")
        print(f"    Full kernel:     {real_time*1000:>8.1f} ms  ({n_s:,} survivors)  [{rate/1e6:.1f}M children/s]")
        print(f"    Crossterm-only:  {cross_time*1000:>8.1f} ms  ({n_v:,} visited, no pruning)")
        print(f"    Cross fraction:  {cross_time/real_time*100:>8.1f}%  of total kernel time")
        print(f"    Pruning+other:   {(real_time-cross_time)*1000:>8.1f} ms  ({(1-cross_time/real_time)*100:.1f}%)")
        print()

    print("=" * 80)
    print("SUMMARY: What to optimize")
    print("=" * 80)
    print()
    total_killed = qc_kills + full_scan_kills
    print(f"Quick-check is responsible for {qc_kills/max(total_killed,1)*100:.1f}% of all kills")
    print(f"Full scan is responsible for {full_scan_kills/max(total_killed,1)*100:.1f}% of all kills")
    print(f"Only {full_scans/visited*100:.2f}% of children require a full window scan")
    print(f"Only {survivors/visited*100:.4f}% of children survive (all pruning)")
    print()
    print("Cross-term update cost model fraction: "
          f"{cross_cost/total_cost*100:.1f}%")
    print("Full window scan cost model fraction:  "
          f"{full_scan_cost/total_cost*100:.1f}%")
    print("Quick-check cost model fraction:       "
          f"{qc_cost/total_cost*100:.1f}%")


if __name__ == '__main__':
    main()
