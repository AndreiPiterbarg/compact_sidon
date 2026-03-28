#!/usr/bin/env python
"""Profile the Gray code kernel at L3 (d_parent=16 -> d_child=32) where pruning
actually works. Since L2 checkpoint is deleted, we generate a few L2-like parents
by taking L3 survivors and collapsing pairs of bins.

Also: analyze what makes L4 fundamentally unprunable.
"""

import os
import sys
import math
import time
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


@njit(cache=False)
def _instrumented_gray_kernel_with_depth(parent_int, n_half_child, m, c_target,
                                          lo_arr, hi_arr):
    """Instrumented kernel tracking window scan depth distribution.

    Returns (stats[12], depth_hist[64])
    stats: same as before
    depth_hist: histogram of scan depth values for full-scan prunes
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    stats = np.zeros(12, dtype=np.int64)
    depth_hist = np.zeros(64, dtype=np.int64)  # depth 0..62 + overflow bucket 63

    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return stats, depth_hist

    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    conv_len = 2 * d_child - 1

    J_MIN = 7
    partial_conv = np.empty(conv_len, dtype=np.int32)

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

    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i
                nz_pos[i] = nz_count
                nz_count += 1

    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = c_target * m_d * m_d * np.float64(ell) * inv_4n

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

    stats[11] = n_active

    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    # Track which ell kills
    ell_kill_hist = np.zeros(ell_count + 1, dtype=np.int64)

    while True:
        stats[0] += 1

        if use_sparse:
            stats[9] += nz_count
            stats[10] += 1

        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + 1.0 + eps_margin + 2.0 * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                stats[1] += 1

        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            pruned = False
            scan_depth = 0
            killing_ell = 0
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                dyn_base_ell = dyn_base_ell_arr[ell_idx]
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                scan_depth += 1
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
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        killing_ell = ell
                        break

            if pruned:
                stats[2] += 1
                stats[7] += scan_depth
                if scan_depth > stats[8]:
                    stats[8] = scan_depth
                bucket = min(scan_depth - 1, 62)
                depth_hist[bucket] += 1
                # Track killing ell
                if killing_ell >= 2 and killing_ell - 2 < ell_count:
                    ell_kill_hist[killing_ell - 2] += 1
            else:
                stats[3] += 1
                depth_hist[63] += 1  # survived

        # Gray code advance
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

        # Subtree pruning
        if j == J_MIN and n_active > J_MIN:
            stats[4] += 1
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
                            lo_clamp = lo_bin if lo_bin >= 0 else 0
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
                    stats[5] += 1
                    inner_product = np.int64(1)
                    for kk in range(J_MIN):
                        inner_product *= np.int64(radix[kk])
                    stats[6] += inner_product
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

    return stats, depth_hist


def main():
    print("=" * 80)
    print("L3 PRUNING PROFILE (d_parent=16 -> d_child=32)")
    print("=" * 80)
    print()

    # Reconstruct L2-like parents from L3 survivors by collapsing adjacent pairs
    l3_path = os.path.join(DATA_DIR, "checkpoint_L3_survivors.npy")
    l3_data = np.load(l3_path, mmap_mode='r')

    rng = np.random.RandomState(123)
    sample_size = 15
    indices = rng.choice(l3_data.shape[0], size=sample_size * 10, replace=False)

    # Collapse d=32 -> d=16 by summing adjacent pairs
    l2_parents = []
    for idx in indices:
        row = np.array(l3_data[idx], dtype=np.int32)
        parent = np.empty(16, dtype=np.int32)
        for i in range(16):
            parent[i] = row[2 * i] + row[2 * i + 1]
        # Check this is a valid L2 parent (unique)
        key = tuple(parent.tolist())
        l2_parents.append(parent)
        if len(l2_parents) >= sample_size:
            break

    del l3_data
    print(f"Reconstructed {len(l2_parents)} L2-like parents (d=16)")
    print()

    # Warmup
    print("JIT warmup...")
    dummy = l2_parents[0].copy()
    d_p = 16
    d_c = 32
    nhc = 16
    result = compute_bin_ranges(dummy, M, C_TARGET, d_c, nhc)
    if result is not None:
        lo, hi, tc = result
        _, _ = _instrumented_gray_kernel_with_depth(dummy, nhc, M, C_TARGET, lo, hi)
    print("  Done.")
    print()

    # Run instrumented kernel on each L2 parent
    all_stats = []
    all_depth_hists = []
    all_times = []
    all_tc = []

    for idx, parent in enumerate(l2_parents):
        d_p = len(parent)
        d_c = 2 * d_p
        nhc = d_c // 2

        result = compute_bin_ranges(parent, M, C_TARGET, d_c, nhc)
        if result is None:
            print(f"  Parent {idx}: EMPTY range")
            all_stats.append(None)
            all_depth_hists.append(None)
            all_times.append(0.0)
            all_tc.append(0)
            continue

        lo_arr, hi_arr, total_children = result
        all_tc.append(total_children)

        ranges = hi_arr - lo_arr + 1
        n_active = int(np.sum(ranges > 1))
        nz_bins = int(np.count_nonzero(parent))

        t0 = time.perf_counter()
        stats, depth_hist = _instrumented_gray_kernel_with_depth(
            parent, nhc, M, C_TARGET, lo_arr, hi_arr)
        t1 = time.perf_counter()
        elapsed = t1 - t0

        all_stats.append(stats)
        all_depth_hists.append(depth_hist)
        all_times.append(elapsed)

        visited = stats[0]
        qc = stats[1]
        full = stats[2]
        surv = stats[3]
        sub_checks = stats[4]
        sub_success = stats[5]
        sub_skipped = stats[6]

        print(f"  Parent {idx}: nz={nz_bins}/{d_p}, n_active={n_active}, "
              f"children={total_children:,}")
        print(f"    Visited: {visited:,} | QC: {qc:,} ({100*qc/max(visited,1):.1f}%) | "
              f"FullScan: {full:,} ({100*full/max(visited,1):.1f}%) | "
              f"Surv: {surv:,} ({100*surv/max(visited,1):.4f}%)")
        print(f"    Subtree: {sub_checks:,} checks, {sub_success:,} hits "
              f"({100*sub_success/max(sub_checks,1):.1f}%), "
              f"skipped {sub_skipped:,} ({100*sub_skipped/max(total_children,1):.1f}%)")
        print(f"    Time: {elapsed:.3f}s ({elapsed/max(visited,1)*1e6:.2f} us/child)")
        print()

    # Aggregates
    valid = [(s, dh, t, tc) for s, dh, t, tc in
             zip(all_stats, all_depth_hists, all_times, all_tc) if s is not None]

    if not valid:
        print("No valid parents!")
        return

    print("=" * 80)
    print("AGGREGATE (L3: d_parent=16 -> d_child=32)")
    print("=" * 80)
    print()

    s_arr = np.array([v[0] for v in valid], dtype=np.int64)
    t_arr = np.array([v[2] for v in valid])
    tc_arr = np.array([v[3] for v in valid])

    total_visited = np.sum(s_arr[:, 0])
    total_qc = np.sum(s_arr[:, 1])
    total_full = np.sum(s_arr[:, 2])
    total_surv = np.sum(s_arr[:, 3])
    total_cart = np.sum(tc_arr)
    total_sub_checks = np.sum(s_arr[:, 4])
    total_sub_success = np.sum(s_arr[:, 5])
    total_sub_skipped = np.sum(s_arr[:, 6])

    print(f"  Cartesian product total:   {total_cart:>14,}")
    print(f"  Children visited:          {total_visited:>14,}")
    print(f"    - Quick-check killed:    {total_qc:>14,} ({100*total_qc/max(total_visited,1):.1f}%)")
    print(f"    - Full-scan pruned:      {total_full:>14,} ({100*total_full/max(total_visited,1):.1f}%)")
    print(f"    - Survived:              {total_surv:>14,} ({100*total_surv/max(total_visited,1):.4f}%)")
    print(f"  Subtree checks:            {total_sub_checks:>14,}")
    print(f"  Subtree successes:         {total_sub_success:>14,} ({100*total_sub_success/max(total_sub_checks,1):.1f}%)")
    print(f"  Subtree children skipped:  {total_sub_skipped:>14,} ({100*total_sub_skipped/max(total_cart,1):.1f}%)")
    print(f"  Total time:                {np.sum(t_arr):>14.2f}s")
    print()

    # Window scan depth distribution (aggregated)
    agg_depth = np.zeros(64, dtype=np.int64)
    for v in valid:
        if v[1] is not None:
            agg_depth += v[1]

    print("  Window scan depth distribution (full-scan prunes):")
    print(f"    {'Depth':>6s} {'Count':>12s} {'Fraction':>10s}")
    for d in range(min(20, 63)):
        if agg_depth[d] > 0:
            frac = agg_depth[d] / max(total_full, 1)
            print(f"    {d+1:>6d} {agg_depth[d]:>12,} {100*frac:>9.1f}%")
    if agg_depth[63] > 0:
        print(f"    {'surv':>6s} {agg_depth[63]:>12,}")
    print()

    # Parent heterogeneity
    print("  Parent heterogeneity:")
    print(f"    {'Metric':<30s} {'Min':>12s} {'Median':>12s} {'Max':>12s} {'Ratio':>8s}")
    tc_f = tc_arr.astype(float)
    vis_f = s_arr[:, 0].astype(float)
    print(f"    {'Cartesian product':<30s} {np.min(tc_f):>12,.0f} {np.median(tc_f):>12,.0f} "
          f"{np.max(tc_f):>12,.0f} {np.max(tc_f)/max(np.min(tc_f),1):>8.1f}x")
    print(f"    {'Children visited':<30s} {np.min(vis_f):>12,.0f} {np.median(vis_f):>12,.0f} "
          f"{np.max(vis_f):>12,.0f} {np.max(vis_f)/max(np.min(vis_f),1):>8.1f}x")
    print(f"    {'Time (s)':<30s} {np.min(t_arr):>12.3f} {np.median(t_arr):>12.3f} "
          f"{np.max(t_arr):>12.3f} {np.max(t_arr)/max(np.min(t_arr),0.001):>8.1f}x")
    surv_f = s_arr[:, 3].astype(float)
    print(f"    {'Survivors':<30s} {np.min(surv_f):>12,.0f} {np.median(surv_f):>12,.0f} "
          f"{np.max(surv_f):>12,.0f} {np.max(surv_f)/max(np.min(surv_f),1):>8.1f}x")
    print()

    # Time breakdown
    print("  Time attribution:")
    total_time = np.sum(t_arr)
    us_per_visit = total_time / max(total_visited, 1) * 1e6
    print(f"    Total: {total_time:.2f}s across {total_visited:,} visited children")
    print(f"    Avg time/visited child: {us_per_visit:.2f} us")
    print()

    # Cost model
    # QC path: ~50 ops (one window recheck)
    # Full scan path: ~ell_count * avg_windows ~ 63*63 ~ 4000 ops per ell, times avg_depth ells
    # Survival rate determines output volume
    if total_full > 0:
        avg_depth = np.sum(s_arr[:, 7]) / total_full
        print(f"    Estimated per-child cost model:")
        print(f"      Quick-check path:  ~1 window recheck")
        print(f"      Full-scan path:    ~{avg_depth:.1f} ell values x ~{2*32-1} windows/ell")
        qc_cost_est = 1.0   # relative
        full_cost_est = avg_depth * 63  # relative
        print(f"      Cost ratio (full/qc): ~{full_cost_est:.0f}x")
        print()
        qc_time_frac = total_qc * qc_cost_est / (total_qc * qc_cost_est + total_full * full_cost_est)
        full_time_frac = 1.0 - qc_time_frac
        print(f"      Estimated time in QC path:   {100*qc_time_frac:.1f}%")
        print(f"      Estimated time in Full path: {100*full_time_frac:.1f}%")


if __name__ == "__main__":
    main()
