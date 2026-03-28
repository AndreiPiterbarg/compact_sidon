#!/usr/bin/env python
"""Profile pruning at each cascade level by running the actual cascade.

Starts from L0 survivors (d=4) and runs L1 (d=8), L2 (d=16), L3 (d=32)
with instrumented kernels to understand pruning behavior at each level.
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
def _instrumented_kernel(parent_int, n_half_child, m, c_target,
                          lo_arr, hi_arr, out_buf):
    """Instrumented Gray code kernel returning stats + survivors.

    Returns (n_surv, stats[12])
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    stats = np.zeros(12, dtype=np.int64)

    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, stats

    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    conv_len = 2 * d_child - 1
    max_survivors = out_buf.shape[0]
    n_surv = 0

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
        dyn_base_ell_arr[ell - 2] = c_target * m_d * m_d * np.float64(ell) * inv_4n

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
                        break

            if pruned:
                stats[2] += 1
                stats[7] += scan_depth
                if scan_depth > stats[8]:
                    stats[8] = scan_depth
            else:
                stats[3] += 1
                # Canonicalize + store survivor
                use_rev = False
                half_d = d_child // 2
                for i in range(half_d):
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

        # Subtree pruning (only for n_active > J_MIN)
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
                    dyn_base_ell = dyn_base_ell_arr[ell - 2]
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
                        fixed_hi = min(hi_bin, fixed_len - 1)
                        if fixed_hi >= lo_bin:
                            lo_clamp = max(lo_bin, 0)
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)
                        unfixed_lo_bin = max(lo_bin, fixed_len)
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = max(unfixed_lo_bin // 2, first_unfixed_parent)
                            p_hi = min(hi_bin // 2, d_parent - 1)
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
                        gc_a[kk] = 0; gc_dir[kk] = 1; gc_focus[kk] = kk
                    gc_focus[0] = next_focus; gc_focus[J_MIN] = J_MIN
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
                                nz_list[nz_count] = ii; nz_pos[ii] = nz_count; nz_count += 1
                            else:
                                nz_pos[ii] = -1
                    if qc_ell > 0:
                        qc_lo2 = max(qc_s - (d_child - 1), 0)
                        qc_hi2 = min(qc_s + qc_ell - 2, d_child - 1)
                        qc_W_int = np.int64(0)
                        for ii in range(qc_lo2, qc_hi2 + 1):
                            qc_W_int += np.int64(child[ii])
                    continue

    return n_surv, stats


def run_level(parents, level, m, c_target, max_parents=None):
    """Run one cascade level with instrumented kernel."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    n_parents = len(parents)
    if max_parents is not None and n_parents > max_parents:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_parents, size=max_parents, replace=False)
        indices.sort()
        parents = parents[indices]
        n_parents = max_parents

    print(f"  Level L{level}: d_parent={d_parent} -> d_child={d_child}, "
          f"{n_parents} parents")

    # Warmup
    dummy = parents[0].copy()
    result = compute_bin_ranges(dummy, m, c_target, d_child, n_half_child)
    if result is not None:
        lo, hi, tc = result
        buf = np.empty((min(tc, 1000), d_child), dtype=np.int32)
        _, _ = _instrumented_kernel(dummy, n_half_child, m, c_target, lo, hi, buf)

    total_stats = np.zeros(12, dtype=np.int64)
    total_cart = 0
    total_time = 0.0
    all_survivors = []
    empty_range_count = 0
    asym_pruned_count = 0
    parent_times = []
    parent_visited = []
    parent_survivors = []

    for i, parent in enumerate(parents):
        result = compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            empty_range_count += 1
            parent_times.append(0.0)
            parent_visited.append(0)
            parent_survivors.append(0)
            continue
        lo_arr, hi_arr, tc = result
        total_cart += tc

        buf_cap = min(tc + 1, 5_000_000)
        buf = np.empty((buf_cap, d_child), dtype=np.int32)

        t0 = time.perf_counter()
        n_surv, stats = _instrumented_kernel(parent, n_half_child, m, c_target,
                                              lo_arr, hi_arr, buf)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if stats[0] == 0:
            asym_pruned_count += 1

        total_stats += stats
        parent_times.append(elapsed)
        parent_visited.append(int(stats[0]))
        parent_survivors.append(n_surv)

        if n_surv > 0 and n_surv <= buf_cap:
            all_survivors.append(buf[:n_surv].copy())

        if (i + 1) % max(n_parents // 5, 1) == 0:
            print(f"    Progress: {i+1}/{n_parents} parents, "
                  f"{int(total_stats[3]):,} survivors so far, {total_time:.1f}s")

    # Collect survivors
    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    # Deduplicate
    if len(survivors) > 0:
        # Simple sort-based dedup
        sort_idx = np.lexsort(survivors.T[::-1])
        survivors = survivors[sort_idx]
        mask = np.ones(len(survivors), dtype=bool)
        for i in range(1, len(survivors)):
            if np.array_equal(survivors[i], survivors[i-1]):
                mask[i] = False
        survivors = survivors[mask]

    total_visited = total_stats[0]
    total_qc = total_stats[1]
    total_full = total_stats[2]
    total_surv_raw = total_stats[3]
    total_sub_checks = total_stats[4]
    total_sub_success = total_stats[5]
    total_sub_skipped = total_stats[6]
    total_scan_depth = total_stats[7]
    max_scan_depth = total_stats[8]

    print()
    print(f"  L{level} Results:")
    print(f"    Parents processed:       {n_parents:>14,}")
    print(f"    Empty-range parents:     {empty_range_count:>14,}")
    print(f"    Asymmetry-pruned:        {asym_pruned_count:>14,}")
    print(f"    Cartesian product total: {total_cart:>14,}")
    print(f"    Children visited:        {total_visited:>14,}")
    if total_cart > 0:
        print(f"      Visit fraction:        {100*total_visited/total_cart:>13.1f}%")
    print(f"    Quick-check kills:       {total_qc:>14,} "
          f"({100*total_qc/max(total_visited,1):.1f}%)")
    print(f"    Full-scan prunes:        {total_full:>14,} "
          f"({100*total_full/max(total_visited,1):.1f}%)")
    print(f"    Survivors (raw):         {total_surv_raw:>14,} "
          f"({100*total_surv_raw/max(total_visited,1):.4f}%)")
    print(f"    Survivors (deduped):     {len(survivors):>14,}")
    print(f"    Subtree checks:          {total_sub_checks:>14,}")
    print(f"    Subtree successes:       {total_sub_success:>14,} "
          f"({100*total_sub_success/max(total_sub_checks,1):.1f}%)")
    print(f"    Subtree skipped:         {total_sub_skipped:>14,}")
    if total_full > 0:
        avg_depth = total_scan_depth / total_full
        print(f"    Avg scan depth:          {avg_depth:>14.2f}")
        print(f"    Max scan depth:          {max_scan_depth:>14}")
    print(f"    Total time:              {total_time:>14.2f}s")
    if total_visited > 0:
        print(f"    Time/visited child:      {total_time/total_visited*1e6:>14.2f} us")

    # Parent heterogeneity
    pt = np.array(parent_times)
    pv = np.array(parent_visited)
    ps = np.array(parent_survivors)
    active_mask = pv > 0
    if np.sum(active_mask) > 1:
        pt_a = pt[active_mask]
        pv_a = pv[active_mask]
        ps_a = ps[active_mask]
        print(f"\n    Parent heterogeneity (active parents only):")
        print(f"      {'Metric':<20s} {'Min':>12s} {'Median':>12s} {'Mean':>12s} "
              f"{'Max':>12s} {'Ratio':>8s}")
        for label, arr in [("Visited", pv_a.astype(float)),
                           ("Survivors", ps_a.astype(float)),
                           ("Time (ms)", pt_a * 1000)]:
            mn, med, mean, mx = np.min(arr), np.median(arr), np.mean(arr), np.max(arr)
            ratio = mx / max(mn, 1e-9)
            print(f"      {label:<20s} {mn:>12.1f} {med:>12.1f} {mean:>12.1f} "
                  f"{mx:>12.1f} {ratio:>8.1f}x")

    print()
    return survivors


def main():
    print("=" * 80)
    print("CASCADE PRUNING PROFILE: L0 -> L1 -> L2 -> L3")
    print(f"Parameters: m={M}, c_target={C_TARGET}")
    print("=" * 80)
    print()

    # Print correction and x_cap at each level
    print("--- LEVEL PARAMETERS ---")
    for level in range(5):
        d_child = 4 * (2 ** level)
        n_half_child = d_child // 2
        corr = correction(M, n_half_child)
        thresh = C_TARGET + corr + 1e-9
        x_cap = int(math.floor(M * math.sqrt(thresh / d_child)))
        x_cap_cs = int(math.floor(M * math.sqrt(C_TARGET / d_child)))
        x_cap = min(x_cap, x_cap_cs, M)
        print(f"  L{level}: d_child={d_child:>3d}, n_half={n_half_child:>2d}, "
              f"correction={corr:.4f}, x_cap={x_cap}, "
              f"avg_mass/bin={M/d_child:.3f}")
    print()

    # Load L0 survivors
    l0_path = os.path.join(DATA_DIR, "checkpoint_L0_survivors.npy")
    l0 = np.load(l0_path)
    print(f"L0 survivors: {len(l0)} compositions of d=4")
    print()

    # L1: d=4 -> d=8
    l1_survivors = run_level(l0, 1, M, C_TARGET)

    # L2: d=8 -> d=16
    if len(l1_survivors) > 0:
        l2_survivors = run_level(l1_survivors, 2, M, C_TARGET, max_parents=500)
    else:
        print("  No L1 survivors -- proof complete at L1!")
        return

    # L3: d=16 -> d=32 (sample)
    if len(l2_survivors) > 0:
        l3_survivors = run_level(l2_survivors, 3, M, C_TARGET, max_parents=100)
    else:
        print("  No L2 survivors -- proof complete at L2!")
        return

    # L4: d=32 -> d=64 (sample)
    if len(l3_survivors) > 0:
        l4_survivors = run_level(l3_survivors, 4, M, C_TARGET, max_parents=10)
    else:
        print("  No L3 survivors -- proof complete at L3!")
        return

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("  The cascade shows exponential growth of survivors because pruning")
    print("  becomes less effective as d grows: the correction term dominates")
    print("  and x_cap shrinks, making all children 'safe' from the bound.")


if __name__ == "__main__":
    main()
