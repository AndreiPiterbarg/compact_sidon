"""Measure quick-check hit rate at d_child=32."""
import sys
import os
import time
import numpy as np
from numba import njit
import math

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import _compute_bin_ranges

M = 20
C_TARGET = 1.4


@njit(cache=False)
def _measure_quick_check(parent_int, n_half_child, m, c_target, lo_arr, hi_arr):
    """Simplified Gray code kernel that counts quick-check vs full-scan kills."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0, 0

    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d
    conv_len = 2 * d_child - 1
    ell_count = 2 * d_child - 1

    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        dyn_base_ell_arr[ell - 2] = c_target * m_d * m_d * np.float64(ell) * inv_4n

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

    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]
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
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

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

    n_quick = 0
    n_full = 0
    n_surv = 0

    while True:
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = (dyn_base_ell_arr[ell_idx_qc] + 1.0 + eps_margin
                        + 2.0 * np.float64(qc_W_int))
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                n_quick += 1

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
                n_windows = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += (np.int64(raw_conv[s_lo + n_cv - 1])
                               - np.int64(raw_conv[s_lo - 1]))
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_x = (dyn_base_ell + 1.0 + eps_margin
                             + 2.0 * np.float64(W_int))
                    dyn_it = np.int64(dyn_x * one_minus_4eps)
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break
            if pruned:
                n_full += 1
            else:
                n_surv += 1

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

    return n_quick, n_full, n_surv


if __name__ == '__main__':
    shard_dir = os.path.join(_project_dir, 'data', '_shards_L2')
    shard_files = sorted([f for f in os.listdir(shard_dir)
                         if f.startswith('shard_') and f.endswith('.npy')
                         and '.m' not in f])
    shard_path = os.path.join(shard_dir, shard_files[0])
    l2_surv = np.array(np.load(shard_path, mmap_mode='r')[:2000])

    # Warmup
    result = _compute_bin_ranges(l2_surv[0], M, C_TARGET, 32, 16)
    if result:
        _measure_quick_check(l2_surv[0], 16, M, C_TARGET, result[0], result[1])

    total_quick = 0
    total_full = 0
    total_surv = 0
    total_children = 0
    for i in range(200):
        result = _compute_bin_ranges(l2_surv[i], M, C_TARGET, 32, 16)
        if result is None:
            continue
        lo_arr, hi_arr, total = result
        if total == 0:
            continue
        nq, nf, ns = _measure_quick_check(
            l2_surv[i], 16, M, C_TARGET, lo_arr, hi_arr)
        total_quick += nq
        total_full += nf
        total_surv += ns
        total_children += total

    print(f"Total children: {total_children:,}")
    print(f"Quick-check kills: {total_quick:,} "
          f"({total_quick / total_children * 100:.1f}%)")
    print(f"Full-scan kills:   {total_full:,} "
          f"({total_full / total_children * 100:.1f}%)")
    print(f"Survivors:         {total_surv:,} "
          f"({total_surv / total_children * 100:.3f}%)")
    qc_rate = total_quick / (total_quick + total_full) * 100
    print(f"Quick-check success rate (among pruned): {qc_rate:.1f}%")
