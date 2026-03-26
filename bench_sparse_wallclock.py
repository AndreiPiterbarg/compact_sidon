"""Wall-clock validation of the iteration-count Amdahl estimate.

The iteration-count analysis treats every "iteration" as equal cost, but:
  - Zero-check iteration: load + branch = ~2-3 cycles
  - Nonzero cross-term iteration: 2 multiplies + 2 adds + 2 stores = ~12-15 cycles
  - Full-scan iteration: 1 add/sub + 1 compare = ~5 cycles

This script compares:
  1. Production Gray code kernel (wall clock)
  2. Instrumented kernel (wall clock, with nnz counting overhead)
  3. Derives the actual cross-term fraction by timing
"""
import sys
import os
import time
import math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction


# ---- Production Gray code kernel (copied from run_cascade.py) ----
@njit(cache=False)
def _gray_production(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
    """Exact copy of production Gray code kernel. Returns (n_surv, 0)."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200

    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

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
    for i in range(d_parent):
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

    while True:
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
                    j_idx = d_child - 1 - i
                    if child[j_idx] < child[i]:
                        use_rev = True
                        break
                    elif child[j_idx] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
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

    return n_surv, 0


# ---- Sparse variant: uses nz_list to skip zero bins ----
@njit(cache=False)
def _gray_sparse(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
    """Gray code kernel with sparse cross-term optimization."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    assert m <= 200

    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    # Sparse data structures
    nz_list = np.empty(d_child, dtype=np.int32)   # indices of nonzero bins
    nz_pos = np.full(d_child, -1, dtype=np.int32)  # reverse: bin -> position in nz_list
    nz_count = 0

    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    # Initialize nz_list
    for i in range(d_child):
        if child[i] != 0:
            nz_list[nz_count] = i
            nz_pos[i] = nz_count
            nz_count += 1

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
    for i in range(d_parent):
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

    while True:
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
                    j_idx = d_child - 1 - i
                    if child[j_idx] < child[i]:
                        use_rev = True
                        break
                    elif child[j_idx] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
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

        # --- SPARSE cross-term loop ---
        # Update nz_list for bins k1, k2
        # k1: old1 -> new1
        if old1 != 0 and new1 == 0:
            # Remove k1 from nz_list
            p = nz_pos[k1]
            nz_count -= 1
            last = nz_list[nz_count]
            nz_list[p] = last
            nz_pos[last] = p
            nz_pos[k1] = -1
        elif old1 == 0 and new1 != 0:
            # Add k1 to nz_list
            nz_list[nz_count] = k1
            nz_pos[k1] = nz_count
            nz_count += 1
        # k2: old2 -> new2
        if old2 != 0 and new2 == 0:
            p = nz_pos[k2]
            nz_count -= 1
            last = nz_list[nz_count]
            nz_list[p] = last
            nz_pos[last] = p
            nz_pos[k2] = -1
        elif old2 == 0 and new2 != 0:
            nz_list[nz_count] = k2
            nz_pos[k2] = nz_count
            nz_count += 1

        # Cross-terms: iterate only nonzero bins, skip k1 and k2
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

    return n_surv, 0


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


def generate_parents(n_half, m, c_target, target_level):
    """Run cascade to generate real parents."""
    from cpu.run_cascade import run_level0, process_parent_fused

    result = run_level0(n_half, m, c_target, verbose=False)
    survivors = result['survivors']
    print(f"  L0: {len(survivors)} survivors")

    for level in range(1, target_level):
        d_parent = survivors.shape[1]
        n_half_child = d_parent
        all_surv = []
        for i in range(len(survivors)):
            surv, _ = process_parent_fused(survivors[i], m, c_target, n_half_child)
            if len(surv) > 0:
                all_surv.append(surv)
        if all_surv:
            survivors = np.vstack(all_surv)
            survivors = np.unique(survivors, axis=0)
        else:
            survivors = np.empty((0, 2 * d_parent), dtype=np.int32)
        print(f"  L{level}: {len(survivors)} survivors")

    return survivors


def run_wallclock_test(name, parents, n_half_child, m, c_target, max_parents):
    d_parent = parents.shape[1]
    d_child = 2 * d_parent

    if max_parents < len(parents):
        indices = np.linspace(0, len(parents)-1, max_parents, dtype=int)
        sample = parents[indices]
    else:
        sample = parents

    # Prepare inputs
    inputs = []
    for pidx in range(len(sample)):
        parent = sample[pidx]
        result = compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, n_children = result
        if n_children <= 1:
            continue
        buf_cap = min(n_children, 500_000)
        out_buf = np.empty((buf_cap, d_child), dtype=np.int32)
        inputs.append((parent, lo_arr, hi_arr, out_buf))

    if not inputs:
        print(f"  {name}: no valid parents")
        return

    print(f"\n  {name}: d_child={d_child}, {len(inputs)} parents")

    # Warm up both kernels
    p, lo, hi, buf = inputs[0]
    _gray_production(p, n_half_child, m, c_target, lo, hi, buf)
    _gray_sparse(p, n_half_child, m, c_target, lo, hi, buf)

    # Verify correctness: both must produce same survivor count
    for p, lo, hi, buf in inputs[:20]:
        buf2 = buf.copy()
        ns_prod, _ = _gray_production(p, n_half_child, m, c_target, lo, hi, buf)
        ns_sparse, _ = _gray_sparse(p, n_half_child, m, c_target, lo, hi, buf2)
        if ns_prod != ns_sparse:
            print(f"  *** CORRECTNESS FAILURE: production={ns_prod}, sparse={ns_sparse} ***")
            print(f"      parent={p}")
            return

    print(f"  Correctness verified on {min(20, len(inputs))} parents.")

    # Time production kernel (3 runs)
    times_prod = []
    for run in range(3):
        t0 = time.perf_counter()
        for p, lo, hi, buf in inputs:
            _gray_production(p, n_half_child, m, c_target, lo, hi, buf)
        times_prod.append(time.perf_counter() - t0)

    # Time sparse kernel (3 runs)
    times_sparse = []
    for run in range(3):
        t0 = time.perf_counter()
        for p, lo, hi, buf in inputs:
            _gray_sparse(p, n_half_child, m, c_target, lo, hi, buf)
        times_sparse.append(time.perf_counter() - t0)

    best_prod = min(times_prod)
    best_sparse = min(times_sparse)
    speedup = best_prod / best_sparse if best_sparse > 0 else float('inf')

    print(f"\n  WALL-CLOCK RESULTS (best of 3 runs):")
    print(f"  Production kernel:  {best_prod:.4f}s")
    print(f"  Sparse kernel:      {best_sparse:.4f}s")
    print(f"  *** ACTUAL SPEEDUP: {speedup:.3f}x ***")
    print(f"  (all runs: prod={[f'{t:.4f}' for t in times_prod]}, "
          f"sparse={[f'{t:.4f}' for t in times_sparse]})")


def main():
    m = 20
    c_target = 1.4
    n_half = 2

    print("=" * 70)
    print("WALL-CLOCK A/B TEST: Production vs Sparse Gray Code Kernel")
    print("=" * 70)

    print("\n--- Generating cascade parents ---")
    l0_surv = generate_parents(n_half, m, c_target, 1)
    l1_surv = generate_parents(n_half, m, c_target, 2)

    # JIT warmup
    print("\nWarming up JIT...")
    dummy = np.array([10, 10], dtype=np.int32)
    dlo = np.array([0, 0], dtype=np.int32)
    dhi = np.array([5, 5], dtype=np.int32)
    dbuf = np.empty((100, 4), dtype=np.int32)
    _gray_production(dummy, 2, 20, 1.4, dlo, dhi, dbuf)
    _gray_sparse(dummy, 2, 20, 1.4, dlo, dhi, dbuf)
    print("JIT warm.")

    # Test L0->L1
    run_wallclock_test('L0->L1', l0_surv, l0_surv.shape[1], m, c_target, 467)

    # Test L1->L2
    run_wallclock_test('L1->L2', l1_surv, l1_surv.shape[1], m, c_target, 2000)

    # Test L2->L3
    print("\n--- Generating L2 parents (may take ~60s) ---")
    l2_surv = generate_parents(n_half, m, c_target, 3)
    if len(l2_surv) > 0:
        run_wallclock_test('L2->L3', l2_surv, l2_surv.shape[1], m, c_target, 500)


if __name__ == '__main__':
    main()
