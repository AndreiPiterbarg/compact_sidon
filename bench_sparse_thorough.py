"""Thorough wall-clock A/B test at the actual target dimension (d_child=64).

Uses L3 parents reconstructed from L4 survivors (sum adjacent pairs).
These are guaranteed to produce >=1 survivor each, so they represent the
hardest parents in the L4 workload (conservative test — easy parents
benefit MORE from sparse because their QC hit rate is higher).

Tests:
  1. Full correctness: survivor arrays must be identical, not just counts
  2. Wall-clock timing at d_child=64 (the real L4 target)
  3. Wall-clock timing at d_child=32 for comparison
  4. Multiple timing runs with statistical reporting
  5. Throughput: children/sec for each kernel
"""
import sys
import os
import time
import math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction


# =====================================================================
# Production Gray code kernel (exact copy from run_cascade.py)
# =====================================================================
@njit(cache=False)
def _gray_production(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
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


# =====================================================================
# Sparse variant with nz_list
# =====================================================================
@njit(cache=False)
def _gray_sparse(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
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

    # Sparse structures
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

        # Update nz_list BEFORE cross-term loop (so we iterate current nonzeros)
        if old1 != 0 and new1 == 0:
            p = nz_pos[k1]
            nz_count -= 1
            last = nz_list[nz_count]
            nz_list[p] = last
            nz_pos[last] = p
            nz_pos[k1] = -1
        elif old1 == 0 and new1 != 0:
            nz_list[nz_count] = k1
            nz_pos[k1] = nz_count
            nz_count += 1
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

        # Sparse cross-term loop
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


# =====================================================================
# Helpers
# =====================================================================
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


def prepare_inputs(parents, m, c_target, n_half_child, max_parents, max_buf=500_000):
    """Prepare kernel inputs, filtering out trivial parents."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent

    if max_parents < len(parents):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(parents), max_parents, replace=False)
        indices.sort()
        sample = parents[indices]
    else:
        sample = parents

    inputs = []
    total_children = 0
    for pidx in range(len(sample)):
        parent = sample[pidx]
        result = compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, n_children = result
        if n_children <= 1:
            continue
        buf_cap = min(n_children, max_buf)
        inputs.append((parent, lo_arr, hi_arr, buf_cap, n_children))
        total_children += n_children

    return inputs, total_children, d_child


def verify_correctness(inputs, n_half_child, m, c_target, d_child, n_to_check):
    """Verify production and sparse produce identical survivor ARRAYS."""
    n_checked = 0
    n_surv_total = 0
    for parent, lo_arr, hi_arr, buf_cap, n_children in inputs[:n_to_check]:
        buf_prod = np.empty((buf_cap, d_child), dtype=np.int32)
        buf_sparse = np.empty((buf_cap, d_child), dtype=np.int32)

        ns_prod, _ = _gray_production(parent, n_half_child, m, c_target,
                                       lo_arr, hi_arr, buf_prod)
        ns_sparse, _ = _gray_sparse(parent, n_half_child, m, c_target,
                                     lo_arr, hi_arr, buf_sparse)

        if ns_prod != ns_sparse:
            print(f"    FAIL: count mismatch prod={ns_prod} sparse={ns_sparse}")
            print(f"    parent={parent}")
            return False

        if ns_prod > 0 and ns_prod <= buf_cap:
            prod_sorted = buf_prod[:ns_prod].copy()
            sparse_sorted = buf_sparse[:ns_sparse].copy()
            # Sort both for comparison (Gray code visits same set but
            # canonicalization makes output order deterministic)
            prod_sorted = prod_sorted[np.lexsort(prod_sorted[:, ::-1].T)]
            sparse_sorted = sparse_sorted[np.lexsort(sparse_sorted[:, ::-1].T)]
            if not np.array_equal(prod_sorted, sparse_sorted):
                print(f"    FAIL: array mismatch (counts match={ns_prod})")
                print(f"    parent={parent}")
                # Find first difference
                for r in range(ns_prod):
                    if not np.array_equal(prod_sorted[r], sparse_sorted[r]):
                        print(f"    row {r}: prod={prod_sorted[r]}")
                        print(f"            sparse={sparse_sorted[r]}")
                        break
                return False

        n_checked += 1
        n_surv_total += ns_prod

    return n_checked, n_surv_total


def time_kernel(kernel, inputs, n_half_child, m, c_target, d_child, n_runs):
    """Time a kernel over all inputs, return list of per-run times."""
    times = []
    total_surv = 0
    for run in range(n_runs):
        t0 = time.perf_counter()
        run_surv = 0
        for parent, lo_arr, hi_arr, buf_cap, n_children in inputs:
            out_buf = np.empty((buf_cap, d_child), dtype=np.int32)
            ns, _ = kernel(parent, n_half_child, m, c_target,
                           lo_arr, hi_arr, out_buf)
            run_surv += ns
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if run == 0:
            total_surv = run_surv
    return times, total_surv


def main():
    m = 20
    c_target = 1.4
    N_TIMING_RUNS = 7

    print("=" * 70)
    print("THOROUGH WALL-CLOCK A/B TEST")
    print(f"  m={m}, c_target={c_target}, {N_TIMING_RUNS} timing runs each")
    print("=" * 70)

    # ── Load L3 parents from L4 survivors ──
    l4 = np.load('data/checkpoint_L4_survivors.npy')
    l3_parents = (l4[:, 0::2] + l4[:, 1::2]).astype(np.int32)
    print(f"\nReconstructed {len(l3_parents):,} L3 parents from L4 survivors")
    print(f"  d_parent=32, d_child=64")
    nnz_parents = np.count_nonzero(l3_parents, axis=1)
    print(f"  parent nnz: min={nnz_parents.min()}, max={nnz_parents.max()}, "
          f"mean={nnz_parents.mean():.1f}")

    # ── Also generate L2 parents for d_child=32 comparison ──
    print("\nGenerating L2 parents via cascade...")
    from cpu.run_cascade import run_level0, process_parent_fused
    r0 = run_level0(2, m, c_target, verbose=False)
    l0_surv = r0['survivors']
    print(f"  L0: {len(l0_surv)} survivors")
    all_l1 = []
    for i in range(len(l0_surv)):
        s, _ = process_parent_fused(l0_surv[i], m, c_target, l0_surv.shape[1])
        if len(s) > 0:
            all_l1.append(s)
    l1_surv = np.unique(np.vstack(all_l1), axis=0) if all_l1 else np.empty((0, 8), dtype=np.int32)
    print(f"  L1: {len(l1_surv)} survivors")
    all_l2 = []
    for i in range(len(l1_surv)):
        s, _ = process_parent_fused(l1_surv[i], m, c_target, l1_surv.shape[1])
        if len(s) > 0:
            all_l2.append(s)
    l2_parents = np.unique(np.vstack(all_l2), axis=0) if all_l2 else np.empty((0, 16), dtype=np.int32)
    print(f"  L2: {len(l2_parents)} survivors (= L3 parents)")

    # ── JIT warmup ──
    print("\nWarming up JIT...")
    dummy = np.array([10, 10], dtype=np.int32)
    dlo = np.array([0, 0], dtype=np.int32)
    dhi = np.array([5, 5], dtype=np.int32)
    dbuf = np.empty((100, 4), dtype=np.int32)
    _gray_production(dummy, 2, 20, 1.4, dlo, dhi, dbuf)
    _gray_sparse(dummy, 2, 20, 1.4, dlo, dhi, dbuf)
    print("JIT warm.\n")

    # ═══════════════════════════════════════════════════
    # TEST 1: d_child=32 (L2->L3)
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 1: L2->L3 (d_child=32)")
    print("=" * 70)

    n_half_child_32 = 16
    inputs_32, total_ch_32, d_child_32 = prepare_inputs(
        l2_parents, m, c_target, n_half_child_32, max_parents=1000)
    print(f"  {len(inputs_32)} parents, {total_ch_32:,} total children")

    # Correctness
    print("\n  Correctness check (full array comparison)...")
    result = verify_correctness(inputs_32, n_half_child_32, m, c_target, d_child_32, 200)
    if result is False:
        print("  *** CORRECTNESS FAILURE - ABORTING ***")
        return
    n_checked, n_surv = result
    print(f"  PASS: {n_checked} parents verified, {n_surv} total survivors matched")

    # Timing
    print(f"\n  Timing ({N_TIMING_RUNS} runs each)...")
    times_prod_32, surv_32 = time_kernel(
        _gray_production, inputs_32, n_half_child_32, m, c_target, d_child_32, N_TIMING_RUNS)
    times_sparse_32, _ = time_kernel(
        _gray_sparse, inputs_32, n_half_child_32, m, c_target, d_child_32, N_TIMING_RUNS)

    best_prod_32 = min(times_prod_32)
    best_sparse_32 = min(times_sparse_32)
    med_prod_32 = sorted(times_prod_32)[N_TIMING_RUNS // 2]
    med_sparse_32 = sorted(times_sparse_32)[N_TIMING_RUNS // 2]

    print(f"\n  Production: best={best_prod_32:.4f}s  median={med_prod_32:.4f}s  "
          f"all={[f'{t:.3f}' for t in times_prod_32]}")
    print(f"  Sparse:     best={best_sparse_32:.4f}s  median={med_sparse_32:.4f}s  "
          f"all={[f'{t:.3f}' for t in times_sparse_32]}")
    print(f"\n  *** SPEEDUP (best): {best_prod_32/best_sparse_32:.3f}x ***")
    print(f"  *** SPEEDUP (median): {med_prod_32/med_sparse_32:.3f}x ***")
    print(f"  Throughput: prod={total_ch_32/best_prod_32/1e6:.1f}M ch/s  "
          f"sparse={total_ch_32/best_sparse_32/1e6:.1f}M ch/s")

    # ═══════════════════════════════════════════════════
    # TEST 2: d_child=64 (L3->L4) — THE REAL TARGET
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 2: L3->L4 (d_child=64) -- THE ACTUAL TARGET")
    print("=" * 70)

    n_half_child_64 = 32
    inputs_64, total_ch_64, d_child_64 = prepare_inputs(
        l3_parents, m, c_target, n_half_child_64, max_parents=2000)
    print(f"  {len(inputs_64)} parents, {total_ch_64:,} total children")

    # Show children-per-parent distribution
    ch_counts = [nc for _, _, _, _, nc in inputs_64]
    ch_arr = np.array(ch_counts)
    print(f"  Children/parent: min={ch_arr.min():,}, max={ch_arr.max():,}, "
          f"mean={ch_arr.mean():,.0f}, median={np.median(ch_arr):,.0f}")

    # Correctness
    print("\n  Correctness check (full array comparison)...")
    result = verify_correctness(inputs_64, n_half_child_64, m, c_target, d_child_64, 500)
    if result is False:
        print("  *** CORRECTNESS FAILURE - ABORTING ***")
        return
    n_checked, n_surv = result
    print(f"  PASS: {n_checked} parents verified, {n_surv} total survivors matched")

    # Timing
    print(f"\n  Timing ({N_TIMING_RUNS} runs each)...")
    times_prod_64, surv_64 = time_kernel(
        _gray_production, inputs_64, n_half_child_64, m, c_target, d_child_64, N_TIMING_RUNS)
    times_sparse_64, _ = time_kernel(
        _gray_sparse, inputs_64, n_half_child_64, m, c_target, d_child_64, N_TIMING_RUNS)

    best_prod_64 = min(times_prod_64)
    best_sparse_64 = min(times_sparse_64)
    med_prod_64 = sorted(times_prod_64)[N_TIMING_RUNS // 2]
    med_sparse_64 = sorted(times_sparse_64)[N_TIMING_RUNS // 2]

    print(f"\n  Production: best={best_prod_64:.4f}s  median={med_prod_64:.4f}s  "
          f"all={[f'{t:.3f}' for t in times_prod_64]}")
    print(f"  Sparse:     best={best_sparse_64:.4f}s  median={med_sparse_64:.4f}s  "
          f"all={[f'{t:.3f}' for t in times_sparse_64]}")
    print(f"\n  *** SPEEDUP (best): {best_prod_64/best_sparse_64:.3f}x ***")
    print(f"  *** SPEEDUP (median): {med_prod_64/med_sparse_64:.3f}x ***")
    print(f"  Throughput: prod={total_ch_64/best_prod_64/1e6:.1f}M ch/s  "
          f"sparse={total_ch_64/best_sparse_64/1e6:.1f}M ch/s")
    print(f"  Survivors: {surv_64}")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  d_child=32:  {best_prod_32/best_sparse_32:.3f}x (best)  {med_prod_32/med_sparse_32:.3f}x (median)")
    print(f"  d_child=64:  {best_prod_64/best_sparse_64:.3f}x (best)  {med_prod_64/med_sparse_64:.3f}x (median)")
    print()

    hours_l3 = 56768 / 3600
    hours_l4 = 250235 / 3600
    saved_l3 = hours_l3 * (1 - best_sparse_32 / best_prod_32) if best_sparse_32 < best_prod_32 else 0
    saved_l4 = hours_l4 * (1 - best_sparse_64 / best_prod_64) if best_sparse_64 < best_prod_64 else 0
    print(f"  L3 (16h baseline): would save {saved_l3:.1f}h")
    print(f"  L4 (70h baseline): would save {saved_l4:.1f}h")
    print()

    note = ""
    if best_prod_64 / best_sparse_64 >= 1.15:
        note = "RECOMMEND implementing for d_child >= 32"
    elif best_prod_64 / best_sparse_64 >= 1.05:
        note = "MARGINAL -- consider only if implementation is quick"
    else:
        note = "NOT RECOMMENDED -- insufficient gain"
    print(f"  VERDICT: {note}")


if __name__ == '__main__':
    main()
