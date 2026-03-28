"""Benchmark Idea 1: Incremental prefix_c maintenance.

A/B comparison of baseline vs. optimized kernel on the same parents.
The optimization maintains prefix_c incrementally (O(1) per Gray step)
instead of recomputing from scratch (O(d_child)) on every quick-check miss.

Both kernels must produce identical survivor counts and identical output.
"""
import sys
import os
import time
import math
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

import numba
from numba import njit

from cpu.run_cascade import (
    _fused_generate_and_prune_gray as _baseline_kernel,
    _compute_bin_ranges,
    run_level0,
    process_parent_fused,
)

M = 20
C_TARGET = 1.4


# =====================================================================
# Idea 1: Incremental prefix_c — modified kernel
# =====================================================================
@njit(cache=False)
def _kernel_idea1(parent_int, n_half_child, m, c_target,
                  lo_arr, hi_arr, out_buf):
    """Identical to baseline except prefix_c is maintained incrementally."""
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

    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    J_MIN = 7
    n_subtree_pruned = 0
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

    # --- IDEA 1 CHANGE: compute prefix_c once at init ---
    prefix_c[0] = 0
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

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
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + 1.0 + eps_margin + 2.0 * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True

        if not quick_killed:
            # --- IDEA 1 CHANGE: prefix_c is already up-to-date, skip recomputation ---
            # (baseline would recompute prefix_c[0..d_child] here in O(d_child))

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

            if not pruned:
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

        # --- IDEA 1 CHANGE: O(1) prefix_c update ---
        prefix_c[k2] += np.int64(delta1)

        # Quick-check W_int update (O(1))
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

                # Subtree pruning uses its own local prefix_c for fixed region
                # We use a temporary to avoid corrupting the maintained prefix_c
                sub_prefix_c = np.empty(fixed_len + 1, dtype=np.int64)
                sub_prefix_c[0] = 0
                for ii in range(fixed_len):
                    sub_prefix_c[ii + 1] = sub_prefix_c[ii] + np.int64(child[ii])

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
                            W_int_fixed = sub_prefix_c[fixed_hi + 1] - sub_prefix_c[lo_clamp]
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

                    # --- IDEA 1 CHANGE: recompute prefix_c after subtree reset ---
                    prefix_c[0] = 0
                    for ii in range(d_child):
                        prefix_c[ii + 1] = prefix_c[ii] + np.int64(child[ii])

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


# =====================================================================
# Benchmark harness
# =====================================================================
def generate_parents(target_level):
    """Generate parents for a given level using the cascade."""
    l0 = run_level0(n_half=2, m=M, c_target=C_TARGET, verbose=False)
    l0_surv = l0['survivors']

    if target_level == 1:
        return l0_surv, 4  # d_parent=4, n_half_child=4

    # Generate L1
    all_l1 = []
    for parent in l0_surv:
        surv, tc = process_parent_fused(parent, M, C_TARGET, 4)
        if len(surv) > 0:
            all_l1.append(surv)
    if all_l1:
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        l1_surv = np.vstack(all_l1)
        _canonicalize_inplace(l1_surv)
        l1_surv = _fast_dedup(l1_surv)
    else:
        l1_surv = np.empty((0, 8), dtype=np.int32)

    if target_level == 2:
        return l1_surv, 8  # d_parent=8, n_half_child=8

    # Generate L2 from a sample of L1
    all_l2 = []
    for parent in l1_surv:
        surv, tc = process_parent_fused(parent, M, C_TARGET, 8)
        if len(surv) > 0:
            all_l2.append(surv)
    if all_l2:
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        l2_surv = np.vstack(all_l2)
        _canonicalize_inplace(l2_surv)
        l2_surv = _fast_dedup(l2_surv)
    else:
        l2_surv = np.empty((0, 16), dtype=np.int32)

    return l2_surv, 16  # d_parent=16, n_half_child=16


def prepare_workload(parents, n_half_child, n_parents=200,
                     min_children=10):
    """Select parents with enough children, sorted by total_children desc."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    valid = []
    for i in range(len(parents)):
        result = _compute_bin_ranges(parents[i], M, C_TARGET,
                                     d_child, n_half_child)
        if result is not None:
            lo, hi, tc = result
            if tc >= min_children:
                valid.append((parents[i].copy(), lo, hi, tc))
    valid.sort(key=lambda x: -x[3])
    return valid[:n_parents]


def run_ab(workload, n_half_child, label, n_runs=5):
    """A/B comparison: baseline vs Idea 1 on the same workload."""
    d_parent = workload[0][0].shape[0]
    d_child = 2 * d_parent
    total_children_all = sum(tc for _, _, _, tc in workload)

    print(f"\n{'='*70}")
    print(f"Idea 1 A/B: {label}  ({len(workload)} parents, "
          f"{total_children_all:,} children)")
    print(f"{'='*70}")

    # --- Warmup both kernels (JIT compile) ---
    p0, lo0, hi0, tc0 = workload[0]
    buf = np.empty((min(tc0, 500_000), d_child), dtype=np.int32)
    _baseline_kernel(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)
    _kernel_idea1(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)
    print("  JIT warmup done.")

    # --- Correctness check: verify identical survivors ---
    print("  Verifying correctness...", end=" ", flush=True)
    mismatch = False
    for parent, lo, hi, tc in workload[:50]:
        buf_cap = min(tc, 5_000_000)
        buf_a = np.empty((buf_cap, d_child), dtype=np.int32)
        buf_b = np.empty((buf_cap, d_child), dtype=np.int32)
        n_a, sub_a = _baseline_kernel(parent, n_half_child, M, C_TARGET,
                                       lo, hi, buf_a)
        n_b, sub_b = _kernel_idea1(parent, n_half_child, M, C_TARGET,
                                    lo, hi, buf_b)
        if n_a != n_b:
            print(f"\n  MISMATCH: baseline={n_a}, idea1={n_b} for parent {parent}")
            mismatch = True
            break
        if n_a > 0 and n_a <= buf_cap:
            a_sorted = buf_a[:n_a].copy()
            b_sorted = buf_b[:n_b].copy()
            a_sorted.sort(axis=0)
            b_sorted.sort(axis=0)
            if not np.array_equal(a_sorted, b_sorted):
                print(f"\n  CONTENT MISMATCH for parent {parent}")
                mismatch = True
                break
    if mismatch:
        print("  FAILED — results differ, aborting benchmark.")
        return
    print("PASS (bit-identical)")

    # --- Timing runs ---
    for variant_name, kernel in [("BASELINE", _baseline_kernel),
                                  ("IDEA 1 ", _kernel_idea1)]:
        timings = []
        for run in range(n_runs):
            total_ch = 0
            total_sv = 0
            t0 = time.perf_counter()
            for parent, lo, hi, tc in workload:
                buf_cap = min(tc, 5_000_000)
                out = np.empty((buf_cap, d_child), dtype=np.int32)
                ns, _ = kernel(parent, n_half_child, M, C_TARGET,
                               lo, hi, out)
                total_ch += tc
                total_sv += ns
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

        timings.sort()
        median = timings[len(timings) // 2]
        best = timings[0]
        throughput = total_ch / median
        print(f"  {variant_name}: median={median:.4f}s  best={best:.4f}s  "
              f"{throughput/1e6:.2f}M ch/s  surv={total_sv:,}")

    # Speedup
    baseline_times = []
    idea1_times = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        for parent, lo, hi, tc in workload:
            buf = np.empty((min(tc, 5_000_000), d_child), dtype=np.int32)
            _baseline_kernel(parent, n_half_child, M, C_TARGET, lo, hi, buf)
        baseline_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for parent, lo, hi, tc in workload:
            buf = np.empty((min(tc, 5_000_000), d_child), dtype=np.int32)
            _kernel_idea1(parent, n_half_child, M, C_TARGET, lo, hi, buf)
        idea1_times.append(time.perf_counter() - t0)

    baseline_times.sort()
    idea1_times.sort()
    b_med = baseline_times[len(baseline_times) // 2]
    i_med = idea1_times[len(idea1_times) // 2]
    speedup = b_med / i_med if i_med > 0 else float('inf')
    delta_pct = (1.0 - i_med / b_med) * 100 if b_med > 0 else 0
    print(f"\n  >>> SPEEDUP: {speedup:.3f}x  ({delta_pct:+.1f}%)")
    print(f"      baseline median={b_med:.4f}s, idea1 median={i_med:.4f}s")


def main():
    print("Benchmark: Idea 1 — Incremental prefix_c Maintenance")
    print("=" * 70)

    # L2 benchmark: d_child=16 (modest expected impact)
    print("\nGenerating L1 survivors as L2 parents...")
    parents, nhc = generate_parents(target_level=2)
    print(f"  Got {len(parents)} parents at d={parents.shape[1]}")

    wl_l2 = prepare_workload(parents, nhc, n_parents=300)
    if wl_l2:
        run_ab(wl_l2, nhc, "L1->L2 (d_child=16)", n_runs=5)

    # L3 benchmark: d_child=32 (main expected impact)
    print("\nGenerating L2 survivors as L3 parents...")
    parents, nhc = generate_parents(target_level=3)
    print(f"  Got {len(parents)} parents at d={parents.shape[1]}")

    wl_l3 = prepare_workload(parents, nhc, n_parents=200,
                             min_children=100)
    if wl_l3:
        run_ab(wl_l3, nhc, "L2->L3 (d_child=32)", n_runs=5)
    else:
        print("  No valid L3 parents found.")


if __name__ == '__main__':
    main()
