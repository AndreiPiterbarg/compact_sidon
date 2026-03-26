"""Empirical evaluation of the CDCL-inspired skip optimization.

Single instrumented kernel measures surplus, |DeltaW|, and skip hit rates.
Timing comparison uses the production kernel as baseline.
"""
import sys
import os
import time
import math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction


def compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child):
    d_parent = len(parent_int)
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
    x_cap = min(x_cap, x_cap_cs, m)
    x_cap = max(x_cap, 0)
    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total = 1
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
        total *= (hi - lo + 1)
    return lo_arr, hi_arr, total


@njit(cache=False)
def _gray_measure(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf,
                  stats):
    """Gray code kernel that measures skip opportunity statistics.

    stats is a flat int64 array:
      [0] n_visited
      [1] n_pruned
      [2] n_skip_exact   (surplus > |exact DeltaW|)
      [3] n_skip_loose   (surplus > 12m+4)
      [4] n_survivors
      [5] sum_surplus
      [6] sum_abs_deltaW
      [7] n_deltaW_computed  (how many times we computed DeltaW)
      [8] max_surplus
      [9] max_abs_deltaW
      [10] n_skip_exact_conservative (surplus > |DeltaW| + thresh_change_bound)
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    loose_bound = np.int64(12 * m + 4)

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
        stats[0] += 1  # n_visited

        # --- Pruning test ---
        quick_killed = False
        surplus = np.int64(0)
        kill_ell = np.int32(0)
        kill_s = np.int32(0)

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
                surplus = ws_qc - dyn_it_qc
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
                        surplus = ws - dyn_it
                        kill_ell = np.int32(ell)
                        kill_s = np.int32(s_lo)
                        qc_ell = kill_ell
                        qc_s = kill_s
                        qc_W_int = W_int
                        break
            if not pruned:
                # Survivor
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
                stats[4] += 1

                # Jump to advance
                j_gc = gc_focus[0]
                if j_gc == n_active:
                    break
                gc_focus[0] = 0
                pos = active_pos[j_gc]
                gc_a[j_gc] += gc_dir[j_gc]
                cursor[pos] = lo_arr[pos] + gc_a[j_gc]
                if gc_a[j_gc] == 0 or gc_a[j_gc] == radix[j_gc] - 1:
                    gc_dir[j_gc] = -gc_dir[j_gc]
                    gc_focus[j_gc] = gc_focus[j_gc + 1]
                    gc_focus[j_gc + 1] = j_gc + 1
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
                continue

        # --- Pruned. Measure surplus and DeltaW. ---
        stats[1] += 1
        if quick_killed:
            qc_ell = kill_ell
            qc_s = kill_s

        stats[5] += surplus
        if surplus > stats[8]:
            stats[8] = surplus

        if surplus > loose_bound:
            stats[3] += 1

        # Compute exact DeltaW for next step on killing window
        next_j = gc_focus[0]
        if next_j < n_active:
            next_pos = active_pos[next_j]
            next_dir_val = gc_dir[next_j]
            k1p = 2 * next_pos
            k2p = k1p + 1
            old1p = child[k1p]
            old2p = child[k2p]
            new1p = old1p + next_dir_val
            new2p = old2p - next_dir_val
            d1p = np.int64(next_dir_val)
            d2p = np.int64(-next_dir_val)

            wlo = int(kill_s)
            whi = int(kill_s + kill_ell - 2)
            dW = np.int64(0)

            # Self-terms
            if wlo <= 2 * k1p <= whi:
                dW += np.int64(new1p) * np.int64(new1p) - np.int64(old1p) * np.int64(old1p)
            if wlo <= 2 * k2p <= whi:
                dW += np.int64(new2p) * np.int64(new2p) - np.int64(old2p) * np.int64(old2p)
            if wlo <= k1p + k2p <= whi:
                dW += np.int64(2) * (np.int64(new1p) * np.int64(new2p) - np.int64(old1p) * np.int64(old2p))

            # Cross-terms (only bins j where k1p+j or k2p+j in window)
            j_lo = min(wlo - k1p, wlo - k2p)
            j_hi = max(whi - k1p, whi - k2p)
            j_lo = max(j_lo, 0)
            j_hi = min(j_hi, d_child - 1)
            for jj in range(j_lo, j_hi + 1):
                if jj == k1p or jj == k2p:
                    continue
                cjp = np.int64(child[jj])
                if cjp == 0:
                    continue
                if wlo <= k1p + jj <= whi:
                    dW += np.int64(2) * d1p * cjp
                if wlo <= k2p + jj <= whi:
                    dW += np.int64(2) * d2p * cjp

            abs_dW = dW if dW >= 0 else -dW
            stats[6] += abs_dW
            stats[7] += 1
            if abs_dW > stats[9]:
                stats[9] = abs_dW

            # Exact skip: surplus > |DeltaW|
            if surplus > abs_dW:
                stats[2] += 1

            # Conservative skip: surplus > |DeltaW| + threshold change bound
            ell_idx_p = kill_ell - 2
            thresh_change = np.int64(two_ell_arr[ell_idx_p] * 2.0 + 1.0)
            if surplus > abs_dW + thresh_change:
                stats[10] += 1

        # --- Gray code advance ---
        j_gc = gc_focus[0]
        if j_gc == n_active:
            break
        gc_focus[0] = 0
        pos = active_pos[j_gc]
        gc_a[j_gc] += gc_dir[j_gc]
        cursor[pos] = lo_arr[pos] + gc_a[j_gc]
        if gc_a[j_gc] == 0 or gc_a[j_gc] == radix[j_gc] - 1:
            gc_dir[j_gc] = -gc_dir[j_gc]
            gc_focus[j_gc] = gc_focus[j_gc + 1]
            gc_focus[j_gc + 1] = j_gc + 1

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

    return n_surv


def generate_parents(n_half, m, c_target, target_level):
    from cpu.run_cascade import run_level0, process_parent_fused
    print(f"  Generating (n_half={n_half}, m={m}, c_target={c_target})...", flush=True)
    result = run_level0(n_half, m, c_target, verbose=False)
    survivors = result['survivors']
    print(f"  L0: {len(survivors)} survivors", flush=True)
    if target_level == 1:
        return survivors
    for level in range(1, target_level):
        d_parent = survivors.shape[1]
        nhc = d_parent
        all_surv = []
        for i in range(len(survivors)):
            surv, _ = process_parent_fused(survivors[i], m, c_target, nhc)
            if len(surv) > 0:
                all_surv.append(surv)
        if all_surv:
            survivors = np.vstack(all_surv)
            survivors = np.unique(survivors, axis=0)
        else:
            survivors = np.empty((0, 2 * d_parent), dtype=np.int32)
        print(f"  L{level}: {len(survivors)} survivors", flush=True)
        if len(survivors) == 0:
            break
    return survivors


def run_benchmark():
    m = 20
    c_target = 1.4
    n_half = 2

    print("=" * 72, flush=True)
    print("  CDCL SKIP OPTIMIZATION — EMPIRICAL EVALUATION", flush=True)
    print("=" * 72, flush=True)

    print("\n--- Generating parents ---", flush=True)
    l0_parents = generate_parents(n_half, m, c_target, 1)
    l1_parents = generate_parents(n_half, m, c_target, 2)

    print("\nJIT warmup...", flush=True)
    dp = np.array([10, 10], dtype=np.int32)
    dl = np.array([0, 0], dtype=np.int32)
    dh = np.array([5, 5], dtype=np.int32)
    db = np.empty((100, 4), dtype=np.int32)
    st = np.zeros(11, dtype=np.int64)
    _gray_measure(dp, 2, 20, 1.4, dl, dh, db, st)
    print("JIT done.\n", flush=True)

    # Import production kernel for timing baseline
    from cpu.run_cascade import _fused_generate_and_prune_gray

    configs = [
        ('L0->L1 (d=4->8)', l0_parents, l0_parents.shape[1]),
        ('L1->L2 (d=8->16)', l1_parents, l1_parents.shape[1]),
    ]

    for name, parents, n_half_child in configs:
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        max_p = min(len(parents), 2000)
        sample = parents[:max_p]

        print(f"\n{'=' * 72}", flush=True)
        print(f"  {name}: {len(sample)} parents, m={m}", flush=True)
        print(f"{'=' * 72}", flush=True)

        # --- Phase 1: Statistics ---
        print("\n  Phase 1: Measuring skip opportunities...", flush=True)
        stats = np.zeros(11, dtype=np.int64)

        # Also collect the production kernel survivor count for validation
        prod_surv_total = 0

        t0 = time.perf_counter()
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

            ns = _gray_measure(parent, n_half_child, m, c_target,
                               lo_arr, hi_arr, out_buf, stats)

            # Also run production kernel for comparison
            out_buf2 = np.empty((buf_cap, d_child), dtype=np.int32)
            ns_prod, _ = _fused_generate_and_prune_gray(
                parent, n_half_child, m, c_target, lo_arr, hi_arr, out_buf2)

            if ns != ns_prod:
                print(f"  WARNING: measurement kernel disagreement at parent {pidx}: "
                      f"measure={ns}, prod={ns_prod}", flush=True)
            prod_surv_total += ns_prod

        t_measure = time.perf_counter() - t0

        n_visited = stats[0]
        n_pruned = stats[1]
        n_skip_exact = stats[2]
        n_skip_loose = stats[3]
        n_survivors = stats[4]
        sum_surplus = stats[5]
        sum_abs_dW = stats[6]
        n_dW = stats[7]
        max_surplus = stats[8]
        max_abs_dW = stats[9]
        n_skip_conservative = stats[10]

        if n_pruned > 0:
            print(f"\n  Total children visited:    {n_visited:>14,}", flush=True)
            print(f"  Pruned:                    {n_pruned:>14,}", flush=True)
            print(f"  Survivors:                 {n_survivors:>14,}", flush=True)
            print(f"  Prod kernel survivors:     {prod_surv_total:>14,}", flush=True)
            print(f"  Survival rate:             {n_survivors/n_visited*100:>13.4f}%", flush=True)
            print(f"  Measurement time:          {t_measure:>13.1f}s", flush=True)

            print(f"\n  SURPLUS STATS:", flush=True)
            avg_surplus = sum_surplus / n_pruned
            print(f"  Mean surplus:              {avg_surplus:>14.1f}", flush=True)
            print(f"  Max surplus:               {max_surplus:>14}", flush=True)

            if n_dW > 0:
                avg_dW = sum_abs_dW / n_dW
                print(f"\n  |DeltaW| STATS ({n_dW:,} computed):", flush=True)
                print(f"  Mean |DeltaW|:             {avg_dW:>14.1f}", flush=True)
                print(f"  Max |DeltaW|:              {max_abs_dW:>14}", flush=True)

            print(f"\n  SKIP RATES (of {n_pruned:,} pruned children):", flush=True)
            print(f"  Exact (surplus > |dW|):      {n_skip_exact:>10,}  = {n_skip_exact/n_pruned*100:.2f}%", flush=True)
            print(f"  Conservative (+thresh):      {n_skip_conservative:>10,}  = {n_skip_conservative/n_pruned*100:.2f}%", flush=True)
            print(f"  Loose (surplus > {12*m+4}):     {n_skip_loose:>10,}  = {n_skip_loose/n_pruned*100:.2f}%", flush=True)

            # Work saved estimate
            # Each skipped child saves: the quick-check O(ell) or full-scan O(d^2) test
            # But NOT the incremental O(d) update (still needed)
            # Quick-check cost ~ ell (say avg 4-8 ops)
            # The DeltaW computation itself costs O(ell) ~ same as quick-check
            # So net savings per skipped child = quick_check_cost - deltaW_cost ≈ 0
            # The ONLY real savings is when we can also skip the incremental update
            qc_cost = 8  # avg ops for quick-check
            dW_cost = 8  # avg ops for DeltaW computation (similar window scan)
            update_cost = 2 * d_child  # incremental conv update
            full_scan_cost = d_child * d_child  # full window scan (worst case)

            # Model: each child costs (update + test)
            # Test cost: 80% quick-check + 20% full scan (typical)
            qc_rate = 0.85
            avg_test_cost = qc_rate * qc_cost + (1 - qc_rate) * full_scan_cost
            per_child_baseline = update_cost + avg_test_cost

            # With skip: skipped children still need update + DeltaW check
            # Non-skipped children: same as baseline
            skip_rate = n_skip_conservative / n_pruned if n_pruned > 0 else 0
            per_child_skip = (skip_rate * (update_cost + dW_cost) +
                              (1 - skip_rate) * (update_cost + avg_test_cost + dW_cost))
            # Note: non-skipped children ALSO pay the dW_cost for trying the skip

            if per_child_skip > 0:
                theoretical_speedup = per_child_baseline / per_child_skip
                print(f"\n  THEORETICAL SPEEDUP ESTIMATE:", flush=True)
                print(f"  d_child={d_child}, update_cost={update_cost}, avg_test_cost={avg_test_cost:.0f}", flush=True)
                print(f"  Skip rate (conservative):  {skip_rate*100:.1f}%", flush=True)
                print(f"  Per-child baseline cost:   {per_child_baseline:.0f} ops", flush=True)
                print(f"  Per-child skip cost:       {per_child_skip:.0f} ops", flush=True)
                print(f"  Theoretical speedup:       {theoretical_speedup:.3f}x", flush=True)
                if theoretical_speedup < 1.0:
                    print(f"  *** SLOWDOWN: DeltaW overhead exceeds savings ***", flush=True)

        # --- Phase 2: Timing ---
        print(f"\n  Phase 2: Timing (3 reps)...", flush=True)
        for rep in range(3):
            t0 = time.perf_counter()
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
                _fused_generate_and_prune_gray(parent, n_half_child, m, c_target,
                                                lo_arr, hi_arr, out_buf)
            t_prod = time.perf_counter() - t0

            t0 = time.perf_counter()
            st2 = np.zeros(11, dtype=np.int64)
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
                _gray_measure(parent, n_half_child, m, c_target,
                              lo_arr, hi_arr, out_buf, st2)
            t_meas = time.perf_counter() - t0

            overhead = t_meas / t_prod if t_prod > 0 else float('inf')
            print(f"  Rep {rep+1}: prod={t_prod:.3f}s  measure={t_meas:.3f}s  "
                  f"overhead={overhead:.2f}x", flush=True)

    print(f"\n{'=' * 72}", flush=True)
    print("  DONE", flush=True)
    print(f"{'=' * 72}", flush=True)


if __name__ == '__main__':
    run_benchmark()
