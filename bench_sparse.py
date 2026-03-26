"""Empirical measurement of quick-check hit rates and sparsity for the
sparse convolution proposal evaluation.

Generates real parents via the cascade (L0->L1->L2), then measures:
1. Quick-check hit rate at L3 (d_child=32) with instrumented Gray code kernel
2. nnz distribution of children during enumeration
3. Work distribution: cross-term vs quick-check vs full-scan iterations
"""
import sys
import os
import time
import math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction


@njit(cache=False)
def _gray_instrumented(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
    """Gray code kernel with instrumentation counters."""
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
        return (0, np.int64(0), np.int64(0), np.int64(0),
                np.int64(0), np.int64(0), np.int64(0), np.int64(0))

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1

    n_visited = np.int64(0)
    n_qc_hit = np.int64(0)
    n_full_scan = np.int64(0)
    total_nnz = np.int64(0)
    total_cross_iters = np.int64(0)
    total_qc_iters = np.int64(0)
    total_fullscan_iters = np.int64(0)

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
        n_visited += 1
        nnz = np.int64(0)
        for i in range(d_child):
            if child[i] != 0:
                nnz += 1
        total_nnz += nnz

        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
                total_qc_iters += 1
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + two_ell_arr[ell_idx_qc] * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                n_qc_hit += 1

        if not quick_killed:
            n_full_scan += 1
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
                    total_fullscan_iters += 1
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
                        total_fullscan_iters += 1
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
                total_cross_iters += 1
        for jj in range(k2 + 1, d_child):
            cj = np.int32(child[jj])
            if cj != 0:
                raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
                total_cross_iters += 1

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

    return (n_surv, n_visited, n_qc_hit, n_full_scan,
            total_nnz, total_cross_iters, total_qc_iters, total_fullscan_iters)


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


def generate_parents_via_cascade(n_half, m, c_target, target_level):
    """Run cascade from scratch to generate real parents at target_level."""
    # Import the cascade functions
    from cpu.run_cascade import run_level0, process_parent_fused

    print(f"  Generating parents via cascade (n_half={n_half}, m={m}, c_target={c_target})...")

    # L0
    t0 = time.perf_counter()
    result = run_level0(n_half, m, c_target, verbose=False)
    survivors = result['survivors']
    print(f"  L0: {len(survivors)} survivors ({time.perf_counter()-t0:.1f}s)")

    if target_level == 1:
        return survivors

    # L1+
    for level in range(1, target_level):
        d_parent = survivors.shape[1]
        n_half_child = d_parent  # n_half_child = d_parent for cascade
        d_child = 2 * d_parent
        t0 = time.perf_counter()
        all_surv = []
        for i in range(len(survivors)):
            surv, _ = process_parent_fused(survivors[i], m, c_target, n_half_child)
            if len(surv) > 0:
                all_surv.append(surv)
        if all_surv:
            survivors = np.vstack(all_surv)
            # Deduplicate
            survivors = np.unique(survivors, axis=0)
        else:
            survivors = np.empty((0, d_child), dtype=np.int32)
        elapsed = time.perf_counter() - t0
        print(f"  L{level}: {len(survivors)} survivors ({elapsed:.1f}s)")

        if len(survivors) == 0:
            break

    return survivors


def run_benchmark():
    m = 20
    c_target = 1.4
    n_half = 2

    print("=" * 70)
    print("SPARSE CONVOLUTION PROPOSAL -- EMPIRICAL EVALUATION")
    print("=" * 70)

    # L4 survivors nnz analysis
    try:
        l4 = np.load('data/checkpoint_L4_survivors.npy')
        if l4.shape[0] > 0:
            nnz = np.count_nonzero(l4, axis=1)
            print(f"\n--- L4 survivors nnz analysis (d={l4.shape[1]}) ---")
            print(f"  Rows: {l4.shape[0]}")
            print(f"  nnz: min={nnz.min()}, max={nnz.max()}, "
                  f"mean={nnz.mean():.1f}, median={np.median(nnz):.0f}")
            print(f"  sparsity: {nnz.mean():.1f}/{l4.shape[1]} = "
                  f"{nnz.mean()/l4.shape[1]:.1%} nonzero")
    except Exception as e:
        print(f"  L4 survivors not available: {e}")

    # Generate L1 parents (L0 survivors) via cascade
    print("\n--- Generating real parents via cascade ---")
    l0_survivors = generate_parents_via_cascade(n_half, m, c_target, target_level=1)
    l1_survivors = generate_parents_via_cascade(n_half, m, c_target, target_level=2)

    # JIT warmup
    print("\nWarming up instrumented kernel...")
    dummy_parent = np.array([10, 10], dtype=np.int32)
    dummy_lo = np.array([0, 0], dtype=np.int32)
    dummy_hi = np.array([5, 5], dtype=np.int32)
    dummy_buf = np.empty((100, 4), dtype=np.int32)
    _gray_instrumented(dummy_parent, 2, 20, 1.4, dummy_lo, dummy_hi, dummy_buf)
    print("JIT warm.\n")

    # Test configurations: parent checkpoint -> child level
    configs = []

    if len(l0_survivors) > 0:
        configs.append(('L0->L1', l0_survivors, l0_survivors.shape[1]))

    if len(l1_survivors) > 0:
        configs.append(('L1->L2', l1_survivors, l1_survivors.shape[1]))

    # For L2->L3: generate L2 parents (this takes ~30s)
    print("--- Generating L2 parents (may take ~30s) ---")
    l2_survivors = generate_parents_via_cascade(n_half, m, c_target, target_level=3)
    if len(l2_survivors) > 0:
        configs.append(('L2->L3', l2_survivors, l2_survivors.shape[1]))

    for name, parents, n_half_child in configs:
        d_parent = parents.shape[1]
        d_child = 2 * d_parent

        print(f"\n{'=' * 70}")
        print(f"  {name}: d_parent={d_parent}, d_child={d_child}, m={m}")
        print(f"  {parents.shape[0]:,} parents available")

        # Sample parents (limit to manageable count for L2->L3)
        if name == 'L2->L3':
            max_parents = min(parents.shape[0], 500)
        elif name == 'L1->L2':
            max_parents = min(parents.shape[0], 2000)
        else:
            max_parents = parents.shape[0]

        # Take evenly spaced sample
        if max_parents < len(parents):
            indices = np.linspace(0, len(parents)-1, max_parents, dtype=int)
            sample = parents[indices]
        else:
            sample = parents
        print(f"  Testing {len(sample):,} parents")
        print(f"{'=' * 70}")

        agg_visited = np.int64(0)
        agg_qc_hit = np.int64(0)
        agg_full_scan = np.int64(0)
        agg_surv = 0
        agg_nnz = np.int64(0)
        agg_cross_iters = np.int64(0)
        agg_qc_iters = np.int64(0)
        agg_fullscan_iters = np.int64(0)
        n_processed = 0
        total_children = 0
        n_asym_pruned = 0

        t0 = time.perf_counter()

        for pidx in range(len(sample)):
            parent = sample[pidx]
            result = compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                n_asym_pruned += 1
                continue
            lo_arr, hi_arr, n_children = result
            if n_children <= 1:
                continue
            total_children += n_children

            buf_cap = min(n_children, 500_000)
            out_buf = np.empty((buf_cap, d_child), dtype=np.int32)

            res = _gray_instrumented(parent, n_half_child, m, c_target,
                                     lo_arr, hi_arr, out_buf)
            ns, nv, nqc, nfs, tnnz, tcross, tqci, tfsi = res

            if nv == 0:
                n_asym_pruned += 1
                continue

            agg_surv += ns
            agg_visited += nv
            agg_qc_hit += nqc
            agg_full_scan += nfs
            agg_nnz += tnnz
            agg_cross_iters += tcross
            agg_qc_iters += tqci
            agg_fullscan_iters += tfsi
            n_processed += 1

            # Progress for long runs
            if n_processed % 200 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    [{n_processed}/{len(sample)}] {elapsed:.1f}s, "
                      f"{agg_visited:,} children visited")

        elapsed = time.perf_counter() - t0

        if agg_visited == 0:
            print("  All parents pruned by asymmetry.")
            continue

        qc_rate = float(agg_qc_hit) / float(agg_visited) * 100
        avg_nnz = float(agg_nnz) / float(agg_visited)
        n_steps = int(agg_visited) - n_processed  # subtract first-child-per-parent (no cross update)
        avg_cross = float(agg_cross_iters) / float(n_steps) if n_steps > 0 else 0

        print(f"\n  RESULTS ({n_processed} parents, {elapsed:.1f}s):")
        print(f"  -------------------------------------------------")
        print(f"  Asymmetry-pruned parents:   {n_asym_pruned:>15,}")
        print(f"  Total children visited:     {int(agg_visited):>15,}")
        print(f"  Cartesian product size:     {total_children:>15,}")
        print(f"  Survivors:                  {agg_surv:>15,}")
        if agg_visited > 0:
            print(f"  Survival rate:              {agg_surv/float(agg_visited)*100:>14.6f}%")
        print()
        print(f"  Quick-check hits:           {int(agg_qc_hit):>15,}")
        print(f"  Full window scans:          {int(agg_full_scan):>15,}")
        print(f"  *** QUICK-CHECK HIT RATE:   {qc_rate:>14.2f}% ***")
        print()
        print(f"  Avg nnz per child:          {avg_nnz:>14.1f} / {d_child}")
        print(f"  Avg nnz ratio:              {avg_nnz/d_child:>14.1%}")
        print(f"  Avg nonzero cross iters:    {avg_cross:>14.1f} / {d_child - 2}")
        print(f"  Cross iter reduction:       {(d_child-2)/avg_cross:>14.2f}x" if avg_cross > 0 else "")
        print()

        total_work = int(agg_cross_iters) + int(agg_qc_iters) + int(agg_fullscan_iters)
        if total_work > 0:
            print(f"  --- WORK DISTRIBUTION (iterations) ---")
            print(f"  Cross-term updates:   {int(agg_cross_iters):>15,}  ({float(agg_cross_iters)/total_work*100:5.1f}%)")
            print(f"  Quick-check sums:     {int(agg_qc_iters):>15,}  ({float(agg_qc_iters)/total_work*100:5.1f}%)")
            print(f"  Full-scan window ops: {int(agg_fullscan_iters):>15,}  ({float(agg_fullscan_iters)/total_work*100:5.1f}%)")
            print(f"  TOTAL:                {total_work:>15,}")
            print()

        if n_steps > 0 and avg_cross > 0:
            current_cross_total = n_steps * (d_child - 2)
            sparse_cross_total = int(agg_cross_iters)
            cross_speedup = current_cross_total / sparse_cross_total

            current_total_est = current_cross_total + int(agg_qc_iters) + int(agg_fullscan_iters)
            sparse_total_est = sparse_cross_total + int(agg_qc_iters) + int(agg_fullscan_iters)
            overall_speedup = current_total_est / sparse_total_est

            print(f"  --- AMDAHL'S LAW SPEEDUP ESTIMATE ---")
            print(f"  Cross-term loop speedup:    {cross_speedup:>13.2f}x")
            print(f"  Cross fraction of total:    {current_cross_total/current_total_est*100:>12.1f}%")
            print(f"  *** OVERALL SPEEDUP:        {overall_speedup:>12.2f}x ***")


if __name__ == '__main__':
    run_benchmark()
