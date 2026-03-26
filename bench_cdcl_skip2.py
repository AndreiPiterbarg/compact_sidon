"""Phase 2: L2->L3 benchmark + batch-skip analysis.

Measures same stats at d=32 (production bottleneck), plus:
- Position-0 sweep lengths and batch-skippability
- Amdahl's law with/without incremental update savings
"""
import sys, os, time, math
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
    lo = np.empty(d_parent, dtype=np.int32)
    hi = np.empty(d_parent, dtype=np.int32)
    total = 1
    for i in range(d_parent):
        b_i = int(parent_int[i])
        l = max(0, b_i - x_cap)
        h = min(b_i, x_cap)
        if l > h:
            return None
        lo[i] = l
        hi[i] = h
        total *= (h - l + 1)
    return lo, hi, total


@njit(cache=False)
def _gray_measure_v2(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf,
                     stats):
    """Stats kernel measuring surplus, DeltaW, and position-0 sweep batching.

    stats[0] n_visited
    stats[1] n_pruned
    stats[2] n_skip_exact
    stats[3] n_skip_conservative
    stats[4] n_survivors
    stats[5] sum_surplus
    stats[6] sum_abs_deltaW
    stats[7] n_dW_computed
    stats[8] max_surplus
    stats[9] max_abs_deltaW
    stats[10] n_pos0_sweeps           (number of position-0 sweeps started)
    stats[11] n_pos0_fully_pruned     (sweeps where ALL children were pruned)
    stats[12] sum_pos0_sweep_len      (total children in position-0 sweeps)
    stats[13] sum_pos0_pruned_in_sweep
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    if np.float64(left_sum_parent) / m_d >= threshold_asym or \
       np.float64(left_sum_parent) / m_d <= 1.0 - threshold_asym:
        return 0

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

    # Position-0 sweep tracking
    in_pos0_sweep = False
    sweep_len = np.int64(0)
    sweep_pruned = np.int64(0)

    while True:
        stats[0] += 1

        # Track position-0 sweeps
        next_j = gc_focus[0]
        is_pos0 = (next_j == 0) if next_j < n_active else False
        if is_pos0 and not in_pos0_sweep:
            in_pos0_sweep = True
            sweep_len = 1
            sweep_pruned = 0
            stats[10] += 1
        elif is_pos0 and in_pos0_sweep:
            sweep_len += 1
        elif not is_pos0 and in_pos0_sweep:
            # Sweep ended
            stats[12] += sweep_len
            stats[13] += sweep_pruned
            if sweep_pruned == sweep_len:
                stats[11] += 1
            in_pos0_sweep = False

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

                # Advance Gray code
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
                k1 = 2 * pos; k2 = k1 + 1
                old1 = np.int32(child[k1]); old2 = np.int32(child[k2])
                child[k1] = cursor[pos]
                child[k2] = parent_int[pos] - cursor[pos]
                new1 = np.int32(child[k1]); new2 = np.int32(child[k2])
                d1 = new1 - old1; d2 = new2 - old2
                raw_conv[2*k1] += new1*new1 - old1*old1
                raw_conv[2*k2] += new2*new2 - old2*old2
                raw_conv[k1+k2] += np.int32(2)*(new1*new2 - old1*old2)
                for jj in range(k1):
                    cj = np.int32(child[jj])
                    if cj != 0:
                        raw_conv[k1+jj] += np.int32(2)*d1*cj
                        raw_conv[k2+jj] += np.int32(2)*d2*cj
                for jj in range(k2+1, d_child):
                    cj = np.int32(child[jj])
                    if cj != 0:
                        raw_conv[k1+jj] += np.int32(2)*d1*cj
                        raw_conv[k2+jj] += np.int32(2)*d2*cj
                if qc_ell > 0:
                    ql = qc_s-(d_child-1); ql = max(ql,0)
                    qh = qc_s+qc_ell-2; qh = min(qh,d_child-1)
                    if ql<=k1<=qh: qc_W_int += np.int64(d1)
                    if ql<=k2<=qh: qc_W_int += np.int64(d2)
                continue

        # --- Pruned ---
        stats[1] += 1
        if quick_killed:
            qc_ell = kill_ell; qc_s = kill_s

        stats[5] += surplus
        if surplus > stats[8]:
            stats[8] = surplus

        if in_pos0_sweep:
            sweep_pruned += 1

        # Compute DeltaW for next step
        nj = gc_focus[0]
        if nj < n_active:
            np_ = active_pos[nj]
            nd = gc_dir[nj]
            k1p = 2*np_; k2p = k1p+1
            o1 = child[k1p]; o2 = child[k2p]
            n1 = o1+nd; n2 = o2-nd
            wlo = int(kill_s)
            whi = int(kill_s+kill_ell-2)
            dW = np.int64(0)
            if wlo<=2*k1p<=whi:
                dW += np.int64(n1)*np.int64(n1) - np.int64(o1)*np.int64(o1)
            if wlo<=2*k2p<=whi:
                dW += np.int64(n2)*np.int64(n2) - np.int64(o2)*np.int64(o2)
            if wlo<=k1p+k2p<=whi:
                dW += np.int64(2)*(np.int64(n1)*np.int64(n2)-np.int64(o1)*np.int64(o2))
            jl = min(wlo-k1p, wlo-k2p); jl = max(jl,0)
            jh = max(whi-k1p, whi-k2p); jh = min(jh, d_child-1)
            for jj in range(jl,jh+1):
                if jj==k1p or jj==k2p: continue
                cjp = np.int64(child[jj])
                if cjp==0: continue
                if wlo<=k1p+jj<=whi: dW += np.int64(2)*np.int64(nd)*cjp
                if wlo<=k2p+jj<=whi: dW += np.int64(2)*np.int64(-nd)*cjp
            abs_dW = dW if dW>=0 else -dW
            stats[6] += abs_dW
            stats[7] += 1
            if abs_dW > stats[9]: stats[9] = abs_dW
            if surplus > abs_dW: stats[2] += 1
            tidx = kill_ell - 2
            tc = np.int64(two_ell_arr[tidx]*2.0+1.0)
            if surplus > abs_dW+tc: stats[3] += 1

        # Advance
        j_gc = gc_focus[0]
        if j_gc == n_active:
            if in_pos0_sweep:
                stats[12] += sweep_len
                stats[13] += sweep_pruned
                if sweep_pruned == sweep_len: stats[11] += 1
            break
        gc_focus[0] = 0
        pos = active_pos[j_gc]
        gc_a[j_gc] += gc_dir[j_gc]
        cursor[pos] = lo_arr[pos] + gc_a[j_gc]
        if gc_a[j_gc] == 0 or gc_a[j_gc] == radix[j_gc] - 1:
            gc_dir[j_gc] = -gc_dir[j_gc]
            gc_focus[j_gc] = gc_focus[j_gc + 1]
            gc_focus[j_gc + 1] = j_gc + 1
        k1 = 2*pos; k2 = k1+1
        old1 = np.int32(child[k1]); old2 = np.int32(child[k2])
        child[k1] = cursor[pos]; child[k2] = parent_int[pos]-cursor[pos]
        new1 = np.int32(child[k1]); new2 = np.int32(child[k2])
        d1 = new1-old1; d2 = new2-old2
        raw_conv[2*k1] += new1*new1-old1*old1
        raw_conv[2*k2] += new2*new2-old2*old2
        raw_conv[k1+k2] += np.int32(2)*(new1*new2-old1*old2)
        for jj in range(k1):
            cj = np.int32(child[jj])
            if cj!=0:
                raw_conv[k1+jj] += np.int32(2)*d1*cj
                raw_conv[k2+jj] += np.int32(2)*d2*cj
        for jj in range(k2+1,d_child):
            cj = np.int32(child[jj])
            if cj!=0:
                raw_conv[k1+jj] += np.int32(2)*d1*cj
                raw_conv[k2+jj] += np.int32(2)*d2*cj
        if qc_ell > 0:
            ql = qc_s-(d_child-1); ql = max(ql,0)
            qh = qc_s+qc_ell-2; qh = min(qh,d_child-1)
            if ql<=k1<=qh: qc_W_int += np.int64(d1)
            if ql<=k2<=qh: qc_W_int += np.int64(d2)
    return n_surv


def generate_parents(n_half, m, c_target, target_level):
    from cpu.run_cascade import run_level0, process_parent_fused
    print(f"  Generating (n_half={n_half}, m={m}, c={c_target})...", flush=True)
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
        survivors = np.vstack(all_surv) if all_surv else np.empty((0,2*d_parent),dtype=np.int32)
        survivors = np.unique(survivors, axis=0) if len(survivors) > 0 else survivors
        print(f"  L{level}: {len(survivors)} survivors", flush=True)
        if len(survivors) == 0: break
    return survivors


def main():
    m, c_target, n_half = 20, 1.4, 2
    print("="*72, flush=True)
    print("  CDCL SKIP — L2->L3 (d=32) + BATCH ANALYSIS", flush=True)
    print("="*72, flush=True)

    print("\n--- Generating L2 parents ---", flush=True)
    l2 = generate_parents(n_half, m, c_target, 3)
    if len(l2) == 0:
        print("No L2 survivors!"); return

    # Sample — 200 parents for reasonable runtime
    max_p = min(len(l2), 200)
    sample = l2[np.linspace(0, len(l2)-1, max_p, dtype=int)]

    print(f"\n  Using {len(sample)} L2 parents (of {len(l2):,})", flush=True)
    print(f"  d_parent=16, d_child=32, m=20", flush=True)

    print("\n  JIT warmup...", flush=True)
    dp = np.array([5,5,5,5], dtype=np.int32)
    dl = np.array([0,0,0,0], dtype=np.int32)
    dh = np.array([3,3,3,3], dtype=np.int32)
    db = np.empty((1000,8), dtype=np.int32)
    st = np.zeros(14, dtype=np.int64)
    _gray_measure_v2(dp, 4, 20, 1.4, dl, dh, db, st)
    print("  JIT done.\n", flush=True)

    # Validation: compare survivor count vs production kernel
    from cpu.run_cascade import _fused_generate_and_prune_gray

    stats = np.zeros(14, dtype=np.int64)
    n_half_child = 16  # d_parent for L2->L3
    d_child = 32
    n_correct = 0

    t0 = time.perf_counter()
    for pidx in range(len(sample)):
        parent = sample[pidx]
        result = compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None: continue
        lo_arr, hi_arr, n_children = result
        if n_children <= 1: continue
        buf_cap = min(n_children, 200_000)
        out1 = np.empty((buf_cap, d_child), dtype=np.int32)
        out2 = np.empty((buf_cap, d_child), dtype=np.int32)

        ns1 = _gray_measure_v2(parent, n_half_child, m, c_target,
                                lo_arr, hi_arr, out1, stats)
        ns2, _ = _fused_generate_and_prune_gray(parent, n_half_child, m, c_target,
                                                  lo_arr, hi_arr, out2)
        if ns1 == ns2:
            n_correct += 1
        else:
            print(f"  MISMATCH parent {pidx}: measure={ns1} prod={ns2}", flush=True)

        if (pidx+1) % 50 == 0:
            elapsed = time.perf_counter()-t0
            print(f"  [{pidx+1}/{len(sample)}] {elapsed:.1f}s, "
                  f"{int(stats[0]):,} children", flush=True)

    elapsed = time.perf_counter()-t0
    print(f"\n  Processed in {elapsed:.1f}s", flush=True)
    print(f"  Correctness: {n_correct}/{len(sample)} match production kernel", flush=True)

    nv = stats[0]; np_ = stats[1]; ns = stats[4]
    n_skip = stats[2]; n_skip_c = stats[3]
    n_dW = stats[7]

    if np_ > 0:
        print(f"\n  RESULTS (d_child=32):", flush=True)
        print(f"  {'='*50}", flush=True)
        print(f"  Visited:    {nv:>14,}", flush=True)
        print(f"  Pruned:     {np_:>14,}", flush=True)
        print(f"  Survived:   {ns:>14,}", flush=True)
        print(f"  Survival:   {ns/nv*100:>13.6f}%", flush=True)

        print(f"\n  Surplus: mean={stats[5]/np_:.1f}, max={stats[8]}", flush=True)
        if n_dW > 0:
            print(f"  |DeltaW|: mean={stats[6]/n_dW:.1f}, max={stats[9]}", flush=True)

        print(f"\n  SKIP RATES:", flush=True)
        print(f"  Exact:        {n_skip:>12,} / {np_:,} = {n_skip/np_*100:.2f}%", flush=True)
        print(f"  Conservative: {n_skip_c:>12,} / {np_:,} = {n_skip_c/np_*100:.2f}%", flush=True)

        # Batch-skip stats
        n_sweeps = stats[10]
        n_fully = stats[11]
        sum_len = stats[12]
        sum_pruned = stats[13]

        if n_sweeps > 0:
            avg_len = sum_len / n_sweeps
            print(f"\n  POSITION-0 SWEEP ANALYSIS:", flush=True)
            print(f"  Total sweeps:     {n_sweeps:>10,}", flush=True)
            print(f"  Avg sweep length: {avg_len:>10.1f}", flush=True)
            print(f"  Fully pruned:     {n_fully:>10,} / {n_sweeps:,} = {n_fully/n_sweeps*100:.1f}%", flush=True)
            print(f"  Children in sweeps: {sum_len:>10,}", flush=True)
            print(f"  Pruned in sweeps:   {sum_pruned:>10,}", flush=True)
            if sum_len > 0:
                print(f"  Prune rate in sweeps: {sum_pruned/sum_len*100:.2f}%", flush=True)

            # Batch-skip estimate: skip fully-pruned sweeps saving (len-1)*O(d) updates
            if n_fully > 0:
                avg_full_len = sum_pruned / n_sweeps  # approx
                saved_updates = n_fully * (avg_len - 1)
                total_updates = nv
                print(f"\n  BATCH-SKIP POTENTIAL:", flush=True)
                print(f"  Updates saved by batch-skip: {saved_updates:>10,.0f} / {total_updates:,}", flush=True)
                print(f"  = {saved_updates/total_updates*100:.2f}% of total updates", flush=True)
                update_frac = 0.5  # fraction of per-child cost that is update
                potential_speedup = 1.0 / (1.0 - update_frac * saved_updates/total_updates)
                print(f"  Batch-skip speedup (est):    {potential_speedup:.3f}x", flush=True)

        # Amdahl's analysis
        print(f"\n  AMDAHL'S LAW (per-child cost breakdown at d=32):", flush=True)
        update = 2*d_child  # O(d) incremental update
        qc = 8             # quick-check O(ell)
        fullscan = d_child*d_child  # full window scan
        qc_rate = 0.88     # from measured data
        test = qc_rate*qc + (1-qc_rate)*fullscan
        total = update + test
        print(f"  Incremental update: {update:>4} ops ({update/total*100:.0f}%)", flush=True)
        print(f"  Test (avg):         {test:>4.0f} ops ({test/total*100:.0f}%)", flush=True)
        print(f"  Total:              {total:>4.0f} ops", flush=True)
        print(f"\n  Skip saves test cost only: max speedup = {total/(total-test):.2f}x (skip all tests)", flush=True)
        print(f"  Skip saves test+update:    max speedup = {total/0.01:.0f}x (impossible limit)", flush=True)
        print(f"  Realistic skip: saves {n_skip_c/np_*100:.0f}% of tests = {total/(total-n_skip_c/np_*test):.3f}x", flush=True)

    print(f"\n{'='*72}", flush=True)
    print("  DONE", flush=True)


if __name__ == '__main__':
    main()
