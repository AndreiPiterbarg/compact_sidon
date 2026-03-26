"""Direct timing: production kernel vs skip-aware kernel at L1->L2 and L2->L3.

The skip kernel pre-checks surplus > |DeltaW| + threshold_bound before
doing the full quick-check/scan. It still does incremental conv update
(required for correctness of subsequent children).
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
    x_cap = min(x_cap, x_cap_cs, m); x_cap = max(x_cap, 0)
    lo = np.empty(d_parent, dtype=np.int32)
    hi = np.empty(d_parent, dtype=np.int32)
    total = 1
    for i in range(d_parent):
        b = int(parent_int[i])
        l = max(0, b - x_cap); h = min(b, x_cap)
        if l > h: return None
        lo[i] = l; hi[i] = h; total *= (h - l + 1)
    return lo, hi, total


@njit(cache=False)
def _gray_skip_kernel(parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf):
    """Gray code with single-step skip: avoid test when surplus > |DeltaW|+guard.

    Still updates raw_conv every step (needed for correct future tests).
    Returns (n_survivors, n_skipped).
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
        return 0, np.int64(0)

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    n_skipped = np.int64(0)

    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent): cursor[i] = lo_arr[i]
    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    qc_ell = np.int32(0); qc_s = np.int32(0); qc_W_int = np.int64(0)

    for i in range(d_parent):
        child[2*i] = cursor[i]; child[2*i+1] = parent_int[i]-cursor[i]

    ell_count = 2*d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2*d_child+1):
        idx = ell-2
        dyn_base_ell_arr[idx] = dyn_base*np.float64(ell)*inv_4n
        two_ell_arr[idx] = 2.0*np.float64(ell)*inv_4n

    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2*d_child)
    for ell in range(2, phase1_end+1):
        ell_order[oi] = np.int32(ell); ell_used[ell-2] = 1; oi += 1
    for ell in (d_child, d_child+1, d_child-1, d_child+2, d_child-2,
                d_child*2, d_child+d_child//2, d_child//2):
        if 2<=ell<=2*d_child and ell_used[ell-2]==0:
            ell_order[oi] = np.int32(ell); ell_used[ell-2]=1; oi+=1
    for ell in range(2, 2*d_child+1):
        if ell_used[ell-2]==0: ell_order[oi]=np.int32(ell); oi+=1

    for k in range(conv_len): raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci!=0:
            raw_conv[2*i] += ci*ci
            for j in range(i+1, d_child):
                cj = np.int32(child[j])
                if cj!=0: raw_conv[i+j] += np.int32(2)*ci*cj

    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        r = hi_arr[i]-lo_arr[i]+1
        if r > 1: active_pos[n_active]=i; radix[n_active]=r; n_active+=1
    gc_a = np.zeros(n_active, dtype=np.int32)
    gc_dir = np.ones(n_active, dtype=np.int32)
    gc_focus = np.empty(n_active+1, dtype=np.int32)
    for i in range(n_active+1): gc_focus[i] = i

    # Skip state
    have_skip_prediction = False
    skip_surplus = np.int64(0)

    while True:
        # --- Check skip prediction ---
        if have_skip_prediction:
            have_skip_prediction = False
            # We predicted this child is pruned. Just verify by recomputing
            # the killing window sum from updated raw_conv.
            n_cv_sk = qc_ell - 1
            ws_sk = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_sk):
                ws_sk += np.int64(raw_conv[k])
            ell_idx_sk = qc_ell - 2
            # Recompute threshold with current W_int
            dyn_x_sk = dyn_base_ell_arr[ell_idx_sk] + two_ell_arr[ell_idx_sk] * np.float64(qc_W_int)
            dyn_it_sk = np.int64(dyn_x_sk * one_minus_4eps)

            if ws_sk > dyn_it_sk:
                # Skip confirmed
                n_skipped += 1
                new_surplus = ws_sk - dyn_it_sk

                # Try to predict NEXT skip
                nj = gc_focus[0]
                if nj < n_active:
                    np2 = active_pos[nj]; nd2 = gc_dir[nj]
                    k1p = 2*np2; k2p = k1p+1
                    wlo = int(qc_s); whi = int(qc_s+qc_ell-2)
                    o1 = child[k1p]; o2 = child[k2p]
                    n1 = o1+nd2; n2 = o2-nd2
                    dW = np.int64(0)
                    if wlo<=2*k1p<=whi: dW+=np.int64(n1)*np.int64(n1)-np.int64(o1)*np.int64(o1)
                    if wlo<=2*k2p<=whi: dW+=np.int64(n2)*np.int64(n2)-np.int64(o2)*np.int64(o2)
                    if wlo<=k1p+k2p<=whi: dW+=np.int64(2)*(np.int64(n1)*np.int64(n2)-np.int64(o1)*np.int64(o2))
                    jl=min(wlo-k1p,wlo-k2p); jl=max(jl,0)
                    jh=max(whi-k1p,whi-k2p); jh=min(jh,d_child-1)
                    for jj in range(jl,jh+1):
                        if jj==k1p or jj==k2p: continue
                        cjp=np.int64(child[jj])
                        if cjp==0: continue
                        if wlo<=k1p+jj<=whi: dW+=np.int64(2)*np.int64(nd2)*cjp
                        if wlo<=k2p+jj<=whi: dW+=np.int64(2)*np.int64(-nd2)*cjp
                    abs_dW = dW if dW>=0 else -dW
                    tc = np.int64(two_ell_arr[ell_idx_sk]*2.0+1.0)
                    if new_surplus > abs_dW + tc:
                        have_skip_prediction = True

                # Advance Gray code + update conv
                j_gc = gc_focus[0]
                if j_gc == n_active: break
                gc_focus[0] = 0
                pos = active_pos[j_gc]
                gc_a[j_gc] += gc_dir[j_gc]
                cursor[pos] = lo_arr[pos]+gc_a[j_gc]
                if gc_a[j_gc]==0 or gc_a[j_gc]==radix[j_gc]-1:
                    gc_dir[j_gc]=-gc_dir[j_gc]
                    gc_focus[j_gc]=gc_focus[j_gc+1]; gc_focus[j_gc+1]=j_gc+1
                k1=2*pos; k2=k1+1
                old1=np.int32(child[k1]); old2=np.int32(child[k2])
                child[k1]=cursor[pos]; child[k2]=parent_int[pos]-cursor[pos]
                new1=np.int32(child[k1]); new2=np.int32(child[k2])
                d1=new1-old1; d2=new2-old2
                raw_conv[2*k1]+=new1*new1-old1*old1
                raw_conv[2*k2]+=new2*new2-old2*old2
                raw_conv[k1+k2]+=np.int32(2)*(new1*new2-old1*old2)
                for jj in range(k1):
                    cj=np.int32(child[jj])
                    if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
                for jj in range(k2+1,d_child):
                    cj=np.int32(child[jj])
                    if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
                if qc_ell>0:
                    ql=qc_s-(d_child-1); ql=max(ql,0)
                    qh=qc_s+qc_ell-2; qh=min(qh,d_child-1)
                    if ql<=k1<=qh: qc_W_int+=np.int64(d1)
                    if ql<=k2<=qh: qc_W_int+=np.int64(d2)
                continue
            # Skip failed — fall through to normal

        # --- Normal test ---
        quick_killed = False
        surplus = np.int64(0)
        if qc_ell > 0:
            n_cv_qc = qc_ell-1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s+n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell-2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc]+two_ell_arr[ell_idx_qc]*np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc*one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True
                surplus = ws_qc - dyn_it_qc

        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i+1] = prefix_c[i]+np.int64(child[i])
            pruned = False
            for ell_oi in range(ell_count):
                if pruned: break
                ell = ell_order[ell_oi]
                n_cv=ell-1; ell_idx=ell-2
                dbn=dyn_base_ell_arr[ell_idx]; tli=two_ell_arr[ell_idx]
                n_windows=conv_len-n_cv+1
                ws=np.int64(0)
                for k in range(n_cv): ws+=np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo>0: ws+=np.int64(raw_conv[s_lo+n_cv-1])-np.int64(raw_conv[s_lo-1])
                    lb=s_lo-(d_child-1)
                    if lb<0: lb=0
                    hb=s_lo+ell-2
                    if hb>d_child-1: hb=d_child-1
                    W_int=prefix_c[hb+1]-prefix_c[lb]
                    dyn_x=dbn+tli*np.float64(W_int)
                    dyn_it=np.int64(dyn_x*one_minus_4eps)
                    if ws>dyn_it:
                        pruned=True; surplus=ws-dyn_it
                        qc_ell=np.int32(ell); qc_s=np.int32(s_lo); qc_W_int=W_int
                        break

            if not pruned:
                use_rev=False
                for i in range(d_child):
                    j_idx=d_child-1-i
                    if child[j_idx]<child[i]: use_rev=True; break
                    elif child[j_idx]>child[i]: break
                if n_surv<max_survivors:
                    if use_rev:
                        for i in range(d_child): out_buf[n_surv,i]=child[d_child-1-i]
                    else:
                        for i in range(d_child): out_buf[n_surv,i]=child[i]
                n_surv+=1

                j_gc=gc_focus[0]
                if j_gc==n_active: break
                gc_focus[0]=0; pos=active_pos[j_gc]
                gc_a[j_gc]+=gc_dir[j_gc]; cursor[pos]=lo_arr[pos]+gc_a[j_gc]
                if gc_a[j_gc]==0 or gc_a[j_gc]==radix[j_gc]-1:
                    gc_dir[j_gc]=-gc_dir[j_gc]; gc_focus[j_gc]=gc_focus[j_gc+1]; gc_focus[j_gc+1]=j_gc+1
                k1=2*pos; k2=k1+1
                old1=np.int32(child[k1]); old2=np.int32(child[k2])
                child[k1]=cursor[pos]; child[k2]=parent_int[pos]-cursor[pos]
                new1=np.int32(child[k1]); new2=np.int32(child[k2])
                d1=new1-old1; d2=new2-old2
                raw_conv[2*k1]+=new1*new1-old1*old1; raw_conv[2*k2]+=new2*new2-old2*old2
                raw_conv[k1+k2]+=np.int32(2)*(new1*new2-old1*old2)
                for jj in range(k1):
                    cj=np.int32(child[jj])
                    if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
                for jj in range(k2+1,d_child):
                    cj=np.int32(child[jj])
                    if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
                if qc_ell>0:
                    ql=qc_s-(d_child-1); ql=max(ql,0); qh=qc_s+qc_ell-2; qh=min(qh,d_child-1)
                    if ql<=k1<=qh: qc_W_int+=np.int64(d1)
                    if ql<=k2<=qh: qc_W_int+=np.int64(d2)
                continue

        # Pruned — try to predict next skip
        if surplus > 0:
            nj = gc_focus[0]
            if nj < n_active:
                np2=active_pos[nj]; nd2=gc_dir[nj]
                k1p=2*np2; k2p=k1p+1
                wlo=int(qc_s); whi=int(qc_s+qc_ell-2)
                o1=child[k1p]; o2=child[k2p]
                n1=o1+nd2; n2=o2-nd2
                dW=np.int64(0)
                if wlo<=2*k1p<=whi: dW+=np.int64(n1)*np.int64(n1)-np.int64(o1)*np.int64(o1)
                if wlo<=2*k2p<=whi: dW+=np.int64(n2)*np.int64(n2)-np.int64(o2)*np.int64(o2)
                if wlo<=k1p+k2p<=whi: dW+=np.int64(2)*(np.int64(n1)*np.int64(n2)-np.int64(o1)*np.int64(o2))
                jl=min(wlo-k1p,wlo-k2p); jl=max(jl,0)
                jh=max(whi-k1p,whi-k2p); jh=min(jh,d_child-1)
                for jj in range(jl,jh+1):
                    if jj==k1p or jj==k2p: continue
                    cjp=np.int64(child[jj])
                    if cjp==0: continue
                    if wlo<=k1p+jj<=whi: dW+=np.int64(2)*np.int64(nd2)*cjp
                    if wlo<=k2p+jj<=whi: dW+=np.int64(2)*np.int64(-nd2)*cjp
                abs_dW = dW if dW>=0 else -dW
                eidx = qc_ell-2
                tc = np.int64(two_ell_arr[eidx]*2.0+1.0)
                if surplus > abs_dW + tc:
                    have_skip_prediction = True

        # Advance
        j_gc=gc_focus[0]
        if j_gc==n_active: break
        gc_focus[0]=0; pos=active_pos[j_gc]
        gc_a[j_gc]+=gc_dir[j_gc]; cursor[pos]=lo_arr[pos]+gc_a[j_gc]
        if gc_a[j_gc]==0 or gc_a[j_gc]==radix[j_gc]-1:
            gc_dir[j_gc]=-gc_dir[j_gc]; gc_focus[j_gc]=gc_focus[j_gc+1]; gc_focus[j_gc+1]=j_gc+1
        k1=2*pos; k2=k1+1
        old1=np.int32(child[k1]); old2=np.int32(child[k2])
        child[k1]=cursor[pos]; child[k2]=parent_int[pos]-cursor[pos]
        new1=np.int32(child[k1]); new2=np.int32(child[k2])
        d1=new1-old1; d2=new2-old2
        raw_conv[2*k1]+=new1*new1-old1*old1; raw_conv[2*k2]+=new2*new2-old2*old2
        raw_conv[k1+k2]+=np.int32(2)*(new1*new2-old1*old2)
        for jj in range(k1):
            cj=np.int32(child[jj])
            if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
        for jj in range(k2+1,d_child):
            cj=np.int32(child[jj])
            if cj!=0: raw_conv[k1+jj]+=np.int32(2)*d1*cj; raw_conv[k2+jj]+=np.int32(2)*d2*cj
        if qc_ell>0:
            ql=qc_s-(d_child-1); ql=max(ql,0); qh=qc_s+qc_ell-2; qh=min(qh,d_child-1)
            if ql<=k1<=qh: qc_W_int+=np.int64(d1)
            if ql<=k2<=qh: qc_W_int+=np.int64(d2)
    return n_surv, n_skipped


def generate_parents(n_half, m, c_target, target_level):
    from cpu.run_cascade import run_level0, process_parent_fused
    print(f"  Generating cascade to L{target_level-1}...", flush=True)
    result = run_level0(n_half, m, c_target, verbose=False)
    survivors = result['survivors']
    print(f"  L0: {len(survivors)}", flush=True)
    if target_level == 1: return survivors
    for level in range(1, target_level):
        d_parent = survivors.shape[1]; nhc = d_parent
        all_surv = []
        for i in range(len(survivors)):
            surv, _ = process_parent_fused(survivors[i], m, c_target, nhc)
            if len(surv) > 0: all_surv.append(surv)
        survivors = np.vstack(all_surv) if all_surv else np.empty((0,2*d_parent),dtype=np.int32)
        if len(survivors)>0: survivors = np.unique(survivors, axis=0)
        print(f"  L{level}: {len(survivors):,}", flush=True)
        if len(survivors)==0: break
    return survivors


def main():
    m, c_target, n_half = 20, 1.4, 2
    print("="*72, flush=True)
    print("  CDCL SKIP — TIMING COMPARISON", flush=True)
    print("="*72, flush=True)

    from cpu.run_cascade import _fused_generate_and_prune_gray

    # Warmup
    print("\nJIT warmup...", flush=True)
    dp = np.array([5,5,5,5], dtype=np.int32)
    dl = np.array([0,0,0,0], dtype=np.int32)
    dh = np.array([3,3,3,3], dtype=np.int32)
    db = np.empty((1000,8), dtype=np.int32)
    _gray_skip_kernel(dp,4,20,1.4,dl,dh,db)
    _fused_generate_and_prune_gray(dp,4,20,1.4,dl,dh,db)
    print("Done.\n", flush=True)

    for level_name, tgt_level, max_parents in [('L1->L2', 2, 500), ('L2->L3', 3, 100)]:
        parents = generate_parents(n_half, m, c_target, tgt_level)
        if len(parents) == 0: continue
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        nhc = d_parent
        sample = parents[:min(len(parents), max_parents)]
        print(f"\n{level_name}: {len(sample)} parents, d_child={d_child}", flush=True)

        # Correctness check
        print("  Correctness check...", flush=True)
        mismatches = 0
        for pidx in range(len(sample)):
            parent = sample[pidx]
            result = compute_bin_ranges(parent, m, c_target, d_child, nhc)
            if result is None: continue
            lo,hi,nc = result
            if nc<=1: continue
            bc = min(nc, 200_000)
            b1 = np.empty((bc,d_child),dtype=np.int32)
            b2 = np.empty((bc,d_child),dtype=np.int32)
            n1,_ = _fused_generate_and_prune_gray(parent,nhc,m,c_target,lo,hi,b1)
            n2,_ = _gray_skip_kernel(parent,nhc,m,c_target,lo,hi,b2)
            if n1 != n2:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    MISMATCH parent {pidx}: prod={n1} skip={n2}", flush=True)
        if mismatches == 0:
            print(f"  PASS: all {len(sample)} parents match.", flush=True)
        else:
            print(f"  FAIL: {mismatches} mismatches!", flush=True)
            return

        # Timing
        print("  Timing (3 reps)...", flush=True)
        for rep in range(3):
            # Production
            t0 = time.perf_counter()
            total_surv = 0
            for pidx in range(len(sample)):
                parent = sample[pidx]
                result = compute_bin_ranges(parent, m, c_target, d_child, nhc)
                if result is None: continue
                lo,hi,nc = result
                if nc<=1: continue
                bc = min(nc,200_000)
                buf = np.empty((bc,d_child),dtype=np.int32)
                ns,_ = _fused_generate_and_prune_gray(parent,nhc,m,c_target,lo,hi,buf)
                total_surv += ns
            t_prod = time.perf_counter()-t0

            # Skip
            t0 = time.perf_counter()
            total_surv2 = 0; total_skipped = 0
            for pidx in range(len(sample)):
                parent = sample[pidx]
                result = compute_bin_ranges(parent, m, c_target, d_child, nhc)
                if result is None: continue
                lo,hi,nc = result
                if nc<=1: continue
                bc = min(nc,200_000)
                buf = np.empty((bc,d_child),dtype=np.int32)
                ns,nsk = _gray_skip_kernel(parent,nhc,m,c_target,lo,hi,buf)
                total_surv2 += ns; total_skipped += int(nsk)
            t_skip = time.perf_counter()-t0

            match = "OK" if total_surv==total_surv2 else "FAIL"
            speedup = t_prod/t_skip if t_skip>0 else 0
            print(f"    Rep {rep+1}: prod={t_prod:.3f}s  skip={t_skip:.3f}s  "
                  f"ratio={speedup:.3f}x  surv={match}  skipped={total_skipped:,}", flush=True)

    print(f"\n{'='*72}", flush=True)
    print("  DONE", flush=True)


if __name__ == '__main__':
    main()
