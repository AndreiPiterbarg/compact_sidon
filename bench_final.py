"""Direct wall-clock A/B test at d_child=64 using reconstructed L3 parents.

No cascade generation needed -- L3 parents are reconstructed from L4 survivors
by summing adjacent pairs: parent[k] = child[2k] + child[2k+1].
"""
import sys, os, time, math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction

# =====================================================================
# Production Gray code kernel
# =====================================================================
@njit(cache=False)
def _kern_prod(parent_int, nhc, m, ct, lo, hi, buf):
    dp = parent_int.shape[0]; dc = 2*dp
    assert m <= 200
    md = np.float64(m); ta = math.sqrt(ct/2.0)
    ls = np.int64(0)
    for i in range(dp//2): ls += np.int64(parent_int[i])
    lf = np.float64(ls)/md
    if lf >= ta or lf <= 1.0 - ta: return 0, 0
    db = ct*md*md + 1.0 + 1e-9*md*md; i4n = 1.0/(4.0*np.float64(nhc))
    eps4 = 1.0 - 4.0*2.220446049250313e-16
    ms = buf.shape[0]; ns = 0; cl = 2*dc-1
    cur = np.empty(dp, dtype=np.int32)
    for i in range(dp): cur[i] = lo[i]
    ch = np.empty(dc, dtype=np.int32)
    rc = np.empty(cl, dtype=np.int32)
    pc = np.empty(dc+1, dtype=np.int64)
    qe = np.int32(0); qs = np.int32(0); qw = np.int64(0)
    for i in range(dp): ch[2*i] = cur[i]; ch[2*i+1] = parent_int[i] - cur[i]
    ec = 2*dc-1
    dba = np.empty(ec, dtype=np.float64); tla = np.empty(ec, dtype=np.float64)
    for e in range(2, 2*dc+1):
        dba[e-2] = db*np.float64(e)*i4n; tla[e-2] = 2.0*np.float64(e)*i4n
    eo = np.empty(ec, dtype=np.int32); eu = np.zeros(ec, dtype=np.int32); oi = 0
    p1e = min(16, 2*dc)
    for e in range(2, p1e+1): eo[oi]=np.int32(e); eu[e-2]=np.int32(1); oi+=1
    for e in (dc, dc+1, dc-1, dc+2, dc-2, dc*2, dc+dc//2, dc//2):
        if 2<=e<=2*dc and eu[e-2]==0: eo[oi]=np.int32(e); eu[e-2]=np.int32(1); oi+=1
    for e in range(2, 2*dc+1):
        if eu[e-2]==0: eo[oi]=np.int32(e); oi+=1
    for k in range(cl): rc[k] = np.int32(0)
    for i in range(dc):
        ci = np.int32(ch[i])
        if ci != 0:
            rc[2*i] += ci*ci
            for j in range(i+1, dc):
                cj = np.int32(ch[j])
                if cj != 0: rc[i+j] += np.int32(2)*ci*cj
    na = 0; ap = np.empty(dp, dtype=np.int32); rx = np.empty(dp, dtype=np.int32)
    for i in range(dp):
        r = hi[i]-lo[i]+1
        if r>1: ap[na]=i; rx[na]=r; na+=1
    ga = np.zeros(na, dtype=np.int32); gd = np.ones(na, dtype=np.int32)
    gf = np.empty(na+1, dtype=np.int32)
    for i in range(na+1): gf[i] = i
    while True:
        qk = False
        if qe > 0:
            nq = qe-1; ws = np.int64(0)
            for k in range(qs, qs+nq): ws += np.int64(rc[k])
            dx = dba[qe-2]+tla[qe-2]*np.float64(qw)
            if ws > np.int64(dx*eps4): qk = True
        if not qk:
            pc[0] = 0
            for i in range(dc): pc[i+1] = pc[i]+np.int64(ch[i])
            pr = False
            for eoi in range(ec):
                if pr: break
                e = eo[eoi]; nv = e-1; nw = cl-nv+1
                dbe = dba[e-2]; tlv = tla[e-2]
                ws = np.int64(0)
                for k in range(nv): ws += np.int64(rc[k])
                for sl in range(nw):
                    if sl > 0: ws += np.int64(rc[sl+nv-1]) - np.int64(rc[sl-1])
                    lb = sl-(dc-1)
                    if lb < 0: lb = 0
                    hb = sl+e-2
                    if hb > dc-1: hb = dc-1
                    wi = pc[hb+1]-pc[lb]
                    dx = dbe+tlv*np.float64(wi)
                    if ws > np.int64(dx*eps4):
                        pr=True; qe=np.int32(e); qs=np.int32(sl); qw=wi; break
            if not pr:
                ur = False
                for i in range(dc):
                    j = dc-1-i
                    if ch[j]<ch[i]: ur=True; break
                    elif ch[j]>ch[i]: break
                if ns < ms:
                    if ur:
                        for i in range(dc): buf[ns,i]=ch[dc-1-i]
                    else:
                        for i in range(dc): buf[ns,i]=ch[i]
                ns += 1
        j = gf[0]
        if j == na: break
        gf[0] = 0; pos = ap[j]; ga[j] += gd[j]; cur[pos] = lo[pos]+ga[j]
        if ga[j]==0 or ga[j]==rx[j]-1: gd[j]=-gd[j]; gf[j]=gf[j+1]; gf[j+1]=j+1
        k1=2*pos; k2=k1+1
        o1=np.int32(ch[k1]); o2=np.int32(ch[k2])
        ch[k1]=cur[pos]; ch[k2]=parent_int[pos]-cur[pos]
        n1=np.int32(ch[k1]); n2=np.int32(ch[k2])
        d1=n1-o1; d2=n2-o2
        rc[2*k1]+=n1*n1-o1*o1; rc[2*k2]+=n2*n2-o2*o2
        rc[k1+k2]+=np.int32(2)*(n1*n2-o1*o2)
        for jj in range(k1):
            cj=np.int32(ch[jj])
            if cj!=0: rc[k1+jj]+=np.int32(2)*d1*cj; rc[k2+jj]+=np.int32(2)*d2*cj
        for jj in range(k2+1, dc):
            cj=np.int32(ch[jj])
            if cj!=0: rc[k1+jj]+=np.int32(2)*d1*cj; rc[k2+jj]+=np.int32(2)*d2*cj
        if qe > 0:
            ql=qs-(dc-1)
            if ql<0: ql=0
            qh=qs+qe-2
            if qh>dc-1: qh=dc-1
            if ql<=k1 and k1<=qh: qw+=np.int64(d1)
            if ql<=k2 and k2<=qh: qw+=np.int64(d2)
    return ns, 0


# =====================================================================
# Sparse variant
# =====================================================================
@njit(cache=False)
def _kern_sparse(parent_int, nhc, m, ct, lo, hi, buf):
    dp = parent_int.shape[0]; dc = 2*dp
    assert m <= 200
    md = np.float64(m); ta = math.sqrt(ct/2.0)
    ls = np.int64(0)
    for i in range(dp//2): ls += np.int64(parent_int[i])
    lf = np.float64(ls)/md
    if lf >= ta or lf <= 1.0 - ta: return 0, 0
    db = ct*md*md + 1.0 + 1e-9*md*md; i4n = 1.0/(4.0*np.float64(nhc))
    eps4 = 1.0 - 4.0*2.220446049250313e-16
    ms = buf.shape[0]; ns = 0; cl = 2*dc-1
    cur = np.empty(dp, dtype=np.int32)
    for i in range(dp): cur[i] = lo[i]
    ch = np.empty(dc, dtype=np.int32)
    rc = np.empty(cl, dtype=np.int32)
    pc = np.empty(dc+1, dtype=np.int64)
    qe = np.int32(0); qs = np.int32(0); qw = np.int64(0)
    # Sparse structures
    nzl = np.empty(dc, dtype=np.int32)
    nzp = np.full(dc, -1, dtype=np.int32)
    nzc = 0
    for i in range(dp): ch[2*i] = cur[i]; ch[2*i+1] = parent_int[i] - cur[i]
    for i in range(dc):
        if ch[i] != 0: nzl[nzc]=i; nzp[i]=nzc; nzc+=1
    ec = 2*dc-1
    dba = np.empty(ec, dtype=np.float64); tla = np.empty(ec, dtype=np.float64)
    for e in range(2, 2*dc+1):
        dba[e-2] = db*np.float64(e)*i4n; tla[e-2] = 2.0*np.float64(e)*i4n
    eo = np.empty(ec, dtype=np.int32); eu = np.zeros(ec, dtype=np.int32); oi = 0
    p1e = min(16, 2*dc)
    for e in range(2, p1e+1): eo[oi]=np.int32(e); eu[e-2]=np.int32(1); oi+=1
    for e in (dc, dc+1, dc-1, dc+2, dc-2, dc*2, dc+dc//2, dc//2):
        if 2<=e<=2*dc and eu[e-2]==0: eo[oi]=np.int32(e); eu[e-2]=np.int32(1); oi+=1
    for e in range(2, 2*dc+1):
        if eu[e-2]==0: eo[oi]=np.int32(e); oi+=1
    for k in range(cl): rc[k] = np.int32(0)
    for i in range(dc):
        ci = np.int32(ch[i])
        if ci != 0:
            rc[2*i] += ci*ci
            for j in range(i+1, dc):
                cj = np.int32(ch[j])
                if cj != 0: rc[i+j] += np.int32(2)*ci*cj
    na = 0; ap = np.empty(dp, dtype=np.int32); rx = np.empty(dp, dtype=np.int32)
    for i in range(dp):
        r = hi[i]-lo[i]+1
        if r>1: ap[na]=i; rx[na]=r; na+=1
    ga = np.zeros(na, dtype=np.int32); gd = np.ones(na, dtype=np.int32)
    gf = np.empty(na+1, dtype=np.int32)
    for i in range(na+1): gf[i] = i
    while True:
        qk = False
        if qe > 0:
            nq = qe-1; ws = np.int64(0)
            for k in range(qs, qs+nq): ws += np.int64(rc[k])
            dx = dba[qe-2]+tla[qe-2]*np.float64(qw)
            if ws > np.int64(dx*eps4): qk = True
        if not qk:
            pc[0] = 0
            for i in range(dc): pc[i+1] = pc[i]+np.int64(ch[i])
            pr = False
            for eoi in range(ec):
                if pr: break
                e = eo[eoi]; nv = e-1; nw = cl-nv+1
                dbe = dba[e-2]; tlv = tla[e-2]
                ws = np.int64(0)
                for k in range(nv): ws += np.int64(rc[k])
                for sl in range(nw):
                    if sl > 0: ws += np.int64(rc[sl+nv-1]) - np.int64(rc[sl-1])
                    lb = sl-(dc-1)
                    if lb < 0: lb = 0
                    hb = sl+e-2
                    if hb > dc-1: hb = dc-1
                    wi = pc[hb+1]-pc[lb]
                    dx = dbe+tlv*np.float64(wi)
                    if ws > np.int64(dx*eps4):
                        pr=True; qe=np.int32(e); qs=np.int32(sl); qw=wi; break
            if not pr:
                ur = False
                for i in range(dc):
                    j = dc-1-i
                    if ch[j]<ch[i]: ur=True; break
                    elif ch[j]>ch[i]: break
                if ns < ms:
                    if ur:
                        for i in range(dc): buf[ns,i]=ch[dc-1-i]
                    else:
                        for i in range(dc): buf[ns,i]=ch[i]
                ns += 1
        j = gf[0]
        if j == na: break
        gf[0] = 0; pos = ap[j]; ga[j] += gd[j]; cur[pos] = lo[pos]+ga[j]
        if ga[j]==0 or ga[j]==rx[j]-1: gd[j]=-gd[j]; gf[j]=gf[j+1]; gf[j+1]=j+1
        k1=2*pos; k2=k1+1
        o1=np.int32(ch[k1]); o2=np.int32(ch[k2])
        ch[k1]=cur[pos]; ch[k2]=parent_int[pos]-cur[pos]
        n1=np.int32(ch[k1]); n2=np.int32(ch[k2])
        d1=n1-o1; d2=n2-o2
        rc[2*k1]+=n1*n1-o1*o1; rc[2*k2]+=n2*n2-o2*o2
        rc[k1+k2]+=np.int32(2)*(n1*n2-o1*o2)
        # Update nz_list
        if o1!=0 and n1==0:
            p=nzp[k1]; nzc-=1; last=nzl[nzc]; nzl[p]=last; nzp[last]=p; nzp[k1]=-1
        elif o1==0 and n1!=0:
            nzl[nzc]=k1; nzp[k1]=nzc; nzc+=1
        if o2!=0 and n2==0:
            p=nzp[k2]; nzc-=1; last=nzl[nzc]; nzl[p]=last; nzp[last]=p; nzp[k2]=-1
        elif o2==0 and n2!=0:
            nzl[nzc]=k2; nzp[k2]=nzc; nzc+=1
        # Sparse cross-terms
        for idx in range(nzc):
            jj = nzl[idx]
            if jj != k1 and jj != k2:
                cj=np.int32(ch[jj])
                rc[k1+jj]+=np.int32(2)*d1*cj; rc[k2+jj]+=np.int32(2)*d2*cj
        if qe > 0:
            ql=qs-(dc-1)
            if ql<0: ql=0
            qh=qs+qe-2
            if qh>dc-1: qh=dc-1
            if ql<=k1 and k1<=qh: qw+=np.int64(d1)
            if ql<=k2 and k2<=qh: qw+=np.int64(d2)
    return ns, 0


def compute_ranges(p, m, ct, dc, nhc):
    dp = len(p); corr = correction(m, nhc)
    thr = ct + corr + 1e-9
    xc = int(math.floor(m*math.sqrt(thr/dc)))
    xcs = int(math.floor(m*math.sqrt(ct/dc)))
    xc = min(xc, xcs, m); xc = max(xc, 0)
    lo = np.empty(dp, dtype=np.int32); hi = np.empty(dp, dtype=np.int32); tc = 1
    for i in range(dp):
        b = int(p[i]); l = max(0, b-xc); h = min(b, xc)
        if l > h: return None
        lo[i] = l; hi[i] = h; tc *= (h-l+1)
    return lo, hi, tc


def main():
    m = 20; ct = 1.4; RUNS = 7
    print("="*70)
    print("FINAL WALL-CLOCK A/B: Production vs Sparse at d_child=64")
    print(f"  m={m}, c_target={ct}, {RUNS} timing runs")
    print("="*70)

    # Reconstruct L3 parents from L4 survivors
    l4 = np.load('data/checkpoint_L4_survivors.npy')
    parents = (l4[:, 0::2] + l4[:, 1::2]).astype(np.int32)
    # Deduplicate parents
    parents = np.unique(parents, axis=0)
    print(f"\n{len(parents):,} unique L3 parents reconstructed from L4 survivors")
    nnz = np.count_nonzero(parents, axis=1)
    print(f"  parent nnz: min={nnz.min()}, max={nnz.max()}, mean={nnz.mean():.1f}")

    nhc = 32; dc = 64

    # Prepare inputs
    print("\nPreparing inputs...")
    np.random.seed(42)
    max_p = min(len(parents), 2000)
    idx = np.random.choice(len(parents), max_p, replace=False)
    idx.sort()
    sample = parents[idx]

    inputs = []
    total_ch = 0
    for i in range(len(sample)):
        r = compute_ranges(sample[i], m, ct, dc, nhc)
        if r is None: continue
        lo, hi, tc = r
        if tc <= 1: continue
        inputs.append((sample[i], lo, hi, min(tc, 500_000), tc))
        total_ch += tc

    print(f"  {len(inputs)} valid parents, {total_ch:,} total children")
    ch_counts = np.array([nc for _,_,_,_,nc in inputs])
    print(f"  children/parent: min={ch_counts.min():,}, max={ch_counts.max():,}, "
          f"mean={ch_counts.mean():,.0f}, median={np.median(ch_counts):,.0f}")

    # JIT warmup
    print("\nJIT warmup...")
    d2 = np.array([10,10], dtype=np.int32)
    dl = np.array([0,0], dtype=np.int32)
    dh = np.array([5,5], dtype=np.int32)
    db = np.empty((100,4), dtype=np.int32)
    _kern_prod(d2, 2, 20, 1.4, dl, dh, db)
    _kern_sparse(d2, 2, 20, 1.4, dl, dh, db)
    # Also warmup at d_parent=32
    p0, l0, h0, bc0, _ = inputs[0]
    b0 = np.empty((bc0, dc), dtype=np.int32)
    _kern_prod(p0, nhc, m, ct, l0, h0, b0)
    _kern_sparse(p0, nhc, m, ct, l0, h0, b0)
    print("JIT warm.")

    # === CORRECTNESS ===
    print(f"\nCorrectnness check on {min(500, len(inputs))} parents...")
    n_fail = 0
    n_checked = 0
    total_surv_checked = 0
    for p, lo, hi, bc, nc in inputs[:500]:
        buf1 = np.empty((bc, dc), dtype=np.int32)
        buf2 = np.empty((bc, dc), dtype=np.int32)
        ns1, _ = _kern_prod(p, nhc, m, ct, lo, hi, buf1)
        ns2, _ = _kern_sparse(p, nhc, m, ct, lo, hi, buf2)
        if ns1 != ns2:
            print(f"  FAIL: count {ns1} vs {ns2}, parent={p}")
            n_fail += 1
            if n_fail >= 3: break
            continue
        if ns1 > 0 and ns1 <= bc:
            s1 = buf1[:ns1].copy(); s2 = buf2[:ns2].copy()
            s1 = s1[np.lexsort(s1[:,::-1].T)]
            s2 = s2[np.lexsort(s2[:,::-1].T)]
            if not np.array_equal(s1, s2):
                print(f"  FAIL: arrays differ, parent={p}")
                n_fail += 1
                if n_fail >= 3: break
                continue
        n_checked += 1
        total_surv_checked += ns1

    if n_fail > 0:
        print(f"  *** {n_fail} FAILURES -- ABORTING ***")
        return
    print(f"  PASS: {n_checked} parents, {total_surv_checked} survivors verified")

    # === TIMING ===
    print(f"\nTiming ({RUNS} runs each on {len(inputs)} parents, {total_ch:,} children)...")

    times_prod = []
    times_sparse = []
    total_surv = 0

    for run in range(RUNS):
        t0 = time.perf_counter()
        rs = 0
        for p, lo, hi, bc, nc in inputs:
            b = np.empty((bc, dc), dtype=np.int32)
            ns, _ = _kern_prod(p, nhc, m, ct, lo, hi, b)
            rs += ns
        times_prod.append(time.perf_counter() - t0)
        if run == 0: total_surv = rs

        t0 = time.perf_counter()
        for p, lo, hi, bc, nc in inputs:
            b = np.empty((bc, dc), dtype=np.int32)
            _kern_sparse(p, nhc, m, ct, lo, hi, b)
        times_sparse.append(time.perf_counter() - t0)

        print(f"  run {run+1}/{RUNS}: prod={times_prod[-1]:.3f}s  sparse={times_sparse[-1]:.3f}s")

    bp = min(times_prod); bs = min(times_sparse)
    mp = sorted(times_prod)[RUNS//2]; ms_ = sorted(times_sparse)[RUNS//2]

    print(f"\n{'='*70}")
    print(f"RESULTS (d_child=64, {len(inputs)} parents, {total_ch:,} children)")
    print(f"{'='*70}")
    print(f"  Production:  best={bp:.4f}s  median={mp:.4f}s")
    print(f"  Sparse:      best={bs:.4f}s  median={ms_:.4f}s")
    print(f"  Survivors:   {total_surv}")
    print(f"\n  SPEEDUP (best):   {bp/bs:.3f}x")
    print(f"  SPEEDUP (median): {mp/ms_:.3f}x")
    print(f"\n  Throughput: prod={total_ch/bp/1e6:.2f}M ch/s  sparse={total_ch/bs/1e6:.2f}M ch/s")

    # Project savings
    l4_hours = 250235 / 3600  # ~69.5h
    if bs < bp:
        saved = l4_hours * (1 - bs/bp)
        print(f"\n  L4 baseline: {l4_hours:.1f}h")
        print(f"  Projected savings: {saved:.1f}h")
        if bp/bs >= 1.15:
            print(f"\n  VERDICT: IMPLEMENT (significant gain)")
        elif bp/bs >= 1.05:
            print(f"\n  VERDICT: MARGINAL (implement only if easy)")
        else:
            print(f"\n  VERDICT: SKIP (not worth the complexity)")
    else:
        print(f"\n  VERDICT: SKIP (sparse is slower)")

    # Note on bias
    print(f"\n  NOTE: These parents are biased -- they all produced L4 survivors.")
    print(f"  Typical parents (producing 0 survivors) have higher QC hit rates,")
    print(f"  making cross-terms a LARGER fraction of their work.")
    print(f"  The actual speedup on the full L4 run may be HIGHER than measured.")


if __name__ == '__main__':
    main()
