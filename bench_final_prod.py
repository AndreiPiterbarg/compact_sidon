"""Wall-clock benchmark using the PRODUCTION kernel (now with sparse optimization).

Compares the modified production kernel against the original (non-sparse) kernel
copied from before the change.
"""
import sys, os, time, math
import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction
from cpu.run_cascade import _fused_generate_and_prune_gray, _compute_bin_ranges


# =====================================================================
# Original kernel (pre-sparse, for A/B comparison)
# =====================================================================
@njit(cache=False)
def _kern_original(parent_int, nhc, m, ct, lo, hi, buf):
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


def main():
    m = 20; ct = 1.4; RUNS = 7
    print("="*70)
    print("PRODUCTION KERNEL BENCHMARK (sparse optimization integrated)")
    print(f"  m={m}, c_target={ct}, {RUNS} timing runs")
    print("="*70)

    l4 = np.load('data/checkpoint_L4_survivors.npy')
    parents = (l4[:, 0::2] + l4[:, 1::2]).astype(np.int32)
    parents = np.unique(parents, axis=0)
    print(f"\n{len(parents):,} unique L3 parents from L4 survivors")

    nhc = 32; dc = 64
    np.random.seed(42)
    max_p = min(len(parents), 2000)
    idx = np.random.choice(len(parents), max_p, replace=False); idx.sort()
    sample = parents[idx]

    inputs = []
    total_ch = 0
    for i in range(len(sample)):
        result = _compute_bin_ranges(sample[i], m, ct, dc, nhc)
        if result is None: continue
        lo, hi, tc = result
        if tc <= 1: continue
        inputs.append((sample[i], lo, hi, min(tc, 500_000), tc))
        total_ch += tc

    print(f"  {len(inputs)} parents, {total_ch:,} total children")

    # JIT warmup
    print("\nJIT warmup...")
    d2 = np.array([10,10], dtype=np.int32)
    dl = np.array([0,0], dtype=np.int32)
    dh = np.array([5,5], dtype=np.int32)
    db = np.empty((100,4), dtype=np.int32)
    _kern_original(d2, 2, 20, 1.4, dl, dh, db)
    # Warmup production kernel at d=64
    p0, l0, h0, bc0, _ = inputs[0]
    b0 = np.empty((bc0, dc), dtype=np.int32)
    _kern_original(p0, nhc, m, ct, l0, h0, b0)
    _fused_generate_and_prune_gray(p0, nhc, m, ct, l0, h0, b0)
    print("JIT warm.")

    # Correctness
    print(f"\nCorrectness check on {min(500, len(inputs))} parents...")
    n_fail = 0
    for p, lo, hi, bc, nc in inputs[:500]:
        buf1 = np.empty((bc, dc), dtype=np.int32)
        buf2 = np.empty((bc, dc), dtype=np.int32)
        ns1, _ = _kern_original(p, nhc, m, ct, lo, hi, buf1)
        ns2, _ = _fused_generate_and_prune_gray(p, nhc, m, ct, lo, hi, buf2)
        if ns1 != ns2:
            print(f"  FAIL: count {ns1} vs {ns2}")
            n_fail += 1; continue
        if ns1 > 0 and ns1 <= bc:
            s1 = buf1[:ns1].copy(); s2 = buf2[:ns2].copy()
            s1 = s1[np.lexsort(s1[:,::-1].T)]
            s2 = s2[np.lexsort(s2[:,::-1].T)]
            if not np.array_equal(s1, s2):
                print(f"  FAIL: arrays differ"); n_fail += 1
    if n_fail > 0:
        print(f"  *** {n_fail} FAILURES ***"); return
    print(f"  PASS")

    # Timing
    print(f"\nTiming ({RUNS} runs each)...")
    times_orig = []; times_prod = []; total_surv = 0
    for run in range(RUNS):
        t0 = time.perf_counter(); rs = 0
        for p, lo, hi, bc, nc in inputs:
            b = np.empty((bc, dc), dtype=np.int32)
            ns, _ = _kern_original(p, nhc, m, ct, lo, hi, b); rs += ns
        times_orig.append(time.perf_counter() - t0)
        if run == 0: total_surv = rs

        t0 = time.perf_counter()
        for p, lo, hi, bc, nc in inputs:
            b = np.empty((bc, dc), dtype=np.int32)
            _fused_generate_and_prune_gray(p, nhc, m, ct, lo, hi, b)
        times_prod.append(time.perf_counter() - t0)
        print(f"  run {run+1}/{RUNS}: original={times_orig[-1]:.3f}s  production(sparse)={times_prod[-1]:.3f}s")

    bo = min(times_orig); bp = min(times_prod)
    mo = sorted(times_orig)[RUNS//2]; mp = sorted(times_prod)[RUNS//2]

    print(f"\n{'='*70}")
    print(f"RESULTS (d_child=64, {len(inputs)} parents, {total_ch:,} children)")
    print(f"{'='*70}")
    print(f"  Original (no sparse): best={bo:.4f}s  median={mo:.4f}s")
    print(f"  Production (sparse):  best={bp:.4f}s  median={mp:.4f}s")
    print(f"  Survivors: {total_surv}")
    print(f"\n  SPEEDUP (best):   {bo/bp:.3f}x")
    print(f"  SPEEDUP (median): {mo/mp:.3f}x")
    print(f"  Throughput: orig={total_ch/bo/1e6:.2f}M ch/s  prod={total_ch/bp/1e6:.2f}M ch/s")

    l4_hours = 250235 / 3600
    if bp < bo:
        saved = l4_hours * (1 - bp/bo)
        print(f"\n  L4 baseline: {l4_hours:.1f}h, projected savings: {saved:.1f}h")


if __name__ == '__main__':
    main()
