"""Fresh independent V2 / V4 / V14 re-verification at the 3-scale 1.292 point.

PARALLEL VERSION: distributes the 6 cross-Bessel pair integrals across
multiple processes (one per pair), so wall time is dominated by the
single longest pair instead of the sum. Each worker uses mpmath at
dps=30 and chunks of width 50 on [0, T=10^5]. Watson tail past T is
bounded analytically.

Run on 2026-05-20 (independent of earlier audit logs).

Independence: this script imports NO project module (no flint, no
kernels.py); the formula derivation is implemented from first principles.
"""
import mpmath as mp
import multiprocessing as mpc
import time


# ---- worker (top-level for multiprocessing pickling) ----

def _compute_pair(args):
    i, j, di_str, dj_str, T_str, chunk_str, dps = args
    mp.mp.dps = dps
    di = mp.mpf(di_str)
    dj = mp.mpf(dj_str)
    T = mp.mpf(T_str)
    chunk = mp.mpf(chunk_str)
    pi = mp.pi
    f = lambda xi: mp.besselj(0, pi*di*xi)**2 * mp.besselj(0, pi*dj*xi)**2
    n_chunks = int(T / chunk)
    total = mp.mpf(0)
    t0 = time.time()
    for k in range(n_chunks):
        x0 = mp.mpf(k) * chunk
        x1 = mp.mpf(k + 1) * chunk
        total += mp.quad(f, [x0, x1])
    last = mp.mpf(n_chunks) * chunk
    if last < T:
        total += mp.quad(f, [last, T])
    tail = mp.mpf(4) / (pi**4 * di * dj * T)
    return (i, j, str(total), str(tail), time.time() - t0)


def _compute_mo(args):
    T_str, chunk_str, dps = args
    mp.mp.dps = dps
    T = mp.mpf(T_str)
    chunk = mp.mpf(chunk_str)
    pi = mp.pi
    f = lambda xi: mp.besselj(0, pi*xi)**4
    n_chunks = int(T / chunk)
    total = mp.mpf(0)
    t0 = time.time()
    for k in range(n_chunks):
        x0 = mp.mpf(k) * chunk
        x1 = mp.mpf(k + 1) * chunk
        total += mp.quad(f, [x0, x1])
    tail = mp.mpf(4) / (pi**4 * T)
    return (str(total), str(tail), time.time() - t0)


def main():
    dps = 30
    mp.mp.dps = dps
    deltas_str = ['0.138', '0.055', '0.025']
    lambdas = [mp.mpf('0.85'), mp.mpf('0.10'), mp.mpf('0.05')]
    T_str = '100000'
    chunk_str = '50'

    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    jobs = [(i, j, deltas_str[i], deltas_str[j], T_str, chunk_str, dps) for i, j in pairs]

    print("=" * 72)
    print(f"PARALLEL V2/V4/V14 audit (2026-05-20). mpmath dps={dps}, T={float(T_str)}.")
    print(f"chunk_size={float(chunk_str)}, ~{int(float(T_str)/float(chunk_str))} chunks per pair.")
    print(f"6 cross-Bessel pairs + 1 MO surrogate; running on {min(7, mpc.cpu_count())} cores.")
    print("=" * 72, flush=True)

    t_global = time.time()
    with mpc.Pool(processes=7) as pool:
        # Schedule the 6 cross-Bessel pairs + 1 MO surrogate
        cross_async = pool.map_async(_compute_pair, jobs)
        mo_async = pool.apply_async(_compute_mo, (('10000', '5', dps),))
        cross_results = cross_async.get()
        mo_result = mo_async.get()

    print(flush=True)
    print(f"Workers done. Wall time: {time.time()-t_global:.1f}s.", flush=True)
    print(flush=True)

    cross = {}
    for i, j, total_str, tail_str, dt in cross_results:
        cross[(i, j)] = (mp.mpf(total_str), mp.mpf(tail_str))
        cross[(j, i)] = (mp.mpf(total_str), mp.mpf(tail_str))
        print(f"C_{i+1}{j+1} (delta_i={deltas_str[i]}, delta_j={deltas_str[j]}):  "
              f"main={mp.nstr(mp.mpf(total_str), 20)}  tail_UB<={mp.nstr(mp.mpf(tail_str), 5)}  "
              f"t={dt:.1f}s")

    # Assemble K_2
    K2_main = mp.mpf(0)
    K2_tail = mp.mpf(0)
    for i in range(3):
        for j in range(3):
            main, tail = cross[(i, j)]
            K2_main += lambdas[i] * lambdas[j] * main
            K2_tail += lambdas[i] * lambdas[j] * tail
    K2_main *= 2
    K2_tail *= 2

    print()
    print(f"K_2 main (mpmath, [0, T={float(T_str)}]):  {mp.nstr(K2_main, 22)}")
    print(f"K_2 Watson tail past T:                  <= {mp.nstr(K2_tail, 6)}")
    print(f"K_2 mpmath enclosure:  [{mp.nstr(K2_main, 18)}, {mp.nstr(K2_main + K2_tail, 18)}]")
    print()
    print(f"Cert arb enclosure:    [4.7888234212591545, 4.7889051816332424]")
    print(f"Cert axiom UB:         47897/10000 = 4.7897")
    print()

    K2_lower_cert = mp.mpf("4.7888234212591545")
    K2_upper_cert = mp.mpf("4.7889051816332424")
    K2_axiom = mp.mpf("4.7897")
    K2_mpmath_UB = K2_main + K2_tail

    inside_cert_lower = K2_main >= K2_lower_cert - mp.mpf("1e-10")
    inside_cert_upper = K2_mpmath_UB <= K2_upper_cert + mp.mpf("1e-3")  # loose: just to confirm we're in the right ballpark
    under_axiom = K2_mpmath_UB <= K2_axiom
    encloses_cert_mid = (K2_main <= mp.mpf("4.788864301446199") <= K2_mpmath_UB)

    print(f"V2 verdict:")
    print(f"  mpmath enclosure contains cert midpoint 4.788864...?  {encloses_cert_mid}")
    print(f"  mpmath upper {mp.nstr(K2_mpmath_UB, 16)} <= axiom 4.7897?  {under_axiom}")
    print(f"  slack to axiom = {mp.nstr(K2_axiom - K2_mpmath_UB, 6)}")
    print(f"  V2: {'PASS' if (encloses_cert_mid and under_axiom) else 'FAIL'}")
    print()

    # V4
    mo_main_str, mo_tail_str, mo_dt = mo_result
    mo_main = mp.mpf(mo_main_str)
    mo_tail = mp.mpf(mo_tail_str)
    mo_upper = mo_main + mo_tail
    print("=" * 72)
    print("V4: MO surrogate int_0^infty J_0(pi xi)^4 dxi")
    print("=" * 72)
    print(f"mpmath: int_0^10000 J_0(pi xi)^4 dxi = {mp.nstr(mo_main, 22)}  (t={mo_dt:.1f}s)")
    print(f"Watson tail past 10000: <= {mp.nstr(mo_tail, 6)}")
    print(f"half-line enclosure: [{mp.nstr(mo_main, 18)}, {mp.nstr(mo_upper, 18)}]")
    print(f"full-line (2x):       [{mp.nstr(2*mo_main, 18)}, {mp.nstr(2*mo_upper, 18)}]")
    print()
    mo_lit = mp.mpf("0.574695")
    agreement_full = abs(2*mo_main - mo_lit) < mp.mpf("1e-5")
    under_full = 2*mo_upper <= mp.mpf("0.5747")
    under_half = mo_upper <= mp.mpf("0.5747")
    print(f"V4 verdict:")
    print(f"  doubled mpmath value matches literature 0.574695 (<1e-5)?  {agreement_full}")
    print(f"  half-line UB <= 0.5747?  {under_half}")
    print(f"  full-line UB <= 0.5747?  {under_full}  (literature convention)")
    print(f"  V4: {'PASS' if agreement_full and under_full else 'FAIL'}")
    print()

    # V14
    print("=" * 72)
    print("V14: per-pair cross-Bessel cross-check (mpmath vs flint anchor values)")
    print("=" * 72)
    flint_raw = {
        (0, 0): mp.mpf("2.08221967490086746"),
        (0, 1): mp.mpf("2.72394150310173977"),
        (0, 2): mp.mpf("3.29371774060815614"),
        (1, 1): mp.mpf("5.224447838177257"),
        (1, 2): mp.mpf("6.59933265626005472"),
        (2, 2): mp.mpf("11.4936508363574674"),
    }
    all_ok = True
    for i, j in pairs:
        mp_val, _ = cross[(i, j)]
        flint_val = flint_raw[(i, j)]
        diff = mp_val - flint_val
        abs_diff = abs(diff)
        rel_err = abs_diff / flint_val
        ok = abs_diff < mp.mpf("1e-10")
        if not ok:
            all_ok = False
        print(f"  C_{i+1}{j+1}: mpmath={mp.nstr(mp_val, 20)}  flint={mp.nstr(flint_val, 20)}")
        print(f"        diff={mp.nstr(diff, 6)}  rel_err={mp.nstr(rel_err, 4)}  |diff|<1e-10: {ok}")
    print()
    print(f"V14: {'PASS' if all_ok else 'FAIL'}")
    print()
    print(f"TOTAL wall time: {time.time()-t_global:.1f}s")


if __name__ == '__main__':
    main()
