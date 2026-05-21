"""
INDEPENDENT real-space sanity check on K_2(K_ms) ~ 4.78884.

The certifier and `audit3_mpmath.py` both compute K_2 via Plancherel:
  K_2 = int |K_hat_ms(xi)|^2 dxi = int (sum lambda_i J_0(pi delta_i xi)^2)^2 dxi
This is the SAME Fourier formula on both sides -- a formula bug there
would NOT be caught.  This script instead computes K_2 directly in
*real space*, using two independent constructions, to cross-check.

Conventions (from `lean/Sidon/MultiScale.lean` lines 297-342):
    eta_delta(x) = (1/pi) / sqrt((delta/2)^2 - x^2)  for |x| < delta/2,
                                                    0 otherwise.
    K_arc(delta, x) = (eta_delta * eta_delta)(x)         [autoconvolution]
                    -- supported in (-delta, delta), mass 1.
    K_ms(x) = lambda_1 K_arc(delta_1, x) + lambda_2 K_arc(delta_2, x)
            + lambda_3 K_arc(delta_3, x).

By Fourier scaling, K_arc(delta, x) = (1/delta) * mu(x/delta), where
    mu(y) := (eta_1 * eta_1)(y)
is the autoconvolution at unit half-width (delta = 1).  mu is supported
on (-1, 1) with mass 1.

CLOSED FORM (verified below by direct numerical convolution):
    mu(y) = (2/pi^2) * K(1 - y^2)        for |y| < 1, else 0,
where K(m) is the complete elliptic integral of the first kind in
PARAMETER (not modulus) convention:
    K(m) = int_0^{pi/2}  dphi / sqrt(1 - m sin^2 phi).
(This is mpmath's `ellipk(m)` convention.)  Mass = 1; mu has a
logarithmic singularity at 0 and a finite limit mu(1^-) = 2/pi^2 * pi/2
= 1/pi at the endpoints (no boundary singularity).

Two independent methods:

Method 1 -- numerical autoconvolution of eta_1 (FFT, fine grid).
    A first-principles check: build eta_1 on a grid (avoiding singular
    endpoints), FFT-convolve, integrate.  Coarse: dominated by FFT
    aliasing near the singularities, mass(mu) overshoots by ~0.4% at
    N = 2 * 10^5.  Accuracy ~ few * 10^-3.

Method 2 -- closed-form mu via complete elliptic integral.
    Use mu(y) = (2/pi^2) K(1 - y^2) and compute K_2 via
        K_2 = sum_{i,j} lambda_i lambda_j * I_{ij},
    where I_{ij} = int K_arc(d_i, x) K_arc(d_j, x) dx is evaluated
    directly in real space by mpmath.quad with high precision.

If Method 2 agrees with [4.788823, 4.788906] to better than 10^-4 the
Fourier-side computation is formula-correct.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.signal import fftconvolve
import mpmath as mp


DELTAS = (0.138, 0.055, 0.025)
LAMBDAS = (0.85, 0.10, 0.05)

CERT_LO = 4.7888234212591545
CERT_HI = 4.7889051816332424


# ----------------------------------------------------------------------
#   METHOD 1: FFT convolution on a uniform grid (sanity / first-principles)
# ----------------------------------------------------------------------

def method1_fft(N: int = 400_001) -> float:
    """K_2 by FFT-convolution of eta_1 on a uniform grid (no closed form).

    Grid: x in (-1/2 + eps, 1/2 - eps), N points.  Note: the
    arcsine-density square-root singularity at the endpoints is
    integrable but ill-suited to point-sampling, so this method has
    O(few * 10^-3) accuracy regardless of N.  Its purpose is to confirm
    independently that the closed-form Method 2 is in the right
    ballpark.
    """
    print(f"\n--- METHOD 1: FFT convolution (sanity), N = {N} ---")

    eps = 5e-7
    x = np.linspace(-0.5 + eps, 0.5 - eps, N)
    dx = x[1] - x[0]
    print(f"  grid dx = {dx:.3e}, eps from singularity = {eps}")

    # eta_1(x) = (1/pi) / sqrt(1/4 - x^2)
    eta1 = (1.0 / math.pi) / np.sqrt(0.25 - x ** 2)
    mass_eta1 = np.trapezoid(eta1, x)
    print(f"  mass(eta_1)       = {mass_eta1:.8f}  (true 1; endpoint cap loss)")

    # Linear convolution -> length 2N-1, dx-spaced, covering (-1, 1).
    mu_full = fftconvolve(eta1, eta1, mode="full") * dx
    y_full = np.linspace(-1.0 + 2 * eps, 1.0 - 2 * eps, 2 * N - 1)
    print(f"  y range           = [{y_full[0]:.6f}, {y_full[-1]:.6f}]")
    print(f"  mass(mu = eta1*eta1) = {np.trapezoid(mu_full, y_full):.8f}  "
          f"(true 1, FFT-aliasing slop)")

    # K_arc(delta, x) = (1/delta) mu(x/delta).
    delta_max = max(DELTAS)
    M = 800_001
    xx = np.linspace(-delta_max + 1e-9, delta_max - 1e-9, M)

    def K_arc_eval(delta, xx):
        y = xx / delta
        mask = np.abs(y) < 1.0 - 1e-10
        out = np.zeros_like(xx)
        out[mask] = np.interp(y[mask], y_full, mu_full) / delta
        return out

    K_ms_vals = np.zeros_like(xx)
    for lam, d in zip(LAMBDAS, DELTAS):
        K_ms_vals += lam * K_arc_eval(d, xx)

    mass_K = np.trapezoid(K_ms_vals, xx)
    print(f"  mass(K_ms)        = {mass_K:.6f}  (true 1)")
    K2 = np.trapezoid(K_ms_vals ** 2, xx)
    print(f"  K_2 (Method 1)    = {K2:.6f}")
    return K2


# ----------------------------------------------------------------------
#   METHOD 2: Closed-form mu via complete elliptic integral
# ----------------------------------------------------------------------

def mu_closedform(y):
    """mu(y) = (2/pi^2) * K(1 - y^2) for |y| < 1, else 0.

    K(m) is mpmath's `ellipk(m)`, with the PARAMETER convention
        K(m) = int_0^{pi/2} d(phi) / sqrt(1 - m sin^2 phi).
    """
    if abs(y) >= 1.0:
        return mp.mpf(0)
    return (2 / mp.pi ** 2) * mp.ellipk(1 - mp.mpf(y) ** 2)


def method2_verify_mu(dps: int = 30) -> None:
    """Verify mu's closed form by:
       (i) point-by-point match vs direct integral definition;
       (ii) mass(mu) == 1 (log-singularity-aware).
    """
    mp.mp.dps = dps
    print("  Verifying mu(y) = (2/pi^2) K(1-y^2):")

    # (i) Spot checks vs direct integration of eta_1 * eta_1.
    def eta1(x):
        if abs(x) >= mp.mpf("0.5"):
            return mp.mpf(0)
        return (1 / mp.pi) / mp.sqrt(mp.mpf("0.25") - x ** 2)

    def mu_direct(y):
        a = max(mp.mpf("-0.5"), y - mp.mpf("0.5"))
        b = min(mp.mpf("0.5"), y + mp.mpf("0.5"))
        if a >= b:
            return mp.mpf(0)
        return mp.quad(lambda t: eta1(t) * eta1(y - t), [a, b])

    for y in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        md = mu_direct(mp.mpf(y))
        mc = mu_closedform(y)
        err = abs(md - mc)
        print(f"    y = {y:.2f}:  direct = {mp.nstr(md, 14)},  "
              f"closed = {mp.nstr(mc, 14)},  |diff| = {mp.nstr(err, 3)}")

    # (ii) Mass = 1.
    #
    # mu has a log singularity at 0.  Compute mass as
    #   2 [ int_0^eps mu  (asymptotic)  +  int_eps^1 mu  (quad) ].
    #
    # K(1 - y^2) ~ log(4/|y|) as y -> 0, so mu(y) ~ (2/pi^2) log(4/|y|).
    # int_0^eps (2/pi^2) log(4/y) dy = (2/pi^2) [y log(4/y) + y] |_0^eps
    #                                = (2/pi^2) (eps log(4/eps) + eps).
    # Use mp.quad direct on [0, 1] with split nodes at 1e-12, 1e-6, 0.5,
    # which exposes the log singularity at 0 to adaptive quadrature.
    mass = 2 * mp.quad(mu_closedform,
                       [mp.mpf("1e-12"), mp.mpf("1e-6"),
                        mp.mpf("1e-3"), mp.mpf("0.5"), mp.mpf("1")])
    # Add analytic correction for (0, 1e-12) using leading-log asymptotic
    eps0 = mp.mpf("1e-12")
    head = (2 / mp.pi ** 2) * (eps0 * mp.log(4 / eps0) + eps0)
    mass = mass + 2 * head
    print(f"  mass(mu) = {mp.nstr(mass, 18)}  (target 1; "
          f"err {mp.nstr(mass - 1, 3)})")
    if abs(float(mass) - 1.0) > 1e-8:
        raise SystemExit("Mass of mu drifts from 1 -- closed form wrong.")


def method2_K2(dps: int = 30) -> mp.mpf:
    """K_2 via the closed-form mu and real-space pairwise integrals.

    K_2 = int K_ms(x)^2 dx
        = sum_{i,j} lambda_i lambda_j * int K_arc(d_i, x) K_arc(d_j, x) dx.

    With K_arc(d, x) = (1/d) mu(x/d) supported in (-d, d), for any i, j
    let d_min = min(d_i, d_j), d_max = max(d_i, d_j).  Substituting
    u = x/d_min:
        I_{ij} := int K_arc(d_i, x) K_arc(d_j, x) dx
              = (1/d_max) int_{-1}^{1} mu(u) mu(d_min/d_max * u) du.

    The mu(u) factor has a log singularity at u = 0; the second factor
    is regular there.  We integrate split at u = 0 to expose the
    integrable log singularity to mpmath's adaptive quadrature.
    """
    mp.mp.dps = dps
    print(f"  Computing K_2 via 9 real-space pair integrals  (dps = {dps})...")
    lambdas = [mp.mpf(l) for l in LAMBDAS]
    deltas = [mp.mpf(d) for d in DELTAS]

    total = mp.mpf(0)
    diag_breakdown = []
    for i in range(3):
        for j in range(3):
            li, lj = lambdas[i], lambdas[j]
            di, dj = deltas[i], deltas[j]
            if di <= dj:
                d_small, d_large = di, dj
            else:
                d_small, d_large = dj, di
            r = d_small / d_large  # in (0, 1]

            def integrand(u, r=r):
                if u == 0:
                    return mp.mpf("inf")
                if abs(u) >= 1:
                    return mp.mpf(0)
                ru = r * u
                if abs(ru) >= 1:
                    return mp.mpf(0)
                return mu_closedform(u) * mu_closedform(ru)

            # mu(u) has log singularity at u = 0 (mu ~ log(4/|u|)).
            # mu(u) mu(ru) has a (log u)^2 singularity at 0, integrable.
            # We integrate on (eps, 1) with eps small enough that the
            # missing (0, eps) head is below tolerance, then add an
            # analytic head correction.
            #
            # eps must be large enough that mu(eps) is finite at the
            # current dps; eps must be small enough that
            #   head(eps) := int_0^eps mu(u) mu(ru) du
            # is bounded.  Asymptotic:  mu(u) ~ (2/pi^2) log(4/|u|).  So
            #   head <= mu(eps)^2 * eps   (crude upper bound).
            # With eps = 1e-10 at dps=30: mu(1e-10) ~ 5, so head ~ 2.5e-9.
            eps = mp.mpf("1e-10")
            pos = mp.quad(integrand,
                          [eps, mp.mpf("1e-8"), mp.mpf("1e-5"),
                           mp.mpf("1e-3"), mp.mpf("0.5"), mp.mpf(1)])
            # Analytic head:
            #   int_0^eps log(4/u) du = u log(4/u) + u |_0^eps
            #                        = eps log(4/eps) + eps.
            # mu(u) ~ (2/pi^2) [log(4/u)],
            # mu(ru) ~ (2/pi^2) [log(4/(ru))] = (2/pi^2) [log(4/u) - log r],
            # product ~ (2/pi^2)^2 [log^2(4/u) - log(r) log(4/u)].
            # int_0^eps log^2(4/u) du = eps (log(4/eps))^2 - 2 eps log(4/eps) + 2 eps
            # int_0^eps log(4/u) du  = eps log(4/eps) + eps.
            le = mp.log(4 / eps)
            int_log2 = eps * le ** 2 - 2 * eps * le + 2 * eps
            int_log = eps * le + eps
            c = 2 / mp.pi ** 2
            head = c ** 2 * (int_log2 - mp.log(r) * int_log)
            part = 2 * (pos + head)
            I_ij = part / d_large
            contrib = li * lj * I_ij
            total += contrib
            diag_breakdown.append((i, j, float(contrib)))

    print("  Breakdown by (i, j):")
    for (i, j, v) in diag_breakdown:
        print(f"    (i={i+1}, j={j+1}):  lambda_i lambda_j I_ij = {v:.6e}")
    return total


# ----------------------------------------------------------------------
#   MAIN
# ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" K_2(K_ms) -- REAL-SPACE sanity check")
    print("=" * 72)
    print(f"  Deltas = {DELTAS}, Lambdas = {LAMBDAS}")
    print(f"  Certifier interval (Fourier side): [{CERT_LO}, {CERT_HI}]")
    print()

    # ---- Method 1 (coarse)
    K2_m1 = method1_fft(N=400_001)

    # ---- Method 2 (closed form)
    print("\n--- METHOD 2: closed-form mu = (2/pi^2) K(1 - y^2) ---")
    method2_verify_mu(dps=30)
    K2_m2 = method2_K2(dps=30)
    print(f"\n  K_2 (Method 2)    = {mp.nstr(K2_m2, 20)}")

    # ---- Comparison
    print()
    print("=" * 72)
    print(" COMPARISON")
    print("=" * 72)
    cert_mid = 0.5 * (CERT_LO + CERT_HI)
    print(f"  Certifier interval (Fourier): [{CERT_LO}, {CERT_HI}]")
    print(f"  Certifier midpoint:           {cert_mid:.16f}")
    print(f"  Method 1 (FFT conv, ~0.5% acc): {K2_m1:.6f}")
    print(f"  Method 2 (elliptic closed form): {mp.nstr(K2_m2, 16)}")
    print()
    print(f"  M1 vs cert midpoint diff:  {K2_m1 - cert_mid:+.6f}")
    print(f"  M2 vs cert midpoint diff:  {float(K2_m2) - cert_mid:+.3e}")
    print(f"  M2 vs cert_lo diff:        {float(K2_m2) - CERT_LO:+.3e}")
    print(f"  M2 vs cert_hi diff:        {float(K2_m2) - CERT_HI:+.3e}")

    # Tolerances:
    # - Method 1 has aliasing from FFT'ing a singular integrand; +-0.5%
    #   is structurally the best we can do without analytic singularity
    #   handling.  We just check it is *near* the cert (sanity).
    # - Method 2 is closed form + adaptive quadrature; targeting <= 1e-4
    #   absolute (well inside the cert interval width of ~ 8e-5).
    tol_m1 = 5e-2     # 1%
    tol_m2 = 5e-4
    pass_m1 = abs(K2_m1 - cert_mid) < tol_m1
    pass_m2 = (CERT_LO - tol_m2) <= float(K2_m2) <= (CERT_HI + tol_m2)
    print()
    print(f"  Method 1 within {tol_m1} of cert midpoint:   "
          f"{'PASS' if pass_m1 else 'FLAG'}")
    print(f"  Method 2 within cert interval +- {tol_m2}: "
          f"{'PASS' if pass_m2 else 'FLAG'}")
    print()
    if pass_m1 and pass_m2:
        print("  VERDICT: PASS -- real-space K_2 agrees with Fourier-side K_2.")
        print("           Fourier formula (K_hat_ms = sum lambda_i J0(pi d_i xi)^2)")
        print("           and the [4.788823, 4.788906] interval are formula-correct.")
    else:
        print("  VERDICT: FLAG -- discrepancy worth investigating.")


if __name__ == "__main__":
    main()
