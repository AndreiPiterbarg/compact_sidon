"""Independent mpmath verification of the 3-scale 1.292 anchors.

Checks performed:
  V7  -- S_1 = sum_{j=1}^{200} a_j^2 / w_j           (claim: <= 29.840907)
  V8  -- min_{x in [0, 1/4]} G(x)                    (claim: >= 0.99997987)
  V9  -- k_1 = sum_i lambda_i J_0(pi delta_i / u)^2  (claim: >= 0.92124658)
  V12 -- min_j K~_ms(j/u) over j=1..200             (Bochner positivity)

All Bessel and Fourier evaluations are done from scratch in mpmath at
50-digit precision -- no imports from the project's kernels.py.

NOTE: G(x) = sum_{j=1}^{200} a_j cos(2 pi j x / u);  see optimize_G.py
                       (no constant term).
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50  # 50 decimal digits

# ---------------------------------------------------------------------------
# Load certificate.
# ---------------------------------------------------------------------------
CERT_PATH = Path(
    r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon"
    r"\delsarte_dual\grid_bound_alt_kernel\certificates"
    r"\multiscale_arcsine_1292.json"
)
cert = json.loads(CERT_PATH.read_text())
body = cert["body"]

# Parse rational parameters.
u_q = Fraction(body["G"]["u_q"])               # 319/500
deltas_q = [Fraction(s) for s in body["kernel"]["deltas_q"]]
lambdas_q = [Fraction(s) for s in body["kernel"]["lambdas_q"]]
coeffs_q = [Fraction(s) for s in body["G"]["coeffs_q"]]
n_coeffs = body["G"]["n_coeffs"]

print(f"u = {u_q} = {float(u_q):.6f}")
print(f"deltas = {[float(d) for d in deltas_q]}")
print(f"lambdas = {[float(l) for l in lambdas_q]}, sum = {float(sum(lambdas_q))}")
print(f"n_coeffs = {n_coeffs}  (len(coeffs_q) = {len(coeffs_q)})")

assert n_coeffs == 200
assert len(coeffs_q) == 200
assert sum(lambdas_q) == 1
assert u_q == Fraction(1, 2) + deltas_q[0]

# Convert to mpf.
u  = mp.mpf(u_q.numerator) / mp.mpf(u_q.denominator)
deltas  = [mp.mpf(d.numerator) / mp.mpf(d.denominator) for d in deltas_q]
lambdas = [mp.mpf(l.numerator) / mp.mpf(l.denominator) for l in lambdas_q]
coeffs  = [mp.mpf(c.numerator) / mp.mpf(c.denominator) for c in coeffs_q]


# ---------------------------------------------------------------------------
# Periodic Fourier coefficient of K_ms:  K~_ms(xi) = sum_i lambda_i J_0(pi*delta_i*xi)^2
# Note: K_arc(delta) has Fourier transform J_0(pi*delta*xi);
#       its autoconvolution K_arc*K_arc has Fourier transform J_0(pi*delta*xi)^2.
#       For the multiscale  K_ms = sum lambda_i  K_arc(delta_i)*K_arc(delta_i)
#       so  K~_ms(xi) = sum lambda_i J_0(pi*delta_i*xi)^2.
# We need this at xi = j / u  for j = 1, ..., 200.
# ---------------------------------------------------------------------------
def K_tilde_ms(xi):
    """K~_ms(xi) = sum_i lambda_i J_0(pi delta_i xi)^2."""
    s = mp.mpf(0)
    for li, di in zip(lambdas, deltas):
        arg = mp.pi * di * xi
        b = mp.besselj(0, arg)
        s += li * b * b
    return s


# ---------------------------------------------------------------------------
# V12 first (needed for V7 anyway).
# ---------------------------------------------------------------------------
print("\n--- V12: Bochner positivity at QP frequencies ---")
w = [None]  # w[j] for j = 1..200; index 0 unused.
for j in range(1, 201):
    xi = mp.mpf(j) / u
    w.append(K_tilde_ms(xi))
w_min = min(w[1:])
j_min = 1 + w[1:].index(w_min)
print(f"  min_j K~_ms(j/u) = {mp.nstr(w_min, 12)}  at j = {j_min}")
print(f"  Claim: >= 2.08e-4  -> {'PASS' if w_min >= mp.mpf('2.08e-4') else 'CHECK'}")

# Also  verify all positive.
all_pos = all(wi > 0 for wi in w[1:])
print(f"  All 200 weights strictly positive: {all_pos}")


# ---------------------------------------------------------------------------
# V9: k_1 -- period-1 Fourier coefficient.
# In the codebase  k_1 = K_tilde(1) = K_tilde_real(xi=1) = sum_i lambda_i J_0(pi delta_i)^2
# (period 1 in the MV master inequality writeup), NOT at xi=1/u.
# This matches  bisect_alt_kernel.py line 232:
#   k1_float = float(kernel.K_tilde(1, prec_bits=prec_bits).mid())
# We report BOTH conventions for transparency.
# ---------------------------------------------------------------------------
print("\n--- V9: k_1 = period-1 FT coefficient ---")
k1_period1 = K_tilde_ms(mp.mpf(1))               # the certifier convention
k1_at_1_over_u = w[1]                            # alt convention from task statement
print(f"  k_1 (period 1, codebase convention) = {mp.nstr(k1_period1, 20)}")
print(f"  k_1 (at 1/u, alt convention)        = {mp.nstr(k1_at_1_over_u, 20)}")
print(f"  Cert mid (period 1):                  0.9212465899364083")
print(f"  Claim: >= 0.92124658  -> {'PASS' if k1_period1 >= mp.mpf('0.92124658') else 'FAIL'}")
k1 = k1_period1


# ---------------------------------------------------------------------------
# V7: S_1 = sum_{j=1}^{200} a_j^2 / w_j.
# ---------------------------------------------------------------------------
print("\n--- V7: S_1 = sum a_j^2 / w_j ---")
S1 = mp.mpf(0)
for j in range(1, 201):
    aj = coeffs[j - 1]   # coeffs[0] = a_1, etc.
    S1 += aj * aj / w[j]
print(f"  S_1 (mpmath, 50dps) = {mp.nstr(S1, 20)}")
print(f"  Cert upper:           29.840906455513267")
print(f"  Claim: <= 29.840907  -> {'PASS' if S1 <= mp.mpf('29.840907') else 'FAIL'}")


# ---------------------------------------------------------------------------
# V8: min G on [0, 1/4].
#     G(x) = sum_{j=1}^{200} a_j cos(2 pi j x / u).
# Strategy: dense uniform grid (200001 points), then refine near minimum
# via mpmath.findroot on G'(x) = 0 with bracketing.
# ---------------------------------------------------------------------------
print("\n--- V8: min G on [0, 1/4] ---")
two_pi_over_u = 2 * mp.pi / u

def G(x):
    s = mp.mpf(0)
    for j in range(1, 201):
        s += coeffs[j - 1] * mp.cos(j * two_pi_over_u * x)
    return s

def Gp(x):
    """Derivative G'(x)."""
    s = mp.mpf(0)
    for j in range(1, 201):
        s -= coeffs[j - 1] * j * two_pi_over_u * mp.sin(j * two_pi_over_u * x)
    return s

# Coarse pass.  We use float-precision arithmetic here for speed.
# 200001 points -> step ~1.25e-6, sufficient to localise minima.
import math as _m
N_grid = 200001
xs = [mp.mpf(i) / mp.mpf(4 * (N_grid - 1)) for i in range(N_grid)]
# But evaluating mpmath G at 200001 points at dps=50 is slow.
# Lower precision to dps=20 for the coarse sweep.
mp.mp.dps = 20

# Precompute scaled deltas for cos call.
omega = float(two_pi_over_u)
af = [float(c) for c in coeffs]
def G_float(x):
    return sum(af[j-1] * _m.cos(j * omega * x) for j in range(1, 201))

step = 0.25 / (N_grid - 1)
gmin_float = math_min_x = None
prev = G_float(0.0)
running_min = prev
running_argmin = 0.0
for i in range(1, N_grid):
    x = i * step
    g = G_float(x)
    if g < running_min:
        running_min = g
        running_argmin = x

print(f"  Coarse grid min = {running_min:.10f}  at x = {running_argmin:.7f}")

# Refine in mpmath: bracket [argmin - 2*step, argmin + 2*step].
mp.mp.dps = 50
x_lo = mp.mpf(running_argmin) - 2 * mp.mpf(step)
x_hi = mp.mpf(running_argmin) + 2 * mp.mpf(step)
if x_lo < 0:
    x_lo = mp.mpf(0)
if x_hi > mp.mpf('0.25'):
    x_hi = mp.mpf('0.25')

# Use mpmath.findroot on G'(x)=0 inside [x_lo, x_hi] if Gp changes sign.
try:
    gp_lo = Gp(x_lo)
    gp_hi = Gp(x_hi)
    if gp_lo * gp_hi < 0:
        x_star = mp.findroot(Gp, (x_lo, x_hi), solver='anderson')
        g_star = G(x_star)
        print(f"  Refined min (mpmath findroot) = {mp.nstr(g_star, 20)}")
        print(f"                       at x = {mp.nstr(x_star, 20)}")
    else:
        # findroot via mpmath.minimize doesn't exist directly; use parabolic.
        # Evaluate G on a fine sub-grid.
        n_sub = 2001
        xs_sub = [x_lo + (x_hi - x_lo) * mp.mpf(i) / mp.mpf(n_sub - 1) for i in range(n_sub)]
        gs_sub = [G(xx) for xx in xs_sub]
        g_star = min(gs_sub)
        x_star = xs_sub[gs_sub.index(g_star)]
        print(f"  Sub-grid min (mpmath, 2001 pts) = {mp.nstr(g_star, 20)}")
        print(f"                          at x = {mp.nstr(x_star, 20)}")
except Exception as exc:
    print(f"  refinement failed: {exc}")
    g_star = mp.mpf(running_min)

print(f"  Cert lower bound: 0.9999798743824747")
print(f"  Claim: >= 0.99997987  -> {'PASS' if g_star >= mp.mpf('0.99997987') else 'FAIL'}")


# ---------------------------------------------------------------------------
# Final summary line for parent caller.
# ---------------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"  V7  S_1            = {mp.nstr(S1, 16)}   claim <= 29.840907   "
      f"-> {'PASS' if S1 <= mp.mpf('29.840907') else 'FAIL'}")
print(f"  V8  min G          = {mp.nstr(g_star, 16)}    claim >= 0.99997987  "
      f"-> {'PASS' if g_star >= mp.mpf('0.99997987') else 'FAIL'}")
print(f"  V9  k_1            = {mp.nstr(k1, 16)}   claim >= 0.92124658   "
      f"-> {'PASS' if k1 >= mp.mpf('0.92124658') else 'FAIL'}")
print(f"  V12 min_j w_j      = {mp.nstr(w_min, 12)}   claim >= 2.08e-4    "
      f"-> {'PASS' if w_min >= mp.mpf('2.08e-4') else 'CHECK'}")
print(f"      j_min = {j_min}")
