"""Hostile-referee independent reimplementation. No project imports."""
import json, hashlib
import mpmath as mp
from fractions import Fraction

mp.mp.dps = 100  # 100 decimal digits

# Multi-scale arcsine parameters from CLAUDE.md / certificate
lam = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]
dlt = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
u_q = Fraction(319, 500)  # 0.638
u_real = mp.mpf(u_q.numerator) / mp.mpf(u_q.denominator)
assert u_real == mp.mpf("0.638")

# --- 1) K_2 = 2 \int_0^infty Khat_ms(xi)^2 d xi, Khat_ms = sum lam_i J0(pi delta_i xi)^2
def Khat(xi):
    return sum(lam[i] * mp.besselj(0, mp.pi * dlt[i] * xi) ** 2 for i in range(3))

def integrand(xi):
    return Khat(xi) ** 2

# Piecewise adaptive quadrature on [0, T] then Watson tail.
T = mp.mpf(100)
# Break interval at oscillation scale of Bessel: zeros of J0(pi*0.138*xi) at xi ~ k * (1/0.138 / 1) ~ 7.246 k
# Use chunks of size 5
chunks = []
xi_low = mp.mpf(0)
step = mp.mpf("2.5")
while xi_low < T:
    xi_hi = min(xi_low + step, T)
    chunks.append((xi_low, xi_hi))
    xi_low = xi_hi

I_main = mp.mpf(0)
for (a, b) in chunks:
    val = mp.quad(integrand, [a, b])
    I_main += val

# Watson tail: |J0(z)|^2 <= 2/(pi z) for z>0.
# So Khat_ms(xi)^2 <= (sum lam_i * 2/(pi^2 delta_i xi))^2 = 4 C^2 / (pi^4 xi^2),
# C = sum lam_i / delta_i.
C = sum(lam[i] / dlt[i] for i in range(3))
# integral_T^infty 4 C^2 / (pi^4 xi^2) d xi = 4 C^2 / (pi^4 T)
tail_bound = 4 * C ** 2 / (mp.pi ** 4 * T)
print(f"C = sum lam_i/delta_i = {mp.nstr(C, 30)}")
print(f"Watson tail integrand bound: 4 C^2 / (pi^4 * T) at T={T}: {mp.nstr(tail_bound, 30)}")

# Doubled K_2:
K2_lo = 2 * I_main
K2_hi = 2 * (I_main + tail_bound)
print()
print(f"K_2 = 2 \int_0^infty Khat_ms^2 d xi (100 dps reimplementation):")
print(f"  Lower: 2 I_[0,T]                = {mp.nstr(K2_lo, 50)}")
print(f"  Upper: 2 I_[0,T] + 2*tail_bound = {mp.nstr(K2_hi, 50)}")
K2_claim = mp.mpf(47897) / mp.mpf(10000)
print(f"  Claimed Lean upper 47897/10000  = {mp.nstr(K2_claim, 50)}")
print(f"  Slack = 47897/10000 - K2_upper  = {mp.nstr(K2_claim - K2_hi, 20)}")
print()

# Cross-check certificate's mid_float:
cert_mid = mp.mpf("4.788864301446199")
print(f"Cert K_2 mid_float            = {cert_mid}")
print(f"|K2_mid_mpmath - cert_mid|    = {mp.nstr(abs((K2_lo+K2_hi)/2 - cert_mid), 10)}")

# --- 4) Watson tail for T=10^5
T_paper = mp.mpf("100000")
tail_paper = 8 * C ** 2 / (mp.pi ** 4 * T_paper)  # one-sided (factor 4) doubled = 8 (over both halves)
# Actually for K_2 = 2*int_0^infty, tail of K_2 itself is 2 * (4 C^2 / pi^4 / T) = 8 C^2 / (pi^4 T)
print()
print(f"Watson tail on K_2 itself (factor 8) at T=10^5: {mp.nstr(tail_paper, 20)}")
print(f"  <= 10^-4? {tail_paper < mp.mpf('1e-4')}")
