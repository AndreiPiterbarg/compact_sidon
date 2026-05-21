"""Hostile referee — K_2 with mpmath at 50 dps. T = 10^5, Watson tail < 8.2e-5."""
import mpmath as mp
mp.mp.dps = 50

lam = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]
dlt = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]

def Khat(xi):
    s = mp.mpf(0)
    for i in range(3):
        s += lam[i] * mp.besselj(0, mp.pi * dlt[i] * xi) ** 2
    return s

def integrand(xi):
    return Khat(xi) ** 2

# Bessel J0 zeros are at z_k ~ (k - 1/4) pi.
# J0(pi delta xi)^2 has zeros where pi delta xi = z_k => xi = z_k/(pi delta).
# Smallest delta = 0.025 => xi-spacing ~ 1/0.025 = 40.
# Largest delta = 0.138 => xi-spacing ~ 1/0.138 ~ 7.25.
# Use chunks of size ~7 for [0, 100], then ~40 beyond.
chunks = []
xi = mp.mpf(0)
while xi < 100:
    nxt = min(xi + 7, mp.mpf(100))
    chunks.append((xi, nxt))
    xi = nxt
while xi < 1000:
    nxt = min(xi + 50, mp.mpf(1000))
    chunks.append((xi, nxt))
    xi = nxt
while xi < 100000:
    nxt = min(xi + 1000, mp.mpf(100000))
    chunks.append((xi, nxt))
    xi = nxt

# Sum the integrand over each chunk using tanh-sinh.
I_total = mp.mpf(0)
for (a, b) in chunks:
    val = mp.quad(integrand, [a, b])
    I_total += val

K2_lo = 2 * I_total

# Watson tail beyond T = 10^5:
T = mp.mpf("100000")
C = sum(lam[i] / dlt[i] for i in range(3))
tail = 8 * C ** 2 / (mp.pi ** 4 * T)
K2_hi = K2_lo + tail

print(f"50-dps independent K_2 reimplementation (T={T}):")
print(f"  K_2 lower (no tail):  {mp.nstr(K2_lo, 40)}")
print(f"  K_2 upper (+ Watson): {mp.nstr(K2_hi, 40)}")
print(f"  Watson tail:          {mp.nstr(tail, 20)}")
print()
K2_claim = mp.mpf(47897)/mp.mpf(10000)
print(f"Claimed Lean upper 47897/10000 = {K2_claim}")
print(f"Slack (claim - upper) = {mp.nstr(K2_claim - K2_hi, 12)}")
print(f"Cert mid_float = 4.788864301446199")
print(f"Cert repr      = [4.7889 +/- 7.66e-5]")
print()
# Compare to cert bracket [4.7888234212591545, 4.788905181633242]:
print("Comparison to certificate brackets:")
print(f"  Cert lower:   4.7888234212591545")
print(f"  Cert upper:   4.788905181633242")
print(f"  mpmath K2_lo: {mp.nstr(K2_lo, 30)}")
print(f"  mpmath K2_hi: {mp.nstr(K2_hi, 30)}")
print(f"  Inside?       {mp.mpf('4.7888234212591545') <= K2_lo and K2_hi <= mp.mpf('4.788905181633242') + tail}")
