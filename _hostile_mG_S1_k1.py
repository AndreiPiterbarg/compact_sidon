"""Reproduce m_G, S_1, k_1 independently from JSON coefficients."""
import json
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 50

p = r'C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\delsarte_dual\grid_bound_alt_kernel\certificates\multiscale_arcsine_1292.json'
with open(p) as f:
    cert = json.load(f)
G = cert['body']['G']
coeffs_q = [Fraction(s) for s in G['coeffs_q']]
u_q = Fraction(G['u_q'])
u = mp.mpf(u_q.numerator)/mp.mpf(u_q.denominator)
n = G['n_coeffs']
assert n == 200
assert len(coeffs_q) == 200
print(f"u = {u_q} = {u}")
print(f"n_coeffs = {n}")

# Build G(x) = sum a_j cos(2 pi j x / u) for j=1..200.
a_mp = [mp.mpf(c.numerator)/mp.mpf(c.denominator) for c in coeffs_q]

def G_eval(x):
    s = mp.mpf(0)
    for j in range(1, n+1):
        s += a_mp[j-1] * mp.cos(2*mp.pi*j*x/u)
    return s

# --- m_G: minimize G on [0, 1/4].
# Dense sample + Brent local refinement.
N_sample = 100000
xs = [mp.mpf(k)/mp.mpf(N_sample) * mp.mpf("0.25") for k in range(N_sample+1)]
# Vectorize-ish: compute G at each sample. (Slow but correct.)
import time
t0 = time.time()
vals = []
for k in range(N_sample+1):
    if k % 20000 == 0:
        print(f"  G sample {k}/{N_sample}, t={time.time()-t0:.1f}s", flush=True)
    vals.append(G_eval(xs[k]))
i_min = min(range(len(vals)), key=lambda i: vals[i])
x_min = xs[i_min]
print(f"Coarse min at x = {x_min}, G(x_min) = {mp.nstr(vals[i_min], 25)}")

# Local refinement using mpmath's minimize via bracket
def G_neg_neg(x):
    return G_eval(mp.mpf(x))
# Refine around x_min with brent
lo = max(xs[max(0, i_min-1)], mp.mpf(0))
hi = min(xs[min(N_sample, i_min+1)], mp.mpf("0.25"))
# Golden-section refinement
phi = (mp.sqrt(5)-1)/2
a, b = lo, hi
for _ in range(80):
    x1 = b - phi*(b-a)
    x2 = a + phi*(b-a)
    if G_eval(x1) < G_eval(x2):
        b = x2
    else:
        a = x1
x_star = (a+b)/2
G_star = G_eval(x_star)
print(f"Refined min: x* = {mp.nstr(x_star, 25)}, G(x*) = {mp.nstr(G_star, 25)}")
print(f"  Cert min_G_lower = 0.9999798743824747")
print(f"  Cert cell center = 64915/262144 = {float(Fraction(64915,262144)):.15f}")

claim_mG = mp.mpf(998)/mp.mpf(1000)
print(f"  Lean axiom: m_G >= 998/1000 = 0.998")
print(f"  m_G - 0.998 = {mp.nstr(G_star - claim_mG, 12)}")

# --- S_1 = sum_j a_j^2 / Ktilde_ms(j),  Ktilde_ms(j) = lam_1 J0(pi j delta_1/u)^2 + lam_2 ... + lam_3 ...
lam = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]
dlt = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]

def Ktilde(j):
    s = mp.mpf(0)
    for i in range(3):
        s += lam[i] * mp.besselj(0, mp.pi * mp.mpf(j) * dlt[i] / u) ** 2
    return s

S1 = mp.mpf(0)
for j in range(1, n+1):
    w = Ktilde(j)
    S1 += a_mp[j-1]**2 / w
print()
print(f"S_1 (independent) = {mp.nstr(S1, 25)}")
print(f"  Cert S_1 mid     = 29.840906455513263")
S1_claim = mp.mpf(29841)/mp.mpf(1000)
print(f"  Lean axiom: S_1 <= 29841/1000 = 29.841")
print(f"  29.841 - S_1 = {mp.nstr(S1_claim - S1, 12)}")

# --- k_1 = Khat_ms(1) [period-u Fourier coefficient at j=1]
k1 = Ktilde(1)
print()
print(f"k_1 = Ktilde_ms(1) = {mp.nstr(k1, 25)}")
print(f"  Cert k_1 mid = 0.9212465899364083")
k1_claim = mp.mpf(9212)/mp.mpf(10000)
print(f"  Lean axiom: k_1 >= 9212/10000 = 0.9212")
print(f"  k_1 - 0.9212 = {mp.nstr(k1 - k1_claim, 12)}")

# --- gain a = (4/u) * m_G^2 / S_1
u_real = u
gain = (4/u_real) * G_star**2 / S1
print()
print(f"gain a = (4/u) * m_G^2 / S_1 = {mp.nstr(gain, 25)}")
gain_claim = mp.mpf(20925)/mp.mpf(100000)
print(f"  Cert gain mid = 0.21009214748668373")
print(f"  Lean axiom: gain >= 20925/100000 = 0.20925")
print(f"  gain - 0.20925 = {mp.nstr(gain - gain_claim, 12)}")

# --- Master Phi(M):
# Phi(M) = M + 1 + sqrt((M-1-2y^2)(K_2 - 1 - 2k_1^2)) - 2/u - a (per CLAUDE.md form)
# But the CLAUDE.md `phi_N1` formula:
# Phi(M, y) = M + 1 + 2 y k_1 + sqrt((M-1-2y^2)(K_2-1-2k_1^2)) - (2/u + a)
# At the closing point we need an outer minimization over y. The certifier does cell search.
# We can at least verify the simpler closing form at y=0:
M = mp.mpf(1292)/mp.mpf(1000)
K2 = mp.mpf(47897)/mp.mpf(10000)
# Try y=0:
phi_y0 = M + 1 + 0 + mp.sqrt((M-1)*(K2-1-2*k1**2)) - (2/u_real + gain_claim)
print()
print(f"Phi(M=1.292, y=0; K2=4.7897, k1>=0.9212, gain>=0.20925) =")
print(f"  Phi = {mp.nstr(phi_y0, 12)}")
# y*: maximizes 2 y k_1 + sqrt((M-1-2y^2)(K_2-1-2k_1^2))
# d/dy: 2 k_1 - 2y * sqrt(K_2-1-2k_1^2) / sqrt(M-1-2y^2) = 0
# => y * sqrt(K_2-1-2k_1^2) = k_1 * sqrt(M-1-2y^2)
# => y^2 * (K_2-1-2k_1^2) = k_1^2 (M-1-2y^2)
# => y^2 * (K_2-1-2k_1^2 + 2 k_1^2) = k_1^2 (M-1)
# => y^2 * (K_2-1) = k_1^2 (M-1)
# => y* = k_1 * sqrt((M-1)/(K_2-1))
y_star = k1 * mp.sqrt((M-1)/(K2-1))
print(f"  y* = k_1 sqrt((M-1)/(K_2-1)) = {mp.nstr(y_star, 20)}")
phi_max = M + 1 + 2*y_star*k1 + mp.sqrt(max(M-1-2*y_star**2, mp.mpf(0)) * (K2-1-2*k1**2)) - (2/u_real + gain_claim)
print(f"  Phi at y* = {mp.nstr(phi_max, 12)}")
# Alternative form combining: max over y is 1 + sqrt((M-1)(K_2-1)) form?
# Actually after maximization: 2 y* k_1 + sqrt((M-1-2y*^2)(K_2-1-2k_1^2))
# At optimum: sqrt((M-1-2y*^2)/(K_2-1)) * (K_2-1-2k_1^2 + 2 k_1^2)/sqrt(...)... let's just plug in.
# Actually closed form is: max_y = sqrt((M-1)(K_2-1))?
# Test: with y* as above, evaluate.
print()
print("Phi(1.292) sign check — if < 0, M=1.292 is forbidden, i.e., C_{1a} >= 1.292.")
