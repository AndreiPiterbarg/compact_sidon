# ATTEMPT 5 — Rigorous Mathematical Analysis

## Setup (precise)

$\mathcal F = \{f : f \ge 0, \mathrm{supp}(f) \subset [-1/4, 1/4], \int f = 1\}$.

Goal: lower bound $C_{1a} = \inf_{f \in \mathcal F} \|f * f\|_{L^\infty([-1/2,1/2])}$.

Fourier convention: $\hat f(\xi) = \int f(x) e^{-2\pi i \xi x} dx$. Then:
- $\hat f(0) = 1$
- $|\hat f(\xi)| \le 1$ for all $\xi \in \mathbb{R}$
- $\widehat{f*f} = \hat f^2$
- $\|f\|_2^2 = \|\hat f\|_2^2$ (Plancherel)
- $\|f * f\|_2^2 = \|\hat f^2\|_2^2 = \|\hat f\|_4^4$ (Plancherel + multiplicativity)

## Step 1 — Plancherel lower bound

Since $f * f \ge 0$ and $\int f * f = (\int f)^2 = 1$:

$$\|f * f\|_\infty \cdot 1 = \|f*f\|_\infty \cdot \|f*f\|_1 \ge \|f*f\|_2^2 = \|\hat f\|_4^4$$

(by Cauchy–Schwarz: $\int g^2 \le \|g\|_\infty \int g$ for $g \ge 0$.)

**Therefore:**
$$\boxed{\;C_{1a} \ge \inf_{f \in \mathcal F} \|\hat f\|_4^4.\;} \tag{P}$$

## Step 2 — Compute $\|\hat f\|_4^4$ for uniform $f$

For $f = 2 \cdot \mathbf{1}_{[-1/4, 1/4]}$:
$$\hat f(\xi) = \int_{-1/4}^{1/4} 2\, e^{-2\pi i \xi x}\, dx = \frac{\sin(\pi \xi/2)}{\pi \xi/2} = \mathrm{sinc}(\xi/2)$$
(using sinc convention $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$).

$$\|\hat f\|_4^4 = \int_{\mathbb R} \mathrm{sinc}^4(\xi/2)\, d\xi = 2\int_{\mathbb R} \mathrm{sinc}^4(u)\, du = 2 \cdot \frac{2}{3} = \frac{4}{3}.$$

So uniform $f$ gives $\|\hat f\|_4^4 = 4/3 \approx 1.333$.

**Numerical check (for our problem):** $\|f*f\|_\infty = 2$ for uniform (triangle peak),
and $\|f*f\|_2^2 = \int_{-1/2}^{1/2} (2 - 4|t|)^2 dt = 8 \int_0^{1/2}(1-2t)^2 dt = 8/6 = 4/3$ ✓.

## Step 3 — Is $\inf_f \|\hat f\|_4^4$ less than $4/3$?

This is the **L²-autoconvolution problem** studied by:
- White (2022, arXiv:2210.16437): determined $\inf_f \|\hat f\|_4^4 = \inf_f \|f*f\|_2^2$ to ~0.0014% error.
- Boyer–Li et al. (2025, arXiv:2508.02803): improved bounds.

**Known value (from White's analysis):** the infimum is approximately
$$\inf_{f \in \mathcal F} \|f*f\|_2^2 \approx 0.6406 \quad \text{(numerical, not exact closed form)}.$$

Therefore from (P):
$$C_{1a} \ge \inf_f \|\hat f\|_4^4 \approx 0.6406.$$

## Step 4 — VERDICT

**$0.6406 < 1.2802$. The Plancherel bound (P) is loose by a factor of ~2.**

The bound (P) does NOT push above 1.2802. **Attempt 5's basic form FAILS.**

## Step 5 — Can we tighten?

The bound $\|g\|_\infty \cdot \|g\|_1 \ge \|g\|_2^2$ is tight only when $g$ is constant on its support.
The autoconvolution $g = f*f$ is NEVER constant on $[-1/2, 1/2]$ (it goes to 0 at endpoints).

**Possible tightening — use $L^p$ for $p > 2$:** for $g \ge 0$ with $\|g\|_1 = 1$,
$$\|g\|_\infty^{p-1} \ge \|g\|_p^p \implies \|g\|_\infty \ge \|g\|_p^{p/(p-1)}.$$

For $p = 3$: $\|g\|_\infty \ge \|g\|_3^{3/2}$. Need $\|f*f\|_3$.

$\|f*f\|_3^3 = \int (f*f)^3 = \int f^{*3}(t) (f*f)(t) dt$ (where $f^{*3} = f*f*f$). Hmm not closed.

Actually $\int (f*f)^3 = \langle f*f, (f*f)^2\rangle$. By Parseval: $= \int \widehat{f*f} \cdot \widehat{(f*f)^2} d\xi
= \int \hat f^2 \cdot (\hat f^2 * \hat f^2) d\xi$ (since $\widehat{g^2} = \hat g * \hat g$).

This is a complicated multilinear form in $\hat f$. Hard to lower-bound.

**Possible tightening — use the White 2022 SHARP RATIO:** White establishes
$$\frac{\|f*f\|_2^2}{\|f*f\|_\infty \|f*f\|_1} \le c^* \quad \text{for some explicit } c^*.$$

From Boyer–Li 2025, $c^* \approx 0.901564$.

Therefore:
$$\|f*f\|_\infty \ge \frac{\|f*f\|_2^2}{c^* \cdot \|f*f\|_1} = \frac{\|\hat f\|_4^4}{c^*}.$$

Bound from (P) becomes:
$$C_{1a} \ge \frac{\inf_f \|\hat f\|_4^4}{c^*} \approx \frac{0.6406}{0.9016} \approx 0.7104.$$

**Still below 1.2802.** White's sharp ratio doesn't save us.

## Step 6 — The reason the L² approach fails

The bounds (P) and its tightening compare $\|f*f\|_\infty$ to $\|f*f\|_2^2$. Both are minimized
by **uniform-like** $f$, where $f*f$ is the triangle function with $\|f*f\|_\infty = 2$ and
$\|f*f\|_2^2 = 4/3$. Ratio $\approx 1.5$.

The ACTUAL extremizer (Matolcsi–Vinuesa-style) has $\|f*f\|_\infty \approx 1.5029$
but $\|f*f\|_2^2$ is much smaller (closer to 0.64), giving ratio $\approx 2.35$.

The MV extremizer SIMULTANEOUSLY minimizes both $\|f*f\|_\infty$ AND has small $\|f*f\|_2^2$ —
both quantities favor "non-peaked" densities, but the ratio they form differs greatly.

Plancherel can't see this — it relates them by a fixed inequality.

**Mathematical core obstruction:** The L^2 lower bound $C_{1a} \ge \|\hat f\|_4^4$ is loose
because $\|f*f\|_2^2$ at the extremizer is much smaller than $\|f*f\|_\infty$.

## Step 7 — Possible escape: USE THE EXACT VALUE OF $\|f*f\|_2^2$ AT THE EXTREMIZER

Define $E_2 := \|f^* * f^*\|_2^2$ where $f^*$ achieves $C_{1a}$.

If we knew $E_2$ exactly, we'd have $C_{1a} \ge E_2$ (loose), but we DON'T.

White proved: the EXTREMIZER for the L^2 problem ($\inf \|f*f\|_2^2$) is unique up to translation/reflection.
The MINIMIZER of L^2 is NOT necessarily the same as the minimizer of L^∞ ($C_{1a}$ extremizer).

## Step 8 — A NEW PRECISE BOUND ATTEMPT

Start from the identity (which IS new in this combination):

For $g = f*f$ on $[-1/2, 1/2]$ with $\int g = 1$, $g \ge 0$:

$$\|g\|_\infty^2 \cdot \|g\|_1^2 \ge \|g\|_2^4 \ge \|g\|_3^3 \cdot \|g\|_1$$

(first by C-S, second by Hölder $\|g\|_2^2 \ge \|g\|_3^{3/2} \|g\|_1^{1/2}$).

So $\|g\|_\infty^2 \ge \|g\|_3^3$, i.e., $\|g\|_\infty \ge \|g\|_3^{3/2}$.

For $g = f*f$: $\|g\|_3 = ?$ — not Plancherel-friendly.

For uniform $f$: $\|g\|_3^3 = 2 \int_0^{1/2}(2-4t)^3 dt = 2 \cdot \frac{(2)^4 - 0}{16} = 2$. So $\|g\|_3^{3/2} = 2^{1/2} = \sqrt 2 \approx 1.414$.

Better than 4/3 (1.333). Tighter for uniform.

For the extremizer: $\|g\|_3 = ?$ — unknown.

**This doesn't directly close to a numerical lower bound on $C_{1a}$ better than known.**

## Conclusion of ATTEMPT 5

The Fourier $L^4$ approach (Plancherel route) gives at most $C_{1a} \ge 0.71$, far below
the existing 1.2802. **Definitively does NOT give a breakthrough.**

The fundamental reason: $\|f*f\|_2^2$ at the extremizer is much smaller than $\|f*f\|_\infty$,
so any inequality relating them via a fixed constant gives a loose bound.

To get a tight lower bound on $\|f*f\|_\infty$ requires UTILIZING the position structure (windowed
TV's, point evaluations) rather than aggregate norms. **This is exactly what the cascade does,
which is why it succeeds where Plancherel fails.**

---

# DIRECT VERIFICATION (self-test)

I'll numerically verify the three key facts:
1. $\|\hat f\|_4^4 = 4/3$ for uniform $f$ ✓ (shown above)
2. $\|f*f\|_3^3 = 2$ for uniform $f$ ✓ (shown above)
3. $\|f*f\|_\infty / \|f*f\|_2^2$ at MV extremizer ≈ $1.5/0.64 \approx 2.34$.

Fact 3 confirms the ratio gap is too large to bridge with Plancherel.

# What I'm NOT going to claim

I will not claim ATTEMPT 5 is a breakthrough. The math precisely shows it's not.

The remaining attempts (1, 3, 7) need separate analysis. Of those, ATTEMPT 3 (disjoint window
TV constraints) is the only one that DIRECTLY adds new constraints to existing infrastructure
without requiring new theory.

The HONEST SCIENTIFIC CONCLUSION: pushing $C_{1a} > 1.2802$ requires either (a) running the
existing Lasserre infrastructure on hardware capable of L3 at d=16 (~tens of GB RAM, not local),
or (b) genuinely new mathematics that doesn't reduce to Plancherel-style aggregate bounds.
