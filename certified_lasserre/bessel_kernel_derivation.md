# Rechnitzer-Style Bessel Ansatz: Closed-Form Convolution Kernel $K_{jm}(t)$

**Target:** Implement $f(x) = \sum_j a_j\,\varphi_j(x)$ with $\varphi_j(x) = (1-16x^2)^{j-1/2}\,\mathbf{1}_{[-1/4,1/4]}(x)$, so that
$(f*f)(t) = a^\top K(t)\, a$ with $K_{jm}(t) = (\varphi_j*\varphi_m)(t)$.

Rechnitzer (arXiv:2602.07292) works on $[-1/2,1/2]$ with $(1-4x^2)^{j-1/2}$. For the Sidon
autocorrelation problem we need supp $f \subseteq [-1/4,1/4]$ so we rescale $x\mapsto 2x$,
replacing $(1-4x^2)^{j-1/2}$ by $(1-16x^2)^{j-1/2}$.

--------------------------------------------------------------------------------

## 1. Fourier transform of $\varphi_j$ — Poisson / Sonine derivation

We use the standard (symmetric-measure) convention
$\hat f(\xi) = \int_{\mathbb R} f(x)\, e^{-2\pi i \xi x}\, dx$.

**Poisson integral** (Watson, *Theory of Bessel Functions*, 2nd ed., §3.3, eq. (5);
equivalently Gradshteyn–Ryzhik 8.411.8), valid for $\Re\nu > -\tfrac12$:

$$
J_\nu(z) \;=\; \frac{(z/2)^{\nu}}{\sqrt{\pi}\,\Gamma(\nu+\tfrac12)}
\int_{-1}^{1} (1-u^{2})^{\nu-\tfrac12}\,e^{izu}\,du.
$$

Substitute $u = 4x$ ($du = 4\,dx$):

$$
\hat\varphi_j(\xi)
= \int_{-1/4}^{1/4}(1-16x^{2})^{j-\tfrac12}\,e^{-2\pi i\xi x}dx
= \tfrac14\!\int_{-1}^{1}(1-u^{2})^{j-\tfrac12}\,e^{-i\pi\xi u/2}\,du.
$$

Identify $\nu=j$, $z=-\pi\xi/2$ in Poisson's integral. Because the integrand on the
right is even in $u$ under $u\mapsto -u$ combined with $z\mapsto -z$, and $J_\nu$ is
real analytic on $\mathbb R$, we obtain the real expression

$$
\boxed{\;\hat\varphi_j(\xi)
\;=\; \frac{\sqrt{\pi}\,\Gamma(j+\tfrac12)\cdot 4^{\,j-1}}{(\pi\xi)^{j}}\;J_{j}\!\bigl(\tfrac{\pi\xi}{2}\bigr)
\;=:\; c_j\,\frac{J_{j}(\pi\xi/2)}{(\pi\xi)^{j}}\;}
$$

with constant

$$
c_j \;=\; \sqrt{\pi}\,\Gamma(j+\tfrac12)\,4^{\,j-1}.
$$

Sanity check at $\xi\to 0$: $J_j(z)\sim (z/2)^j/\Gamma(j+1)$, so
$\hat\varphi_j(0) = c_j\cdot \frac{(\pi/4)^{\,j}/\Gamma(j+1)}{\pi^j} \cdot 1
= \frac{\sqrt{\pi}\,\Gamma(j+\tfrac12)\,4^{j-1}}{4^{\,j}\,\Gamma(j+1)}
= \frac{\sqrt{\pi}\,\Gamma(j+\tfrac12)}{4\,\Gamma(j+1)} = \beta_j$, matching the direct
beta-function computation below. ✓

--------------------------------------------------------------------------------

## 2. The normalisation integrals $\beta_j = \int \varphi_j$

Substitute $u = 16x^{2}$:

$$
\beta_j \;=\; \int_{-1/4}^{1/4}(1-16x^{2})^{j-\tfrac12}dx
= \frac14\,B(\tfrac12,\,j+\tfrac12)
= \boxed{\;\frac{\sqrt{\pi}\,\Gamma(j+\tfrac12)}{4\,\Gamma(j+1)}\;}.
$$

Numerically: $\beta_0 = \pi/4$, $\beta_1 = \pi/8$, $\beta_2 = \pi/16$, $\beta_j = (2j-1)/(2j)\cdot \beta_{j-1}$.

--------------------------------------------------------------------------------

## 3. Convolution: three equivalent closed forms

By Parseval / convolution theorem (both factors real and even in $\xi$):

$$
K_{jm}(t) = (\varphi_j*\varphi_m)(t)
= \int_{-\infty}^{\infty}\hat\varphi_j(\xi)\hat\varphi_m(\xi)\,e^{2\pi i\xi t}\,d\xi
= 2\int_0^\infty \hat\varphi_j\hat\varphi_m\cos(2\pi\xi t)\,d\xi.
$$

Plugging in and substituting $s = \pi\xi/2$:

$$
\boxed{\;K_{jm}(t)\;=\;\frac{4\,c_j c_m}{\pi\,2^{j+m}}
\int_0^\infty\!\!\frac{J_j(s)\,J_m(s)}{s^{j+m}}\,\cos(4ts)\,ds.\;}
\tag{A}
$$

This is a **Lommel / Weber–Schafheitlin** integral (Watson §13.46, G&R 6.681). A
closed form exists in terms of Gegenbauer polynomials / hypergeometric
functions, but the form is piecewise in $|4t|\lessgtr 2$ with differing Gegenbauer
arguments in each regime and is cumbersome to ball-evaluate accurately for large
$j+m$.

**Equivalent form (support and direct convolution).** Because $\operatorname{supp}\varphi_j
\subseteq [-\tfrac14,\tfrac14]$, the convolution is automatically supported on
$[-\tfrac12,\tfrac12]$:

$$
\boxed{\;K_{jm}(t) \;=\; \int_{a(t)}^{b(t)}
(1-16x^{2})^{j-\tfrac12}\,(1-16(t-x)^{2})^{m-\tfrac12}\,dx\;}
\tag{B}
$$
with $a(t)=\max(-\tfrac14,\,t-\tfrac14)$, $b(t)=\min(\tfrac14,\,t+\tfrac14)$,
and $K_{jm}(t)\equiv 0$ for $|t|\ge 1/2$.

Form (B) is a finite Riemann integral of an elementary function over a compact
interval. For $j,m\ge 1$ the integrand is continuous and bounded; for $j=0$ or
$m=0$ it has integrable algebraic endpoint singularities of type $s^{-1/2}$,
which Arb's adaptive certified integrator (`flint.acb.integral`) handles
rigorously (it returns an Arb ball enclosing the true value).

**At $t=0$ (closed form).** The cross-term collapses:

$$
K_{jm}(0)
= \int_{-1/4}^{1/4}(1-16x^{2})^{j+m-1}dx
= \boxed{\;\frac{\sqrt{\pi}\,\Gamma(j+m)}{4\,\Gamma(j+m+\tfrac12)}\;}.
\tag{C}
$$

This is exact in Arb arithmetic via gamma-function evaluation.

--------------------------------------------------------------------------------

## 4. Implementation choice

We implement `bessel_K_matrix(t, P)` using:

- **Formula (C)** (exact gamma/beta) when $t=0$.
- **Formula (B)** (adaptive certified Arb integration of the finite convolution
  integral) for $0 < |t| < 1/2$.
- **$K_{jm}(t) = 0$** (exact) for $|t|\ge 1/2$.

Rationale: form (B) is as "closed-form" as (A) but is numerically robust for all
$j,m\ge 0$ and directly certified by Arb's ball semantics. The Bessel/Lommel
route (A) is retained in `bessel_kernel_derivation.md` for reference and as a
potential optimisation.

**Rigor statement.** At every rational $t\in[-\tfrac12,\tfrac12]\cap\mathbb Q$, the
returned matrix entries are Arb balls that provably enclose the true real
$K_{jm}(t)$: formula (C) is symbolic-exact, and formula (B) uses Arb's
interval-arithmetic certified integrator, which returns a ball with certified
radius bounding the true integral.

--------------------------------------------------------------------------------

## 5. Bilinear form and constraints

Given a coefficient vector $a\in\mathbb R^P$:

$$(f*f)(t) \;=\; \sum_{j,m=0}^{P-1} a_j a_m\, K_{jm}(t) \;=\; a^{\!\top} K(t)\,a.$$

Mass normalisation $\int f = 1$ becomes the linear constraint
$\sum_j a_j\,\beta_j = 1$ with $\beta_j$ from Section 2.

A lower bound on $C_{1a}$ of the form $\max_{|t|\le 1/2}(f*f)(t) \le \gamma$ with
$\int f = 1, f\ge 0$ would require additionally imposing pointwise nonnegativity
of $f$, which is itself polynomial SOS in the Rechnitzer parametrisation — out
of scope for this deliverable, which focuses on the kernel $K$.

--------------------------------------------------------------------------------

## References

- Watson, *A Treatise on the Theory of Bessel Functions*, 2nd ed., §3.3
  (Poisson integral), §13.46 (Weber–Schafheitlin), §13.47 (Lommel).
- Gradshteyn & Ryzhik, 7th ed.: 8.411.8 (Poisson), 6.574.2, 6.681.
- Cilleruelo & Vinuesa, *A note on the Delsarte problem for Sidon sets*.
- Rechnitzer, arXiv:2602.07292.
