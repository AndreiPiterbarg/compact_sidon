# Project Report: A New Lower Bound for the Supremum of Autoconvolutions

> A new lower bound on the autoconvolution constant
> $$ C_{1a} \;\ge\; \frac{1292}{1000} \;=\; 1.292, $$
> improving the previously published rigorous LB of $1.27481$
> (Matolcsi-Vinuesa 2010) and $1.28$ (Cloninger-Steinerberger 2017);
> the unaudited $1.2802$ figure attributed to Xie (2026, Grok) is not
> independently verified and our internal reproduction of it is
> unsound (see `project_cs_1.2802_invalid.md`). The argument is closed by interval arithmetic in
> `flint.arb` at 256-bit precision and mechanized in Lean 4 across
> **30 modules (~15.6 kLoC** on top of mathlib `v4.29.1`): 13 core
> `lean/Sidon/*.lean` (7655 LoC) plus 17 `lean/Sidon/Constructor/*.lean`
> (7922 LoC, all analytic content, axiom-free except
> `LatticePositivity`). There are **two headline theorems**, both
> `lake build` green with 0 `sorry`:
> the **conditional** `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`,
> which carries an analytic admissibility-bundle hypothesis
> `ExtremiserPrimitives f` and reaches exactly **two**
> *verifiable-by-computation* user axioms (5 total with the 3 Lean-core
> logical axioms); and the **unconditional**
> `Sidon.MultiScale.C1a_ge_1292_unconditional`, which takes ONLY
> admissibility, *constructs* that bundle via
> `ExtremiserPrimitives.of_admissible`, and reaches **four**
> verifiable-by-computation axioms (7 total). All four numerical axioms
> are rigorously certified, logically decidable, *not* conjectural --
> analogues of MV 2010's Mathematica citations, the FlySpeck convention.
> The end-to-end audit (`audit_consistency.py`) passes with verdict
> `ALL CHECKS PASS`.
>
> **Rigor claim:** we have established $C_{1a} \ge 1.292$ to **at least
> the rigor of Matolcsi–Vinuesa 2010** (the accepted, peer-reviewed
> proof of $C_{1a} \ge 1.2748$). MV's proof = (i) formal analytic lemmas
> proved on paper, (ii) numerical values cited from Mathematica,
> (iii) algebraic assembly. We match this and strengthen each layer:
> the analytic content is proven in the manuscript by direct citation to
> MO 2009 / MV 2010 applied to $K_{\rm ms}$ AND mechanized axiom-free in
> Lean (the unconditional headline *constructs* the whole MV-Lemma-3.1
> bundle); the four numerical axioms are the *same kind* of
> computer-checked fact as MV's Mathematica citations but backed by
> `flint.arb` at 256-bit (rigorous, not heuristic CAS), mpmath-corroborated,
> hash-anchored; and the assembly is exact rational arithmetic with a
> positive closing margin $307/3190000$. **Honest caveats:** the four
> numerical axioms are decidable and arb-backed but *not eliminated*
> (a pure-Lean-kernel proof would need Flyspeck-scale verified interval
> arithmetic in mathlib); verification is by rigorous AI-agent self-audit,
> NOT third-party referee review.

| | |
|---|---|
| **Authors** | Andrei Piterbarg, Jai Bajaj, Derrick Vincent |
| **Manuscript** | [`lower_bound_proof.tex`](lower_bound_proof.tex) / [`lower_bound_proof.pdf`](lower_bound_proof.pdf) |
| **Lean formalization** | [`lean/Sidon/`](lean/Sidon/) |
| **Numerical certificate** | [`delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json) |


---

## 1. The Result

For every nonnegative $f \in L^1(\mathbb{R})$ supported on
$(-\tfrac14, \tfrac14)$ with $\int f > 0$ and
$\|f * f\|_{L^\infty} < \infty$,

$$ \frac{\|f * f\|_{L^\infty}}{\bigl(\int f\bigr)^2} \;\ge\; \frac{1292}{1000}. $$

Taking the infimum, $C_{1a} \ge 1292/1000 = 1.292$. Combined with the
upper bound $C_{1a} \le 1.502862$ from Georgiev-Gomez Serrano-Tao-Wagner
(AlphaEvolve, arXiv:2511.02864), the constant is now bracketed in
$[1.292, 1.502862]$. The improvement is

| | $C_{1a} \ge$ | Source |
|---|---|---|
| Erdős-Turán (1941) | $1$ | classical |
| Martin-O'Bryant (2009) | $1.262$ | arXiv:0807.5121 |
| Matolcsi-Vinuesa (2010) | $1.27481$ | arXiv:0907.1379 |
| Cloninger-Steinerberger (2017) | $1.28$ | arXiv:1403.7988 |
| Xie (2026, Grok-assisted, unpublished/unaudited) | $1.2802$ | -- |
| **This work Piterbarg-Bajaj-Vincent (2026)** | **$1.292$** | manuscript at root |

The lift over the prior published lower bound is
$1.292 - 1.28 = 0.012$ (Cloninger-Steinerberger 2017); over the
rigorous analytic Matolcsi-Vinuesa baseline,
$1.292 - 1.27481 = 0.01719$.

## 2. Method

The proof refines the Matolcsi-Vinuesa dual framework along four
axes:

1. **Three-scale arcsine kernel.** The single arcsine kernel
   $K_{\rm arc}(\delta_1; \cdot)$ used by MV is replaced by the convex
   combination
   $$ K_{\rm ms} = \sum_{i=1}^{3} \lambda_i\, K_{\rm arc}(\delta_i; \cdot), $$
   with
   $(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$ and
   $(\lambda_1, \lambda_2, \lambda_3) = (85, 10, 5)/100$. The smaller
   scales fill the gaps left by the first Bessel zero of
   $J_0(\pi \delta_1 \xi)^2$, lowering the dominant denominator $S_1$
   from $\approx 87.4$ (single-scale) to $\le 29.841$.

2. **200-mode cosine multiplier.** The trigonometric multiplier $G$ is
   re-optimized as a 200-cosine expansion (rather than the 119 modes
   of MV2010), solved by a convex QP minimizing
   $\sum_j a_j^2 / \widetilde{K_{\rm ms}}(j)$ subject to $G \ge 1$ on
   a 5001-point grid in $[0, 1/4]$. Coefficients are rationalized to
   $a_j \in \mathbb{Q}$ with denominator $10^8$.

3. **Rigorous interval arithmetic.** Every analytic functional
   entering the master inequality is bounded in `flint.arb` at
   256-bit precision and rounded outward to an exact rational.

4. **Quadratic strict-failure witness.** The $z_1$-free quadratic
   master inequality
   $$ \Phi(M) \;=\; M + 1 + \sqrt{(M-1)(K_2 - 1)} \;\ge\; \tau \;=\; \tfrac{2}{u} + a $$
   is closed by exhibiting a rational $M_0 = 1292/1000$ at which
   $\Phi(M_0) < \tau$ strictly. The certified upper bound on $K_2$
   makes $\Phi$ an over-estimate, so the strict failure forces
   $R(f) > M_0$ for every admissible $f$.

## 3. The Certified Anchors

The five real functionals certified in `flint.arb` at 256-bit
precision (all rationals are sourced from
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)):

| Functional | Direction | Certifier value | Rational slack | Lean theorem |
|---|---|---|---|---|
| $k_1 = \widehat{K_{\rm ms}}(1)$ | $\ge$ | $0.92124658993\ldots$ | $9212/10000$ | `k_one_lower_bound` |
| $K_2 = \|K_{\rm ms}\|_{L^2}^2$ | $\le$ | $\in [4.78882342, 4.78890519]$ | $47897/10000$ | `K_two_upper_bound` |
| $S_1 = \sum_j a_j^2 / \widetilde{K}(j)$ | $\le$ | $29.84090646\ldots$ | $29841/1000$ | `S_one_upper_bound` |
| $m_G = \min_{[0,1/4]} G$ | $\ge$ | $0.99997987\ldots$ | $998/1000$ | `min_G_lower_bound` |
| $a = (4/u)\, m_G^2 / S_1$ | $\ge$ | $0.21009214\ldots$ | $20925/100000$ | `gain_lower_bound` |

The **strict-failure margin** at the rational witness
$M_0 = 1292/1000$, $K_2 \le 47897/10000$ is exactly

$$ \tau - \Phi(M_0) \;=\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\approx\; 9.624 \times 10^{-5}. $$

The certifier itself reports a tighter
$M_{\rm cert} \ge 66167/51200 \approx 1.29232422$ (production driver)
when the analytic anchors are coupled rather than rationalized
separately. The headline target $1292/1000$ is the looser rational
floor used in the published statement and in the Lean axiom.

## 4. Lean 4 Formalization

The analytic chain is mechanized in
[`lean/Sidon/`](lean/Sidon/) across **30 modules totalling
~15.6 kLoC** (13 core, 7655 LoC; 17 `Constructor/`, 7922 LoC) on top
of Mathlib pinned to `v4.29.1`, commit
[`5e932f97dd25535344f80f9dd8da3aab83df0fe6`](https://github.com/leanprover-community/mathlib4/commit/5e932f97dd25535344f80f9dd8da3aab83df0fe6).
The `v4.29.1` bump (post-Nov 2025) unlocked the $L^2$-Plancherel API
(`MeasureTheory.Lp.fourierTransformₗᵢ`) and convolution--Fourier
duality (`Real.fourier_mul_convolution_eq`), on which the Parseval
infrastructure of this project depends. The full development builds
cleanly under `lake build` with **zero `sorry` tactics** across all
modules.

**Recent fixes (Wave-12 multi-agent audit; Option B 2026-05-20; S1+S2 refactor).** Two
math-fidelity corrections landed during the Wave-12 multi-agent audit
and are reflected in the post-Wave-12 build: (i) the `f ∘ f` convention
was tightened from a pointwise product to the *convolutional* form
matching MV 2010, and (ii) `K_arc` is now defined as the autoconvolution
$\eta_\delta * \eta_\delta$ rather than the bare arcsine density. A
subsequent refinement (**Option B**, 2026-05-20) replaces the previously
opaque analytic primitive `gain_analytic := gainLowerQ + 1` with
*concrete defined* real expressions, so the two
verifiable-by-computation axioms now genuinely bound real analytic
functionals. Specifically:

- The 200 QP coefficient numerators are embedded in
  `Sidon.MultiScale.qpNumerators : List ℤ` (common denominator $10^8$),
  with `qpNumerators_length : qpNumerators.length = 200` verified by
  `native_decide`.
- `G_concrete x := Σⱼ (qpNumerators[j]/10⁸) · cos(2π (j+1) x / u_real)`
  is the concrete cosine sum, and
  `min_G_analytic := sInf (G_concrete '' Set.Icc 0 (1/4))` is the
  analytic infimum.
- `Ktilde_ms j := λ₁·besselJ0(πjδ₁/u)² + λ₂·besselJ0(πjδ₂/u)² +
  λ₃·besselJ0(πjδ₃/u)²` is the Bessel-form period-$u$ Fourier
  coefficient (`besselJ0` is the axiom-free `Sidon.Bessel` power
  series), and `S_1_analytic := Σⱼ aⱼ²/Ktilde_ms(j)` is the QP
  denominator sum.
- `gain_analytic := (4 / uQ_real) * min_G_analytic^2 / S_1_analytic` is
  a non-opaque `noncomputable def`.

A later **S1+S2 refactor** (2026-05) made two further changes:
(iii) the bundle's `hEq3` equality was replaced by the inequality form
`hEq3_ge : 2/u + 2·u²·S_cos ≤ LHS1 + LHS2`, which is discharge-able
from finite-`J` Parseval plus Bochner positivity `K̃_ms(j) ≥ 0` alone —
no period-`u` Poisson summation required (new supporting theorems
`mv_eq3_ge` and `mv_eq3_ge_of_eq` live in `MVLemmas.lean`); (iv) the
two vacuously-true Schwartz-class headline variants and the
`SchwartzAtomic` / `SchwartzAtomicResidual` records were removed
(Paley–Wiener + Carlson's theorem: no nontrivial Schwartz `f`
compactly supported in `(-1/4, 1/4)` can satisfy
`ParsevalSplitSchwartz`), reducing the core module count from fifteen to
thirteen. Subsequently the **17-module `Sidon.Constructor.*` layer**
(7922 LoC) was added to *construct* the `ExtremiserPrimitives` bundle
from raw admissibility, bringing the total to **30 modules / ~15.6 kLoC**
and adding the **unconditional** headline `C1a_ge_1292_unconditional`
alongside the original conditional one.

All fixes are build-clean (`lake build` green, 0 sorries). The
**conditional** headline `autoconvolution_ratio_ge_1292_1000` reaches a
5-axiom budget (3 Lean core + 2 verifiable-by-computation, with the
bundle assumed); the **unconditional** headline
`C1a_ge_1292_unconditional` reaches 7 (3 Lean core + 4
verifiable-by-computation, with the bundle constructed).

The exported headline theorems are the conditional
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
(hypothesised on the `ExtremiserPrimitives` bundle), with
decimal restatement `autoconvolution_ratio_ge_1_292` and flipped
form `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`)
exported from the same namespace, and the unconditional
`Sidon.MultiScale.C1a_ge_1292_unconditional` (in
`Sidon.Constructor.Assembly`).

| Module | Lines | Role | Axioms |
|--------|------:|------|--------|
| [`Sidon.Defs`](lean/Sidon/Defs.lean) | 55 | Core definitions (`autoconvolution_ratio`, etc.; convolutional `f ∘ f`). | 0 |
| [`Sidon.Bessel`](lean/Sidon/Bessel.lean) | 958 | Bessel $J_0$ power series, autoconvolution arcsine FT identity, Watson tail bound. | 0 |
| [`Sidon.FourierAux`](lean/Sidon/FourierAux.lean) | 606 | Schwartz Plancherel, $L^p$ bridge, $L^1$-pairing. | 0 |
| [`Sidon.TorusParseval`](lean/Sidon/TorusParseval.lean) | 785 | Period-$u$ Parseval, lattice Fourier, bilinear pairing. | 0 |
| [`Sidon.MVLemmas`](lean/Sidon/MVLemmas.lean) | 767 | MV Lemma 3.1 Eqs.(1)--(4) + inner-product floor + `mv_eq3_ge` / `mv_eq3_ge_of_eq`. | 0 |
| [`Sidon.MasterFromLemmas`](lean/Sidon/MasterFromLemmas.lean) | 130 | Algebraic assembly Eqs.(1)--(4) $\Rightarrow$ Eq.(6). | 0 |
| [`Sidon.BundleDefs`](lean/Sidon/BundleDefs.lean) | 488 | `ExtremiserPrimitives` record. | 0 |
| [`Sidon.BundleEq1`](lean/Sidon/BundleEq1.lean) | 347 | Discharge of bundle field `hEq1` (MV Eq.(1)). | 0 |
| [`Sidon.BundleEq2Schwartz`](lean/Sidon/BundleEq2Schwartz.lean) | 624 | Discharge of bundle field `hEq2` (MV Eq.(2)). | 0 |
| [`Sidon.BundleEq3Schwartz`](lean/Sidon/BundleEq3Schwartz.lean) | 371 | Discharge of bundle field `hEq3_ge` (MV Eq.(3), inequality form). | 0 |
| [`Sidon.BundleEq4`](lean/Sidon/BundleEq4.lean) | 445 | Discharge of bundle field `hEq4` (MV Eq.(4)). | 0 |
| [`Sidon.BilinearParseval`](lean/Sidon/BilinearParseval.lean) | 434 | Bilinear Parseval pairings used by the bundle discharges. | 0 |
| [`Sidon.MultiScale`](lean/Sidon/MultiScale.lean) | 1645 | Conditional headline + admissibility bundle + 3 numerical axioms (K2, gain, min_G). | 3 (verifiable-by-computation) |
| [`Sidon.Constructor.*`](lean/Sidon/Constructor/) (17 modules) | 7922 | Bundle constructor `of_admissible`, unconditional headline `C1a_ge_1292_unconditional`, all supporting analytic infrastructure. | 1 (`K_ms_fourier_lattice_pos_active`, in `LatticePositivity`) |

(`MultiScale.lean` declares three numerical axioms; only two —
`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ` — are reached
by the conditional headline, while `min_G_analytic_ge_minGLowerQ` is
reached only along the unconditional constructor path.)

The **conditional** headline theorem's dependency closure reaches
exactly **two verifiable-by-computation user axioms** (also: "rigorously
certified numerical assertions"), both analogues of "Mathematica
computed this value" in MV 2010, but backed by `flint.arb` at 256-bit
precision (the **unconditional** headline reaches two more — see below).

These axioms are *verifiable-by-computation* in the following
precise sense:

- Each is a logically *decidable* inequality about a specific real
  number (a concrete integral / finite sum over the explicit kernel
  $K_{\rm ms}$ and multiplier $G$).
- Each is backed by a specific reproducible algorithm
  (`flint.arb` at 256-bit precision via
  `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`).
- Each is **not yet** discharged inside Lean only because the
  corresponding Bessel interval-arithmetic infrastructure is not in
  mathlib.
- The arrangement is functionally equivalent to delegating the
  computation to an external oracle (the FlySpeck formalisation of
  Kepler's conjecture used the same convention).

These are *not* conjectural axioms; they are not RH-style assertions
undecidable from within the system. They are statements about
specific integers that any sufficient implementation of interval
arithmetic + the Bessel power series can decide.

| Axiom | Statement | Justification |
|---|---|---|
| `K2_analytic_le_K2UpperQ` | $\int K_{\rm ms}(x)^2\,dx \le \texttt{K2UpperQ} = 47897/10000$, where $K_{\rm ms}$ is the explicit three-scale arcsine kernel. | Closed-form $K_2$ of the three-scale arcsine kernel, evaluated in arb interval arithmetic. Certifier interval $[4.788823, 4.788906]$, slack margin $\approx 7.9 \times 10^{-4}$. Paper Lemma 4.2. |
| `gain_analytic_ge_gainLowerQ` | $\texttt{gain\_analytic} := (4/u_{\rm real}) \cdot \texttt{min\_G\_analytic}^2 / \texttt{S\_1\_analytic} \ge \texttt{gainLowerQ} = 20925/100000$; the RHS is a concrete `noncomputable def` over the 200 embedded QP coefficients `qpNumerators` (denominator $10^8$) and the Bessel-form Fourier coefficient `Ktilde_ms` built from `Sidon.Bessel.besselJ0`. | Cosine $G$'s $(\min G)^2/S_1$ ratio, optimised by QP and arb-verified (coupled-arb $\ge 0.21009214$, margin $\approx 8.4 \times 10^{-4}$). Paper Lemmas 4.3--4.5. |

The two axioms above are the **entire** kernel-specific axiom budget of
the **conditional** headline `autoconvolution_ratio_ge_1292_1000`
(which carries the `ExtremiserPrimitives` bundle as a hypothesis).
The **unconditional** headline
`Sidon.MultiScale.C1a_ge_1292_unconditional` *constructs* that bundle
from raw admissibility via `ExtremiserPrimitives.of_admissible`, and so
reaches **two further** sanctioned verifiable-by-computation axioms:
`min_G_analytic_ge_minGLowerQ` ($\min_{[0,1/4]} G \ge 998/1000$, used to
discharge the bundle's `min_G`-positivity field — declared in
`Sidon.MultiScale` but reached only along the constructor path), and the
active-set positivity axiom below (needed because the QP active-set
period-$u$ denominators must be strictly positive; structural prefix
folded in as a Lean theorem). The four-axiom total is the unconditional
budget:

| Axiom | Statement | Justification |
|---|---|---|
| `min_G_analytic_ge_minGLowerQ` (in `Sidon.MultiScale`) | $\texttt{min\_G\_analytic} \ge \texttt{minGLowerQ} = 998/1000$, where `min_G_analytic` is $\mathrm{sInf}$ of the concrete cosine sum $G$ over $[0,1/4]$. | Taylor-2 branch-and-bound in `flint.arb` (`delsarte_dual/grid_bound/G_min.py`, 32768 cells, 256-bit), arb lower endpoint $\approx 0.99998$; the rational comparison $998/1000 \le \min G$ (slack $\approx 1.9\times10^{-3}$) is re-checked by `audit_consistency.py` (Sections B and D). Same role as MV 2010's Mathematica $m_G$ citation, stricter. |

| Axiom | Statement | Justification |
|---|---|---|
| `K_ms_fourier_lattice_pos_active` (in `Sidon.Constructor.LatticePositivity`) | $\texttt{K\_ms\_fourier\_lattice}(j) = \sum_i \lambda_i\, J_0(\pi \delta_i\, j/u)^2 > 0$ for all $1 \le j \le 200$ (the active-set $S_1$ denominators are nonzero, i.e. the three $J_0$ zero-sets do not coincide at any $j/u$). | Logically *decidable* finite conjunction of 200 strict $J_0$-sum inequalities. Prefix $1\le j\le 16$ is a Lean **theorem** (`pos_of_abs_le_16`, axiom-free) via $\delta_3$ and $J_0(z)\ge 1-(z/2)^2$ on $\lvert z\rvert<2$; tail $17\le j\le 200$ is arb-verified at 256-bit by `delsarte_dual/grid_bound_alt_kernel/audit_lattice_positivity.py`, which reports $\min_{1\le j\le 200}\widetilde{K_{\rm ms}}(j) \approx 2.0817\times10^{-4}$ (certified arb lower endpoint, at $j=147$). Analogue of MV 2010's Mathematica $J_0$ citation, stricter. **Not** derivable from `gain_analytic_ge_gainLowerQ` (which records only a rational *upper* bound on $S_1$; Lean's $x/0=0$ convention lets a zero denominator decrease $S_1$ and still satisfy the bound, so the gain axiom cannot force positivity). |

The headline theorem additionally takes an **analytic
admissibility-bundle hypothesis**:

```lean
structure ExtremiserPrimitives (f : ℝ → ℝ) where
  m_G S_G S_cos LHS1 LHS2 : ℝ
  -- binding fields: tie the reals above to the canonical analytic functionals
  m_G_eq   : m_G = min_G_analytic
  S_G_eq   : S_G = uQ_real * S_1_analytic / 2
  S_cos_eq : S_cos = Sidon.MultiScale.S_cos f
  LHS1_eq  : LHS1 = Sidon.MultiScale.LHS1 f
  LHS2_eq  : LHS2 = Sidon.MultiScale.LHS2 f
  K2_ge_1  : 1 ≤ K2_analytic
  gain_eq  : gain_analytic = 2 * m_G^2 / S_G
  R_ge_1   : 1 ≤ autoconvolution_ratio f
  S_G_pos  : 0 < S_G
  hEq1     : LHS1 ≤ autoconvolution_ratio f
  hEq2     : LHS2 ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                    * Real.sqrt (K2_analytic - 1)
  hEq3_ge  : 2/uQ + 2 * uQ^2 * S_cos ≤ LHS1 + LHS2
  hEq4     : uQ^2 * S_cos ≥ m_G^2 / S_G
```

The four fields `hEq1`, `hEq2`, `hEq3_ge`, `hEq4` are Lean restatements
of MV Lemma 3.1 Eqs.(1)--(4) for the specific $(f, K_{\rm ms})$ pair.
The five `*_eq` binding fields tie the record's real parameters
`m_G, S_G, S_cos, LHS1, LHS2` to the concrete analytic functionals
(`min_G_analytic`, `uQ_real·S_1_analytic/2`, and the canonical
`Sidon.MultiScale.{S_cos, LHS1, LHS2} f`), so the bundle forces the
canonical $(f, K_{\rm ms})$ values and is no longer satisfiable by
arbitrary reals -- a strengthening over the prior free-real-parameter
form. `K2_ge_1` and `R_ge_1` remain provable positivity hypothesis
fields (discharge needs downstream infra; documented with TODO
comments -- they are not axioms). MO~2009 Lemmas~3.1--3.4 / MV~2010
Lemma 3.1 apply to $K_{\rm ms}$ directly (a pdf supported in
$[-\delta_1, \delta_1]$ with $\widetilde{K_{\rm ms}}(j) \ge 0$ and
$K_{\rm ms} \in L^2$), and the paper discharges `hEq1`, `hEq2`, `hEq4`
by direct citation. The
inequality form `hEq3_ge` (replacing the earlier equality `hEq3`) is
*genuinely* Lean-derivable from finite-`J` Parseval plus Bochner
positivity `K̃_ms(j) ≥ 0` alone — no period-`u` Poisson summation
needed; the supporting theorems are `mv_eq3_ge` and `mv_eq3_ge_of_eq`
in `Sidon.MVLemmas`. `Sidon.MasterFromLemmas` chains the bundle
fields axiom-free into the master inequality; the building blocks for
a single-call constructor of the remaining three fields live in
`Sidon.TorusParseval` and `Sidon.FourierAux` but have not yet been
packaged into a one-line mathlib invocation. The bundle's status as
a named Lean hypothesis is therefore a mathlib-API note, not a
logical gap.

The previous macro axiom `MV_master_inequality_for_extremiser`
(single user axiom bundling all analytic + numerical content into one
statement) is now a Lean *theorem*, derived from the bundle hypothesis
plus the two verifiable-by-computation axioms.

The following dependent statements are Lean **theorems** (no `sorry`,
no axioms):

| Theorem | Role |
|---|---|
| `MV_master_inequality_for_extremiser` | The MV master inequality at slack rationals, specialised to $K_{\rm ms}$. *Now a theorem*, replacing the prior macro axiom of the same name. |
| `MV_master_via_slack_monotonicity` | Real-algebraic lift from analytic anchors $(\texttt{K\_2\_analytic}, \texttt{gain\_analytic})$ to slack rationals via `Real.sqrt` monotonicity. |
| `MV_master_inequality_from_MV_lemmas` | Full chain from MV Eqs.(1)--(4) (bundle fields) to the slack-anchored master inequality. |
| `master_inequality_M_lower` | Quadratic inversion (Prop 5.1 of the paper); case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. |
| `K_two_upper_bound` | The slack rational $K_2 \le 47897/10000$ dominates the certifier-reported $K_2 \le 4.788906$ (one-line `norm_num`). |
| `k_one_lower_bound` | Slack-soundness for $k_1$. |
| `S_one_upper_bound` | Slack-soundness for $S_1$. |
| `min_G_lower_bound` | Slack-soundness for $m_G$. |
| `gain_lower_bound` | Slack-soundness for $a$. |

The headline theorem is

```lean
theorem Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
    (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤)
    (P : ExtremiserPrimitives f) :
    autoconvolution_ratio f ≥ (1292 : ℝ) / 1000
```

with decimal restatement `autoconvolution_ratio_ge_1_292` and the
flipped form `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`)
exported from the same namespace -- all three taking the same
`ExtremiserPrimitives f` bundle hypothesis as their fifth argument.
The bundle is "MO 2009 / MV 2010 cited inequalities + 5 axioms":
`hEq1`, `hEq2`, `hEq4` are paper-discharged by direct citation,
while `hEq3_ge` is genuinely Lean-derivable from finite-`J` Parseval
plus Bochner positivity.

#### Unconditional headline `C1a_ge_1292_unconditional`

The formalization additionally exports an **unconditional** headline
(over the admissible class) that takes **no** `ExtremiserPrimitives`
hypothesis:

```lean
theorem Sidon.MultiScale.C1a_ge_1292_unconditional
    (f : ℝ → ℝ)
    (hf_int    : Integrable f volume)
    (hf_L2     : MeasureTheory.MemLp f 2 volume)
    (hf_supp   : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one    : ∫ x, f x ∂volume = 1) :
    autoconvolution_ratio f ≥ (1292 : ℝ) / 1000
```

(symbol resolves through `Sidon.Constructor.Assembly`; it builds the
finiteness side-condition `h_conv_fin` from `MemLp f 2` and feeds the
conditional headline). The MV Lemma 3.1 bundle is **constructed, not
assumed**: the record is produced by
`ExtremiserPrimitives.of_admissible`, whose proof discharges all four
Eqs.(1)--(4) fields (plus the positivity fields) directly from the
admissibility hypotheses and the sanctioned numerical axioms. Building
it required a new **`lean/Sidon/Constructor/`** module layer (17 files,
7922 LoC, all axiom-free except `LatticePositivity`): the
$L^1 \cap L^2$ convolution Fourier identity (`ConvFourier`),
`MemLp K_ms 2` via a three-function Hölder/Young estimate
$L^{4/3}\star L^{4/3}\to L^2$ (`KernelL2`, `YoungConvolution`), a
period-$u$ Poisson sampling lemma generalizing mathlib's
period-1/continuous-only version (`PoissonSampling`, `PoissonSummable`,
`PeriodUParseval`), the MO 2009 Lemma 2.1 period-$u$ Parseval pairing
(`MOLemma21`), period-1 Parseval for the $f\circ f$ bound
(`Eq2Period1`, `Eq2Split`), the Cauchy--Schwarz floor
(`CauchySchwarzFloor`), and the glue/assembly (`FieldsEasy`,
`FieldsParseval`, `FieldEq4`, `Glue`, `Assembly`). The effort also
**fixed two real bugs**: a convention/scaling error in the bundle's
`S_cos` term (the master used a $2u^2$ coefficient where the correct one
is $2/u$) — corrected in both the Lean source and `lower_bound_proof.tex`
— and a false `h_parseval_split` form.

Because `of_admissible` discharges the positivity fields using
`Sidon.MultiScale.min_G_analytic_ge_minGLowerQ` and the active-set
positivity, the unconditional headline's dependency closure reaches the
Lean-core trio plus **four** verifiable-by-computation axioms:
`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`,
`min_G_analytic_ge_minGLowerQ`, and
`Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`
(vs. **two** for the conditional headline, which carries the bundle as a
hypothesis). All four are logically decidable, `flint.arb`-backed at
256-bit precision, the same standard as MV's Mathematica citations —
this is MV-parity computer-assisted rigor, **not** a fully
kernel-checked numeric proof.

The slack-anchor substitution is monotonically sound: the master
inequality is increasing in $K_2 - 1$ and in $a$, so any true bound
on the analytic functionals transports to a valid bound at the slack
rationals. The five `norm_num`-checked slack-soundness theorems
above discharge that the rational slacks are on the correct side of
the certifier-reported decimals.

### Comparison with Matolcsi--Vinuesa (2010): axiom budget

The published MV paper (J. Math. Anal. Appl. **372** (2010),
439--447) proves $C_{1a} \ge 1.2748$ by:

1. Formally proving Lemmas 3.1 (Eqs.(1)--(4), via Martin--O'Bryant),
   3.3 ($z_1$ refinement), 3.4 ($\sin$ bound).
2. **Citing Mathematica** for $J_0(\pi \cdot 0.138)^2$, $m_G$, $S_1$,
   $a = 0.0713$.
3. Combining algebraically.

This work matches the architecture, but strengthens each layer:

1. **Analytic content formally proved in Lean** -- ~15.6 kLoC across 30
   modules, all axiom-free except the four numerical axioms, spanning
   Bessel power series, the (autoconvolution) arcsine
   Fourier-transform identity, $L^2$-Plancherel (via mathlib `v4.29.1`),
   period-$u$ torus Parseval, the four MV Lemma 3.1 atomic primitives
   together with their dedicated discharge modules
   (`BundleEq1`/`BundleEq2Schwartz`/`BundleEq3Schwartz`/`BundleEq4`),
   the bilinear Parseval pairings, the master inequality assembly, AND
   the entire 17-module `Constructor/` layer that *constructs* the
   MV-Lemma-3.1 bundle for any admissible $f$. The analytic content MV
   proved on paper is thus mechanized in Lean, not merely assumed.
2. **Verifiable-by-computation axioms** (2 for the conditional headline,
   4 for the unconditional) -- analogues *in role* of
   MV's Mathematica citations, backed by `flint.arb` at 256-bit
   precision (proven interval bounds rather than heuristic
   floating-point), anchored to a SHA-256-stamped certificate and
   re-derived independently via mpmath at 30–50 decimal digits
   (`audit3_mpmath.py`, `docs/audit_3scale_reverification*.md`).
3. **Admissibility bundle** `ExtremiserPrimitives f` --
   the analogue of MV invoking "by Lemma 3.1 (Martin--O'Bryant)". A
   *hypothesis* of the conditional headline, *constructed* from raw
   admissibility for the unconditional headline; not an axiom.
4. **Assembly** -- exact rational arithmetic with a positive closing
   margin $307/3190000$ (true threshold $M^\ast = 1.29203$), machine-checked
   in Lean with no `sorry`.

**Categorisation of the axiom budget.** Lean's `#print axioms`
output mixes three categorically distinct kinds of dependency:

- **Logical axioms** -- `propext`, `Classical.choice`,
  `Quot.sound`. Lean 4 core; trusted without proof; cannot be
  derived by any finite computation.
- **Verifiable-by-computation axioms** -- the two numerical ones
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`) reached by
  the conditional headline, plus two more reached only on the
  unconditional-headline path (`min_G_analytic_ge_minGLowerQ` and the
  active-set positivity `K_ms_fourier_lattice_pos_active`). All four are
  logically *decidable* statements about specific real numbers, certified
  by a reproducible `flint.arb` algorithm and mpmath-corroborated;
  currently un-formalised in Lean only because mathlib lacks a Bessel
  interval-arithmetic library.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`.
  Not an axiom: a *hypothesis* of the conditional headline, and
  *constructed* from admissibility for the unconditional headline.

The distinction between *conjectural* and
*verifiable-by-computation* axioms matters because the natural
critic's question -- "are you assuming something unprovable?" --
has a clean answer here: **no**. All four numerical axioms are
provable. They are simply not yet formalised in Lean for
engineering reasons (no Bessel interval-arithmetic library in
mathlib). They are not RH-style assertions undecidable from within
the system; they are statements about specific integers that any
sufficient implementation of interval arithmetic + the Bessel
power series can decide.

This is a standard axiom architecture for computer-assisted
real-number proofs (Flyspeck cited Kepler's interval arithmetic;
the polynomial-method cap-set proof cited Lagrange polynomial
bounds; the PFR formalisation cited numerical Plünnecke--Ruzsa
constants). The mathematical content of the proof
is in the Lean theorems; the verifiable-by-computation axioms
encode only "evaluate this specific integral and compare it to
this specific rational".

**Honesty caveats.**

- This is computer-assisted rigor at MV's standard: the
  verifiable-by-computation axioms (2 for the conditional headline,
  4 for the unconditional) are decidable and arb-backed
  (mpmath-corroborated), but **not eliminated**. A pure-Lean-kernel
  proof would need Flyspeck-scale verified interval arithmetic in
  mathlib, which does not exist today.
- The **conditional** headline carries `ExtremiserPrimitives f` as a
  named hypothesis record; the **unconditional** headline *constructs*
  it from raw admissibility (`of_admissible`, in the `Constructor/`
  layer). The bundle fields are Lean restatements of MO~2009 / MV~2010
  outputs at $(f, K_{\rm ms})$, discharged in the paper by
  direct citation (those lemmas apply to $K_{\rm ms}$ directly as an
  admissible kernel), exactly as MV~2010 discharged its single-arcsine
  instance via MO~2009.
- The verifiable-by-computation axioms depend on trusting the
  `flint.arb` library (peer-reviewed -- Johansson 2017, IEEE TC --
  but not Lean-verified).
- Replacing the verifiable-by-computation axioms with verified Lean
  numerics would require a separate multi-year subproject (rigorous
  interval arithmetic + verified Gauss--Legendre quadrature +
  verified Bessel + Taylor branch-and-bound, ~6000--10000 lines),
  comparable to the Flyspeck effort for Kepler, with no upside for
  the mathematical claim.
- Verification is by rigorous AI-agent self-audit across multiple
  independent passes (numerical re-derivation, proof tracing, certifier
  re-runs, `#print axioms`), **not** third-party referee review.

## 5. Repository Layout

```
compact_sidon/
├── lower_bound_proof.tex             # The manuscript
├── lower_bound_proof.pdf             # Compiled output
├── audit_consistency.py              # Cross-source audit
├── REPORT.md                         # This file
├── README.md                         # Project overview
│
├── lean/                             # Lean 4 formalization (~15.6 kLoC, 30 modules)
│   ├── Sidon/Defs.lean               # Shared definitions (55 lines, 0 axioms)
│   ├── Sidon/Bessel.lean             # Bessel J0 power series, arcsine FT (958, 0 axioms)
│   ├── Sidon/FourierAux.lean         # Schwartz Plancherel, L^p bridge (606, 0 axioms)
│   ├── Sidon/TorusParseval.lean      # Period-u Parseval, lattice Fourier (785, 0 axioms)
│   ├── Sidon/MVLemmas.lean           # MV Lemma 3.1 Eqs.(1)-(4), mv_eq3_ge (767, 0 axioms)
│   ├── Sidon/MasterFromLemmas.lean   # Master inequality assembly (130, 0 axioms)
│   ├── Sidon/BundleDefs.lean         # ExtremiserPrimitives record (488, 0 axioms)
│   ├── Sidon/BundleEq1.lean          # Discharge of hEq1 (MV Eq.(1)) (347, 0 axioms)
│   ├── Sidon/BundleEq2Schwartz.lean  # Discharge of hEq2 (MV Eq.(2)) (624, 0 axioms)
│   ├── Sidon/BundleEq3Schwartz.lean  # Discharge of hEq3_ge (MV Eq.(3), inequality) (371, 0 axioms)
│   ├── Sidon/BundleEq4.lean          # Discharge of hEq4 (MV Eq.(4)) (445, 0 axioms)
│   ├── Sidon/BilinearParseval.lean   # Bilinear Parseval pairings (434, 0 axioms)
│   ├── Sidon/MultiScale.lean         # Conditional headline, 3 numerical axioms K2/gain/min_G (1645 lines)
│   ├── Sidon/Constructor/            # 17 modules (7922 LoC): of_admissible + unconditional headline
│   │                                 #   + analytic infra; axiom-free except LatticePositivity
│   │                                 #   (4th axiom K_ms_fourier_lattice_pos_active)
│   ├── Sidon.lean                    # Top-level module entry
│   └── AxiomCheck{BundleDefs,Fourier,MV,Torus}.lean # Per-module axiom inventories
│
├── delsarte_dual/                    # The arb certifier
│   ├── grid_bound/                   # Single-scale MV machinery + certify.py verifier
│   ├── grid_bound_alt_kernel/        # Three-scale kernel, QP for G, bisect driver
│   │   └── certificates/
│   │       ├── reference_anchors.json    # Canonical 256-bit anchors
│   │       └── multiscale_arcsine_1292.json  # Fresh-run certificate
│   └── README.md
│
├── tests/                            # pytest suite
│   ├── grid_bound_alt_kernel/        # Kernel admissibility, Bochner positivity, QP
│   └── README.md
│
├── docs/                             # Public documentation
│   ├── proof_outline.md              # Mathematical summary (sections, key formulas)
│   ├── reproducibility.md            # Exact reproduction commands
│   ├── formalization.md              # Lean module description
│   ├── verification.md               # 14-task verification checklist
│   └── presentation/                 # Slide deck (.pptx + figures)
│
└── archive/                          # Earlier exploration (cs-cascade,
                                      # Lasserre SDP, attempts/, agent_experiments, etc.)
```

## 6. Reproducing the Result

### Compile the manuscript

```bash
pdflatex -interaction=nonstopmode lower_bound_proof.tex
```

(No external `.bib` file: the bibliography is inlined via
`thebibliography`.) Output: no overfull/underfull/undefined
warnings.

### Regenerate the numerical certificate

```bash
pip install python-flint cvxpy numpy
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

The driver runs at 256-bit precision and emits
`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
with the five anchors, the bisection history, the terminal cell
list, and a SHA-256 body hash.

### Independent verification

```bash
python -m delsarte_dual.grid_bound.certify \
    delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json
```

`certify.py` is a stand-alone verifier that imports only
`python-flint` primitives. Exit code `0` iff the certificate body
hash matches, every anchor is recomputable in arb at the declared
precision, the terminal cells cover $[0, \mu(M_{\rm cert})]$
contiguously, and every cell has $\Phi < 0$ upper bound.

### Build the Lean formalization

```bash
cd lean && lake build              # all 30 modules (13 core + 17 Constructor)
lake env lean AxiomCheckMV.lean    # per-module axiom inventories
                                   # (also: BundleDefs, Fourier, Torus)
```

The four `AxiomCheck*.lean` files print the axiom closure of their
corresponding module imports. The **conditional** headline's axiom
dependency closure
(via `#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
after `lake build`) consists of Lean's three core logical axioms
(`Classical.choice`, `propext`, `Quot.sound`) plus exactly two
*verifiable-by-computation* user axioms (rigorously certified
numerical assertions, both backed by `flint.arb` at 256-bit
precision):

```
'Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ]
```

The **unconditional** headline reaches two more (four total), because
its constructor `of_admissible` discharges the bundle's positivity
fields:

```
'Sidon.MultiScale.C1a_ge_1292_unconditional' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ,
   Sidon.MultiScale.min_G_analytic_ge_minGLowerQ,
   Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active]
```

The conditional headline additionally takes an analytic
admissibility-bundle hypothesis `ExtremiserPrimitives f` (its fifth
argument) packaging the four MV Lemma 3.1 outputs (Eqs.(1)--(4))
for the specific $(f, K_{\rm ms})$ pair. For the unconditional headline
this bundle is **constructed, not assumed**: the `Constructor/` layer
bridges mathlib's $L^2$ Plancherel API to the concrete period-$u$
Parseval splits, so `of_admissible` produces the witness from raw
admissibility for general (non-Schwartz) admissible $f$.

### Run the cross-source audit

```bash
python audit_consistency.py             # summary verdict
python audit_consistency.py --verbose   # print every individual check
```

Verifies that the numerical anchors in
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json),
the slack rationals in
[`lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean), the
decimal claims in `docs/{proof_outline,reproducibility,formalization,verification}.md`,
the headline-bound claims across READMEs, and the Proposition 5.1
arithmetic in
[`lower_bound_proof.tex`](lower_bound_proof.tex) are all mutually
consistent and on the correct side of the arb endpoints.

### Run the pytest suite

```bash
pytest tests/grid_bound_alt_kernel/
```

Covers kernel admissibility, Bochner positivity of $\widehat{K_{\rm ms}}$,
the QP solver convergence, and the single-scale baseline check
against the published Matolcsi-Vinuesa value $1.27481$.

## 7. Cross-Source Audit Framework

`audit_consistency.py` is the project's source of truth for
quantitative consistency. Each run re-derives the analytic anchors in
`flint.arb` at 256-bit precision and verifies every quantitative
claim against the freshly computed ground truth across eight
sections:

| Section | What it checks |
|---|---|
| A | Kernel-parameter consistency (rationals declared in Lean / LaTeX / code agree exactly). |
| B | Slack-rational soundness (every Lean rational anchor is a true bound on the arb endpoint). |
| C | Lean axiom RHS soundness (each of the five $\{k_1, K_2, S_1, m_G, a\}$ slack comparisons is rationally true). |
| D | Tight-decimal claim soundness (every decimal value asserted in READMEs / JSON / docstrings / LaTeX is on the correct side of the arb endpoint). |
| E | LaTeX Proposition 5.1 strict-failure arithmetic (exact rational verification of every step in the closing chain). |
| F | LaTeX per-lemma slack-value claims (e.g. "slack $\ge 9.3 \times 10^{-5}$"). |
| G | $K_2 = \text{bulk} + \text{tail}$ decomposition (Watson tail bound and the constant $C = \sum_i \lambda_i / \delta_i$). |
| H | Published bound consistency ($M_{\rm cert}$ production $\ge 1.29232422$; slack-anchor bisection $\ge 1.29215650$; headline $\ge 1292/1000$). |

**Current status: every check passes, verdict `ALL CHECKS PASS`.**

## 8. Project History (Selected)

The repository carries a substantial earlier-exploration layer under
`archive/`, including:

- A multiscale branch-and-prune cascade extending the CS17 method,
  archived at `archive/attempts/cs_writeup_legacy/` (writeup) and
  `archive/cloninger-steinerberger/` (code).
- A Lasserre SDP hierarchy track with correlative sparsity for $d \in
  \{32, 64, 128\}$, archived at `archive/attempts/lasserre_writeup/` and
  `archive/coarse_lp_bnb/`.
- A two-scale arcsine kernel precursor that produced
  $C_{1a} \ge 1651/1280 \approx 1.28984$, documented in
  [`archive/attempts/multiscale_arcsine.md`](archive/attempts/multiscale_arcsine.md).
- Earlier Hölder, KBK, AlphaEvolve-dual, cohn-elkies, and minimum-overlap
  attempts under `archive/`.

Each historical attempt is preserved with its decision record (often
including the FLAG that closed it).

## 9. Trust Boundary

The published bound rests on the following components:

| Component | Trust |
|---|---|
| Lean 4 kernel | Foundational (assumed sound). |
| Mathlib (`v4.29.1`, commit `5e932f97dd`) | Community-verified library; the bump unlocked `MeasureTheory.Lp.fourierTransformₗᵢ` and `Real.fourier_mul_convolution_eq` used by the project's Parseval infrastructure. |
| `K2_analytic_le_K2UpperQ` axiom (verifiable-by-computation) | Asserts $\int K_{\rm ms}^2 \le 47897/10000$ for the explicit three-scale arcsine kernel. Discharged externally by `flint.arb` at 256-bit precision; certifier interval $[4.788823, 4.788906]$, slack margin $\approx 7.9 \times 10^{-4}$. Logically decidable; not yet a Lean theorem only because mathlib lacks a Bessel interval-arithmetic library. Analogue of MV 2010's Mathematica citation of $K_2$. |
| `gain_analytic_ge_gainLowerQ` axiom (verifiable-by-computation) | Asserts the concrete defined functional $\texttt{gain\_analytic} = (4/u_{\rm real}) \cdot \texttt{min\_G\_analytic}^2 / \texttt{S\_1\_analytic} \ge 20925/100000$ over the 200 embedded `qpNumerators` and the Bessel-form `Ktilde_ms`. Discharged externally by `flint.arb` (coupled-arb $\ge 0.21009214$, margin $\approx 8.4 \times 10^{-4}$). Logically decidable; analogue of MV 2010's Mathematica citation of $a$. Reached by both headlines. |
| `min_G_analytic_ge_minGLowerQ` axiom (verifiable-by-computation) | Asserts $\min_{[0,1/4]} G \ge 998/1000$ for the concrete 200-mode cosine $G$. Discharged externally by `flint.arb` Taylor-2 B&B (arb lower endpoint $\approx 0.99998$). Logically decidable; analogue of MV 2010's Mathematica citation of $m_G$. **Reached only on the unconditional-headline path** (discharges the bundle's positivity field inside `of_admissible`). |
| `K_ms_fourier_lattice_pos_active` axiom (verifiable-by-computation, in `Sidon.Constructor.LatticePositivity`) | Asserts $\widetilde{K_{\rm ms}}(j) = \sum_i \lambda_i J_0(\pi\delta_i j/u)^2 > 0$ for all $1 \le j \le 200$ (active-set $S_1$ denominators nonzero). Prefix $1\le j\le 16$ is a Lean **theorem**; tail arb-verified ($\min \approx 2.0817\times10^{-4}$ at $j=147$). Logically decidable. **Reached only on the unconditional-headline path.** |
| `ExtremiserPrimitives f` bundle | Encodes the four MV Lemma 3.1 outputs (Eqs.(1)--(4)) for the specific $(f, K_{\rm ms})$ pair. Not a Lean axiom. The **conditional** headline takes it as an explicit hypothesis; the **unconditional** headline *constructs* it from raw admissibility via `ExtremiserPrimitives.of_admissible` (the $L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge is implemented in the `Sidon.Constructor.*` layer). |
| `python-flint` / Arb library | Standard interval-arithmetic backend (Johansson 2017). Peer-reviewed; not itself Lean-verified. Numerical axioms are additionally mpmath-corroborated. |
| Numerical anchors | All five anchors are reproduced exactly by `bisect_alt_kernel.py` and independently re-verified by `grid_bound/certify.py`. |
| Rational slack substitution into the master inequality | Sound by monotonicity in $K_2 - 1$ and $a$; the five `norm_num`-decided slack-soundness theorems confirm the rationals lie on the correct side of the certifier's decimal output. **This step is a Lean theorem** (`MV_master_via_slack_monotonicity`), not an axiom. |
| Lean-side assembly (`MV_master_inequality_for_extremiser`, `master_inequality_M_lower`) | Pure Lean *theorems*; `master_inequality_M_lower` is case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. No external dependency. The previous macro axiom of the same name has been promoted to a theorem. |

No component is required beyond those listed.


## References

- [Manuscript: `lower_bound_proof.pdf`](lower_bound_proof.pdf)
- [Lean module: `lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean)
- [Numerical certificate: `reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)
- [Audit script: `audit_consistency.py`](audit_consistency.py)
- [Proof outline: `docs/proof_outline.md`](docs/proof_outline.md)
- [Reproducibility: `docs/reproducibility.md`](docs/reproducibility.md)
- [Formalization notes: `docs/formalization.md`](docs/formalization.md)
- [Verification checklist: `docs/verification.md`](docs/verification.md)

External:
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions
  with an application to Sidon sets.* Proc. Amer. Math. Soc. 145
  (2017), 3191-3200, arXiv:1403.7988.
- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of
  autoconvolutions.* J. Math. Anal. Appl. 372 (2010), 439-447,
  arXiv:0907.1379.
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with
  applications to additive number theory.* Illinois J. Math. 53
  (2009), 219-235, arXiv:0807.5121.
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius
  interval arithmetic.* IEEE Trans. Comput. 66(8) (2017), 1281-1292.
- de Moura, L., Ullrich, S. *The Lean 4 theorem prover and programming
  language.* CADE 28, LNCS 12699, Springer, 2021, 625-635.
- The mathlib community. *The Lean mathematical library.* CPP 2020,
  367-381.
- Georgiev, B., G&oacute;mez-Serrano, J., Tao, T., Wagner, A.Z.
  *Mathematical exploration and discovery at scale.* arXiv:2511.02864.
