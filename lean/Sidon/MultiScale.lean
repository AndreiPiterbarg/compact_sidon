/-
Sidon Autocorrelation Constant: Rigorous Lower Bound `C_{1a} ≥ 1.292`
=====================================================================

This module formalises the headline theorem of *A New Lower Bound for
the Supremum of Autoconvolutions* (Piterbarg, Bajaj, Vincent).

The autoconvolution ratio of an admissible test function `f`,

```
    R(f)  =  ‖f * f‖_∞ / (∫ f)²,
```

is bounded below uniformly over nonnegative `f` supported on `[-1/4, 1/4]`
with `∫ f > 0`.  Matolcsi–Vinuesa (2010, arXiv:0907.1379) established the
master inequality

```
    R(f) + 1 + √(R(f) - 1)·√(K_2 - 1)  ≥  2/u + a,
```

where `K = ∑ λ_i K_arc(δ_i, ·)` is an admissible kernel (a convex
combination of arcsine kernels) and the gain term `a = (4/u)·min_G²/S_1`
is determined by a finite cosine `G` on `[0, 1/4]`.  Inverting this
quadratic in `R(f)` produces a rational lower bound `M_target` on
`C_{1a}`.

This file formalises the 3-scale instance

```
    δ₁ = 138/1000,    λ₁ = 85/100,
    δ₂ =  55/1000,    λ₂ = 10/100,
    δ₃ =  25/1000,    λ₃ =  5/100,
    u  = 638/1000  (= 1/2 + δ₁),
```

with a 200-coefficient `G` re-optimised at this kernel, yielding

```
    C_{1a}  ≥  1292/1000  =  1.292.
```

Bochner admissibility of `K̂` is automatic: each `J₀(πδᵢξ)²` is the square
of a real Bessel function and the convex combination preserves
nonnegativity.

Numerical anchors
-----------------
Each of the five numerical inputs is computed by a `flint.arb` certifier
at 256-bit precision (`delsarte_dual/grid_bound_alt_kernel/`):

  * `K_2`     ∈ `[4.788823, 4.788906]`  (XI_MAX = 1e5, closed-form tail)
  * `k_1`     ≥ `0.92124658`              (essentially exact, rad < 1e-76)
  * `S_1`     ≤ `29.840907`
  * `min G`   ≥ `0.99997987`             (Taylor B&B, 32768 cells on [0, 1/4])
  * `gain a`  ≥ `0.21009214`             (script-reported, coupled in arb)

The rational bounds declared below are slack relaxations of these
arb-interval endpoints; the certifier's cell-search bisection reaches
`M_cert = 66167/51200 ≈ 1.29232422`, which exceeds the rational target
`1292/1000` by `≈ 3.2 × 10⁻⁴`.  The strict-failure proof in this file
uses the looser rational anchors and clears `1292/1000` with margin
`307/3190000 ≈ 9.6 × 10⁻⁵`.

Axioms
------
The headline theorem reaches exactly **two** numerical-only user
axioms in its dependency closure:

  * `K2_analytic_le_K2UpperQ`  (paper Lemma 4.2)
      `K_2(K_ms) ≤ K2UpperQ = 47897/10000` — the closed-form `K_2` of
      the 3-scale arcsine kernel evaluated in `arb` interval
      arithmetic at 256-bit precision.

  * `gain_analytic_ge_gainLowerQ`  (paper Lemmas 4.3-4.5)
      `gain_analytic ≥ gainLowerQ = 20925/100000` — the cosine `G`'s
      `(min G)²/S_1` ratio, optimised by QP and arb-verified.

These are analogues of "Mathematica computed this value" in MV's
paper — they are certifier outputs, not analytic content.  Both
quantities are defined symbolically as concrete real integrals /
finite sums over the explicit 3-scale arcsine kernel `K_ms` and the
QP-optimised cosine `G`.

The headline factors through the wire-up theorem
`MV_master_inequality_for_extremiser` (a Lean *theorem*, not an
axiom) which takes the four MV Lemma 3.1 atomic primitives as
**hypotheses** (period-`u` Parseval splits, the F-bound, etc.) and
discharges them to the slack-rational form using the two numerical
axioms above.

The four atomic analytic primitives for `(f, K_ms)` are *not*
provable from the available mathlib infrastructure for general
non-Schwartz admissible `f` (the L¹∩L² periodisation bridge for
`f*f` and `f∘f` on `ℝ/uℤ` is the missing piece).  They are therefore
collected in a single residual `ExtremiserPrimitives f` hypothesis
bundle, which the general headline `autoconvolution_ratio_ge_1292_1000`
(and its display alias `C1a_ge_1292`) asks the consumer to supply.
The paper discharges those fields by direct citation to MO 2009
Lemmas 2.1, 2.2, 3.2, 3.3 / MV 2010 Lemma 3.1.  See the
documentation of `ExtremiserPrimitives` for the precise mathematical
content.  (The previously-exported Schwartz-class variants
`autoconvolution_ratio_ge_1292_1000_schwartz` /
`_schwartz_residual` and their backing modules
`Sidon.MultiScaleSchwartz` / `Sidon.SchwartzAtomicDischarge` were
retired by the S1+S2 refactor when the Schwartz `ParsevalSplitSchwartz`
predicate was shown to be vacuously satisfiable by Paley-Wiener +
Carlson; only the general bundle survives.)

The five rational comparisons recording the slack soundness
(`K_two_upper_bound`, `k_one_lower_bound`, `S_one_upper_bound`,
`min_G_lower_bound`, `gain_lower_bound`) and the algebraic inversion
(`master_inequality_M_lower`) are now Lean *theorems*.

The algebraic/monotonicity surface of the axiom is moreover *exposed*
by two zero-axiom wire-up theorems:

  * `MV_master_via_slack_monotonicity` — the *real-algebraic* lift from
    the master inequality at analytic anchors `(K_2_analytic,
    gain_analytic)` to the slack rationals `(K2UpperQ, gainLowerQ)`.

  * `MV_master_inequality_from_MV_lemmas` — the full chain from the
    four MV-Lemma-3.1 atomic conclusions (Eqs. 1–4, formalised in
    `Sidon.MVLemmas`) + the two kernel-specific analytic bounds
    (Lemmas 4.2–4.5) to the axiom's conclusion.  Composes
    `Sidon.Master.master_inequality_from_lemmas` with
    `MV_master_via_slack_monotonicity`.

So the *only* content the axiom currently encapsulates is the
discharge of the four MV-Lemma-3.1 atomic conclusions on `ℝ` (which
needs L¹ ∩ L² Plancherel for non-Schwartz functions, periodisation of
`f*f` and `f∘f`, and the inner-product floor on `(f*f-1/u)·G`) plus
the kernel-specific certifier outputs.

No `sorry`, no conjectural axioms.

References
----------
* Matolcsi, M., Vinuesa, C.  *Improved bounds on the supremum of
  autoconvolutions.*  J. Math. Anal. Appl. **372** (2010), 439-447,
  arXiv:0907.1379.
* Cloninger, A., Steinerberger, S.  *On suprema of autoconvolutions
  with an application to Sidon sets.*  Proc. Amer. Math. Soc.
  **145** (2017), 3191-3200, arXiv:1403.7988.
* Cohn, H., Elkies, N.  *New upper bounds on sphere packings I.*
  Ann. of Math. (2) 157 (2003), 689–714.
-/

import Mathlib
import Sidon.Defs
import Sidon.Bessel
import Sidon.MVLemmas
import Sidon.MasterFromLemmas

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open scoped Pointwise
open MeasureTheory

namespace Sidon.MultiScale

/-! ## Kernel anchors -/

/-- First arcsine scale `δ₁ = 138/1000`. -/
def delta1Q : ℚ := 138 / 1000

/-- Second arcsine scale `δ₂ = 55/1000`. -/
def delta2Q : ℚ := 55 / 1000

/-- Third arcsine scale `δ₃ = 25/1000`. -/
def delta3Q : ℚ := 25 / 1000

/-- Mixture weight `λ₁ = 85/100`. -/
def lambda1Q : ℚ := 85 / 100

/-- Mixture weight `λ₂ = 10/100`. -/
def lambda2Q : ℚ := 10 / 100

/-- Mixture weight `λ₃ = 5/100`. -/
def lambda3Q : ℚ := 5 / 100

/-- Weights are a partition of unity. -/
theorem lambdas_sum_one : lambda1Q + lambda2Q + lambda3Q = 1 := by
  unfold lambda1Q lambda2Q lambda3Q; norm_num

/-- Weights are nonnegative. -/
theorem lambdas_nonneg :
    (0 : ℚ) ≤ lambda1Q ∧ (0 : ℚ) ≤ lambda2Q ∧ (0 : ℚ) ≤ lambda3Q := by
  unfold lambda1Q lambda2Q lambda3Q
  refine ⟨?_, ?_, ?_⟩ <;> norm_num

/-- Period of the MV cosine basis: `u = 638/1000 = 1/2 + δ₁`. -/
def uQ : ℚ := 638 / 1000

/-- Verifies the closed-form relation `u = 1/2 + δ₁`. -/
theorem uQ_eq : uQ = (1 : ℚ) / 2 + delta1Q := by
  unfold uQ delta1Q; norm_num

/-- Rational target of the bound: `M_target = 1292/1000 = 1.292`. -/
def MTargetQ : ℚ := 1292 / 1000

/-! ## Numerical anchors

Each rational below is a slack relaxation of an arb-interval bound
reported by the `flint.arb` certifier at 256-bit precision.
-/

/-- Slack upper bound on `K_2 = ‖K̂‖₂² = 2·∫_0^∞ K̂(ξ)² dξ`.

    Certifier returns `K_2 ∈ [4.788823, 4.788906]`; we use `4.7897`. -/
def K2UpperQ : ℚ := 47897 / 10000

/-- Slack lower bound on `k_1 = K̂(1)`.

    Certifier returns `k_1 ≈ 0.92124658993` (radius `< 10⁻⁷⁶`), so in
    particular `k_1 ≥ 0.92124658`; we use the slack rational `0.9212`. -/
def K1LowerQ : ℚ := 9212 / 10000

/-- Slack upper bound on the dual sum `S_1 = ∑_{j=1}^{200} a_j² / K̂(j/u)`.

    Certifier returns `S_1 ≤ 29.840907`; we use `29.841`. -/
def S1UpperQ : ℚ := 29841 / 1000

/-- Slack lower bound on `min_{x ∈ [0,1/4]} G(x)` for the 200-coefficient
    re-optimised `G`.

    Certifier returns `min G ≥ 0.99997987`; we use `0.998`. -/
def minGLowerQ : ℚ := 998 / 1000

/-- Slack lower bound on the gain `a = (4/u)·min_G²/S_1`.

    Plugging the rational floors directly gives
    `(4/u)·(998/1000)²/(29841/1000) ≈ 0.20926`; we use `0.20925`.
    The certifier reports the tighter coupled value `a ≥ 0.21009214`.

    `a` is invariant under rescaling `G ↦ cG` (`min G ↦ c·min G`,
    `S_1 ↦ c²·S_1`), so `min G < 1` for the re-optimised G is benign. -/
def gainLowerQ : ℚ := 20925 / 100000

/-- The gain lower bound is strictly positive. -/
theorem gainLowerQ_pos : (0 : ℚ) < gainLowerQ := by
  unfold gainLowerQ; norm_num

/-- The rational `gainLowerQ` is dominated by the certifier's reported
    coupled-arb value `0.21009214`. -/
theorem gainLowerQ_below_certifier_value :
    (gainLowerQ : ℚ) ≤ 21009214 / 100000000 := by
  unfold gainLowerQ; norm_num

/-! ## Real coercions of the rational kernel parameters -/

/-- Real-coerced first arcsine scale. -/
noncomputable def delta1 : ℝ := (delta1Q : ℝ)
/-- Real-coerced second arcsine scale. -/
noncomputable def delta2 : ℝ := (delta2Q : ℝ)
/-- Real-coerced third arcsine scale. -/
noncomputable def delta3 : ℝ := (delta3Q : ℝ)
/-- Real-coerced first mixture weight. -/
noncomputable def lambda1 : ℝ := (lambda1Q : ℝ)
/-- Real-coerced second mixture weight. -/
noncomputable def lambda2 : ℝ := (lambda2Q : ℝ)
/-- Real-coerced third mixture weight. -/
noncomputable def lambda3 : ℝ := (lambda3Q : ℝ)
/-- Real-coerced period. -/
noncomputable def uQ_real : ℝ := (uQ : ℝ)

/-- The three scales are strictly positive. -/
theorem delta_positivities : 0 < delta1 ∧ 0 < delta2 ∧ 0 < delta3 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    (first | (unfold delta1 delta1Q; push_cast; norm_num)
           | (unfold delta2 delta2Q; push_cast; norm_num)
           | (unfold delta3 delta3Q; push_cast; norm_num))

/-- The period `u` is strictly positive. -/
theorem uQ_real_pos : 0 < uQ_real := by
  unfold uQ_real uQ; push_cast; norm_num

/-- The period `u` is strictly larger than `1/2`. -/
theorem uQ_real_gt_half : (1 : ℝ) / 2 < uQ_real := by
  unfold uQ_real uQ; push_cast; norm_num

/-- The period `u` satisfies `1/2 ≤ u`. -/
theorem uQ_real_ge_half : (1 : ℝ) / 2 ≤ uQ_real := le_of_lt uQ_real_gt_half

/-! ## Multi-scale arcsine kernel `K_ms`

The kernel used in the paper and the Python certifier is the
*autoconvolution* of the half-arcsine density `η_δ` on `(-δ/2, δ/2)`,
not the bare arcsine density.  The bare arcsine density is not
Bochner-admissible (its Fourier transform `J₀(2πδξ)` oscillates and
can be negative) and has `∫ K² = ∞`; the autoconvolution form has
nonnegative Fourier transform `J₀(πδξ)²` and finite `L²` norm, which
is what the numerical axiom `K2_analytic ≤ K2UpperQ` quantifies.

**Definition** (this file):
  * `eta δ x := (1/δ)·(2/π) / √(1 - (2x/δ)²) · 𝟙_{|x| < δ/2}`  — half-arcsine
    density on `(-δ/2, δ/2)`, mass 1, Fourier transform `J₀(πδξ)`.
  * `K_arc(δ, x) := (η_δ * η_δ)(x)`  — the *autoconvolution*, supported
    on `(-δ, δ)`, mass 1, Fourier transform `J₀(πδξ)²` (nonneg ⇒
    Bochner-admissible).
  * `K_ms x := λ₁·K_arc(δ₁, x) + λ₂·K_arc(δ₂, x) + λ₃·K_arc(δ₃, x)`.

Properties (now genuinely provable for the autoconvolution form):
  * `K_ms ≥ 0`            (convolution of nonneg integrands is nonneg).
  * `∫ K_ms = 1`          (`∫ η * η = (∫η)² = 1` via Fubini, then linear
                            combination).
  * `supp(K_ms) ⊆ [-δ₁, δ₁]`  (`supp(η * η) ⊆ supp η + supp η =
                                (-δ/2, δ/2) + (-δ/2, δ/2) ⊆ [-δ, δ]`).
  * `K_ms x = 0` for `|x| ≥ δ₁` (by construction).
  * `K_ms ∈ L²`           (convolution of L¹ × L² is L²; in fact `η`
                            is L² so `η * η ∈ L²`).
-/

/-- The half-arcsine density of half-width `δ/2`:

  `η_δ(x) = (1/δ)·(2/π) / √(1 - (2x/δ)²) · 𝟙_{|x| < δ/2}`.

Algebraic simplification (`1 - (2x/δ)² = 4·((δ/2)² - x²)/δ²` for `δ > 0`)
gives the equivalent form

  `η_δ(x) = (1/π) / √((δ/2)² - x²) · 𝟙_{|x| < δ/2}`,

which is the bare arcsine density of half-width `δ/2`.  We use the
simplified form for the Lean definition (mass 1 follows from
`Sidon.Bessel.arcsine_moment_integral` at `k = 0` and width `δ/2`).
`η_δ` has Fourier transform `J₀(πδξ)`. -/
noncomputable def eta (δ x : ℝ) : ℝ :=
  if |x| < δ / 2 then (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - x^2))⁻¹ else 0

/-- The arcsine MV kernel of half-width `δ`: the autoconvolution `η_δ * η_δ`.
This is supported in `(-δ, δ)`, has mass `1`, and Fourier transform
`J₀(πδξ)²`. -/
noncomputable def K_arc (δ : ℝ) (x : ℝ) : ℝ :=
  MeasureTheory.convolution (eta δ) (eta δ) (ContinuousLinearMap.mul ℝ ℝ)
    MeasureTheory.volume x

/-- The 3-scale arcsine kernel
    `K_ms(x) = λ₁·K_arc(δ₁, x) + λ₂·K_arc(δ₂, x) + λ₃·K_arc(δ₃, x)`. -/
noncomputable def K_ms (x : ℝ) : ℝ :=
  lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x + lambda3 * K_arc delta3 x

/-- The half-arcsine density is nonnegative. -/
theorem eta_nonneg (δ x : ℝ) (hδ : 0 < δ) : 0 ≤ eta δ x := by
  unfold eta
  split_ifs with h
  · apply mul_nonneg
    · apply div_nonneg (by norm_num : (0:ℝ) ≤ 1); exact le_of_lt Real.pi_pos
    · exact inv_nonneg.mpr (Real.sqrt_nonneg _)
  · exact le_refl 0

/-- The autoconvolution kernel is nonnegative.  The convolution
`(η * η)(x) = ∫ η(t)·η(x-t) dt` is an integral of a pointwise-nonnegative
integrand (each `η` factor is `≥ 0`), so the result is `≥ 0`. -/
theorem K_arc_nonneg (δ : ℝ) (hδ : 0 < δ) (x : ℝ) : 0 ≤ K_arc δ x := by
  unfold K_arc
  exact convolution_nonneg (eta_nonneg δ · hδ) (eta_nonneg δ · hδ) x

/-- The multi-scale arcsine kernel is nonnegative. -/
theorem K_ms_nonneg (x : ℝ) : 0 ≤ K_ms x := by
  unfold K_ms
  have hδ := delta_positivities
  have hlams := lambdas_nonneg
  have hl1 : 0 ≤ lambda1 := by unfold lambda1; exact_mod_cast hlams.1
  have hl2 : 0 ≤ lambda2 := by unfold lambda2; exact_mod_cast hlams.2.1
  have hl3 : 0 ≤ lambda3 := by unfold lambda3; exact_mod_cast hlams.2.2
  have h1 : 0 ≤ lambda1 * K_arc delta1 x :=
    mul_nonneg hl1 (K_arc_nonneg _ hδ.1 _)
  have h2 : 0 ≤ lambda2 * K_arc delta2 x :=
    mul_nonneg hl2 (K_arc_nonneg _ hδ.2.1 _)
  have h3 : 0 ≤ lambda3 * K_arc delta3 x :=
    mul_nonneg hl3 (K_arc_nonneg _ hδ.2.2 _)
  linarith

/-- `eta δ x = 0` whenever `|x| ≥ δ/2`. -/
theorem eta_eq_zero_outside (δ x : ℝ) (h : δ / 2 ≤ |x|) : eta δ x = 0 := by
  unfold eta
  split_ifs with hcase
  · linarith
  · rfl

/-- The support of `eta δ` is contained in `Ioo (-δ/2) (δ/2)`. -/
theorem eta_support_subset (δ : ℝ) :
    Function.support (eta δ) ⊆ Set.Ioo (-(δ/2)) (δ/2) := by
  intro x hx
  by_contra h_not
  apply hx
  show eta δ x = 0
  have h_abs : δ/2 ≤ |x| := by
    by_contra h_lt
    push_neg at h_lt
    exact h_not ⟨(abs_lt.mp h_lt).1, (abs_lt.mp h_lt).2⟩
  exact eta_eq_zero_outside δ x h_abs

/-- `eta δ` equals the indicator of `Ioo (-δ/2) (δ/2)` of `(1/π)·(sqrt((δ/2)²-x²))⁻¹`. -/
theorem eta_eq_indicator (δ : ℝ) :
    ∀ x, eta δ x =
      (Set.Ioo (-(δ/2)) (δ/2)).indicator
        (fun x => (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - x^2))⁻¹) x := by
  intro x
  unfold eta
  by_cases h : |x| < δ/2
  · have hx_mem : x ∈ Set.Ioo (-(δ/2)) (δ/2) := by
      rw [abs_lt] at h; exact ⟨h.1, h.2⟩
    rw [if_pos h, Set.indicator_of_mem hx_mem]
  · have hx_not_mem : x ∉ Set.Ioo (-(δ/2)) (δ/2) := by
      intro hmem
      apply h
      rw [abs_lt]
      exact ⟨hmem.1, hmem.2⟩
    rw [if_neg h, Set.indicator_of_notMem hx_not_mem]

/-- The integral of `eta δ` over `ℝ` equals `1` (mass of half-arcsine of width `δ/2`).

Routes through `Sidon.Bessel.arcsine_moment_integral` at width `δ/2` and `k=0`. -/
theorem eta_integral_eq_one (δ : ℝ) (hδ : 0 < δ) :
    ∫ x, eta δ x ∂volume = 1 := by
  have hδ_half : 0 < δ/2 := by linarith
  -- Step 1: rewrite η as the indicator form on Ioo(-δ/2, δ/2).
  have h_pt := eta_eq_indicator δ
  -- Step 2: convert integral.
  calc ∫ x, eta δ x ∂volume
      = ∫ x, (Set.Ioo (-(δ/2)) (δ/2)).indicator
            (fun x => (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - x^2))⁻¹) x ∂volume := by
        apply integral_congr_ae
        exact Filter.Eventually.of_forall h_pt
    _ = ∫ x in Set.Ioo (-(δ/2)) (δ/2),
            (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - x^2))⁻¹ ∂volume := by
        rw [integral_indicator measurableSet_Ioo]
    _ = (1 / Real.pi) *
            ∫ x in Set.Ioo (-(δ/2)) (δ/2), (Real.sqrt ((δ/2)^2 - x^2))⁻¹ ∂volume := by
        rw [integral_const_mul]
    _ = (1 / Real.pi) * Real.pi := by
        -- This is arcsine_moment_integral at k=0 with width δ/2.
        have h_arc := Sidon.Bessel.arcsine_moment_integral hδ_half 0
        simp only [Nat.mul_zero, pow_zero, Nat.choose_self, Nat.cast_one, mul_one,
                   div_one] at h_arc
        -- h_arc: ∫ x in Ioo(-δ/2, δ/2), 1 / √((δ/2)² - x²) = π
        have heq : (fun x : ℝ => (Real.sqrt ((δ/2)^2 - x^2))⁻¹) =
                   (fun x => 1 / Real.sqrt ((δ/2)^2 - x^2)) := by
          funext x; rw [one_div]
        rw [heq]
        rw [h_arc]
    _ = 1 := by
        rw [one_div, inv_mul_cancel₀ Real.pi_ne_zero]

/-- `eta δ` is integrable: its integral equals `1 ≠ 0`. -/
theorem eta_integrable (δ : ℝ) (hδ : 0 < δ) : Integrable (eta δ) volume := by
  apply Integrable.of_integral_ne_zero
  rw [eta_integral_eq_one δ hδ]
  exact one_ne_zero

/-- The autoconvolution `K_arc(δ, ·) = η_δ * η_δ` is integrable. -/
theorem K_arc_integrable (δ : ℝ) (hδ : 0 < δ) :
    Integrable (K_arc δ) volume := by
  unfold K_arc
  exact (eta_integrable δ hδ).integrable_convolution _ (eta_integrable δ hδ)

/-- The autoconvolution kernel integrates to `1`:
`∫ K_arc(δ, ·) = ∫ (η_δ * η_δ) = (∫ η_δ) · (∫ η_δ) = 1 · 1 = 1`,
via `MeasureTheory.integral_convolution`. -/
theorem K_arc_integral_eq_one (δ : ℝ) (hδ : 0 < δ) :
    ∫ x, K_arc δ x ∂volume = 1 := by
  unfold K_arc
  rw [integral_convolution (ContinuousLinearMap.mul ℝ ℝ)
        (eta_integrable δ hδ) (eta_integrable δ hδ)]
  -- The result is mul (∫ η) (∫ η) = (∫ η) * (∫ η) in ℝ.
  rw [eta_integral_eq_one δ hδ]
  -- Now goal: (ContinuousLinearMap.mul ℝ ℝ) 1 1 = 1.
  simp [ContinuousLinearMap.mul_apply']

/-- The support of `K_arc(δ, ·)` is contained in `[-δ, δ]`.
Specifically, `K_arc(δ, x) = 0` whenever `|x| ≥ δ` (since
`supp(η * η) ⊆ supp(η) + supp(η) ⊆ (-δ/2, δ/2) + (-δ/2, δ/2) ⊆ (-δ, δ)`). -/
theorem K_arc_eq_zero_outside (δ x : ℝ) (h : δ ≤ |x|) : K_arc δ x = 0 := by
  unfold K_arc
  -- Use support_convolution_subset: supp(f ⋆ g) ⊆ supp(f) + supp(g).
  -- supp(η δ) ⊆ Ioo(-δ/2, δ/2), so supp(η * η) ⊆ Ioo(-δ, δ).
  -- |x| ≥ δ means x ∉ Ioo(-δ, δ), hence (η*η)(x) = 0.
  by_contra h_ne
  -- h_ne : convolution η η at x ≠ 0
  -- So x ∈ support of convolution.
  have hx_supp :
      x ∈ Function.support
        (MeasureTheory.convolution (eta δ) (eta δ)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) := h_ne
  have h_subset :=
    MeasureTheory.support_convolution_subset
      (L := ContinuousLinearMap.mul ℝ ℝ)
      (f := eta δ) (g := eta δ) (μ := MeasureTheory.volume)
  -- h_subset : support (η * η) ⊆ support (η) + support (η)
  have hx_sum : x ∈ Function.support (eta δ) + Function.support (eta δ) :=
    h_subset hx_supp
  -- Now show this is impossible when |x| ≥ δ.
  rw [Set.mem_add] at hx_sum
  obtain ⟨a, ha, b, hb, hab⟩ := hx_sum
  have ha_supp : a ∈ Set.Ioo (-(δ/2)) (δ/2) := eta_support_subset δ ha
  have hb_supp : b ∈ Set.Ioo (-(δ/2)) (δ/2) := eta_support_subset δ hb
  have ha_lt : -(δ/2) < a := ha_supp.1
  have ha_gt : a < δ/2 := ha_supp.2
  have hb_lt : -(δ/2) < b := hb_supp.1
  have hb_gt : b < δ/2 := hb_supp.2
  -- x = a + b ∈ (-δ, δ), but |x| ≥ δ — contradiction.
  have h_sum_lt : -δ < a + b := by linarith
  have h_sum_gt : a + b < δ := by linarith
  have h_x_lt : -δ < x := hab ▸ h_sum_lt
  have h_x_gt : x < δ := hab ▸ h_sum_gt
  -- |x| < δ contradicts δ ≤ |x|.
  have : |x| < δ := by
    rw [abs_lt]; exact ⟨h_x_lt, h_x_gt⟩
  linarith

/-! ## Concrete QP cosine multiplier `G` (200 rational coefficients)

The 200 rational coefficients `a_j` (j = 1..200) of the QP-optimised
cosine multiplier `G(x) = ∑_{j=1}^{200} a_j cos(2π j x / u)`, as produced
by the certifier `delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`
and recorded in `multiscale_arcsine_1292.json` (SHA-256 `5fa9ae37…`),
common denominator `10⁸`. -/

/-- The 200 QP coefficient numerators with common denominator `10⁸`. -/
def qpNumerators : List ℤ := [
    190807542,  -- a_1
    -161500769,  -- a_2
    117725683,  -- a_3
    -72411108,  -- a_4
    30847109,  -- a_5
    1993162,  -- a_6
    -18299784,  -- a_7
    21082470,  -- a_8
    -16524158,  -- a_9
    5261307,  -- a_10
    6185741,  -- a_11
    -10943773,  -- a_12
    11993526,  -- a_13
    -9820731,  -- a_14
    2796912,  -- a_15
    2763767,  -- a_16
    -4328666,  -- a_17
    5508653,  -- a_18
    -3927028,  -- a_19
    -571906,  -- a_20
    2099377,  -- a_21
    -1810436,  -- a_22
    2005291,  -- a_23
    825591,  -- a_24
    -4305400,  -- a_25
    4283170,  -- a_26
    -4059618,  -- a_27
    3511151,  -- a_28
    140934,  -- a_29
    -2582597,  -- a_30
    2366839,  -- a_31
    -2935072,  -- a_32
    1912738,  -- a_33
    1428526,  -- a_34
    -2454528,  -- a_35
    2401573,  -- a_36
    -3191125,  -- a_37
    1486416,  -- a_38
    1049751,  -- a_39
    -1143221,  -- a_40
    1649652,  -- a_41
    -2261780,  -- a_42
    329757,  -- a_43
    841180,  -- a_44
    -303503,  -- a_45
    945843,  -- a_46
    -636964,  -- a_47
    -1391607,  -- a_48
    1582797,  -- a_49
    -1178999,  -- a_50
    1734673,  -- a_51
    -500734,  -- a_52
    -1195281,  -- a_53
    914797,  -- a_54
    -1057636,  -- a_55
    1298174,  -- a_56
    395895,  -- a_57
    -1369491,  -- a_58
    1029562,  -- a_59
    -1577383,  -- a_60
    1234867,  -- a_61
    453329,  -- a_62
    -762491,  -- a_63
    767003,  -- a_64
    -1456373,  -- a_65
    667266,  -- a_66
    452672,  -- a_67
    -222697,  -- a_68
    502005,  -- a_69
    -798765,  -- a_70
    -328538,  -- a_71
    828997,  -- a_72
    -520932,  -- a_73
    958258,  -- a_74
    -664839,  -- a_75
    -467125,  -- a_76
    497666,  -- a_77
    -441751,  -- a_78
    808177,  -- a_79
    -31703,  -- a_80
    -783218,  -- a_81
    612217,  -- a_82
    -859612,  -- a_83
    898076,  -- a_84
    110472,  -- a_85
    -526920,  -- a_86
    481538,  -- a_87
    -934099,  -- a_88
    630008,  -- a_89
    202556,  -- a_90
    -224246,  -- a_91
    369798,  -- a_92
    -695613,  -- a_93
    77022,  -- a_94
    419609,  -- a_95
    -301792,  -- a_96
    619948,  -- a_97
    -617206,  -- a_98
    -104323,  -- a_99
    247328,  -- a_100
    -212046,  -- a_101
    541894,  -- a_102
    -167518,  -- a_103
    -435152,  -- a_104
    371893,  -- a_105
    -526946,  -- a_106
    672507,  -- a_107
    -32942,  -- a_108
    -345464,  -- a_109
    311249,  -- a_110
    -651636,  -- a_111
    553585,  -- a_112
    79000,  -- a_113
    -180475,  -- a_114
    271733,  -- a_115
    -602415,  -- a_116
    241222,  -- a_117
    221598,  -- a_118
    -179998,  -- a_119
    430827,  -- a_120
    -588470,  -- a_121
    87616,  -- a_122
    118702,  -- a_123
    -74805,  -- a_124
    388386,  -- a_125
    -275450,  -- a_126
    -216971,  -- a_127
    226003,  -- a_128
    -294260,  -- a_129
    531518,  -- a_130
    -162360,  -- a_131
    -222553,  -- a_132
    190068,  -- a_133
    -419128,  -- a_134
    509286,  -- a_135
    -46411,  -- a_136
    -148325,  -- a_137
    166210,  -- a_138
    -461511,  -- a_139
    353868,  -- a_140
    72115,  -- a_141
    -125576,  -- a_142
    256378,  -- a_143
    -505158,  -- a_144
    246098,  -- a_145
    37609,  -- a_146
    -21844,  -- a_147
    231060,  -- a_148
    -330404,  -- a_149
    -765,  -- a_150
    135470,  -- a_151
    -143512,  -- a_152
    359659,  -- a_153
    -272106,  -- a_154
    -59002,  -- a_155
    117843,  -- a_156
    -217422,  -- a_157
    389002,  -- a_158
    -181959,  -- a_159
    -66018,  -- a_160
    103412,  -- a_161
    -274905,  -- a_162
    351216,  -- a_163
    -97474,  -- a_164
    -61165,  -- a_165
    131136,  -- a_166
    -322832,  -- a_167
    298544,  -- a_168
    -71186,  -- a_169
    -13710,  -- a_170
    118566,  -- a_171
    -265799,  -- a_172
    162333,  -- a_173
    11409,  -- a_174
    -72968,  -- a_175
    202267,  -- a_176
    -273812,  -- a_177
    127885,  -- a_178
    3903,  -- a_179
    -84625,  -- a_180
    220386,  -- a_181
    -232038,  -- a_182
    95935,  -- a_183
    4673,  -- a_184
    -113707,  -- a_185
    238537,  -- a_186
    -223671,  -- a_187
    124269,  -- a_188
    -44843,  -- a_189
    -68069,  -- a_190
    159126,  -- a_191
    -136234,  -- a_192
    74755,  -- a_193
    -11706,  -- a_194
    -84315,  -- a_195
    147032,  -- a_196
    -141088,  -- a_197
    118984,  -- a_198
    -83669,  -- a_199
    29587  -- a_200
  ]

theorem qpNumerators_length : qpNumerators.length = 200 := by
  native_decide

/-- The QP-optimised cosine multiplier
`G(x) = ∑_{j=1}^{200} a_j · cos(2π j x / u)`,
where `a_j = qpNumerators[j-1] / 10⁸`. -/
noncomputable def G_concrete (x : ℝ) : ℝ :=
  ∑ i ∈ Finset.range 200,
    ((qpNumerators.getD i 0 : ℝ) / 100000000) *
    Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real)

/-- The period-`u` Fourier coefficient of `K_ms` at integer index `j`,
in closed form using the Bessel function:
`K̃_ms(j) = ∑_i λ_i · J₀(π j δ_i / u)²`.
This is the standard formula for the autoconvolution arcsine kernel. -/
noncomputable def Ktilde_ms (j : ℕ) : ℝ :=
  lambda1 * (Sidon.Bessel.besselJ0 (Real.pi * (j : ℝ) * delta1 / uQ_real))^2 +
  lambda2 * (Sidon.Bessel.besselJ0 (Real.pi * (j : ℝ) * delta2 / uQ_real))^2 +
  lambda3 * (Sidon.Bessel.besselJ0 (Real.pi * (j : ℝ) * delta3 / uQ_real))^2

/-- The QP-side dual sum `S_1 = ∑_{j=1}^{200} a_j² / K̃_ms(j)`,
as a real-valued analytic functional of the concrete coefficients
and the Bessel-form K̃_ms. -/
noncomputable def S_1_analytic : ℝ :=
  ∑ i ∈ Finset.range 200,
    (((qpNumerators.getD i 0 : ℝ) / 100000000))^2 / Ktilde_ms (i + 1)

/-- The analytic minimum `min_{x ∈ [0, 1/4]} G_concrete(x)`. -/
noncomputable def min_G_analytic : ℝ :=
  sInf (G_concrete '' Set.Icc (0 : ℝ) (1/4))

/-! ## Analytic primitives `K2_analytic` and `gain_analytic`

These are the two analytic functionals that drive the master
inequality.  They are defined symbolically over the explicit
3-scale kernel `K_ms`, and bounded numerically by the certifier.

`K2_analytic := ∫ ‖K_ms‖²` is the L² norm squared of `K_ms` on `ℝ`.
This equals `2·∫_0^∞ K̂_ms(ξ)² dξ` (by Plancherel on ℝ), which is
what the `flint.arb` certifier reports.

`gain_analytic := (4/u)·min_G²/S_1` is a **concrete defined** real
expression formed from the 200 QP coefficients (`G_concrete` /
`min_G_analytic`) and the Bessel-form period-`u` Fourier coefficients
of `K_ms` (`Ktilde_ms` / `S_1_analytic`).  The certifier reports a
numerical lower bound for this functional, which is what the slack
rational `gainLowerQ = 20925/100000` records.
-/

/-- The L² norm squared of `K_ms`: `K_2(K_ms) := ∫ ‖K_ms(x)‖² dx`.

This is a concrete real integral over the explicit multi-scale
arcsine kernel.  The numerical axiom `K2_analytic_le_K2UpperQ`
asserts that this integral is bounded above by the slack rational
`K2UpperQ`. -/
noncomputable def K2_analytic : ℝ := ∫ x, K_ms x ^ 2 ∂volume

/-- Analytic gain functional `(4/u) · min_G² / S_1`.  This is now a
**concrete defined** real expression involving the 200 QP coefficients
and the Bessel-form `K̃_ms`, not an opaque symbol.  The numerical axiom
`gain_analytic_ge_gainLowerQ` (below) bounds this analytic functional. -/
noncomputable def gain_analytic : ℝ :=
  (4 / uQ_real) * min_G_analytic^2 / S_1_analytic

/-! ## Numerical kernel-specific axioms

These are the **only** kernel-specific user axioms in the
dependency closure of the headline theorem.  Both are *numerical*:
they assert that the analytic primitives `K2_analytic` and
`gain_analytic` lie on the correct side of the slack rationals
`K2UpperQ` and `gainLowerQ`.

These axioms are analogues of "Mathematica computed this value" in
MV's paper — they are certifier outputs, not analytic content.  The
underlying numerical computations are performed by `flint.arb` at
256-bit precision (`delsarte_dual/grid_bound_alt_kernel/`).
-/

/-- (NUMERICAL AXIOM 1, paper Lemma 4.2) `K_2(K_ms) ≤ K2UpperQ`.

The `flint.arb` certifier reports `K_2(K_ms) ∈ [4.788823, 4.788906]`
(radius `< 10⁻⁴`) at 256-bit precision via:

  * Bochner-positivity check on `K̂_ms(ξ) = ∑ᵢ λᵢ · J₀(πδᵢξ)²`;
  * Numerical integration `2·∫_0^∞ K̂_ms(ξ)² dξ` with truncation at
    `XI_MAX = 1e5` and closed-form Watson tail correction;
  * Plancherel identification `K_2 = ‖K̂‖₂² = ∫ ‖K‖²` on `ℝ`.

The slack `K2UpperQ = 47897/10000 = 4.7897` exceeds the certifier's
upper endpoint with margin `≈ 7.9 × 10⁻⁴`.

*Provenance.* The published Matolcsi–Vinuesa (2010) proof assumes the
analogous bound `‖K‖₂² ≤ 0.5747/δ` for the single-scale arcsine kernel
(citing Mathematica / the Martin–O'Bryant Lemma 3.2 surrogate); this
axiom plays the strictly-more-rigorous multi-scale *analogous role*,
certified by `flint.arb` interval arithmetic rather than heuristic CAS
numerics. It is **not the same fact**: it is a *new* numerical
inequality specific to the three-scale kernel `K_ms`, not a statement
contained in MV 2010. -/
axiom K2_analytic_le_K2UpperQ : K2_analytic ≤ (K2UpperQ : ℝ)

/-- (NUMERICAL AXIOM 2, paper Lemmas 4.3-4.5) `gain_analytic ≥ gainLowerQ`.

The `flint.arb` certifier reports `gain_analytic ≥ 0.21009214`
(coupled-arb value, radius `< 10⁻⁸`) via:

  * Solve a 200-coefficient QP in `mosek` / `clarabel` for the
    cosine `G(x) = ∑_{j=1}^{200} aⱼ·cos(2πjx/u)` minimising
    `S_1 = ∑_j aⱼ² / K̂_ms(j/u)` subject to `G(x) ≥ 1` on `[0, 1/4]`;
  * Round the optimal `aⱼ` to rationals (denominator `10⁸`);
  * Verify `min_{x ∈ [0, 1/4]} G(x) ≥ 0.99997987` via Taylor-2 B&B
    in arb interval arithmetic (32768 cells on `[0, 1/4]`);
  * Compute `S_1 ≤ 29.840907` from the rationals;
  * Bound `gain = (4/u)·(min G)²/S_1 ≥ 0.21009214`.

The slack `gainLowerQ = 20925/100000 = 0.20925` is below the
certifier's lower endpoint with margin `≈ 8.4 × 10⁻⁴`.

*Provenance.* The published Matolcsi–Vinuesa (2010) proof assumes the
analogous gain value `a = 0.0713` for its single-scale construction
(citing Mathematica for `m_G`, `S_1` and the resulting `a`); this axiom
plays the strictly-more-rigorous multi-scale *analogous role* (the
larger value `≥ 0.20925` is what the re-optimised 200-cosine `G` buys),
certified by `flint.arb` rather than heuristic CAS numerics. It is
**not the same fact**: it is a *new* numerical inequality specific to
the three-scale kernel and the re-optimised 200-cosine `G`, not a
statement contained in MV 2010. -/
axiom gain_analytic_ge_gainLowerQ : gain_analytic ≥ (gainLowerQ : ℝ)

/-! ## Slack-rational soundness theorems

Each statement below is a pure rational comparison between the slack
rational chosen above and the decimal value reported by the `flint.arb`
certifier (paper Lemmas 4.1-4.5).  They are *not* axioms about the
analytic functionals `K_2`, `k_1`, `S_1`, `min_G`, `a` (those bounds are
discharged externally by the certifier and used inside the master
inequality wire-up); they record that the rational slack used
downstream is on the correct side of the certifier output.
Each is a one-line `norm_num` check.
-/

/-- (N1) The Lean slack `K2UpperQ = 47897/10000` is at least the
    certifier's reported upper endpoint `K_2 ≤ 4.788906`. -/
theorem K_two_upper_bound : (K2UpperQ : ℝ) ≥ (4788906 / 1000000 : ℝ) := by
  unfold K2UpperQ; norm_num

/-- (N2) The Lean slack `K1LowerQ = 9212/10000` is at most the
    certifier's reported lower endpoint `k_1 ≥ 0.92124658`. -/
theorem k_one_lower_bound : (K1LowerQ : ℝ) ≤ (92124658 / 100000000 : ℝ) := by
  unfold K1LowerQ; norm_num

/-- (N3) The Lean slack `S1UpperQ = 29841/1000` is at least the
    certifier's reported upper endpoint `S_1 ≤ 29.84091`. -/
theorem S_one_upper_bound : (S1UpperQ : ℝ) ≥ (2984091 / 100000 : ℝ) := by
  unfold S1UpperQ; norm_num

/-- (N4) The Lean slack `minGLowerQ = 998/1000` is at most the
    certifier's reported lower endpoint `min_G ≥ 0.9999798`. -/
theorem min_G_lower_bound : (minGLowerQ : ℝ) ≤ (9999798 / 10000000 : ℝ) := by
  unfold minGLowerQ; norm_num

/-- (N5) The Lean slack `gainLowerQ = 20925/100000` is at most the
    certifier's reported lower endpoint `a ≥ 0.21009214` (coupled-arb)
    and at most the looser rational floor `(4/u)·(998/1000)²/(29841/1000)
    ≈ 0.20926` derivable from N3+N4. -/
theorem gain_lower_bound : (gainLowerQ : ℝ) ≤ (21009214 / 100000000 : ℝ) := by
  unfold gainLowerQ; norm_num

/-! ## Slack-monotonicity wire-up

The lemma below packages the *purely-real-algebraic* step that lifts the
master inequality at the analytic anchors `(K_2_analytic, gain_analytic)`
to the slack rationals `(K2UpperQ, gainLowerQ)` used in the axiom
statement.

It takes:
  * `h_K2_bound`     : `K_2_analytic ≤ K2UpperQ`           (Lemma 4.2)
  * `h_K2_ge_1`      : `1 ≤ K_2_analytic`                  (forced by `∫K=1`,
                                                            Bochner positivity)
  * `h_gain_bound`   : `2·m_G²/S_G ≥ gainLowerQ`           (Lemma 4.3-4.5)
  * `h_R_ge_1`       : `1 ≤ R(f)`                         (from `∫f²·∫1 ≥ (∫f)²`
                                                            via Cauchy-Schwarz
                                                            plus convolution
                                                            structure)
  * `h_master`       : the master inequality at the analytic anchors
                       (Eq.(6) for `Minf = R(f)`, `K_2 = K_2_analytic`,
                        `2 m_G²/S_G = gain_analytic`).

and produces the axiom's conclusion (master inequality at the slack
anchors).

The wire-up is *real-algebraic* monotonicity: replacing `K_2_analytic`
with the upper bound `K2UpperQ` weakens the LHS factor
`√(K_2 - 1)` to a larger value, and replacing `2·m_G²/S_G` with the
lower bound `gainLowerQ` weakens the RHS.  No analytic input beyond the
five hypotheses is required.

This lemma is *theorem*-level (zero axioms beyond the three Lean core
axioms); together with the analytic discharge of `h_master` (now the
theorem `MV_master_inequality_for_extremiser`, which depends only on
the two numerical axioms `K2_analytic_le_K2UpperQ` and
`gain_analytic_ge_gainLowerQ`), it proves
`autoconvolution_ratio_ge_1292_1000`. -/

/-- Real-algebraic monotonicity lift of the MV master inequality from
the analytic anchors `(K_2_analytic, 2·m_G²/S_G)` to the slack
rationals `(K2UpperQ, gainLowerQ)`.  MV (2010) Eq. (6) at the slack
anchors; no analytic input beyond the five hypotheses is required. -/
theorem MV_master_via_slack_monotonicity
    (f : ℝ → ℝ)
    (K_2_analytic m_G_sq S_G : ℝ)
    (h_K2_bound : K_2_analytic ≤ (K2UpperQ : ℝ))
    (_h_K2_ge_1 : 1 ≤ K_2_analytic)
    (h_gain_bound : 2 * m_G_sq / S_G ≥ (gainLowerQ : ℝ))
    (_h_R_ge_1 : 1 ≤ autoconvolution_ratio f)
    (h_master :
      autoconvolution_ratio f + 1 +
        Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt (K_2_analytic - 1)
        ≥ 2 / (uQ : ℝ) + 2 * m_G_sq / S_G) :
    autoconvolution_ratio f + 1 +
      Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
      ≥ 2 / (uQ : ℝ) + (gainLowerQ : ℝ) := by
  -- Step 1: `√(K_2_analytic - 1) ≤ √(K2UpperQ - 1)` by `Real.sqrt_le_sqrt`.
  have h_K2_sub : K_2_analytic - 1 ≤ (K2UpperQ : ℝ) - 1 := by linarith
  have h_sqrt_K2 :
      Real.sqrt (K_2_analytic - 1) ≤ Real.sqrt ((K2UpperQ : ℝ) - 1) :=
    Real.sqrt_le_sqrt h_K2_sub
  -- Step 2: `√(R(f) - 1) ≥ 0` so the multiplication preserves the
  -- inequality.
  have h_sqrt_R_nn : 0 ≤ Real.sqrt (autoconvolution_ratio f - 1) :=
    Real.sqrt_nonneg _
  have h_LHS_le :
      autoconvolution_ratio f + 1 +
          Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt (K_2_analytic - 1)
        ≤ autoconvolution_ratio f + 1 +
          Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1) := by
    have h_mul := mul_le_mul_of_nonneg_left h_sqrt_K2 h_sqrt_R_nn
    linarith
  -- Step 3: chain with `h_master ≥ 2/u + 2·m_G²/S_G ≥ 2/u + gainLowerQ`.
  have h_RHS_ge :
      2 / (uQ : ℝ) + 2 * m_G_sq / S_G ≥ 2 / (uQ : ℝ) + (gainLowerQ : ℝ) := by
    linarith
  linarith

/-! ## Full atomic-inputs wire-up

The lemma below composes
  * `Sidon.Master.master_inequality_from_lemmas` (chains MV Eqs.(1)-(4)
    into Eq.(6) algebraically), and
  * `MV_master_via_slack_monotonicity` (lifts the analytic anchors to
    the slack rationals via real-algebraic monotonicity).

The hypotheses are precisely the conclusions of `Sidon.MV.mv_eq1`,
`Sidon.MV.mv_eq2`, `Sidon.MV.mv_eq3`, `Sidon.MV.mv_eq4`, plus the two
kernel-specific analytic bounds (5)-(6).  Currently those four
hypotheses are *theorems* (zero axioms) of `Sidon.MVLemmas`, but they
take atomic sub-hypotheses (Parseval splits, periodisation identities,
inner-product floor) as inputs because the underlying analytic
discharge is not yet formalised on the real line for non-Schwartz
data.

This is the algebraic spine of `MV_master_inequality_for_extremiser`
(now a Lean theorem, see below).  The remaining gap is the
*analytic* one: instantiating the atomic hypotheses of `mv_eq2` /
`mv_eq3` for the concrete `(f, K_ms)` pair on ℝ — this is the content
packaged in the `ExtremiserPrimitives f` bundle that the headline
theorem takes as a hypothesis.

Note the rational `K2UpperQ - 1 = 37897/10000 > 0`, so the slack
`√(K2UpperQ - 1)` is defined and nonzero. -/

/-- Algebraic chain of MV Lemma 3.1 Eqs. (1)–(4) into the MV master
inequality at the slack rationals.  Composes
`Sidon.Master.master_inequality_from_lemmas` (the MV Eq. (6) assembly)
with `MV_master_via_slack_monotonicity` (the lift to slack anchors)
under the two kernel-specific bounds `K_2 ≤ K2UpperQ` and
`2·m_G²/S_G ≥ gainLowerQ`. -/
theorem MV_master_inequality_from_MV_lemmas
    (f : ℝ → ℝ)
    (K_2_analytic m_G S_G S_cos LHS1 LHS2 : ℝ)
    (h_K2_bound : K_2_analytic ≤ (K2UpperQ : ℝ))
    (h_K2_ge_1 : 1 ≤ K_2_analytic)
    (h_gain_bound : 2 * m_G ^ 2 / S_G ≥ (gainLowerQ : ℝ))
    (h_R_ge_1 : 1 ≤ autoconvolution_ratio f)
    (h_S_G_pos : 0 < S_G)
    -- MV Eq.(1) conclusion:
    (hEq1 : LHS1 ≤ autoconvolution_ratio f)
    -- MV Eq.(2) conclusion:
    (hEq2 : LHS2 ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                          * Real.sqrt (K_2_analytic - 1))
    -- MV Eq.(3) (inequality form, Bochner-discharge-able):
    (hEq3_ge : 2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2 * S_cos ≤ LHS1 + LHS2)
    -- MV Eq.(4) conclusion:
    (hEq4 : (uQ : ℝ) ^ 2 * S_cos ≥ m_G ^ 2 / S_G) :
    autoconvolution_ratio f + 1 +
      Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
      ≥ 2 / (uQ : ℝ) + (gainLowerQ : ℝ) := by
  -- `uQ > 0` (rational arithmetic).
  have huQ_pos : (0 : ℝ) < (uQ : ℝ) := by unfold uQ; push_cast; norm_num
  -- Assemble Eq.(6) at the analytic anchors via `master_inequality_from_lemmas`.
  have h_master :
      autoconvolution_ratio f + 1 +
        Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt (K_2_analytic - 1)
        ≥ 2 / (uQ : ℝ) + 2 * m_G ^ 2 / S_G :=
    Sidon.Master.master_inequality_from_lemmas
      (autoconvolution_ratio f) K_2_analytic m_G S_G (uQ : ℝ)
      S_cos LHS1 LHS2 huQ_pos h_S_G_pos hEq1 hEq2 hEq3_ge hEq4
  -- Lift to the slack rationals.
  exact MV_master_via_slack_monotonicity f K_2_analytic (m_G ^ 2) S_G
          h_K2_bound h_K2_ge_1 h_gain_bound h_R_ge_1 h_master

/-! ## Axiom (structural content)

The MV master inequality, specialised to the 3-scale kernel.  The
Fourier reduction on `ℝ/uℤ` proceeds exactly as in the single-scale case
(MV 2010, Lemma 3.1): replacing `J₀(πδξ)²` by the λ-weighted sum
`∑ᵢ λᵢ J₀(πδᵢξ)²` preserves Bochner-positivity (a positive linear
combination of squared real Bessel functions remains positive
semi-definite as a Fourier transform), and the cosine `G` enters only
through the gain parameter `a`.

**Remaining analytic content (what an axiom-free proof would require):**

Per `Sidon.Master.master_inequality_from_lemmas`, the inequality below
factors through four atomic analytic primitives (MV Lemma 3.1 Eqs. 1-4):

  1. `mv_eq1` — `∫(f*f)·K ≤ M_∞`.  Discharged by `Sidon.MV.mv_eq1`
     unconditionally from `K ≥ 0`, `∫K = 1`, and the a.e. bound
     `f*f ≤ M_∞`.

  2. `mv_eq2` — `∫(f∘f)·K ≤ 1 + √(M_∞-1)·√(K_2-1)`.  Reduced by
     `Sidon.MV.mv_eq2_full` to three atomic primitives:
       (a) **`h_parseval_split`** — Period-`u` Parseval split for
           `∫(f∘f)·K` against `K` on the torus `ℝ/uℤ`.
       (b) **`h_F_bound`** — `∑ |f̂(j/u)|⁴ ≤ M_∞ - 1` (Parseval +
           Hölder on `f*f`).
       (c) **`h_K_bound`** — `∑ K̂(j/u)² ≤ K_2 - 1` (Parseval-at-lattice,
           available as `Sidon.TorusParseval.plancherel_at_lattice_period_u`
           applied to `K` minus its constant term).

  3. `mv_eq3` — `∫((f*f)+(f∘f))·K = 2/u + 2u²·∑ Re(f̃)²·K̃`.  Reduced
     by `Sidon.MV.mv_eq3` to three atomic primitives:
       (a) `h_torus_split` — Period-`u` Parseval split.
       (b) `h_constant_term` — Constant-term mass identity `c₀ = 2/u`.
       (c) `h_tail_form` — Fourier expansion of the tail.

  4. `mv_eq4` — `u²·∑ Re(f̃)²·K̃ ≥ m_G² / ∑ G̃²/K̃`.  Discharged by
     `Sidon.MV.mv_eq4` unconditionally from `K̃(j) > 0`, the
     inner-product floor `u·∑ Re(f̃)·G̃ ≥ m_G`, and `m_G ≥ 0`.

Plus the kernel-specific analytic bounds (discharged externally by the
`flint.arb` certifier in `delsarte_dual/grid_bound_alt_kernel/`):

  5. `K_2(K_ms) ≤ K2UpperQ` (Lemma 4.2) — closed-form `K_2` of the
     3-scale arcsine kernel, evaluated in `arb` interval arithmetic.

  6. `2·m_G²/S_G ≥ gainLowerQ` (Lemma 4.3-4.5) — the cosine `G`'s
     `(min G)² / S_1` ratio, optimised by QP and arb-verified.

The wire-up of (1)-(4) into the master inequality is provided
unconditionally by `Sidon.Master.master_inequality_from_lemmas`; the
slack lift (5)-(6) is provided unconditionally by
`MV_master_via_slack_monotonicity` (above).  What remains:

  * The **periodisation lemma** for `f*f` and `f∘f` on `ℝ/uℤ` (needed
    for `h_parseval_split` of `mv_eq2_full`, and `h_torus_split` of
    `mv_eq3`) — *not* yet in mathlib in a directly usable form (see
    `Sidon.FourierAux`, the "Gap statement for L¹ ∩ L²").
  * The **F-bound** translating `‖f*f‖_{L²}² ≤ ‖f*f‖_{L^∞}` into a
    discrete Parseval sum.
  * The **kernel-specific analytic bounds** (5) and (6), currently
    discharged by the certifier rather than reproved in Lean.

Discharging these in Lean for any concrete `f` would require the
L¹∩L² Plancherel bridge (see `Sidon.FourierAux`'s documentation of
the gap).  In the present formalisation the bundle is taken as a
hypothesis record discharged in the paper by direct citation to MO
2009 Lemmas 2.1, 2.2, 3.2, 3.3 / MV 2010 Lemma 3.1 applied at
`(f, K_ms)`; this is the same citation set MV 2010 itself uses at
the single-arcsine point.  (A Schwartz-class lighter bundle was
attempted in earlier drafts but its `ParsevalSplitSchwartz`
predicate turned out to be vacuously satisfiable by Paley-Wiener +
Carlson, and the variant was retired by the S1+S2 refactor.) -/

/-! ## Extremiser analytic primitives bundle

For the headline theorem we package the four MV Lemma 3.1 atomic
primitives (Eqs.(1)-(4)) as a single bundle `ExtremiserPrimitives f`
indexed by the admissible function `f`.  The bundle records:

  * `LHS1, LHS2, S_cos, m_G, S_G` — the analytic anchors, each **bound**
    (via `m_G_eq, S_G_eq, S_cos_eq, LHS1_eq, LHS2_eq`) to the concrete
    analytic functional at `(f, K_ms)` — not a free real;
  * `R(f) ≥ 1`, `S_G > 0` — positivity;
  * The four MV Lemma 3.1 outputs (Eqs.(1)-(4)), instantiated at the
    **definitional analytic primitive `K2_analytic = ∫ K_ms²`**.

This is an analytic structure on `(f, K_ms)` whose existence is
proved externally by the L¹∩L² Plancherel + period-`u` Parseval
infrastructure of `Sidon.TorusParseval` + `Sidon.FourierAux`.  For
arbitrary admissible `f` (non-Schwartz, only L¹∩L²) the proof
requires bridging mathlib's `Lp 2`-norm Plancherel to the concrete
integral `∫ K̂² = K_2_analytic`, which is **not** yet expressible
as a single mathlib statement.

Crucially, the bundle's `hEq2` field references `K2_analytic`
(the **definition**), not a free real parameter; this forces the
master-inequality wire-up to discharge `K2_analytic ≤ K2UpperQ`
from the **numerical axiom** `K2_analytic_le_K2UpperQ`, putting
that axiom in the headline's dependency closure.

Likewise, the bundle's `hEq4` field is stated against `gain_analytic`
(in the form `m_G²/S_G`), forcing the wire-up to discharge
`gain_analytic ≥ gainLowerQ` from `gain_analytic_ge_gainLowerQ`.

The structure itself, the wire-up theorem
`MV_master_inequality_for_extremiser`, and the headline theorems are
**relocated** to `Sidon.BundleDefs` (see the note below) so the binding
fields can reference the concrete `Sidon.BundleDefs.{S_cos, LHS1, LHS2}`
functionals (which depend on `K_ms` and `autocorr`, hence live
downstream of this module). -/

/-! ## Extremiser bundle + headline — relocated to `Sidon.BundleDefs`

The `ExtremiserPrimitives f` structure, the wire-up theorem
`MV_master_inequality_for_extremiser`, and the headline theorems
(`autoconvolution_ratio_ge_1292_1000` and its aliases) are defined in
`Sidon.BundleDefs` (re-opening `namespace Sidon.MultiScale`), so that the
bundle's binding fields (`S_cos_eq`, `LHS1_eq`, `LHS2_eq`) can reference
the concrete analytic functionals `Sidon.BundleDefs.S_cos`,
`Sidon.BundleDefs.LHS1`, `Sidon.BundleDefs.LHS2`.  Those functionals
depend on `K_ms` (defined here) and `autocorr`, so they live in a module
that imports this one; the bundle must therefore be declared downstream.
The fully-qualified names `Sidon.MultiScale.ExtremiserPrimitives`,
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`, etc. are
preserved by re-opening the namespace in `Sidon.BundleDefs`.

The bundle-independent quadratic-in-`M` inversion
`master_inequality_M_lower` remains in this module (below). -/

/-! ## Quadratic-in-`M` inversion (arithmetic theorem)

The quadratic-in-`M` inversion of the master inequality at these
anchors: for any `a_lower ≥ gainLowerQ` and any `M` with

```
    M + 1 + √(M - 1)·√(K_2 - 1)  ≥  2/u + a_lower,
```

one has `M ≥ 1292/1000`.  At `M = 1292/1000` the LHS is at most

```
    Φ(M) ≤ 66879/20000 = 3.34395,
```

strictly below the threshold `2/u + gainLowerQ = 4267003/1276000`, with
margin `307/3190000 ≈ 9.6 × 10⁻⁵`.  The proof is by case analysis on
`M ≤ 1` (where `Real.sqrt (M-1) = 0` and the LHS is `≤ 2`) versus
`M > 1` (where the rational bound `√((M-1)(K_2-1)) ≤ 105195/100000`
holds for `M ≤ 1292/1000` because `(105195/100000)² > (M-1)(K_2-1)`).
-/

/-- Quadratic-in-`M` inversion of the master inequality at the slack
rationals: any `M` satisfying `M + 1 + √(M-1)·√(K2UpperQ - 1) ≥
2/u + a_lower` with `a_lower ≥ gainLowerQ` is bounded below by
`MTargetQ = 1292/1000`.  Strict-margin closed-form arithmetic
inversion. -/
theorem master_inequality_M_lower :
  ∀ (a_lower : ℝ),
    a_lower ≥ (gainLowerQ : ℝ) →
    ∀ (M : ℝ),
      M + 1 + Real.sqrt (M - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
        ≥ 2 / (uQ : ℝ) + a_lower →
      M ≥ (MTargetQ : ℝ) := by
  intros a_lower h_a M h_MI
  -- Strategy: if `M < MTargetQ` then LHS ≤ 66879/20000 < 4267003/1276000 ≤ RHS.
  by_contra h_lt
  push_neg at h_lt
  have hMT : (MTargetQ : ℝ) = 1292 / 1000 := by unfold MTargetQ; push_cast; ring
  have hK2 : (K2UpperQ : ℝ) - 1 = 37897 / 10000 := by
    unfold K2UpperQ; push_cast; ring
  have h_RHS_ge : 2 / (uQ : ℝ) + a_lower ≥ 4267003 / 1276000 := by
    have h_sum : (2 / (uQ : ℝ)) + (gainLowerQ : ℝ) = 4267003 / 1276000 := by
      unfold uQ gainLowerQ; push_cast; ring
    linarith [h_a]
  -- Bound LHS by 66879/20000.
  have h_LHS_le : M + 1 + Real.sqrt (M - 1) * Real.sqrt ((K2UpperQ : ℝ) - 1)
                  ≤ 66879 / 20000 := by
    rw [hK2]
    by_cases hM1 : M ≤ 1
    · -- `M ≤ 1`: `Real.sqrt (M - 1) = 0` since `M - 1 ≤ 0`.
      have h_sqrt_zero : Real.sqrt (M - 1) = 0 :=
        Real.sqrt_eq_zero'.mpr (by linarith)
      rw [h_sqrt_zero, zero_mul, add_zero]
      have h_lt' : M < 1292 / 1000 := by rw [hMT] at h_lt; exact h_lt
      linarith
    · -- `M > 1`: `√(M-1)·√(K_2-1) = √((M-1)(K_2-1)) ≤ 105195/100000`.
      push_neg at hM1
      have h_M_m1_nn : (0 : ℝ) ≤ M - 1 := by linarith
      rw [← Real.sqrt_mul h_M_m1_nn]
      have h_lt' : M < 1292 / 1000 := by rw [hMT] at h_lt; exact h_lt
      have h_prod_le : (M - 1) * (37897 / 10000) ≤ 11065924 / 10000000 := by
        nlinarith
      have h_pos : (0 : ℝ) ≤ 105195 / 100000 := by norm_num
      have h_sq_ge : (11065924 / 10000000 : ℝ) ≤ (105195 / 100000) ^ 2 := by
        norm_num
      have h_sqrt_le : Real.sqrt ((M - 1) * (37897 / 10000)) ≤ 105195 / 100000 :=
        calc Real.sqrt ((M - 1) * (37897 / 10000))
            ≤ Real.sqrt ((105195 / 100000 : ℝ) ^ 2) := by
              exact Real.sqrt_le_sqrt (le_trans h_prod_le h_sq_ge)
          _ = 105195 / 100000 := Real.sqrt_sq h_pos
      linarith
  -- 66879/20000 < 4267003/1276000.
  have h_strict : (66879 / 20000 : ℝ) < 4267003 / 1276000 := by norm_num
  linarith [h_LHS_le, h_RHS_ge, h_MI, h_strict]

/-! ## Main theorem — relocated to `Sidon.BundleDefs`

The headline `autoconvolution_ratio_ge_1292_1000` (and its aliases
`autoconvolution_ratio_ge_1_292`, `C1a_ge_1292`) is defined in
`Sidon.BundleDefs`, re-opening `namespace Sidon.MultiScale`, so its
`ExtremiserPrimitives f` bundle can bind the analytic functionals
`Sidon.BundleDefs.S_cos`, `LHS1`, `LHS2` (which depend on `K_ms` and
`autocorr`, hence live downstream of this module).  The headline keeps
its fully-qualified name `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`.
Its proof uses `master_inequality_M_lower` (above) and the relocated
`MV_master_inequality_for_extremiser`. -/

end Sidon.MultiScale
