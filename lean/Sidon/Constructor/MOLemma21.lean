/-
Sidon Autocorrelation Project — MO 2009 Lemma 2.1 (period-`u` Parseval primitive).
==================================================================================

This file attacks the deepest remaining analytic primitive of the
constructor layer: the period-`u` Parseval pairing (MO 2009 Lemma 2.1)
that feeds the three residual hypotheses of
`Sidon.MultiScale.C1a_ge_1292_unconditional`:

  * **(P3)** `h_torus_eq_full` of `field_hEq3_ge` :
      `LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f`.
  * **(P2)** `h_parseval_split` of `field_hEq2_canonical` :
      `∫ autocorr f · K_ms = 1 + ∑_{j∈J} (Re 𝓕f(j/u))²·K̂_ms(j)`.
  * **(P4)** `h_cs` of `field_hEq4` :
      `S_cos f ≥ 2·min_G²/S_1` (Cauchy–Schwarz floor).

================================================================================
## The mathematical core (MO 2009 Lemma 2.1, period-`u`)
================================================================================

For `g, K : ℝ → ℝ` with `supp g ⊆ (-1/2, 1/2)`, `supp K ⊆ (-δ₁, δ₁)`,
`δ₁ = u - 1/2` (so `1/2 + δ₁ = u`, the boundary non-overlap):

  `∫_ℝ g(x)·K(x) dx = (1/u)·∑_{j∈ℤ} 𝓕g(j/u)·conj(𝓕K(j/u))`.

The subtle point — and the whole difficulty — is that `g = (f*f) + autocorr f`
**overflows** the period window `(-u/2, u/2) = (-0.319, 0.319)` (its support
reaches `(-1/2, 1/2)`).  So the existing bilinear period-`u` Parseval
`Sidon.BilinearParseval.bilinear_parseval_period_u_inv`, which requires *both*
factors to fit one period, does **not** apply directly.

The correct MO route pairs through the *period-fitting* kernel `K_ms`
(`supp K_ms ⊆ (-δ₁, δ₁) ⊆ (-u/2, u/2)`):

  1. `∫_ℝ g·K_ms = ∫_{period} g·K_ms`  (`K_ms = 0` outside `(-δ₁,δ₁)`);
  2. expand `K_ms` in its period-`u` `L²` Fourier series on `AddCircle u`
     (`K_ms` *does* fit one period), pair against `g` in `L²(period)` via
     the orthonormal-basis bilinear Parseval
     `Sidon.TorusParseval.bilinear_parseval_addCircle_Lp_tsum`;
  3. identify `fourierCoeff(lift K_ms) j = (1/u)·𝓕K_ms(j/u)`
     (`Sidon.BilinearParseval.fourierCoeff_toLp_eq_period_u`, `K_ms` fits);
  4. identify `fourierCoeff(lift g_per) j = (1/u)·𝓕g(j/u)` — the **period-`u`
     Poisson sampling** of the periodisation `g_per(x) := ∑ₖ g(x+ku)`,
     valid because on `supp K_ms ⊆ (-δ₁,δ₁)` the periodisation agrees with
     `g` (the `k≠0` translates `g(·+ku)`, supported in `(-1/2−ku, 1/2−ku)`,
     do not meet `(-δ₁,δ₁)` precisely because `1/2 + δ₁ = u`).

Step 4 is the genuine period-`u` Poisson sampling identity (a period-`u`
Poisson sampling lemma for an overflowing-but-periodisable, *non-continuous*
`g`); mathlib's `Real.fourierCoeff_tsum_comp_add` is hardcoded to period 1
and requires a *continuous* base with a uniform summability bound, neither
of which holds for our `L¹∩L²` `g`.  It is **proved** (for the periodisation
`g_per`, hypothesis-free) in `Sidon.Constructor.PoissonSampling`
(`period_u_poisson_sampling`), which also assembles the hypothesis-free MO
2.1 core and the exact P3 identity / `PeriodUParsevalTrue` consumed by the
bundle (`mo_lemma_21_core_via_periodization`,
`field_P3_periodUParsevalTrue_via_periodization`).

NOTE.  The naive *truncation*-lift form of Step 4 — equating the Fourier
coefficient of `liftIoc u (-(u/2)) g` (which truncates `g` to one period)
with `(1/u)·𝓕g(j/u)` — is FALSE for our overflowing `g`, and is **not**
present in this file.  Only the periodisation route (in `PoissonSampling`)
is sound; see the note there.

================================================================================
## What is proved here (no `sorry`, no new axioms)
================================================================================

ALL of the following are FULLY PROVED (axioms: only `propext`,
`Classical.choice`, `Quot.sound` — the Lean core trio).  They are the
support lemmas reused by the hypothesis-free periodisation route in
`Sidon.Constructor.PoissonSampling`:

  * `g`, `g_complex` definitions and the support / integrability /
    `MemLp` facts for both `g_ℂ` and `K_ms_ℂ` over one period; in
    particular **`g_complex_memLp_period`** (`g ∈ L²(period)`, proved
    from the everywhere bound `‖g x‖ ≤ 2∫f²` on the finite interval).
  * `inner_g_Kms_eq` : the `Lp²(AddCircle u)` inner product of the lifts
    equals `(1/u)·∫_ℝ g·K_ms`  (uses `K_ms = 0` off the period).
  * `fourierCoeff_Kms_eq` : `fourierCoeff(lift K_ms) j = (1/u)·𝓕K_ms(j/u)`.
  * the `Re`/`±j` collapse (`re_summand`), realness of `𝓕K_ms`
    (`conj_fourierIntegral_K_ms`), the `j = 0` term value
    (`abstract_zero_term`), `LHS1_add_LHS2_eq_integral`,
    `integral_g_Kms_complex_eq`, and the abstract sum `S_cos_abstract`.
  * **the Fourier–Bessel identity** `bessel_identity`
    (`(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`), built from
    `Sidon.Bessel.besselJ0_arcsine_fourier` + the autoconvolution Fourier
    theorem + additivity over the three scales (a ~250-line development
    not present in mathlib), and the bridge `S_cos_abstract_eq_S_cos`.

No `sorry`, no new axioms in this file.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.ConvFourier
import Sidon.Constructor.PeriodUParseval
import Sidon.BilinearParseval
import Sidon.BundleEq2Schwartz
import Sidon.BundleEq3Schwartz
import Sidon.Bessel
import Sidon.Constructor.KernelL2
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical Pointwise

namespace Sidon.Constructor.MOLemma21

open Sidon.MultiScale
open Sidon.FourierAux (autocorr autocorr_def)
open Sidon.Constructor.PeriodU (g)

noncomputable section

/-! ## Period and support preliminaries -/

/-- `δ₁ < u/2`: the kernel half-width is strictly inside the half-period.
(`δ₁ = 0.138`, `u/2 = 0.319`.) -/
theorem delta1_lt_u_half : delta1 < uQ_real / 2 :=
  Sidon.BundleEq2Schwartz.delta1_lt_u_half

/-- `K_ms x = 0` whenever `|x| ≥ δ₁`. -/
theorem K_ms_zero_outside (x : ℝ) (h : delta1 ≤ |x|) : K_ms x = 0 :=
  Sidon.BundleEq2Schwartz.K_ms_eq_zero_outside x h

/-- Complex lift of `K_ms` is supported in `Ioc (-(u/2)) (u/2)`. -/
theorem K_ms_complex_support :
    Function.support (fun x => ((K_ms x : ℝ) : ℂ))
      ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) :=
  Sidon.BundleEq2Schwartz.K_ms_complex_support

/-! ## Integrability of `g = (f*f) + autocorr f` (real and complex) -/

/-- `autocorr f` is integrable from `Integrable f` (`autocorr f = f̌ ⋆ f`). -/
theorem autocorr_integrable_local (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (autocorr f) volume := by
  have hrefl : Integrable (fun y => f (-y)) volume := by
    have := hf.comp_neg; simpa [Function.comp] using this
  have hconv : Integrable
      (MeasureTheory.convolution (fun y => f (-y)) f
        (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
    hrefl.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf
  have h_eq : autocorr f = MeasureTheory.convolution (fun y => f (-y)) f
      (ContinuousLinearMap.mul ℝ ℝ) volume := by
    funext x
    rw [autocorr_def]
    simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
    rw [← MeasureTheory.integral_neg_eq_self (fun t => f (-t) * f (x - t)) volume]
    refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
    simp only [neg_neg]
    have : x - -s = x + s := by ring
    rw [this]
  rw [h_eq]; exact hconv

/-- `Integrable (g f)`: sum of two `L¹` convolutions. -/
theorem g_integrable (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (g f) volume := by
  unfold g
  -- `f*f` is integrable.
  have hconv : Integrable
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
    hf.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf
  -- `autocorr f` is integrable.
  have hauto : Integrable (autocorr f) volume := autocorr_integrable_local f hf
  exact hconv.add hauto

/-- The complex lift `g_ℂ x := (g f x : ℂ)`. -/
def g_complex (f : ℝ → ℝ) : ℝ → ℂ := fun x => ((g f x : ℝ) : ℂ)

/-- `Integrable (g_complex f)`. -/
theorem g_complex_integrable (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (g_complex f) volume :=
  (g_integrable f hf).ofReal

/-! ### `g ∈ L²` on one period (fully proved)

`g = (f*f) + autocorr f` is everywhere bounded by `2·∫f²` for `f ∈ L²`
(Cauchy–Schwarz: both `|f*f x|` and `|autocorr f x|` are `≤ ∫f²`).  A
bounded a.e.-strongly-measurable function on the finite-measure interval
`Ioc (-(u/2)) (u/2)` is in `L²`. -/

/-- **Pointwise Cauchy–Schwarz bound on `f*f`.**  `|(f*f) x| ≤ ∫f²`. -/
theorem convff_abs_le (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume) (x : ℝ) :
    |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x|
      ≤ ∫ x, f x ^ 2 ∂volume := by
  -- The reflected translate `t ↦ f (x - t)` is in `L²`.
  have hgx : MemLp (fun t => f (x - t)) 2 volume := by
    have h1 : MemLp (fun t => f (x + t)) 2 volume := by
      have := hf_L2.comp_measurePreserving
        (measurePreserving_add_left (volume : Measure ℝ) x)
      simpa [Function.comp] using this
    have := h1.comp_measurePreserving (Measure.measurePreserving_neg (volume : Measure ℝ))
    simpa [Function.comp, sub_eq_add_neg] using this
  have h2 : (ENNReal.ofReal (2:ℝ)) = (2 : ENNReal) := by norm_num
  have hfabs : MemLp (fun t => |f t|) (ENNReal.ofReal (2:ℝ)) volume := by
    rw [h2]; exact hf_L2.abs
  have hgabs : MemLp (fun t => |f (x - t)|) (ENNReal.ofReal (2:ℝ)) volume := by
    rw [h2]; exact hgx.abs
  -- `|(f*f) x| ≤ ∫ |f t|·|f (x-t)|`.
  have h_abs_le :
      |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x|
        ≤ ∫ t, |f t| * |f (x - t)| ∂volume := by
    rw [MeasureTheory.convolution_def]
    calc |∫ t, (ContinuousLinearMap.mul ℝ ℝ) (f t) (f (x - t)) ∂volume|
        ≤ ∫ t, |(ContinuousLinearMap.mul ℝ ℝ) (f t) (f (x - t))| ∂volume :=
          abs_integral_le_integral_abs
      _ = ∫ t, |f t| * |f (x - t)| ∂volume := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
          simp only [ContinuousLinearMap.mul_apply', abs_mul]
  -- Cauchy–Schwarz.
  have hcs := MeasureTheory.integral_mul_le_Lp_mul_Lq_of_nonneg
    (μ := volume) (p := (2:ℝ)) (q := (2:ℝ))
    (Real.HolderConjugate.two_two)
    (f := fun t => |f t|) (g := fun t => |f (x - t)|)
    (Filter.Eventually.of_forall fun t => abs_nonneg _)
    (Filter.Eventually.of_forall fun t => abs_nonneg _)
    hfabs hgabs
  simp only [] at hcs
  have hsq1 : ∫ t, (|f t|) ^ (2:ℝ) ∂volume = ∫ t, f t ^ 2 ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
    show (|f t|) ^ (2:ℝ) = f t ^ 2
    rw [Real.rpow_two, sq_abs, sq]
  have hsq2 : ∫ t, (|f (x - t)|) ^ (2:ℝ) ∂volume = ∫ t, f t ^ 2 ∂volume := by
    have htrans : ∫ t, (|f (x - t)|) ^ (2:ℝ) ∂volume
                    = ∫ t, (|f t|) ^ (2:ℝ) ∂volume := by
      -- substitute `t ↦ x - t` (measure-preserving reflection + translation).
      have := MeasureTheory.integral_sub_left_eq_self
        (fun t => (|f t|) ^ (2:ℝ)) volume x
      simpa using this
    rw [htrans, hsq1]
  rw [hsq1, hsq2] at hcs
  have hI_nonneg : (0 : ℝ) ≤ ∫ t, f t ^ 2 ∂volume :=
    integral_nonneg (fun t => sq_nonneg _)
  have hprod :
      (∫ t, f t ^ 2 ∂volume) ^ ((1:ℝ) / 2) * (∫ t, f t ^ 2 ∂volume) ^ ((1:ℝ) / 2)
        = ∫ t, f t ^ 2 ∂volume := by
    rw [← Real.rpow_add' hI_nonneg (by norm_num)]; norm_num
  rw [hprod] at hcs
  exact le_trans h_abs_le hcs

/-- **Pointwise Cauchy–Schwarz bound on `autocorr f`.**  `|autocorr f x| ≤ ∫f²`
(local copy of `Sidon.Constructor.Glue.autocorr_abs_le`, avoiding the
heavy `Glue` import). -/
theorem autocorr_abs_le_local (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume) (x : ℝ) :
    |autocorr f x| ≤ ∫ x, f x ^ 2 ∂volume := by
  have hgx : MemLp (fun t => f (x + t)) 2 volume := by
    have := hf_L2.comp_measurePreserving
      (measurePreserving_add_left (volume : Measure ℝ) x)
    simpa [Function.comp] using this
  have h2 : (ENNReal.ofReal (2:ℝ)) = (2 : ENNReal) := by norm_num
  have hfabs : MemLp (fun t => |f t|) (ENNReal.ofReal (2:ℝ)) volume := by
    rw [h2]; exact hf_L2.abs
  have hgabs : MemLp (fun t => |f (x + t)|) (ENNReal.ofReal (2:ℝ)) volume := by
    rw [h2]; exact hgx.abs
  have h_abs_le :
      |autocorr f x| ≤ ∫ t, |f t| * |f (x + t)| ∂volume := by
    rw [autocorr_def]
    calc |∫ t, f t * f (x + t) ∂volume|
        ≤ ∫ t, |f t * f (x + t)| ∂volume := abs_integral_le_integral_abs
      _ = ∫ t, |f t| * |f (x + t)| ∂volume := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
          simp only [abs_mul]
  have hcs := MeasureTheory.integral_mul_le_Lp_mul_Lq_of_nonneg
    (μ := volume) (p := (2:ℝ)) (q := (2:ℝ))
    (Real.HolderConjugate.two_two)
    (f := fun t => |f t|) (g := fun t => |f (x + t)|)
    (Filter.Eventually.of_forall fun t => abs_nonneg _)
    (Filter.Eventually.of_forall fun t => abs_nonneg _)
    hfabs hgabs
  simp only [] at hcs
  have hsq1 : ∫ t, (|f t|) ^ (2:ℝ) ∂volume = ∫ t, f t ^ 2 ∂volume := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
    show (|f t|) ^ (2:ℝ) = f t ^ 2
    rw [Real.rpow_two, sq_abs, sq]
  have hsq2 : ∫ t, (|f (x + t)|) ^ (2:ℝ) ∂volume = ∫ t, f t ^ 2 ∂volume := by
    have htrans : ∫ t, (|f (x + t)|) ^ (2:ℝ) ∂volume
                    = ∫ t, (|f t|) ^ (2:ℝ) ∂volume := by
      have hcomm : (fun t => (|f (x + t)|) ^ (2:ℝ))
                    = (fun t => (fun y => (|f y|) ^ (2:ℝ)) (t + x)) := by
        funext t; rw [add_comm]
      rw [hcomm, MeasureTheory.integral_add_right_eq_self
        (fun y => (|f y|) ^ (2:ℝ)) x]
    rw [htrans, hsq1]
  rw [hsq1, hsq2] at hcs
  have hI_nonneg : (0 : ℝ) ≤ ∫ t, f t ^ 2 ∂volume :=
    integral_nonneg (fun t => sq_nonneg _)
  have hprod :
      (∫ t, f t ^ 2 ∂volume) ^ ((1:ℝ) / 2) * (∫ t, f t ^ 2 ∂volume) ^ ((1:ℝ) / 2)
        = ∫ t, f t ^ 2 ∂volume := by
    rw [← Real.rpow_add' hI_nonneg (by norm_num)]; norm_num
  rw [hprod] at hcs
  exact le_trans h_abs_le hcs

/-- `g_complex f` is a.e.-strongly-measurable. -/
theorem g_complex_aestronglyMeasurable (f : ℝ → ℝ) (hf : Integrable f volume) :
    AEStronglyMeasurable (g_complex f) volume :=
  (g_complex_integrable f hf).aestronglyMeasurable

/-- **`g ∈ L²` on one period.**  `MemLp (g_complex f) 2 (restrict (Ioc (-(u/2)) (u/2)))`,
proved from the everywhere bound `‖g_complex f x‖ ≤ 2·∫f²` and the finite
measure of the interval. -/
theorem g_complex_memLp_period (f : ℝ → ℝ) (hf : Integrable f volume)
    (hf_L2 : MemLp f 2 volume) :
    MemLp (g_complex f) 2 (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
  -- Finite measure on the bounded interval.
  haveI : IsFiniteMeasure (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
    refine ⟨?_⟩
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  -- a.e.-strong-measurability on the restriction.
  have hmeas : AEStronglyMeasurable (g_complex f)
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    (g_complex_aestronglyMeasurable f hf).restrict
  -- Everywhere bound `‖g_complex f x‖ ≤ 2·∫f²`.
  set C : ℝ := 2 * ∫ x, f x ^ 2 ∂volume with hC_def
  have h_bound : ∀ᵐ x ∂(volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))),
      ‖g_complex f x‖ ≤ C := by
    refine Filter.Eventually.of_forall fun x => ?_
    -- `‖g_complex f x‖ = |g f x| ≤ |f*f x| + |autocorr f x| ≤ ∫f² + ∫f² = C`.
    unfold g_complex g
    rw [Complex.norm_real]
    rw [Real.norm_eq_abs]
    calc |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
            + autocorr f x|
        ≤ |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x|
            + |autocorr f x| := abs_add_le _ _
      _ ≤ (∫ x, f x ^ 2 ∂volume) + (∫ x, f x ^ 2 ∂volume) :=
          add_le_add (convff_abs_le f hf_L2 x)
            (autocorr_abs_le_local f hf_L2 x)
      _ = C := by rw [hC_def]; ring
  exact MemLp.of_bound hmeas C h_bound

/-- The complex lift of `K_ms`. -/
def K_ms_complex : ℝ → ℂ := fun x => ((K_ms x : ℝ) : ℂ)

/-- `K_ms` is integrable (finite sum of integrable `λᵢ·K_arc(δᵢ)`). -/
theorem K_ms_integrable_local : Integrable K_ms volume := by
  unfold K_ms
  have hδ := delta_positivities
  have h1 : Integrable (fun x => lambda1 * K_arc delta1 x) volume :=
    (K_arc_integrable _ hδ.1).const_mul _
  have h2 : Integrable (fun x => lambda2 * K_arc delta2 x) volume :=
    (K_arc_integrable _ hδ.2.1).const_mul _
  have h3 : Integrable (fun x => lambda3 * K_arc delta3 x) volume :=
    (K_arc_integrable _ hδ.2.2).const_mul _
  exact (h1.add h2).add h3

/-- `∫ K_ms = 1` (convex combination of unit-mass arcsine kernels). -/
theorem K_ms_integral_eq_one_local : ∫ x, K_ms x ∂volume = 1 := by
  unfold K_ms
  have hδ := delta_positivities
  have h1 : Integrable (fun x => lambda1 * K_arc delta1 x) volume :=
    (K_arc_integrable _ hδ.1).const_mul _
  have h2 : Integrable (fun x => lambda2 * K_arc delta2 x) volume :=
    (K_arc_integrable _ hδ.2.1).const_mul _
  have h3 : Integrable (fun x => lambda3 * K_arc delta3 x) volume :=
    (K_arc_integrable _ hδ.2.2).const_mul _
  have step_a :
      ∫ x, (lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x)
              + lambda3 * K_arc delta3 x ∂volume
        = (∫ x, lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x ∂volume)
            + ∫ x, lambda3 * K_arc delta3 x ∂volume :=
    integral_add (h1.add h2) h3
  have step_b :
      ∫ x, lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x ∂volume
        = (∫ x, lambda1 * K_arc delta1 x ∂volume)
            + ∫ x, lambda2 * K_arc delta2 x ∂volume :=
    integral_add h1 h2
  rw [step_a, step_b, integral_const_mul, integral_const_mul, integral_const_mul,
      K_arc_integral_eq_one _ hδ.1, K_arc_integral_eq_one _ hδ.2.1,
      K_arc_integral_eq_one _ hδ.2.2]
  have hsum := lambdas_sum_one
  have hsum_real : lambda1 + lambda2 + lambda3 = 1 := by
    unfold lambda1 lambda2 lambda3
    have := congrArg (fun q : ℚ => (q : ℝ)) hsum
    push_cast at this
    linarith
  linarith [hsum_real]

/-- `Integrable K_ms_complex`. -/
theorem K_ms_complex_integrable : Integrable K_ms_complex volume :=
  (K_ms_integrable_local).ofReal

/-- `𝓕(g_complex f)(ξ) = (𝓕f ξ)² + 𝓕f ξ · conj(𝓕f ξ)` — re-export of
`Sidon.Constructor.PeriodU.fourierIntegral_g`. -/
theorem fourierIntegral_g_complex (f : ℝ → ℝ) (hf : Integrable f volume)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (ξ : ℝ) :
    Real.fourierIntegral (g_complex f) ξ
      = (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) ^ 2
        + Real.fourierIntegral (fun x => (f x : ℂ)) ξ
            * starRingEnd ℂ (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) :=
  Sidon.Constructor.PeriodU.fourierIntegral_g f hf h_joint ξ

/-! ## `MemLp` of `K_ms_complex` on one period

`K_ms ∈ L²(ℝ)` (`Sidon.Constructor.KernelL2.K_ms_memLp_two`), so its
complex lift is in `L²(ℝ)`, and the restriction to any subset is in `L²`. -/

/-- `K_ms_complex ∈ L²(ℝ)`. -/
theorem K_ms_complex_memLp_volume : MemLp K_ms_complex 2 volume := by
  have h := Sidon.Constructor.K_ms_memLp_two
  -- `MemLp K_ms 2` for real `K_ms`; coerce to ℂ.
  have h2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2 volume := h.ofReal (K := ℂ)
  simpa [K_ms_complex] using h2

/-- `K_ms_complex ∈ L²(volume.restrict (Ioc (-(u/2)) (u/2)))`. -/
theorem K_ms_complex_memLp_restrict :
    MemLp K_ms_complex 2 (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
  K_ms_complex_memLp_volume.restrict _

/-! ## The period-`u` Fourier coefficient of `K_ms` (fully proved)

`K_ms` fits one period (`supp ⊆ (-δ₁,δ₁) ⊆ (-u/2,u/2)`), so its
period-`u` Fourier coefficient samples the full-line transform. -/

/-- `fourierCoeff (lift K_ms_complex) j = (1/u)·𝓕K_ms_complex(j/u)`.

This is the period-fitting case, discharged by
`Sidon.BilinearParseval.fourierCoeff_toLp_eq_period_u`. -/
theorem fourierCoeff_Kms_eq (j : ℤ) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    fourierCoeff
        ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
            K_ms_complex K_ms_complex_memLp_restrict).toLp
          (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex) : AddCircle uQ_real → ℂ) j
      = (1 / uQ_real : ℂ) * Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  -- Support of `K_ms_complex` is in `Ioc (-(u/2)) (u/2)`.
  have hsupp : Function.support K_ms_complex ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) :=
    K_ms_complex_support
  exact Sidon.BilinearParseval.fourierCoeff_toLp_eq_period_u uQ_real uQ_real_pos
    K_ms_complex hsupp K_ms_complex_memLp_restrict j

/-! ## The `Lp²(AddCircle u)` inner product equals `(1/u)·∫_ℝ g·K_ms`

The `Lp²` inner product of the lifts of `g` and `K_ms` equals
`(1/u)·∫_{period} g·conj(K_ms)`, and since `K_ms` is real and vanishes
off the period this is `(1/u)·∫_ℝ g·K_ms`. -/

/-- The `conj` of `K_ms_complex` is `K_ms_complex` (real-valued). -/
theorem conj_K_ms_complex (x : ℝ) :
    starRingEnd ℂ (K_ms_complex x) = K_ms_complex x := by
  unfold K_ms_complex
  exact Complex.conj_ofReal _

/-- The interval integral over one period of `g_complex · K_ms_complex`
equals the full-line integral, since `K_ms_complex = 0` off the period. -/
theorem intervalIntegral_g_Kms_eq_full (f : ℝ → ℝ) :
    (∫ x in (-(uQ_real/2))..(uQ_real/2),
        g_complex f x * starRingEnd ℂ (K_ms_complex x))
      = ∫ x, g_complex f x * starRingEnd ℂ (K_ms_complex x) ∂volume := by
  rw [intervalIntegral.integral_of_le (by linarith [uQ_real_pos] : -(uQ_real/2) ≤ uQ_real/2)]
  refine MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ?_
  intro x hx
  -- For `x ∉ Ioc (-(u/2)) (u/2)`: `K_ms_complex x = 0`, so the product is `0`.
  have hK0 : K_ms_complex x = 0 := by
    -- `x ∉ Ioc` ⟹ `x ∉ supp K_ms_complex` ⟹ `K_ms_complex x = 0`.
    by_contra h_ne
    exact hx (K_ms_complex_support h_ne)
  rw [hK0, map_zero, mul_zero]

/-- **The inner-product = integral identity.**

For `g_complex f ∈ L²(period)`,
`⟪lift K_ms, lift g⟫ = (1/u)·∫_ℝ g·K_ms` (where on the RHS `K_ms` is
multiplied without conjugation since it is real). -/
theorem inner_g_Kms_eq (f : ℝ → ℝ)
    (hg_L2_period : MemLp (g_complex f) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    @inner ℂ _ _
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          K_ms_complex K_ms_complex_memLp_restrict).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex))
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          (g_complex f) hg_L2_period).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (g_complex f)))
      = (uQ_real⁻¹ : ℂ) * ∫ x, g_complex f x * K_ms_complex x ∂volume := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  -- `inner_toLp_eq_intervalIntegral` with g := g_complex f, h := K_ms_complex.
  rw [Sidon.BilinearParseval.inner_toLp_eq_intervalIntegral uQ_real uQ_real_pos
    (g_complex f) K_ms_complex hg_L2_period K_ms_complex_memLp_restrict]
  -- Now: `u⁻¹ · ∫_{period} g·conj(K_ms) = u⁻¹ · ∫_ℝ g·K_ms`.
  rw [intervalIntegral_g_Kms_eq_full f]
  congr 1
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  simp only [conj_K_ms_complex]

/-! ## Note: the period-`u` Poisson sampling primitive (lives in `PoissonSampling`)

For the period-fitting `K_ms` we proved `fourierCoeff (lift K_ms) j =
(1/u)·𝓕K_ms(j/u)` outright (`fourierCoeff_Kms_eq`).  For `g = (f*f) +
autocorr f`, whose support `(-1/2,1/2)` **overflows** the period
`(-u/2,u/2)`, the *truncation* lift `liftIoc u (-(u/2)) g` does **not**
sample `𝓕g(j/u)` (its coefficient is `(1/u)·𝓕(g·1_{period})(j/u)`).  The
correct MO 2009 Lemma 2.1 route uses the **periodisation**
`g_per(x) := ∑ₖ g(x+ku)`, whose period-`u` Fourier coefficient *does*
sample `𝓕g` at the lattice, and which agrees with `g` on
`supp K_ms ⊆ (-δ₁,δ₁)` (the non-overlap `1/2 + δ₁ = u`).

That periodisation route — the *true* period-`u` Poisson sampling
identity and the hypothesis-free MO 2.1 core / `PeriodUParsevalTrue`
that the bundle consumes — is fully proved (no Poisson hypothesis) in
`Sidon.Constructor.PoissonSampling`
(`period_u_poisson_sampling`,
`mo_lemma_21_core_via_periodization`,
`field_P3_periodUParsevalTrue_via_periodization`).  The supporting
lemmas in *this* file (`fourierCoeff_Kms_eq`, `inner_g_Kms_eq`,
`re_summand`, `bessel_identity`, `S_cos_abstract_eq_S_cos`,
`abstract_zero_term`, `LHS1_add_LHS2_eq_integral`,
`integral_g_Kms_complex_eq`, …) are reused there. -/

/-! ## Realness of `𝓕K_ms` (even real kernel)

`K_ms` is real-valued and even (`Sidon.BundleEq3Schwartz.K_ms_even`), so
`𝓕K_ms(-ξ) = conj(𝓕K_ms(ξ))` (conjugate symmetry) **and** `𝓕K_ms(-ξ) =
𝓕K_ms(ξ)` (evenness), forcing `conj(𝓕K_ms(ξ)) = 𝓕K_ms(ξ)`, i.e.
`𝓕K_ms(ξ)` is real. -/

/-- `𝓕K_ms` is invariant under negation of the frequency: `𝓕K_ms(-ξ) =
𝓕K_ms(ξ)` (because `K_ms` is even). -/
theorem fourierIntegral_K_ms_even (ξ : ℝ) :
    Real.fourierIntegral K_ms_complex (-ξ) = Real.fourierIntegral K_ms_complex ξ := by
  unfold K_ms_complex
  -- `𝓕(K_ms)(-ξ) = ∫ 𝐞(-(v·-ξ))·K_ms(v) = ∫ 𝐞(-(-v·ξ))·K_ms(v)`; substitute `v ↦ -v`
  -- and use `K_ms(-v) = K_ms(v)`.
  show FourierTransform.fourier (fun x => ((K_ms x : ℝ) : ℂ)) (-ξ)
      = FourierTransform.fourier (fun x => ((K_ms x : ℝ) : ℂ)) ξ
  rw [Real.fourier_real_eq, Real.fourier_real_eq]
  rw [show (∫ v, (𝐞 (-(v * ξ)) : Circle) • ((K_ms v : ℝ) : ℂ))
        = ∫ v, (𝐞 (-((-v) * ξ)) : Circle) • ((K_ms (-v) : ℝ) : ℂ) from
      (MeasureTheory.integral_neg_eq_self
        (fun v => (𝐞 (-(v * ξ)) : Circle) • ((K_ms v : ℝ) : ℂ)) volume).symm]
  refine integral_congr_ae (Filter.Eventually.of_forall fun v => ?_)
  simp only [Sidon.BundleEq3Schwartz.K_ms_even]
  congr 2
  ring

/-- `𝓕K_ms(ξ)` is real-valued: `conj(𝓕K_ms(ξ)) = 𝓕K_ms(ξ)`. -/
theorem conj_fourierIntegral_K_ms (ξ : ℝ) :
    starRingEnd ℂ (Real.fourierIntegral K_ms_complex ξ)
      = Real.fourierIntegral K_ms_complex ξ := by
  -- `conj(𝓕K_ms(ξ)) = 𝓕K_ms(-ξ)` (conj symmetry) `= 𝓕K_ms(ξ)` (evenness).
  have h_conj :
      Real.fourierIntegral K_ms_complex (-ξ)
        = starRingEnd ℂ (Real.fourierIntegral K_ms_complex ξ) := by
    have := Sidon.FourierAux.fourierIntegral_real_conj K_ms
      (by simpa [K_ms_complex] using K_ms_complex_integrable) ξ
    simpa [K_ms_complex] using this
  rw [← h_conj, fourierIntegral_K_ms_even]

/-- `𝓕K_ms(ξ) = ((𝓕K_ms(ξ)).re : ℂ)` (the imaginary part vanishes). -/
theorem fourierIntegral_K_ms_eq_re (ξ : ℝ) :
    Real.fourierIntegral K_ms_complex ξ
      = (((Real.fourierIntegral K_ms_complex ξ).re : ℝ) : ℂ) := by
  have h := conj_fourierIntegral_K_ms ξ
  -- `conj z = z ↔ z.im = 0`.
  rw [Complex.ext_iff] at h ⊢
  refine ⟨rfl, ?_⟩
  -- `h.2 : (conj z).im = z.im`, i.e. `-z.im = z.im`, so `z.im = 0`.
  have him := h.2
  simp only [Complex.conj_im] at him
  simp only [Complex.ofReal_im]
  linarith

/-! ## Real-part reduction of the per-frequency summand

Each summand `𝓕g(j/u)·conj(𝓕K_ms(j/u))` has real part
`2·(Re 𝓕f(j/u))²·(𝓕K_ms(j/u)).re`.  This is the `Re`/`±j` collapse: since
`𝓕g = (𝓕f)² + 𝓕f·conj(𝓕f)` and `𝓕K_ms(j/u)` is real, the imaginary part
of the product does not contribute to the (real) integral. -/

/-- The real part of the period-`u` summand `𝓕g(j/u)·conj(𝓕K_ms(j/u))`
collapses to `2·(Re 𝓕f(j/u))²·(𝓕K_ms(j/u)).re`. -/
theorem re_summand (f : ℝ → ℝ) (hf : Integrable f volume)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (j : ℤ) :
    (Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))).re
      = 2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
          * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re := by
  -- Replace `conj(𝓕K_ms)` by the real cast `((𝓕K_ms).re : ℂ)`.
  rw [conj_fourierIntegral_K_ms, fourierIntegral_K_ms_eq_re (j / uQ_real : ℝ)]
  -- Expand `𝓕g = (𝓕f)² + 𝓕f·conj(𝓕f)`.
  rw [fourierIntegral_g_complex f hf h_joint (j / uQ_real : ℝ)]
  -- Apply the `re_mul_real` collapse from `PeriodUParseval`.
  exact Sidon.Constructor.PeriodU.re_mul_real
    (Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ))
    ((Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re)

/-! ## Supporting identities for the abstract period-`u` Parseval (real form)

`LHS1 f + LHS2 f = ∫_ℝ g·K_ms` and the real-cast bridge
`∫ g_complex·K_ms_complex = ((∫ g·K_ms : ℝ) : ℂ)`, reused by the
hypothesis-free periodisation route in `Sidon.Constructor.PoissonSampling`
(`mo_lemma_21_real_via_periodization`).  The abstract sum is kept over
`(𝓕K_ms).re`; the bridge to the project's `S_cos` is the Bessel FT
identity `bessel_identity` (proved below). -/

/-- The combined integral `LHS1 f + LHS2 f` equals `∫_ℝ g·K_ms` (real). -/
theorem LHS1_add_LHS2_eq_integral (f : ℝ → ℝ)
    (hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume)
    (hAuto_prod_int : Integrable (fun x => autocorr f x * K_ms x) volume) :
    LHS1 f + LHS2 f = ∫ x, g f x * K_ms x ∂volume := by
  unfold LHS1 LHS2 g
  rw [show (fun x => ((MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) volume) x + autocorr f x) * K_ms x)
        = (fun x => (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x
          + autocorr f x * K_ms x) from by funext x; ring]
  rw [integral_add hConv_prod_int hAuto_prod_int]

/-- The complex integral `∫ g_complex·K_ms_complex` is the real cast of
`∫ g·K_ms`. -/
theorem integral_g_Kms_complex_eq (f : ℝ → ℝ) :
    (∫ x, g_complex f x * K_ms_complex x ∂volume)
      = ((∫ x, g f x * K_ms x ∂volume : ℝ) : ℂ) := by
  rw [show ((∫ x, g f x * K_ms x ∂volume : ℝ) : ℂ)
        = ∫ x, ((g f x * K_ms x : ℝ) : ℂ) ∂volume from
      (integral_ofReal (f := fun x => g f x * K_ms x)).symm]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  unfold g_complex K_ms_complex
  push_cast
  ring

/-! ## The abstract `S_cos` and the abstract-form P3

We isolate the `j = 0` term and rewrite the tail in the project's
`if j = 0` convention, obtaining the abstract analogue of P3 with the
*abstract* coefficient `(𝓕K_ms(j/u)).re` in place of the Bessel
`K_ms_fourier_lattice j`. -/

/-- The abstract bilinear cosine sum, with the *abstract* FT coefficient
`(𝓕K_ms(j/u)).re` in place of the Bessel `K_ms_fourier_lattice j`. -/
def S_cos_abstract (f : ℝ → ℝ) : ℝ :=
  ∑' j : ℤ, if j = 0 then 0 else
    ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
      * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re

/-- The `j = 0` term of the abstract sum is `2`: `2·(∫f)²·(∫K_ms) = 2·1·1`. -/
theorem abstract_zero_term (f : ℝ → ℝ) (hf_one : ∫ x, f x ∂volume = 1) :
    2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) ((0 : ℤ) / uQ_real : ℝ)).re) ^ 2
        * (Real.fourierIntegral K_ms_complex ((0 : ℤ) / uQ_real : ℝ)).re
      = 2 := by
  -- `(0 : ℤ)/u = 0`.
  have h0 : ((0 : ℤ) : ℝ) / uQ_real = 0 := by push_cast; ring
  rw [h0]
  -- `(𝓕f(0)).re = ∫f = 1`.
  rw [Sidon.FourierAux.fourierIntegral_zero_real f, hf_one]
  -- `(𝓕K_ms(0)).re = ∫K_ms = 1`.
  have hK0 : (Real.fourierIntegral K_ms_complex 0).re = 1 := by
    have heq : Real.fourierIntegral K_ms_complex 0 = ((∫ x, K_ms x ∂volume : ℝ) : ℂ) := by
      unfold K_ms_complex
      rw [Sidon.FourierAux.fourierIntegral_zero (fun x => ((K_ms x : ℝ) : ℂ))]
      exact integral_ofReal (𝕜 := ℂ) (f := K_ms)
    rw [heq, Complex.ofReal_re, K_ms_integral_eq_one_local]
  rw [hK0]
  norm_num

/-! ## The Fourier–Bessel identity `(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`

This is the bridge from the *abstract* FT coefficient `(𝓕K_ms(j/u)).re`
to the project's *Bessel closed form*
`K_ms_fourier_lattice j = ∑ᵢ λᵢ·J₀(πδᵢ·j/u)²`.  FULLY PROVED from:

  * `Sidon.Bessel.besselJ0_arcsine_fourier` (the FT-cosine integral
    representation of `J₀` at width `δ/2`), giving the keystone
    `(𝓕η_δ)(ξ).re = J₀(πδξ)`;
  * evenness of `η_δ` (`Sidon.BundleEq3Schwartz.eta_even`) ⟹ `𝓕η_δ` is
    real;
  * the autoconvolution Fourier theorem
    `Sidon.Constructor.fourierIntegral_autoconvolution_real`
    (`𝓕(η_δ*η_δ) = (𝓕η_δ)²`), giving `(𝓕K_arc δ)(ξ).re = J₀(πδξ)²`;
  * additivity of `𝓕` (`Sidon.Constructor.PeriodU.fourierIntegral_add`)
    over the three-scale sum, with real constants `λᵢ` pulled out.

This `~200`-line Fourier–Bessel development is **not** present in mathlib
`v4.29.1` (no `𝓕(arcsine density) = J₀` lemma is exposed); proving it
here removes what would otherwise be a residual hypothesis. -/

/-- For integrable real `f`, `(𝓕(↑f))(w).re = ∫ f(v)·cos(2π v w) dv`
(local copy of `Sidon.Constructor.fourierIntegral_re_eq_cos_integral`, to
avoid the heavy `FieldEq4` import). -/
theorem fourierIntegral_re_eq_cos_integral_local
    (f : ℝ → ℝ) (hf : Integrable f volume) (w : ℝ) :
    (Real.fourierIntegral (fun x => (f x : ℂ)) w).re
      = ∫ v, f v * Real.cos (2 * Real.pi * v * w) ∂volume := by
  have hfℂ : Integrable (fun x => (f x : ℂ)) volume := by simpa using hf.ofReal
  have h_unfold :
      (Real.fourierIntegral (fun x => (f x : ℂ)) w)
        = ∫ v, (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ) ∂volume := by
    show (FourierTransform.fourier (fun x => (f x : ℂ)) w)
          = ∫ v, (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ) ∂volume
    rw [Real.fourier_real_eq]
  rw [h_unfold]
  have h_int_cplx :
      Integrable (fun v : ℝ => (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ)) volume := by
    have hconv := (Real.fourierIntegral_convergent_iff (V := ℝ)
      (f := fun x => (f x : ℂ)) w).mpr hfℂ
    refine hconv.congr (Filter.Eventually.of_forall (fun v => ?_))
    simp only []
    have hinner : (inner ℝ v w : ℝ) = v * w := by rw [real_inner_comm]; rfl
    rw [hinner]
  rw [← RCLike.re_to_complex]
  refine (integral_re h_int_cplx).symm.trans ?_
  refine integral_congr_ae (Filter.Eventually.of_forall (fun v => ?_))
  simp only []
  rw [RCLike.re_to_complex]
  rw [Circle.smul_def, Real.fourierChar_apply, smul_eq_mul, Complex.mul_re]
  simp only [Complex.ofReal_re, Complex.ofReal_im, sub_zero, mul_zero]
  rw [Complex.exp_ofReal_mul_I_re]
  rw [show (2 * Real.pi * -(v * w)) = -(2 * Real.pi * v * w) by ring, Real.cos_neg]
  ring

/-- **Keystone.** `(𝓕(η_δ))(ξ).re = J₀(πδξ)`. -/
theorem fourierIntegral_eta_re (δ : ℝ) (hδ : 0 < δ) (ξ : ℝ) :
    (Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ).re
      = Sidon.Bessel.besselJ0 (Real.pi * δ * ξ) := by
  have hδhalf : 0 < δ / 2 := by linarith
  rw [fourierIntegral_re_eq_cos_integral_local (eta δ) (eta_integrable δ hδ) ξ]
  rw [show (fun v => eta δ v * Real.cos (2 * Real.pi * v * ξ))
        = (Set.Ioo (-(δ/2)) (δ/2)).indicator
            (fun v => (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - v^2))⁻¹
              * Real.cos (2 * Real.pi * v * ξ)) from by
      funext v
      rw [eta_eq_indicator δ v]
      by_cases hv : v ∈ Set.Ioo (-(δ/2)) (δ/2)
      · rw [Set.indicator_of_mem hv, Set.indicator_of_mem hv]
      · rw [Set.indicator_of_notMem hv, Set.indicator_of_notMem hv, zero_mul]]
  rw [integral_indicator measurableSet_Ioo]
  have hbessel := Sidon.Bessel.besselJ0_arcsine_fourier (δ := δ/2) hδhalf ξ
  have harg : 2 * Real.pi * (δ/2) * ξ = Real.pi * δ * ξ := by ring
  rw [harg] at hbessel
  rw [hbessel]
  rw [show (fun x : ℝ => 1 / Real.pi * (Real.sqrt ((δ/2)^2 - x^2))⁻¹
            * Real.cos (2 * Real.pi * x * ξ))
        = (fun x : ℝ => (1 / Real.pi) *
            ((Real.sqrt ((δ/2)^2 - x^2))⁻¹ * Real.cos (2 * Real.pi * x * ξ))) from by
      funext x; ring]
  rw [MeasureTheory.integral_const_mul]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioo (fun x _ => ?_)
  rw [show (2 * Real.pi * x * ξ) = (2 * Real.pi * ξ * x) by ring]
  rw [div_eq_mul_inv]
  ring

/-- `𝓕(η_δ)(ξ) = ((J₀(πδξ) : ℝ) : ℂ)` (real, eta even). -/
theorem fourierIntegral_eta_real (δ : ℝ) (hδ : 0 < δ) (ξ : ℝ) :
    Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ
      = ((Sidon.Bessel.besselJ0 (Real.pi * δ * ξ) : ℝ) : ℂ) := by
  have h_int : Integrable (fun x => ((eta δ x : ℝ) : ℂ)) volume :=
    (eta_integrable δ hδ).ofReal
  have h_even : Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) (-ξ)
      = Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ := by
    show FourierTransform.fourier (fun x => ((eta δ x : ℝ) : ℂ)) (-ξ)
        = FourierTransform.fourier (fun x => ((eta δ x : ℝ) : ℂ)) ξ
    rw [Real.fourier_real_eq, Real.fourier_real_eq]
    rw [show (∫ v, (𝐞 (-(v * ξ)) : Circle) • ((eta δ v : ℝ) : ℂ))
          = ∫ v, (𝐞 (-((-v) * ξ)) : Circle) • ((eta δ (-v) : ℝ) : ℂ) from
        (MeasureTheory.integral_neg_eq_self
          (fun v => (𝐞 (-(v * ξ)) : Circle) • ((eta δ v : ℝ) : ℂ)) volume).symm]
    refine integral_congr_ae (Filter.Eventually.of_forall fun v => ?_)
    simp only [Sidon.BundleEq3Schwartz.eta_even]
    congr 2
    ring
  have h_conj : Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) (-ξ)
      = starRingEnd ℂ (Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ) :=
    Sidon.FourierAux.fourierIntegral_real_conj (eta δ) h_int ξ
  have h_real : starRingEnd ℂ (Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ)
      = Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ := by
    rw [← h_conj, h_even]
  have him : (Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ).im = 0 := by
    rw [Complex.ext_iff] at h_real
    have := h_real.2; simp only [Complex.conj_im] at this; linarith
  rw [Complex.ext_iff]
  refine ⟨?_, ?_⟩
  · rw [Complex.ofReal_re, fourierIntegral_eta_re δ hδ ξ]
  · rw [Complex.ofReal_im, him]

/-- `(𝓕(K_arc δ))(ξ).re = J₀(πδξ)²` via the autoconvolution Fourier theorem. -/
theorem fourierIntegral_K_arc_re (δ : ℝ) (hδ : 0 < δ) (ξ : ℝ) :
    (Real.fourierIntegral (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ).re
      = (Sidon.Bessel.besselJ0 (Real.pi * δ * ξ)) ^ 2 := by
  have h_conv : Real.fourierIntegral (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ
      = (Real.fourierIntegral (fun x => ((eta δ x : ℝ) : ℂ)) ξ) ^ 2 := by
    unfold K_arc
    exact Sidon.Constructor.fourierIntegral_autoconvolution_real (eta δ) (eta_integrable δ hδ) ξ
  rw [h_conv, fourierIntegral_eta_real δ hδ ξ]
  rw [show (((Sidon.Bessel.besselJ0 (Real.pi * δ * ξ) : ℝ) : ℂ)) ^ 2
        = (((Sidon.Bessel.besselJ0 (Real.pi * δ * ξ) ^ 2 : ℝ)) : ℂ) by push_cast; ring]
  rw [Complex.ofReal_re]

/-- `(𝓕(c·K_arc δ))(ξ).re = c·J₀(πδξ)²` for real constant `c`. -/
theorem fourierIntegral_const_K_arc_re (c δ : ℝ) (hδ : 0 < δ) (ξ : ℝ) :
    (Real.fourierIntegral (fun x => ((c * K_arc δ x : ℝ) : ℂ)) ξ).re
      = c * (Sidon.Bessel.besselJ0 (Real.pi * δ * ξ)) ^ 2 := by
  have h_smul : (fun x => ((c * K_arc δ x : ℝ) : ℂ))
      = (fun x => (c : ℂ) * ((K_arc δ x : ℝ) : ℂ)) := by funext x; push_cast; ring
  rw [h_smul]
  have h_const : Real.fourierIntegral (fun x => (c : ℂ) * ((K_arc δ x : ℝ) : ℂ)) ξ
      = (c : ℂ) * Real.fourierIntegral (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ := by
    show FourierTransform.fourier (fun x => (c : ℂ) * ((K_arc δ x : ℝ) : ℂ)) ξ
        = (c : ℂ) * FourierTransform.fourier (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ
    rw [Real.fourier_real_eq, Real.fourier_real_eq]
    rw [show (fun v => (𝐞 (-(v * ξ)) : Circle) • ((c : ℂ) * ((K_arc δ v : ℝ) : ℂ)))
          = (fun v => (c : ℂ) * ((𝐞 (-(v * ξ)) : Circle) • ((K_arc δ v : ℝ) : ℂ))) from by
        funext v; rw [Circle.smul_def, Circle.smul_def, smul_eq_mul, smul_eq_mul]; ring]
    exact integral_const_mul (c : ℂ) _
  rw [h_const, Complex.mul_re]
  rw [fourierIntegral_K_arc_re δ hδ ξ]
  rw [Complex.ofReal_re, Complex.ofReal_im]
  have him : (Real.fourierIntegral (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ).im = 0 := by
    have h_conv : Real.fourierIntegral (fun x => ((K_arc δ x : ℝ) : ℂ)) ξ
        = (((Sidon.Bessel.besselJ0 (Real.pi * δ * ξ) ^ 2 : ℝ)) : ℂ) := by
      unfold K_arc
      rw [Sidon.Constructor.fourierIntegral_autoconvolution_real (eta δ) (eta_integrable δ hδ) ξ,
        fourierIntegral_eta_real δ hδ ξ]
      push_cast; ring
    rw [h_conv, Complex.ofReal_im]
  rw [him]; ring

/-- **The Fourier–Bessel identity.**  `(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`. -/
theorem bessel_identity (j : ℤ) :
    (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re
      = K_ms_fourier_lattice j := by
  have hδ := delta_positivities
  set ξ : ℝ := (j / uQ_real : ℝ) with hξ
  have hI1 : Integrable (fun x => ((lambda1 * K_arc delta1 x : ℝ) : ℂ)) volume :=
    ((K_arc_integrable _ hδ.1).const_mul lambda1).ofReal
  have hI2 : Integrable (fun x => ((lambda2 * K_arc delta2 x : ℝ) : ℂ)) volume :=
    ((K_arc_integrable _ hδ.2.1).const_mul lambda2).ofReal
  have hI3 : Integrable (fun x => ((lambda3 * K_arc delta3 x : ℝ) : ℂ)) volume :=
    ((K_arc_integrable _ hδ.2.2).const_mul lambda3).ofReal
  have h_split : Real.fourierIntegral K_ms_complex ξ
      = Real.fourierIntegral (fun x => ((lambda1 * K_arc delta1 x : ℝ) : ℂ)) ξ
        + Real.fourierIntegral (fun x => ((lambda2 * K_arc delta2 x : ℝ) : ℂ)) ξ
        + Real.fourierIntegral (fun x => ((lambda3 * K_arc delta3 x : ℝ) : ℂ)) ξ := by
    unfold K_ms_complex MultiScale.K_ms
    rw [show (fun x => (((lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x
              + lambda3 * K_arc delta3 x : ℝ)) : ℂ))
          = (fun x => (((lambda1 * K_arc delta1 x : ℝ) : ℂ)
              + ((lambda2 * K_arc delta2 x : ℝ) : ℂ))
              + ((lambda3 * K_arc delta3 x : ℝ) : ℂ)) from by funext x; push_cast; ring]
    rw [Sidon.Constructor.PeriodU.fourierIntegral_add
          (fun x => ((lambda1 * K_arc delta1 x : ℝ) : ℂ)
            + ((lambda2 * K_arc delta2 x : ℝ) : ℂ))
          (fun x => ((lambda3 * K_arc delta3 x : ℝ) : ℂ)) (hI1.add hI2) hI3 ξ]
    rw [Sidon.Constructor.PeriodU.fourierIntegral_add
          (fun x => ((lambda1 * K_arc delta1 x : ℝ) : ℂ))
          (fun x => ((lambda2 * K_arc delta2 x : ℝ) : ℂ)) hI1 hI2 ξ]
  rw [h_split, Complex.add_re, Complex.add_re]
  rw [fourierIntegral_const_K_arc_re lambda1 delta1 hδ.1 ξ,
      fourierIntegral_const_K_arc_re lambda2 delta2 hδ.2.1 ξ,
      fourierIntegral_const_K_arc_re lambda3 delta3 hδ.2.2 ξ]
  unfold K_ms_fourier_lattice
  rw [hξ]

/-! ## Bridge to the exact P3 (`S_cos`) form

The abstract sum uses the FT coefficient `(𝓕K_ms(j/u)).re`; the project's
`S_cos f` uses the Bessel closed form `K_ms_fourier_lattice j`.  These are
equal by `bessel_identity` (FULLY PROVED above).  The exact P3 identity
`LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f` (= the bundle field
`h_torus_eq_full`) is assembled — hypothesis-free, via the periodisation
Poisson sampling — in
`Sidon.Constructor.PoissonSampling.field_P3_periodUParsevalTrue_via_periodization`,
which reuses `S_cos_abstract_eq_S_cos` below. -/

/-- `S_cos_abstract f = S_cos f` (via the proved `bessel_identity`). -/
theorem S_cos_abstract_eq_S_cos (f : ℝ → ℝ) :
    S_cos_abstract f = S_cos f := by
  unfold S_cos_abstract S_cos
  refine tsum_congr (fun j => ?_)
  split_ifs with hj
  · rfl
  · rw [bessel_identity j]

end

end Sidon.Constructor.MOLemma21
