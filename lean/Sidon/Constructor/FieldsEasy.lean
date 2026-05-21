/-
Sidon Autocorrelation Project — "easy" bundle-field obligations.
================================================================

For a general admissible `f` (nonneg, `L¹ ∩ L²`, supported in
`(-1/4, 1/4)`, mass `1`), this file discharges three of the
`ExtremiserPrimitives` bundle fields needed by the constructor that
makes `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
unconditional:

  * `field_K2_ge_1`  : `1 ≤ K2_analytic`           (f-independent).
  * `field_R_ge_1`   : `1 ≤ autoconvolution_ratio f`.
  * `field_hEq1`     : `LHS1 f ≤ autoconvolution_ratio f`.

The only sanctioned external hypothesis is `hK_sq_int :
Integrable (K_ms²)` (the BLOCKED `L²`-membership of `K_ms`, which
mathlib `v4.29.1` cannot yet establish for lack of a `Lᵖ` Young
inequality / `J₀⁴ ∈ L¹` theory).

A shared prerequisite, `conv_eLpNorm_top_ne_top`, proves that the
autoconvolution `f*f` is essentially bounded (`‖f*f‖_∞ < ⊤`) by an
everywhere Cauchy–Schwarz bound `(f*f)(x) ≤ ‖f‖_{L²}²`.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.BundleDefs
import Sidon.BundleEq1
import Sidon.BundleEq2Schwartz
import Sidon.Constructor.KernelFacts

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical ENNReal Pointwise
open MeasureTheory

namespace Sidon.Constructor

noncomputable section

/-! ## Shared prerequisite PRE-G — `f*f` is essentially bounded

For `f ∈ L¹ ∩ L²` we have the everywhere bound
`(f*f)(x) = ∫ f(t)·f(x-t) dt ≤ ‖f‖_{L²}²` by Cauchy–Schwarz and
translation invariance.  Hence `eLpNormEssSup (f*f) ≤ ‖f‖_{L²}² < ⊤`.

We work at the level of extended-nonneg reals: with
`C := ∫⁻ t, ‖f t‖ₑ²`, the bound `‖(f*f)(x)‖ₑ ≤ C` holds for every `x`
and `C < ⊤` because `f ∈ L²`. -/

/-- Abbreviation matching the spelled-out convolution inside
`autoconvolution_ratio` / `LHS1`. -/
private def convff (f : ℝ → ℝ) : ℝ → ℝ :=
  MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume

private lemma convff_def (f : ℝ → ℝ) :
    convff f = MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := rfl

/-- The squared `L²`-`enorm` mass `C := ∫⁻ ‖f t‖ₑ²`. -/
private def L2sqEnorm (f : ℝ → ℝ) : ℝ≥0∞ := ∫⁻ t, ‖f t‖ₑ ^ (2 : ℝ) ∂volume

/-- `C < ⊤` for `f ∈ L²`. -/
private lemma L2sqEnorm_lt_top {f : ℝ → ℝ} (hf_L2 : MemLp f 2 volume) :
    L2sqEnorm f < ⊤ := by
  have h := lintegral_rpow_enorm_lt_top_of_eLpNorm_lt_top
    (p := (2 : ℝ≥0∞)) (μ := volume) (f := f)
    (by norm_num) (by norm_num) hf_L2.eLpNorm_lt_top
  -- `(2 : ℝ≥0∞).toReal = 2`.
  simpa [L2sqEnorm] using h

/-- **Everywhere** Cauchy–Schwarz bound on the autoconvolution:
`‖(f*f)(x)‖ₑ ≤ ∫⁻ t, ‖f t‖ₑ²` for every `x`. -/
private lemma convff_enorm_le {f : ℝ → ℝ} (hf_L2 : MemLp f 2 volume) (x : ℝ) :
    ‖convff f x‖ₑ ≤ L2sqEnorm f := by
  -- a.e.-measurability of `t ↦ ‖f t‖ₑ` and `t ↦ ‖f (x-t)‖ₑ`.
  have h_aesm : AEStronglyMeasurable f volume := hf_L2.aestronglyMeasurable
  have hmeas : AEMeasurable (fun t => ‖f t‖ₑ) volume := h_aesm.enorm
  have hmeas_sub : AEMeasurable (fun t => ‖f (x - t)‖ₑ) volume :=
    (h_aesm.comp_quasiMeasurePreserving
      (quasiMeasurePreserving_sub_left_of_right_invariant volume x)).enorm
  -- Unfold the convolution as an integral.
  have h_unfold : convff f x = ∫ t, f t * f (x - t) ∂volume := by
    rw [convff_def, convolution_def]
    simp only [ContinuousLinearMap.mul_apply']
  rw [h_unfold]
  -- Pull the enorm inside the integral.
  calc ‖∫ t, f t * f (x - t) ∂volume‖ₑ
      ≤ ∫⁻ t, ‖f t * f (x - t)‖ₑ ∂volume :=
        enorm_integral_le_lintegral_enorm _
    _ = ∫⁻ t, (fun t => ‖f t‖ₑ) t * (fun t => ‖f (x - t)‖ₑ) t ∂volume := by
        congr 1; ext t; rw [enorm_mul]
    _ ≤ (∫⁻ t, (fun t => ‖f t‖ₑ) t ^ (2 : ℝ) ∂volume) ^ (1 / (2 : ℝ))
          * (∫⁻ t, (fun t => ‖f (x - t)‖ₑ) t ^ (2 : ℝ) ∂volume) ^ (1 / (2 : ℝ)) :=
        ENNReal.lintegral_mul_le_Lp_mul_Lq volume
          (Real.holderConjugate_iff.mpr ⟨by norm_num, by norm_num⟩)
          hmeas hmeas_sub
    _ = L2sqEnorm f := by
        -- Translation invariance: ∫⁻ ‖f(x-t)‖ₑ² = ∫⁻ ‖f t‖ₑ².
        simp only []
        have h_transl :
            (∫⁻ t, ‖f (x - t)‖ₑ ^ (2 : ℝ) ∂volume)
              = ∫⁻ t, ‖f t‖ₑ ^ (2 : ℝ) ∂volume :=
          lintegral_sub_left_eq_self (fun t => ‖f t‖ₑ ^ (2 : ℝ)) x
        rw [h_transl]
        -- A^(1/2) · A^(1/2) = A^(1/2+1/2) = A^1 = A.
        rw [← ENNReal.rpow_add_of_nonneg _ _ (by norm_num) (by norm_num)]
        norm_num [L2sqEnorm]

/-- **PRE-G.**  The autoconvolution `f*f` is essentially bounded:
`eLpNorm (f*f) ⊤ volume ≠ ⊤`, for `f ∈ L²`. -/
theorem conv_eLpNorm_top_ne_top {f : ℝ → ℝ} (hf_L2 : MemLp f 2 volume) :
    MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤ := by
  -- `eLpNorm ⊤ = eLpNormEssSup`, which is `≤ L2sqEnorm f < ⊤`.
  rw [eLpNorm_exponent_top]
  have h_bound : eLpNormEssSup (convff f) volume ≤ L2sqEnorm f :=
    eLpNormEssSup_le_of_ae_enorm_bound
      (Filter.Eventually.of_forall (convff_enorm_le hf_L2))
  have h_lt : eLpNormEssSup (convff f) volume < ⊤ :=
    lt_of_le_of_lt h_bound (L2sqEnorm_lt_top hf_L2)
  -- `convff f` is defeq to the spelled-out convolution.
  exact ne_of_lt h_lt

/-! ## Target 1 — `field_K2_ge_1` (f-independent) -/

/-- **Target 1.**  The kernel `L²`-mass functional is `≥ 1`.

This is `f`-independent.  It follows from Cauchy–Schwarz applied to
`K_ms` on `[-δ₁, δ₁]` (length `2δ₁ = 0.276 < 1`), using `∫ K_ms = 1`.
The only blocked input is the `L²`-integrability of `K_ms`, supplied
as `hK_sq_int`. -/
theorem field_K2_ge_1
    (hK_sq_int : Integrable (fun x => Sidon.MultiScale.K_ms x ^ 2) volume) :
    1 ≤ Sidon.MultiScale.K2_analytic := by
  -- Assemble the prerequisites for `K2_analytic_ge_one_of_integral_one`.
  have h_K_int : Integrable Sidon.MultiScale.K_ms volume :=
    Sidon.BundleEq1.K_ms_integrable
  have h_K_one : ∫ x, Sidon.MultiScale.K_ms x ∂volume = 1 :=
    Sidon.BundleEq1.K_ms_integral_eq_one
  -- Support: `δ₁ < |x| → K_ms x = 0` (from the `≤` form).
  have h_K_supp : ∀ x, Sidon.MultiScale.delta1 < |x| →
      Sidon.MultiScale.K_ms x = 0 := fun x hx =>
    Sidon.BundleEq2Schwartz.K_ms_eq_zero_outside x (le_of_lt hx)
  -- Integrable-on restrictions.
  have h_K_int_on :
      IntegrableOn Sidon.MultiScale.K_ms
        (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1) volume :=
    h_K_int.integrableOn
  have h_K_sq_int_on :
      IntegrableOn (fun x => Sidon.MultiScale.K_ms x ^ 2)
        (Set.Icc (-Sidon.MultiScale.delta1) Sidon.MultiScale.delta1) volume :=
    hK_sq_int.integrableOn
  exact Sidon.BundleDefs.K2_analytic_ge_one_of_integral_one
    h_K_int h_K_int_on h_K_sq_int_on hK_sq_int h_K_supp h_K_one

/-! ## Target 2 — `field_R_ge_1`

`1 ≤ autoconvolution_ratio f` for an admissible nonneg `f` with mass `1`.

Argument (MV's "mass floor"): `∫(f*f) = (∫f)² = 1` (Fubini); the
support of `f*f` lies in `(-1/2, 1/2)` (length `1`); hence
`1 = ∫(f*f) = ∫_{supp} (f*f) ≤ ‖f*f‖_∞ · vol(supp) ≤ ‖f*f‖_∞`. -/

/-- Support of `f*f` lies in `(-1/2, 1/2)` for `f` supported in
`(-1/4, 1/4)`. -/
private lemma convff_support_subset {f : ℝ → ℝ}
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (convff f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  -- `x ∈ support (f*f) ⊆ support f + support f`.
  have h_sub : Function.support (convff f) ⊆
      Function.support f + Function.support f := by
    rw [convff_def]; exact support_convolution_subset _
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (h_sub hx)
  have ha' := hf_supp ha
  have hb' := hf_supp hb
  simp only [Set.mem_Ioo] at ha' hb' ⊢
  subst hab
  exact ⟨by linarith [ha'.1, hb'.1], by linarith [ha'.2, hb'.2]⟩

/-- **Mass floor.**  `ENNReal.ofReal 1 ≤ eLpNorm (f*f) ⊤ volume`
for an admissible nonneg `f` with mass `1`. -/
private lemma convff_essSup_ge_one {f : ℝ → ℝ}
    (hf_int : Integrable f volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    ENNReal.ofReal 1 ≤ eLpNorm (convff f) ⊤ volume := by
  -- `g := f*f`, nonneg, integrable, supported in `s := Ioo(-1/2, 1/2)`.
  have h_g_nonneg : ∀ x, 0 ≤ convff f x := fun x => by
    rw [convff_def]; exact convolution_nonneg hf_nonneg hf_nonneg x
  have h_g_int : Integrable (convff f) volume := by
    rw [convff_def]; exact hf_int.integrable_convolution _ hf_int
  -- `∫ g = (∫f)·(∫f) = 1` (Fubini via `integral_convolution`).
  have h_g_one : ∫ x, convff f x ∂volume = 1 := by
    rw [convff_def, integral_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int hf_int]
    simp only [ContinuousLinearMap.mul_apply', hf_one, mul_one]
  set s : Set ℝ := Set.Ioo (-(1/2 : ℝ)) (1/2) with hs_def
  have hs_meas : MeasurableSet s := measurableSet_Ioo
  have hs_vol : volume s = 1 := by
    rw [hs_def, Real.volume_Ioo]; norm_num
  have h_supp_s : Function.support (convff f) ⊆ s := convff_support_subset hf_supp
  set C : ℝ≥0∞ := eLpNormEssSup (convff f) volume with hC_def
  -- ∫⁻ ‖g‖ₑ = ofReal (∫ g) = ofReal 1.
  have h_lint_eq : (∫⁻ x, ‖convff f x‖ₑ ∂volume) = ENNReal.ofReal 1 := by
    have h_ofReal :
        (fun x => ‖convff f x‖ₑ) = fun x => ENNReal.ofReal (convff f x) := by
      ext x; rw [Real.enorm_of_nonneg (h_g_nonneg x)]
    rw [h_ofReal, ← ofReal_integral_eq_lintegral_ofReal h_g_int
      (Filter.Eventually.of_forall h_g_nonneg), h_g_one]
  -- ∫⁻ ‖g‖ₑ = ∫⁻_s ‖g‖ₑ (zero outside s).
  have h_restrict : (∫⁻ x, ‖convff f x‖ₑ ∂volume)
      = ∫⁻ x in s, ‖convff f x‖ₑ ∂volume := by
    rw [← lintegral_indicator hs_meas]
    congr 1; ext x
    by_cases hx : x ∈ s
    · rw [Set.indicator_of_mem hx]
    · rw [Set.indicator_of_notMem hx]
      have : convff f x = 0 := by
        by_contra h; exact hx (h_supp_s h)
      rw [this]; simp
  -- ∫⁻_s ‖g‖ₑ ≤ ∫⁻_s C = C · vol s = C.
  have h_le_C : (∫⁻ x in s, ‖convff f x‖ₑ ∂volume) ≤ C := by
    calc (∫⁻ x in s, ‖convff f x‖ₑ ∂volume)
        ≤ ∫⁻ _ in s, C ∂volume := by
          refine setLIntegral_mono_ae' hs_meas ?_
          filter_upwards [ae_le_eLpNormEssSup (f := convff f) (μ := volume)]
            with x hx _ using hx
      _ = C * volume s := setLIntegral_const s C
      _ = C := by rw [hs_vol, mul_one]
  -- Chain: ofReal 1 = ∫⁻ ‖g‖ₑ = ∫⁻_s ‖g‖ₑ ≤ C = eLpNorm ⊤.
  rw [eLpNorm_exponent_top, ← hC_def]
  calc ENNReal.ofReal 1 = ∫⁻ x, ‖convff f x‖ₑ ∂volume := h_lint_eq.symm
    _ = ∫⁻ x in s, ‖convff f x‖ₑ ∂volume := h_restrict
    _ ≤ C := h_le_C

/-- **Target 2.**  `1 ≤ autoconvolution_ratio f` for an admissible
nonneg `f` with mass `1`. -/
theorem field_R_ge_1 (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    1 ≤ autoconvolution_ratio f := by
  -- `∫(f*f) = 1` (Fubini).
  have h_conv_one :
      ∫ x, (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x ∂volume = 1 := by
    rw [integral_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int hf_int]
    simp only [ContinuousLinearMap.mul_apply', hf_one, mul_one]
  -- Support measure ≤ ofReal 1.
  have h_conv_supp_meas :
      volume (Function.support (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)) ≤ ENNReal.ofReal 1 := by
    have h_sub : Function.support (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := convff_support_subset hf_supp
    calc volume (Function.support _) ≤ volume (Set.Ioo (-(1/2 : ℝ)) (1/2)) :=
          measure_mono h_sub
      _ = ENNReal.ofReal 1 := by rw [Real.volume_Ioo]; norm_num
  -- Nonnegativity.
  have h_conv_nonneg : ∀ x, 0 ≤ (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x :=
    convolution_nonneg hf_nonneg hf_nonneg
  -- Mass floor: essSup ≥ 1.
  have h_essSup_ge : ENNReal.ofReal 1 ≤ eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ volume :=
    convff_essSup_ge_one hf_int hf_supp hf_nonneg hf_one
  -- Finiteness (PRE-G).
  have h_essSup_fin : eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ volume ≠ ⊤ :=
    conv_eLpNorm_top_ne_top hf_L2
  exact Sidon.BundleDefs.R_ge_one_of_data hf_one h_conv_one h_conv_supp_meas
    h_conv_nonneg h_essSup_ge h_essSup_fin

/-! ## Target 3 — `field_hEq1` (MV Lemma 3.1 Eq.(1) discharge) -/

/-- **Target 3.**  The MV Lemma 3.1 Eq.(1) bundle field
`LHS1 f ≤ autoconvolution_ratio f`, valid for an admissible nonneg `f`
with mass `1`.

The convolution integrability `h_prod_int` is produced from
`Sidon.BundleEq1.conv_K_ms_integrable`, whose `AEStronglyMeasurable`
prerequisite comes from `Integrable.integrable_convolution` (our `f`
is integrable).  The finiteness `h_conv_fin` is `PRE-G`. -/
theorem field_hEq1 (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    Sidon.MultiScale.LHS1 f ≤ autoconvolution_ratio f := by
  -- `PRE-G`: finiteness of `‖f*f‖_∞`.  `BundleEq1.conv f` is defeq to the
  -- spelled-out convolution, so `conv_eLpNorm_top_ne_top` applies.
  have h_conv_fin :
      eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ volume ≠ ⊤ :=
    conv_eLpNorm_top_ne_top hf_L2
  -- `f*f` is integrable, hence a.e.-strongly-measurable.
  have h_conv_int :
      Integrable (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) volume :=
    hf_int.integrable_convolution _ hf_int
  have h_conv_meas :
      AEStronglyMeasurable (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) volume :=
    h_conv_int.aestronglyMeasurable
  -- Integrability of `(f*f) · K_ms`.
  have h_prod_int :
      Integrable (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x
          * Sidon.MultiScale.K_ms x) volume :=
    Sidon.BundleEq1.conv_K_ms_integrable hf_nonneg h_conv_meas h_conv_fin
  -- `Sidon.BundleEq1.hEq1_discharge` concludes `BundleEq1.LHS1 f ≤
  -- autoconvolution_ratio f`; `BundleEq1.LHS1 f` is defeq to
  -- `Sidon.MultiScale.LHS1 f`.
  exact Sidon.BundleEq1.hEq1_discharge f hf_nonneg hf_supp hf_int hf_one
    h_conv_fin h_prod_int

end -- noncomputable section

end Sidon.Constructor
