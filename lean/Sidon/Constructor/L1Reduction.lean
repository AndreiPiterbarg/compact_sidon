/-
Sidon Autocorrelation Project — L¹ reduction (removing the `MemLp f 2` hypothesis).
================================================================================

The unconditional headline `Sidon.MultiScale.C1a_ge_1292_unconditional`
takes admissibility *including* `MemLp f 2` (square-integrability).  The
literature constant `C₁ₐ = inf R(f)` is an infimum over the full
nonnegative `L¹` class.  Matolcsi–Vinuesa 2010 §2 bridge the two by the
Schinzel–Schmidt 2002 step-function reduction.  This file formalises that
bridge directly, by the elementary truncation `fₙ = min (f, n)`:

  * `fₙ` is bounded with compact support, hence in `L¹ ∩ L²`;
  * `fₙ ≤ f` (nonneg), so `fₙ ⋆ fₙ ≤ f ⋆ f` pointwise a.e., giving
    `‖fₙ ⋆ fₙ‖_∞ ≤ ‖f ⋆ f‖_∞`;
  * `∫ fₙ → ∫ f` (dominated convergence);
  * the L² headline applies to (the normalisation of) `fₙ`.

**Why the conclusion is in `ℝ≥0∞`, not via `autoconvolution_ratio`.**
`autoconvolution_ratio f = (eLpNorm (f⋆f) ⊤).toReal / (∫f)²`, and mathlib's
`(⊤).toReal = 0`.  There exist admissible `L¹` functions with
`‖f⋆f‖_∞ = ∞` (e.g. `f(t) = t^{-1+ε}` on `(0,1/4)`, `0 < ε < 1/2`:
`f ∈ L¹`, `f ∉ L²`, `(f⋆f)(x) ∼ x^{-1+2ε} → ∞`), for which
`autoconvolution_ratio f = 0 < 1.292`.  So `∀ f∈L¹, R(f) ≥ 1.292` is
literally FALSE under the `.toReal` junk-value convention.  The
mathematically correct fully-unconditional statement is the `ℝ≥0∞`
inequality `ENNReal.ofReal (1292/1000) ≤ eLpNorm (f⋆f) ⊤`, which is true
for ALL admissible `L¹ f` (trivially when the left-hand target meets `⊤`).
This is exactly `C₁ₐ ≥ 1.292` over the full class.  (Note: `f⋆f` bounded
does NOT imply `f ∈ L²` — `t^{-1/2}` has `f⋆f ≡ π` yet `f ∉ L²` — so the
truncation/approximation argument is genuinely required; there is no
shortcut splitting on boundedness.)

Headline: `Sidon.Constructor.C1a_ge_1292_L1`.

No `sorry`, no new `axiom`.  (Inherits the four sanctioned
verifiable-by-computation numerical axioms via the L² headline.)
-/

import Mathlib
import Sidon.Constructor.Assembly
import Sidon.Constructor.FieldsEasy

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical ENNReal
open MeasureTheory

namespace Sidon.Constructor

noncomputable section

/-- The mul-convolution autoconvolution `f ⋆ f` used throughout the project. -/
local notation3 "conv " f => MeasureTheory.convolution f f
  (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume

/-- Truncation `fₙ(x) = min (f x) n`. -/
def gtrunc (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ := fun x => min (f x) (n : ℝ)

variable {f : ℝ → ℝ}

theorem gtrunc_nonneg (hf_nonneg : ∀ x, 0 ≤ f x) (n : ℕ) (x : ℝ) :
    0 ≤ gtrunc f n x :=
  le_min (hf_nonneg x) (Nat.cast_nonneg n)

theorem gtrunc_le (n : ℕ) (x : ℝ) : gtrunc f n x ≤ f x := min_le_left _ _

theorem gtrunc_le_n (n : ℕ) (x : ℝ) : gtrunc f n x ≤ (n : ℝ) := min_le_right _ _

theorem gtrunc_supp (n : ℕ) :
    Function.support (gtrunc f n) ⊆ Function.support f := by
  intro x hx
  simp only [Function.mem_support, gtrunc] at hx ⊢
  intro hfx
  apply hx
  rw [hfx]
  exact min_eq_left (Nat.cast_nonneg n)

theorem gtrunc_aesm (hf_int : Integrable f volume) (n : ℕ) :
    AEStronglyMeasurable (gtrunc f n) volume :=
  (hf_int.aemeasurable.min aemeasurable_const).aestronglyMeasurable

theorem gtrunc_integrable (hf_int : Integrable f volume) (hf_nonneg : ∀ x, 0 ≤ f x)
    (n : ℕ) : Integrable (gtrunc f n) volume := by
  refine hf_int.mono' (gtrunc_aesm hf_int n) (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (gtrunc_nonneg hf_nonneg n x)]
  exact gtrunc_le n x

theorem gtrunc_sq_integrable (hf_int : Integrable f volume) (hf_nonneg : ∀ x, 0 ≤ f x)
    (n : ℕ) : Integrable (fun x => gtrunc f n x ^ 2) volume := by
  refine (hf_int.const_mul (n : ℝ)).mono'
    ((continuous_pow 2).comp_aestronglyMeasurable (gtrunc_aesm hf_int n))
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  -- gₙ² = gₙ·gₙ ≤ n·gₙ ≤ n·f
  have h1 : gtrunc f n x ^ 2 ≤ (n : ℝ) * gtrunc f n x := by
    rw [sq]
    exact mul_le_mul_of_nonneg_right (gtrunc_le_n n x) (gtrunc_nonneg hf_nonneg n x)
  have h2 : (n : ℝ) * gtrunc f n x ≤ (n : ℝ) * f x :=
    mul_le_mul_of_nonneg_left (gtrunc_le n x) (Nat.cast_nonneg n)
  linarith

theorem gtrunc_memLp (hf_int : Integrable f volume) (hf_nonneg : ∀ x, 0 ≤ f x)
    (n : ℕ) : MemLp (gtrunc f n) 2 volume :=
  (memLp_two_iff_integrable_sq (gtrunc_aesm hf_int n)).mpr
    (gtrunc_sq_integrable hf_int hf_nonneg n)

/-- `∫ fₙ → ∫ f` by dominated convergence (dominated by `f`). -/
theorem gtrunc_tendsto_integral (hf_int : Integrable f volume) (hf_nonneg : ∀ x, 0 ≤ f x) :
    Filter.Tendsto (fun n => ∫ x, gtrunc f n x ∂volume) Filter.atTop
      (nhds (∫ x, f x ∂volume)) := by
  refine tendsto_integral_of_dominated_convergence f
    (fun n => (gtrunc_aesm hf_int n).aemeasurable.aestronglyMeasurable)
    hf_int (fun n => ae_of_all _ fun x => ?_) (ae_of_all _ fun x => ?_)
  · rw [Real.norm_eq_abs, abs_of_nonneg (gtrunc_nonneg hf_nonneg n x)]
    exact gtrunc_le n x
  · -- pointwise: min (f x) n → f x
    have hev : (fun _ : ℕ => f x) =ᶠ[Filter.atTop] (fun n => gtrunc f n x) := by
      filter_upwards [Filter.eventually_ge_atTop ⌈f x⌉₊] with n hn
      have hle : f x ≤ (n : ℝ) := le_trans (Nat.le_ceil (f x)) (by exact_mod_cast hn)
      exact (min_eq_left hle).symm
    exact Filter.Tendsto.congr' hev tendsto_const_nhds

/-- Convolution `essSup`-monotonicity: if `0 ≤ g ≤ f` (both integrable),
then `‖g⋆g‖_∞ ≤ ‖f⋆f‖_∞`. -/
theorem conv_essSup_mono {g : ℝ → ℝ}
    (hf_int : Integrable f volume) (hg_int : Integrable g volume)
    (hf_nonneg : ∀ x, 0 ≤ f x) (hg_nonneg : ∀ x, 0 ≤ g x)
    (hgf : ∀ x, g x ≤ f x) :
    eLpNorm (conv g) ⊤ volume ≤ eLpNorm (conv f) ⊤ volume := by
  have h_ae : ∀ᵐ x ∂volume, (conv g) x ≤ (conv f) x := by
    filter_upwards [hg_int.ae_convolution_exists (L := ContinuousLinearMap.mul ℝ ℝ) hg_int,
      hf_int.ae_convolution_exists (L := ContinuousLinearMap.mul ℝ ℝ) hf_int]
      with x hxg hxf
    rw [convolution_def, convolution_def]
    apply integral_mono hxg hxf
    intro t
    simp only [ContinuousLinearMap.mul_apply']
    exact mul_le_mul (hgf t) (hgf (x - t)) (hg_nonneg (x - t)) (hf_nonneg t)
  refine eLpNorm_mono_ae_real ?_
  filter_upwards [h_ae] with x hx
  rw [Real.norm_eq_abs, abs_of_nonneg (convolution_nonneg hg_nonneg hg_nonneg x)]
  exact hx

/-- **Scale-free `ℝ≥0∞` form of the L² headline.**  For admissible L² `g`
with `∫ g > 0`,
`ofReal(1.292) · ofReal((∫g)²) ≤ ‖g⋆g‖_∞`.  Proved by normalising
`h = (∫g)⁻¹ • g` (so `∫h = 1`) and feeding the unconditional headline. -/
theorem L2_essSup_scalefree {g : ℝ → ℝ}
    (hg_int : Integrable g volume) (hg_L2 : MemLp g 2 volume)
    (hg_supp : Function.support g ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hg_nonneg : ∀ x, 0 ≤ g x) (hg_pos : 0 < ∫ x, g x ∂volume) :
    ENNReal.ofReal (1292/1000) * ENNReal.ofReal ((∫ x, g x ∂volume) ^ 2)
      ≤ eLpNorm (conv g) ⊤ volume := by
  set c : ℝ := ∫ x, g x ∂volume with hc
  have hc_ne : c ≠ 0 := ne_of_gt hg_pos
  have hci_pos : 0 < c⁻¹ := inv_pos.mpr hg_pos
  -- normalised pdf h = c⁻¹ • g
  set h : ℝ → ℝ := c⁻¹ • g with hh
  have hh_apply : ∀ x, h x = c⁻¹ * g x := fun x => rfl
  have hh_nonneg : ∀ x, 0 ≤ h x := fun x => by
    rw [hh_apply]; exact mul_nonneg (le_of_lt hci_pos) (hg_nonneg x)
  have hh_supp : Function.support h ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4) := by
    intro x hx
    apply hg_supp
    simp only [Function.mem_support, hh_apply] at hx ⊢
    intro h0
    exact hx (by rw [h0, mul_zero])
  have hh_int : Integrable h volume := by
    rw [hh]; exact hg_int.smul c⁻¹
  have hh_L2 : MemLp h 2 volume := hg_L2.const_smul c⁻¹
  have hh_one : ∫ x, h x ∂volume = 1 := by
    simp only [hh_apply]
    rw [integral_const_mul, ← hc, inv_mul_cancel₀ hc_ne]
  -- L² headline
  have hR := Sidon.MultiScale.C1a_ge_1292_unconditional h hh_int hh_L2 hh_supp hh_nonneg hh_one
  -- unfold the ratio; (∫h)² = 1
  have hEf_fin : eLpNorm (conv h) ⊤ volume ≠ ⊤ := conv_eLpNorm_top_ne_top hh_L2
  have hRatio : (eLpNorm (conv h) ⊤ volume).toReal ≥ (1292/1000 : ℝ) := by
    have := hR
    unfold autoconvolution_ratio at this
    simp only [hh_one, one_pow, div_one] at this
    exact this
  -- E_h ≥ ofReal(1.292)
  have hEh_ge : ENNReal.ofReal (1292/1000) ≤ eLpNorm (conv h) ⊤ volume := by
    rw [ENNReal.ofReal_le_iff_le_toReal hEf_fin]
    exact hRatio
  -- relate h⋆h = c⁻² • (g⋆g)
  have hconv_eq : (conv h) = (c⁻¹ * c⁻¹) • (conv g) := by
    rw [hh]
    rw [MeasureTheory.smul_convolution, MeasureTheory.convolution_smul, smul_smul]
  have hEh_eq : eLpNorm (conv h) ⊤ volume
      = ENNReal.ofReal (c⁻¹ * c⁻¹) * eLpNorm (conv g) ⊤ volume := by
    rw [hconv_eq, eLpNorm_const_smul]
    congr 1
    rw [Real.enorm_eq_ofReal_abs, abs_of_nonneg (by positivity)]
  -- so ofReal(c⁻²)·E_g ≥ ofReal(1.292); multiply by ofReal(c²)
  rw [hEh_eq] at hEh_ge
  have hcc : ENNReal.ofReal (c ^ 2) * ENNReal.ofReal (c⁻¹ * c⁻¹) = 1 := by
    rw [← ENNReal.ofReal_mul (by positivity)]
    rw [show c ^ 2 * (c⁻¹ * c⁻¹) = 1 by field_simp]
    exact ENNReal.ofReal_one
  calc ENNReal.ofReal (1292/1000) * ENNReal.ofReal (c ^ 2)
      = ENNReal.ofReal (c ^ 2) * ENNReal.ofReal (1292/1000) := by ring
    _ ≤ ENNReal.ofReal (c ^ 2)
          * (ENNReal.ofReal (c⁻¹ * c⁻¹) * eLpNorm (conv g) ⊤ volume) :=
        mul_le_mul_left' hEh_ge _
    _ = (ENNReal.ofReal (c ^ 2) * ENNReal.ofReal (c⁻¹ * c⁻¹))
          * eLpNorm (conv g) ⊤ volume := by rw [mul_assoc]
    _ = eLpNorm (conv g) ⊤ volume := by rw [hcc, one_mul]

/-- **Fully unconditional headline (full nonnegative `L¹` class).**
For admissible `L¹ f` (nonneg, supported in `(-1/4, 1/4)`, `∫f = 1`) — with
NO `MemLp f 2` hypothesis — the autoconvolution `essSup` satisfies
`ENNReal.ofReal (1292/1000) ≤ ‖f ⋆ f‖_∞`.  This is `C₁ₐ ≥ 1.292` over the
full class, in the `ℝ≥0∞` formulation that is correct for unbounded `f⋆f`.

The `MemLp f 2` step of admissibility is discharged here by the
Schinzel–Schmidt truncation `fₙ = min (f, n)` — the identical reduction
Matolcsi–Vinuesa 2010 §2 invoke. -/
theorem C1a_ge_1292_L1 (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    ENNReal.ofReal (1292/1000) ≤ eLpNorm (conv f) ⊤ volume := by
  -- For every n: ofReal(1.292)·ofReal((∫fₙ)²) ≤ ‖f⋆f‖_∞.
  have key : ∀ n : ℕ,
      ENNReal.ofReal (1292/1000) * ENNReal.ofReal ((∫ x, gtrunc f n x ∂volume) ^ 2)
        ≤ eLpNorm (conv f) ⊤ volume := by
    intro n
    rcases eq_or_lt_of_le (integral_nonneg (f := gtrunc f n) (gtrunc_nonneg hf_nonneg n))
      with h0 | hpos
    · -- ∫fₙ = 0 ⟹ LHS = 0
      rw [← h0]
      simp
    · -- ∫fₙ > 0 ⟹ Lemma A + monotonicity
      calc ENNReal.ofReal (1292/1000) * ENNReal.ofReal ((∫ x, gtrunc f n x ∂volume) ^ 2)
          ≤ eLpNorm (conv (gtrunc f n)) ⊤ volume :=
            L2_essSup_scalefree (gtrunc_integrable hf_int hf_nonneg n)
              (gtrunc_memLp hf_int hf_nonneg n)
              (fun x hx => hf_supp (gtrunc_supp n hx))
              (gtrunc_nonneg hf_nonneg n) hpos
        _ ≤ eLpNorm (conv f) ⊤ volume :=
            conv_essSup_mono hf_int (gtrunc_integrable hf_int hf_nonneg n)
              hf_nonneg (gtrunc_nonneg hf_nonneg n) (gtrunc_le n)
  -- Take n → ∞: ofReal(1.292)·ofReal((∫fₙ)²) → ofReal(1.292).
  have htend_int : Filter.Tendsto (fun n => ∫ x, gtrunc f n x ∂volume) Filter.atTop
      (nhds (1 : ℝ)) := by
    have := gtrunc_tendsto_integral hf_int hf_nonneg
    rwa [hf_one] at this
  have htend_sq : Filter.Tendsto (fun n => (∫ x, gtrunc f n x ∂volume) ^ 2) Filter.atTop
      (nhds (1 : ℝ)) := by
    have := htend_int.pow 2
    simpa using this
  have htend_ofReal : Filter.Tendsto
      (fun n => ENNReal.ofReal ((∫ x, gtrunc f n x ∂volume) ^ 2)) Filter.atTop
      (nhds (ENNReal.ofReal 1)) :=
    (ENNReal.continuous_ofReal.tendsto _).comp htend_sq
  have htend : Filter.Tendsto
      (fun n => ENNReal.ofReal (1292/1000) * ENNReal.ofReal ((∫ x, gtrunc f n x ∂volume) ^ 2))
      Filter.atTop (nhds (ENNReal.ofReal (1292/1000))) := by
    have hmul := ENNReal.Tendsto.const_mul (a := ENNReal.ofReal (1292/1000)) htend_ofReal
      (Or.inr ENNReal.ofReal_ne_top)
    simpa using hmul
  exact le_of_tendsto' htend key

end

end Sidon.Constructor
