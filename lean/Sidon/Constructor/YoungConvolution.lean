/-
Sidon Autocorrelation Project — Young's convolution inequality
`L^{4/3} ⋆ L^{4/3} → L²`, specialized to the autoconvolution of a
nonnegative, `L^{4/3}` function (`η_δ`).
==================================================================

GOAL.  Mathlib `v4.29.1` has **no** `MemLp`/`eLpNorm` form of Young's
convolution inequality (`Mathlib/Analysis/Convolution.lean` proves only
existence, `L¹⋆L¹→L¹` integrability, continuity, support), **no**
continuous-form Minkowski integral inequality, and **no**
Hausdorff–Young (`𝓕 : L^{4/3} → L⁴`).  But it *does* have:

  * the `n`-function Hölder inequality for the Lebesgue integral
    (`ENNReal.lintegral_prod_norm_pow_le`),
  * Tonelli (`lintegral_lintegral_swap`),
  * translation-invariance of `lintegral` (`lintegral_sub_left_eq_self`,
    `lintegral_sub_right_eq_self`),
  * the `L¹ ⋆ L¹ → L¹` machinery (`Integrable.ae_convolution_exists`).

We assemble Young's inequality at the *exact* exponent triple
`(4/3, 4/3, 2)` (`3/4 + 3/4 = 1/2 + 1`) from these pieces, via the
standard three-function Hölder split

    g(t)·g(x-t) = [g(t)^{4/3} g(x-t)^{4/3}]^{1/2}
                    · [g(t)^{4/3}]^{1/4} · [g(x-t)^{4/3}]^{1/4},

with Hölder weights `(1/2, 1/4, 1/4)` summing to `1`.  Squaring and
integrating in `x` (Tonelli) gives

    ∫⁻ (η⋆η)²  ≤  (∫⁻ η^{4/3})³  <  ⊤,

i.e. `η ⋆ η ∈ L²`.  No `sorry`, no new axioms.

This file proves the abstract specialized Young inequality
`memLp_two_of_autoconv_of_memLp_fourThirds` and then the concrete
`K_arc δ ∈ L²` (`K_arc_memLp_two`).
-/

import Mathlib
import Sidon.MultiScale

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical ENNReal Convolution
open MeasureTheory

namespace Sidon.Constructor

open Sidon.MultiScale

noncomputable section

/-! ## The `ℝ≥0∞`-valued lift of a nonnegative function and its `L^{4/3}` mass -/

/-- The `ℝ≥0∞` mass `∫⁻ x, (ENNReal.ofReal (f x))^(4/3)`.  For `f = η_δ`
this is finite (that is exactly `η_δ ∈ L^{4/3}`). -/
private def fourThirdsMass (f : ℝ → ℝ) : ℝ≥0∞ :=
  ∫⁻ x, (ENNReal.ofReal (f x)) ^ (4 / 3 : ℝ) ∂volume

/-- For a measurable nonnegative `f ∈ L^{4/3}`, the `ℝ≥0∞` mass is finite. -/
private lemma fourThirdsMass_lt_top {f : ℝ → ℝ} (hmeas : Measurable f)
    (hnn : ∀ x, 0 ≤ f x) (hLp : MemLp f (4 / 3) volume) :
    fourThirdsMass f < ⊤ := by
  -- `MemLp f (4/3)` means `eLpNorm f (4/3) < ⊤`, i.e. `∫⁻ ‖f‖ₑ^(4/3) < ⊤`.
  have hp0 : (4 / 3 : ℝ≥0∞) ≠ 0 := by norm_num
  have hptop : (4 / 3 : ℝ≥0∞) ≠ ∞ := by finiteness
  have h := (eLpNorm_lt_top_iff_lintegral_rpow_enorm_lt_top hp0 hptop).1 hLp.2
  -- `(4/3 : ℝ≥0∞).toReal = 4/3`, and for `f ≥ 0`, `‖f x‖ₑ = ENNReal.ofReal (f x)`.
  have htoReal : (4 / 3 : ℝ≥0∞).toReal = (4 / 3 : ℝ) := by
    rw [show (4 / 3 : ℝ≥0∞) = ENNReal.ofReal (4 / 3) by
      rw [ENNReal.ofReal_div_of_pos (by norm_num)]; norm_num]
    rw [ENNReal.toReal_ofReal (by norm_num)]
  rw [htoReal] at h
  have hcongr : (fun x => ‖f x‖ₑ ^ (4 / 3 : ℝ))
      = (fun x => (ENNReal.ofReal (f x)) ^ (4 / 3 : ℝ)) := by
    funext x
    rw [Real.enorm_eq_ofReal (hnn x)]
  rw [hcongr] at h
  exact h

/-! ## The three-function Hölder split, in `ℝ≥0∞` -/

/-- **Pointwise (in `x`) Hölder bound.**  For a measurable nonnegative `f`,
writing `g = ENNReal.ofReal ∘ f`, the squared convolution mass
`(∫⁻ t, g t · g (x-t))²` is bounded by `(G ⋆ G)(x) · M`, where
`G = g^{4/3}`, `M = ∫⁻ g^{4/3}`, and `(G⋆G)(x) = ∫⁻ t, G t · G (x-t)`.

This is the heart of Young at `(4/3, 4/3, 2)`: the three-function Hölder
split with weights `(1/2, 1/4, 1/4)`. -/
private lemma sq_convMass_le (f : ℝ → ℝ) (hmeas : Measurable f) (x : ℝ) :
    (∫⁻ t, ENNReal.ofReal (f t) * ENNReal.ofReal (f (x - t)) ∂volume) ^ 2
      ≤ (∫⁻ t, (ENNReal.ofReal (f t)) ^ (4 / 3 : ℝ)
                * (ENNReal.ofReal (f (x - t))) ^ (4 / 3 : ℝ) ∂volume)
        * fourThirdsMass f := by
  classical
  set g : ℝ → ℝ≥0∞ := fun y => ENNReal.ofReal (f y) with hg_def
  have hg_meas : Measurable g := (ENNReal.measurable_ofReal.comp hmeas)
  -- The three functions of the split, as functions of `t` (with `x` fixed):
  --   a t = g t ^ (4/3) * g (x - t) ^ (4/3),  weight 1/2
  --   b t = g t ^ (4/3),                       weight 1/4
  --   c t = g (x - t) ^ (4/3),                 weight 1/4
  -- Identity: g t * g (x-t) = a t ^ (1/2) * b t ^ (1/4) * c t ^ (1/4).
  set a : ℝ → ℝ≥0∞ := fun t => g t ^ (4 / 3 : ℝ) * g (x - t) ^ (4 / 3 : ℝ) with ha_def
  set b : ℝ → ℝ≥0∞ := fun t => g t ^ (4 / 3 : ℝ) with hb_def
  set c : ℝ → ℝ≥0∞ := fun t => g (x - t) ^ (4 / 3 : ℝ) with hc_def
  -- Index the three functions by `Fin 3`.
  set F : Fin 3 → ℝ → ℝ≥0∞ := ![a, b, c] with hF_def
  set p : Fin 3 → ℝ := ![1/2, 1/4, 1/4] with hp_def
  -- Pointwise: g t * g (x-t) = ∏ i, F i t ^ p i.
  have hpoint : ∀ t, g t * g (x - t) = ∏ i, F i t ^ p i := by
    intro t
    rw [Fin.prod_univ_three]
    simp only [hF_def, hp_def, ha_def, hb_def, hc_def, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]
    -- RHS = (g t^(4/3) g (x-t)^(4/3))^(1/2) * (g t^(4/3))^(1/4) * (g (x-t)^(4/3))^(1/4)
    rw [ENNReal.mul_rpow_of_nonneg _ _ (by norm_num : (0:ℝ) ≤ 1/2)]
    -- collapse iterated rpows: (g·^(4/3))^q = g·^(4/3*q)
    rw [← ENNReal.rpow_mul (g t) (4/3) (1/2), ← ENNReal.rpow_mul (g (x - t)) (4/3) (1/2),
        ← ENNReal.rpow_mul (g t) (4/3) (1/4), ← ENNReal.rpow_mul (g (x - t)) (4/3) (1/4)]
    -- regroup the two `g t` powers and the two `g (x-t)` powers, then add exponents
    rw [show (g t ^ (4/3 * (1/2) : ℝ) * g (x - t) ^ (4/3 * (1/2) : ℝ)) * g t ^ (4/3 * (1/4) : ℝ)
              * g (x - t) ^ (4/3 * (1/4) : ℝ)
            = (g t ^ (4/3 * (1/2) : ℝ) * g t ^ (4/3 * (1/4) : ℝ))
              * (g (x - t) ^ (4/3 * (1/2) : ℝ) * g (x - t) ^ (4/3 * (1/4) : ℝ)) from by ring]
    rw [← ENNReal.rpow_add_of_nonneg _ _ (by norm_num) (by norm_num),
        ← ENNReal.rpow_add_of_nonneg _ _ (by norm_num) (by norm_num)]
    norm_num
  -- Apply the n-function Hölder inequality.
  have hsum : ∑ i, p i = 1 := by
    rw [Fin.sum_univ_three]; simp only [hp_def, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]; norm_num
  have hp_nn : ∀ i ∈ Finset.univ, 0 ≤ p i := by
    intro i _
    fin_cases i <;> norm_num [hp_def]
  have hF_meas : ∀ i ∈ Finset.univ, AEMeasurable (F i) volume := by
    intro i _
    fin_cases i <;>
      simp only [hF_def, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
        Matrix.cons_val_two, Matrix.tail_cons, ha_def, hb_def, hc_def]
    · exact ((hg_meas.pow_const _).mul
        ((hg_meas.comp (measurable_const.sub measurable_id)).pow_const _)).aemeasurable
    · exact (hg_meas.pow_const _).aemeasurable
    · exact ((hg_meas.comp (measurable_const.sub measurable_id)).pow_const _).aemeasurable
  have hHolder := ENNReal.lintegral_prod_norm_pow_le (μ := volume) Finset.univ hF_meas hsum hp_nn
  -- Rewrite the LHS integrand back to `g t * g (x-t)`.
  have hLHS : ∫⁻ t, ∏ i, F i t ^ p i ∂volume
      = ∫⁻ t, g t * g (x - t) ∂volume := by
    refine lintegral_congr fun t => ?_
    exact (hpoint t).symm
  rw [hLHS] at hHolder
  rw [Fin.prod_univ_three] at hHolder
  simp only [hF_def, hp_def, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.tail_cons] at hHolder
  -- `hHolder : (∫⁻ t, g t * g(x-t)) ≤ (∫⁻ a)^(1/2) * (∫⁻ b)^(1/4) * (∫⁻ c)^(1/4)`.
  -- Square both sides.
  have hsq := ENNReal.rpow_le_rpow hHolder (by norm_num : (0:ℝ) ≤ 2)
  rw [← ENNReal.rpow_natCast _ 2]
  -- Compute the RHS square.
  have hRHS : ((∫⁻ t, a t ∂volume) ^ (1/2 : ℝ) * (∫⁻ t, b t ∂volume) ^ (1/4 : ℝ)
                * (∫⁻ t, c t ∂volume) ^ (1/4 : ℝ)) ^ (2 : ℝ)
      = (∫⁻ t, a t ∂volume) * ((∫⁻ t, b t ∂volume) ^ (1/2 : ℝ) * (∫⁻ t, c t ∂volume) ^ (1/2 : ℝ)) := by
    rw [ENNReal.mul_rpow_of_nonneg _ _ (by norm_num : (0:ℝ) ≤ 2)]
    rw [ENNReal.mul_rpow_of_nonneg _ _ (by norm_num : (0:ℝ) ≤ 2)]
    rw [← ENNReal.rpow_mul, ← ENNReal.rpow_mul, ← ENNReal.rpow_mul]
    norm_num
    ring
  -- Now relate `∫⁻ b` and `∫⁻ c` to `fourThirdsMass f` and bound.
  have hb_eq : (∫⁻ t, b t ∂volume) = fourThirdsMass f := rfl
  have hc_eq : (∫⁻ t, c t ∂volume) = fourThirdsMass f := by
    rw [hc_def, fourThirdsMass]
    -- `∫⁻ t, g (x - t)^(4/3) = ∫⁻ t, g t^(4/3)` by translation invariance.
    have := lintegral_sub_left_eq_self (μ := volume) (fun y => g y ^ (4 / 3 : ℝ)) x
    simpa [hg_def] using this
  calc (∫⁻ t, g t * g (x - t) ∂volume) ^ (2 : ℝ)
      ≤ ((∫⁻ t, a t ∂volume) ^ (1/2 : ℝ) * (∫⁻ t, b t ∂volume) ^ (1/4 : ℝ)
            * (∫⁻ t, c t ∂volume) ^ (1/4 : ℝ)) ^ (2 : ℝ) := hsq
    _ = (∫⁻ t, a t ∂volume)
          * ((∫⁻ t, b t ∂volume) ^ (1/2 : ℝ) * (∫⁻ t, c t ∂volume) ^ (1/2 : ℝ)) := hRHS
    _ = (∫⁻ t, a t ∂volume) * fourThirdsMass f := by
          rw [hb_eq, hc_eq, ← ENNReal.rpow_add_of_nonneg _ _ (by norm_num) (by norm_num)]
          norm_num

/-! ## Tonelli: the total mass of the `L¹` autoconvolution of `g^{4/3}` -/

/-- `∫⁻ x, (∫⁻ t, G t · G (x-t)) = (∫⁻ G)²` for measurable `G : ℝ → ℝ≥0∞`.
(Tonelli + translation-invariance of `lintegral`.) -/
private lemma lintegral_convMass_eq_sq {G : ℝ → ℝ≥0∞} (hG : Measurable G) :
    (∫⁻ x, ∫⁻ t, G t * G (x - t) ∂volume ∂volume) = (∫⁻ y, G y ∂volume) ^ 2 := by
  -- Swap the order of integration (`x` outer, `t` inner) → (`t` outer, `x` inner).
  rw [lintegral_lintegral_swap]
  · -- `∫⁻ t, G t · (∫⁻ x, G (x-t)) = ∫⁻ t, G t · (∫⁻ G) = (∫⁻ G)·(∫⁻ G)`.
    have hinner : ∀ t : ℝ, (∫⁻ x, G t * G (x - t) ∂volume) = G t * ∫⁻ y, G y ∂volume := by
      intro t
      have hmeas_shift : Measurable (fun x : ℝ => G (x - t)) :=
        hG.comp (measurable_id.sub_const t)
      have := lintegral_const_mul (μ := volume) (G t) hmeas_shift
      rw [this]
      congr 1
      exact lintegral_sub_right_eq_self G t
    rw [lintegral_congr hinner, lintegral_mul_const _ hG, sq]
  · -- joint measurability of `(x,t) ↦ G t · G (x-t)`
    apply Measurable.aemeasurable
    exact (hG.comp measurable_snd).mul (hG.comp (measurable_fst.sub measurable_snd))

/-! ## The bridge: `ofReal (η⋆η)(x) = ∫⁻ t, ofReal(η t)·ofReal(η(x-t))` a.e. -/

/-- For integrable nonnegative `f`, the autoconvolution value
`(f ⋆ f)(x) = ∫ t, f t · f (x-t)` lifts to the `lintegral` of the
`ℝ≥0∞`-products, for a.e. `x` (the set where the convolution integrand is
integrable). -/
private lemma ofReal_autoconv_eq_lintegral_ae (f : ℝ → ℝ)
    (hf : Integrable f volume) (hnn : ∀ x, 0 ≤ f x) :
    ∀ᵐ x ∂volume,
      ENNReal.ofReal
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume x)
        = ∫⁻ t, ENNReal.ofReal (f t) * ENNReal.ofReal (f (x - t)) ∂volume := by
  filter_upwards [hf.ae_convolution_exists (ContinuousLinearMap.mul ℝ ℝ) hf] with x hx
  -- `hx : ConvolutionExistsAt f f x (mul ℝ ℝ) volume`, i.e. integrability of the integrand.
  rw [MeasureTheory.convolution_def]
  simp only [ContinuousLinearMap.mul_apply']
  -- `ConvolutionExistsAt f f x (mul ℝ ℝ) = Integrable (fun t => f t * f (x - t))`.
  have hint : Integrable (fun t => f t * f (x - t)) volume := by
    have := hx
    rw [ConvolutionExistsAt] at this
    simpa [ContinuousLinearMap.mul_apply'] using this
  -- `ofReal (∫ ...) = ∫⁻ ofReal (...)` since the integrand is integrable & nonneg.
  rw [ofReal_integral_eq_lintegral_ofReal hint
        (Filter.Eventually.of_forall fun t => mul_nonneg (hnn t) (hnn (x - t)))]
  refine lintegral_congr fun t => ?_
  rw [ENNReal.ofReal_mul (hnn t)]

/-! ## Young's inequality `L^{4/3} ⋆ L^{4/3} → L²`, specialized -/

/-- **Young's convolution inequality at `(4/3, 4/3, 2)`, autoconvolution
form.**  If `f : ℝ → ℝ` is measurable, nonnegative, integrable, and in
`L^{4/3}`, then its autoconvolution `f ⋆ f` lies in `L²`.

(`3/4 + 3/4 = 1/2 + 1`, the Young exponent relation.) -/
theorem memLp_two_of_autoconv_of_memLp_fourThirds (f : ℝ → ℝ)
    (hmeas : Measurable f) (hnn : ∀ x, 0 ≤ f x)
    (hf : Integrable f volume) (hLp : MemLp f (4 / 3) volume) :
    MemLp (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) 2 volume := by
  set K := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume with hK_def
  -- The convolution is `AEStronglyMeasurable` (it is integrable, by `L¹ ⋆ L¹ → L¹`).
  have hKmeas : AEStronglyMeasurable K volume :=
    (hf.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf).aestronglyMeasurable
  refine ⟨hKmeas, ?_⟩
  -- Reduce `eLpNorm K 2 < ⊤` to `∫⁻ ‖K x‖ₑ² < ⊤`.
  rw [eLpNorm_lt_top_iff_lintegral_rpow_enorm_lt_top (by norm_num) (by finiteness)]
  -- `(2 : ℝ≥0∞).toReal = 2`.
  have htwo : (2 : ℝ≥0∞).toReal = (2 : ℝ) := by norm_num
  rw [htwo]
  set M : ℝ≥0∞ := fourThirdsMass f with hM_def
  have hM_lt : M < ⊤ := fourThirdsMass_lt_top hmeas hnn hLp
  -- `G := (ofReal ∘ f)^(4/3)`, measurable, with `∫⁻ G = M`.
  set G : ℝ → ℝ≥0∞ := fun y => (ENNReal.ofReal (f y)) ^ (4 / 3 : ℝ) with hG_def
  have hG_meas : Measurable G := (ENNReal.measurable_ofReal.comp hmeas).pow_const _
  have hG_mass : (∫⁻ y, G y ∂volume) = M := rfl
  -- Pointwise a.e. bound: `‖K x‖ₑ² ≤ (G ⋆ G)(x) · M`.
  have hbound : ∀ᵐ x ∂volume, ‖K x‖ₑ ^ (2 : ℝ)
      ≤ (∫⁻ t, G t * G (x - t) ∂volume) * M := by
    filter_upwards [ofReal_autoconv_eq_lintegral_ae f hf hnn] with x hx
    -- `K x ≥ 0`, so `‖K x‖ₑ = ofReal (K x)`.
    have hKx_nn : 0 ≤ K x := by
      rw [hK_def]; exact convolution_nonneg hnn hnn x
    rw [Real.enorm_eq_ofReal hKx_nn, ENNReal.rpow_two, hx]
    -- Now use the three-function Hölder bound.
    have := sq_convMass_le f hmeas x
    rw [← ENNReal.rpow_two] at this ⊢
    -- rewrite the `a`-integral target `∫⁻ a` as `∫⁻ G t · G (x-t)`.
    calc (∫⁻ t, ENNReal.ofReal (f t) * ENNReal.ofReal (f (x - t)) ∂volume) ^ (2 : ℝ)
        ≤ (∫⁻ t, (ENNReal.ofReal (f t)) ^ (4 / 3 : ℝ)
                  * (ENNReal.ofReal (f (x - t))) ^ (4 / 3 : ℝ) ∂volume) * fourThirdsMass f := this
      _ = (∫⁻ t, G t * G (x - t) ∂volume) * M := by rw [hG_def, hM_def]
  -- Integrate the bound over `x`.
  calc ∫⁻ x, ‖K x‖ₑ ^ (2 : ℝ) ∂volume
      ≤ ∫⁻ x, (∫⁻ t, G t * G (x - t) ∂volume) * M ∂volume :=
        lintegral_mono_ae hbound
    _ = (∫⁻ x, ∫⁻ t, G t * G (x - t) ∂volume ∂volume) * M := by
        rw [lintegral_mul_const]
        exact (Measurable.lintegral_prod_right (by
          have : Measurable (fun p : ℝ × ℝ => G p.2 * G (p.1 - p.2)) :=
            (hG_meas.comp measurable_snd).mul (hG_meas.comp (measurable_fst.sub measurable_snd))
          exact this))
    _ = (∫⁻ y, G y ∂volume) ^ 2 * M := by rw [lintegral_convMass_eq_sq hG_meas]
    _ = M ^ 2 * M := by rw [hG_mass]
    _ < ⊤ := by
        apply ENNReal.mul_lt_top
        · exact ENNReal.pow_lt_top hM_lt
        · exact hM_lt

end

section AxiomAudit
#guard_msgs (drop info) in
#print axioms memLp_two_of_autoconv_of_memLp_fourThirds
end AxiomAudit

end Sidon.Constructor
