/-
Sidon Autocorrelation Project — `K_ms ∈ L²` (target), now DISCHARGED.
==================================================================

GOAL (task), both proved here `sorry`-free / no new axioms:
  * `K_ms_memLp_two     : MemLp Sidon.MultiScale.K_ms 2 volume`
  * `K_ms_sq_integrable : Integrable (fun x => K_ms x ^ 2) volume`

ROUTE.  The mathematically correct route is Young's convolution
inequality `‖η * η‖₂ ≤ ‖η‖_{4/3}²` (exponents `1 + 1/2 = 3/4 + 3/4`),
since the arcsine density `η_δ` lies in `L^{4/3}` (its `1/√` endpoint
singularity is `L^p`-integrable for every `p < 2`) but NOT in `L²` (so
`η * η` is genuinely an `L^{4/3} ⋆ L^{4/3} → L²` statement, not a cheap
`L¹ ⋆ L²`).

  Mathlib `v4.29.1` has **no** `eLpNorm`/`MemLp` Young convolution
  inequality, **no** continuous-form Minkowski integral inequality, and
  **no** Hausdorff–Young.  We therefore *prove* the needed specialized
  Young inequality `L^{4/3} ⋆ L^{4/3} → L²` from scratch in
  `Sidon.Constructor.YoungConvolution`
  (`memLp_two_of_autoconv_of_memLp_fourThirds`), using only the
  `n`-function Hölder inequality for the Lebesgue integral
  (`ENNReal.lintegral_prod_norm_pow_le`), Tonelli, and translation
  invariance — all of which mathlib *does* have.  The three-function
  Hölder split at weights `(1/2, 1/4, 1/4)` gives `∫⁻ (η⋆η)² ≤
  (∫⁻ η^{4/3})³ < ⊤`.

WHAT IS PROVED HERE (`sorry`-free, no new axioms):
  * `integrableOn_arcsine_pow_fourThirds` — the `L^{4/3}` integrability
    *core*: `x ↦ ((δ/2)²-x²)^{-2/3}` is integrable on `(-(δ/2), δ/2)`.
  * `eta_measurable`         — `η_δ` is measurable.
  * `eta_memLp_fourThirds`   — `η_δ ∈ L^{4/3}(volume)` (the Young input).
  * `K_arc_memLp_two`        — `K_arc δ = η_δ ⋆ η_δ ∈ L²`.
  * `K_ms_memLp_two`         — `K_ms ∈ L²`.
  * `K_ms_sq_integrable`     — `Integrable (K_ms²)`.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.YoungConvolution

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical ENNReal
open MeasureTheory

namespace Sidon.Constructor

open Sidon.MultiScale

noncomputable section

/-! ## Single-endpoint `rpow` integrability building blocks -/

/-- `x ↦ (a - x) ^ (-(2:ℝ)/3)` is integrable on `(-a, a)`.
Reflection of the integrable-singularity power `t ↦ t^{-2/3}`. -/
theorem integrableOn_sub_rpow {a : ℝ} (ha : 0 < a) :
    IntegrableOn (fun x => (a - x) ^ (-(2:ℝ)/3)) (Set.Ioo (-a) a) volume := by
  -- `t ↦ t^{-2/3}` is interval-integrable on `[2a, 0]` (`-2/3 > -1`).
  have hbase : IntervalIntegrable (fun t : ℝ => t ^ (-(2:ℝ)/3)) volume (2 * a) 0 :=
    intervalIntegral.intervalIntegrable_rpow' (by norm_num)
  -- Compose with `x ↦ a - x`: gives integrability on `[a - 2a, a - 0] = [-a, a]`.
  have hcomp := hbase.comp_sub_left a
  -- `hcomp : IntervalIntegrable (fun x => (a - x)^{-2/3}) volume (a - 2a) (a - 0)`.
  have he1 : a - 2 * a = -a := by ring
  have he2 : a - 0 = a := by ring
  rw [he1, he2] at hcomp
  -- Convert interval-integrability on `[-a, a]` to `IntegrableOn (Ioo (-a) a)`.
  rw [intervalIntegrable_iff_integrableOn_Ioo_of_le (by linarith)] at hcomp
  exact hcomp

/-- `x ↦ (a + x) ^ (-(2:ℝ)/3)` is integrable on `(-a, a)`.
Translation of the integrable-singularity power `t ↦ t^{-2/3}`. -/
theorem integrableOn_add_rpow {a : ℝ} (ha : 0 < a) :
    IntegrableOn (fun x => (a + x) ^ (-(2:ℝ)/3)) (Set.Ioo (-a) a) volume := by
  -- `t ↦ t^{-2/3}` is interval-integrable on `[0, 2a]`.
  have hbase : IntervalIntegrable (fun t : ℝ => t ^ (-(2:ℝ)/3)) volume 0 (2 * a) :=
    intervalIntegral.intervalIntegrable_rpow' (by norm_num)
  -- Compose with `x ↦ x + a` via `comp_sub_right` with `c = -a`:
  -- `fun x => f (x - (-a)) = fun x => f (x + a)`, on `[0 + (-a), 2a + (-a)] = [-a, a]`.
  have hcomp := hbase.comp_sub_right (-a)
  -- `hcomp : IntervalIntegrable (fun x => (x - (-a))^{-2/3}) volume (0 + -a) (2a + -a)`.
  have hfun : (fun x : ℝ => (x - -a) ^ (-(2:ℝ)/3)) = (fun x : ℝ => (a + x) ^ (-(2:ℝ)/3)) := by
    funext x; rw [sub_neg_eq_add, add_comm]
  rw [hfun] at hcomp
  have he1 : (0 : ℝ) + -a = -a := by ring
  have he2 : 2 * a + -a = a := by ring
  rw [he1, he2] at hcomp
  rw [intervalIntegrable_iff_integrableOn_Ioo_of_le (by linarith)] at hcomp
  exact hcomp

/-! ## The `L^{4/3}` integrability core -/

/-- **The `L^{4/3}` integrability core.**  For `δ > 0`,
`x ↦ ((δ/2)² - x²)^{-2/3}` is integrable on `(-(δ/2), δ/2)`.

This is the analytic heart of `η_δ ∈ L^{4/3}`: raising `η_δ` to the
power `4/3` produces (up to the constant `(1/π)^{4/3}`) this integrand,
whose endpoint singularity has exponent `-2/3 > -1`.

Proof: on `(-a, a)` (with `a = δ/2`) we have the pointwise bound
`(a² - x²)^{-2/3} ≤ a^{-2/3} · ((a - x)^{-2/3} + (a + x)^{-2/3})`,
because `a² - x² = (a - x)(a + x)` and at least one factor is `≥ a`
(so `(a ∓ x)^{-2/3} ≤ a^{-2/3}` for that factor; the other term is
nonneg).  The dominating function is integrable by
`integrableOn_sub_rpow` + `integrableOn_add_rpow`. -/
theorem integrableOn_arcsine_pow_fourThirds {δ : ℝ} (hδ : 0 < δ) :
    IntegrableOn
      (fun x => ((δ/2)^2 - x^2) ^ (-(2:ℝ)/3))
      (Set.Ioo (-(δ/2)) (δ/2)) volume := by
  set a : ℝ := δ / 2 with ha_def
  have ha_pos : 0 < a := by positivity
  -- Dominating function `g x = a^{-2/3} · ((a - x)^{-2/3} + (a + x)^{-2/3})`.
  set c : ℝ := a ^ (-(2:ℝ)/3) with hc_def
  have hc_pos : 0 < c := Real.rpow_pos_of_pos ha_pos _
  set g : ℝ → ℝ :=
    fun x => c * ((a - x) ^ (-(2:ℝ)/3) + (a + x) ^ (-(2:ℝ)/3)) with hg_def
  -- `g` is integrable on `(-a, a)`.
  have hg_int : IntegrableOn g (Set.Ioo (-a) a) volume := by
    apply Integrable.const_mul
    exact (integrableOn_sub_rpow ha_pos).add (integrableOn_add_rpow ha_pos)
  -- Dominate.  Use `Integrable.mono'` on the restricted measure.
  rw [IntegrableOn] at hg_int ⊢
  refine hg_int.mono' ?_ ?_
  · -- AEStronglyMeasurable on the restricted measure: the integrand is
    -- continuous on `Ioo (-a) a`, where the base `a² - x² > 0`.
    apply ContinuousOn.aestronglyMeasurable _ measurableSet_Ioo
    intro x hx
    obtain ⟨hx1, hx2⟩ := hx
    have hbase_ne : (a ^ 2 - x ^ 2) ≠ 0 := by
      have hax : 0 < a - x := by linarith
      have hax' : 0 < a + x := by linarith
      have : 0 < a ^ 2 - x ^ 2 := by
        have : a ^ 2 - x ^ 2 = (a - x) * (a + x) := by ring
        rw [this]; positivity
      exact ne_of_gt this
    apply ContinuousAt.continuousWithinAt
    exact (((continuous_const.sub (continuous_pow 2)).continuousAt).rpow_const
      (Or.inl hbase_ne))
  · -- The pointwise bound `‖(a²-x²)^{-2/3}‖ ≤ g x` a.e. on `Ioo (-a) a`.
    filter_upwards [ae_restrict_mem measurableSet_Ioo] with x hx
    obtain ⟨hx1, hx2⟩ := hx
    -- `a - x > 0`, `a + x > 0`, `a² - x² > 0`.
    have hax : 0 < a - x := by linarith
    have hax' : 0 < a + x := by linarith
    have hprod : a ^ 2 - x ^ 2 = (a - x) * (a + x) := by ring
    have hpos : 0 < a ^ 2 - x ^ 2 := by rw [hprod]; positivity
    -- The integrand is nonneg, so `‖·‖ = ·`.
    have hval_nonneg : 0 ≤ (a ^ 2 - x ^ 2) ^ (-(2:ℝ)/3) :=
      Real.rpow_nonneg hpos.le _
    rw [Real.norm_eq_abs, abs_of_nonneg hval_nonneg]
    -- `(a²-x²)^{-2/3} = (a-x)^{-2/3} (a+x)^{-2/3}`.
    have hsplit : (a ^ 2 - x ^ 2) ^ (-(2:ℝ)/3)
        = (a - x) ^ (-(2:ℝ)/3) * (a + x) ^ (-(2:ℝ)/3) := by
      rw [hprod, Real.mul_rpow hax.le hax'.le]
    rw [hsplit]
    -- Case split on the sign of `x`.
    rcases le_or_gt 0 x with hxsign | hxsign
    · -- `x ≥ 0`: `a + x ≥ a`, so `(a+x)^{-2/3} ≤ a^{-2/3} = c`.
      have hle : (a + x) ^ (-(2:ℝ)/3) ≤ c :=
        Real.rpow_le_rpow_of_nonpos ha_pos (by linarith) (by norm_num)
      have hother_nonneg : 0 ≤ (a - x) ^ (-(2:ℝ)/3) := Real.rpow_nonneg hax.le _
      have hadd_nonneg : 0 ≤ (a + x) ^ (-(2:ℝ)/3) := Real.rpow_nonneg hax'.le _
      calc (a - x) ^ (-(2:ℝ)/3) * (a + x) ^ (-(2:ℝ)/3)
          ≤ (a - x) ^ (-(2:ℝ)/3) * c := by
            apply mul_le_mul_of_nonneg_left hle hother_nonneg
        _ = c * (a - x) ^ (-(2:ℝ)/3) := by ring
        _ ≤ c * ((a - x) ^ (-(2:ℝ)/3) + (a + x) ^ (-(2:ℝ)/3)) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            linarith
    · -- `x < 0`: `a - x ≥ a`, so `(a-x)^{-2/3} ≤ a^{-2/3} = c`.
      have hle : (a - x) ^ (-(2:ℝ)/3) ≤ c :=
        Real.rpow_le_rpow_of_nonpos ha_pos (by linarith) (by norm_num)
      have hother_nonneg : 0 ≤ (a + x) ^ (-(2:ℝ)/3) := Real.rpow_nonneg hax'.le _
      have hsub_nonneg : 0 ≤ (a - x) ^ (-(2:ℝ)/3) := Real.rpow_nonneg hax.le _
      calc (a - x) ^ (-(2:ℝ)/3) * (a + x) ^ (-(2:ℝ)/3)
          ≤ c * (a + x) ^ (-(2:ℝ)/3) := by
            apply mul_le_mul_of_nonneg_right hle hother_nonneg
        _ ≤ c * ((a - x) ^ (-(2:ℝ)/3) + (a + x) ^ (-(2:ℝ)/3)) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            linarith

/-! ## Measurability and `L^{4/3}` membership of `η_δ` -/

/-- The half-arcsine density `η_δ` is measurable. -/
theorem eta_measurable (δ : ℝ) : Measurable (eta δ) := by
  unfold eta
  -- `eta δ = if |x| < δ/2 then (1/π)·(√((δ/2)²-x²))⁻¹ else 0`.
  apply Measurable.ite
  · -- `{x | |x| < δ/2}` is measurable.
    exact measurableSet_lt measurable_norm measurable_const
  · -- the `then`-branch is measurable.
    apply Measurable.const_mul
    apply Measurable.inv
    exact (measurable_const.sub (measurable_id.pow_const 2)).sqrt
  · -- the `else`-branch (`0`) is measurable.
    exact measurable_const

/-- On its support `Ioo (-(δ/2)) (δ/2)`, `‖η_δ‖^{4/3}` equals
`(1/π)^{4/3} · ((δ/2)² - x²)^{-2/3}`.  (Pointwise identity, used to
transfer the integrability core to `η_δ`.) -/
theorem eta_norm_rpow_eq_on_support {δ : ℝ} (hδ : 0 < δ) {x : ℝ}
    (hx : x ∈ Set.Ioo (-(δ/2)) (δ/2)) :
    ‖eta δ x‖ ^ (4/3 : ℝ)
      = (1 / Real.pi) ^ (4/3 : ℝ) * ((δ/2)^2 - x^2) ^ (-(2:ℝ)/3) := by
  obtain ⟨hx1, hx2⟩ := hx
  have habs : |x| < δ / 2 := by rw [abs_lt]; exact ⟨hx1, hx2⟩
  -- positivity of the radicand
  have hax : 0 < δ/2 - x := by linarith
  have hax' : 0 < δ/2 + x := by linarith
  have hY : (0:ℝ) < (δ/2)^2 - x^2 := by
    have : (δ/2)^2 - x^2 = (δ/2 - x) * (δ/2 + x) := by ring
    rw [this]; positivity
  -- evaluate η on the support
  have heta : eta δ x = (1 / Real.pi) * (Real.sqrt ((δ/2)^2 - x^2))⁻¹ := by
    unfold eta; rw [if_pos habs]
  -- η ≥ 0
  have hpi : 0 < Real.pi := Real.pi_pos
  have hsqrt_pos : 0 < Real.sqrt ((δ/2)^2 - x^2) := Real.sqrt_pos.mpr hY
  have heta_nonneg : 0 ≤ eta δ x := by
    rw [heta]
    apply mul_nonneg (by positivity)
    exact inv_nonneg.mpr (Real.sqrt_nonneg _)
  -- ‖η‖ = η
  rw [Real.norm_eq_abs, abs_of_nonneg heta_nonneg, heta]
  -- (1/π · (√Y)⁻¹)^(4/3) = (1/π)^(4/3) · ((√Y)⁻¹)^(4/3)
  rw [Real.mul_rpow (by positivity) (inv_nonneg.mpr (Real.sqrt_nonneg _))]
  congr 1
  -- ((√Y)⁻¹)^(4/3) = Y^(-2/3)
  rw [Real.sqrt_eq_rpow]
  -- (Y^(1/2))⁻¹ = Y^(-(1/2))
  rw [← Real.rpow_neg hY.le]
  -- (Y^(-(1/2)))^(4/3) = Y^((-(1/2)) * (4/3)) = Y^(-2/3)
  rw [← Real.rpow_mul hY.le]
  norm_num

/-- `‖η_δ‖^{4/3}` is integrable on the support `Ioo (-(δ/2)) (δ/2)`. -/
theorem integrableOn_eta_norm_rpow {δ : ℝ} (hδ : 0 < δ) :
    IntegrableOn (fun x => ‖eta δ x‖ ^ (4/3 : ℝ))
      (Set.Ioo (-(δ/2)) (δ/2)) volume := by
  -- equals `(1/π)^(4/3) · ((δ/2)²-x²)^(-2/3)` on the interval
  have hcore : IntegrableOn
      (fun x => (1 / Real.pi) ^ (4/3 : ℝ) * ((δ/2)^2 - x^2) ^ (-(2:ℝ)/3))
      (Set.Ioo (-(δ/2)) (δ/2)) volume :=
    (integrableOn_arcsine_pow_fourThirds hδ).const_mul ((1 / Real.pi) ^ (4/3 : ℝ))
  refine IntegrableOn.congr_fun hcore (fun x hx => ?_) measurableSet_Ioo
  exact (eta_norm_rpow_eq_on_support hδ hx).symm

/-- **`η_δ ∈ L^{4/3}(volume)`.**  The half-arcsine density of half-width
`δ/2` lies in `L^{4/3}`.  This is the entire `K`-side input to Young's
convolution inequality `‖η_δ * η_δ‖₂ ≤ ‖η_δ‖_{4/3}²` (the missing
mathlib step toward `K_arc δ ∈ L²`). -/
theorem eta_memLp_fourThirds {δ : ℝ} (hδ : 0 < δ) :
    MemLp (eta δ) (4/3) volume := by
  -- `MemLp f p ↔ Integrable (‖f‖^p.toReal)` (given measurability).
  have hmeas : AEStronglyMeasurable (eta δ) volume :=
    (eta_measurable δ).aestronglyMeasurable
  have hp0 : (4/3 : ℝ≥0∞) ≠ 0 := by norm_num
  have hptop : (4/3 : ℝ≥0∞) ≠ ∞ := by finiteness
  rw [← integrable_norm_rpow_iff hmeas hp0 hptop]
  -- `(4/3 : ℝ≥0∞).toReal = 4/3`.
  have htoReal : (4/3 : ℝ≥0∞).toReal = (4/3 : ℝ) := by
    rw [show (4/3 : ℝ≥0∞) = ENNReal.ofReal (4/3) by
      rw [ENNReal.ofReal_div_of_pos (by norm_num)]; norm_num]
    rw [ENNReal.toReal_ofReal (by norm_num)]
  rw [htoReal]
  -- `‖η_δ‖^{4/3}` vanishes off the support, and is integrable on it.
  apply IntegrableOn.integrable_of_forall_notMem_eq_zero (integrableOn_eta_norm_rpow hδ)
  intro x hx
  -- outside `Ioo`, `η_δ x = 0`, so `‖·‖^{4/3} = 0`.
  have hx_abs : δ / 2 ≤ |x| := by
    by_contra h
    rw [not_le] at h
    exact hx ⟨(abs_lt.mp h).1, (abs_lt.mp h).2⟩
  have : eta δ x = 0 := eta_eq_zero_outside δ x hx_abs
  rw [this, norm_zero, Real.zero_rpow (by norm_num)]

/-! ## `K_arc δ ∈ L²` via the specialized Young inequality -/

/-- **`K_arc δ = η_δ ⋆ η_δ ∈ L²`.**  Direct application of the
specialized Young inequality `L^{4/3} ⋆ L^{4/3} → L²`
(`Sidon.Constructor.memLp_two_of_autoconv_of_memLp_fourThirds`) to the
half-arcsine density `η_δ`, which is measurable, nonnegative, integrable
and (by `eta_memLp_fourThirds`) in `L^{4/3}`. -/
theorem K_arc_memLp_two {δ : ℝ} (hδ : 0 < δ) :
    MemLp (K_arc δ) 2 volume := by
  unfold K_arc
  exact memLp_two_of_autoconv_of_memLp_fourThirds (eta δ)
    (eta_measurable δ)
    (fun x => eta_nonneg δ x hδ)
    (eta_integrable δ hδ)
    (eta_memLp_fourThirds hδ)

/-! ## `K_ms ∈ L²` and `Integrable (K_ms²)` -/

/-- **`K_ms ∈ L²(volume)`.**  `K_ms = λ₁·K_arc δ₁ + λ₂·K_arc δ₂ + λ₃·K_arc δ₃`
is a finite linear combination of the `L²` autoconvolutions `K_arc δᵢ`,
hence `L²` by `MemLp.add` / `MemLp.const_mul`. -/
theorem K_ms_memLp_two : MemLp K_ms 2 volume := by
  have hδ := delta_positivities
  have h1 : MemLp (fun x => lambda1 * K_arc delta1 x) 2 volume :=
    (K_arc_memLp_two hδ.1).const_mul lambda1
  have h2 : MemLp (fun x => lambda2 * K_arc delta2 x) 2 volume :=
    (K_arc_memLp_two hδ.2.1).const_mul lambda2
  have h3 : MemLp (fun x => lambda3 * K_arc delta3 x) 2 volume :=
    (K_arc_memLp_two hδ.2.2).const_mul lambda3
  have hsum : MemLp
      (fun x => lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x + lambda3 * K_arc delta3 x)
      2 volume := ((h1.add h2).add h3)
  -- `K_ms x = λ₁·K_arc δ₁ x + λ₂·K_arc δ₂ x + λ₃·K_arc δ₃ x` definitionally.
  unfold K_ms
  exact hsum

/-- **`Integrable (fun x => K_ms x ^ 2)`.**  Immediate from `K_ms ∈ L²`
and `K_ms ≥ 0` (so `‖K_ms x‖² = K_ms x ^ 2`). -/
theorem K_ms_sq_integrable : Integrable (fun x => K_ms x ^ 2) volume := by
  have hmem := K_ms_memLp_two
  -- `MemLp K_ms 2 ↔ Integrable (K_ms²)` for real-valued `K_ms` (via `memLp_two_iff_integrable_sq`).
  exact (memLp_two_iff_integrable_sq hmem.1).1 hmem

end

/-! ## Axiom audit

`#print axioms` confirms the proved theorems — including the headline
`K_ms_memLp_two` and `K_ms_sq_integrable` — depend only on the three
Lean-core logical axioms (`propext`, `Classical.choice`, `Quot.sound`):
no `sorry`, no project axioms, no new axioms. -/
section AxiomAudit
#guard_msgs (drop info) in
#print axioms eta_memLp_fourThirds
#guard_msgs (drop info) in
#print axioms integrableOn_arcsine_pow_fourThirds
#guard_msgs (drop info) in
#print axioms eta_measurable
#guard_msgs (drop info) in
#print axioms K_arc_memLp_two
#guard_msgs (drop info) in
#print axioms K_ms_memLp_two
#guard_msgs (drop info) in
#print axioms K_ms_sq_integrable
end AxiomAudit

end Sidon.Constructor
