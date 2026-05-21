/-
Sidon Autocorrelation Project — Period-`u` Poisson Sampling (MO 2009 Lemma 2.1, Step 4).
=========================================================================================

This file attacks the **single** genuinely-missing mathlib-shaped primitive
isolated in `Sidon.Constructor.MOLemma21` as the named `Prop`
`PeriodUPoissonSampling`: the period-`u` Poisson sampling identity

  `fourierCoeff (liftIoc u (-(u/2)) g_per) j = (1/u) · 𝓕g(j/u)`     (∀ j : ℤ)

for `g : ℝ → ℂ` an `L¹` function and `g_per(x) := ∑'_k g(x + k·u)` its
period-`u` periodisation.

================================================================================
## Mathematical statement and the route taken
================================================================================

mathlib's only Poisson-sampling lemma `Real.fourierCoeff_tsum_comp_add`
(`Mathlib/Analysis/Fourier/PoissonSummation.lean`) is hardcoded to **period 1**
and requires the base `f : C(ℝ,ℂ)` **continuous** with a locally-uniform
summability bound.  Our `g = (f*f) + autocorr f` is `L¹ ∩ L²` but **not**
continuous, and the period is `u = 0.638 ≠ 1`.  We overcome both via the
**measure-theoretic sum/integral swap** (`integral_tsum_of_summable_integral_norm`),
which needs only `L¹`-summability, never continuity:

  `fourierCoeff (liftIoc u (-(u/2)) g_per) j`
    `= (1/u) · ∫_{-u/2}^{u/2} 𝐞(-j·x/u) · g_per(x) dx`        (`fourierCoeff_eq_intervalIntegral`)
    `= (1/u) · ∫_{-u/2}^{u/2} ∑'_k 𝐞(-j·x/u) · g(x + k·u) dx`  (`liftIoc` agrees with `g_per` on the period)
    `= (1/u) · ∑'_k ∫_{-u/2}^{u/2} 𝐞(-j·x/u) · g(x + k·u) dx`  (`integral_tsum`, L¹-swap)
    `= (1/u) · ∑'_k ∫_{-u/2 + k·u}^{u/2 + k·u} 𝐞(-j·y/u) · g(y) dy`   (substitute `y = x + k·u`; the character is `u`-periodic so `𝐞(-j·(y-k·u)/u) = 𝐞(-j·y/u)`)
    `= (1/u) · ∫_ℝ 𝐞(-j·y/u) · g(y) dy`                       (the half-open intervals `(-u/2 + k·u, u/2 + k·u]` tile `ℝ`)
    `= (1/u) · 𝓕g(j/u)`.

The last reassembly is mathlib's
`MeasureTheory.Integrable.hasSum_intervalIntegral_comp_mul_add` /
`hasSum_intervalIntegral_comp_add_int` family (the integral over `ℝ` is the
sum of integrals over the tiling intervals, for an integrable function).

This is the **TRUE** period-`u` Poisson sampling identity (for the
*periodisation* `g_per`).  See the docstring of `period_u_poisson_sampling`
for the exact statement, and the note at the end of the file on why the
naive *truncation*-lift form (the Fourier coefficient of
`liftIoc u (-(u/2)) g` equal to `(1/u)·𝓕g(j/u)`) is FALSE when `supp g`
overflows one period, as ours does — only the periodisation form below is
sound, which is why this file supplies the hypothesis-free MO 2.1 core.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.BilinearParseval
import Sidon.TorusParseval
import Sidon.Constructor.MOLemma21

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical Pointwise

namespace Sidon.Constructor.PoissonSampling

open Sidon.MultiScale

noncomputable section

/-! ## The period-`u` periodisation of an `L¹` function -/

/-- The period-`u` periodisation `g_per(x) := ∑'_k g(x + k·u)`. -/
def periodization (u : ℝ) (g : ℝ → ℂ) : ℝ → ℂ :=
  fun x => ∑' k : ℤ, g (x + k • u)

/-- The lattice character `χ_j(x) := exp(-2πi·j·x/u)`, written so its
`u`-periodicity is `χ_j(x + k·u) = χ_j(x)` (`k : ℤ`). -/
def latticeChar (u : ℝ) (j : ℤ) (x : ℝ) : ℂ :=
  Complex.exp (↑(-2 * π * x * (j / u : ℝ)) * Complex.I)

/-- The character is invariant under shifts by `k·u` (`k : ℤ`): `χ_j(x + k·u) = χ_j(x)`. -/
theorem latticeChar_add_zsmul (u : ℝ) (hu : u ≠ 0) (j : ℤ) (k : ℤ) (x : ℝ) :
    latticeChar u j (x + k • u) = latticeChar u j x := by
  unfold latticeChar
  -- exp(α) = exp(β) when α - β ∈ 2πi·ℤ; here α - β = -2πi·(k·u)·(j/u) = -2πi·k·j.
  rw [Complex.exp_eq_exp_iff_exists_int]
  refine ⟨-(k * j), ?_⟩
  -- `(x + k•u)·(j/u) = x·(j/u) + k·j` since `u ≠ 0`.
  have hku : ((k • u : ℝ) : ℂ) = (k : ℂ) * (u : ℂ) := by
    push_cast [zsmul_eq_mul]; ring
  have huc : (u : ℂ) ≠ 0 := by exact_mod_cast hu
  push_cast [zsmul_eq_mul]
  field_simp
  ring

/-- The character has unit modulus: `‖latticeChar u j x‖ = 1`. -/
theorem latticeChar_norm_eq_one (u : ℝ) (j : ℤ) (x : ℝ) :
    ‖latticeChar u j x‖ = 1 := by
  unfold latticeChar
  rw [Complex.norm_exp]
  simp only [Complex.mul_re, Complex.I_re, Complex.I_im, Complex.ofReal_re,
    Complex.ofReal_im, mul_zero, mul_one, sub_zero, Real.exp_zero]

/-! ## The `L¹` summability of the translated family

For `g ∈ L¹(ℝ)`, the family `k ↦ ∫_{(-u/2,u/2]} ‖g(x + k·u)‖ dx` is summable
with sum `∫_ℝ ‖g‖`.  This is the Tonelli/`L¹` domination underlying the
sum/integral swap; no continuity is used.  We package it as the
`Summable (fun k ↦ ∫ x in (-u/2)..(u/2), ‖term k x‖)` hypothesis of
`integral_tsum_of_summable_integral_norm`. -/

variable {u : ℝ}

/-- Each translate `x ↦ g (x + k • u)` is integrable when `g` is. -/
theorem integrable_comp_add_zsmul (g : ℝ → ℂ) (hg : Integrable g volume) (k : ℤ) :
    Integrable (fun x => g (x + k • u)) volume := by
  have := hg.comp_add_right (k • u)
  simpa using this

/-- `∫ x in (-u/2)..(u/2), g(x + k·u) = ∫ y in (-u/2 + k·u)..(u/2 + k·u), g y`. -/
theorem intervalIntegral_comp_add_zsmul (g : ℝ → ℂ) (k : ℤ) :
    (∫ x in (-(u/2))..(u/2), g (x + k • u))
      = ∫ y in (-(u/2) + k • u)..(u/2 + k • u), g y := by
  exact intervalIntegral.integral_comp_add_right (a := -(u/2)) (b := u/2) g (k • u)

/-- The translated `Ioc`-integrals of `‖g‖` are summable with sum `∫_ℝ ‖g‖`.
This is the `L¹` domination: the half-open intervals `(-u/2 + k·u, u/2 + k·u]`
tile `ℝ`, so by `hasSum_integral_iUnion` the sum of `∫_{tile} ‖g‖` is `∫_ℝ ‖g‖`. -/
theorem hasSum_intervalIntegral_norm_comp_add_zsmul
    (hu : 0 < u) (g : ℝ → ℂ) (hg : Integrable g volume) :
    HasSum (fun k : ℤ => ∫ x in (-(u/2))..(u/2), ‖g (x + k • u)‖)
      (∫ y, ‖g y‖ ∂volume) := by
  -- Reduce to set-integrals over the tiling `Ioc (-u/2 + k•u) (u/2 + k•u)`.
  have hle : ∀ k : ℤ, (-(u/2) + k • u : ℝ) ≤ (u/2 + k • u : ℝ) := by
    intro k
    have h2 : -(u/2) ≤ u/2 := by linarith
    linarith
  have h_reindex : ∀ k : ℤ,
      (∫ x in (-(u/2))..(u/2), ‖g (x + k • u)‖)
        = ∫ y in Set.Ioc (-(u/2) + k • u) (u/2 + k • u), ‖g y‖ ∂volume := by
    intro k
    rw [show (∫ x in (-(u/2))..(u/2), ‖g (x + k • u)‖)
          = ∫ x in (-(u/2))..(u/2), (fun y => ‖g y‖) (x + k • u) from rfl]
    rw [intervalIntegral.integral_comp_add_right (fun y => ‖g y‖) (k • u)]
    rw [intervalIntegral.integral_of_le (hle k)]
  -- The tiling intervals: `s k := Ioc (-(u/2) + k•u) (u/2 + k•u)`.
  -- These equal `Ioc (a + k•u) (a + (k+1)•u)` with `a = -(u/2)` since
  -- `a + (k+1)•u = -(u/2) + u + k•u = u/2 + k•u`.
  have h_s_eq : ∀ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)
      = Set.Ioc (-(u/2) + k • u) (-(u/2) + (k + 1) • u) := by
    intro k
    congr 1
    rw [add_smul, one_smul]; ring
  -- `⋃ k, s k = univ`.
  have h_union : (⋃ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)) = Set.univ := by
    simp_rw [h_s_eq]
    exact iUnion_Ioc_add_zsmul hu (-(u/2))
  -- Disjointness.
  have h_disj : Pairwise (Function.onFun Disjoint
      (fun k : ℤ => Set.Ioc (-(u/2) + k • u) (u/2 + k • u))) := by
    simp_rw [h_s_eq]
    exact Set.pairwise_disjoint_Ioc_add_zsmul (-(u/2)) u
  -- `IntegrableOn ‖g‖ (⋃ k, s k) = IntegrableOn ‖g‖ univ`.
  have hInt_on : IntegrableOn (fun y => ‖g y‖)
      (⋃ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)) volume := by
    rw [h_union]
    rw [integrableOn_univ]
    exact hg.norm
  have h_hasSum := MeasureTheory.hasSum_integral_iUnion
    (μ := volume) (f := fun y => ‖g y‖)
    (s := fun k : ℤ => Set.Ioc (-(u/2) + k • u) (u/2 + k • u))
    (fun k => measurableSet_Ioc) h_disj hInt_on
  rw [h_union] at h_hasSum
  rw [setIntegral_univ] at h_hasSum
  -- Rewrite each interval-integral summand into the set-integral form.
  rw [show (fun k : ℤ => ∫ x in (-(u/2))..(u/2), ‖g (x + k • u)‖)
        = (fun k : ℤ => ∫ y in Set.Ioc (-(u/2) + k • u) (u/2 + k • u), ‖g y‖ ∂volume) from
      funext h_reindex]
  exact h_hasSum

/-! ## The integrand and the sum/integral swap

We now compute `fourierCoeff (liftIoc u (-(u/2)) (periodization u g)) j`.  After
`fourierCoeff_eq_intervalIntegral` it becomes `(1/u)·∫_{-u/2}^{u/2} fourier(-j)(↑x)·(liftIoc gpr)(↑x)`.
On the period the lift agrees with `gpr = periodization u g`, and the character
`fourier(-j)(↑x)` equals `latticeChar u j x`.  We then push the `tsum` defining
`gpr` past `latticeChar` and swap with the integral. -/

/-- On the fundamental domain `Ioc (-(u/2)) (u/2)`, the integrand
`fourier (-j) (↑x) • (liftIoc u (-(u/2)) (periodization u g)) (↑x)` equals the
pointwise `tsum` `∑'_k latticeChar u j x · g (x + k·u)`. -/
theorem integrand_eq_tsum_on_period
    (hu : 0 < u) (g : ℝ → ℂ) (j : ℤ) {x : ℝ} (hx : x ∈ Set.Ioc (-(u/2)) (u/2)) :
    haveI : Fact (0 < u) := ⟨hu⟩
    (fourier (-j) (↑x : AddCircle u) : ℂ)
        • (AddCircle.liftIoc u (-(u/2)) (periodization u g)) (↑x : AddCircle u)
      = ∑' k : ℤ, latticeChar u j x * g (x + k • u) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  -- `liftIoc` agrees with `periodization u g` on the period.
  have hx' : x ∈ Set.Ioc (-(u/2)) (-(u/2) + u) := by
    have he : -(u/2) + u = u/2 := by ring
    rw [he]; exact hx
  rw [AddCircle.liftIoc_coe_apply hx']
  -- `fourier (-j) (↑x) = latticeChar u j x`.
  rw [fourier_coe_apply]
  have h_char : Complex.exp (2 * ↑π * Complex.I * ↑(-j) * ↑x / ↑u)
      = latticeChar u j x := by
    unfold latticeChar
    congr 1
    push_cast
    ring
  rw [h_char]
  -- `latticeChar u j x • (∑'_k g(x + k•u)) = ∑'_k latticeChar u j x · g(x + k•u)`.
  unfold periodization
  rw [smul_eq_mul, ← tsum_mul_left]

/-- **Sum/integral swap.**  `∫_{-u/2}^{u/2} ∑'_k (latticeChar u j x · g(x+k·u)) dx
= ∑'_k ∫_{-u/2}^{u/2} latticeChar u j x · g(x+k·u) dx`, justified by the `L¹`
domination `hasSum_intervalIntegral_norm_comp_add_zsmul`. -/
theorem intervalIntegral_tsum_swap
    (hu : 0 < u) (g : ℝ → ℂ) (hg : Integrable g volume) (j : ℤ) :
    (∫ x in (-(u/2))..(u/2), ∑' k : ℤ, latticeChar u j x * g (x + k • u))
      = ∑' k : ℤ, ∫ x in (-(u/2))..(u/2), latticeChar u j x * g (x + k • u) := by
  have hle : -(u/2) ≤ u/2 := by linarith
  -- Convert interval integrals to set-integrals over `Ioc` (both sides).
  rw [intervalIntegral.integral_of_le hle]
  rw [show (fun k : ℤ => ∫ x in (-(u/2))..(u/2), latticeChar u j x * g (x + k • u))
        = (fun k : ℤ => ∫ x in Set.Ioc (-(u/2)) (u/2), latticeChar u j x * g (x + k • u) ∂volume)
      from by funext k; rw [intervalIntegral.integral_of_le hle]]
  -- Each translate `x ↦ latticeChar · g(x+k•u)` is integrable on the interval.
  -- |latticeChar u j x| = 1, so ‖term k x‖ = ‖g(x+k•u)‖.
  have h_term_int : ∀ k : ℤ, Integrable
      (fun x => latticeChar u j x * g (x + k • u))
      (volume.restrict (Set.Ioc (-(u/2)) (u/2))) := by
    intro k
    have hgk : Integrable (fun x => g (x + k • u))
        (volume.restrict (Set.Ioc (-(u/2)) (u/2))) :=
      (integrable_comp_add_zsmul g hg k).restrict
    -- multiply by the bounded measurable `latticeChar`.
    refine Integrable.bdd_mul' (c := 1) hgk ?_ ?_
    · -- a.e.-strong-measurability of latticeChar
      apply Continuous.aestronglyMeasurable
      unfold latticeChar
      fun_prop
    · -- ‖latticeChar u j x‖ ≤ 1
      refine Filter.Eventually.of_forall (fun x => ?_)
      rw [latticeChar_norm_eq_one u j x]
  -- The `‖·‖` integrals are summable (the `L¹` domination).
  have h_norm_summable : Summable
      (fun k : ℤ => ∫ x in Set.Ioc (-(u/2)) (u/2), ‖latticeChar u j x * g (x + k • u)‖ ∂volume) := by
    -- ‖latticeChar · g(x+k•u)‖ = ‖g(x+k•u)‖ pointwise.
    have h_eq : ∀ k : ℤ,
        (∫ x in Set.Ioc (-(u/2)) (u/2), ‖latticeChar u j x * g (x + k • u)‖ ∂volume)
          = ∫ x in Set.Ioc (-(u/2)) (u/2), ‖g (x + k • u)‖ ∂volume := by
      intro k
      refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
      show ‖latticeChar u j x * g (x + k • u)‖ = ‖g (x + k • u)‖
      rw [norm_mul, latticeChar_norm_eq_one u j x, one_mul]
    rw [show (fun k : ℤ => ∫ x in Set.Ioc (-(u/2)) (u/2),
              ‖latticeChar u j x * g (x + k • u)‖ ∂volume)
          = (fun k : ℤ => ∫ x in Set.Ioc (-(u/2)) (u/2), ‖g (x + k • u)‖ ∂volume) from
        funext h_eq]
    -- This family is summable since it `HasSum` to `∫_ℝ ‖g‖`.
    have hHS := hasSum_intervalIntegral_norm_comp_add_zsmul hu g hg
    rw [show (fun k : ℤ => ∫ x in (-(u/2))..(u/2), ‖g (x + k • u)‖)
          = (fun k : ℤ => ∫ x in Set.Ioc (-(u/2)) (u/2), ‖g (x + k • u)‖ ∂volume) from by
        funext k; rw [intervalIntegral.integral_of_le hle]] at hHS
    exact hHS.summable
  -- Apply the measure-theoretic swap.
  rw [← integral_tsum_of_summable_integral_norm h_term_int h_norm_summable]

/-! ## Substitution and reassembly

Each per-`k` integral, after the substitution `y = x + k·u` (translation
invariance, `intervalIntegral.integral_comp_add_right`) and the
`u`-periodicity of the character (`latticeChar_add_zsmul`), is the integral of
`latticeChar u j y · g y` over the tile `(-u/2 + k·u, u/2 + k·u]`.  Summing the
tiles reassembles `∫_ℝ latticeChar u j y · g y = 𝓕g(j/u)`. -/

/-- The per-`k` integral equals the integral of `latticeChar u j · g` over the
tile `Ioc (-u/2 + k·u) (u/2 + k·u)`. -/
theorem term_integral_eq_tile_integral
    (g : ℝ → ℂ) (j : ℤ) (hu : u ≠ 0) (k : ℤ) :
    (∫ x in (-(u/2))..(u/2), latticeChar u j x * g (x + k • u))
      = ∫ y in (-(u/2) + k • u)..(u/2 + k • u), latticeChar u j y * g y := by
  -- substitute `y = x + k•u`.
  rw [show (fun x => latticeChar u j x * g (x + k • u))
        = (fun x => (fun y => latticeChar u j (y - k • u) * g y) (x + k • u)) from by
      funext x; simp only [add_sub_cancel_right]]
  rw [intervalIntegral.integral_comp_add_right
        (fun y => latticeChar u j (y - k • u) * g y) (k • u)]
  -- The shifted character `latticeChar u j (y - k•u) = latticeChar u j y` (periodicity).
  refine intervalIntegral.integral_congr (fun y _ => ?_)
  have hper : latticeChar u j (y - k • u) = latticeChar u j y := by
    have := latticeChar_add_zsmul u hu j (-k) y
    simp only [neg_smul, ← sub_eq_add_neg] at this
    exact this
  rw [hper]

/-- **Reassembly.**  `∑'_k ∫_{tile k} latticeChar u j y · g y = ∫_ℝ latticeChar u j y · g y`. -/
theorem tsum_tile_integral_eq_full
    (hu : 0 < u) (g : ℝ → ℂ) (hg : Integrable g volume) (j : ℤ) :
    (∑' k : ℤ, ∫ y in (-(u/2) + k • u)..(u/2 + k • u), latticeChar u j y * g y)
      = ∫ y, latticeChar u j y * g y ∂volume := by
  have hle : ∀ k : ℤ, (-(u/2) + k • u : ℝ) ≤ (u/2 + k • u : ℝ) := by
    intro k
    have h2 : -(u/2) ≤ u/2 := by linarith
    linarith
  -- `latticeChar u j · g` is integrable (bounded char times integrable `g`).
  have hFint : Integrable (fun y => latticeChar u j y * g y) volume := by
    refine Integrable.bdd_mul' (c := 1) hg ?_ ?_
    · apply Continuous.aestronglyMeasurable; unfold latticeChar; fun_prop
    · exact Filter.Eventually.of_forall (fun y => le_of_eq (latticeChar_norm_eq_one u j y))
  -- Use the tiling set-integral `HasSum`.
  have h_s_eq : ∀ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)
      = Set.Ioc (-(u/2) + k • u) (-(u/2) + (k + 1) • u) := by
    intro k; congr 1; rw [add_smul, one_smul]; ring
  have h_union : (⋃ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)) = Set.univ := by
    simp_rw [h_s_eq]; exact iUnion_Ioc_add_zsmul hu (-(u/2))
  have h_disj : Pairwise (Function.onFun Disjoint
      (fun k : ℤ => Set.Ioc (-(u/2) + k • u) (u/2 + k • u))) := by
    simp_rw [h_s_eq]; exact Set.pairwise_disjoint_Ioc_add_zsmul (-(u/2)) u
  have hInt_on : IntegrableOn (fun y => latticeChar u j y * g y)
      (⋃ k : ℤ, Set.Ioc (-(u/2) + k • u) (u/2 + k • u)) volume := by
    rw [h_union, integrableOn_univ]; exact hFint
  have h_hasSum := MeasureTheory.hasSum_integral_iUnion
    (μ := volume) (f := fun y => latticeChar u j y * g y)
    (s := fun k : ℤ => Set.Ioc (-(u/2) + k • u) (u/2 + k • u))
    (fun k => measurableSet_Ioc) h_disj hInt_on
  rw [h_union, setIntegral_univ] at h_hasSum
  -- Convert tile interval-integrals to set-integrals.
  rw [show (fun k : ℤ => ∫ y in (-(u/2) + k • u)..(u/2 + k • u), latticeChar u j y * g y)
        = (fun k : ℤ => ∫ y in Set.Ioc (-(u/2) + k • u) (u/2 + k • u),
            latticeChar u j y * g y ∂volume) from by
      funext k; rw [intervalIntegral.integral_of_le (hle k)]]
  exact h_hasSum.tsum_eq

/-- `∫_ℝ latticeChar u j y · g y = 𝓕g(j/u)`. -/
theorem integral_latticeChar_eq_fourierIntegral (g : ℝ → ℂ) (j : ℤ) :
    (∫ y, latticeChar u j y * g y ∂volume)
      = Real.fourierIntegral g (j / u : ℝ) := by
  rw [Sidon.TorusParseval.fourierIntegral_lattice_exp_form u g j]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun y => ?_))
  show latticeChar u j y * g y
      = Complex.exp (↑(-2 * π * y * (j / u : ℝ)) * Complex.I) • g y
  rw [smul_eq_mul]
  rfl

/-! ## The main period-`u` Poisson sampling theorem (for the periodisation)

This is the genuine mathlib-shaped primitive: the period-`u` Fourier coefficient
of the periodisation `liftIoc u (-(u/2)) (periodization u g)` samples the full-line
Fourier transform of `g` at the lattice `j/u`.  Proved for arbitrary `g ∈ L¹(ℝ)`
(no continuity, any period `u > 0`) via the `L¹` sum/integral swap. -/

/-- **Period-`u` Poisson sampling (periodisation form).**

For `g : ℝ → ℂ` with `Integrable g volume` and any `u > 0`:

  `fourierCoeff (liftIoc u (-(u/2)) (periodization u g)) j = (1/u) · 𝓕g(j/u)`.

This is the period-`u`, `L¹` generalisation of mathlib's
`Real.fourierCoeff_tsum_comp_add` (which is hardcoded to period 1 and requires a
*continuous* base with a uniform summability bound).  We never assume continuity:
the sum/integral swap is the measure-theoretic `integral_tsum_of_summable_integral_norm`,
dominated by `g ∈ L¹`. -/
theorem period_u_poisson_sampling
    (hu : 0 < u) (g : ℝ → ℂ) (hg : Integrable g volume) (j : ℤ) :
    haveI : Fact (0 < u) := ⟨hu⟩
    fourierCoeff (AddCircle.liftIoc u (-(u/2)) (periodization u g)) j
      = (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ) := by
  haveI : Fact (0 < u) := ⟨hu⟩
  have hu_ne : u ≠ 0 := ne_of_gt hu
  -- Step 1: fourierCoeff as a (1/u)-scaled interval integral over `[-(u/2), u/2]`.
  rw [fourierCoeff_eq_intervalIntegral _ j (-(u/2))]
  have hbound : -(u/2) + u = u/2 := by ring
  rw [hbound]
  have hle : -(u/2) ≤ u/2 := by linarith
  -- Step 2: rewrite the integrand on the period as the pointwise `tsum`.
  -- Convert to a set-integral over `Ioc`, then rewrite the integrand pointwise.
  rw [intervalIntegral.integral_of_le hle]
  rw [setIntegral_congr_fun (μ := volume) measurableSet_Ioc
        (g := fun x => ∑' k : ℤ, latticeChar u j x * g (x + k • u))
        (fun x hx => integrand_eq_tsum_on_period hu g j hx)]
  rw [← intervalIntegral.integral_of_le hle]
  -- Step 3: swap sum and integral.
  rw [intervalIntegral_tsum_swap hu g hg j]
  -- Step 4: per-`k` substitution to the tile integral.
  rw [show (fun k : ℤ => ∫ x in (-(u/2))..(u/2), latticeChar u j x * g (x + k • u))
        = (fun k : ℤ => ∫ y in (-(u/2) + k • u)..(u/2 + k • u), latticeChar u j y * g y) from by
      funext k; exact term_integral_eq_tile_integral g j hu_ne k]
  -- Step 5: reassemble the tiles into `∫_ℝ`.
  rw [tsum_tile_integral_eq_full hu g hg j]
  -- Step 6: `∫_ℝ latticeChar u j · g = 𝓕g(j/u)`.
  rw [integral_latticeChar_eq_fourierIntegral g j]
  -- Final: `(1/u : ℝ) • z = (1/u : ℂ) * z`.
  show ((1 / u : ℝ) : ℝ) • Real.fourierIntegral g (j / u : ℝ)
      = (1 / u : ℂ) * Real.fourierIntegral g (j / u : ℝ)
  rw [Complex.real_smul]
  push_cast
  ring

/-! ============================================================================
## Bridge to the Sidon `g = (f*f) + autocorr f` and to `MOLemma21`
============================================================================

The naive *truncation*-lift coefficient identity — Fourier coefficient of
`liftIoc u (-(u/2)) (g_complex f)` equal to `(1/u)·𝓕(g_complex f)(j/u)` —
is FALSE.  Because `supp (g f) = (-1/2, 1/2)` **overflows** the period
`(-u/2, u/2) = (-0.319, 0.319)`, the truncation lift and the periodisation
lift `liftIoc u (-(u/2)) (periodization u (g_complex f))` are **genuinely
different functions** (they disagree on `(u/2, 1/2)`); the per-coefficient
identity holds verbatim only for the periodisation, which
`period_u_poisson_sampling` proves.

What IS true — and what the MO 2.1 pairing actually needs — is that the
periodisation agrees with `g` **on `supp K_ms ⊆ (-δ₁, δ₁)`** (the non-overlap
`1/2 + δ₁ = u`).  Hence the truncation and periodisation lifts pair
*identically* against `lift K_ms`, so the MO 2.1 core identity holds
hypothesis-free via the periodisation (`mo_lemma_21_core_via_periodization`
below).  We develop that bridge here. -/

open Sidon.Constructor.MOLemma21 (g_complex K_ms_complex)
open Sidon.Constructor.PeriodU (g)
open Sidon.FourierAux (autocorr autocorr_def)

/-- Support of `g f = (f*f) + autocorr f` lies in `Ioo (-1/2) (1/2)` for `f`
supported in `Ioo (-1/4) (1/4)`. -/
theorem g_support_subset {f : ℝ → ℝ}
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (g f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  -- `support (g f) ⊆ support (f*f) ∪ support (autocorr f)`.
  have h_union : Function.support (g f)
      ⊆ Function.support (MeasureTheory.convolution f f
            (ContinuousLinearMap.mul ℝ ℝ) volume)
        ∪ Function.support (autocorr f) := by
    intro x hx
    have hx' : g f x ≠ 0 := hx
    by_contra h_not
    rw [Set.mem_union] at h_not
    push_neg at h_not
    obtain ⟨h1, h2⟩ := h_not
    rw [Function.mem_support, not_not] at h1 h2
    apply hx'
    unfold g
    rw [h1, h2, add_zero]
  refine h_union.trans (Set.union_subset ?_ ?_)
  · -- support (f*f) ⊆ supp f + supp f ⊆ Ioo(-1/2,1/2).
    intro x hx
    have h_sub : Function.support (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) ⊆
        Function.support f + Function.support f := support_convolution_subset _
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (h_sub hx)
    have ha' := hf_supp ha; have hb' := hf_supp hb
    simp only [Set.mem_Ioo] at ha' hb' ⊢
    subst hab
    exact ⟨by linarith [ha'.1, hb'.1], by linarith [ha'.2, hb'.2]⟩
  · -- support (autocorr f) ⊆ supp (f∘neg) + supp f ⊆ Ioo(-1/2,1/2).
    intro x hx
    have h_eq : autocorr f = MeasureTheory.convolution (fun y => f (-y)) f
        (ContinuousLinearMap.mul ℝ ℝ) volume := by
      funext y
      rw [autocorr_def]
      simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
      rw [← MeasureTheory.integral_neg_eq_self (fun t => f (-t) * f (y - t)) volume]
      refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
      simp only [neg_neg]
      rw [show y - -s = y + s by ring]
    rw [h_eq] at hx
    have h_sub : Function.support (MeasureTheory.convolution (fun y => f (-y)) f
        (ContinuousLinearMap.mul ℝ ℝ) volume) ⊆
        Function.support (fun y => f (-y)) + Function.support f :=
      support_convolution_subset _
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (h_sub hx)
    -- `a ∈ support (f∘neg)` ⟹ `-a ∈ support f` ⟹ `-a ∈ Ioo(-1/4,1/4)` ⟹ `a ∈ Ioo(-1/4,1/4)`.
    have ha_neg : f (-a) ≠ 0 := ha
    have ha' := hf_supp ha_neg
    have hb' := hf_supp hb
    simp only [Set.mem_Ioo] at ha' hb' ⊢
    subst hab
    constructor
    · linarith [ha'.2, hb'.1]
    · linarith [ha'.1, hb'.2]

/-- `u = 1/2 + δ₁` as reals (from `uQ_eq`). -/
theorem uQ_real_eq_half_add_delta1 : uQ_real = 1/2 + delta1 := by
  unfold uQ_real delta1
  have := congrArg (fun q : ℚ => (q : ℝ)) uQ_eq
  push_cast at this
  linarith [this]

/-- **The non-overlap agreement.**  For `x ∈ Ioo (-δ₁) δ₁` (the open support of
`K_ms`), the periodisation `periodization u (g_complex f)` agrees with the bare
`g_complex f`: the `k ≠ 0` translates `g(x + k·u)` vanish because
`1/2 + δ₁ = u` pushes them outside `supp g ⊆ (-1/2, 1/2)`. -/
theorem periodization_eq_on_Ioo_delta1 (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    {x : ℝ} (hx : x ∈ Set.Ioo (-delta1) delta1) :
    periodization uQ_real (g_complex f) x = g_complex f x := by
  have hu : uQ_real = 1/2 + delta1 := uQ_real_eq_half_add_delta1
  have hδ : 0 < delta1 := (delta_positivities).1
  have hg_supp : Function.support (g f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
    g_support_subset hf_supp
  simp only [Set.mem_Ioo] at hx
  -- Every `k ≠ 0` term vanishes: `g_complex f (x + k•u) = 0`.
  have h_vanish : ∀ k : ℤ, k ≠ 0 → g_complex f (x + k • uQ_real) = 0 := by
    intro k hk
    have hgk : g f (x + k • uQ_real) = 0 := by
      -- `x + k•u ∉ Ioo(-1/2,1/2)` ⟹ not in support ⟹ `= 0`.
      by_contra hne
      have hmem : (x + k • uQ_real) ∈ Set.Ioo (-(1/2 : ℝ)) (1/2) := hg_supp hne
      simp only [Set.mem_Ioo] at hmem
      rcases lt_or_gt_of_ne hk with hkneg | hkpos
      · -- k ≤ -1: `x + k•u ≤ δ₁ + (-1)·u = δ₁ - u = -1/2`, contradicting `> -1/2`.
        have hk1 : (k : ℝ) ≤ -1 := by exact_mod_cast Int.le_sub_one_of_lt hkneg
        have hku : (k • uQ_real : ℝ) = (k : ℝ) * uQ_real := by
          rw [zsmul_eq_mul]
        have hupos : 0 < uQ_real := by rw [hu]; linarith
        have : (k : ℝ) * uQ_real ≤ -uQ_real := by nlinarith [hk1, hupos]
        rw [hku] at hmem
        -- `x < δ₁` and `k•u ≤ -u`, so `x + k•u < δ₁ - u = -1/2`.
        nlinarith [hmem.1, hx.2, this, hu]
      · -- k ≥ 1: `x + k•u ≥ -δ₁ + u = 1/2`, contradicting `< 1/2`.
        have hk1 : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hkpos
        have hku : (k • uQ_real : ℝ) = (k : ℝ) * uQ_real := by rw [zsmul_eq_mul]
        have hupos : 0 < uQ_real := by rw [hu]; linarith
        have : uQ_real ≤ (k : ℝ) * uQ_real := by nlinarith [hk1, hupos]
        rw [hku] at hmem
        nlinarith [hmem.2, hx.1, this, hu]
    unfold g_complex
    rw [hgk]
    simp
  -- The tsum collapses to the `k = 0` term.
  unfold periodization
  rw [tsum_eq_single 0 (fun k hk => h_vanish k hk)]
  simp

/-! ### `g_per ∈ L²` on one period

On `Ioc (-(u/2)) (u/2)`, only the translates `k ∈ {-1, 0, 1}` of `g` are nonzero
(`supp g ⊆ (-1/2, 1/2)` and `u = 0.638`), so the periodisation equals the finite
sum `∑_{k=-1}^{1} g(·+k·u)` there, which is in `L²` (each translate is `L²`). -/

/-- On the period `Ioc (-(u/2)) (u/2)`, the periodisation is the finite sum over
`k ∈ {-1, 0, 1}` of the translates `g_complex f (·+ k·u)`. -/
theorem periodization_eq_finsum_on_period (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    {x : ℝ} (hx : x ∈ Set.Ioc (-(uQ_real/2)) (uQ_real/2)) :
    periodization uQ_real (g_complex f) x
      = ∑ k ∈ Finset.Icc (-1 : ℤ) 1, g_complex f (x + k • uQ_real) := by
  have hu : uQ_real = 1/2 + delta1 := uQ_real_eq_half_add_delta1
  have hδ : 0 < delta1 := (delta_positivities).1
  have hδlt : delta1 < uQ_real / 2 := Sidon.Constructor.MOLemma21.delta1_lt_u_half
  have hg_supp : Function.support (g f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
    g_support_subset hf_supp
  simp only [Set.mem_Ioc] at hx
  -- Outside `{-1,0,1}` the translate vanishes.
  have h_vanish : ∀ k : ℤ, k ∉ Finset.Icc (-1 : ℤ) 1 → g_complex f (x + k • uQ_real) = 0 := by
    intro k hk
    simp only [Finset.mem_Icc, not_and_or, not_le] at hk
    have hgk : g f (x + k • uQ_real) = 0 := by
      by_contra hne
      have hmem : (x + k • uQ_real) ∈ Set.Ioo (-(1/2 : ℝ)) (1/2) := hg_supp hne
      simp only [Set.mem_Ioo] at hmem
      have hku : (k • uQ_real : ℝ) = (k : ℝ) * uQ_real := by rw [zsmul_eq_mul]
      have hupos : 0 < uQ_real := by rw [hu]; linarith
      rcases hk with hkneg | hkpos
      · -- k ≤ -2: `x + k•u ≤ u/2 - 2u = -3u/2 < -1/2` (since u > 1/3). Contradiction with `> -1/2`.
        have hk2 : (k : ℝ) ≤ -2 := by exact_mod_cast Int.le_sub_one_of_lt hkneg
        rw [hku] at hmem
        nlinarith [hmem.1, hx.2, hk2, hupos, hu, hδlt]
      · -- k ≥ 2: `x + k•u > -u/2 + 2u = 3u/2 > 1/2`. Contradiction with `< 1/2`.
        have hk2 : (2 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hkpos
        rw [hku] at hmem
        nlinarith [hmem.2, hx.1, hk2, hupos, hu, hδlt]
    unfold g_complex; rw [hgk]; simp
  -- The tsum equals the finite sum over `Icc (-1) 1`.
  unfold periodization
  rw [tsum_eq_sum (s := Finset.Icc (-1 : ℤ) 1) (fun k hk => h_vanish k hk)]

/-- `g_per ∈ L²` on one period, via the finite-sum decomposition. -/
theorem periodization_memLp_period (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
  -- The finite sum `∑_{k=-1}^1 g_complex f (·+k•u)` is `MemLp` on the restriction.
  have h_sum_memLp : MemLp
      (∑ k ∈ Finset.Icc (-1 : ℤ) 1, fun x => g_complex f (x + k • uQ_real)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
    refine memLp_finset_sum' (Finset.Icc (-1 : ℤ) 1) (fun k _ => ?_)
    -- each translate `x ↦ g_complex f (x + k•u)` is `L²` on the finite-measure period:
    -- it is everywhere bounded by `2∫f²` (same Cauchy–Schwarz bound as `g`), so `MemLp`
    -- on the finite interval.
    haveI : IsFiniteMeasure (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
      refine ⟨?_⟩
      rw [Measure.restrict_apply_univ, Real.volume_Ioc]
      exact ENNReal.ofReal_lt_top
    set C : ℝ := 2 * ∫ x, f x ^ 2 ∂volume with hC_def
    have hmeas : AEStronglyMeasurable (fun x => g_complex f (x + k • uQ_real))
        (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
      have h1 : AEStronglyMeasurable (g_complex f) volume :=
        Sidon.Constructor.MOLemma21.g_complex_aestronglyMeasurable f hf_int
      have h2 := h1.comp_measurePreserving
        (measurePreserving_add_right (volume : Measure ℝ) (k • uQ_real))
      exact (by simpa [Function.comp] using h2 : AEStronglyMeasurable
        (fun x => g_complex f (x + k • uQ_real)) volume).restrict
    have h_bound : ∀ᵐ x ∂(volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))),
        ‖g_complex f (x + k • uQ_real)‖ ≤ C := by
      refine Filter.Eventually.of_forall fun x => ?_
      -- `‖g_complex f y‖ = |g f y| ≤ |f*f y| + |autocorr f y| ≤ 2∫f² = C` at `y = x + k•u`.
      unfold g_complex Sidon.Constructor.PeriodU.g
      rw [Complex.norm_real, Real.norm_eq_abs]
      calc |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume)
              (x + k • uQ_real) + autocorr f (x + k • uQ_real)|
          ≤ |(MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume)
              (x + k • uQ_real)| + |autocorr f (x + k • uQ_real)| := abs_add_le _ _
        _ ≤ (∫ x, f x ^ 2 ∂volume) + (∫ x, f x ^ 2 ∂volume) :=
            add_le_add (Sidon.Constructor.MOLemma21.convff_abs_le f hf_L2 _)
              (Sidon.Constructor.MOLemma21.autocorr_abs_le_local f hf_L2 _)
        _ = C := by rw [hC_def]; ring
    exact MemLp.of_bound hmeas C h_bound
  -- Now transfer via the ae-agreement on the period.
  refine MemLp.ae_eq ?_ h_sum_memLp
  refine (ae_restrict_iff' measurableSet_Ioc).mpr ?_
  refine Filter.Eventually.of_forall (fun x hx => ?_)
  rw [Finset.sum_apply]
  exact (periodization_eq_finsum_on_period f hf_supp hx).symm

/-! ### The inner product `⟪lift K_ms, lift g_per⟫ = (1/u)·∫_ℝ g·K_ms`

The periodisation pairs against `lift K_ms` *exactly as `g` does*, since
`g_per = g` on `supp K_ms ⊆ (-δ₁, δ₁)` and `K_ms = 0` off `(-δ₁, δ₁)`. -/

/-- The interval integral over one period of `g_per·conj(K_ms)` equals the
full-line integral `∫_ℝ g·K_ms`: `K_ms` vanishes off `(-δ₁, δ₁)`, and on
`(-δ₁, δ₁)` the periodisation `g_per` agrees with `g`. -/
theorem intervalIntegral_gper_Kms_eq_full (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    (∫ x in (-(uQ_real/2))..(uQ_real/2),
        periodization uQ_real (g_complex f) x * starRingEnd ℂ (K_ms_complex x))
      = ∫ x, g_complex f x * K_ms_complex x ∂volume := by
  have hδ : 0 < delta1 := (delta_positivities).1
  have hδlt : delta1 < uQ_real / 2 := Sidon.Constructor.MOLemma21.delta1_lt_u_half
  -- Reduce both to the full-line integral of `g·K_ms` (real, conj-free).
  rw [intervalIntegral.integral_of_le (by linarith [uQ_real_pos] : -(uQ_real/2) ≤ uQ_real/2)]
  -- On `Ioc (-(u/2)) (u/2)`, the integrand `g_per·conj(K_ms)` equals `g·K_ms`.
  have h_integrand : ∀ x ∈ Set.Ioc (-(uQ_real/2)) (uQ_real/2),
      periodization uQ_real (g_complex f) x * starRingEnd ℂ (K_ms_complex x)
        = g_complex f x * K_ms_complex x := by
    intro x hx
    by_cases hxδ : x ∈ Set.Ioo (-delta1) delta1
    · -- On `supp K_ms`: `g_per = g`, `conj K_ms = K_ms`.
      rw [periodization_eq_on_Ioo_delta1 f hf_supp hxδ,
        Sidon.Constructor.MOLemma21.conj_K_ms_complex]
    · -- Off `(-δ₁,δ₁)`: `K_ms x = 0`, both products vanish.
      have hK0 : K_ms_complex x = 0 := by
        have hxabs : delta1 ≤ |x| := by
          rw [Set.mem_Ioo, not_and_or, not_lt, not_lt] at hxδ
          rcases hxδ with h | h
          · -- `x ≤ -δ₁` ⟹ `|x| = -x ≥ δ₁`.
            rw [abs_of_nonpos (by linarith)]; linarith
          · -- `δ₁ ≤ x` ⟹ `|x| = x ≥ δ₁`.
            rw [le_abs]; left; exact h
        unfold K_ms_complex
        rw [Sidon.Constructor.MOLemma21.K_ms_zero_outside x hxabs]
        simp
      rw [hK0, map_zero, mul_zero, mul_zero]
  rw [setIntegral_congr_fun (μ := volume) measurableSet_Ioc h_integrand]
  -- `∫_{Ioc} g·K_ms = ∫_ℝ g·K_ms` (`K_ms = 0` off the period).
  refine MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ?_
  intro x hx
  have hK0 : K_ms_complex x = 0 := by
    by_contra h_ne
    exact hx (Sidon.Constructor.MOLemma21.K_ms_complex_support h_ne)
  rw [hK0, mul_zero]

/-- **The inner product = integral identity (periodisation form).**
`⟪lift K_ms, lift g_per⟫ = (1/u)·∫_ℝ g·K_ms`. -/
theorem inner_gper_Kms_eq (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hgper_L2_period : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    @inner ℂ _ _
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          K_ms_complex Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex))
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          (periodization uQ_real (g_complex f)) hgper_L2_period).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (g_complex f))))
      = (uQ_real⁻¹ : ℂ) * ∫ x, g_complex f x * K_ms_complex x ∂volume := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  rw [Sidon.BilinearParseval.inner_toLp_eq_intervalIntegral uQ_real uQ_real_pos
    (periodization uQ_real (g_complex f)) K_ms_complex hgper_L2_period
    Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict]
  rw [intervalIntegral_gper_Kms_eq_full f hf_supp]

/-- The `toLp` representative of `lift g_per` has Fourier coefficient
`(1/u)·𝓕g(j/u)` — the `toLp`-wrapped form of `period_u_poisson_sampling`. -/
theorem fourierCoeff_toLp_gper_eq (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hgper_L2_period : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) (j : ℤ) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    fourierCoeff
        ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
            (periodization uQ_real (g_complex f)) hgper_L2_period).toLp
          (AddCircle.liftIoc uQ_real (-(uQ_real/2))
            (periodization uQ_real (g_complex f))) : AddCircle uQ_real → ℂ) j
      = (1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  -- `fourierCoeff (toLp _) j = fourierCoeff (liftIoc g_per) j` (ae-bridge).
  have h_ae :
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          (periodization uQ_real (g_complex f)) hgper_L2_period).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (g_complex f)))
          : AddCircle uQ_real → ℂ)
        =ᵐ[@AddCircle.haarAddCircle uQ_real ⟨uQ_real_pos⟩]
          AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (g_complex f)) :=
    MemLp.coeFn_toLp _
  rw [fourierCoeff_congr_ae h_ae]
  -- `g_complex f` is integrable; apply `period_u_poisson_sampling`.
  have hgint : Integrable (g_complex f) volume :=
    Sidon.Constructor.MOLemma21.g_complex_integrable f hf_int
  exact period_u_poisson_sampling uQ_real_pos (g_complex f) hgint j

/-! ### The hypothesis-free MO 2.1 core (periodisation form)

The MO 2.1 core complex identity, proved via the periodisation lift `Fper`
(not the truncation `F`), with the coefficient identity supplied by the
*proved* `period_u_poisson_sampling` and the inner product by
`inner_gper_Kms_eq` — so it carries **no** Poisson hypothesis. -/

/-- **MO 2009 Lemma 2.1 (core identity, hypothesis-free via periodisation).**

`∫_ℝ g·K_ms = (1/u)·∑'_j 𝓕g(j/u)·conj(𝓕K_ms(j/u))`,

with the period-`u` Poisson sampling discharged by the proved
`period_u_poisson_sampling` (no hypothesis).  Takes the (proved) `L²(period)`
membership of `g_per` and the support hypothesis. -/
theorem mo_lemma_21_core_via_periodization (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hgper_L2_period : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    (∫ x, g_complex f x * K_ms_complex x ∂volume)
      = (1 / uQ_real : ℂ) * ∑' j : ℤ,
          Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
            * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  have hu_ne : (uQ_real : ℝ) ≠ 0 := ne_of_gt uQ_real_pos
  set Fper := (Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
      (periodization uQ_real (g_complex f)) hgper_L2_period).toLp
    (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (g_complex f))) with hFper_def
  set H := (Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
      K_ms_complex Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict).toLp
    (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex) with hH_def
  have h_parseval :
      ∑' j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle uQ_real → ℂ) j)
        = @inner ℂ _ _ H Fper :=
    Sidon.TorusParseval.bilinear_parseval_addCircle_Lp_tsum uQ_real uQ_real_pos Fper H
  have h_coef_F : ∀ j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                    = (1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ) := by
    intro j; rw [hFper_def]; exact fourierCoeff_toLp_gper_eq f hf_int hgper_L2_period j
  have h_coef_H : ∀ j : ℤ, fourierCoeff (H : AddCircle uQ_real → ℂ) j
                    = (1 / uQ_real : ℂ) * Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ) := by
    intro j; rw [hH_def]; exact Sidon.Constructor.MOLemma21.fourierCoeff_Kms_eq j
  have h_inner : @inner ℂ _ _ H Fper
      = (uQ_real⁻¹ : ℂ) * ∫ x, g_complex f x * K_ms_complex x ∂volume := by
    rw [hH_def, hFper_def]; exact inner_gper_Kms_eq f hf_supp hgper_L2_period
  -- Substitute coefficients into the tsum and factor out `1/u²`.
  have h_tsum_eq :
      ∑' j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle uQ_real → ℂ) j)
        = ((1 / uQ_real ^ 2 : ℝ) : ℂ) * ∑' j : ℤ,
            Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)) := by
    rw [← tsum_mul_left]
    refine tsum_congr (fun j => ?_)
    rw [h_coef_F j, h_coef_H j, map_mul]
    rw [show starRingEnd ℂ (1 / uQ_real : ℂ) = (1 / uQ_real : ℂ) by
        rw [show ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) by push_cast; rfl,
          Complex.conj_ofReal]]
    rw [show (((1 / uQ_real ^ 2 : ℝ)) : ℂ) = (1 / uQ_real : ℂ) * (1 / uQ_real : ℂ) by
        push_cast; field_simp]
    ring
  rw [h_tsum_eq, h_inner] at h_parseval
  set T : ℂ := ∑' j : ℤ,
      Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)) with hT_def
  set I : ℂ := ∫ x, g_complex f x * K_ms_complex x ∂volume with hI_def
  have hu_ne_c : (uQ_real : ℂ) ≠ 0 := by exact_mod_cast hu_ne
  have hmul := congrArg (fun z => (uQ_real : ℂ) * z) h_parseval
  simp only at hmul
  have hL : (uQ_real : ℂ) * (((1 / uQ_real ^ 2 : ℝ) : ℂ) * T) = (1 / uQ_real : ℂ) * T := by
    have : (uQ_real : ℂ) * ((1 / uQ_real ^ 2 : ℝ) : ℂ) = (1 / uQ_real : ℂ) := by
      push_cast; field_simp
    rw [← mul_assoc, this]
  have hR : (uQ_real : ℂ) * ((uQ_real⁻¹ : ℂ) * I) = I := by
    rw [← mul_assoc, mul_inv_cancel₀ hu_ne_c, one_mul]
  rw [hL, hR] at hmul
  rw [hmul]

/-! ### From the hypothesis-free core to `PeriodUParsevalTrue`

We now chain the periodisation core through the real-part reduction (`re_summand`),
the `j = 0` term, and the Fourier–Bessel identity (`bessel_identity`,
`S_cos_abstract_eq_S_cos`) — starting from the hypothesis-free core — to obtain
`LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f`, i.e.
`Sidon.Constructor.PeriodU.PeriodUParsevalTrue f`, with **no Poisson hypothesis**. -/

open Sidon.MultiScale (LHS1 LHS2 S_cos)
open Sidon.Constructor.MOLemma21 (S_cos_abstract)

/-- **MO 2.1, abstract real form (hypothesis-free via periodisation).**
`∫_ℝ g·K_ms = (1/u)·∑'_j 2·(Re 𝓕f(j/u))²·(𝓕K_ms(j/u)).re`. -/
theorem mo_lemma_21_real_via_periodization (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (hgper_L2_period : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))))
    (h_summable : Summable (fun j : ℤ =>
        Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)))) :
    (∫ x, g f x * K_ms x ∂volume)
      = (1 / uQ_real) * ∑' j : ℤ,
          2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
            * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re := by
  have hcore := mo_lemma_21_core_via_periodization f hf_int hf_supp hgper_L2_period
  rw [Sidon.Constructor.MOLemma21.integral_g_Kms_complex_eq f] at hcore
  have hre := congrArg Complex.re hcore
  rw [Complex.ofReal_re] at hre
  rw [hre]
  rw [show ((1 / uQ_real : ℂ) * ∑' j : ℤ,
            Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))).re
        = (1 / uQ_real) * (∑' j : ℤ,
            Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))).re from by
      rw [show ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) by push_cast; rfl]
      rw [Complex.re_ofReal_mul]]
  rw [show (∑' j : ℤ,
            Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))).re
        = ∑' j : ℤ,
            (Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))).re from
      (Complex.reCLM.map_tsum h_summable)]
  congr 1
  refine tsum_congr (fun j => ?_)
  exact Sidon.Constructor.MOLemma21.re_summand f hf_int h_joint j

/-- **The exact P3 identity (`PeriodUParsevalTrue`), hypothesis-free via the
periodisation.**  `LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f`, with the period-`u`
Poisson sampling discharged by the proved `period_u_poisson_sampling` — no Poisson
hypothesis remains.  The remaining inputs are the standard admissibility data
(`Integrable f`, `MemLp f 2`, `∫f = 1`, support, joint integrability), the two
product integrabilities, and two summabilities (all provable / threaded). -/
theorem field_P3_periodUParsevalTrue_via_periodization (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume)
    (hAuto_prod_int : Integrable (fun x => autocorr f x * K_ms x) volume)
    (h_summable : Summable (fun j : ℤ =>
        Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))))
    (h_summable_re : Summable (fun j : ℤ =>
        2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
          * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re)) :
    Sidon.Constructor.PeriodU.PeriodUParsevalTrue f := by
  have hu_ne : (uQ_real : ℝ) ≠ 0 := ne_of_gt uQ_real_pos
  -- `g_per ∈ L²(period)` (proved).
  have hgper_L2 : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    periodization_memLp_period f hf_int hf_L2 hf_supp
  -- Unfold the target: `LHS1 + LHS2 = 2/u + (2/u)·S_cos f`.
  show LHS1 f + LHS2 f = 2 / uQ_real + (2 / uQ_real) * S_cos f
  -- `LHS1 + LHS2 = ∫ g·K`.
  rw [Sidon.Constructor.MOLemma21.LHS1_add_LHS2_eq_integral f hConv_prod_int hAuto_prod_int]
  -- `∫ g·K = (1/u)·∑'_j 2(Re 𝓕f)²(𝓕K_ms).re`.
  rw [mo_lemma_21_real_via_periodization f hf_int hf_supp h_joint hgper_L2 h_summable]
  -- Split the tsum: `j = 0` term + tail.
  set a : ℤ → ℝ := fun j =>
      2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
        * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re with ha_def
  have h_split : ∑' j : ℤ, a j = a 0 + ∑' j : ℤ, (if j = 0 then 0 else a j) :=
    h_summable_re.tsum_eq_add_tsum_ite 0
  rw [h_split]
  have ha0 : a 0 = 2 := Sidon.Constructor.MOLemma21.abstract_zero_term f hf_one
  rw [ha0]
  -- The tail equals `2 · S_cos_abstract f`.
  have h_tail : (∑' j : ℤ, (if j = 0 then 0 else a j)) = 2 * S_cos_abstract f := by
    unfold S_cos_abstract
    rw [← tsum_mul_left]
    refine tsum_congr (fun j => ?_)
    rw [ha_def]; simp only []
    split_ifs with hj
    · ring
    · ring
  rw [h_tail]
  -- `S_cos_abstract f = S_cos f` (Fourier–Bessel).
  rw [Sidon.Constructor.MOLemma21.S_cos_abstract_eq_S_cos f]
  -- `(1/u)·(2 + 2·S) = 2/u + (2/u)·S`.
  ring

/-- **An alternate (weaker, un-normalized) `hEq3_ge`-shaped inequality,
hypothesis-free via the periodisation Poisson sampling.**
`2/u + 2u²·S_cos f ≤ LHS1 f + LHS2 f`, over the **un-normalized** full-line
`S_cos f` with the literal `2u²` coefficient, derived from the proved period-`u`
Poisson sampling (`period_u_poisson_sampling`) with **no** Poisson hypothesis,
through the lossy step `2u² ≤ 2/u` (`field_hEq3_ge_of_true_parseval`).

NOTE — naming caution.  This is TRUE but **NOT** the form consumed by the live
constructor `Sidon.MultiScale.ExtremiserPrimitives.of_admissible`.  The struct
field `ExtremiserPrimitives.hEq3_ge` is stated over the **`u³`-normalized**
`S_cos f / u³` (TIGHT — an equality after the `u³` rescale, no lossy step), and
the live headline discharges it via
`Sidon.Constructor.FieldsParseval.field_hEq3_ge`
(fed the genuine `2/u`-coefficient Parseval equality
`field_P3_periodUParsevalTrue_via_periodization`).  This theorem instead targets
the un-normalized `2u²` form via the weaker `2u² ≤ 2/u` bound and is retained as
an alternate; do not confuse it with the struct field of the same stem name. -/
theorem field_hEq3_ge_via_periodization (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume)
    (hAuto_prod_int : Integrable (fun x => autocorr f x * K_ms x) volume)
    (h_summable : Summable (fun j : ℤ =>
        Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))))
    (h_summable_re : Summable (fun j : ℤ =>
        2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
          * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re)) :
    2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2 * S_cos f ≤ LHS1 f + LHS2 f :=
  Sidon.Constructor.PeriodU.field_hEq3_ge_of_true_parseval f
    (field_P3_periodUParsevalTrue_via_periodization f hf_int hf_L2 hf_one hf_supp
      h_joint hConv_prod_int hAuto_prod_int h_summable h_summable_re)

end

end Sidon.Constructor.PoissonSampling
