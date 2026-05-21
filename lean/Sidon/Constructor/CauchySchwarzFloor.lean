/-
Sidon Autocorrelation Project — Constructor: the MV Eq.(4) Cauchy–Schwarz floor
================================================================================

This module discharges the single residual hypothesis `h_cs` of
`Sidon.Constructor.FieldEq4.field_hEq4` (and hence of the unconditional
headline `Sidon.MultiScale.C1a_ge_1292_unconditional`):

  `h_cs : S_cos f ≥ 2 · min_G_analytic² / S_1_analytic`

for a general admissible `f` (`Integrable f`, `MemLp f 2`,
`support f ⊆ Ioo (-1/4, 1/4)`, `∀ x, 0 ≤ f x`, `∫ f = 1`).

**Why this needs NO period-`u` Poisson lemma.**  `G_concrete` is a
*finite* trigonometric polynomial `∑_{k=1}^{200} a_k cos(2π k x/u)`, so
the inner-product floor `∫ f·G ≥ m_G` is a finite sum, derived by
linearity of the integral over the `200` cosine terms (each
`∫ f cos(2π k x/u) = (𝓕f(k/u)).re` is elementary — see
`Sidon.Constructor.fourierIntegral_re_eq_cos_integral`).  Cauchy–Schwarz
(Sedrakyan/Titu, `Sidon.MV.mv_eq4_core`) on the resulting finite sum
gives `m_G² ≤ (∑_{k=1}^{200} c_k² K̃(k)) · S_1_analytic`, where
`c_k = (𝓕f(k/u)).re`.  The full tsum `S_cos f` dominates twice the
finite truncation (the `±k` pairing, using `(𝓕f(k/u)).re` even in `k`
via the cosine-integral form, and `K̃(-k) = K̃(k)` from `besselJ0` even),
giving `S_cos f ≥ 2 · ∑_{k=1}^{200} c_k² K̃(k) ≥ 2 · m_G² / S_1_analytic`.

The **only** non-elementary input is *summability* of the `S_cos`
family.  This is supplied by the period-`u` Plancherel sum-of-squares
`Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum` (which gives
`Summable (j ↦ ‖(1/u)·𝓕f(j/u)‖²)`, valid because
`supp f ⊆ (-1/4,1/4) ⊆ Ioc (-u/2, u/2)` for `u = 0.638 > 1/2`); the
`S_cos` summand is dominated by the real-part-square (via
`K̃(j) ≤ 1`, `Sidon.Constructor.Glue.S_cos_summand_le_fside`), so
summability transfers by comparison.

No `sorry`, no new `axiom`.  The lattice-positivity input
`hK4` enters via the sanctioned axiom
`Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.MVLemmas
import Sidon.TorusParseval
import Sidon.Constructor.FieldEq4
import Sidon.Constructor.Glue
import Sidon.Constructor.LatticePositivity

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical
open MeasureTheory

namespace Sidon.Constructor

open Sidon.MultiScale

noncomputable section

/-! ## Notation: the real-part cosine data `c_j = (𝓕f(j/u)).re` -/

/-- The real part of the Fourier transform of `f` at the lattice point
`j/u` — equivalently `∫ f(x) cos(2π j x/u) dx`
(`fourierIntegral_re_eq_cos_integral`). -/
private def cData (f : ℝ → ℝ) (j : ℤ) : ℝ :=
  (Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)).re

/-! ## Part 1 — the `S_cos` summand is even in `j`

The summand `cData f j ^ 2 · K_ms_fourier_lattice j` is even in `j`:
`cData f (-j) = cData f j` (cosine is even) and
`K_ms_fourier_lattice (-j) = K_ms_fourier_lattice j` (`besselJ0` even). -/

/-- `cData` is even: `(𝓕f(-j/u)).re = (𝓕f(j/u)).re`, because both equal
`∫ f(x) cos(2π j x/u) dx` and `cos` is even. -/
private theorem cData_neg (f : ℝ → ℝ) (hf : Integrable f volume) (j : ℤ) :
    cData f (-j) = cData f j := by
  unfold cData
  rw [fourierIntegral_re_eq_cos_integral f hf (((-j : ℤ) : ℝ) / uQ_real)]
  rw [fourierIntegral_re_eq_cos_integral f hf ((j : ℝ) / uQ_real)]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  simp only []
  have harg : 2 * Real.pi * x * (((-j : ℤ) : ℝ) / uQ_real)
                = -(2 * Real.pi * x * ((j : ℝ) / uQ_real)) := by
    push_cast; ring
  rw [harg, Real.cos_neg]

/-- `K_ms_fourier_lattice` is even in `j` (each `besselJ0` argument flips
sign, and `besselJ0` is even). -/
private theorem K_ms_fourier_lattice_neg (j : ℤ) :
    K_ms_fourier_lattice (-j) = K_ms_fourier_lattice j := by
  unfold K_ms_fourier_lattice
  have e1 : Real.pi * delta1 * (((-j : ℤ) : ℝ) / uQ_real)
              = -(Real.pi * delta1 * ((j : ℝ) / uQ_real)) := by push_cast; ring
  have e2 : Real.pi * delta2 * (((-j : ℤ) : ℝ) / uQ_real)
              = -(Real.pi * delta2 * ((j : ℝ) / uQ_real)) := by push_cast; ring
  have e3 : Real.pi * delta3 * (((-j : ℤ) : ℝ) / uQ_real)
              = -(Real.pi * delta3 * ((j : ℝ) / uQ_real)) := by push_cast; ring
  rw [e1, e2, e3, Sidon.Bessel.besselJ0_even, Sidon.Bessel.besselJ0_even,
    Sidon.Bessel.besselJ0_even]

/-- The `S_cos` summand `g j := cData f j² · K̃(j)` is even in `j`. -/
private theorem sCosTerm_neg (f : ℝ → ℝ) (hf : Integrable f volume) (j : ℤ) :
    cData f (-j) ^ 2 * K_ms_fourier_lattice (-j)
      = cData f j ^ 2 * K_ms_fourier_lattice j := by
  rw [cData_neg f hf j, K_ms_fourier_lattice_neg j]

/-! ## Part 2 — summability of the `S_cos` family

`supp f ⊆ Ioo (-1/4, 1/4) ⊆ Ioc (-u/2, u/2)` (since `u = 0.638 > 1/2`),
so the period-`u` Plancherel sum-of-squares
`Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum` applies, giving
`Summable (j ↦ ‖(1/u)·𝓕f(j/u)‖²)`.  Pulling out the constant `(1/u)²`
gives `Summable (j ↦ ‖𝓕f(j/u)‖²)`, and `cData f j² ≤ ‖𝓕f(j/u)‖²`
(`re² ≤ norm²`) plus `K̃(j) ≤ 1` give summability of the `S_cos`
summand by comparison. -/

/-- `(-(u/2), u/2)` contains `(-1/4, 1/4)` for `u = uQ_real = 0.638`. -/
private theorem Ioo_subset_Ioc_half :
    Set.Ioo (-(1/4 : ℝ)) (1/4) ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
  have hu : uQ_real = 638 / 1000 := by unfold uQ_real uQ; push_cast; ring
  rw [hu]
  intro x hx
  constructor
  · linarith [hx.1]
  · linarith [hx.2]

/-- Summability of `j ↦ ‖𝓕f(j/u)‖²` for admissible `f`, from the
period-`u` Plancherel `HasSum`. -/
private theorem summable_normSq_fourier
    (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ => ‖Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)‖ ^ 2) := by
  have hu_pos : 0 < uQ_real := uQ_real_pos
  -- Complexify.
  set F : ℝ → ℂ := fun x => (f x : ℂ) with hF
  -- support and L² of `F` on the period window.
  have hsupp_F : Function.support F ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
    intro x hx
    have hx' : f x ≠ 0 := by
      simp only [hF, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
      exact hx
    exact Ioo_subset_Ioc_half (hf_supp (by simpa [Function.mem_support] using hx'))
  have hL2_F : MemLp F 2 (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
    have hFc : MemLp F 2 volume := by simpa [hF] using hf_L2.ofReal
    exact hFc.restrict _
  -- Plancherel HasSum for the scaled coefficient.
  have hHasSum := Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
    uQ_real hu_pos F hsupp_F hL2_F
  have hSummable_scaled :
      Summable (fun j : ℤ => ‖(1 / uQ_real : ℂ) * Real.fourierIntegral F (j / uQ_real : ℝ)‖ ^ 2) :=
    hHasSum.summable
  -- `‖(1/u)·z‖² = (1/u²)·‖z‖²`, so pull the constant out.
  have h_norm_eq : ∀ j : ℤ,
      ‖(1 / uQ_real : ℂ) * Real.fourierIntegral F (j / uQ_real : ℝ)‖ ^ 2
        = (1 / uQ_real ^ 2) * ‖Real.fourierIntegral F (j / uQ_real : ℝ)‖ ^ 2 := by
    intro j
    rw [norm_mul, mul_pow]
    congr 1
    have h_cast : ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) := by push_cast; rfl
    rw [h_cast, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (by positivity : (0 : ℝ) < 1 / uQ_real), div_pow, one_pow]
  rw [summable_congr h_norm_eq] at hSummable_scaled
  -- Divide out the nonzero constant `(1/u²)`.
  have hconst_ne : (1 / uQ_real ^ 2 : ℝ) ≠ 0 := by positivity
  have := (summable_mul_left_iff (a := (1 / uQ_real ^ 2 : ℝ)) hconst_ne).mp hSummable_scaled
  -- `F (j/u) = 𝓕f(j/u)` definitionally (`F = fun x => (f x : ℂ)`).
  exact this

/-- Summability of the `S_cos` summand `j ↦ cData f j² · K̃(j)` for
admissible `f` (comparison with `‖𝓕f(j/u)‖²` via `re² ≤ norm²` and
`K̃(j) ≤ 1`). -/
private theorem summable_sCosTerm
    (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ => cData f j ^ 2 * K_ms_fourier_lattice j) := by
  have h_maj := summable_normSq_fourier f hf_L2 hf_supp
  refine Summable.of_nonneg_of_le (fun j => ?_) (fun j => ?_) h_maj
  · -- nonneg
    exact mul_nonneg (sq_nonneg _) (K_ms_fourier_lattice_nonneg j)
  · -- `cData f j² · K̃(j) ≤ cData f j² ≤ ‖𝓕f(j/u)‖²`.
    have h1 : cData f j ^ 2 * K_ms_fourier_lattice j ≤ cData f j ^ 2 := by
      nlinarith [Sidon.Constructor.Glue.K_ms_fourier_lattice_le_one j,
        K_ms_fourier_lattice_nonneg j, sq_nonneg (cData f j)]
    have h2 : cData f j ^ 2
        ≤ ‖Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)‖ ^ 2 := by
      unfold cData
      have hre := Complex.abs_re_le_norm
        (Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real))
      nlinarith [hre, abs_nonneg
        (Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)).re,
        norm_nonneg (Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)),
        sq_abs (Real.fourierIntegral (fun x => (f x : ℂ)) ((j : ℝ) / uQ_real)).re]
    linarith [h1, h2]

/-- Summability of the `S_cos` family in its `if j = 0 then 0 else …`
form (matching `Sidon.MultiScale.S_cos`). -/
private theorem summable_sCos_if
    (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ => if j = 0 then (0 : ℝ) else
      cData f j ^ 2 * K_ms_fourier_lattice j) := by
  -- Differs from the unconditioned family only at `j = 0`.
  have hfin : {j : ℤ | cData f j ^ 2 * K_ms_fourier_lattice j
        ≠ (if j = 0 then (0 : ℝ) else cData f j ^ 2 * K_ms_fourier_lattice j)}.Finite := by
    refine Set.Finite.subset (Set.finite_singleton (0 : ℤ)) ?_
    intro j hj
    simp only [Set.mem_setOf_eq] at hj
    by_contra hjne
    apply hj
    rw [if_neg (by simpa using hjne)]
  exact (summable_congr_cofinite hfin).mp (summable_sCosTerm f hf_L2 hf_supp)

/-! ## Part 3 — the inner-product floor (finite trig polynomial)

`G_concrete x = ∑_{i<200} a_i cos(2π(i+1)x/u)` (`a_i = qpNumerators[i]/10⁸`).
The floor `∑_i a_i · cData f (i+1) ≥ min_G_analytic` is derived by:
  * `∫ f·G_concrete = ∑_i a_i · ∫ f cos(2π(i+1)x/u) = ∑_i a_i · cData f (i+1)`
    — FINITE-sum linearity + `fourierIntegral_re_eq_cos_integral` (NO Poisson);
  * `∫ f·G_concrete ≥ min_G_analytic · ∫ f = min_G_analytic` — pointwise
    `min_G_analytic · f x ≤ f x · G_concrete x` (`G_concrete ≥ min_G_analytic`
    on `[-1/4,1/4] ⊇ supp f`, using `G_concrete` even + `csInf_le`), `f ≥ 0`,
    `∫ f = 1`. -/

/-- `G_concrete` is even: `cos(2π(i+1)(-x)/u) = cos(2π(i+1)x/u)`. -/
private theorem G_concrete_even (x : ℝ) : G_concrete (-x) = G_concrete x := by
  unfold G_concrete
  refine Finset.sum_congr rfl (fun i _ => ?_)
  congr 1
  rw [show 2 * Real.pi * ((i + 1 : ℕ) : ℝ) * (-x) / uQ_real
        = -(2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) by ring, Real.cos_neg]

/-- `G_concrete` is continuous. -/
private theorem G_concrete_continuous : Continuous G_concrete := by
  unfold G_concrete; fun_prop

/-- The image `G_concrete '' [0,1/4]` is bounded below (compact image of a
compact set under a continuous map). -/
private theorem G_concrete_image_bddBelow :
    BddBelow (G_concrete '' Set.Icc (0 : ℝ) (1/4)) :=
  (isCompact_Icc.image G_concrete_continuous).bddBelow

/-- `min_G_analytic ≤ G_concrete x` for `x ∈ [0,1/4]` (`csInf_le`). -/
private theorem min_G_le_on_unit {x : ℝ} (hx : x ∈ Set.Icc (0 : ℝ) (1/4)) :
    min_G_analytic ≤ G_concrete x := by
  unfold min_G_analytic
  exact csInf_le G_concrete_image_bddBelow ⟨x, hx, rfl⟩

/-- `min_G_analytic ≤ G_concrete x` for `x ∈ [-1/4,1/4]` (via evenness). -/
private theorem min_G_le_on_quarter {x : ℝ} (hx : x ∈ Set.Icc (-(1/4 : ℝ)) (1/4)) :
    min_G_analytic ≤ G_concrete x := by
  -- `|x| ∈ [0,1/4]` and `G_concrete x = G_concrete |x|`.
  have habs : |x| ∈ Set.Icc (0 : ℝ) (1/4) := by
    rw [Set.mem_Icc]
    exact ⟨abs_nonneg x, by rw [abs_le]; exact ⟨hx.1, hx.2⟩⟩
  have hGabs : G_concrete x = G_concrete |x| := by
    rcases abs_choice x with h | h
    · rw [h]
    · rw [h, G_concrete_even]
  rw [hGabs]
  exact min_G_le_on_unit habs

/-- Each cosine term `f · (a_i cos(2π(i+1)x/u))` is integrable (`f`
integrable, cosine bounded). -/
private theorem integrable_f_cos (f : ℝ → ℝ) (hf : Integrable f volume) (i : ℕ) :
    Integrable (fun x => ((qpNumerators.getD i 0 : ℝ) / 100000000)
      * Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) * f x) volume := by
  -- `(a_i cos(...))` is bounded by `|a_i|`; times integrable `f`.
  -- `Integrable.bdd_mul` (receiver = integrable `f`) gives `Integrable (b·f)`.
  exact hf.bdd_mul
    (Continuous.aestronglyMeasurable (by fun_prop))
    (Filter.Eventually.of_forall (fun x => by
      rw [Real.norm_eq_abs, abs_mul]
      have hcos : |Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real)| ≤ 1 :=
        Real.abs_cos_le_one _
      calc |(qpNumerators.getD i 0 : ℝ) / 100000000|
            * |Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real)|
          ≤ |(qpNumerators.getD i 0 : ℝ) / 100000000| * 1 :=
            mul_le_mul_of_nonneg_left hcos (abs_nonneg _)
        _ = |(qpNumerators.getD i 0 : ℝ) / 100000000| := by ring))

/-- `f · G_concrete` is integrable. -/
private theorem integrable_f_G (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (fun x => f x * G_concrete x) volume := by
  have h_eq : (fun x => f x * G_concrete x)
      = (fun x => ∑ i ∈ Finset.range 200,
          ((qpNumerators.getD i 0 : ℝ) / 100000000)
            * Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) * f x) := by
    funext x
    unfold G_concrete
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun i _ => by ring)
  rw [h_eq]
  exact integrable_finset_sum _ (fun i _ => integrable_f_cos f hf i)

/-- **Inner-product identity.** `∫ f·G_concrete = ∑_i a_i · cData f (i+1)`,
a FINITE sum (no Poisson summation). -/
private theorem inner_product_eq (f : ℝ → ℝ) (hf : Integrable f volume) :
    ∫ x, f x * G_concrete x ∂volume
      = ∑ i ∈ Finset.range 200,
          ((qpNumerators.getD i 0 : ℝ) / 100000000) * cData f (i + 1) := by
  -- Expand `G_concrete` and swap `∫ ∑ = ∑ ∫`.
  have h_eq : (fun x => f x * G_concrete x)
      = (fun x => ∑ i ∈ Finset.range 200,
          ((qpNumerators.getD i 0 : ℝ) / 100000000)
            * Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) * f x) := by
    funext x
    unfold G_concrete
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun i _ => by ring)
  rw [h_eq, integral_finset_sum _ (fun i _ => integrable_f_cos f hf i)]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  -- `∫ a_i cos · f = a_i · ∫ f cos = a_i · cData f (i+1)`.
  set a : ℝ := (qpNumerators.getD i 0 : ℝ) / 100000000 with ha
  have h_pull :
      ∫ x, a * Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) * f x ∂volume
        = a * ∫ x, f x * Real.cos (2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real) ∂volume := by
    rw [← integral_const_mul]
    refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
    ring
  rw [h_pull]
  congr 1
  -- `∫ f cos(2π(i+1)x/u) = cData f (i+1)`.  (`cData f (i+1)` has `ℤ`-cast arg
  -- `((↑i+1:ℤ):ℝ)`; the cosine arg has `ℕ`-cast `((i+1:ℕ):ℝ)` — reconcile.)
  unfold cData
  rw [fourierIntegral_re_eq_cos_integral f hf ((((i : ℤ) + 1 : ℤ) : ℝ) / uQ_real)]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  simp only []
  have harg : 2 * Real.pi * ((i + 1 : ℕ) : ℝ) * x / uQ_real
                = 2 * Real.pi * x * ((((i : ℤ) + 1 : ℤ) : ℝ) / uQ_real) := by
    push_cast; ring
  rw [harg]

/-- **The inner-product floor.** `∑_i a_i · cData f (i+1) ≥ min_G_analytic`,
for admissible `f`. -/
private theorem inner_floor
    (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    (∑ i ∈ Finset.range 200,
        ((qpNumerators.getD i 0 : ℝ) / 100000000) * cData f (i + 1))
      ≥ min_G_analytic := by
  rw [← inner_product_eq f hf_int]
  -- Pointwise floor `min_G_analytic · f x ≤ f x · G_concrete x`.
  have h_pointwise : ∀ x, min_G_analytic * f x ≤ f x * G_concrete x := by
    intro x
    by_cases hx : x ∈ Function.support f
    · have hx_ioo := hf_supp hx
      have hx_icc : x ∈ Set.Icc (-(1/4 : ℝ)) (1/4) :=
        ⟨le_of_lt hx_ioo.1, le_of_lt hx_ioo.2⟩
      have hG : min_G_analytic ≤ G_concrete x := min_G_le_on_quarter hx_icc
      nlinarith [mul_le_mul_of_nonneg_right hG (hf_nonneg x), hf_nonneg x]
    · have hfx : f x = 0 := by by_contra h; exact hx h
      rw [hfx, mul_zero, zero_mul]
  have h_mG_int : Integrable (fun x => min_G_analytic * f x) volume := hf_int.const_mul _
  have h_mono : ∫ x, min_G_analytic * f x ∂volume ≤ ∫ x, f x * G_concrete x ∂volume :=
    integral_mono h_mG_int (integrable_f_G f hf_int) h_pointwise
  have h_LHS : ∫ x, min_G_analytic * f x ∂volume = min_G_analytic := by
    rw [integral_const_mul, hf_one]; ring
  rw [h_LHS] at h_mono
  exact h_mono

/-! ## Part 4 — finite Cauchy–Schwarz and the `±k` truncation

`mv_eq4_core` (Sedrakyan/Titu) on `J = range 200 ↦ (i+1)` gives
`(∑ a_i c_{i+1})² ≤ (∑ c_{i+1}² K̃(i+1)) · (∑ a_i²/K̃(i+1))`.  The second
factor is exactly `S_1_analytic` (after the `getD`/`Ktilde_ms`
reconciliation).  The full tsum `S_cos f` dominates twice the finite
truncation `X := ∑_{i<200} c_{i+1}² K̃(i+1)` (the `±k` pairing). -/

/-- The QP-active frequency set `{1,…,200} ⊆ ℤ`. -/
private def Jpos : Finset ℤ := Finset.Icc (1 : ℤ) 200

/-- Reindexing: `∑_{j ∈ Jpos} h j = ∑_{i ∈ range 200} h ((i:ℤ)+1)` (clean
`Finset.sum_bij` between `range 200` and `Icc 1 200`). -/
private theorem sum_Jpos_eq_range (h : ℤ → ℝ) :
    (∑ j ∈ Jpos, h j) = ∑ i ∈ Finset.range 200, h ((i : ℤ) + 1) := by
  unfold Jpos
  rw [Finset.sum_bij (fun (i : ℕ) (_ : i ∈ Finset.range 200) => ((i : ℤ) + 1))]
  · intro a ha; simp only [Finset.mem_range] at ha; simp only [Finset.mem_Icc]; omega
  · intro a _ b _ hab; simpa using hab
  · intro b hb; simp only [Finset.mem_Icc] at hb
    exact ⟨(b - 1).toNat, by simp only [Finset.mem_range]; omega, by omega⟩
  · intro a _; rfl

/-- `K̃(j) > 0` for every `j ∈ Jpos` (the active set), via `hK4_active`. -/
private theorem K_pos_on_Jpos : ∀ j ∈ Jpos, 0 < K_ms_fourier_lattice j := by
  intro j hj
  simp only [Jpos, Finset.mem_Icc] at hj
  exact Sidon.Constructor.LatticePositivity.hK4_active j hj.1 hj.2

/-- The finite truncation `X := ∑_{i<200} cData f (i+1)² · K̃(i+1)`, equal
to the `Jpos`-sum of the `S_cos` summand. -/
private theorem X_eq_Jpos_sum (f : ℝ → ℝ) :
    (∑ i ∈ Finset.range 200, cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1))
      = ∑ j ∈ Jpos, cData f j ^ 2 * K_ms_fourier_lattice j := by
  rw [sum_Jpos_eq_range (fun j => cData f j ^ 2 * K_ms_fourier_lattice j)]

/-- **Finite Cauchy–Schwarz** (`mv_eq4_core`): `(∑ a_i c_{i+1})² ≤ X · S_1`. -/
private theorem finite_CS (f : ℝ → ℝ) :
    (∑ i ∈ Finset.range 200,
        ((qpNumerators.getD i 0 : ℝ) / 100000000) * cData f (i + 1)) ^ 2
      ≤ (∑ i ∈ Finset.range 200, cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1))
          * S_1_analytic := by
  -- Coefficient family on ℤ: `aQ j = qpNumerators[j-1]/10⁸`.
  set aQ : ℤ → ℝ := fun j => (qpNumerators.getD (j.toNat - 1) 0 : ℝ) / 100000000 with haQ
  -- `aQ ((i:ℤ)+1) = qpNumerators.getD i 0 / 1e8`  (since `(↑i+1).toNat - 1 = i`).
  have htoNat : ∀ i : ℕ, ((i : ℤ) + 1).toNat - 1 = i := by intro i; omega
  have haQ_eval : ∀ i : ℕ, aQ ((i : ℤ) + 1) = (qpNumerators.getD i 0 : ℝ) / 100000000 := by
    intro i; simp only [haQ, htoNat i]
  -- Core C-S on `Jpos` with weights `K_ms_fourier_lattice`.
  have hcore := Sidon.MV.mv_eq4_core (cData f) aQ K_ms_fourier_lattice Jpos K_pos_on_Jpos
  -- LHS: `∑ cData·aQ` over Jpos  =  `∑ aᵢ·cData(i+1)` over range.
  have hL : (∑ j ∈ Jpos, cData f j * aQ j)
              = ∑ i ∈ Finset.range 200,
                  ((qpNumerators.getD i 0 : ℝ) / 100000000) * cData f (i + 1) := by
    rw [sum_Jpos_eq_range (fun j => cData f j * aQ j)]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [haQ_eval i, mul_comm]
  -- middle factor: `∑ cData²·K̃` over Jpos = X.
  have hM : (∑ j ∈ Jpos, (cData f j) ^ 2 * K_ms_fourier_lattice j)
              = ∑ i ∈ Finset.range 200, cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1) :=
    (X_eq_Jpos_sum f).symm
  -- right factor: `∑ aQ²/K̃` over Jpos = S_1_analytic.
  have hR : (∑ j ∈ Jpos, (aQ j) ^ 2 / K_ms_fourier_lattice j) = S_1_analytic := by
    rw [sum_Jpos_eq_range (fun j => (aQ j) ^ 2 / K_ms_fourier_lattice j)]
    unfold S_1_analytic
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [haQ_eval i]
    congr 1
    -- `K_ms_fourier_lattice (↑i+1) = Ktilde_ms (i+1)`.
    rw [show ((i : ℤ) + 1) = (((i + 1 : ℕ)) : ℤ) by push_cast; ring,
      ← Sidon.Constructor.Ktilde_ms_eq_fourier_lattice (i + 1)]
  rw [hL] at hcore
  rw [hM, hR] at hcore
  exact hcore

/-- **The `±k` truncation.** The full tsum `S_cos f` dominates twice the
finite truncation `X`: `S_cos f ≥ 2 · X`, using the `±k` evenness of the
summand and nonneg truncation (`Summable.sum_le_tsum`). -/
private theorem two_X_le_S_cos
    (f : ℝ → ℝ) (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    2 * (∑ i ∈ Finset.range 200, cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1))
      ≤ S_cos f := by
  classical
  -- The summand family `g j = if j=0 then 0 else cData f j² · K̃(j)`.
  set g : ℤ → ℝ := fun j => if j = 0 then (0 : ℝ) else
      cData f j ^ 2 * K_ms_fourier_lattice j with hg
  have hg_summable : Summable g := summable_sCos_if f hf_L2 hf_supp
  have hg_nonneg : ∀ j, 0 ≤ g j := fun j => by
    simp only [hg]; exact S_cos_summand_nonneg f j
  -- The finite index set `S = {±1,…,±200}`.
  set Sset : Finset ℤ := Jpos ∪ Jpos.image (fun j => -j) with hSset
  -- `S_cos f = ∑' g`, and `∑_{j∈S} g j ≤ ∑' g`.
  have h_tsum_ge : (∑ j ∈ Sset, g j) ≤ S_cos f := by
    have : S_cos f = ∑' j, g j := by unfold S_cos; rfl
    rw [this]
    exact hg_summable.sum_le_tsum Sset (fun j _ => hg_nonneg j)
  -- `∑_{j∈S} g j = 2·X`: disjoint union of `Jpos` (positive) and `-Jpos`,
  -- and `g (-k) = g k` for `k ∈ Jpos`.
  have h_disj : Disjoint Jpos (Jpos.image (fun j => -j)) := by
    rw [Finset.disjoint_left]
    intro a ha hna
    simp only [Jpos, Finset.mem_image, Finset.mem_Icc] at ha hna
    obtain ⟨k, hk, hak⟩ := hna
    omega
  have hg_even : ∀ k ∈ Jpos, g (-k) = g k := by
    intro k hk
    simp only [Jpos, Finset.mem_Icc] at hk
    simp only [hg]
    rw [if_neg (by omega : -k ≠ 0), if_neg (by omega : k ≠ 0)]
    exact sCosTerm_neg f hf_int k
  -- Sum over the union.
  have h_sum_union :
      (∑ j ∈ Sset, g j) = (∑ j ∈ Jpos, g j) + ∑ j ∈ Jpos.image (fun j => -j), g j := by
    rw [hSset, Finset.sum_union h_disj]
  have h_sum_neg :
      (∑ j ∈ Jpos.image (fun j => -j), g j) = ∑ k ∈ Jpos, g k := by
    rw [Finset.sum_image (by intro a _ b _ hab; simpa using hab)]
    exact Finset.sum_congr rfl hg_even
  have h_S_eq_2X : (∑ j ∈ Sset, g j) = 2 * ∑ j ∈ Jpos, g j := by
    rw [h_sum_union, h_sum_neg]; ring
  -- `∑_{j∈Jpos} g j = X` (drop the `if`, reindex).
  have hX_eq : (∑ j ∈ Jpos, g j)
      = ∑ i ∈ Finset.range 200, cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1) := by
    have h_drop : (∑ j ∈ Jpos, g j) = ∑ j ∈ Jpos, cData f j ^ 2 * K_ms_fourier_lattice j := by
      refine Finset.sum_congr rfl (fun j hj => ?_)
      simp only [hg, Jpos, Finset.mem_Icc] at hj ⊢
      rw [if_neg (by omega : j ≠ 0)]
    rw [h_drop, ← X_eq_Jpos_sum f]
  rw [h_S_eq_2X, hX_eq] at h_tsum_ge
  exact h_tsum_ge

/-! ## The Cauchy–Schwarz floor `h_cs`

Assembling Parts 3–4: from the floor `Φ := ∑ aᵢ·cData f (i+1) ≥
min_G_analytic ≥ 0`, square to `Φ² ≥ min_G_analytic²`; finite C-S gives
`Φ² ≤ X · S_1_analytic`; so `min_G_analytic² ≤ X · S_1_analytic`, i.e.
`X ≥ min_G_analytic²/S_1_analytic`; and `S_cos f ≥ 2X`, whence
`S_cos f ≥ 2·min_G_analytic²/S_1_analytic`.

The single numerical input is `hmin_nonneg : 0 ≤ min_G_analytic` — the
`flint.arb` / `G_min.py` certifier output `min G ≈ 0.99997 > 0` (paper
Lemma 4.3).  It is genuinely *independent* of the two existing numerical
axioms, which only constrain `min_G_analytic²` (via the gain functional)
and hence force `min_G_analytic ≠ 0` but not its sign. -/

/-- **MV Eq.(4) Cauchy–Schwarz floor (`h_cs`).**  For admissible `f` and
the certifier sign `0 ≤ min_G_analytic`,
  `S_cos f ≥ 2 · min_G_analytic² / S_1_analytic`. -/
theorem cauchy_schwarz_floor
    (f : ℝ → ℝ) (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hmin_nonneg : 0 ≤ min_G_analytic) :
    S_cos f ≥ 2 * min_G_analytic ^ 2 / S_1_analytic := by
  set Φ : ℝ := ∑ i ∈ Finset.range 200,
      ((qpNumerators.getD i 0 : ℝ) / 100000000) * cData f (i + 1) with hΦ
  set X : ℝ := ∑ i ∈ Finset.range 200,
      cData f (i + 1) ^ 2 * K_ms_fourier_lattice ((i : ℤ) + 1) with hX
  have hS1_pos : 0 < S_1_analytic := Sidon.Constructor.S_1_analytic_pos
  -- floor: Φ ≥ min ≥ 0.
  have hfloor : Φ ≥ min_G_analytic := inner_floor f hf_int hf_supp hf_nonneg hf_one
  -- Φ² ≥ min².
  have hΦsq : min_G_analytic ^ 2 ≤ Φ ^ 2 := by
    have h := mul_self_le_mul_self hmin_nonneg hfloor
    nlinarith [h]
  -- C-S: Φ² ≤ X·S_1.
  have hCS : Φ ^ 2 ≤ X * S_1_analytic := finite_CS f
  -- min² ≤ X·S_1  ⟹  X ≥ min²/S_1.
  have hXge : X ≥ min_G_analytic ^ 2 / S_1_analytic := by
    rw [ge_iff_le, div_le_iff₀ hS1_pos]
    linarith [hΦsq, hCS]
  -- S_cos ≥ 2X ≥ 2 min²/S_1.
  have h2X : 2 * X ≤ S_cos f := two_X_le_S_cos f hf_int hf_L2 hf_supp
  rw [ge_iff_le]
  have : 2 * min_G_analytic ^ 2 / S_1_analytic = 2 * (min_G_analytic ^ 2 / S_1_analytic) := by
    ring
  rw [this]
  linarith [h2X, hXge]

end

end Sidon.Constructor
