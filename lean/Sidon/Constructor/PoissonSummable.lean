/-
Sidon Autocorrelation Project — Summability of the period-`u` Fourier family.
================================================================================

This file discharges, **for general admissible `f`**, the two summability
side-conditions that `Sidon.Constructor.PoissonSampling` threads through its
capstone theorems
`field_P3_periodUParsevalTrue_via_periodization` and
`field_hEq3_ge_via_periodization`:

  * `h_summable`    : `Summable (j ↦ 𝓕g(j/u) · conj(𝓕K_ms(j/u)))`
  * `h_summable_re` : `Summable (j ↦ 2·(Re 𝓕f(j/u))²·(𝓕K_ms(j/u)).re)`

where `g = (f*f) + autocorr f` (`g_complex f`).

================================================================================
## The route: period-`u` Plancherel for one-period-supported `L²` functions
================================================================================

The single non-elementary input is the **period-`u` Plancherel
sum-of-squares**.  mathlib's `hasSum_sq_fourierCoeffOn` states, for any
`h ∈ L²(Ioc a b)` (no support hypothesis),

  `HasSum (i ↦ ‖fourierCoeffOn hab h i‖²) ((b-a)⁻¹ • ∫_a^b ‖h‖²)`,

and `fourierCoeffOn hab h i = fourierCoeff (liftIoc (b-a) a h) i` by `rfl`.

Applied to the **periodisation** `g_per := periodization u (g_complex f)`
(which is in `L²` on one period — `PoissonSampling.periodization_memLp_period`),
the now-proven period-`u` Poisson sampling identity
`PoissonSampling.period_u_poisson_sampling` gives
`fourierCoeff (liftIoc u (-(u/2)) g_per) j = (1/u)·𝓕g(j/u)`, so
`‖fourierCoeffOn _ g_per j‖² = (1/u²)·‖𝓕g(j/u)‖²`.  Dividing out the
constant `1/u²` yields the building block

  `Summable (j ↦ ‖𝓕g(j/u)‖²)`.

For the period-fitting kernel `K_ms` (supported in `(-δ₁, δ₁) ⊆ Ioc(-u/2,u/2)`)
`Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum` gives directly
`Summable (j ↦ ‖𝓕K_ms(j/u)‖²)`.

The two target families then follow by comparison:

  * `h_summable` (complex products): `‖𝓕g(j/u)·conj(𝓕K_ms(j/u))‖
    ≤ ½(‖𝓕g(j/u)‖² + ‖𝓕K_ms(j/u)‖²)` (AM–GM), and `Summable.of_norm`.
  * `h_summable_re` (real reduction): `(𝓕K_ms(j/u)).re = K̃_ms(j) ∈ [0,1]`
    (`MOLemma21.bessel_identity` + `K_ms_fourier_lattice` bounds) and
    `(Re 𝓕f(j/u))² ≤ ‖𝓕f(j/u)‖²` (`Complex.abs_re_le_norm`), so the
    summand is squeezed `0 ≤ · ≤ 2·‖𝓕f(j/u)‖²` and dominated by the
    `f`-Plancherel family `Summable (j ↦ ‖𝓕f(j/u)‖²)`.

No `sorry`, no new `axiom`.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.TorusParseval
import Sidon.Constructor.MOLemma21
import Sidon.Constructor.PoissonSampling
import Sidon.Constructor.Glue

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical Pointwise

namespace Sidon.Constructor.PoissonSummable

open Sidon.MultiScale
open Sidon.Constructor.MOLemma21 (g_complex K_ms_complex bessel_identity)
open Sidon.Constructor.PoissonSampling (periodization periodization_memLp_period
  period_u_poisson_sampling)

noncomputable section

/-! ## Building block 1 — `Summable (j ↦ ‖𝓕g(j/u)‖²)`

Period-`u` Plancherel on the periodisation `g_per`, then the proven Poisson
sampling identity to reindex onto `𝓕g(j/u)`. -/

/-- `fourierCoeffOn hab h j = fourierCoeff (liftIoc u a h) j` when `b - a = u`.
A re-export (with `u a b` as variables, so the dependent `Fact (0 < b-a)`
instance is `subst`-able) of mathlib's definitional unfolding of
`fourierCoeffOn`.  Mirrors the proven `Sidon.TorusParseval`-internal helper. -/
theorem fourierCoeffOn_eq_fourierCoeff_liftIoc_of_eq
    (u a b : ℝ) (hab : a < b) (huba : b - a = u) (h : ℝ → ℂ) (j : ℤ) :
    haveI : Fact (0 < u) := ⟨huba ▸ sub_pos.mpr hab⟩
    fourierCoeffOn hab h j = fourierCoeff (AddCircle.liftIoc u a h) j := by
  haveI hu_fact : Fact (0 < b - a) := ⟨sub_pos.mpr hab⟩
  haveI : Fact (0 < u) := ⟨huba ▸ hu_fact.out⟩
  have hdef : fourierCoeffOn hab h j = fourierCoeff (AddCircle.liftIoc (b - a) a h) j := rfl
  rw [hdef]; subst huba; rfl

/-- The scaled period-`u` Plancherel `HasSum` for the periodisation `g_per`:
`HasSum (j ↦ ‖(1/u)·𝓕g(j/u)‖²) (u⁻¹ • ∫_{-u/2}^{u/2} ‖g_per‖²)`.
Obtained from mathlib's `hasSum_sq_fourierCoeffOn` (no support hypothesis,
only `g_per ∈ L²(period)`) plus `period_u_poisson_sampling`. -/
theorem hasSum_normSq_scaled_fourierIntegral_g
    (f : ℝ → ℝ) (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    HasSum
      (fun j : ℤ => ‖(1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2)
      (uQ_real⁻¹ • ∫ x in (-(uQ_real/2))..(uQ_real/2),
        ‖periodization uQ_real (g_complex f) x‖ ^ 2) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  -- `g_per ∈ L²(period)`.
  have hgper_L2 : MemLp (periodization uQ_real (g_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    periodization_memLp_period f hf_int hf_L2 hf_supp
  have hab : -(uQ_real/2) < uQ_real/2 := by linarith [uQ_real_pos]
  -- mathlib's support-free period Plancherel for `g_per` on `Ioc (-(u/2)) (u/2)`.
  have hHS := hasSum_sq_fourierCoeffOn (a := -(uQ_real/2)) (b := uQ_real/2)
    (f := periodization uQ_real (g_complex f)) hab hgper_L2
  -- `(b - a)⁻¹ = u⁻¹`.
  have h_diff : (uQ_real/2 - -(uQ_real/2) : ℝ) = uQ_real := by ring
  rw [h_diff] at hHS
  -- Reindex each `fourierCoeffOn` to the scaled `𝓕g(j/u)` via Poisson sampling.
  have h_coef : ∀ j : ℤ,
      ‖fourierCoeffOn (a := -(uQ_real/2)) (b := uQ_real/2) hab
          (periodization uQ_real (g_complex f)) j‖ ^ 2
        = ‖(1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2 := by
    intro j
    congr 1
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    have hgint : Integrable (g_complex f) volume :=
      Sidon.Constructor.MOLemma21.g_complex_integrable f hf_int
    -- The proven period-`u` Poisson sampling identity (stated for `liftIoc uQ_real`).
    have hpp := period_u_poisson_sampling uQ_real_pos (g_complex f) hgint j
    -- `fourierCoeffOn hab g_per j` is *defeq* `fourierCoeff (liftIoc (b-a) (-(u/2)) g_per) j`;
    -- the `b - a = u` rewrite (under the dependent `Fact` instance) is handled by `simp only`.
    have hco : fourierCoeffOn (a := -(uQ_real/2)) (b := uQ_real/2) hab
          (periodization uQ_real (g_complex f)) j
        = fourierCoeff (AddCircle.liftIoc uQ_real (-(uQ_real/2))
            (periodization uQ_real (g_complex f))) j :=
      fourierCoeffOn_eq_fourierCoeff_liftIoc_of_eq uQ_real (-(uQ_real/2)) (uQ_real/2)
        hab h_diff (periodization uQ_real (g_complex f)) j
    rw [hco, hpp]
  rw [show (fun j : ℤ => ‖fourierCoeffOn (a := -(uQ_real/2)) (b := uQ_real/2) hab
            (periodization uQ_real (g_complex f)) j‖ ^ 2)
        = (fun j : ℤ => ‖(1 / uQ_real : ℂ)
            * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2)
      from funext h_coef] at hHS
  exact hHS

/-- **Building block 1.** `Summable (j ↦ ‖𝓕g(j/u)‖²)` for admissible `f`. -/
theorem summable_normSq_fourierIntegral_g
    (f : ℝ → ℝ) (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ => ‖Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2) := by
  have hu_pos : 0 < uQ_real := uQ_real_pos
  have hSummable_scaled :
      Summable (fun j : ℤ =>
        ‖(1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2) :=
    (hasSum_normSq_scaled_fourierIntegral_g f hf_int hf_L2 hf_supp).summable
  -- `‖(1/u)·z‖² = (1/u²)·‖z‖²`, divide the nonzero constant out.
  have h_norm_eq : ∀ j : ℤ,
      ‖(1 / uQ_real : ℂ) * Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2
        = (1 / uQ_real ^ 2) * ‖Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2 := by
    intro j
    rw [norm_mul, mul_pow]
    congr 1
    have h_cast : ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) := by push_cast; rfl
    rw [h_cast, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (by positivity : (0 : ℝ) < 1 / uQ_real), div_pow, one_pow]
  rw [summable_congr h_norm_eq] at hSummable_scaled
  have hconst_ne : (1 / uQ_real ^ 2 : ℝ) ≠ 0 := by positivity
  exact (summable_mul_left_iff (a := (1 / uQ_real ^ 2 : ℝ)) hconst_ne).mp hSummable_scaled

/-! ## Building block 2 — `Summable (j ↦ ‖𝓕K_ms(j/u)‖²)`

`K_ms` fits one period, so `Sidon.TorusParseval.plancherel_at_lattice_period_u`
applies directly. -/

/-- **Building block 2.** `Summable (j ↦ ‖𝓕K_ms(j/u)‖²)`. -/
theorem summable_normSq_fourierIntegral_Kms :
    Summable (fun j : ℤ => ‖Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ ^ 2) := by
  have hu_pos : 0 < uQ_real := uQ_real_pos
  have hsupp : Function.support K_ms_complex ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) :=
    Sidon.Constructor.MOLemma21.K_ms_complex_support
  have hL2 : MemLp K_ms_complex 2 (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict
  have hHS := Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
    uQ_real hu_pos K_ms_complex hsupp hL2
  have hSummable_scaled :
      Summable (fun j : ℤ =>
        ‖(1 / uQ_real : ℂ) * Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ ^ 2) :=
    hHS.summable
  have h_norm_eq : ∀ j : ℤ,
      ‖(1 / uQ_real : ℂ) * Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ ^ 2
        = (1 / uQ_real ^ 2) * ‖Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ ^ 2 := by
    intro j
    rw [norm_mul, mul_pow]
    congr 1
    have h_cast : ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) := by push_cast; rfl
    rw [h_cast, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (by positivity : (0 : ℝ) < 1 / uQ_real), div_pow, one_pow]
  rw [summable_congr h_norm_eq] at hSummable_scaled
  have hconst_ne : (1 / uQ_real ^ 2 : ℝ) ≠ 0 := by positivity
  exact (summable_mul_left_iff (a := (1 / uQ_real ^ 2 : ℝ)) hconst_ne).mp hSummable_scaled

/-! ## Building block 3 — `Summable (j ↦ ‖𝓕f(j/u)‖²)`

`supp f ⊆ Ioo(-1/4,1/4) ⊆ Ioc(-u/2,u/2)`, so the period-`u` Plancherel
applies to the complexified `f` directly. -/

/-- `(-(u/2), u/2)` contains `(-1/4, 1/4)` for `u = uQ_real = 0.638`. -/
private theorem Ioo_subset_Ioc_half :
    Set.Ioo (-(1/4 : ℝ)) (1/4) ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
  have hu : uQ_real = 638 / 1000 := by unfold uQ_real uQ; push_cast; ring
  rw [hu]
  intro x hx
  exact ⟨by linarith [hx.1], by linarith [hx.2]⟩

/-- **Building block 3.** `Summable (j ↦ ‖𝓕f(j/u)‖²)` for admissible `f`. -/
theorem summable_normSq_fourierIntegral_f
    (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ =>
      ‖Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)‖ ^ 2) := by
  have hu_pos : 0 < uQ_real := uQ_real_pos
  set F : ℝ → ℂ := fun x => (f x : ℂ) with hF
  have hsupp_F : Function.support F ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
    intro x hx
    have hx' : f x ≠ 0 := by
      simp only [hF, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
      exact hx
    exact Ioo_subset_Ioc_half (hf_supp (by simpa [Function.mem_support] using hx'))
  have hL2_F : MemLp F 2 (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
    have hFc : MemLp F 2 volume := by simpa [hF] using hf_L2.ofReal
    exact hFc.restrict _
  have hHS := Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
    uQ_real hu_pos F hsupp_F hL2_F
  have hSummable_scaled :
      Summable (fun j : ℤ => ‖(1 / uQ_real : ℂ) * Real.fourierIntegral F (j / uQ_real : ℝ)‖ ^ 2) :=
    hHS.summable
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
  have hconst_ne : (1 / uQ_real ^ 2 : ℝ) ≠ 0 := by positivity
  exact (summable_mul_left_iff (a := (1 / uQ_real ^ 2 : ℝ)) hconst_ne).mp hSummable_scaled

/-! ## Target 1 — `h_summable` (the complex products)

`‖𝓕g(j/u)·conj(𝓕K_ms(j/u))‖ = ‖𝓕g(j/u)‖·‖𝓕K_ms(j/u)‖
  ≤ ½(‖𝓕g(j/u)‖² + ‖𝓕K_ms(j/u)‖²)` (AM–GM), and the RHS family is summable
(blocks 1 + 2).  Hence absolute summability ⟹ summability. -/

/-- **Target 1 (verbatim `PoissonSampling.field_*` hypothesis `h_summable`).**
`Summable (j ↦ 𝓕g(j/u) · conj(𝓕K_ms(j/u)))`. -/
theorem summable_fourier_product
    (f : ℝ → ℝ) (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ =>
        Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))) := by
  -- Majorant: `½(‖𝓕g‖² + ‖𝓕K_ms‖²)`.
  have h_g := summable_normSq_fourierIntegral_g f hf_int hf_L2 hf_supp
  have h_K := summable_normSq_fourierIntegral_Kms
  have h_maj : Summable (fun j : ℤ =>
      (1/2) * (‖Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ ^ 2
        + ‖Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ ^ 2)) :=
    ((h_g.add h_K).mul_left (1/2))
  -- Absolute summability via AM–GM term bound.
  refine Summable.of_norm (Summable.of_nonneg_of_le (fun j => ?_) (fun j => ?_) h_maj)
  · positivity
  · -- `‖𝓕g·conj(𝓕K_ms)‖ = ‖𝓕g‖·‖𝓕K_ms‖ ≤ ½(‖𝓕g‖² + ‖𝓕K_ms‖²)`.
    rw [norm_mul, RCLike.norm_conj]
    set A : ℝ := ‖Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)‖ with hA
    set B : ℝ := ‖Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)‖ with hB
    nlinarith [sq_nonneg (A - B), norm_nonneg
      (Real.fourierIntegral (g_complex f) (j / uQ_real : ℝ)),
      norm_nonneg (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))]

/-! ## Target 2 — `h_summable_re` (the real reduction)

`(𝓕K_ms(j/u)).re = K̃_ms(j) ∈ [0,1]` (Fourier–Bessel + lattice bounds) and
`(Re 𝓕f(j/u))² ≤ ‖𝓕f(j/u)‖²`, so the summand is squeezed
`0 ≤ 2·(Re𝓕f)²·(𝓕K_ms).re ≤ 2·‖𝓕f(j/u)‖²` and dominated by block 3. -/

/-- **Target 2 (verbatim `PoissonSampling.field_*` hypothesis `h_summable_re`).**
`Summable (j ↦ 2·(Re 𝓕f(j/u))²·(𝓕K_ms(j/u)).re)`. -/
theorem summable_fourier_re
    (f : ℝ → ℝ) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ =>
        2 * ((Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re) ^ 2
          * (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re) := by
  -- Majorant `2·‖𝓕f(j/u)‖²` (block 3 scaled).
  have h_f := summable_normSq_fourierIntegral_f f hf_L2 hf_supp
  have h_maj : Summable (fun j : ℤ =>
      2 * ‖Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)‖ ^ 2) :=
    h_f.mul_left 2
  refine Summable.of_nonneg_of_le (fun j => ?_) (fun j => ?_) h_maj
  · -- nonneg: `(𝓕K_ms).re = K̃(j) ≥ 0`.
    have hKre : (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re
        = K_ms_fourier_lattice j := bessel_identity j
    rw [hKre]
    have hK0 : 0 ≤ K_ms_fourier_lattice j := K_ms_fourier_lattice_nonneg j
    positivity
  · -- `2·(Re𝓕f)²·K̃(j) ≤ 2·(Re𝓕f)² ≤ 2·‖𝓕f‖²`, using `0 ≤ K̃ ≤ 1` and `re² ≤ norm²`.
    have hKre : (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)).re
        = K_ms_fourier_lattice j := bessel_identity j
    rw [hKre]
    have hK1 : K_ms_fourier_lattice j ≤ 1 :=
      Sidon.Constructor.Glue.K_ms_fourier_lattice_le_one j
    have hK0 : 0 ≤ K_ms_fourier_lattice j := K_ms_fourier_lattice_nonneg j
    have hre := Complex.abs_re_le_norm
      (Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ))
    set c : ℝ := (Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)).re with hc
    set N : ℝ := ‖Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)‖ with hN
    have hcsq : c ^ 2 ≤ N ^ 2 := by
      have : |c| ≤ N := hre
      nlinarith [this, abs_nonneg c, norm_nonneg
        (Real.fourierIntegral (fun x => (f x : ℂ)) (j / uQ_real : ℝ)), sq_abs c]
    nlinarith [hcsq, hK0, hK1, sq_nonneg c]

end

end Sidon.Constructor.PoissonSummable
