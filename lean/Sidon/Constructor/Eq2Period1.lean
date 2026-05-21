/-
Sidon Autocorrelation Project — MV Eq.(2) via the PERIOD-1 Parseval split.
==========================================================================

This file discharges the bundle field `hEq2` of the unconditional Sidon
headline `Sidon.MultiScale.C1a_ge_1292_unconditional` **fully from
admissibility**, eliminating the previously-residual hypotheses
`h_parseval_split` and `h_F_bound`.

================================================================================
## Why PERIOD 1 (not period `u`)
================================================================================

The deepest obstruction recorded in `Sidon.Constructor.Eq2Split` was that
`autocorr f` and `f*f` have support up to `(-1/2, 1/2)`, which *overflows*
the period-`u` window `(-u/2, u/2) = (-0.319, 0.319)`.  The resolution is
that BOTH `autocorr f` and `K_ms` (supp `⊆ (-δ₁,δ₁) ⊆ (-1/2,1/2)`) fit a
**length-1** period.  So the bilinear period-`u` Parseval workhorse
`Sidon.BilinearParseval.bilinear_parseval_period_u`, applied at `u = 1`,
pairs `autocorr f` against `K_ms` *directly* — no periodisation needed:

  `∫_ℝ autocorr f · K_ms
     = ∑'_{r∈ℤ} 𝓕(autocorr f)(r) · conj(𝓕K_ms(r))`
     = ∑'_{r∈ℤ} ‖𝓕f(r)‖² · (𝓕K_ms(r)).re`               (Wiener–Khinchin)
     = 1 + ∑'_{r≠0} ‖𝓕f(r)‖² · (𝓕K_ms(r)).re`            (`r=0` term `= 1·1`).

The constant `1` matches MV's Eq.(2) split exactly (the period-`u` split of
`Eq2Split.autocorr_period_u_split` carried `1/u`; that was the wrong
normalisation for the *bound*).

================================================================================
## The two `L²`-Fourier moment bounds (now PROVED)
================================================================================

  * **K-bound** `∑'_{r≠0}(𝓕K_ms(r)).re² ≤ K2_analytic − 1` — period-1
    Plancherel for `K_ms` (`∑'_r ‖𝓕K_ms(r)‖² = K2_analytic`,
    `Sidon.BundleEq2Schwartz.plancherel_K_ms_K2_analytic_period1`) minus the
    `r=0` term `=1`; `‖𝓕K_ms(r)‖² = (𝓕K_ms(r)).re²` since `𝓕K_ms` is real.

  * **F-bound** `∑'_{r≠0}‖𝓕f(r)‖⁴ ≤ R(f) − 1` — `‖𝓕f(r)‖⁴ = ‖𝓕(f*f)(r)‖²`
    (`𝓕(f*f)=(𝓕f)²`, `ConvFourier.fourierIntegral_autoconvolution_real`),
    period-1 Plancherel for `f*f` (which ALSO fits one period:
    supp `(f*f) ⊆ (-1/2,1/2)`) gives `∑'_r ‖𝓕(f*f)(r)‖² = ∫_{-1/2}^{1/2}|f*f|²
    = ∫_ℝ (f*f)²`, and the Fourier–Hölder bound `∫(f*f)² ≤ ‖f*f‖_∞·∫(f*f) =
    R(f)·1` (a.e. bound `f*f ≤ ‖f*f‖_∞` from `ae_le_essSup` + `∫(f*f)=1`),
    minus the `r=0` term `‖𝓕(f*f)(0)‖² = (∫ f*f)² = 1`.

The tail Cauchy–Schwarz uses finite-sum CS (`Real.sum_mul_le_sqrt_mul_sqrt`)
on every partial sum plus `Real.tsum_le_of_sum_le` (nonneg-term family).

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.MOLemma21
import Sidon.Constructor.ConvFourier
import Sidon.Constructor.FieldsEasy
import Sidon.Constructor.FieldsParseval
import Sidon.Constructor.Eq2Split
import Sidon.BilinearParseval
import Sidon.BundleEq2Schwartz
import Sidon.TorusParseval
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical Pointwise

namespace Sidon.Constructor.Eq2Period1

open Sidon.MultiScale
open Sidon.FourierAux (autocorr autocorr_def)
open Sidon.Constructor.MOLemma21 (K_ms_complex K_ms_complex_memLp_volume
  conj_fourierIntegral_K_ms fourierIntegral_K_ms_eq_re K_ms_zero_outside
  autocorr_abs_le_local autocorr_integrable_local)

noncomputable section

/-! ## Period-1 support and `L²` facts -/

/-- `δ₁ < 1/2` (`δ₁ = 0.138`). -/
theorem delta1_lt_half : delta1 < (1/2 : ℝ) := by
  unfold delta1 delta1Q; push_cast; norm_num

/-- `K_ms_complex` is supported in `Ioo (-1/2) (1/2)` (kernel half-width
`δ₁ = 0.138 < 1/2`). -/
theorem K_ms_complex_supp_half :
    Function.support K_ms_complex ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  have hx_ne : K_ms x ≠ 0 := by
    simp only [K_ms_complex, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
    exact hx
  -- `K_ms x ≠ 0` ⟹ `|x| < δ₁ < 1/2`.
  by_contra hxout
  rw [Set.mem_Ioo, not_and_or, not_lt, not_lt] at hxout
  have hδ1 : delta1 ≤ |x| := by
    rcases hxout with h | h
    · rw [abs_of_nonpos (by linarith [delta1_lt_half])]
      linarith [delta1_lt_half]
    · rw [le_abs]; left; linarith [delta1_lt_half]
  exact hx_ne (K_ms_zero_outside x hδ1)

/-- `K_ms_complex ∈ L²(volume)` (re-export). -/
theorem K_ms_complex_memLp2 : MemLp K_ms_complex 2 volume := K_ms_complex_memLp_volume

/-- The complex lift of `autocorr f`. -/
def autocorr_complex (f : ℝ → ℝ) : ℝ → ℂ := fun x => ((autocorr f x : ℝ) : ℂ)

/-- `autocorr_complex f` is integrable. -/
theorem autocorr_complex_integrable (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (autocorr_complex f) volume :=
  (autocorr_integrable_local f hf).ofReal

/-- Support of `autocorr f ⊆ Ioo (-1/2) (1/2)` for admissible `f`.  (Reuse the
proof from `Sidon.Constructor.Eq2Split`.) -/
theorem autocorr_support_subset {f : ℝ → ℝ}
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (autocorr f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
  Sidon.Constructor.Eq2Split.autocorr_support_subset hf_supp

/-- Support of `autocorr_complex f ⊆ Ioo (-1/2) (1/2)`. -/
theorem autocorr_complex_support {f : ℝ → ℝ}
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (autocorr_complex f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  have hx_ne : autocorr f x ≠ 0 := by
    simp only [autocorr_complex, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
    exact hx
  exact autocorr_support_subset hf_supp (by simpa [Function.mem_support] using hx_ne)

/-- `autocorr_complex f ∈ L²(volume)`: bounded everywhere by `∫f²` and
compactly supported in `Ioo (-1/2) (1/2)`. -/
theorem autocorr_complex_memLp2 (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    MemLp (autocorr_complex f) 2 volume := by
  -- bounded by `C := ∫f²`, support in the finite-measure interval `Ioo (-1/2) (1/2)`.
  set C : ℝ := ∫ x, f x ^ 2 ∂volume with hC_def
  -- `autocorr_complex f = 0` outside `Ioo (-1/2) (1/2)`; on it, `‖·‖ ≤ C`.
  have h_supp : Function.support (autocorr_complex f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
    autocorr_complex_support hf_supp
  -- Use `MemLp.of_bound` on the whole line with the everywhere bound, then it's L² since
  -- the function is supported on a finite-measure set.  Cleanest: bound the L² norm via
  -- `MemLp` of the indicator-times-constant majorant.
  -- We instead show: `autocorr_complex f` is a.e.-strongly-measurable and bounded by the
  -- `L²` function `C · 𝟙_{Ioo(-1/2,1/2)}`.
  have hmeas : AEStronglyMeasurable (autocorr_complex f) volume :=
    (autocorr_complex_integrable f hf_int).aestronglyMeasurable
  -- Majorant `g x = C · 𝟙_{Ioo(-1/2,1/2)}(x)` is in `L²`.
  set g : ℝ → ℝ := Set.indicator (Set.Ioo (-(1/2 : ℝ)) (1/2)) (fun _ => C) with hg_def
  have hg_memLp : MemLp g 2 volume := by
    rw [hg_def]
    have hμs : volume (Set.Ioo (-(1/2 : ℝ)) (1/2)) ≠ ⊤ := by
      rw [Real.volume_Ioo]; exact ENNReal.ofReal_ne_top
    exact memLp_indicator_const 2 measurableSet_Ioo C (Or.inr hμs)
  -- pointwise `‖autocorr_complex f x‖ ≤ g x`.
  have h_ptwise : ∀ x, ‖autocorr_complex f x‖ ≤ g x := by
    intro x
    by_cases hx : x ∈ Set.Ioo (-(1/2 : ℝ)) (1/2)
    · rw [hg_def, Set.indicator_of_mem hx]
      unfold autocorr_complex
      rw [Complex.norm_real, Real.norm_eq_abs]
      exact autocorr_abs_le_local f hf_L2 x
    · -- outside: autocorr_complex f x = 0 (support), g x = 0.
      have h0 : autocorr_complex f x = 0 := by
        by_contra hne
        exact hx (h_supp (by simpa [Function.mem_support] using hne))
      rw [h0, hg_def, Set.indicator_of_notMem hx, norm_zero]
  exact hg_memLp.mono' hmeas (Filter.Eventually.of_forall h_ptwise)

/-! ## The period-1 summand and its realness

For integer `r`, the period-1 bilinear Parseval summand
`𝓕(autocorr f)(r)·conj(𝓕K_ms(r))` equals the **real** number
`‖𝓕f(r)‖²·(𝓕K_ms(r)).re` (Wiener–Khinchin + realness of `𝓕K_ms`). -/

/-- The squared modulus of the period-1 lattice Fourier coefficient of `f`. -/
def Fmod_sq (f : ℝ → ℝ) (r : ℤ) : ℝ :=
  ‖Real.fourierIntegral (fun x => ((f x : ℂ))) (r : ℝ)‖ ^ 2

/-- `0 ≤ Fmod_sq f r`. -/
theorem Fmod_sq_nonneg (f : ℝ → ℝ) (r : ℤ) : 0 ≤ Fmod_sq f r := sq_nonneg _

/-- The period-1 lattice real Fourier coefficient of `K_ms`. -/
def Khat1 (r : ℤ) : ℝ := (Real.fourierIntegral K_ms_complex (r : ℝ)).re

/-- Per-summand identity: `𝓕(autocorr f)(r)·conj(𝓕K_ms(r)) = (Fmod_sq f r · Khat1 r : ℂ)`. -/
theorem summand_eq (f : ℝ → ℝ) (hf : Integrable f volume) (r : ℤ) :
    Real.fourierIntegral (autocorr_complex f) (r : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (r : ℝ))
      = ((Fmod_sq f r * Khat1 r : ℝ) : ℂ) := by
  -- `𝓕(autocorr f)(ξ) = 𝓕f(ξ)·conj(𝓕f(ξ))` (Wiener–Khinchin).
  have h_ac : Real.fourierIntegral (autocorr_complex f) (r : ℝ)
      = Real.fourierIntegral (fun x => ((f x : ℂ))) (r : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral (fun x => ((f x : ℂ))) (r : ℝ)) := by
    have := Sidon.Constructor.fourierIntegral_autocorr_real f hf (r : ℝ)
    simpa [autocorr_complex] using this
  -- `conj(𝓕K_ms(r)) = 𝓕K_ms(r) = ((𝓕K_ms(r)).re : ℂ)`.
  have h_K : starRingEnd ℂ (Real.fourierIntegral K_ms_complex (r : ℝ))
      = ((Khat1 r : ℝ) : ℂ) := by
    rw [conj_fourierIntegral_K_ms, fourierIntegral_K_ms_eq_re]; rfl
  rw [h_ac, h_K]
  set z := Real.fourierIntegral (fun x => ((f x : ℂ))) (r : ℝ) with hz
  have hzz : z * starRingEnd ℂ z = ((‖z‖ ^ 2 : ℝ) : ℂ) := by
    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
  rw [hzz, Fmod_sq]; push_cast; ring

/-- The complex integral `∫ autocorr_ℂ·conj(K_ms_ℂ)` is the real cast of
`∫ autocorr·K_ms` (`K_ms` real, so `conj` is trivial). -/
theorem integral_autocorr_Kms_complex_eq (f : ℝ → ℝ) :
    (∫ x, autocorr_complex f x * starRingEnd ℂ (K_ms_complex x) ∂volume)
      = ((∫ x, autocorr f x * K_ms x ∂volume : ℝ) : ℂ) := by
  rw [show ((∫ x, autocorr f x * K_ms x ∂volume : ℝ) : ℂ)
        = ∫ x, ((autocorr f x * K_ms x : ℝ) : ℂ) ∂volume from
      (integral_ofReal (f := fun x => autocorr f x * K_ms x)).symm]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  simp only [autocorr_complex, K_ms_complex, Complex.conj_ofReal]
  push_cast; ring

/-! ## Period-1 Plancherel summability inputs -/

/-- `Summable (r ↦ Fmod_sq f r)` (period-1 Plancherel for `f`; `f` fits one
period as `supp f ⊆ (-1/4,1/4) ⊆ (-1/2,1/2)`). -/
theorem summable_Fmod_sq (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun r : ℤ => Fmod_sq f r) := by
  -- Period-1 Plancherel: `f` fits `Ioc (-1/2) (1/2)`.
  set F : ℝ → ℂ := fun x => (f x : ℂ) with hF
  have hsupp_F : Function.support F ⊆ Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2) := by
    intro x hx
    have hx' : f x ≠ 0 := by
      simp only [hF, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
      exact hx
    have := hf_supp (by simpa [Function.mem_support] using hx')
    rw [Set.mem_Ioo] at this; rw [Set.mem_Ioc]
    constructor <;> linarith [this.1, this.2]
  have hL2_F : MemLp F 2 (volume.restrict (Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2))) := by
    have hFc : MemLp F 2 volume := by simpa [hF] using hf_L2.ofReal
    exact hFc.restrict _
  have hHS := Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
    (1 : ℝ) one_pos F hsupp_F hL2_F
  have hSummable_scaled :
      Summable (fun j : ℤ => ‖(1 / (1:ℝ) : ℂ) * Real.fourierIntegral F (j / (1:ℝ) : ℝ)‖ ^ 2) :=
    hHS.summable
  have h_eq : ∀ j : ℤ,
      ‖(1 / (1:ℝ) : ℂ) * Real.fourierIntegral F (j / (1:ℝ) : ℝ)‖ ^ 2 = Fmod_sq f j := by
    intro j
    rw [Fmod_sq]
    have hj1 : ((j : ℝ) / (1:ℝ)) = (j : ℝ) := by norm_num
    rw [hj1]
    simp [hF]
  rw [summable_congr h_eq] at hSummable_scaled
  exact hSummable_scaled

/-- `Summable (r ↦ Khat1 r ^ 2)` (period-1 Plancherel for `K_ms`). -/
theorem summable_Khat1_sq :
    Summable (fun r : ℤ => Khat1 r ^ 2) := by
  have hsumm := Sidon.BundleEq2Schwartz.K_ms_lattice_summable_period1
    Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict
  -- `‖𝓕K_ms(r)‖² = Khat1 r ^ 2` (𝓕K_ms real).
  have h_eq : ∀ r : ℤ, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ)) ((r : ℝ))‖ ^ 2
      = Khat1 r ^ 2 := by
    intro r
    rw [Khat1]
    rw [show (fun x => ((K_ms x : ℝ) : ℂ)) = K_ms_complex from rfl]
    conv_lhs => rw [fourierIntegral_K_ms_eq_re (r : ℝ)]
    rw [Complex.norm_real, Real.norm_eq_abs, sq_abs]
  rw [summable_congr h_eq] at hsumm
  exact hsumm

/-- The (real) period-1 summand family `r ↦ Fmod_sq f r · Khat1 r` is summable.
Absolute summability via AM–GM: `|Fmod_sq·Khat1| ≤ ½(Fmod_sq² + Khat1²)`, both
summable (period-1 Plancherel for `f` and `K_ms`). -/
theorem summable_real_summand (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun r : ℤ => Fmod_sq f r * Khat1 r) := by
  have h_F2 : Summable (fun r : ℤ => Fmod_sq f r ^ 2) := by
    have h_f := summable_Fmod_sq f hf_L2 hf_supp
    -- `Fmod_sq` summable nonneg ⟹ bounded by `B := ∑' Fmod_sq` (`le_tsum`), so
    -- `Fmod_sq² ≤ B·Fmod_sq` is summable.
    set B : ℝ := ∑' r : ℤ, Fmod_sq f r with hB_def
    refine Summable.of_nonneg_of_le (fun r => sq_nonneg _) (fun r => ?_) (h_f.mul_left B)
    have hBr : Fmod_sq f r ≤ B := by
      have h := h_f.sum_le_tsum {r} (fun i _ => Fmod_sq_nonneg f i)
      simpa using h
    nlinarith [Fmod_sq_nonneg f r, hBr]
  have h_K2 : Summable (fun r : ℤ => Khat1 r ^ 2) := summable_Khat1_sq
  have h_maj : Summable (fun r : ℤ => (1/2) * (Fmod_sq f r ^ 2 + Khat1 r ^ 2)) :=
    (h_F2.add h_K2).mul_left (1/2)
  -- Absolute summability: `‖Fmod_sq·Khat1‖ ≤ ½(Fmod_sq²+Khat1²)`.
  refine Summable.of_norm (Summable.of_nonneg_of_le (fun r => norm_nonneg _) (fun r => ?_) h_maj)
  -- `‖Fmod_sq·Khat1‖ = Fmod_sq·|Khat1| ≤ ½(Fmod_sq²+Khat1²)` (AM–GM).
  rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (Fmod_sq_nonneg f r)]
  nlinarith [sq_nonneg (Fmod_sq f r - |Khat1 r|), sq_abs (Khat1 r),
    Fmod_sq_nonneg f r, abs_nonneg (Khat1 r)]

/-- `Summable (r ↦ Fmod_sq f r · |Khat1 r|)` (same AM–GM majorant). -/
theorem summable_real_summand_abs (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun r : ℤ => Fmod_sq f r * |Khat1 r|) := by
  have h_F2 : Summable (fun r : ℤ => Fmod_sq f r ^ 2) := by
    have h_f := summable_Fmod_sq f hf_L2 hf_supp
    set B : ℝ := ∑' r : ℤ, Fmod_sq f r with hB_def
    refine Summable.of_nonneg_of_le (fun r => sq_nonneg _) (fun r => ?_) (h_f.mul_left B)
    have hBr : Fmod_sq f r ≤ B := by
      have := h_f.sum_le_tsum {r} (fun i _ => Fmod_sq_nonneg f i); simpa using this
    nlinarith [Fmod_sq_nonneg f r, hBr]
  have h_K2 : Summable (fun r : ℤ => Khat1 r ^ 2) := summable_Khat1_sq
  have h_maj : Summable (fun r : ℤ => (1/2) * (Fmod_sq f r ^ 2 + Khat1 r ^ 2)) :=
    (h_F2.add h_K2).mul_left (1/2)
  refine Summable.of_nonneg_of_le (fun r => mul_nonneg (Fmod_sq_nonneg f r) (abs_nonneg _))
    (fun r => ?_) h_maj
  nlinarith [sq_nonneg (Fmod_sq f r - |Khat1 r|), sq_abs (Khat1 r),
    Fmod_sq_nonneg f r, abs_nonneg (Khat1 r)]

/-! ## The period-1 Parseval split (constant `1`)

`bilinear_parseval_period_u` at `u = 1`, applied to `autocorr_complex f`
(supp `⊆ (-1/2,1/2)`) and `K_ms_complex` (supp `⊆ (-δ₁,δ₁) ⊆ (-1/2,1/2)`),
both fitting one period. -/

/-- The `r=0` term of the real period-1 summand is `1`:
`Fmod_sq f 0 · Khat1 0 = ‖𝓕f(0)‖²·(𝓕K_ms(0)).re = 1·1 = 1`. -/
theorem summand_zero (f : ℝ → ℝ) (hf_one : ∫ x, f x ∂volume = 1)
    (hK_one : ∫ x, K_ms x ∂volume = 1) :
    Fmod_sq f 0 * Khat1 0 = 1 := by
  have hF0 : Fmod_sq f 0 = 1 := by
    rw [Fmod_sq]
    have h0 : ((0 : ℤ) : ℝ) = 0 := by norm_num
    rw [h0, Sidon.FourierAux.fourierIntegral_zero (fun x => ((f x : ℝ) : ℂ))]
    rw [show (∫ x, ((f x : ℝ) : ℂ) ∂volume) = ((∫ x, f x ∂volume : ℝ) : ℂ) from
        integral_ofReal (𝕜 := ℂ) (f := f)]
    rw [hf_one]; simp
  have hK0 : Khat1 0 = 1 := by
    rw [Khat1]
    have h0 : ((0 : ℤ) : ℝ) = 0 := by norm_num
    rw [h0, show K_ms_complex = (fun x => ((K_ms x : ℝ) : ℂ)) from rfl,
      Sidon.BundleEq2Schwartz.fourierIntegral_K_ms_zero, hK_one]
    simp
  rw [hF0, hK0]; ring

/-- **The period-1 Parseval split of `∫ autocorr f · K_ms` (constant `1`).**

For admissible `f`,
  `∫ autocorr f · K_ms = 1 + ∑'_{r≠0} ‖𝓕f(r)‖² · (𝓕K_ms(r)).re`. -/
theorem autocorr_period1_split (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_one : ∫ x, f x ∂volume = 1) :
    ∫ x, autocorr f x * K_ms x ∂volume
      = 1 + ∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r * Khat1 r := by
  have hK_one : ∫ x, K_ms x ∂volume = 1 :=
    Sidon.Constructor.K_ms_integral_eq_one_local
  -- Period-1 bilinear Parseval (u = 1): autocorr_ℂ and K_ms_ℂ both fit one period.
  have hac_int : Integrable (autocorr_complex f) volume := autocorr_complex_integrable f hf_int
  have hac_L2 : MemLp (autocorr_complex f) 2 volume :=
    autocorr_complex_memLp2 f hf_int hf_L2 hf_supp
  have hac_supp : Function.support (autocorr_complex f) ⊆ Set.Ioo (-((1:ℝ)/2)) ((1:ℝ)/2) :=
    autocorr_complex_support hf_supp
  have hK_int : Integrable K_ms_complex volume :=
    (Sidon.Constructor.K_ms_integrable_local).ofReal
  have hK_L2 : MemLp K_ms_complex 2 volume := K_ms_complex_memLp2
  have hK_supp : Function.support K_ms_complex ⊆ Set.Ioo (-((1:ℝ)/2)) ((1:ℝ)/2) :=
    K_ms_complex_supp_half
  have h_par := Sidon.BilinearParseval.bilinear_parseval_period_u
    (1 : ℝ) one_pos (autocorr_complex f) K_ms_complex
    hac_int hK_int hac_supp hK_supp hac_L2 hK_L2
  -- LHS `1 * ∫ ... = ∫ ...`; RHS reindex `j/1 = j`.
  rw [show ((1:ℝ) : ℂ) = 1 from Complex.ofReal_one, one_mul] at h_par
  have h_idx : ∀ j : ℤ, (j : ℝ) / (1:ℝ) = (j : ℝ) := fun j => by norm_num
  rw [show (fun j : ℤ => Real.fourierIntegral (autocorr_complex f) (j / (1:ℝ) : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / (1:ℝ) : ℝ)))
        = (fun j : ℤ => Real.fourierIntegral (autocorr_complex f) (j : ℝ)
              * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j : ℝ)))
      from by funext j; rw [h_idx j]] at h_par
  -- Each summand is `(Fmod_sq·Khat1 : ℂ)`.
  set s : ℤ → ℝ := fun r => Fmod_sq f r * Khat1 r with hs_def
  have hs_summable : Summable s := summable_real_summand f hf_L2 hf_supp
  have h_summand : ∀ r : ℤ,
      Real.fourierIntegral (autocorr_complex f) (r : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (r : ℝ))
        = ((s r : ℝ) : ℂ) := by
    intro r; rw [hs_def]; exact summand_eq f hf_int r
  rw [tsum_congr h_summand] at h_par
  rw [show (∑' r : ℤ, ((s r : ℝ) : ℂ)) = ((∑' r : ℤ, s r : ℝ) : ℂ) from
      (Complex.ofRealCLM.map_tsum hs_summable).symm] at h_par
  rw [integral_autocorr_Kms_complex_eq f] at h_par
  -- Take real parts: `∫ ac·K = ∑' s`.
  have hre := congrArg Complex.re h_par
  rw [Complex.ofReal_re, Complex.ofReal_re] at hre
  rw [hre]
  -- Split off `r = 0`.
  have h_split : ∑' r : ℤ, s r = s 0 + ∑' r : ℤ, (if r = 0 then 0 else s r) :=
    hs_summable.tsum_eq_add_tsum_ite 0
  rw [h_split]
  have hs0 : s 0 = 1 := by rw [hs_def]; exact summand_zero f hf_one hK_one
  rw [hs0, hs_def]

/-! ## The K-bound `∑'_{r≠0} Khat1² ≤ K2_analytic − 1` -/

/-- `∑'_{r≠0} Khat1 r ^ 2 ≤ K2_analytic − 1` (period-1 Plancherel for `K_ms`
minus the `r=0` term `= 1`). -/
theorem K_bound_tsum (hK_one : ∫ x, K_ms x ∂volume = 1) :
    (∑' r : ℤ, if r = 0 then 0 else Khat1 r ^ 2) ≤ K2_analytic - 1 := by
  -- `∑'_r Khat1 r² = K2_analytic` (period-1 Plancherel, 𝓕K_ms real).
  have h_plan : ∑' r : ℤ, Khat1 r ^ 2 = K2_analytic := by
    have h := Sidon.BundleEq2Schwartz.plancherel_K_ms_K2_analytic_period1
      Sidon.Constructor.MOLemma21.K_ms_complex_memLp_restrict
    rw [← h]
    refine tsum_congr (fun r => ?_)
    rw [Khat1, show (fun x => ((K_ms x : ℝ) : ℂ)) = K_ms_complex from rfl]
    conv_rhs => rw [fourierIntegral_K_ms_eq_re (r : ℝ)]
    rw [Complex.norm_real, Real.norm_eq_abs, sq_abs]
  have h_summable : Summable (fun r : ℤ => Khat1 r ^ 2) := summable_Khat1_sq
  have hK0 : Khat1 0 = 1 := by
    rw [Khat1]
    have h0 : ((0 : ℤ) : ℝ) = 0 := by norm_num
    rw [h0, show K_ms_complex = (fun x => ((K_ms x : ℝ) : ℂ)) from rfl,
      Sidon.BundleEq2Schwartz.fourierIntegral_K_ms_zero, hK_one]; simp
  -- split off `r = 0`.
  have h_split : ∑' r : ℤ, Khat1 r ^ 2
      = Khat1 0 ^ 2 + ∑' r : ℤ, (if r = 0 then 0 else Khat1 r ^ 2) :=
    h_summable.tsum_eq_add_tsum_ite 0
  rw [h_plan, hK0] at h_split
  -- `K2_analytic = 1 + tail` ⟹ `tail = K2 - 1`.
  linarith [h_split]

/-! ## The F-bound `∑'_{r≠0} Fmod_sq² ≤ R(f) − 1`

`Fmod_sq f r ^ 2 = ‖𝓕f(r)‖⁴ = ‖𝓕(f*f)(r)‖²`; period-1 Plancherel for `f*f`
gives `∑'_r = ∫_ℝ (f*f)²`; the `r=0` term is `1`; Hölder bounds the integral
by `‖f*f‖_∞ = R(f)`. -/

/-- `f*f` (real autoconvolution). -/
private def convff (f : ℝ → ℝ) : ℝ → ℝ :=
  MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume

/-- `convff f x ≤ R(f) = ‖f*f‖_∞` a.e. (from `ae_le_essSup`), using `∫f = 1`. -/
theorem convff_ae_le_ratio (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_nonneg : ∀ x, 0 ≤ f x) (hf_one : ∫ x, f x ∂volume = 1) :
    ∀ᵐ x ∂volume, convff f x ≤ autoconvolution_ratio f := by
  -- `R(f) = (eLpNorm convff ⊤).toReal` (since `∫ f = 1`).
  have hR_eq : autoconvolution_ratio f = (eLpNorm (convff f) ⊤ volume).toReal := by
    unfold autoconvolution_ratio convff
    rw [hf_one]; norm_num
  rw [hR_eq, eLpNorm_exponent_top]
  -- `convff f ≥ 0`, so `convff f x = ‖convff f x‖`.
  have h_nn : ∀ x, 0 ≤ convff f x := convolution_nonneg hf_nonneg hf_nonneg
  -- essSup of `‖convff‖ₑ` is finite.
  have h_fin : eLpNormEssSup (convff f) volume ≠ ⊤ := by
    have := Sidon.Constructor.conv_eLpNorm_top_ne_top hf_L2
    rwa [eLpNorm_exponent_top] at this
  -- a.e. `‖convff f x‖ₑ ≤ essSup`.
  have h_ae := ENNReal.ae_le_essSup (μ := volume) (fun x => ‖convff f x‖ₑ)
  -- `eLpNormEssSup (convff f) volume = essSup (‖convff f ·‖ₑ) volume` definitionally.
  show ∀ᵐ x ∂volume, convff f x ≤ (essSup (fun x => ‖convff f x‖ₑ) volume).toReal
  have h_fin' : essSup (fun x => ‖convff f x‖ₑ) volume ≠ ⊤ := h_fin
  filter_upwards [h_ae] with x hx
  -- `‖convff f x‖ₑ = ofReal (convff f x)`; convert to reals.
  rw [Real.enorm_eq_ofReal (h_nn x)] at hx
  have hle := (ENNReal.toReal_le_toReal (by finiteness) h_fin').mpr hx
  rwa [ENNReal.toReal_ofReal (h_nn x)] at hle

/-- `convff f` is integrable. -/
theorem convff_integrable (f : ℝ → ℝ) (hf_int : Integrable f volume) :
    Integrable (convff f) volume :=
  hf_int.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int

/-- `∫ convff f = (∫ f)² = 1` (mass identity, `∫f = 1`). -/
theorem convff_integral_eq_one (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_one : ∫ x, f x ∂volume = 1) :
    ∫ x, convff f x ∂volume = 1 := by
  rw [convff, integral_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int hf_int,
    ContinuousLinearMap.mul_apply', hf_one]; ring

/-- `convff f` support `⊆ Ioo (-1/2) (1/2)`. -/
theorem convff_support (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (convff f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  have h_sub : Function.support (convff f) ⊆
      Function.support f + Function.support f := by
    rw [convff]; exact support_convolution_subset _
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (h_sub hx)
  have ha' := hf_supp ha; have hb' := hf_supp hb
  simp only [Set.mem_Ioo] at ha' hb' ⊢
  subst hab
  exact ⟨by linarith [ha'.1, hb'.1], by linarith [ha'.2, hb'.2]⟩

/-- The Fourier–Hölder bound `∫ (f*f)² ≤ R(f)` (using `∫(f*f) = 1`). -/
theorem convff_sq_integral_le_ratio (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_nonneg : ∀ x, 0 ≤ f x) (hf_one : ∫ x, f x ∂volume = 1) :
    ∫ x, convff f x ^ 2 ∂volume ≤ autoconvolution_ratio f := by
  set M : ℝ := autoconvolution_ratio f with hM_def
  have h_nn : ∀ x, 0 ≤ convff f x := convolution_nonneg hf_nonneg hf_nonneg
  have h_ae_le : ∀ᵐ x ∂volume, convff f x ≤ M :=
    convff_ae_le_ratio f hf_int hf_L2 hf_nonneg hf_one
  have h_conv_int : Integrable (convff f) volume := convff_integrable f hf_int
  -- a.e. `(f*f)² ≤ M·(f*f)`.
  have h_sq_le : ∀ᵐ x ∂volume, convff f x ^ 2 ≤ M * convff f x := by
    filter_upwards [h_ae_le] with x hx
    nlinarith [h_nn x, hx]
  -- `M·(f*f)` integrable.
  have h_Mconv_int : Integrable (fun x => M * convff f x) volume := h_conv_int.const_mul M
  -- `(f*f)²` integrable: nonneg, a.e. ≤ integrable `M·(f*f)`.
  have h_sq_int : Integrable (fun x => convff f x ^ 2) volume := by
    refine Integrable.mono' h_Mconv_int ?_ ?_
    · exact (h_conv_int.aestronglyMeasurable.pow 2)
    · filter_upwards [h_sq_le] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
      exact hx
  -- integrate the a.e. bound.
  have h_int_le : ∫ x, convff f x ^ 2 ∂volume ≤ ∫ x, M * convff f x ∂volume :=
    integral_mono_ae h_sq_int h_Mconv_int h_sq_le
  rw [integral_const_mul, convff_integral_eq_one f hf_int hf_one, mul_one] at h_int_le
  exact h_int_le

/-- Complex lift of `f*f`. -/
def convff_complex (f : ℝ → ℝ) : ℝ → ℂ := fun x => ((convff f x : ℝ) : ℂ)

/-- `convff_complex f ∈ L²(restrict period-1)`: bounded by `∫f²` on the finite
interval `Ioc (-1/2) (1/2)`. -/
theorem convff_complex_memLp2_restrict (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume) :
    MemLp (convff_complex f) 2 (volume.restrict (Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2))) := by
  haveI : IsFiniteMeasure (volume.restrict (Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2))) := by
    refine ⟨?_⟩
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top
  set C : ℝ := ∫ x, f x ^ 2 ∂volume with hC_def
  have h_conv_int : Integrable (convff f) volume := convff_integrable f hf_int
  have hmeas : AEStronglyMeasurable (convff_complex f)
      (volume.restrict (Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2))) :=
    (h_conv_int.ofReal.aestronglyMeasurable).restrict
  refine MemLp.of_bound hmeas C ?_
  refine Filter.Eventually.of_forall (fun x => ?_)
  unfold convff_complex convff
  rw [Complex.norm_real, Real.norm_eq_abs]
  -- `|f*f x| ≤ ∫f²` (Cauchy–Schwarz; reuse the FieldsEasy enorm bound is awkward; use convff_abs).
  exact Sidon.Constructor.MOLemma21.convff_abs_le f hf_L2 x

/-- `convff_complex f` support `⊆ Ioc (-1/2) (1/2)`. -/
theorem convff_complex_support (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (convff_complex f) ⊆ Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2) := by
  intro x hx
  have hx' : convff f x ≠ 0 := by
    simp only [convff_complex, Function.mem_support, ne_eq, Complex.ofReal_eq_zero] at hx
    exact hx
  have := convff_support f hf_supp (by simpa [Function.mem_support] using hx')
  rw [Set.mem_Ioo] at this; rw [Set.mem_Ioc]
  exact ⟨this.1, le_of_lt this.2⟩

/-- `‖𝓕(convff_complex f)(r)‖² = Fmod_sq f r ^ 2` (`𝓕(f*f)=(𝓕f)²`). -/
theorem normSq_fourier_convff_eq (f : ℝ → ℝ) (hf_int : Integrable f volume) (r : ℤ) :
    ‖Real.fourierIntegral (convff_complex f) (r : ℝ)‖ ^ 2 = Fmod_sq f r ^ 2 := by
  have h_ft : Real.fourierIntegral (convff_complex f) (r : ℝ)
      = (Real.fourierIntegral (fun x => ((f x : ℂ))) (r : ℝ)) ^ 2 := by
    have := Sidon.Constructor.fourierIntegral_autoconvolution_real f hf_int (r : ℝ)
    simpa [convff_complex, convff] using this
  rw [h_ft, Fmod_sq, norm_pow]

/-- The `r=0` term: `‖𝓕(convff_complex f)(0)‖² = (∫ f*f)² = 1`. -/
theorem normSq_fourier_convff_zero (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_one : ∫ x, f x ∂volume = 1) :
    ‖Real.fourierIntegral (convff_complex f) ((0 : ℤ) : ℝ)‖ ^ 2 = 1 := by
  have h0 : ((0 : ℤ) : ℝ) = 0 := by norm_num
  rw [h0, Sidon.FourierAux.fourierIntegral_zero (convff_complex f)]
  rw [show (∫ x, convff_complex f x ∂volume) = ((∫ x, convff f x ∂volume : ℝ) : ℂ) from
      integral_ofReal (𝕜 := ℂ) (f := convff f)]
  rw [convff_integral_eq_one f hf_int hf_one]; simp

/-- **The F-bound** `∑'_{r≠0} Fmod_sq f r ^ 2 ≤ R(f) − 1`. -/
theorem F_bound_tsum (f : ℝ → ℝ)
    (hf_int : Integrable f volume) (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x) (hf_one : ∫ x, f x ∂volume = 1) :
    (∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r ^ 2) ≤ autoconvolution_ratio f - 1 := by
  -- period-1 Plancherel for `convff_complex f` (fits one period).
  have hsupp : Function.support (convff_complex f) ⊆ Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2) :=
    convff_complex_support f hf_supp
  have hL2 : MemLp (convff_complex f) 2 (volume.restrict (Set.Ioc (-((1:ℝ)/2)) ((1:ℝ)/2))) :=
    convff_complex_memLp2_restrict f hf_int hf_L2
  have h_plan := Sidon.TorusParseval.plancherel_at_lattice_period_u
    (1 : ℝ) one_pos (convff_complex f) hsupp hL2
  -- LHS reindex `r/1 = r`; identify summand with `Fmod_sq²`.
  have h_idx : ∀ r : ℤ, (r : ℝ) / (1:ℝ) = (r : ℝ) := fun r => by norm_num
  rw [show (fun r : ℤ => ‖Real.fourierIntegral (convff_complex f) (r / (1:ℝ) : ℝ)‖ ^ 2)
        = (fun r : ℤ => Fmod_sq f r ^ 2)
      from by funext r; rw [h_idx r]; exact normSq_fourier_convff_eq f hf_int r] at h_plan
  -- RHS: `1 · ∫_{-1/2}^{1/2}|convff|² = ∫_ℝ (convff)²`.
  have h_rhs : (1 : ℝ) * ∫ x in (-((1:ℝ)/2))..((1:ℝ)/2), ‖convff_complex f x‖ ^ 2
      = ∫ x, convff f x ^ 2 ∂volume := by
    rw [one_mul]
    -- `‖convff_complex f x‖² = (convff f x)²`.
    have h_pt : ∀ x, ‖convff_complex f x‖ ^ 2 = convff f x ^ 2 := by
      intro x; unfold convff_complex; rw [Complex.norm_real, Real.norm_eq_abs, sq_abs]
    rw [show (fun x => ‖convff_complex f x‖ ^ 2) = (fun x => convff f x ^ 2) from
        funext h_pt]
    -- interval integral over (-1/2,1/2) = full-line integral (supp fits).
    rw [intervalIntegral.integral_of_le (by norm_num : (-((1:ℝ)/2)) ≤ (1:ℝ)/2)]
    refine MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero (fun x hx => ?_)
    -- outside `Ioc (-1/2,1/2)`: convff f x = 0.
    have hx0 : convff f x = 0 := by
      by_contra hne
      have hmem := convff_support f hf_supp (by simpa [Function.mem_support] using hne)
      rw [Set.mem_Ioo] at hmem
      exact hx ⟨hmem.1, le_of_lt hmem.2⟩
    rw [hx0]; ring
  rw [h_rhs] at h_plan
  -- `∑'_r Fmod_sq² = ∫(convff)² ≤ M`; split off `r = 0`.
  have h_summable : Summable (fun r : ℤ => Fmod_sq f r ^ 2) := by
    have hscaled := (Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
      (1 : ℝ) one_pos (convff_complex f) hsupp hL2).summable
    refine hscaled.congr (fun r => ?_)
    have h_idx' : (r : ℝ) / (1:ℝ) = (r : ℝ) := by norm_num
    rw [show ‖(1 / (1:ℝ) : ℂ) * Real.fourierIntegral (convff_complex f) (r / (1:ℝ) : ℝ)‖ ^ 2
          = ‖Real.fourierIntegral (convff_complex f) (r : ℝ)‖ ^ 2 from by
        rw [h_idx']; norm_num]
    exact normSq_fourier_convff_eq f hf_int r
  have hF0 : Fmod_sq f 0 ^ 2 = 1 := by
    have hz := normSq_fourier_convff_zero f hf_int hf_one
    rw [normSq_fourier_convff_eq f hf_int 0] at hz
    exact hz
  have h_split : ∑' r : ℤ, Fmod_sq f r ^ 2
      = Fmod_sq f 0 ^ 2 + ∑' r : ℤ, (if r = 0 then 0 else Fmod_sq f r ^ 2) :=
    h_summable.tsum_eq_add_tsum_ite 0
  rw [h_plan, hF0] at h_split
  have hM := convff_sq_integral_le_ratio f hf_int hf_L2 hf_nonneg hf_one
  linarith [h_split, hM]

/-! ## The tail Cauchy–Schwarz and the final `field_hEq2`

The tail `T := ∑'_{r≠0} Fmod_sq f r · Khat1 r` is bounded via
`T ≤ ∑'_{r≠0} Fmod_sq · |Khat1| ≤ √(∑Fmod_sq²)·√(∑Khat1²)`
(NNReal Hölder `p=q=2`, `inner_le_Lp_mul_Lq_tsum`). -/

/-- Squared-tail summabilities (real, `if r=0 then 0 else …` form). -/
theorem summable_Fmod_sq_sq_tail (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun r : ℤ => if r = 0 then (0:ℝ) else Fmod_sq f r ^ 2) := by
  have h := summable_Fmod_sq f hf_L2 hf_supp
  have hsq : Summable (fun r : ℤ => Fmod_sq f r ^ 2) := by
    set B : ℝ := ∑' r : ℤ, Fmod_sq f r with hB
    refine Summable.of_nonneg_of_le (fun r => sq_nonneg _) (fun r => ?_) (h.mul_left B)
    have hBr : Fmod_sq f r ≤ B := by
      have := h.sum_le_tsum {r} (fun i _ => Fmod_sq_nonneg f i); simpa using this
    nlinarith [Fmod_sq_nonneg f r, hBr]
  have hfin : {r : ℤ | Fmod_sq f r ^ 2
      ≠ if r = 0 then (0:ℝ) else Fmod_sq f r ^ 2}.Finite := by
    refine Set.Finite.subset (Set.finite_singleton (0:ℤ)) (fun r hr => ?_)
    simp only [Set.mem_setOf_eq] at hr
    by_contra hne; exact hr (by rw [if_neg (by simpa using hne)])
  exact (summable_congr_cofinite hfin).mp hsq

theorem summable_Khat1_sq_tail :
    Summable (fun r : ℤ => if r = 0 then (0:ℝ) else Khat1 r ^ 2) := by
  have hfin : {r : ℤ | Khat1 r ^ 2
      ≠ if r = 0 then (0:ℝ) else Khat1 r ^ 2}.Finite := by
    refine Set.Finite.subset (Set.finite_singleton (0:ℤ)) (fun r hr => ?_)
    simp only [Set.mem_setOf_eq] at hr
    by_contra hne; exact hr (by rw [if_neg (by simpa using hne)])
  exact (summable_congr_cofinite hfin).mp summable_Khat1_sq

/-- The tail Cauchy–Schwarz bound:
`∑'_{r≠0} Fmod_sq f r · Khat1 r ≤ √(∑'_{r≠0} Fmod_sq²) · √(∑'_{r≠0} Khat1²)`.
Real route: for every finite `s`, the partial sum is `≤` the product of
square-root tail norms (finite CS `Real.sum_mul_le_sqrt_mul_sqrt` + partial ≤
tsum), then `Real.tsum_le_of_sum_le`. -/
theorem tail_cauchy_schwarz (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    (∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r * Khat1 r)
      ≤ Real.sqrt (∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r ^ 2)
          * Real.sqrt (∑' r : ℤ, if r = 0 then 0 else Khat1 r ^ 2) := by
  set SF : ℝ := ∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r ^ 2 with hSF
  set SK : ℝ := ∑' r : ℤ, if r = 0 then 0 else Khat1 r ^ 2 with hSK
  have hSF_summ : Summable (fun r : ℤ => if r = 0 then (0:ℝ) else Fmod_sq f r ^ 2) :=
    summable_Fmod_sq_sq_tail f hf_L2 hf_supp
  have hSK_summ : Summable (fun r : ℤ => if r = 0 then (0:ℝ) else Khat1 r ^ 2) :=
    summable_Khat1_sq_tail
  -- nonneg tail-product family `g`.
  set g : ℤ → ℝ := fun r => if r = 0 then 0 else Fmod_sq f r * |Khat1 r| with hg
  have hg_nn : ∀ r, 0 ≤ g r := by
    intro r; simp only [hg]; split_ifs
    · exact le_refl 0
    · exact mul_nonneg (Fmod_sq_nonneg f r) (abs_nonneg _)
  -- For every finite `s`: `∑_s g ≤ √SF·√SK`.
  have h_fin_le : ∀ s : Finset ℤ, (∑ r ∈ s, g r) ≤ Real.sqrt SF * Real.sqrt SK := by
    intro s
    -- finite CS on `s` with families `Fmod_sq·1_{≠0}` and `|Khat1|·1_{≠0}`.
    set u : ℤ → ℝ := fun r => if r = 0 then 0 else Fmod_sq f r with hu
    set v : ℤ → ℝ := fun r => if r = 0 then 0 else |Khat1 r| with hv
    have h_g_eq : ∀ r ∈ s, g r = u r * v r := by
      intro r _; simp only [hg, hu, hv]; split_ifs with h
      · ring
      · rfl
    have hCS := Real.sum_mul_le_sqrt_mul_sqrt s u v
    -- `∑_s g = ∑_s u·v`.
    rw [Finset.sum_congr rfl h_g_eq]
    -- `u r ^ 2` / `v r ^ 2` agree with the SF / SK tail summands.
    have hu_sq_eq : (∑ r ∈ s, u r ^ 2)
        = ∑ r ∈ s, (if r = 0 then (0:ℝ) else Fmod_sq f r ^ 2) := by
      refine Finset.sum_congr rfl (fun r _ => ?_); simp only [hu]; split_ifs <;> ring
    have hv_sq_eq : (∑ r ∈ s, v r ^ 2)
        = ∑ r ∈ s, (if r = 0 then (0:ℝ) else Khat1 r ^ 2) := by
      refine Finset.sum_congr rfl (fun r _ => ?_); simp only [hv]; split_ifs with h
      · ring
      · rw [sq_abs]
    have hu2 : (∑ r ∈ s, u r ^ 2) ≤ SF := by
      rw [hu_sq_eq, hSF]; exact hSF_summ.sum_le_tsum s (fun r _ => by split_ifs <;> positivity)
    have hv2 : (∑ r ∈ s, v r ^ 2) ≤ SK := by
      rw [hv_sq_eq, hSK]; exact hSK_summ.sum_le_tsum s (fun r _ => by split_ifs <;> positivity)
    have h1 : Real.sqrt (∑ r ∈ s, u r ^ 2) ≤ Real.sqrt SF := Real.sqrt_le_sqrt hu2
    have h2 : Real.sqrt (∑ r ∈ s, v r ^ 2) ≤ Real.sqrt SK := Real.sqrt_le_sqrt hv2
    calc ∑ r ∈ s, u r * v r
        ≤ Real.sqrt (∑ r ∈ s, u r ^ 2) * Real.sqrt (∑ r ∈ s, v r ^ 2) := hCS
      _ ≤ Real.sqrt SF * Real.sqrt SK :=
          mul_le_mul h1 h2 (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  -- `∑' g ≤ √SF·√SK`.
  have h_tsum_g : (∑' r : ℤ, g r) ≤ Real.sqrt SF * Real.sqrt SK :=
    Real.tsum_le_of_sum_le hg_nn h_fin_le
  -- `T = ∑'_{r≠0} Fmod_sq·Khat1 ≤ ∑'_{r≠0} Fmod_sq·|Khat1| = ∑' g`.
  have h_T_le_g : (∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r * Khat1 r) ≤ ∑' r : ℤ, g r := by
    -- summable tail-product (real summand family).
    have h_summ : Summable (fun r : ℤ => if r = 0 then (0:ℝ) else Fmod_sq f r * Khat1 r) := by
      have hbase := summable_real_summand f hf_L2 hf_supp
      have hfin : {r : ℤ | Fmod_sq f r * Khat1 r
          ≠ if r = 0 then (0:ℝ) else Fmod_sq f r * Khat1 r}.Finite := by
        refine Set.Finite.subset (Set.finite_singleton (0:ℤ)) (fun r hr => ?_)
        simp only [Set.mem_setOf_eq] at hr
        by_contra hne; exact hr (by rw [if_neg (by simpa using hne)])
      exact (summable_congr_cofinite hfin).mp hbase
    have hg_summ : Summable g := by
      have hbase := summable_real_summand_abs f hf_L2 hf_supp
      have hfin : {r : ℤ | Fmod_sq f r * |Khat1 r| ≠ g r}.Finite := by
        refine Set.Finite.subset (Set.finite_singleton (0:ℤ)) (fun r hr => ?_)
        simp only [hg, Set.mem_setOf_eq] at hr
        by_contra hne; exact hr (by rw [if_neg (by simpa using hne)])
      exact (summable_congr_cofinite hfin).mp hbase
    refine Summable.tsum_le_tsum (fun r => ?_) h_summ hg_summ
    simp only [hg]; split_ifs with h
    · exact le_refl 0
    · exact mul_le_mul_of_nonneg_left (le_abs_self _) (Fmod_sq_nonneg f r)
  linarith [h_T_le_g, h_tsum_g]

/-! ## The unconditional `field_hEq2`

Assembling: period-1 split (`LHS2 = 1 + T`), tail CS
(`T ≤ √(SF)·√(SK)`), F-bound (`SF ≤ R(f)−1`), K-bound (`SK ≤ K2−1`), and
sqrt monotonicity. -/

/-- **`field_hEq2`, fully discharged from admissibility.**

For admissible `f`,
`LHS2 f ≤ 1 + √(autoconvolution_ratio f − 1) · √(K2_analytic − 1)`,
with NO residual `h_parseval_split` / `h_F_bound` / `h_K_bound`. -/
theorem field_hEq2_unconditional (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    LHS2 f ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                  * Real.sqrt (K2_analytic - 1) := by
  have hK_one : ∫ x, K_ms x ∂volume = 1 :=
    Sidon.Constructor.K_ms_integral_eq_one_local
  -- `LHS2 = 1 + T`, `T = ∑'_{r≠0} Fmod_sq·Khat1`.
  have h_split := autocorr_period1_split f hf_int hf_L2 hf_supp hf_one
  show ∫ x, autocorr f x * K_ms x ∂volume ≤ _
  rw [h_split]
  set T : ℝ := ∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r * Khat1 r with hT
  set SF : ℝ := ∑' r : ℤ, if r = 0 then 0 else Fmod_sq f r ^ 2 with hSF
  set SK : ℝ := ∑' r : ℤ, if r = 0 then 0 else Khat1 r ^ 2 with hSK
  -- tail CS: `T ≤ √SF·√SK`.
  have h_cs : T ≤ Real.sqrt SF * Real.sqrt SK := tail_cauchy_schwarz f hf_L2 hf_supp
  -- F-bound, K-bound.
  have h_F : SF ≤ autoconvolution_ratio f - 1 :=
    F_bound_tsum f hf_int hf_L2 hf_supp hf_nonneg hf_one
  have h_K : SK ≤ K2_analytic - 1 := K_bound_tsum hK_one
  -- sqrt monotonicity (`SF, SK ≥ 0`).
  have hSF_nn : 0 ≤ SF := by
    rw [hSF]; exact tsum_nonneg (fun r => by split_ifs <;> positivity)
  have hSK_nn : 0 ≤ SK := by
    rw [hSK]; exact tsum_nonneg (fun r => by split_ifs <;> positivity)
  have h_sqrtF : Real.sqrt SF ≤ Real.sqrt (autoconvolution_ratio f - 1) :=
    Real.sqrt_le_sqrt h_F
  have h_sqrtK : Real.sqrt SK ≤ Real.sqrt (K2_analytic - 1) :=
    Real.sqrt_le_sqrt h_K
  have h_prod : Real.sqrt SF * Real.sqrt SK
      ≤ Real.sqrt (autoconvolution_ratio f - 1) * Real.sqrt (K2_analytic - 1) :=
    mul_le_mul h_sqrtF h_sqrtK (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  linarith [h_cs, h_prod]

end

end Sidon.Constructor.Eq2Period1
