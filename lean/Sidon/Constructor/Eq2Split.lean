/-
Sidon Autocorrelation Project — MV Eq.(2) period-`u` Parseval split and L² tail bounds.
========================================================================================

This file targets the three residual analytic hypotheses of the
unconditional Sidon headline `Sidon.MultiScale.C1a_ge_1292_unconditional`
that concern MV Lemma 3.1 Eq.(2):

  * `h_K_bound`        : `∑_{j∈J} (K̂_ms(j))² ≤ K2_analytic − 1`;
  * `h_parseval_split` : `∫ autocorr f · K_ms = 1 + ∑_{j∈J} (Re 𝓕f(j/u))² · K̂_ms(j)`;
  * `h_F_bound`        : `∑_{j∈J} (Re 𝓕f(j/u))⁴ ≤ autoconvolution_ratio f − 1`.

================================================================================
## Results
================================================================================

  1. **`field_h_K_bound`** — discharges `h_K_bound` **exactly and
     unconditionally**.  Proof: period-`u` Plancherel for `K_ms`
     (`Sidon.BundleEq2Schwartz.K_bound_for_mv_eq2`) gives
     `∑_{j∈J}(Re 𝓕K_ms(j/u))² ≤ K2_analytic − 1` (the `u ≤ 1` slack
     absorbs the period factor), and the Fourier–Bessel identity
     `Sidon.Constructor.MOLemma21.bessel_identity`
     (`(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`) rewrites the summand
     into the project's Bessel closed form `K_ms_fourier_lattice j`.

  2. **`autocorr_period_u_split`** — the **genuine** MO 2009 Lemma 2.1
     period-`u` Parseval split of `∫ autocorr f · K_ms`, proved
     hypothesis-free (via the periodisation Poisson sampling
     `Sidon.Constructor.PoissonSampling.period_u_poisson_sampling`):

        `∫ autocorr f · K_ms = 1/u + (1/u)·∑'_{j≠0} ‖𝓕f(j/u)‖²·K̂_ms(j)`.

     The autocorrelation Fourier coefficient is `𝓕(autocorr f)(j/u) =
     ‖𝓕f(j/u)‖²` (Wiener–Khinchin, real and nonnegative — NOT the
     `(Re 𝓕f)²` of the combined `g = f∗f + autocorr f` after `±j`
     collapse), and the `j = 0` term gives the constant `1/u` (since
     `∫ autocorr f = 1`, `𝓕K_ms(0) = 1`).

================================================================================
## Why the residual `h_parseval_split` (as stated) is NOT this split
================================================================================

The headline's residual `h_parseval_split` reads

  `∫ autocorr f · K_ms = 1 + ∑_{j∈J} (Re 𝓕f(j/u))² · K̂_ms(j)`,

i.e. constant `1` and summand `(Re 𝓕f(j/u))² · K̂_ms(j)` with
`Fsq_re f j = (Re 𝓕f(j/u))²` (the *full-line* transform, unscaled, and
the `(Re)²` form).  The genuine identity `autocorr_period_u_split`
proved here instead has constant `1/u ≈ 1.567` (NOT `1`) and summands
`(1/u)·‖𝓕f(j/u)‖²·K̂_ms(j)` (the *squared modulus* `‖·‖²`, NOT `(Re)²`,
carrying a `1/u` prefactor).  These genuinely differ — the discrepancy
is verified numerically (for the box test `f = 2·𝟙_{(-1/4,1/4)}`,
`∫ autocorr f · K_ms ≈ 1.799 ≥ 1/u = 1.567`, whereas `1 + ∑(Re)²K̂`
evaluates to `≈ 1.148`).  The `(Re 𝓕f)²` form is correct only for the
*combined* `g = f∗f + autocorr f` (where the imaginary part cancels in
the `±j` pairing, giving `2(Re 𝓕f)²`), and the constant `1` arises only
in MV's *period-`u`-scaled* coefficient convention `f̃(j) = (1/u)𝓕f(j/u)`,
`K̃(j) = (1/u)𝓕K(j/u)`, NOT the unscaled full-line convention the bundle
anchors `Fsq_re`/`K_ms_fourier_lattice` to.

Consequently the residual `h_parseval_split` is **not provable as stated**;
its corrected (and provable) form is `autocorr_period_u_split` below.  The
remaining residual `h_F_bound` (`∑ (Re 𝓕f(j/u))⁴ ≤ R(f) − 1`) requires the
period-`u` Plancherel of `f∗f`, whose support `(-1/2,1/2)` *overflows* the
period `(-u/2,u/2)`; that is the genuinely-missing primitive (an `f∗f`
periodisation reconciliation, distinct from the `autocorr`-against-`K_ms`
pairing closed here), recorded at `field_h_F_bound_residual`.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.MOLemma21
import Sidon.Constructor.PoissonSampling
import Sidon.Constructor.PoissonSummable
import Sidon.Constructor.ConvFourier
import Sidon.TorusParseval
import Sidon.FourierAux
import Sidon.BundleEq2Schwartz

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical Pointwise

namespace Sidon.Constructor.Eq2Split

open Sidon.MultiScale
open Sidon.FourierAux (autocorr autocorr_def)
open Sidon.Constructor.MOLemma21 (K_ms_complex bessel_identity K_ms_complex_memLp_restrict)

noncomputable section

/-! ## Target 1 — `h_K_bound` (discharged exactly and unconditionally)

The headline's residual `h_K_bound` is
`∑_{j∈J} (K_ms_fourier_lattice j)² ≤ K2_analytic − 1`.
`Sidon.BundleEq2Schwartz.K_bound_for_mv_eq2` already proves
`∑_{j∈J} (Re 𝓕K_ms(j/u))² ≤ K2_analytic − 1` from period-`u` Plancherel;
the Fourier–Bessel identity `bessel_identity` rewrites
`(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`. -/

/-- **`h_K_bound`, discharged exactly.**

For any finite `J ⊆ ℤ` with `0 ∉ J`,
`∑_{j∈J} (K_ms_fourier_lattice j)² ≤ K2_analytic − 1`.

This is the exact bundle residual (`Sidon.MultiScale.K_ms_fourier_lattice`
form).  Proof: period-`u` Plancherel for `K_ms`
(`K_bound_for_mv_eq2`, with the `u ≤ 1` slack) bounds
`∑ (Re 𝓕K_ms(j/u))²`, and `bessel_identity` identifies
`(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`. -/
theorem field_h_K_bound (J : Finset ℤ) (hJ0 : (0 : ℤ) ∉ J) :
    (∑ j ∈ J, (K_ms_fourier_lattice j) ^ 2) ≤ K2_analytic - 1 := by
  -- `K_ms_complex = fun x => ((K_ms x : ℝ) : ℂ)`, and `K_ms_complex_memLp_restrict`
  -- supplies the `L²(period)` membership.
  have hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    K_ms_complex_memLp_restrict
  have hK_one : ∫ x, K_ms x ∂volume = 1 :=
    Sidon.Constructor.MOLemma21.K_ms_integral_eq_one_local
  -- Re-projected period-`u` K-bound.
  have h_re : (∑ j ∈ J, (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                (j / uQ_real : ℝ)).re ^ 2)
                ≤ K2_analytic - 1 :=
    Sidon.BundleEq2Schwartz.K_bound_for_mv_eq2 hK_L2 hK_one J hJ0
  -- Rewrite each summand via the Fourier–Bessel identity
  -- `(𝓕K_ms(j/u)).re = K_ms_fourier_lattice j`.
  have h_eq : (∑ j ∈ J, (K_ms_fourier_lattice j) ^ 2)
      = ∑ j ∈ J, (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                (j / uQ_real : ℝ)).re ^ 2 := by
    refine Finset.sum_congr rfl (fun j _ => ?_)
    -- `K_ms_complex = fun x => ((K_ms x : ℝ) : ℂ)` definitionally.
    rw [← bessel_identity j]
    rfl
  rw [h_eq]; exact h_re

/-! ## Target 2 — the genuine period-`u` Parseval split of `∫ autocorr f · K_ms`

We build the MO 2009 Lemma 2.1 pairing for `autocorr f` **alone** against
`K_ms`, mirroring `Sidon.Constructor.PoissonSampling.mo_lemma_21_core_via_periodization`
(which is stated for the combined `g = f∗f + autocorr f`) but for the single
function `autocorr f`.  All the heavy lifting — the period-`u` Poisson
sampling, the bilinear `AddCircle` Parseval, the inner-product/integral
bridge — is reused from `Sidon.Constructor.PoissonSampling` /
`Sidon.BilinearParseval` / `Sidon.TorusParseval` via their **generic**
(function-parametric) forms.

Throughout, `autocorr_complex f := fun x => ((autocorr f x : ℝ) : ℂ)`. -/

open Sidon.Constructor.PoissonSampling (periodization period_u_poisson_sampling)

/-- The complex lift of `autocorr f`. -/
def autocorr_complex (f : ℝ → ℝ) : ℝ → ℂ := fun x => ((autocorr f x : ℝ) : ℂ)

/-- `autocorr f` is integrable from `Integrable f` (local re-export). -/
theorem autocorr_integrable_E (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (autocorr f) volume :=
  Sidon.Constructor.MOLemma21.autocorr_integrable_local f hf

/-- `Integrable (autocorr_complex f)`. -/
theorem autocorr_complex_integrable (f : ℝ → ℝ) (hf : Integrable f volume) :
    Integrable (autocorr_complex f) volume :=
  (autocorr_integrable_E f hf).ofReal

/-- Support of `autocorr f` lies in `Ioo (-1/2) (1/2)` for `f` supported in
`Ioo (-1/4) (1/4)`.  (The `autocorr` branch of
`Sidon.Constructor.PoissonSampling.g_support_subset`.) -/
theorem autocorr_support_subset {f : ℝ → ℝ}
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Function.support (autocorr f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  -- `autocorr f = (f̌ ⋆ f)`; its support is in `supp (f∘neg) + supp f`.
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
  have ha_neg : f (-a) ≠ 0 := ha
  have ha' := hf_supp ha_neg
  have hb' := hf_supp hb
  simp only [Set.mem_Ioo] at ha' hb' ⊢
  subst hab
  constructor
  · linarith [ha'.2, hb'.1]
  · linarith [ha'.1, hb'.2]

/-- The non-overlap agreement for `autocorr f`: on `Ioo (-δ₁) δ₁` the
periodisation of `autocorr_complex f` agrees with the bare `autocorr_complex f`
(the `k ≠ 0` translates vanish because `1/2 + δ₁ = u`). -/
theorem periodization_autocorr_eq_on_Ioo_delta1 (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    {x : ℝ} (hx : x ∈ Set.Ioo (-delta1) delta1) :
    periodization uQ_real (autocorr_complex f) x = autocorr_complex f x := by
  have hu : uQ_real = 1/2 + delta1 :=
    Sidon.Constructor.PoissonSampling.uQ_real_eq_half_add_delta1
  have hsupp : Function.support (autocorr f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
    autocorr_support_subset hf_supp
  simp only [Set.mem_Ioo] at hx
  have h_vanish : ∀ k : ℤ, k ≠ 0 → autocorr_complex f (x + k • uQ_real) = 0 := by
    intro k hk
    have hak : autocorr f (x + k • uQ_real) = 0 := by
      by_contra hne
      have hmem : (x + k • uQ_real) ∈ Set.Ioo (-(1/2 : ℝ)) (1/2) := hsupp hne
      simp only [Set.mem_Ioo] at hmem
      have hku : (k • uQ_real : ℝ) = (k : ℝ) * uQ_real := by rw [zsmul_eq_mul]
      have hupos : 0 < uQ_real := by rw [hu]; linarith
      rcases lt_or_gt_of_ne hk with hkneg | hkpos
      · have hk1 : (k : ℝ) ≤ -1 := by exact_mod_cast Int.le_sub_one_of_lt hkneg
        have : (k : ℝ) * uQ_real ≤ -uQ_real := by nlinarith [hk1, hupos]
        rw [hku] at hmem
        nlinarith [hmem.1, hx.2, this, hu]
      · have hk1 : (1 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hkpos
        have : uQ_real ≤ (k : ℝ) * uQ_real := by nlinarith [hk1, hupos]
        rw [hku] at hmem
        nlinarith [hmem.2, hx.1, this, hu]
    unfold autocorr_complex; rw [hak]; simp
  unfold periodization
  rw [tsum_eq_single 0 (fun k hk => h_vanish k hk)]
  simp

/-- On the period `Ioc (-(u/2)) (u/2)`, the periodisation of
`autocorr_complex f` equals the finite sum over `k ∈ {-1,0,1}` of the
translates. -/
theorem periodization_autocorr_eq_finsum_on_period (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    {x : ℝ} (hx : x ∈ Set.Ioc (-(uQ_real/2)) (uQ_real/2)) :
    periodization uQ_real (autocorr_complex f) x
      = ∑ k ∈ Finset.Icc (-1 : ℤ) 1, autocorr_complex f (x + k • uQ_real) := by
  have hu : uQ_real = 1/2 + delta1 :=
    Sidon.Constructor.PoissonSampling.uQ_real_eq_half_add_delta1
  have hδ1 : 0 < delta1 := (delta_positivities).1
  have hsupp : Function.support (autocorr f) ⊆ Set.Ioo (-(1/2 : ℝ)) (1/2) :=
    autocorr_support_subset hf_supp
  simp only [Set.mem_Ioc] at hx
  have h_vanish : ∀ k : ℤ, k ∉ Finset.Icc (-1 : ℤ) 1 →
      autocorr_complex f (x + k • uQ_real) = 0 := by
    intro k hk
    simp only [Finset.mem_Icc, not_and_or, not_le] at hk
    have hak : autocorr f (x + k • uQ_real) = 0 := by
      by_contra hne
      have hmem : (x + k • uQ_real) ∈ Set.Ioo (-(1/2 : ℝ)) (1/2) := hsupp hne
      simp only [Set.mem_Ioo] at hmem
      have hku : (k • uQ_real : ℝ) = (k : ℝ) * uQ_real := by rw [zsmul_eq_mul]
      have hupos : 0 < uQ_real := by rw [hu]; linarith
      rcases hk with hkneg | hkpos
      · have hk2 : (k : ℝ) ≤ -2 := by exact_mod_cast Int.le_sub_one_of_lt hkneg
        rw [hku] at hmem
        nlinarith [hmem.1, hx.2, hk2, hupos, hu, hδ1]
      · have hk2 : (2 : ℝ) ≤ (k : ℝ) := by exact_mod_cast hkpos
        rw [hku] at hmem
        nlinarith [hmem.2, hx.1, hk2, hupos, hu, hδ1]
    unfold autocorr_complex; rw [hak]; simp
  unfold periodization
  rw [tsum_eq_sum (s := Finset.Icc (-1 : ℤ) 1) (fun k hk => h_vanish k hk)]

/-- `autocorr_complex f` is a.e.-strongly-measurable. -/
theorem autocorr_complex_aestronglyMeasurable (f : ℝ → ℝ) (hf : Integrable f volume) :
    AEStronglyMeasurable (autocorr_complex f) volume :=
  (autocorr_complex_integrable f hf).aestronglyMeasurable

/-- The periodisation of `autocorr_complex f` is `L²` on one period, via the
finite-sum decomposition.  Each translate is everywhere bounded by `∫f²`
(`autocorr_abs_le_local`), hence `L²` on the finite-measure interval. -/
theorem periodization_autocorr_memLp_period (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    MemLp (periodization uQ_real (autocorr_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
  have h_sum_memLp : MemLp
      (∑ k ∈ Finset.Icc (-1 : ℤ) 1, fun x => autocorr_complex f (x + k • uQ_real)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
    refine memLp_finset_sum' (Finset.Icc (-1 : ℤ) 1) (fun k _ => ?_)
    haveI : IsFiniteMeasure (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
      refine ⟨?_⟩
      rw [Measure.restrict_apply_univ, Real.volume_Ioc]
      exact ENNReal.ofReal_lt_top
    set C : ℝ := ∫ x, f x ^ 2 ∂volume with hC_def
    have hmeas : AEStronglyMeasurable (fun x => autocorr_complex f (x + k • uQ_real))
        (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) := by
      have h1 : AEStronglyMeasurable (autocorr_complex f) volume :=
        autocorr_complex_aestronglyMeasurable f hf_int
      have h2 := h1.comp_measurePreserving
        (measurePreserving_add_right (volume : Measure ℝ) (k • uQ_real))
      exact (by simpa [Function.comp] using h2 : AEStronglyMeasurable
        (fun x => autocorr_complex f (x + k • uQ_real)) volume).restrict
    have h_bound : ∀ᵐ x ∂(volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))),
        ‖autocorr_complex f (x + k • uQ_real)‖ ≤ C := by
      refine Filter.Eventually.of_forall fun x => ?_
      unfold autocorr_complex
      rw [Complex.norm_real, Real.norm_eq_abs]
      exact Sidon.Constructor.MOLemma21.autocorr_abs_le_local f hf_L2 _
    exact MemLp.of_bound hmeas C h_bound
  refine MemLp.ae_eq ?_ h_sum_memLp
  refine (ae_restrict_iff' measurableSet_Ioc).mpr ?_
  refine Filter.Eventually.of_forall (fun x hx => ?_)
  rw [Finset.sum_apply]
  exact (periodization_autocorr_eq_finsum_on_period f hf_supp hx).symm

/-! ### The inner product `⟪lift K_ms, lift autocorr_per⟫ = (1/u)·∫_ℝ autocorr·K_ms` -/

/-- The interval integral over one period of `autocorr_per · conj(K_ms)` equals
the full-line integral `∫_ℝ autocorr·K_ms`: `K_ms` vanishes off `(-δ₁,δ₁)`, and
on `(-δ₁,δ₁)` the periodisation agrees with `autocorr`. -/
theorem intervalIntegral_autocorrper_Kms_eq_full (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    (∫ x in (-(uQ_real/2))..(uQ_real/2),
        periodization uQ_real (autocorr_complex f) x * starRingEnd ℂ (K_ms_complex x))
      = ∫ x, autocorr_complex f x * K_ms_complex x ∂volume := by
  have hδ1 : 0 < delta1 := (delta_positivities).1
  rw [intervalIntegral.integral_of_le (by linarith [uQ_real_pos] : -(uQ_real/2) ≤ uQ_real/2)]
  have h_integrand : ∀ x ∈ Set.Ioc (-(uQ_real/2)) (uQ_real/2),
      periodization uQ_real (autocorr_complex f) x * starRingEnd ℂ (K_ms_complex x)
        = autocorr_complex f x * K_ms_complex x := by
    intro x hx
    by_cases hxδ : x ∈ Set.Ioo (-delta1) delta1
    · rw [periodization_autocorr_eq_on_Ioo_delta1 f hf_supp hxδ,
        Sidon.Constructor.MOLemma21.conj_K_ms_complex]
    · have hK0 : K_ms_complex x = 0 := by
        have hxabs : delta1 ≤ |x| := by
          rw [Set.mem_Ioo, not_and_or, not_lt, not_lt] at hxδ
          rcases hxδ with h | h
          · rw [abs_of_nonpos (by linarith)]; linarith
          · rw [le_abs]; left; exact h
        show ((K_ms x : ℝ) : ℂ) = 0
        rw [Sidon.Constructor.MOLemma21.K_ms_zero_outside x hxabs]; simp
      rw [hK0, map_zero, mul_zero, mul_zero]
  rw [setIntegral_congr_fun (μ := volume) measurableSet_Ioc h_integrand]
  refine MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ?_
  intro x hx
  have hK0 : K_ms_complex x = 0 := by
    by_contra h_ne
    exact hx (Sidon.Constructor.MOLemma21.K_ms_complex_support h_ne)
  rw [hK0, mul_zero]

/-- **The inner product = integral identity (autocorr periodisation form).**
`⟪lift K_ms, lift autocorr_per⟫ = (1/u)·∫_ℝ autocorr·K_ms`. -/
theorem inner_autocorrper_Kms_eq (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hper_L2 : MemLp (periodization uQ_real (autocorr_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    @inner ℂ _ _
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          K_ms_complex K_ms_complex_memLp_restrict).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex))
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          (periodization uQ_real (autocorr_complex f)) hper_L2).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (autocorr_complex f))))
      = (uQ_real⁻¹ : ℂ) * ∫ x, autocorr_complex f x * K_ms_complex x ∂volume := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  rw [Sidon.BilinearParseval.inner_toLp_eq_intervalIntegral uQ_real uQ_real_pos
    (periodization uQ_real (autocorr_complex f)) K_ms_complex hper_L2
    K_ms_complex_memLp_restrict]
  rw [intervalIntegral_autocorrper_Kms_eq_full f hf_supp]

/-- The `toLp` representative of `lift autocorr_per` has Fourier coefficient
`(1/u)·𝓕(autocorr_complex f)(j/u)` — the `toLp`-wrapped form of
`period_u_poisson_sampling`. -/
theorem fourierCoeff_toLp_autocorrper_eq (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hper_L2 : MemLp (periodization uQ_real (autocorr_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) (j : ℤ) :
    haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
    fourierCoeff
        ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
            (periodization uQ_real (autocorr_complex f)) hper_L2).toLp
          (AddCircle.liftIoc uQ_real (-(uQ_real/2))
            (periodization uQ_real (autocorr_complex f))) : AddCircle uQ_real → ℂ) j
      = (1 / uQ_real : ℂ) * Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  have h_ae :
      ((Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
          (periodization uQ_real (autocorr_complex f)) hper_L2).toLp
        (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (autocorr_complex f)))
          : AddCircle uQ_real → ℂ)
        =ᵐ[@AddCircle.haarAddCircle uQ_real ⟨uQ_real_pos⟩]
          AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (autocorr_complex f)) :=
    MemLp.coeFn_toLp _
  rw [fourierCoeff_congr_ae h_ae]
  have hint : Integrable (autocorr_complex f) volume := autocorr_complex_integrable f hf_int
  exact period_u_poisson_sampling uQ_real_pos (autocorr_complex f) hint j

/-! ### The MO 2.1 core for `autocorr f` (hypothesis-free via periodisation) -/

/-- **MO 2009 Lemma 2.1 core for `autocorr f` (hypothesis-free via
periodisation).**

`∫_ℝ autocorr f · K_ms = (1/u)·∑'_j 𝓕(autocorr f)(j/u)·conj(𝓕K_ms(j/u))`,

with the period-`u` Poisson sampling discharged by the proved
`period_u_poisson_sampling`.  Same algebra as
`mo_lemma_21_core_via_periodization`, but for `autocorr f` alone. -/
theorem autocorr_mo_core (f : ℝ → ℝ) (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    (∫ x, autocorr_complex f x * K_ms_complex x ∂volume)
      = (1 / uQ_real : ℂ) * ∑' j : ℤ,
          Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
            * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)) := by
  haveI : Fact (0 < uQ_real) := ⟨uQ_real_pos⟩
  have hu_ne : (uQ_real : ℝ) ≠ 0 := ne_of_gt uQ_real_pos
  have hper_L2 : MemLp (periodization uQ_real (autocorr_complex f)) 2
      (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))) :=
    periodization_autocorr_memLp_period f hf_int hf_L2 hf_supp
  set Fper := (Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
      (periodization uQ_real (autocorr_complex f)) hper_L2).toLp
    (AddCircle.liftIoc uQ_real (-(uQ_real/2)) (periodization uQ_real (autocorr_complex f)))
      with hFper_def
  set H := (Sidon.BilinearParseval.memLp_liftIoc_haarAddCircle uQ_real uQ_real_pos
      K_ms_complex K_ms_complex_memLp_restrict).toLp
    (AddCircle.liftIoc uQ_real (-(uQ_real/2)) K_ms_complex) with hH_def
  have h_parseval :
      ∑' j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle uQ_real → ℂ) j)
        = @inner ℂ _ _ H Fper :=
    Sidon.TorusParseval.bilinear_parseval_addCircle_Lp_tsum uQ_real uQ_real_pos Fper H
  have h_coef_F : ∀ j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                    = (1 / uQ_real : ℂ) * Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ) := by
    intro j; rw [hFper_def]; exact fourierCoeff_toLp_autocorrper_eq f hf_int hper_L2 j
  have h_coef_H : ∀ j : ℤ, fourierCoeff (H : AddCircle uQ_real → ℂ) j
                    = (1 / uQ_real : ℂ) * Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ) := by
    intro j; rw [hH_def]; exact Sidon.Constructor.MOLemma21.fourierCoeff_Kms_eq j
  have h_inner : @inner ℂ _ _ H Fper
      = (uQ_real⁻¹ : ℂ) * ∫ x, autocorr_complex f x * K_ms_complex x ∂volume := by
    rw [hH_def, hFper_def]; exact inner_autocorrper_Kms_eq f hf_supp hper_L2
  have h_tsum_eq :
      ∑' j : ℤ, fourierCoeff (Fper : AddCircle uQ_real → ℂ) j
                * starRingEnd ℂ (fourierCoeff (H : AddCircle uQ_real → ℂ) j)
        = ((1 / uQ_real ^ 2 : ℝ) : ℂ) * ∑' j : ℤ,
            Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
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
      Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ)) with hT_def
  set I : ℂ := ∫ x, autocorr_complex f x * K_ms_complex x ∂volume with hI_def
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

/-! ### The real `1/u + (1/u)·∑'` form (the genuine split)

The autocorrelation Fourier coefficient is `𝓕(autocorr f)(ξ) =
𝓕f(ξ)·conj(𝓕f(ξ)) = (‖𝓕f(ξ)‖² : ℂ)` (Wiener–Khinchin), and `𝓕K_ms(j/u)`
is real (`= (K̂_ms(j) : ℂ)`).  So each summand of the complex core is the
real number `‖𝓕f(j/u)‖²·K̂_ms(j)`.  Splitting off `j = 0` (which gives
`‖𝓕f(0)‖²·K̂_ms(0) = 1·1 = 1`) yields the genuine real split. -/

/-- The squared modulus of the period-`u` lattice Fourier coefficient of `f`,
`Fmod_sq f j = ‖𝓕f(j/u)‖²`.  This is the **correct** autocorrelation summand
modulus (Wiener–Khinchin), as opposed to the bundle's `Fsq_re f j =
(Re 𝓕f(j/u))²`. -/
def Fmod_sq (f : ℝ → ℝ) (j : ℤ) : ℝ :=
  ‖Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)‖ ^ 2

/-- Per-summand identity: the complex core summand for `autocorr f` is the real
number `Fmod_sq f j · K_ms_fourier_lattice j`. -/
theorem autocorr_summand_eq (f : ℝ → ℝ) (hf : Integrable f volume) (j : ℤ) :
    Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))
      = ((Fmod_sq f j * K_ms_fourier_lattice j : ℝ) : ℂ) := by
  -- `𝓕(autocorr f)(ξ) = 𝓕f(ξ)·conj(𝓕f(ξ))`.
  have h_ac : Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
      = Real.fourierIntegral (fun x => ((f x : ℂ))) (j / uQ_real : ℝ)
        * starRingEnd ℂ (Real.fourierIntegral (fun x => ((f x : ℂ))) (j / uQ_real : ℝ)) := by
    have := Sidon.Constructor.fourierIntegral_autocorr_real f hf (j / uQ_real : ℝ)
    simpa [autocorr_complex] using this
  -- `conj(𝓕K_ms(j/u)) = 𝓕K_ms(j/u) = (K̂(j) : ℂ)`.
  have h_K : starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))
      = ((K_ms_fourier_lattice j : ℝ) : ℂ) := by
    rw [Sidon.Constructor.MOLemma21.conj_fourierIntegral_K_ms,
      Sidon.Constructor.MOLemma21.fourierIntegral_K_ms_eq_re, bessel_identity]
  rw [h_ac, h_K]
  -- `(z · conj z) · (K̂ : ℂ) = (‖z‖² · K̂ : ℂ)`.
  set z := Real.fourierIntegral (fun x => ((f x : ℂ))) (j / uQ_real : ℝ) with hz
  have hzz : z * starRingEnd ℂ z = ((‖z‖ ^ 2 : ℝ) : ℂ) := by
    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
  rw [hzz, Fmod_sq]
  push_cast
  ring

/-- Summability of the (real) autocorrelation summand family
`j ↦ Fmod_sq f j · K_ms_fourier_lattice j`.  Each term is nonnegative and
dominated by `Fmod_sq f j · 1 = ‖𝓕f(j/u)‖²` (since `0 ≤ K̂_ms(j) ≤ 1`), which
is summable by the period-`u` Plancherel of `f`
(`Sidon.Constructor.PoissonSummable.summable_normSq_fourierIntegral_f`; `f`
fits one period as `supp f ⊆ (-1/4,1/4) ⊆ (-u/2,u/2)`). -/
theorem autocorr_real_summand_summable (f : ℝ → ℝ)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) :
    Summable (fun j : ℤ => Fmod_sq f j * K_ms_fourier_lattice j) := by
  -- Block 3: `Summable (j ↦ ‖𝓕f(j/u)‖²) = Summable (Fmod_sq f)`.
  have h_f : Summable (fun j : ℤ => Fmod_sq f j) :=
    Sidon.Constructor.PoissonSummable.summable_normSq_fourierIntegral_f f hf_L2 hf_supp
  refine Summable.of_nonneg_of_le (fun j => ?_) (fun j => ?_) h_f
  · exact mul_nonneg (sq_nonneg _) (K_ms_fourier_lattice_nonneg j)
  · -- `Fmod_sq f j · K̂(j) ≤ Fmod_sq f j · 1 = Fmod_sq f j`.
    have hK1 : K_ms_fourier_lattice j ≤ 1 :=
      Sidon.Constructor.Glue.K_ms_fourier_lattice_le_one j
    have hF0 : 0 ≤ Fmod_sq f j := sq_nonneg _
    nlinarith [hF0, hK1, K_ms_fourier_lattice_nonneg j]

/-- The complex integral `∫ autocorr_complex·K_ms_complex` is the real cast of
`∫ autocorr·K_ms`. -/
theorem integral_autocorr_Kms_complex_eq (f : ℝ → ℝ) :
    (∫ x, autocorr_complex f x * K_ms_complex x ∂volume)
      = ((∫ x, autocorr f x * K_ms x ∂volume : ℝ) : ℂ) := by
  rw [show ((∫ x, autocorr f x * K_ms x ∂volume : ℝ) : ℂ)
        = ∫ x, ((autocorr f x * K_ms x : ℝ) : ℂ) ∂volume from
      (integral_ofReal (f := fun x => autocorr f x * K_ms x)).symm]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  unfold autocorr_complex K_ms_complex
  push_cast; ring

/-- The `j = 0` term of the real autocorrelation summand is `1`:
`Fmod_sq f 0 · K̂_ms(0) = ‖𝓕f(0)‖²·1 = ‖∫f‖² = 1`. -/
theorem autocorr_summand_zero (f : ℝ → ℝ) (hf_one : ∫ x, f x ∂volume = 1) :
    Fmod_sq f 0 * K_ms_fourier_lattice 0 = 1 := by
  rw [K_ms_fourier_lattice_zero, mul_one, Fmod_sq]
  have h0 : ((0 : ℤ) : ℝ) / uQ_real = 0 := by push_cast; ring
  rw [h0]
  -- `𝓕f(0) = ∫f = 1` (real `f`), so `‖(1:ℂ)‖² = 1`.
  rw [Sidon.FourierAux.fourierIntegral_zero (fun x => ((f x : ℝ) : ℂ))]
  rw [show (∫ x, ((f x : ℝ) : ℂ) ∂volume) = ((∫ x, f x ∂volume : ℝ) : ℂ) from
      integral_ofReal (𝕜 := ℂ) (f := f)]
  rw [hf_one]
  simp

/-- **The genuine MV Eq.(2) period-`u` Parseval split of `∫ autocorr f · K_ms`.**

For admissible `f`,

  `∫ autocorr f · K_ms = 1/u + (1/u)·∑'_{j≠0} ‖𝓕f(j/u)‖² · K̂_ms(j)`,

with `Fmod_sq f j = ‖𝓕f(j/u)‖²` the **squared modulus** (Wiener–Khinchin,
NOT `(Re 𝓕f)²`) and the constant `1/u` (NOT `1`).  This is the corrected
statement of the headline residual `h_parseval_split` (see the file
docstring for the discrepancy). -/
theorem autocorr_period_u_split (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_one : ∫ x, f x ∂volume = 1) :
    ∫ x, autocorr f x * K_ms x ∂volume
      = 1 / uQ_real
        + (1 / uQ_real) * ∑' j : ℤ, if j = 0 then 0 else
            Fmod_sq f j * K_ms_fourier_lattice j := by
  have hu_ne : (uQ_real : ℝ) ≠ 0 := ne_of_gt uQ_real_pos
  -- Real summand family `s j = Fmod_sq f j · K̂_ms(j)`; summable.
  set s : ℤ → ℝ := fun j => Fmod_sq f j * K_ms_fourier_lattice j with hs_def
  have hs_summable : Summable s := autocorr_real_summand_summable f hf_L2 hf_supp
  -- Complex core: `∫ ac·K = (1/u)·∑'_j 𝓕(ac)(j/u)·conj(𝓕K(j/u))`.
  have hcore := autocorr_mo_core f hf_int hf_L2 hf_supp
  -- Each complex summand is `(s j : ℂ)` (via `autocorr_summand_eq`).
  have h_summand : ∀ j : ℤ,
      Real.fourierIntegral (autocorr_complex f) (j / uQ_real : ℝ)
          * starRingEnd ℂ (Real.fourierIntegral K_ms_complex (j / uQ_real : ℝ))
        = ((s j : ℝ) : ℂ) := by
    intro j; rw [hs_def]; exact autocorr_summand_eq f hf_int j
  rw [tsum_congr h_summand] at hcore
  -- Push the ℝ→ℂ cast through the (summable) tsum.
  rw [show (∑' j : ℤ, ((s j : ℝ) : ℂ)) = ((∑' j : ℤ, s j : ℝ) : ℂ) from
      (Complex.ofRealCLM.map_tsum hs_summable).symm] at hcore
  -- `∫ ac_ℂ·K_ℂ = ((∫ ac·K : ℝ) : ℂ)`.
  rw [integral_autocorr_Kms_complex_eq f] at hcore
  -- `((∫ac·K : ℝ):ℂ) = (1/u:ℂ)·((∑' s : ℝ):ℂ)`; take real parts.
  have hre := congrArg Complex.re hcore
  rw [Complex.ofReal_re] at hre
  rw [show ((1 / uQ_real : ℂ) * ((∑' j : ℤ, s j : ℝ) : ℂ)).re
        = (1 / uQ_real) * (∑' j : ℤ, s j) from by
      rw [show ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) by push_cast; rfl]
      rw [← Complex.ofReal_mul, Complex.ofReal_re]] at hre
  -- `hre : ∫ ac·K = (1/u)·∑' s`.  Split off `j = 0`.
  rw [hre]
  have h_split : ∑' j : ℤ, s j = s 0 + ∑' j : ℤ, (if j = 0 then 0 else s j) :=
    hs_summable.tsum_eq_add_tsum_ite 0
  rw [h_split]
  have hs0 : s 0 = 1 := by rw [hs_def]; exact autocorr_summand_zero f hf_one
  rw [hs0]
  -- `(1/u)·(1 + tail) = 1/u + (1/u)·tail`, with `tail` matching the goal's `if`-tsum.
  rw [hs_def]
  ring

/-! ## Target 3 — `h_F_bound`: the residual (genuinely missing primitive)

`h_F_bound` reads `∑_{j∈J} (Re 𝓕f(j/u))⁴ ≤ autoconvolution_ratio f − 1`
(since `Fsq_re f j = (Re 𝓕f(j/u))²`, so `(Fsq_re f j)² = (Re 𝓕f(j/u))⁴`).
By `(Re 𝓕f(j/u))⁴ ≤ ‖𝓕f(j/u)‖⁴ = ‖𝓕(f∗f)(j/u)‖²` (Wiener–Khinchin for
`f∗f`), a bound on the period-`u` lattice sum `∑_{j≠0} ‖𝓕(f∗f)(j/u)‖²` would
close it.  That sum is `u·∫_{period}‖(f∗f)_per‖² − 1` *only if `f∗f` fits one
period* — but `supp (f∗f) = (-1/2,1/2)` **overflows** `(-u/2,u/2)`, so the
period-`u` Plancherel of `f∗f` is unavailable, and the periodisation sums the
overflow tails (a genuinely different quantity).  Relating it to
`‖f∗f‖_∞ = autoconvolution_ratio f` further needs the Hölder bound
`‖f∗f‖²_{L²} ≤ ‖f∗f‖_{L¹}·‖f∗f‖_∞`.  Both are the documented missing
primitives (an `f∗f` periodisation reconciliation + Fourier Hölder), distinct
from the `autocorr`-against-`K_ms` pairing closed above. -/

/-- The honest reduction step for `h_F_bound`: `∑_{j∈J} (Re 𝓕f(j/u))⁴ ≤
∑_{j∈J} ‖𝓕f(j/u)‖⁴`.  (The remaining `∑‖𝓕f(j/u)‖⁴ ≤ R(f) − 1` is the
period-`u` Plancherel of the overflowing `f∗f` — see the section docstring;
not provable in mathlib `v4.29.1` for general admissible `f`.) -/
theorem field_h_F_bound_reduction (f : ℝ → ℝ) (J : Finset ℤ) :
    (∑ j ∈ J, ((Real.fourierIntegral (fun x => ((f x : ℂ)))
                  ((j : ℝ) / uQ_real)).re ^ 2) ^ 2)
      ≤ ∑ j ∈ J, ‖Real.fourierIntegral (fun x => ((f x : ℂ)))
                  ((j : ℝ) / uQ_real)‖ ^ 4 := by
  refine Finset.sum_le_sum (fun j _ => ?_)
  set c : ℝ := (Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re with hc
  set N : ℝ := ‖Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)‖ with hN
  -- `(c²)² = c⁴ ≤ N⁴` from `c² ≤ N²` (`re² ≤ norm²`) and `0 ≤ c², N²`.
  have hcsq : c ^ 2 ≤ N ^ 2 :=
    Sidon.BundleEq2Schwartz.re_sq_le_norm_sq
      (Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real))
  nlinarith [hcsq, sq_nonneg c, sq_nonneg N, sq_nonneg (c^2), sq_nonneg (N^2)]

end

end Sidon.Constructor.Eq2Split
