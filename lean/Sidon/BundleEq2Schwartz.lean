/-
Sidon Autocorrelation Project — Bundle Eq.(2) K-side period-1 Plancherel
========================================================================

This file provides the unconditional K-side discharge of MV Lemma 3.1
Eq.(2) for the multi-scale kernel `K_ms`, in both the period-`u` and
period-`1` normalisations.  Concretely, it proves

```
  ∑_{r ∈ J} (Re 𝓕K_ms(r))² ≤ K2_analytic - 1   (period-1)
  ∑_{j ∈ J} (Re 𝓕K_ms(j/u))² ≤ K2_analytic - 1 (period-u, via u ≤ 1 slack)
```

via mathlib period-`u` Plancherel applied to `K_ms` (supported in
`[-δ₁, δ₁] ⊂ Ioc(-u/2, u/2)` for `u = 0.638`).  These are the K-side
ingredients for the general bundle field

```
  hEq2 : LHS2 ≤ 1 + √(autoconvolution_ratio f - 1) * √(K2_analytic - 1)
```

(see `Sidon.MultiScale.ExtremiserPrimitives`).

The Schwartz-specific F-side and bilinear Parseval-split `Prop`s and
their assembly into `hEq2_schwartz_from_atomic` previously also lived
in this file; they have been removed because the Schwartz instance
they served was vacuous (`SchwartzAtomic f_s` is unsatisfiable: the
Parseval split required `f̂(r) = 0` for cofinite `r`, which combined
with Paley–Wiener + Carlson forces `f ≡ 0`).

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas
import Sidon.MultiScale
import Sidon.TorusParseval
import Sidon.FourierAux

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open scoped BigOperators
open scoped Classical
open scoped Real
open scoped FourierTransform Topology SchwartzMap
open MeasureTheory Complex

namespace Sidon.BundleEq2Schwartz

open Sidon.MultiScale
open Sidon.FourierAux (autocorr)

noncomputable section

/-! ## Preliminaries: real-valued Schwartz functions
-/

/-- The MV Eq.(2) LHS for a Schwartz `f`:
`LHS2_schwartz f := ∫ (autocorr f) · K_ms`, where
`autocorr f x := ∫ t, f(t)·f(x+t) dt` is the convolutional
autocorrelation (MV's `f∘f`). -/
def LHS2_schwartz (f : 𝓢(ℝ, ℝ)) : ℝ :=
  ∫ x, autocorr (f : ℝ → ℝ) x * K_ms x ∂volume

/-! ## `K_ms` support: K_ms(x) = 0 for |x| ≥ δ₁ -/

/-- The first scale `δ₁ = 138/1000` is strictly less than `u/2 = 319/1000`. -/
lemma delta1_lt_u_half : delta1 < uQ_real / 2 := by
  unfold delta1 delta1Q uQ_real uQ
  push_cast
  norm_num

/-- `K_arc(δ, x) = 0` whenever `|x| ≥ δ`.  Re-exported from
`Sidon.MultiScale.K_arc_eq_zero_outside` (now proved via
`MeasureTheory.support_convolution_subset`). -/
lemma K_arc_eq_zero_outside (δ x : ℝ) (h : δ ≤ |x|) : K_arc δ x = 0 :=
  Sidon.MultiScale.K_arc_eq_zero_outside δ x h

/-- `K_ms x = 0` whenever `|x| ≥ δ₁` (the largest scale). -/
lemma K_ms_eq_zero_outside (x : ℝ) (h : delta1 ≤ |x|) : K_ms x = 0 := by
  show K_ms x = 0
  unfold K_ms
  have h_delta_order : delta3 ≤ delta2 ∧ delta2 ≤ delta1 := by
    unfold delta1 delta2 delta3 delta1Q delta2Q delta3Q
    refine ⟨?_, ?_⟩ <;> (push_cast; norm_num)
  have h1 : K_arc delta1 x = 0 := K_arc_eq_zero_outside delta1 x h
  have h2 : K_arc delta2 x = 0 := K_arc_eq_zero_outside delta2 x (le_trans h_delta_order.2 h)
  have h3 : K_arc delta3 x = 0 :=
    K_arc_eq_zero_outside delta3 x (le_trans (le_trans h_delta_order.1 h_delta_order.2) h)
  rw [h1, h2, h3]
  ring

/-- Support of `K_ms` (complex lift) lies in `Ioc(-(u/2), u/2)`. -/
lemma K_ms_complex_support :
    Function.support (fun x => ((K_ms x : ℝ) : ℂ))
      ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
  intro x hx
  have hK_ne : K_ms x ≠ 0 := by
    intro heq
    apply hx
    show ((K_ms x : ℝ) : ℂ) = 0
    rw [heq]
    rfl
  have h_lt : |x| < delta1 := by
    by_contra h_ge
    push_neg at h_ge
    exact hK_ne (K_ms_eq_zero_outside x h_ge)
  have hx_abs : |x| < uQ_real / 2 := lt_trans h_lt delta1_lt_u_half
  refine ⟨?_, ?_⟩
  · have := abs_lt.mp hx_abs; linarith
  · have := abs_lt.mp hx_abs; linarith

/-- Support of `K_ms²` (real) lies in `Ioc(-(u/2), u/2)`. -/
lemma K_ms_sq_support :
    Function.support (fun x => K_ms x ^ 2) ⊆ Set.Ioc (-(uQ_real/2)) (uQ_real/2) := by
  intro x hx
  have hK_ne : K_ms x ≠ 0 := by
    intro heq
    apply hx
    show K_ms x ^ 2 = 0
    rw [heq]; ring
  have h_lt : |x| < delta1 := by
    by_contra h_ge
    push_neg at h_ge
    exact hK_ne (K_ms_eq_zero_outside x h_ge)
  have hx_abs : |x| < uQ_real / 2 := lt_trans h_lt delta1_lt_u_half
  refine ⟨?_, ?_⟩
  · have := abs_lt.mp hx_abs; linarith
  · have := abs_lt.mp hx_abs; linarith

/-! ## Period-1 support lemmas for `K_ms`

`K_ms` has support in `[-δ₁, δ₁]` with `δ₁ = 138/1000 < 1/2`, hence is
also supported in `Ioc(-(1/2), 1/2)`.  This stronger inclusion lets us
apply *period-1* Parseval to `K_ms` (the natural normalization, since
both `K_ms` and `autocorr f` of an `f` supported in `(-1/4, 1/4)` lie
inside the length-`1` interval `(-1/2, 1/2)`). -/

/-- `δ₁ = 138/1000 < 1/2`. -/
lemma delta1_lt_one_half : delta1 < 1 / 2 := by
  unfold delta1 delta1Q
  push_cast
  norm_num

/-- Support of `K_ms` (complex lift) lies in `Ioc(-(1/2), 1/2)`. -/
lemma K_ms_complex_support_period1 :
    Function.support (fun x => ((K_ms x : ℝ) : ℂ))
      ⊆ Set.Ioc (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  have hK_ne : K_ms x ≠ 0 := by
    intro heq
    apply hx
    show ((K_ms x : ℝ) : ℂ) = 0
    rw [heq]
    rfl
  have h_lt : |x| < delta1 := by
    by_contra h_ge
    push_neg at h_ge
    exact hK_ne (K_ms_eq_zero_outside x h_ge)
  have hx_abs : |x| < 1 / 2 := lt_trans h_lt delta1_lt_one_half
  refine ⟨?_, ?_⟩
  · have := abs_lt.mp hx_abs; linarith
  · have := abs_lt.mp hx_abs; linarith

/-- Support of `K_ms²` (real) lies in `Ioc(-(1/2), 1/2)`. -/
lemma K_ms_sq_support_period1 :
    Function.support (fun x => K_ms x ^ 2) ⊆ Set.Ioc (-(1/2 : ℝ)) (1/2) := by
  intro x hx
  have hK_ne : K_ms x ≠ 0 := by
    intro heq
    apply hx
    show K_ms x ^ 2 = 0
    rw [heq]; ring
  have h_lt : |x| < delta1 := by
    by_contra h_ge
    push_neg at h_ge
    exact hK_ne (K_ms_eq_zero_outside x h_ge)
  have hx_abs : |x| < 1 / 2 := lt_trans h_lt delta1_lt_one_half
  refine ⟨?_, ?_⟩
  · have := abs_lt.mp hx_abs; linarith
  · have := abs_lt.mp hx_abs; linarith

/-! ## Pointwise real-to-complex norm identity -/

/-- For real-valued integrands, `‖((r : ℝ) : ℂ)‖² = r²`. -/
lemma norm_sq_ofReal (r : ℝ) : ‖((r : ℝ) : ℂ)‖ ^ 2 = r ^ 2 := by
  rw [Complex.norm_real, Real.norm_eq_abs, sq_abs]

/-! ## K2_analytic as a restricted integral -/

/-- `K2_analytic = ∫_{(-u/2, u/2]} K_ms²` (restriction since K_ms vanishes
outside the support). -/
theorem K2_analytic_eq_restricted_integral :
    K2_analytic = ∫ x in (-(uQ_real/2))..(uQ_real/2), K_ms x ^ 2 := by
  show K2_analytic = _
  unfold K2_analytic
  exact (intervalIntegral.integral_eq_integral_of_support_subset
    K_ms_sq_support).symm

/-- `K2_analytic = ∫_{(-1/2, 1/2]} K_ms²` (period-1 restriction since K_ms
vanishes outside `[-δ₁, δ₁] ⊂ (-1/2, 1/2)`). -/
theorem K2_analytic_eq_restricted_integral_period1 :
    K2_analytic = ∫ x in (-(1/2 : ℝ))..(1/2), K_ms x ^ 2 := by
  show K2_analytic = _
  unfold K2_analytic
  exact (intervalIntegral.integral_eq_integral_of_support_subset
    K_ms_sq_support_period1).symm

/-! ## Plancherel-at-lattice for K_ms -/

/-- Plancherel-at-lattice for K_ms in `K2_analytic` form:
`∑'_j ‖𝓕K_ms(j/u)‖² = u · K2_analytic`. -/
theorem plancherel_K_ms_K2_analytic
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    ∑' j : ℤ, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      (j / uQ_real : ℝ)‖ ^ 2
      = uQ_real * K2_analytic := by
  have h := Sidon.TorusParseval.plancherel_at_lattice_period_u
    uQ_real uQ_real_pos (fun x => ((K_ms x : ℝ) : ℂ))
    K_ms_complex_support hK_L2
  have h_pt : ∀ x : ℝ, ‖((K_ms x : ℝ) : ℂ)‖ ^ 2 = K_ms x ^ 2 := fun x => norm_sq_ofReal _
  have h_int_eq :
      ∫ x in (-(uQ_real/2))..(uQ_real/2), ‖((K_ms x : ℝ) : ℂ)‖ ^ 2
        = ∫ x in (-(uQ_real/2))..(uQ_real/2), K_ms x ^ 2 := by
    apply intervalIntegral.integral_congr
    intro x _
    exact h_pt x
  rw [h_int_eq] at h
  rw [K2_analytic_eq_restricted_integral]
  exact h

/-! ## Period-1 MemLp transport for K_ms

The user-supplied L² hypothesis `hK_L2_torus` for `K_ms` is stated on
`Ioc(-u/2, u/2)`.  Since `K_ms` is supported in `[-δ₁, δ₁] ⊂ (-1/2, 1/2)`,
the same hypothesis (combined with the support inclusion) implies the
period-1 MemLp on `Ioc(-1/2, 1/2)`.  The L² norms over both intervals
are equal because `K_ms` vanishes on the difference. -/

/-- `K_ms_complex = indicator (Ioc(-u/2, u/2)) K_ms_complex` everywhere on `ℝ`
(by the support inclusion). -/
lemma K_ms_complex_eq_indicator_period_u :
    (fun x => ((K_ms x : ℝ) : ℂ))
      = (Set.Ioc (-(uQ_real/2)) (uQ_real/2)).indicator (fun x => ((K_ms x : ℝ) : ℂ)) := by
  funext x
  by_cases hmem : x ∈ Set.Ioc (-(uQ_real/2)) (uQ_real/2)
  · rw [Set.indicator_of_mem hmem]
  · rw [Set.indicator_of_notMem hmem]
    have h_not_in_supp : x ∉ Function.support (fun y => ((K_ms y : ℝ) : ℂ)) := by
      intro h_in_supp
      exact hmem (K_ms_complex_support h_in_supp)
    show ((K_ms x : ℝ) : ℂ) = 0
    have : (fun y => ((K_ms y : ℝ) : ℂ)) x = 0 := by
      by_contra h_ne
      exact h_not_in_supp h_ne
    exact this

/-- AE-strong-measurability of `K_ms_complex` on full `volume`,
derived from the period-`u` restricted AE-strong-measurability + support. -/
lemma K_ms_complex_aestronglyMeasurable_volume
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    AEStronglyMeasurable (fun x => ((K_ms x : ℝ) : ℂ)) volume := by
  have h_set_meas : MeasurableSet (Set.Ioc (-(uQ_real/2)) (uQ_real/2)) :=
    measurableSet_Ioc
  -- `AEStronglyMeasurable f (volume.restrict s)` ↔ `AEStronglyMeasurable (indicator s f) volume`
  -- (for measurable `s` and `f` with `f = indicator s f`).
  have h_ind : AEStronglyMeasurable
      ((Set.Ioc (-(uQ_real/2)) (uQ_real/2)).indicator (fun x => ((K_ms x : ℝ) : ℂ))) volume :=
    (aestronglyMeasurable_indicator_iff h_set_meas).mpr hK_L2.1
  -- Rewrite using `K_ms_complex = indicator s K_ms_complex` (support fact).
  rw [K_ms_complex_eq_indicator_period_u]
  exact h_ind

/-- L²-finiteness of `K_ms_complex` on full `volume` (eLpNorm `< ⊤`),
derived from the period-`u` restricted L² + support. -/
lemma K_ms_complex_eLpNorm_volume_lt_top
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    eLpNorm (fun x => ((K_ms x : ℝ) : ℂ)) 2 volume < ⊤ := by
  have h_eq : eLpNorm (fun x => ((K_ms x : ℝ) : ℂ)) 2
                  (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))
              = eLpNorm (fun x => ((K_ms x : ℝ) : ℂ)) 2 volume :=
    eLpNorm_restrict_eq_of_support_subset K_ms_complex_support
  rw [← h_eq]
  exact hK_L2.2

/-- `MemLp K_ms_C 2 volume` from the period-`u` restricted MemLp. -/
lemma K_ms_complex_MemLp_volume
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2 volume :=
  ⟨K_ms_complex_aestronglyMeasurable_volume hK_L2,
   K_ms_complex_eLpNorm_volume_lt_top hK_L2⟩

/-- `MemLp K_ms_C 2 (volume.restrict (Ioc(-1/2, 1/2)))` from the period-`u` MemLp. -/
lemma K_ms_complex_MemLp_period1
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
      (volume.restrict (Set.Ioc (-(1/2 : ℝ)) (1/2))) :=
  (K_ms_complex_MemLp_volume hK_L2).restrict _

/-- Plancherel-at-integer-lattice for K_ms (period-1 form):
`∑'_r ‖𝓕K_ms(r)‖² = K2_analytic`.

This is the period-1 Parseval normalisation, applicable since
`K_ms` is supported in `[-δ₁, δ₁] ⊂ (-1/2, 1/2)`. -/
theorem plancherel_K_ms_K2_analytic_period1
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    ∑' r : ℤ, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      ((r : ℝ))‖ ^ 2
      = K2_analytic := by
  have hK_L2_p1 := K_ms_complex_MemLp_period1 hK_L2
  -- Apply `plancherel_at_lattice_period_u` with `u := 1`.
  have h := Sidon.TorusParseval.plancherel_at_lattice_period_u
    (1 : ℝ) one_pos (fun x => ((K_ms x : ℝ) : ℂ))
    K_ms_complex_support_period1 hK_L2_p1
  -- The LHS of `h` is `∑'_j ‖𝓕K_ms(j/1)‖² = ∑'_j ‖𝓕K_ms(j)‖²`.
  -- We need to rewrite `(j : ℝ) / 1 = (j : ℝ)`.
  have h_div_one : ∀ j : ℤ, (j : ℝ) / 1 = (j : ℝ) := fun j => by ring
  have h_fn_eq :
      (fun j : ℤ => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                            ((j : ℝ) / 1)‖ ^ 2)
        = (fun r : ℤ => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                ((r : ℝ))‖ ^ 2) := by
    funext j
    rw [h_div_one j]
  rw [h_fn_eq] at h
  -- The RHS is `1 * ∫_{(-1/2, 1/2]} ‖K_ms_C‖² = ∫_{(-1/2, 1/2]} K_ms²`.
  have h_pt : ∀ x : ℝ, ‖((K_ms x : ℝ) : ℂ)‖ ^ 2 = K_ms x ^ 2 := fun x => norm_sq_ofReal _
  have h_int_eq :
      ∫ x in (-(1/2 : ℝ))..(1/2), ‖((K_ms x : ℝ) : ℂ)‖ ^ 2
        = ∫ x in (-(1/2 : ℝ))..(1/2), K_ms x ^ 2 := by
    apply intervalIntegral.integral_congr
    intro x _
    exact h_pt x
  rw [h_int_eq] at h
  rw [K2_analytic_eq_restricted_integral_period1]
  -- `1 * S = S`.
  have h_unfold : (1 : ℝ) * ∫ x in (-(1/2 : ℝ))..(1/2), K_ms x ^ 2 =
                  ∫ x in (-(1/2 : ℝ))..(1/2), K_ms x ^ 2 := by ring
  rw [h_unfold] at h
  exact h

/-! ## Constant-term identification: `𝓕K_ms(0) = ∫K_ms` -/

/-- `Real.fourierIntegral (K_ms : ℝ → ℂ) 0 = ((∫ K_ms : ℝ) : ℂ)`. -/
theorem fourierIntegral_K_ms_zero :
    Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ)) 0
      = ((∫ x, K_ms x ∂volume : ℝ) : ℂ) := by
  rw [Sidon.FourierAux.fourierIntegral_zero (fun x => ((K_ms x : ℝ) : ℂ))]
  -- ∫ x, ((K_ms x : ℝ) : ℂ) = ((∫ K_ms : ℝ) : ℂ).
  have h := integral_ofReal (𝕜 := ℂ) (f := K_ms) (μ := volume)
  -- h : ∫ x, ((K_ms x : ℝ) : ℂ) = ↑(∫ K_ms)
  exact h

/-- The `j = 0` lattice term, assuming `∫ K_ms = 1`. -/
theorem K_ms_lattice_zero_term
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1) :
    ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                          ((0 : ℤ) / uQ_real : ℝ)‖ ^ 2 = 1 := by
  have h0 : ((0 : ℤ) / uQ_real : ℝ) = 0 := by push_cast; ring
  rw [h0]
  rw [fourierIntegral_K_ms_zero]
  rw [h_K_int_one]
  rw [norm_sq_ofReal]
  ring

/-! ## Summability of the K_ms lattice FT squared norms -/

/-- Summability of the K_ms lattice FT squared moduli, from Plancherel. -/
theorem K_ms_lattice_summable
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    Summable (fun j : ℤ => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                  (j / uQ_real : ℝ)‖ ^ 2) := by
  have h_hasSum :=
    Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
      uQ_real uQ_real_pos (fun x => ((K_ms x : ℝ) : ℂ))
      K_ms_complex_support hK_L2
  have h_summable_scaled := h_hasSum.summable
  have h_u_pos : (0 : ℝ) < uQ_real := uQ_real_pos
  -- Each scaled term equals (1/u²) times the FT squared norm.
  have h_eq_fn : (fun j : ℤ => ‖(1 / uQ_real : ℂ) *
                          Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                (j / uQ_real : ℝ)‖ ^ 2)
                      = (fun j : ℤ => (1 / uQ_real ^ 2) *
                          ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                (j / uQ_real : ℝ)‖ ^ 2) := by
    funext j
    have h_cast : ((1 / uQ_real : ℂ)) = ((1 / uQ_real : ℝ) : ℂ) := by push_cast; rfl
    rw [norm_mul, mul_pow, h_cast, Complex.norm_real, Real.norm_eq_abs]
    have h_pos : (0 : ℝ) < 1 / uQ_real := by
      apply div_pos one_pos h_u_pos
    rw [abs_of_pos h_pos]
    rw [div_pow, one_pow]
  rw [h_eq_fn] at h_summable_scaled
  -- summable (1/u² * F) ↔ summable F, since 1/u² ≠ 0.
  have h_factor_ne : (1 / uQ_real ^ 2 : ℝ) ≠ 0 := by
    apply div_ne_zero one_ne_zero
    exact pow_ne_zero 2 (ne_of_gt h_u_pos)
  exact (summable_mul_left_iff h_factor_ne).mp h_summable_scaled

/-! ## Tail sum: K-bound at the lattice -/

/-- For a summable nonneg series `F` indexed by `ℤ`,
`∑' j, F j - F 0 = ∑ j ∈ Jᶜ, F j` for any finite `J` containing `0`,
in the limit form (we use a hypothesis-friendly version).

The exact version we use: for finite `J ⊆ ℤ` with `0 ∉ J`, and `F ≥ 0` summable,
`∑ j ∈ J, F j ≤ ∑' j, F j - F 0`. -/
theorem finsum_le_tsum_minus_zero
    {F : ℤ → ℝ} (hF_nn : ∀ j, 0 ≤ F j) (hF_summable : Summable F)
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J) :
    (∑ j ∈ J, F j) ≤ (∑' j : ℤ, F j) - F 0 := by
  classical
  -- Insert 0 into J: ∑ j ∈ J, F j = ∑ j ∈ insert 0 J, F j - F 0.
  have hF0 : F 0 = F 0 := rfl
  -- Use: ∑ j ∈ J, F j + F 0 = ∑ j ∈ insert 0 J, F j ≤ ∑' j, F j.
  have h_insert : (∑ j ∈ insert (0 : ℤ) J, F j) = F 0 + ∑ j ∈ J, F j :=
    Finset.sum_insert hJ_no_zero
  -- ∑ j ∈ insert 0 J, F j ≤ ∑' j, F j (sum_le_tsum for nonneg summable).
  have h_le : (∑ j ∈ insert (0 : ℤ) J, F j) ≤ ∑' j : ℤ, F j :=
    hF_summable.sum_le_tsum (s := insert (0 : ℤ) J) (fun i _ => hF_nn i)
  linarith

/-- **Lattice form of `h_K_bound`** for `K_ms`:

For any finite `J ⊆ ℤ` with `0 ∉ J`,
`∑ j ∈ J, ‖𝓕K_ms(j/u)‖² ≤ u · K2_analytic - 1`.

Assuming `∫ K_ms = 1` and `K_ms` is in `L²` on the torus. -/
theorem K_bound_lattice
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))))
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1)
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J) :
    (∑ j ∈ J, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      (j / uQ_real : ℝ)‖ ^ 2)
      ≤ uQ_real * K2_analytic - 1 := by
  set F : ℤ → ℝ :=
    fun j => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                    (j / uQ_real : ℝ)‖ ^ 2 with hF_def
  have h_F_nn : ∀ j, 0 ≤ F j := fun j => sq_nonneg _
  have h_F_summable : Summable F := K_ms_lattice_summable hK_L2
  have h_F_zero : F 0 = 1 := K_ms_lattice_zero_term h_K_int_one
  have h_plan : ∑' j : ℤ, F j = uQ_real * K2_analytic :=
    plancherel_K_ms_K2_analytic hK_L2
  have h_le := finsum_le_tsum_minus_zero h_F_nn h_F_summable J hJ_no_zero
  rw [h_plan, h_F_zero] at h_le
  exact h_le

/-! ## Discharge of `h_K_bound` for `mv_eq2_full`

`mv_eq2_full` expects `h_K_bound : ∑ j ∈ J, Khat j ^ 2 ≤ K2 - 1`.
Setting `Khat j := Re(𝓕K_ms(j/u))` and `K2 := K2_analytic`, we have:

  `Khat j ^ 2 = (Re ·)² ≤ ‖·‖²` (since (Re z)² ≤ |z|²)

so `∑ Khat j² ≤ ∑ ‖𝓕K_ms(j/u)‖² ≤ u·K2 - 1 ≤ K2 - 1` (since `u ≤ 1`).
-/

/-- `u ≤ 1` for our `u = 638/1000`. -/
lemma uQ_real_le_one : uQ_real ≤ 1 := by
  show uQ_real ≤ 1
  unfold uQ_real uQ; push_cast; norm_num

/-- `(Re z)² ≤ ‖z‖²` for any `z : ℂ`. -/
lemma re_sq_le_norm_sq (z : ℂ) : z.re ^ 2 ≤ ‖z‖ ^ 2 := by
  -- `Complex.re_sq_le_normSq : z.re * z.re ≤ normSq z`
  -- `Complex.normSq_eq_norm_sq : normSq z = ‖z‖^2`
  have h_re := Complex.re_sq_le_normSq z
  have h_normSq : (Complex.normSq z) = ‖z‖ ^ 2 := Complex.normSq_eq_norm_sq z
  have h_sq : z.re ^ 2 = z.re * z.re := by ring
  rw [h_sq]
  linarith [h_re, h_normSq.symm ▸ h_re]

/-- **K-bound for `mv_eq2_full`** (Re-projected) for `K_ms`:

`∑ j ∈ J, (Re(𝓕K_ms(j/u)))² ≤ K2_analytic - 1`,
where the rescaling `u ≤ 1` absorbs the period-`u` factor. -/
theorem K_bound_for_mv_eq2
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))))
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1)
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J) :
    (∑ j ∈ J, (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      (j / uQ_real : ℝ)).re ^ 2)
      ≤ K2_analytic - 1 := by
  -- ∑ (Re z j)² ≤ ∑ ‖z j‖²  (Re² ≤ ‖·‖²)
  --             ≤ u · K2 - 1  (K_bound_lattice)
  --             ≤ K2 - 1  (u ≤ 1)
  have h_re_le : ∀ j ∈ J,
      (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                            (j / uQ_real : ℝ)).re ^ 2
        ≤ ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                (j / uQ_real : ℝ)‖ ^ 2 :=
    fun j _ => re_sq_le_norm_sq _
  have h_sum_re_le : (∑ j ∈ J, (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      (j / uQ_real : ℝ)).re ^ 2)
                      ≤ ∑ j ∈ J, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                        (j / uQ_real : ℝ)‖ ^ 2 :=
    Finset.sum_le_sum h_re_le
  have h_lattice : (∑ j ∈ J, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                      (j / uQ_real : ℝ)‖ ^ 2)
                    ≤ uQ_real * K2_analytic - 1 :=
    K_bound_lattice hK_L2 h_K_int_one J hJ_no_zero
  have h_K2_nn : 0 ≤ K2_analytic := by
    show 0 ≤ _
    unfold K2_analytic
    exact integral_nonneg (fun x => sq_nonneg _)
  have h_uK2_le_K2 : uQ_real * K2_analytic ≤ K2_analytic := by
    have := mul_le_mul_of_nonneg_right uQ_real_le_one h_K2_nn
    linarith
  linarith

/-! ## Hypothesis forms of the remaining atomic primitives

The two remaining `mv_eq2_full` atomic primitives (`h_F_bound` and
`h_parseval_split`) cannot be discharged from existing mathlib +
project infrastructure without bridging the bilinear period-`u`
Parseval identity (300-500 LOC) and Poisson summation on `f*f`
whose support `(-1/2, 1/2)` exceeds the torus interval `(-u/2, u/2)`.

We expose them as precise `Prop` hypotheses with the exact statement
needed to close `hEq2_schwartz` unconditionally. -/

/-! ## Period-1 K-bound chain (correct normalisation for the Schwartz path)

The autocorrelation `autocorr f` is supported in `(-1/2, 1/2)`, which
*overflows* the period-`u` interval `(-u/2, u/2)` (since `u/2 = 0.319
< 0.5`).  The Parseval split for `∫(autocorr f)·K_ms` must therefore
use **period-1** Parseval: both `autocorr f` and
`K_ms ⊆ [-δ₁, δ₁] ⊂ (-1/2, 1/2)` fit a length-1 period.  The period-1
normalisation has prefactor `1` and integer lattice frequencies
`r ∈ ℤ` (NOT the period-`u` lattice `j/u`).  The lemmas below mirror
the period-`u` chain at period `1`, where the Plancherel identity is
exact (`∑' r ‖𝓕K_ms(r)‖² = K2_analytic`, no `u ≤ 1` slack). -/

/-- Summability of the period-1 K_ms lattice FT squared moduli. -/
theorem K_ms_lattice_summable_period1
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2)))) :
    Summable (fun r : ℤ => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                  ((r : ℝ))‖ ^ 2) := by
  have hK_L2_p1 := K_ms_complex_MemLp_period1 hK_L2
  have h_hasSum :=
    Sidon.TorusParseval.plancherel_at_lattice_period_u_hasSum
      (1 : ℝ) one_pos (fun x => ((K_ms x : ℝ) : ℂ))
      K_ms_complex_support_period1 hK_L2_p1
  have h_summable_scaled := h_hasSum.summable
  have h_eq_fn : (fun j : ℤ => ‖(1 / (1 : ℝ) : ℂ) *
                          Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                (j / (1 : ℝ) : ℝ)‖ ^ 2)
                      = (fun r : ℤ =>
                          ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                                ((r : ℝ))‖ ^ 2) := by
    funext j
    have hj1 : ((j : ℝ) / (1 : ℝ)) = (j : ℝ) := by norm_num
    rw [hj1]
    norm_num
  rw [h_eq_fn] at h_summable_scaled
  exact h_summable_scaled

/-- The period-1 `r = 0` lattice term equals `1`, assuming `∫ K_ms = 1`. -/
theorem K_ms_lattice_zero_term_period1
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1) :
    ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                          (((0 : ℤ) : ℝ))‖ ^ 2 = 1 := by
  have h0 : (((0 : ℤ) : ℝ)) = (0 : ℝ) := by norm_num
  rw [h0, fourierIntegral_K_ms_zero, h_K_int_one, norm_sq_ofReal]
  ring

/-- Period-1 lattice K-bound: `∑ r ∈ J, ‖𝓕K_ms(r)‖² ≤ K2_analytic - 1`.
    Exact (no `u ≤ 1` slack), since period-1 Plancherel yields exactly
    `K2_analytic`. -/
theorem K_bound_lattice_period1
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))))
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1)
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J) :
    (∑ r ∈ J, ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      ((r : ℝ))‖ ^ 2)
      ≤ K2_analytic - 1 := by
  set F : ℤ → ℝ :=
    fun r => ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                    ((r : ℝ))‖ ^ 2 with hF_def
  have h_F_nn : ∀ r, 0 ≤ F r := fun r => sq_nonneg _
  have h_F_summable : Summable F := K_ms_lattice_summable_period1 hK_L2
  have h_F_zero : F 0 = 1 := K_ms_lattice_zero_term_period1 h_K_int_one
  have h_plan : ∑' r : ℤ, F r = K2_analytic :=
    plancherel_K_ms_K2_analytic_period1 hK_L2
  have h_le := finsum_le_tsum_minus_zero h_F_nn h_F_summable J hJ_no_zero
  rw [h_plan, h_F_zero] at h_le
  exact h_le

/-- Period-1 K-bound for `mv_eq2_full` (Re-projected):
    `∑ r ∈ J, (Re(𝓕K_ms(r)))² ≤ K2_analytic - 1`. -/
theorem K_bound_for_mv_eq2_period1
    (hK_L2 : MemLp (fun x => ((K_ms x : ℝ) : ℂ)) 2
              (volume.restrict (Set.Ioc (-(uQ_real/2)) (uQ_real/2))))
    (h_K_int_one : ∫ x, K_ms x ∂volume = 1)
    (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J) :
    (∑ r ∈ J, (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ))
                                      ((r : ℝ))).re ^ 2)
      ≤ K2_analytic - 1 := by
  have h_re_le : ∀ r ∈ J,
      (Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ)) ((r : ℝ))).re ^ 2
        ≤ ‖Real.fourierIntegral (fun x => ((K_ms x : ℝ) : ℂ)) ((r : ℝ))‖ ^ 2 :=
    fun r _ => re_sq_le_norm_sq _
  have h_sum_re_le := Finset.sum_le_sum h_re_le
  have h_lattice := K_bound_lattice_period1 hK_L2 h_K_int_one J hJ_no_zero
  linarith

end -- noncomputable section

end Sidon.BundleEq2Schwartz
