/-
Sidon Autocorrelation Project — Convolution Fourier Identity (Lemma N1).
=======================================================================

This file proves the convolution Fourier theorem in the two forms the
downstream MV machinery consumes:

  * **(A)** `𝓕(f ⋆ f)(ξ) = (𝓕 f ξ)²`           (autoconvolution),
  * **(B)** `𝓕(autocorr f)(ξ) = 𝓕f(ξ) · conj(𝓕f(ξ)) = |𝓕f(ξ)|²`
            (Wiener–Khinchin / autocorrelation).

Here `f : ℝ → ℝ` is integrable, and for (B) the joint product
`(x,t) ↦ f t · f (x+t)` is integrable on `ℝ²` (which holds e.g. for
`f ∈ L¹` with compact support).

**Relation to mathlib.**  Mathlib `v4.29.1` already proves the
multiplicative convolution theorem `Real.fourier_mul_convolution_eq`
(`Mathlib/Analysis/Fourier/Convolution.lean:149`), but with hypotheses
`Integrable f ∧ Continuous f` for *both* factors.  Our extremiser `f`
is only `L¹ ∩ L²` with compact support — it need NOT be continuous.
We therefore reprove the underlying identity from Fubini, requiring only
`Integrable f` (continuity is not needed; `Integrable.convolution_integrand`
supplies the joint `L¹(ℝ²)` bound that mathlib's proof obtained from
continuity via `integrable_prod_sub`).

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.FourierAux

namespace Sidon.Constructor

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Convolution

set_option maxHeartbeats 4000000

noncomputable section

/-! ## The general `L¹` multiplicative convolution theorem (no continuity) -/

/-- The character factor as a complex number has modulus `1`. -/
private lemma norm_fourierChar_mul (a : ℝ) : ‖(𝐞 (-a) : ℂ)‖ = 1 := by
  rw [Real.fourierChar_apply, Complex.norm_exp]
  simp

/-- The complex autoconvolution integrand `(p : ℝ×ℝ) ↦ f(p.2)·g(p.1 - p.2)`
is integrable on `ℝ²` for integrable `f g`.  (Specialisation of
`Integrable.convolution_integrand` to `B = mul ℂ ℂ`; the variable order
matches `Function.uncurry (fun x t => f t * g (x - t))`.) -/
private lemma integrable_mul_sub
    (f g : ℝ → ℂ) (hf : Integrable f volume) (hg : Integrable g volume) :
    Integrable (fun p : ℝ × ℝ => f p.2 * g (p.1 - p.2)) (volume.prod volume) := by
  have h := hf.convolution_integrand (ContinuousLinearMap.mul ℂ ℂ) hg
  simpa [ContinuousLinearMap.mul_apply'] using h

/-- **Convolution Fourier theorem, `L¹` version (complex, no continuity).**

For integrable `f g : ℝ → ℂ`,
`𝓕(f ⋆[mul ℂ ℂ] g)(ξ) = 𝓕 f ξ · 𝓕 g ξ`.

This strengthens mathlib's `Real.fourier_mul_convolution_eq` by dropping
the `Continuous` hypotheses. -/
theorem fourierIntegral_convolution_mul
    (f g : ℝ → ℂ) (hf : Integrable f volume) (hg : Integrable g volume) (ξ : ℝ) :
    Real.fourierIntegral
        (fun x => (MeasureTheory.convolution f g (ContinuousLinearMap.mul ℂ ℂ) volume) x) ξ
      = Real.fourierIntegral f ξ * Real.fourierIntegral g ξ := by
  -- Damped integrand in the order `(p.1 = x, p.2 = t)`, integrable on `ℝ²`.
  -- (Norm equals that of the undamped integrand `integrable_mul_sub`, since `‖𝐞‖ = 1`.)
  have hdamp : Integrable
      (fun p : ℝ × ℝ => (𝐞 (-(p.1 * ξ)) : ℂ) * (f p.2 * g (p.1 - p.2)))
        (volume.prod volume) := by
    refine (integrable_mul_sub f g hf hg).norm.mono' ?_ ?_
    · refine AEStronglyMeasurable.mul ?_ (integrable_mul_sub f g hf hg).aestronglyMeasurable
      have hc : Continuous (fun p : ℝ × ℝ => (𝐞 (-(p.1 * ξ)) : ℂ)) :=
        continuous_subtype_val.comp (Real.continuous_fourierChar.comp (by continuity))
      exact hc.aestronglyMeasurable
    · filter_upwards with p
      rw [norm_mul, norm_fourierChar_mul, one_mul]
  -- Helper: unfold `𝓕 h ξ = ∫ v, (𝐞(-(v·ξ)):ℂ) * h v`.
  have hFT : ∀ (h : ℝ → ℂ) (w : ℝ),
      Real.fourierIntegral h w = ∫ v, (𝐞 (-(v * w)) : ℂ) * h v := by
    intro h w
    show FourierTransform.fourier h w = _
    rw [Real.fourier_real_eq]
    simp_rw [Circle.smul_def, smul_eq_mul]
  calc
    Real.fourierIntegral
        (fun x => (MeasureTheory.convolution f g (ContinuousLinearMap.mul ℂ ℂ) volume) x) ξ
        = ∫ x, (𝐞 (-(x * ξ)) : ℂ) * ∫ t, f t * g (x - t) := by
          rw [hFT]
          refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
          simp [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
    _ = ∫ x, ∫ t, (𝐞 (-(x * ξ)) : ℂ) * (f t * g (x - t)) := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
          simp only []
          exact (integral_const_mul (𝐞 (-(x * ξ)) : ℂ) (fun t => f t * g (x - t))).symm
    _ = ∫ t, ∫ x, (𝐞 (-(x * ξ)) : ℂ) * (f t * g (x - t)) := by
          exact MeasureTheory.integral_integral_swap hdamp
    _ = ∫ t, f t * ∫ x, (𝐞 (-(x * ξ)) : ℂ) * g (x - t) := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
          simp only []
          rw [show (fun x => (𝐞 (-(x * ξ)) : ℂ) * (f t * g (x - t)))
                = (fun x => f t * ((𝐞 (-(x * ξ)) : ℂ) * g (x - t))) from by
                  funext x; ring]
          exact integral_const_mul (f t) (fun x => (𝐞 (-(x * ξ)) : ℂ) * g (x - t))
    _ = ∫ t, f t * ((𝐞 (-(t * ξ)) : ℂ) * Real.fourierIntegral g ξ) := by
          refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
          simp only []
          congr 1
          -- substitute `x = s + t`:  ∫_x 𝐞(-(x·ξ)) g(x-t) dx = 𝐞(-(t·ξ)) ∫_s 𝐞(-(s·ξ)) g(s) ds
          rw [← MeasureTheory.integral_add_right_eq_self
                (fun x => (𝐞 (-(x * ξ)) : ℂ) * g (x - t)) t]
          rw [hFT g ξ]
          rw [show (𝐞 (-(t * ξ)) : ℂ) * ∫ v, (𝐞 (-(v * ξ)) : ℂ) * g v
                = ∫ v, (𝐞 (-(t * ξ)) : ℂ) * ((𝐞 (-(v * ξ)) : ℂ) * g v) from
              (integral_const_mul (𝐞 (-(t * ξ)) : ℂ)
                (fun v => (𝐞 (-(v * ξ)) : ℂ) * g v)).symm]
          refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
          have hadd : (𝐞 (-((s + t) * ξ)) : ℂ) = (𝐞 (-(t * ξ)) : ℂ) * (𝐞 (-(s * ξ)) : ℂ) := by
            rw [← Circle.coe_mul, ← AddChar.map_add_eq_mul]
            congr 2
            ring
          have hsub : s + t - t = s := by ring
          simp only [hsub]
          rw [hadd]
          ring
    _ = Real.fourierIntegral f ξ * Real.fourierIntegral g ξ := by
          rw [hFT f ξ]
          rw [show (∫ v, (𝐞 (-(v * ξ)) : ℂ) * f v) * Real.fourierIntegral g ξ
                = ∫ v, ((𝐞 (-(v * ξ)) : ℂ) * f v) * Real.fourierIntegral g ξ from
              (integral_mul_const (Real.fourierIntegral g ξ)
                (fun v => (𝐞 (-(v * ξ)) : ℂ) * f v)).symm]
          refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
          ring

/-! ## Target (A): autoconvolution Fourier identity for real `f`

For real `f : ℝ → ℝ` integrable, the Fourier transform of the
(`ℝ`-bilinear) autoconvolution, cast to `ℂ`, equals the square of `𝓕 f`. -/

/-- The complex cast of the real autoconvolution equals the complex
autoconvolution of the cast: for `f : ℝ → ℝ`,
`((f ⋆[mul ℝ ℝ] f) x : ℂ) = ((↑f) ⋆[mul ℂ ℂ] (↑f)) x`. -/
private lemma ofReal_convolution_eq (f : ℝ → ℝ) (x : ℝ) :
    ((MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x : ℂ)
      = (MeasureTheory.convolution (fun y => (f y : ℂ)) (fun y => (f y : ℂ))
          (ContinuousLinearMap.mul ℂ ℂ) volume) x := by
  simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
  rw [show (fun t => (f t : ℂ) * (f (x - t) : ℂ))
        = (fun t => ((f t * f (x - t) : ℝ) : ℂ)) from by funext t; push_cast; ring]
  exact (integral_ofReal (f := fun t => f t * f (x - t))).symm

/-- **Target (A): autoconvolution Fourier identity (real `f`).**

For `f : ℝ → ℝ` integrable (in particular for `f ∈ L¹ ∩ L²` with compact
support `⊆ (-1/4, 1/4)`),
`𝓕(f ⋆[mul ℝ ℝ] f)(ξ) = (𝓕 (↑f) ξ)²`. -/
theorem fourierIntegral_autoconvolution_real
    (f : ℝ → ℝ) (hf : Integrable f volume) (ξ : ℝ) :
    Real.fourierIntegral
        (fun x => ((MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x : ℂ)) ξ
      = (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) ^ 2 := by
  -- Cast the real convolution to the complex one, then apply the general lemma.
  have hfℂ : Integrable (fun y => (f y : ℂ)) volume := hf.ofReal
  rw [show (fun x => ((MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x : ℂ))
        = (fun x => (MeasureTheory.convolution (fun y => (f y : ℂ)) (fun y => (f y : ℂ))
            (ContinuousLinearMap.mul ℂ ℂ) volume) x) from
      funext (ofReal_convolution_eq f)]
  rw [fourierIntegral_convolution_mul (fun y => (f y : ℂ)) (fun y => (f y : ℂ)) hfℂ hfℂ ξ]
  ring

/-! ## Target (B): autocorrelation Fourier identity (Wiener–Khinchin)

For real `f : ℝ → ℝ` integrable, `𝓕(autocorr f)(ξ) = 𝓕f(ξ)·conj(𝓕f(ξ))`,
i.e. the squared modulus `|𝓕f(ξ)|²`.  The autocorrelation
`autocorr f x = ∫ t, f t · f (x + t)` is the convolution of the reversal
`f̌ y := f (-y)` with `f` (substitute `s = -t`), and `𝓕 f̌ = conj(𝓕 f)`
for real `f` (`Sidon.FourierAux.fourierIntegral_real_conj`). -/

/-- The reversal of a real function, lifted to `ℂ`: `y ↦ (f (-y) : ℂ)`. -/
private def reflectC (f : ℝ → ℝ) : ℝ → ℂ := fun y => (f (-y) : ℂ)

/-- `autocorr f x = (f̌ ⋆[mul ℂ ℂ] f) x` after casting to `ℂ`, where
`f̌ y = f (-y)`.  (Change of variables `s = -t` in the defining integral.) -/
private lemma ofReal_autocorr_eq_convolution (f : ℝ → ℝ) (x : ℝ) :
    ((Sidon.FourierAux.autocorr f x : ℝ) : ℂ)
      = (MeasureTheory.convolution (reflectC f) (fun y => (f y : ℂ))
          (ContinuousLinearMap.mul ℂ ℂ) volume) x := by
  rw [Sidon.FourierAux.autocorr_def]
  simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply', reflectC]
  -- RHS: substitute `t = -s` (reflection invariance), then recognise the real cast.
  rw [← MeasureTheory.integral_neg_eq_self
        (fun t => (f (-t) : ℂ) * (f (x - t) : ℂ)) volume]
  simp only [neg_neg]
  rw [show (fun t => (f t : ℂ) * (f (x - -t) : ℂ))
        = (fun t => ((f t * f (x + t) : ℝ) : ℂ)) from by
        funext t
        have : x - -t = x + t := by ring
        rw [this]; push_cast; ring]
  exact (integral_ofReal (f := fun t => f t * f (x + t))).symm

/-- **Target (B): autocorrelation Fourier identity (real `f`).**

For `f : ℝ → ℝ` integrable (in particular for `f ∈ L¹ ∩ L²` with compact
support `⊆ (-1/4, 1/4)`),
`𝓕(autocorr f)(ξ) = 𝓕(↑f) ξ · conj(𝓕(↑f) ξ)`  (`= |𝓕f(ξ)|²`). -/
theorem fourierIntegral_autocorr_real
    (f : ℝ → ℝ) (hf : Integrable f volume) (ξ : ℝ) :
    Real.fourierIntegral (fun x => ((Sidon.FourierAux.autocorr f x : ℝ) : ℂ)) ξ
      = Real.fourierIntegral (fun x => (f x : ℂ)) ξ *
          starRingEnd ℂ (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) := by
  have hfℂ : Integrable (fun y => (f y : ℂ)) volume := hf.ofReal
  -- The reflection `f̌` is integrable (reflection preserves the `L¹` norm).
  have hreflect : Integrable (reflectC f) volume := by
    have := hfℂ.comp_neg
    simpa [reflectC, Function.comp] using this
  -- Rewrite `autocorr` as the convolution `f̌ ⋆ f`, then apply the general FT lemma.
  rw [show (fun x => ((Sidon.FourierAux.autocorr f x : ℝ) : ℂ))
        = (fun x => (MeasureTheory.convolution (reflectC f) (fun y => (f y : ℂ))
            (ContinuousLinearMap.mul ℂ ℂ) volume) x) from
      funext (ofReal_autocorr_eq_convolution f)]
  rw [fourierIntegral_convolution_mul (reflectC f) (fun y => (f y : ℂ)) hreflect hfℂ ξ]
  -- `𝓕 f̌ ξ = 𝓕 f (-ξ) = conj (𝓕 f ξ)`; align factors by commutativity.
  have hrefl_ft : Real.fourierIntegral (reflectC f) ξ
      = Real.fourierIntegral (fun x => (f x : ℂ)) (-ξ) := by
    show FourierTransform.fourier (reflectC f) ξ
        = FourierTransform.fourier (fun x => (f x : ℂ)) (-ξ)
    rw [Real.fourier_real_eq, Real.fourier_real_eq]
    -- Reverse-substitute `v ↦ -v` on the RHS, then match integrands.
    rw [show (∫ v, (𝐞 (-(v * -ξ)) : Circle) • ((f v : ℂ)))
          = ∫ v, (𝐞 (-((-v) * -ξ)) : Circle) • ((f (-v) : ℂ)) from
        (MeasureTheory.integral_neg_eq_self
          (fun v => (𝐞 (-(v * -ξ)) : Circle) • ((f v : ℂ))) volume).symm]
    refine integral_congr_ae (Filter.Eventually.of_forall fun v => ?_)
    simp only [reflectC]
    congr 2
    ring
  rw [hrefl_ft, Sidon.FourierAux.fourierIntegral_real_conj f hfℂ ξ]
  rw [mul_comm]

end

end Sidon.Constructor
