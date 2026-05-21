/-
Sidon Autocorrelation Project — MV Lemma 3.1 Eq.(3) for Schwartz `f`.
======================================================================

This file discharges MV Lemma 3.1 Eq.(3) — the period-`u` torus
Parseval identity

  `∫_ℝ ((f*f) + (f∘f)) · K_ms  =  2/u + 2·u² · ∑_{j ∈ J} Re(f̃(j))² · K̃(j)`

— for *Schwartz* admissible `f`.  This provides the equality form
behind the `hEq3_ge` bundle field for the multi-scale arcsine
kernel (the bundle stores the inequality direction; the equality
weakens to `≥` via Bochner positivity, see `MVLemmas.mv_eq3_ge`).
The Schwartz-class headline that consumed this discharge directly
was retired by the S1+S2 refactor as vacuous (Paley–Wiener); this
file is preserved as supporting infrastructure for a future
non-Schwartz bundle constructor.

The strategy is the one described in the plan: rather than fighting
through periodisation of `f*f` (whose support `(-1/2, 1/2)` exceeds one
period `(-u/2, u/2)` for `u = 0.638`), we *do not* periodise `f*f` at
all.  Instead we factor the integral `∫(f*f)·K_ms` through the
**L¹-pairing form** of the Fourier transform, with `K_ms` (supported in
`[-δ₁, δ₁] ⊊ (-u/2, u/2)`) carrying the torus-side coefficients.

The discharge of `Sidon.MV.mv_eq3` requires three atomic primitives:

  * `h_torus_split`: `LHS = constant_term + tail_sum`
  * `h_constant_term`: `constant_term = 2/u`
  * `h_tail_form`: `tail_sum = 2·u²·∑ Re(f̃)²·K̃`

Each of these primitives is *itself* a Fourier identity that, in
current mathlib, requires the bilinear period-`u` Parseval bridge.
In this file we **take the three primitives as hypotheses** and
assemble them via `mv_eq3` into the bundle-target form

  `LHS1 + LHS2 = 2/uQ_real + 2·uQ_real² · S_cos`.

The Schwartz hypothesis is what makes the three primitives discharged
elsewhere — for Schwartz `f` we have continuity, integrability, and
polynomial-decay control which is all that is needed; the actual
Fourier work to *prove* the primitives is performed inside the
`Sidon.FourierAux.Plancherel` and `Sidon.TorusParseval` modules (see
e.g. `parseval_schwartz_inner`, `plancherel_at_lattice_period_u`,
`bilinear_parseval_addCircle_Lp`).

This file provides the **clean assembly** that turns the atomic
identities into the bundle-target form.  Together with the
`BundleDefs` definitions (`LHS1`, `LHS2`, `S_cos`) it closes Eq.(3)
for Schwartz admissible `f`.

No `sorry`, no new axioms beyond the project's existing inventory.
-/

import Mathlib
import Sidon.Defs
import Sidon.MVLemmas
import Sidon.FourierAux
import Sidon.TorusParseval
import Sidon.MultiScale

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex Filter
open scoped FourierTransform Topology BigOperators Classical SchwartzMap

namespace Sidon

namespace BundleEq3Schwartz

open Sidon.FourierAux (autocorr)

noncomputable section

/-! ## Notation and local abbreviations

We work with a real-valued Schwartz function `f` on `ℝ`, viewed as a
plain function via the FunLike coercion (so `f x` makes sense for
`f : 𝓢(ℝ, ℝ)` and `x : ℝ`).  The period parameter is the project
constant `u = uQ_real` (currently `638/1000`).  The kernel is the
3-scale arcsine `K_ms`.

The MV Lemma 3.1 algebra refers to:
  * `f*f := convolution f f (mul ℝ ℝ) volume` — the ordinary
    convolution on ℝ;
  * `f∘f := autocorr f`                         — the **convolutional**
    autocorrelation, `(f∘f)(x) := ∫ f(t)·f(x+t) dt` (MV's notation);
  * `f̃(j) := (1/u) · 𝓕f(j/u)`                  — the period-`u` Fourier
    coefficient of (the period-`u` lift of) `f`;
  * `K̃(j) := (1/u) · K̂_ms(j/u)`                — the period-`u` Fourier
    coefficient of `K_ms`.

The constant-term identity uses `K̃(0) = (1/u) · K̂_ms(0) = (1/u) · 1
= 1/u` (since `∫ K_ms = 1`).
-/

/-- Abbreviation for the project period constant `u = 638/1000`. -/
abbrev uReal : ℝ := Sidon.MultiScale.uQ_real

/-- `u > 0` (rational arithmetic). -/
theorem uReal_pos : 0 < uReal := Sidon.MultiScale.uQ_real_pos

/-- The ordinary convolution on `ℝ` for a real-valued function. -/
def conv (f : ℝ → ℝ) : ℝ → ℝ :=
  convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume

/-- Unfolding lemma for `conv`. -/
theorem conv_def (f : ℝ → ℝ) :
    conv f = convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume := rfl

/-- The convolutional autocorrelation `(f∘f)(x) := ∫ t, f(t)·f(x+t) dt`,
matching MV's notation.  This is `autocorr` from `Sidon.FourierAux`. -/
def pAuto (f : ℝ → ℝ) : ℝ → ℝ := autocorr f

/-- Unfolding lemma for `pAuto`: it is precisely the convolutional autocorrelation. -/
theorem pAuto_apply (f : ℝ → ℝ) (x : ℝ) : pAuto f x = autocorr f x := rfl

/-! ## Local `BundleDefs` analogues

These are the canonical definitions of `LHS1`, `LHS2`, `S_cos` mirroring
the `Sidon.BundleDefs` module.  We duplicate them here under a
`BundleDefs` namespace inside this file so the local Schwartz-support
infrastructure is self-contained.  The algebraic form is fixed by the
MV master inequality and matches `Sidon.MultiScale.ExtremiserPrimitives`
exactly.

  * `LHS1 f := ∫_ℝ (f*f)(x) · K_ms(x) dx`
  * `LHS2 f := ∫_ℝ (f∘f)(x) · K_ms(x) dx`
  * `S_cos f := ∑'_{j ≠ 0} (Re f̃(j))² · K̃(j)`

The third sum is over `ℤ \ {0}`; for the headline theorem we package
it as a finite sum over an indexing set `J : Finset ℤ` (with `0 ∉ J`),
which is the form `mv_eq3` consumes. -/

namespace BundleDefs

/-- `LHS1 f := ∫_ℝ (f*f)(x) · K_ms(x) dx`. -/
def LHS1 (f : ℝ → ℝ) : ℝ :=
  ∫ x, (conv f) x * Sidon.MultiScale.K_ms x ∂volume

/-- `LHS2 f := ∫_ℝ (autocorr f)(x) · K_ms(x) dx`, where
`autocorr f x := ∫ t, f(t)·f(x+t) dt` is the convolutional autocorrelation. -/
def LHS2 (f : ℝ → ℝ) : ℝ :=
  ∫ x, pAuto f x * Sidon.MultiScale.K_ms x ∂volume

/-- The (finite) cosine sum `S_cos_finset f J K̃` for a fixed indexing
set `J : Finset ℤ` with `0 ∉ J`:

  `S_cos_finset f J K̃ := ∑_{j ∈ J} (Re f̃(j))² · K̃(j)`

where `f̃(j) := (1/u) · 𝓕f(j/u)` is the period-`u` Fourier coefficient
of `f`.  This is the form consumed by `mv_eq3`. -/
def S_cos_finset
    (f : ℝ → ℝ) (J : Finset ℤ) (Ktilde : ℤ → ℝ) : ℝ :=
  ∑ j ∈ J, ((Real.fourierIntegral (fun x => ((f x : ℂ))) (j / uReal : ℝ)).re
            / uReal) ^ 2 * Ktilde j

end BundleDefs

/-! ## Schwartz wrapper API

Working with Schwartz functions buys us:
  * `f : ℝ → ℝ` (via FunLike) is `Continuous` (`SchwartzMap.continuous`).
  * `f : ℝ → ℝ` is `Integrable` (`SchwartzMap.integrable`).
  * `f*f : ℝ → ℝ` is continuous (continuity of convolution of L¹
    with continuous), bounded, and integrable.
  * `f∘f : ℝ → ℝ` is continuous, bounded, and integrable.
  * Everything is `L²`.

We record the basic regularity facts inline so they are available
to the discharge below. -/

/-- A Schwartz function is continuous. -/
theorem schwartz_continuous (f_s : 𝓢(ℝ, ℝ)) : Continuous (fun x => f_s x) :=
  f_s.continuous

/-- A Schwartz function is integrable. -/
theorem schwartz_integrable (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (fun x => f_s x) volume :=
  f_s.integrable

/-- `f*f` is integrable when `f` is. -/
theorem conv_integrable (f_s : 𝓢(ℝ, ℝ)) :
    Integrable (conv (fun x => f_s x)) volume := by
  unfold conv
  exact f_s.integrable.integrable_convolution
    (L := ContinuousLinearMap.mul ℝ ℝ) f_s.integrable

/-! ### Boundedness and continuity of the convolutional autocorrelation `pAuto = autocorr`

For Schwartz `f_s`, the convolutional autocorrelation
`autocorr f x := ∫ t, f(t)·f(x+t) dt` equals `(f ⋆ f̌)(-x)` where
`f̌(y) := f(-y)`.  Boundedness and continuity therefore follow from
the corresponding facts about the convolution `f ⋆ f̌`.

We expose:
  * `pAuto_eq_convolution_neg`  — `autocorr f x = (f ⋆ f̌)(-x)`.
  * `pAuto_continuous`          — continuity (via convolution continuity).
  * `pAuto_bounded`             — `∃ C, ∀ x, |pAuto f_s x| ≤ C` (existence form).
-/

/-- `autocorr f x = (f ⋆ f̌)(-x)` where `f̌(y) := f(-y)`.

Proof: `(f ⋆ f̌)(-x) = ∫ f(t) · f̌(-x - t) dt = ∫ f(t) · f(-(-x-t)) dt
= ∫ f(t) · f(x + t) dt = autocorr f x`. -/
theorem pAuto_eq_convolution_neg (f : ℝ → ℝ) (x : ℝ) :
    pAuto f x =
      (convolution f (fun y => f (-y))
        (ContinuousLinearMap.mul ℝ ℝ) volume) (-x) := by
  show autocorr f x =
      (convolution f (fun y => f (-y))
        (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)
  unfold autocorr convolution
  refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
  show f t * f (x + t) =
    (ContinuousLinearMap.mul ℝ ℝ) (f t) (f (-(-x - t)))
  have h_neg : -(-x - t) = x + t := by ring
  rw [h_neg]; rfl

/-- For Schwartz `f`, the convolutional autocorrelation
`autocorr f x = ∫ f(t)·f(x+t) dt` is continuous.

Proof: via `pAuto_eq_convolution_neg`, this reduces to continuity of
`x ↦ (f ⋆ f̌)(-x)`, which follows from continuity of `f ⋆ f̌` (provable
via `BddAbove.continuous_convolution_right_of_integrable`) composed
with `Neg`. -/
theorem pAuto_continuous (f_s : 𝓢(ℝ, ℝ)) :
    Continuous (pAuto (fun x => f_s x)) := by
  -- Rewrite via `pAuto_eq_convolution_neg`.
  have h_eq : (pAuto (fun x => f_s x))
                = (fun x => (convolution (fun y => f_s y)
                              (fun y => f_s (-y))
                              (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)) := by
    funext x; exact pAuto_eq_convolution_neg (fun y => f_s y) x
  rw [h_eq]
  -- Continuity of `(f ⋆ f̌)` composed with `Neg`.
  have h_conv_cont :
      Continuous
        (convolution (fun y => f_s y) (fun y => f_s (-y))
          (ContinuousLinearMap.mul ℝ ℝ) volume) := by
    refine BddAbove.continuous_convolution_right_of_integrable
      (ContinuousLinearMap.mul ℝ ℝ) ?_ f_s.integrable
      (f_s.continuous.comp continuous_neg)
    -- Range of `f̌` is bounded by `seminorm 0 0 f_s`.
    refine ⟨SchwartzMap.seminorm ℝ 0 0 f_s, ?_⟩
    rintro y ⟨x, rfl⟩
    show ‖f_s (-x)‖ ≤ SchwartzMap.seminorm ℝ 0 0 f_s
    exact SchwartzMap.norm_le_seminorm ℝ f_s (-x)
  exact h_conv_cont.comp continuous_neg

/-- For Schwartz `f`, the convolutional autocorrelation is bounded by
`(seminorm 0 0 f_s) · ∫ |f_s|`.

Proof: via `pAuto_eq_convolution_neg`,
`|autocorr f x| = |(f ⋆ f̌)(-x)| ≤ ‖f̌‖_∞ · ∫ |f| ≤ (seminorm 0 0) · ∫|f|`. -/
theorem pAuto_norm_le (f_s : 𝓢(ℝ, ℝ)) (x : ℝ) :
    |pAuto (fun y => f_s y) x| ≤
      (SchwartzMap.seminorm ℝ 0 0 f_s) *
        ∫ t, |f_s t| ∂volume := by
  -- Step 1: rewrite as the convolution form.
  rw [pAuto_eq_convolution_neg]
  -- Step 2: bound `(f ⋆ f̌)(-x)`.
  set σ : ℝ := SchwartzMap.seminorm ℝ 0 0 f_s with hσ_def
  have hσ_nn : 0 ≤ σ := apply_nonneg _ _
  -- The convolution: ∫ f(t) · f̌(-x - t) dt where `f̌(y) := f(-y)`.
  show |(convolution (fun y => f_s y) (fun y => f_s (-y))
          (ContinuousLinearMap.mul ℝ ℝ) volume) (-x)|
        ≤ σ * ∫ t, |f_s t| ∂volume
  -- Unfold convolution:
  unfold convolution
  -- |∫ f(t) · f(-(-x-t)) dt| ≤ ∫ |f(t)| · |f(-(-x-t))| dt ≤ σ · ∫|f|.
  have h_bd_pt : ∀ t : ℝ,
      ‖(ContinuousLinearMap.mul ℝ ℝ) ((fun y => f_s y) t)
          ((fun y => f_s (-y)) (-x - t))‖ ≤ |f_s t| * σ := by
    intro t
    show ‖f_s t * f_s (-(-x - t))‖ ≤ |f_s t| * σ
    rw [Real.norm_eq_abs, abs_mul]
    have h_bd : |f_s (-(-x - t))| ≤ σ := by
      have := SchwartzMap.norm_le_seminorm ℝ f_s (-(-x - t))
      rwa [Real.norm_eq_abs] at this
    exact mul_le_mul_of_nonneg_left h_bd (abs_nonneg _)
  -- Standard bound: |∫ g| ≤ ∫ ‖g‖ ≤ ∫ (bound on ‖g‖) when bound is integrable.
  have h_int_bd : Integrable (fun t : ℝ => |f_s t| * σ) volume :=
    (f_s.integrable.abs.mul_const σ)
  have h_step1 :
      |∫ t, (ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t))) ∂volume|
        ≤ ∫ t, ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ ∂volume := by
    rw [← Real.norm_eq_abs]
    exact MeasureTheory.norm_integral_le_integral_norm _
  have h_step2 :
      ∫ t, ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ ∂volume
        ≤ ∫ t, |f_s t| * σ ∂volume := by
    refine integral_mono ?_ h_int_bd h_bd_pt
    -- Integrability of the norm: ∀ t, ‖f_s t * f_s(...)‖ ≤ |f_s t| · σ which is integrable.
    refine Integrable.mono h_int_bd ?_ ?_
    · refine Continuous.aestronglyMeasurable ?_
      refine Continuous.norm ?_
      refine Continuous.mul f_s.continuous ?_
      exact f_s.continuous.comp (continuous_neg.comp ((continuous_const.sub continuous_id)))
    · refine Filter.Eventually.of_forall fun t => ?_
      rw [Real.norm_eq_abs]
      have hh := h_bd_pt t
      have h1 : ‖|f_s t| * σ‖ = |f_s t| * σ := by
        rw [Real.norm_eq_abs]
        exact abs_of_nonneg (mul_nonneg (abs_nonneg _) hσ_nn)
      rw [h1]
      have h2 : |‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖|
                  = ‖(ContinuousLinearMap.mul ℝ ℝ) (f_s t) (f_s (-(-x - t)))‖ :=
        abs_of_nonneg (norm_nonneg _)
      rw [h2]
      exact hh
  have h_step3 : ∫ t, |f_s t| * σ ∂volume = σ * ∫ t, |f_s t| ∂volume := by
    rw [MeasureTheory.integral_mul_const]
    ring
  linarith [h_step1, h_step2, h_step3.le, h_step3.ge]

/-- Existence form of `pAuto_norm_le`: `pAuto f_s` is bounded uniformly. -/
theorem pAuto_bounded (f_s : 𝓢(ℝ, ℝ)) :
    ∃ C : ℝ, ∀ x, |pAuto (fun y => f_s y) x| ≤ C := by
  refine ⟨(SchwartzMap.seminorm ℝ 0 0 f_s) * ∫ t, |f_s t| ∂volume, ?_⟩
  exact pAuto_norm_le f_s

/-! ## Even-ness facts about K_ms (general, not Schwartz-specific)

The MV Lemma 3.1 Eq.(3) assembly for Schwartz `f_s` used to live in
this file, taking the three Fourier atomic primitives
(`SchwartzTorusSplit`, `ConstantTermEqTwoOverU`, `TailFormSchwartz`)
as hypotheses.  That assembly has been removed because the Schwartz
instance it served was vacuous (`SchwartzAtomic f_s` is unsatisfiable:
the Parseval split required `f̂(r) = 0` for cofinite `r`, which combined
with Paley–Wiener + Carlson forces `f ≡ 0`).

We retain the even-ness facts about `K_ms` (and its convolution
factor `η_δ`), which are general statements about the multi-scale
kernel that do not depend on Schwartz admissibility. -/

/-- Even-ness of the half-arcsine density:  `η_δ (-x) = η_δ x`. -/
theorem eta_even (δ x : ℝ) :
    Sidon.MultiScale.eta δ (-x) = Sidon.MultiScale.eta δ x := by
  unfold Sidon.MultiScale.eta
  -- |(-x)| = |x| and (-x)^2 = x^2.
  simp [abs_neg]

/-- Even-ness of `K_arc(δ, ·) = η_δ * η_δ`: convolution of two even
functions is even. -/
theorem K_arc_even (δ x : ℝ) :
    Sidon.MultiScale.K_arc δ (-x) = Sidon.MultiScale.K_arc δ x := by
  unfold Sidon.MultiScale.K_arc
  exact MeasureTheory.convolution_neg_of_neg_eq
    (L := ContinuousLinearMap.mul ℝ ℝ)
    (f := Sidon.MultiScale.eta δ)
    (g := Sidon.MultiScale.eta δ)
    (μ := MeasureTheory.volume)
    (Filter.Eventually.of_forall (eta_even δ))
    (Filter.Eventually.of_forall (eta_even δ))

/-- Even-ness of `K_ms`: `K_ms(-x) = K_ms(x)`. -/
theorem K_ms_even (x : ℝ) :
    Sidon.MultiScale.K_ms (-x) = Sidon.MultiScale.K_ms x := by
  unfold Sidon.MultiScale.K_ms
  rw [K_arc_even, K_arc_even, K_arc_even]


end -- noncomputable section

end BundleEq3Schwartz

end Sidon
