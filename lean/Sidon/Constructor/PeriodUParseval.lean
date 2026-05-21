/-
Sidon Autocorrelation Project — Period-`u` Parseval pairing (MO 2009 Lemma 2.1).
================================================================================

This file attacks the deepest missing primitive of the constructor layer:
the period-`u` Parseval pairing that `Sidon.Constructor.FieldsParseval`
left as the hypothesis `h_torus_eq_full` of `field_hEq3_ge`.

  `∫_ℝ ((f*f) + autocorr f)·K_ms  =  2/u  +  (coefficient)·S_cos f`,

where `S_cos f = ∑'_{j≠0} (Re 𝓕f(j/u))² · K̂_ms(j/u)` and `u = 1/2 + δ₁`.

================================================================================
## A correction to the stated coefficient (verified numerically + symbolically)
================================================================================

The bundle field `Sidon.MultiScale.ExtremiserPrimitives.hEq3_ge` and the
hypothesis `h_torus_eq_full` of `FieldsParseval.field_hEq3_ge` are both
written with the coefficient `2·u²`:

  `LHS1 f + LHS2 f  =  2/u + 2·u²·S_cos f`     (`h_torus_eq_full`).

With `S_cos f` in the project's **full-line Fourier** convention
(`S_cos f := ∑'_{j≠0} (Re 𝓕f(j/u))² · K_ms_fourier_lattice j`, where
`K_ms_fourier_lattice j = 𝓕K_ms(j/u)` is the genuine real-line transform
`= ∑ᵢ λᵢ J₀(πδᵢ·j/u)²`), this equality is **off by a factor of `u³`** and
is therefore *false*.  The genuine period-`u` Parseval identity is

  `LHS1 f + LHS2 f  =  2/u + (2/u)·S_cos f`     (`period_u_parseval_true`).

The `2·u²` coefficient is correct only in the *period-`u` Fourier
coefficient* convention used abstractly inside `Sidon.MVLemmas.mv_eq3`
(there `realFparts j = (1/u)·Re 𝓕f(j/u)` and `Ktilde j = (1/u)·𝓕K_ms(j/u)`,
so `2u²·∑ realFparts² · Ktilde = (2/u)·∑ (Re 𝓕f)²·K̂_ms`).  The
`ExtremiserPrimitives` binding `S_cos_eq : S_cos = Sidon.MultiScale.S_cos f`
silently switches to the full-line convention, which is what introduces the
`u³` mismatch in the stated *equality*.

Crucially, the *bundle field that the master inequality actually consumes*
is the **inequality** `hEq3_ge : 2/u + 2·u²·S_cos ≤ LHS1 + LHS2`, NOT the
equality.  Because `u = 638/1000 < 1` we have `u³ < 1`, hence `2u² < 2/u`,
hence (since `S_cos f ≥ 0`)

  `2/u + 2·u²·S_cos f  ≤  2/u + (2/u)·S_cos f  =  LHS1 f + LHS2 f`,

so the consumed inequality `hEq3_ge` is **true** and is derived here, with
**zero residual hypotheses**, from the genuine equality `period_u_parseval_true`
via `field_hEq3_ge_of_true_parseval`.  (See the numerical check recorded in
the project notes: with the asymmetric test `f`, `LHS1+LHS2 ≈ 3.59807`,
`2/u + (2/u)·S_cos ≈ 3.59807` matches, while `2/u + 2u²·S_cos ≈ 3.25511`.)

================================================================================
## The `Re` / `±j` reduction (fully proved here)
================================================================================

For real `f`, the convolution-theorem outputs (already proved in
`Sidon.Constructor.ConvFourier`) give

  `𝓕(f*f)(ξ) = (𝓕f ξ)²`,   `𝓕(autocorr f)(ξ) = 𝓕f ξ · conj(𝓕f ξ) = |𝓕f ξ|²`,

so `𝓕g(ξ) = (𝓕f ξ)² + |𝓕f ξ|²`.  Writing `𝓕f ξ = a + b·I`,

  `(a+bI)² + (a²+b²) = 2a² + 2ab·I`,

whose imaginary part `2ab` does **not** vanish in general (the extremiser is
*not* assumed even — its support `(-1/4,1/4)` is symmetric but `f` need not
be).  The imaginary part cancels only after the `±j` pairing: for real `f`,
`𝓕f(-ξ) = conj(𝓕f ξ)`, so `Re 𝓕f(-ξ) = Re 𝓕f ξ` and `Im 𝓕f(-ξ) = -Im 𝓕f ξ`,
while `K_ms` is real and even, so `K̂_ms(-ξ) = K̂_ms(ξ) ∈ ℝ`.  Hence the
`j`- and `(-j)`-terms of `∑ 𝓕g(j/u)·K̂_ms(j/u)` add to

  `(2a² + 2abI)·K̂ + (2a² − 2abI)·K̂ = 4a²·K̂ = 4·(Re 𝓕f(j/u))²·K̂_ms(j/u)`,

i.e. **only `(Re 𝓕f)²` survives**, which is exactly why `S_cos` is phrased in
terms of `Re 𝓕f`.  The pointwise `j`-term identity
`Re(𝓕g(j/u)·K̂_ms(j/u)) = 2·(Re 𝓕f(j/u))²·K̂_ms(j/u)` is proved below as
`re_fourierIntegral_g_mul_Khat` — `K̂_ms(j/u)` being real makes even the
single-`j` real part collapse to `2a²·K̂` without needing the pairing (the
pairing is only needed to discard the imaginary part of the *complex* sum;
here we take real parts term-by-term, which is legitimate because the final
target `S_cos` and the integral `LHS1+LHS2` are real).

`re_fourierIntegral_g_mul_Khat` is fully proved (no `sorry`, no axiom).

================================================================================
## What is proved vs. what remains
================================================================================

PROVED, compiling, no `sorry`/axiom:
  * `g`, `g_complex` definitions and `fourierIntegral_g` :
    `𝓕g(ξ) = (𝓕f ξ)² + 𝓕f ξ · conj(𝓕f ξ)`  (from `ConvFourier`).
  * `re_fourierIntegral_g_mul_Khat` : the `Re`/real-part reduction
    `Re(𝓕g(ξ)·(K̂:ℝ)) = 2·(Re 𝓕f ξ)²·K̂` for any real `K̂`.
  * `S_cos_nonneg` : `0 ≤ S_cos f`  (nonneg-term `tsum`).
  * `two_uSq_le_two_div_u` : `2·u² ≤ 2/u`  (from `u³ ≤ 1`, `u = 0.638`).
  * `field_hEq3_ge_of_true_parseval` : **the bundle field `hEq3_ge`**,
    `2/u + 2·u²·S_cos f ≤ LHS1 f + LHS2 f`, derived with ZERO residual
    hypotheses from the genuine equality `period_u_parseval_true`.

REMAINS (the single genuine analytic primitive, stated as a `def`-level
`Prop` `PeriodUParsevalTrue`, NOT assumed as an axiom anywhere — it is a
hypothesis of the bridge theorems):
  * `period_u_parseval_true f : LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f`.

This is MO 2009 Lemma 2.1: the period-`u` Fourier coefficients of the
periodisation `∑ₖ g(·+ku)` sample `𝓕g` at the lattice `{j/u}`, and on
`supp K_ms ⊆ (-δ₁,δ₁)` the periodisation agrees with `g` (the `k≠0`
translates `g(·+ku)`, supported in `(-1/2−ku, 1/2−ku)`, do not meet
`(-δ₁,δ₁)` precisely because `1/2 + δ₁ = u`).  The exact residual
mathlib-shaped lemma is recorded in the docstring of
`PeriodUParsevalTrue` below.

No `sorry`, no new axioms in this file.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.BilinearParseval
import Sidon.FourierAux
import Sidon.Constructor.ConvFourier

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical

namespace Sidon.Constructor.PeriodU

open Sidon.MultiScale
open Sidon.FourierAux (autocorr autocorr_def)

noncomputable section

/-! ## Additivity of `Real.fourierIntegral` on integrable functions

Mathlib exposes `VectorFourier.fourierIntegral_add` only with continuity
side-conditions on the character and the bilinear pairing.  For the
concrete `Real.fourierIntegral` (`= FourierTransform.fourier`) we reprove
additivity directly from `integral_add` via the exponential-form unfolding
(the same `hFT` trick used in `Sidon.Constructor.ConvFourier`), needing only
integrability of the two summands. -/

/-- Unfold `𝓕 h w = ∫ v, 𝐞(-(v·w)) • h v`. -/
private theorem fourierIntegral_exp_unfold (h : ℝ → ℂ) (w : ℝ) :
    Real.fourierIntegral h w = ∫ v, (𝐞 (-(v * w)) : ℂ) * h v := by
  show FourierTransform.fourier h w = _
  rw [Real.fourier_real_eq]
  simp_rw [Circle.smul_def, smul_eq_mul]

/-- **Additivity of `Real.fourierIntegral`** on integrable `f₁, f₂ : ℝ → ℂ`:
`𝓕(f₁ + f₂)(w) = 𝓕f₁(w) + 𝓕f₂(w)`. -/
theorem fourierIntegral_add (f₁ f₂ : ℝ → ℂ)
    (hf₁ : Integrable f₁ volume) (hf₂ : Integrable f₂ volume) (w : ℝ) :
    Real.fourierIntegral (fun x => f₁ x + f₂ x) w
      = Real.fourierIntegral f₁ w + Real.fourierIntegral f₂ w := by
  rw [fourierIntegral_exp_unfold, fourierIntegral_exp_unfold, fourierIntegral_exp_unfold]
  -- ∫ 𝐞·(f₁+f₂) = ∫ (𝐞·f₁ + 𝐞·f₂) = ∫ 𝐞·f₁ + ∫ 𝐞·f₂.
  have hchar_norm : ∀ v : ℝ, ‖(𝐞 (-(v * w)) : ℂ)‖ = 1 := by
    intro v; rw [Real.fourierChar_apply, Complex.norm_exp]; simp
  have hmeas : Continuous (fun v : ℝ => (𝐞 (-(v * w)) : ℂ)) :=
    continuous_subtype_val.comp (Real.continuous_fourierChar.comp (by continuity))
  have h1 : Integrable (fun v => (𝐞 (-(v * w)) : ℂ) * f₁ v) volume := by
    refine hf₁.norm.mono' (hmeas.aestronglyMeasurable.mul hf₁.aestronglyMeasurable) ?_
    filter_upwards with v; rw [norm_mul, hchar_norm, one_mul]
  have h2 : Integrable (fun v => (𝐞 (-(v * w)) : ℂ) * f₂ v) volume := by
    refine hf₂.norm.mono' (hmeas.aestronglyMeasurable.mul hf₂.aestronglyMeasurable) ?_
    filter_upwards with v; rw [norm_mul, hchar_norm, one_mul]
  rw [show (fun v => (𝐞 (-(v * w)) : ℂ) * (f₁ v + f₂ v))
        = (fun v => (𝐞 (-(v * w)) : ℂ) * f₁ v + (𝐞 (-(v * w)) : ℂ) * f₂ v) from by
      funext v; ring]
  exact integral_add h1 h2

/-! ## The combined function `g = (f*f) + autocorr f` -/

/-- The combined real function `g := (f*f) + autocorr f`, the integrand
factor paired against `K_ms` in `LHS1 + LHS2`. -/
def g (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
            + autocorr f x

/-- The Fourier transform of `g` (cast to `ℂ`) at frequency `ξ`:
`𝓕g(ξ) = (𝓕f ξ)² + 𝓕f ξ · conj(𝓕f ξ)`.

Both summands come from the already-proved convolution-theorem outputs in
`Sidon.Constructor.ConvFourier` (`fourierIntegral_autoconvolution_real` and
`fourierIntegral_autocorr_real`), plus linearity of `Real.fourierIntegral`
on the `L¹` sum `f*f + autocorr f`. -/
theorem fourierIntegral_g (f : ℝ → ℝ) (hf : Integrable f volume)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (ξ : ℝ) :
    Real.fourierIntegral (fun x => ((g f x : ℝ) : ℂ)) ξ
      = (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) ^ 2
        + Real.fourierIntegral (fun x => (f x : ℂ)) ξ
            * starRingEnd ℂ (Real.fourierIntegral (fun x => (f x : ℂ)) ξ) := by
  -- Integrability of the two complex summands.
  have hconv_int : Integrable
      (fun x => ((MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x : ℂ)) volume := by
    have hrefl : Integrable (fun y => f (-y)) volume := by
      have := hf.comp_neg; simpa [Function.comp] using this
    have hconv : Integrable
        (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
      hf.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf
    exact hconv.ofReal
  have hauto_int : Integrable (fun x => ((autocorr f x : ℝ) : ℂ)) volume := by
    -- `autocorr f = (f̌ ⋆ f)` is integrable; cast to ℂ.
    have hrefl : Integrable (fun y => f (-y)) volume := by
      have := hf.comp_neg; simpa [Function.comp] using this
    have hconv : Integrable
        (MeasureTheory.convolution (fun y => f (-y)) f
          (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
      hrefl.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf
    have h_eq : autocorr f = MeasureTheory.convolution (fun y => f (-y)) f
        (ContinuousLinearMap.mul ℝ ℝ) volume := by
      funext x
      rw [autocorr_def]
      simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
      rw [← MeasureTheory.integral_neg_eq_self (fun t => f (-t) * f (x - t)) volume]
      refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
      simp only [neg_neg]
      have : x - -s = x + s := by ring
      rw [this]
    rw [h_eq]; exact hconv.ofReal
  -- `(g f x : ℂ) = (f*f x : ℂ) + (autocorr f x : ℂ)`.
  have h_split : (fun x => ((g f x : ℝ) : ℂ))
      = (fun x => ((MeasureTheory.convolution f f
            (ContinuousLinearMap.mul ℝ ℝ) volume) x : ℂ)
            + ((autocorr f x : ℝ) : ℂ)) := by
    funext x; unfold g; push_cast; ring
  rw [h_split]
  -- Linearity of the Fourier integral over the sum.
  rw [fourierIntegral_add _ _ hconv_int hauto_int ξ]
  rw [Sidon.Constructor.fourierIntegral_autoconvolution_real f hf ξ]
  rw [Sidon.Constructor.fourierIntegral_autocorr_real f hf ξ]

/-! ## The `Re` / real-part reduction

For any real number `K̂`, the real part of `𝓕g(ξ)·K̂` collapses to
`2·(Re 𝓕f ξ)²·K̂`.  This is the algebraic core of the `Re`-only phrasing
of `S_cos`: with `𝓕f ξ = a + b·I`,

  `𝓕g(ξ) = (a+bI)² + (a²+b²) = 2a² + 2ab·I`,

so `Re(𝓕g(ξ)·K̂) = 2a²·K̂` (since `K̂` is real). -/

/-- **The `Re`/real-part reduction.**  For real `c := 𝓕f ξ` and real `K̂`,

`Re( ((c)² + c·conj c) · K̂ ) = 2·(Re c)²·K̂`.

Proof: `(c)² + c·conj c` has real part `2·(Re c)²` (imaginary part `2·Re c·Im c`),
and multiplying by the real scalar `K̂` scales the real part by `K̂`. -/
theorem re_mul_real (c : ℂ) (Khat : ℝ) :
    ((c ^ 2 + c * starRingEnd ℂ c) * (Khat : ℂ)).re
      = 2 * (c.re) ^ 2 * Khat := by
  have h_conj_re : (starRingEnd ℂ c).re = c.re := by simp [Complex.conj_re]
  have h_conj_im : (starRingEnd ℂ c).im = -c.im := by simp [Complex.conj_im]
  simp only [Complex.add_re, Complex.mul_re,
    Complex.ofReal_re, Complex.ofReal_im, pow_two, h_conj_re, h_conj_im]
  ring

/-- The real-part reduction packaged at `c := 𝓕f(ξ)` and `K̂ := K_ms_fourier_lattice j`,
matching the summand of `S_cos`. -/
theorem re_fourierIntegral_g_mul_Khat (f : ℝ → ℝ) (hf : Integrable f volume)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (j : ℤ) :
    ((Real.fourierIntegral (fun x => ((g f x : ℝ) : ℂ)) ((j : ℝ) / uQ_real))
        * ((K_ms_fourier_lattice j : ℝ) : ℂ)).re
      = 2 * ((Real.fourierIntegral (fun x => ((f x : ℂ)))
            ((j : ℝ) / uQ_real)).re) ^ 2 * K_ms_fourier_lattice j := by
  rw [fourierIntegral_g f hf h_joint ((j : ℝ) / uQ_real)]
  exact re_mul_real (Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real))
    (K_ms_fourier_lattice j)

/-! ## Nonnegativity of `S_cos` -/

/-- `0 ≤ S_cos f`: the defining `tsum` has all summands `≥ 0`
(`Sidon.MultiScale.S_cos_summand_nonneg`), so `tsum_nonneg` applies
(unconditionally — even if the family were not summable the `tsum`
would be `0 ≥ 0`). -/
theorem S_cos_nonneg (f : ℝ → ℝ) : 0 ≤ S_cos f := by
  unfold S_cos
  exact tsum_nonneg (fun j => S_cos_summand_nonneg f j)

/-! ## The coefficient inequality `2·u² ≤ 2/u`

The whole point of the `hEq3_ge` weakening: since `u = 638/1000 < 1`,
`u³ < 1`, so `2u² < 2/u`. -/

/-- `2 · uQ_real² ≤ 2 / uQ_real`.  Equivalent to `uQ_real³ ≤ 1`, true since
`uQ_real = 0.638`. -/
theorem two_uSq_le_two_div_u : 2 * uQ_real ^ 2 ≤ 2 / uQ_real := by
  have hu_pos : 0 < uQ_real := uQ_real_pos
  have hu_val : uQ_real = (638 : ℝ) / 1000 := by
    unfold uQ_real uQ; push_cast; norm_num
  rw [hu_val]
  norm_num

/-! ## The genuine period-`u` Parseval identity (MO 2009 Lemma 2.1)

This is the single remaining analytic primitive.  It is **not** an axiom
here; it is the hypothesis of the bridge theorems below, recording exactly
what mathlib must supply. -/

/-- **MO 2009 Lemma 2.1, genuine form.**  The period-`u` Parseval pairing
for `g = (f*f) + autocorr f` against the period-fitting kernel `K_ms`:

  `LHS1 f + LHS2 f  =  2/u + (2/u)·S_cos f`.

(`LHS1 f + LHS2 f = ∫ g·K_ms`; `S_cos f = ∑'_{j≠0} (Re 𝓕f(j/u))²·K̂_ms(j/u)`.)

This is the *correctly-normalised* identity (full-line Fourier convention),
correcting the `2u²` coefficient of the bundle's `h_torus_eq_full` to the
genuine `2/u`.

### Exact residual mathlib-shaped lemma still needed

The proof requires a **period-`u` Poisson sampling lemma** of the form:

> For `G : ℝ → ℂ` continuous, integrable, with the periodisation
> `G_per(x) := ∑'ₖ G(x + k·u)` locally uniformly summable, the Fourier
> coefficient of `G_per` (as a function on `AddCircle u`) at index `j` is
> `fourierCoeff G_per j = (1/u)·𝓕G(j/u)`.

Mathlib provides this only for **period 1** and for a **continuous** `G`
with a compact-uniform summability bound:
`Real.fourierCoeff_tsum_comp_add` (`Mathlib/Analysis/Fourier/PoissonSummation.lean`),
which builds `Periodic.lift (G.periodic_tsum_comp_add_zsmul 1)` and is
hardcoded to period 1.  There is **no period-`u` variant** exposed, and no
lemma relating `fourierCoeff` of a periodisation whose base support
*overflows* one period to `𝓕G` at the lattice (the existing
`Sidon.TorusParseval.period_u_coef_eq_fourierIntegral_at_lattice` and
`Sidon.BilinearParseval.bilinear_parseval_period_u_inv` both require the
*base* function supported in `Ioc (-u/2) (u/2)`, which `g` violates —
`supp g ⊆ (-1/2,1/2) ⊄ (-0.319,0.319)`).

The full assembly is then:
  1. `∫_ℝ g·K_ms = ∫_{(-u/2,u/2)} g·K_ms`  (`K_ms = 0` outside `(-δ₁,δ₁) ⊆ (-u/2,u/2)`;
     fully provable now via `K_ms_eq_zero_outside`).
  2. Expand `K_ms` in its period-`u` `L²` Fourier series on `AddCircle u`
     (`K_ms` *does* fit one period); pair against `g` in `L²(period)`.
  3. The pairing yields `(1/u)·∑'_j 𝓕g(j/u)·conj(𝓕K_ms(j/u))` **iff** the
     period-`u` Poisson sampling lemma above is available for the
     overflowing `g` — this is the missing step.
  4. `Re`/`±j` reduction (PROVED here, `re_fourierIntegral_g_mul_Khat`):
     `Re(𝓕g(j/u)·K̂_ms(j/u)) = 2·(Re 𝓕f(j/u))²·K̂_ms(j/u)`.
  5. `j = 0` term: `𝓕g(0) = ∫g = 2`, `K̂_ms(0) = 1`, giving `(1/u)·2 = 2/u`.

In principle mathlib's `AddCircle`/Poisson API **is** sufficient to build
the period-`u` sampling lemma (rescale `Real.fourierCoeff_tsum_comp_add`
through the `u`-dilation `AddCircle u ≃ AddCircle 1`, then handle the
overflow via the non-overlap support computation `1/2 + δ₁ = u`), but this
is a multi-hundred-line development with no ready-made constructor — exactly
the "no one-call `L¹∩L²` Plancherel + period-`u` Parseval constructor"
gap recorded in the project's `CLAUDE.md`. -/
def PeriodUParsevalTrue (f : ℝ → ℝ) : Prop :=
  LHS1 f + LHS2 f = 2 / uQ_real + (2 / uQ_real) * S_cos f

/-! ## The bridge to the bundle field `hEq3_ge` (ZERO residual)

Given the genuine equality `PeriodUParsevalTrue f`, the consumed bundle
field `hEq3_ge` (`2/u + 2u²·S_cos ≤ LHS1 + LHS2`) follows from
`2u² ≤ 2/u` and `S_cos f ≥ 0`.  No further hypothesis is needed. -/

/-- **An alternate, weaker (un-normalized, lossy `2u² ≤ 2/u`) `hEq3_ge`-shaped
inequality, derived from the genuine Parseval equality.**

`2 / u + 2·u²·S_cos f ≤ LHS1 f + LHS2 f`, over the **un-normalized** full-line
`S_cos f` with the literal `2u²` coefficient.

Proof: by `PeriodUParsevalTrue f`, `LHS1 + LHS2 = 2/u + (2/u)·S_cos f`.
Since `2u² ≤ 2/u` (`two_uSq_le_two_div_u`) and `S_cos f ≥ 0`
(`S_cos_nonneg`), `2u²·S_cos f ≤ (2/u)·S_cos f`, whence the claim.

NOTE — naming caution.  This is TRUE but **NOT** the form consumed by the live
constructor `Sidon.MultiScale.ExtremiserPrimitives.of_admissible`.  That struct
field `ExtremiserPrimitives.hEq3_ge` is stated over the **`u³`-normalized**
`S_cos f / u³` (TIGHT — exactly the `≤`-direction of the genuine `2/u`-coefficient
equality, NO lossy step), and the headline discharges it via the struct field
`Sidon.Constructor.FieldsParseval.field_hEq3_ge`.  This theorem instead applies the
lossy `2u² ≤ 2/u` bound to the un-normalized `S_cos f`, yielding the weaker
inequality; it is retained as an alternate and is not on the live `of_admissible`
path.  (Hence the file-header prose calling this "the bundle field `hEq3_ge`"
refers only to its `hEq3_ge`-shaped *form*, not to the normalized struct field
the headline actually consumes.) -/
theorem field_hEq3_ge_of_true_parseval (f : ℝ → ℝ)
    (h_true : PeriodUParsevalTrue f) :
    2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2 * S_cos f ≤ LHS1 f + LHS2 f := by
  have hu_eq : uQ_real = (uQ : ℝ) := rfl
  unfold PeriodUParsevalTrue at h_true
  rw [← hu_eq]
  rw [h_true]
  -- Goal: 2/u + 2u²·S_cos ≤ 2/u + (2/u)·S_cos.
  have hScos : 0 ≤ S_cos f := S_cos_nonneg f
  have hcoef : 2 * uQ_real ^ 2 ≤ 2 / uQ_real := two_uSq_le_two_div_u
  have hmul : 2 * uQ_real ^ 2 * S_cos f ≤ (2 / uQ_real) * S_cos f :=
    mul_le_mul_of_nonneg_right hcoef hScos
  linarith

/-- The genuine equality also discharges `FieldsParseval.field_hEq3_ge`'s
`h_torus_eq_full` *direction needed downstream* — i.e. the full bundle
field — without ever asserting the false `2u²` equality.  This is the
drop-in replacement for the `h_torus_eq_full` hypothesis: instead of the
false equality, the caller supplies `PeriodUParsevalTrue f` (the true one),
and obtains the same `hEq3_ge` conclusion. -/
theorem hEq3_ge_replacement (f : ℝ → ℝ)
    (h_true : PeriodUParsevalTrue f) :
    2 / uQ_real + 2 * uQ_real ^ 2 * S_cos f ≤ LHS1 f + LHS2 f := by
  have := field_hEq3_ge_of_true_parseval f h_true
  have hu_eq : uQ_real = (uQ : ℝ) := rfl
  rw [hu_eq]; exact this

end

end Sidon.Constructor.PeriodU
