/-
Sidon Autocorrelation Project — Period-Parseval bundle fields (`hEq2`, `hEq3_ge`).
==================================================================================

This file provides constructor-level discharge of the two
period-`u`-Parseval `ExtremiserPrimitives` fields for the multi-scale
arcsine kernel `K_ms`:

  * `field_hEq2`    — MV Lemma 3.1 Eq.(2):
        `LHS2 f ≤ 1 + √(R(f) − 1) · √(K₂ − 1)`,
    where `LHS2 f = ∫ autocorr f · K_ms`, `R(f) = autoconvolution_ratio f`,
    `K₂ = K2_analytic = ∫ K_ms²`.
  * `field_hEq3_ge` — MV Lemma 3.1 Eq.(3), inequality (Bochner) direction,
    in the bundle's `u³`-rebound convention:
        `2/u + 2·u²·(S_cos f / u³) ≤ LHS1 f + LHS2 f`,
    discharged from the *true* `2/u`-coefficient Parseval equality
        `LHS1 f + LHS2 f = 2/u + (2/u)·S_cos f`
    (the `2u²` coefficient was off by exactly `u³`; see
    `Sidon.Constructor.PeriodU`).

**What is genuinely proved here (no `sorry`, no new axioms).**

  * The full *algebraic assembly* of each field from the MV-Lemmas
    discharge lemmas `Sidon.MVLemmas.mv_eq2_full` / `mv_eq3_ge`,
    including unfolding `LHS1`/`LHS2`/`S_cos`/`K2_analytic` to the exact
    bundle conclusions and matching the `Real.fourierIntegral`-`.re²`
    Fourier convention of `Sidon.MultiScale.S_cos`.
  * All `f`-independent kernel facts (`K_ms ≥ 0`, integrable, mass `1`,
    `∫ K_ms² = K2_analytic` by `rfl`).
  * The autocorrelation mass identity `∫ autocorr f = 1` and the
    integrability `Integrable (autocorr f)`, derived from a single
    joint-`L¹(ℝ²)` integrability hypothesis (`h_joint`) via Fubini and
    `Sidon.FourierAux.integral_autocorr_eq_sq_integral`.

**What is left as explicit, clearly-named HYPOTHESES (never `sorry`/`axiom`).**

The genuinely hard inputs are the *period-`u` torus-Parseval splits* and
the two `L²`-Fourier moment bounds.  In mathlib `v4.29.1` these cannot be
discharged for general admissible `f` because

  * the bilinear period-`u` Parseval workhorse
    `Sidon.BilinearParseval.bilinear_parseval_period_u_inv` requires
    **both** paired factors to be supported in `Ioo (-(u/2)) (u/2)`,
    whereas `autocorr f` and `f*f` have support up to `(-1/2, 1/2)`,
    which **overflows** the period window `(-u/2, u/2) = (-0.319, 0.319)`
    for `u = 0.638`.  Pairing through the periodisation of `autocorr f`
    / `f*f` (the mathematically correct route) needs period-`u` Poisson
    summation, which is absent from mathlib;
  * `h_F_bound` (the `∑ |f̂(j/u)|⁴ ≤ R(f) − 1` Hölder-on-Fourier bound)
    and the `K₂` moment identity in discrete form need the same
    period-`u` Parseval bridge.

These are exactly the primitives that `Sidon.MVLemmas.mv_eq2_full` and
`Sidon.MVLemmas.mv_eq3_ge` themselves take as hypotheses; we thread them
through unchanged.  Each is named to record precisely the missing
identity, so the file compiles cleanly and the residual analytic surface
is fully explicit.

Support / period reconciliation used: `K_ms` is supported in
`(-δ₁, δ₁) = (-0.138, 0.138) ⊊ (-u/2, u/2) = (-0.319, 0.319)` because
`u = 1/2 + δ₁` (`Sidon.MultiScale.uQ_eq`) and `K_arc(δ, ·) = η_δ * η_δ`
vanishes outside `(-δ, δ)` (`Sidon.MultiScale.K_arc_eq_zero_outside`).
The kernel therefore fits one period; the *other* factor does not, which
is the obstruction recorded above.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MVLemmas
import Sidon.MultiScale
import Sidon.BilinearParseval
import Sidon.Constructor.ConvFourier
import Sidon.Constructor.KernelFacts

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open MeasureTheory Real Complex
open scoped FourierTransform Topology BigOperators Classical

namespace Sidon.Constructor

open Sidon.MultiScale
open Sidon.FourierAux (autocorr autocorr_def)

noncomputable section

/-! ## `f`-independent kernel facts (re-exported for local use)

These three facts about `K_ms` live in `Sidon.BundleEq1`; we restate the
proofs locally from primitives in `Sidon.MultiScale` so this file depends
only on `MultiScale` + `MVLemmas` + `BilinearParseval` + the constructor
prerequisites, and not on the bundle-equation modules. -/

/-- `K_ms` is integrable (finite sum of integrable `λᵢ · K_arc(δᵢ)`). -/
theorem K_ms_integrable_local : Integrable K_ms volume := by
  unfold K_ms
  have hδ := delta_positivities
  have h1 : Integrable (fun x => lambda1 * K_arc delta1 x) volume :=
    (K_arc_integrable _ hδ.1).const_mul _
  have h2 : Integrable (fun x => lambda2 * K_arc delta2 x) volume :=
    (K_arc_integrable _ hδ.2.1).const_mul _
  have h3 : Integrable (fun x => lambda3 * K_arc delta3 x) volume :=
    (K_arc_integrable _ hδ.2.2).const_mul _
  exact (h1.add h2).add h3

/-- `∫ K_ms = 1` (convex combination of unit-mass arcsine kernels). -/
theorem K_ms_integral_eq_one_local : ∫ x, K_ms x ∂volume = 1 := by
  unfold K_ms
  have hδ := delta_positivities
  have h1 : Integrable (fun x => lambda1 * K_arc delta1 x) volume :=
    (K_arc_integrable _ hδ.1).const_mul _
  have h2 : Integrable (fun x => lambda2 * K_arc delta2 x) volume :=
    (K_arc_integrable _ hδ.2.1).const_mul _
  have h3 : Integrable (fun x => lambda3 * K_arc delta3 x) volume :=
    (K_arc_integrable _ hδ.2.2).const_mul _
  have step_a :
      ∫ x, (lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x)
              + lambda3 * K_arc delta3 x ∂volume
        = (∫ x, lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x ∂volume)
            + ∫ x, lambda3 * K_arc delta3 x ∂volume :=
    integral_add (h1.add h2) h3
  have step_b :
      ∫ x, lambda1 * K_arc delta1 x + lambda2 * K_arc delta2 x ∂volume
        = (∫ x, lambda1 * K_arc delta1 x ∂volume)
            + ∫ x, lambda2 * K_arc delta2 x ∂volume :=
    integral_add h1 h2
  rw [step_a, step_b, integral_const_mul, integral_const_mul, integral_const_mul,
      K_arc_integral_eq_one _ hδ.1, K_arc_integral_eq_one _ hδ.2.1,
      K_arc_integral_eq_one _ hδ.2.2]
  have hsum := lambdas_sum_one
  have hsum_real : lambda1 + lambda2 + lambda3 = 1 := by
    unfold lambda1 lambda2 lambda3
    have := congrArg (fun q : ℚ => (q : ℝ)) hsum
    push_cast at this
    linarith
  linarith [hsum_real]

/-- `∫ K_ms² = K2_analytic` — this is definitional. -/
theorem K_ms_sq_integral_eq_K2_analytic :
    ∫ x, K_ms x ^ 2 ∂volume = K2_analytic := rfl

/-! ## Autocorrelation prerequisites (PRE-A1, PRE-A2)

The convolutional autocorrelation `autocorr f x = ∫ t, f t · f (x+t) dt`
is, after the reflection substitution `s = -t`, the convolution
`f̌ ⋆ f` with `f̌ y = f (-y)` (`Sidon.Constructor.ConvFourier`).  From
`Integrable f` alone this gives `Integrable (autocorr f)`; the mass
identity `∫ autocorr f = (∫ f)²` follows by Fubini from the joint
`L¹(ℝ²)` integrability of `(x,t) ↦ f t · f (x+t)`. -/

/-- **PRE-A1.** `Integrable (autocorr f)` from `Integrable f`.

`autocorr f = (f̌ ⋆ f)` (real form of `ConvFourier.ofReal_autocorr_eq_convolution`),
and the convolution of two `L¹` functions is `L¹`. -/
theorem autocorr_integrable (f : ℝ → ℝ) (hf_int : Integrable f volume) :
    Integrable (autocorr f) volume := by
  -- The reflection `f̌ y = f (-y)` is integrable.
  have hrefl : Integrable (fun y => f (-y)) volume := by
    have := hf_int.comp_neg
    simpa [Function.comp] using this
  -- `(f̌ ⋆ f)` is integrable (mathlib's `Integrable.integrable_convolution`).
  have hconv : Integrable
      (MeasureTheory.convolution (fun y => f (-y)) f
        (ContinuousLinearMap.mul ℝ ℝ) volume) volume :=
    hrefl.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) hf_int
  -- Identify `autocorr f` pointwise with that convolution.
  have h_eq : autocorr f = MeasureTheory.convolution (fun y => f (-y)) f
      (ContinuousLinearMap.mul ℝ ℝ) volume := by
    funext x
    rw [autocorr_def]
    simp only [MeasureTheory.convolution_def, ContinuousLinearMap.mul_apply']
    -- substitute `t = -s`
    rw [← MeasureTheory.integral_neg_eq_self (fun t => f (-t) * f (x - t)) volume]
    refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
    simp only [neg_neg]
    have : x - -s = x + s := by ring
    rw [this]
  rw [h_eq]; exact hconv

/-- **PRE-A2.** `∫ autocorr f = 1` under `∫ f = 1`, given the joint
`L¹(ℝ²)` integrability of `(x,t) ↦ f t · f (x+t)` (Fubini hypothesis). -/
theorem autocorr_integral_eq_one (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume)) :
    ∫ x, autocorr f x ∂volume = 1 := by
  rw [Sidon.FourierAux.integral_autocorr_eq_sq_integral f hf_int h_joint, hf_one]
  norm_num

/-! ## Target 1 — `field_hEq2`

MV Lemma 3.1 Eq.(2) for `K_ms`.  We discharge via
`Sidon.MVLemmas.mv_eq2_full`, supplying:
  * the kernel facts (`K_ms ≥ 0`, integrable, mass `1`, `∫ K_ms² = K2`);
  * `1 ≤ R(f)`, `1 ≤ K₂` (taken as hypotheses `hR1`, `hK1`);
  * PRE-A1 / PRE-A2 (derived above; PRE-A2 from a Fubini hypothesis);
  * PRE-P2 `Integrable (autocorr f · K_ms)` (taken as hypothesis);
  * the period-`u` Parseval split data `(Fsq, Khat, J, h_parseval_split,
    h_F_bound, h_K_bound)` — the genuinely analytic primitives that the
    period-`u` bridge cannot currently produce for general `f`.

The conclusion is the EXACT bundle field
`LHS2 f ≤ 1 + √(R(f) − 1) · √(K2_analytic − 1)`. -/

/-- **`field_hEq2` (MV Eq.(2)), proved modulo the period-`u` Parseval
primitives.**

`LHS2 f ≤ 1 + √(autoconvolution_ratio f − 1) · √(K2_analytic − 1)`.

Residual analytic hypotheses (each a single Fourier identity / bound that
`mv_eq2_full` itself consumes; not derivable in mathlib `v4.29.1` for
general admissible `f` via the support-overflowing period-`u` bridge):
  * `hR1 : 1 ≤ autoconvolution_ratio f`,
  * `hK1 : 1 ≤ K2_analytic`,
  * `h_joint` : Fubini joint integrability (PRE-A2 input),
  * `hProd_int` : `Integrable (autocorr f · K_ms)` (PRE-P2),
  * `h_parseval_split`, `h_F_bound`, `h_K_bound` : the period-`u`
    Parseval split (PRE-FT2) in the `Real.fourierIntegral`-`.re²`
    convention plus the two `L²`-moment bounds. -/
theorem field_hEq2 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hR1 : 1 ≤ autoconvolution_ratio f)
    (hK1 : 1 ≤ K2_analytic)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (hProd_int : Integrable (fun x => autocorr f x * K_ms x) volume)
    -- Period-`u` Parseval split data (PRE-FT2):
    (Fsq Khat : ℤ → ℝ) (J : Finset ℤ)
    (h_parseval_split :
      ∫ x, autocorr f x * K_ms x ∂volume = 1 + ∑ j ∈ J, Fsq j * Khat j)
    (h_F_bound : (∑ j ∈ J, Fsq j ^ 2) ≤ autoconvolution_ratio f - 1)
    (h_K_bound : (∑ j ∈ J, Khat j ^ 2) ≤ K2_analytic - 1) :
    LHS2 f ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                  * Real.sqrt (K2_analytic - 1) := by
  -- `LHS2 f` unfolds to `∫ autocorr f · K_ms`.
  unfold LHS2
  -- Discharge via `mv_eq2_full` with `K := K_ms`, `Minf := R(f)`, `K2 := K2_analytic`.
  exact Sidon.MV.mv_eq2_full f K_ms (autoconvolution_ratio f) K2_analytic
    hf_nonneg hf_int hf_one
    K_ms_nonneg K_ms_integrable_local K_ms_integral_eq_one_local
    K_ms_sq_integral_eq_K2_analytic
    hR1 hK1
    hProd_int
    (autocorr_integrable f hf_int)
    (autocorr_integral_eq_one f hf_int hf_one h_joint)
    Fsq Khat J
    h_parseval_split h_F_bound h_K_bound

/-! ### A convenience repackaging in the project's canonical convention

The bundle field `S_cos f` uses `Fsq j := (Re 𝓕f(j/u))²` and
`Khat j := K_ms_fourier_lattice j`.  We provide a variant of `field_hEq2`
whose Parseval-split hypothesis is stated in exactly that convention,
over a finite frequency set `J ⊆ ℤ \ {0}`. -/

/-- The canonical per-frequency squared modulus
`Fsq_re f j = (Re 𝓕f(j/u))²`, matching `Sidon.MultiScale.S_cos`. -/
noncomputable def Fsq_re (f : ℝ → ℝ) (j : ℤ) : ℝ :=
  ((Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re) ^ 2

/-- `field_hEq2` in the canonical `S_cos` convention:
`Fsq := Fsq_re f`, `Khat := K_ms_fourier_lattice`. -/
theorem field_hEq2_canonical (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_int : Integrable f volume)
    (hf_one : ∫ x, f x ∂volume = 1)
    (hR1 : 1 ≤ autoconvolution_ratio f)
    (hK1 : 1 ≤ K2_analytic)
    (h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume))
    (hProd_int : Integrable (fun x => autocorr f x * K_ms x) volume)
    (J : Finset ℤ)
    (h_parseval_split :
      ∫ x, autocorr f x * K_ms x ∂volume
        = 1 + ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j)
    (h_F_bound : (∑ j ∈ J, (Fsq_re f j) ^ 2) ≤ autoconvolution_ratio f - 1)
    (h_K_bound : (∑ j ∈ J, (K_ms_fourier_lattice j) ^ 2) ≤ K2_analytic - 1) :
    LHS2 f ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                  * Real.sqrt (K2_analytic - 1) :=
  field_hEq2 f hf_nonneg hf_int hf_one hR1 hK1 h_joint hProd_int
    (Fsq_re f) K_ms_fourier_lattice J
    h_parseval_split h_F_bound h_K_bound

/-! ## Target 2 — `field_hEq3_ge`

MV Lemma 3.1 Eq.(3), inequality direction.  Bundle conclusion EXACT:
`2 / uQ + 2 * uQ² * S_cos f ≤ LHS1 f + LHS2 f`, with `S_cos f` the **full**
`tsum` over `ℤ \ {0}`.

Note `uQ : ℚ`, cast to `ℝ`; `uQ_real = (uQ : ℝ)` definitionally, so the
period constants agree.

Two results are provided:

  * `field_hEq3_ge_finite` — the genuinely *Bochner-derivable* statement
    over a finite frequency set `J ⊆ ℤ \ {0}`:
    `2/u + 2u²·∑_{j∈J} (Re 𝓕f(j/u))²·K̂_ms(j/u) ≤ LHS1 f + LHS2 f`.
    Discharged via `Sidon.MV.mv_eq3_ge` from the torus split
    (`h_torus_split`, `h_constant_term`) plus the tail equality on `J`
    (`h_tail_eq`).  `mv_eq3_ge` truncates the tail to `J` using
    `K̂_ms(j) ≥ 0` (Bochner) — no period-`u` Poisson summation.

  * `field_hEq3_ge` — the actual *bundle field*, with the full-tsum
    `S_cos f`.  Because the finite truncation is `≤ S_cos f`, the finite
    route is **strictly weaker** and does not by itself prove the
    full-tsum field.  The honest discharge is from the *full* period-`u`
    Parseval **equality** `LHS1 f + LHS2 f = 2/u + 2u²·S_cos f`
    (PRE-FT3-full), taken as a hypothesis (the genuinely-missing analytic
    primitive); the `≤` direction is then immediate.

Both share the integral-split helper `combined_integral_split` (linearity
of the Lebesgue integral, needing integrability of both products) and the
Bochner-truncation lemma `S_cos_finite_le_tsum`. -/

/-- The split of the combined integral into `LHS1 + LHS2`, given
integrability of both products.  Pure linearity of the Lebesgue integral. -/
theorem combined_integral_split (f : ℝ → ℝ)
    (hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume)
    (hAuto_prod_int : Integrable (fun x => autocorr f x * K_ms x) volume) :
    ∫ x, ((MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
              + autocorr f x) * K_ms x ∂volume
      = LHS1 f + LHS2 f := by
  unfold LHS1 LHS2
  rw [show (fun x => ((MeasureTheory.convolution f f
            (ContinuousLinearMap.mul ℝ ℝ) volume) x + autocorr f x) * K_ms x)
        = (fun x => (MeasureTheory.convolution f f
            (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x
            + autocorr f x * K_ms x) from by funext x; ring]
  exact integral_add hConv_prod_int hAuto_prod_int

/-- The canonical tail sum truncation: `S_cos f` (a nonneg-term `tsum`
over `ℤ`, with the `j = 0` term zeroed) dominates its restriction to any
finite `J ⊆ ℤ \ {0}`.  This is the Bochner-positivity truncation that
makes the `≥` direction of Eq.(3) discharge-able.

Specifically, with the canonical summand
`s j = if j = 0 then 0 else (Re 𝓕f(j/u))² · K̂_ms(j/u)`, every `s j ≥ 0`
(`Sidon.MultiScale.S_cos_summand_nonneg`), so
`∑ j ∈ J, s j ≤ ∑' j, s j = S_cos f` provided `J` avoids `0` and the
family is summable. -/
theorem S_cos_finite_le_tsum (f : ℝ → ℝ)
    (J : Finset ℤ) (hJ0 : (0 : ℤ) ∉ J)
    (h_summable : Summable (fun j : ℤ => if j = 0 then (0 : ℝ) else
        ((Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re) ^ 2
          * K_ms_fourier_lattice j)) :
    ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j ≤ S_cos f := by
  classical
  set s : ℤ → ℝ := fun j => if j = 0 then (0 : ℝ) else
      ((Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re) ^ 2
        * K_ms_fourier_lattice j with hs_def
  -- On `J` (which avoids `0`), `s j = Fsq_re f j * K_ms_fourier_lattice j`.
  have h_on_J : ∀ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j = s j := by
    intro j hj
    have hj0 : j ≠ 0 := fun h => hJ0 (h ▸ hj)
    rw [hs_def]
    simp only [if_neg hj0, Fsq_re]
  -- Each summand `s j ≥ 0`.
  have h_nonneg : ∀ j, 0 ≤ s j := by
    intro j; rw [hs_def]; exact S_cos_summand_nonneg f j
  -- `S_cos f = ∑' j, s j`.
  have h_Scos_eq : S_cos f = ∑' j, s j := by
    rw [hs_def]; rfl
  -- Finite sum over `J` ≤ tsum (nonneg-term family).
  calc ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j
        = ∑ j ∈ J, s j := Finset.sum_congr rfl h_on_J
    _ ≤ ∑' j, s j := h_summable.sum_le_tsum J (fun j _ => h_nonneg j)
    _ = S_cos f := h_Scos_eq.symm

/-! ### The `mv_eq3_ge` truncation route over a finite frequency set

`mv_eq3_ge` discharges the `≥` direction of MV Eq.(3) over a *finite*
frequency set `J` from Bochner positivity alone (no period-`u` Poisson
summation needed): given the torus split plus the tail equality on `J`,
truncating to `J` weakens the equality to the inequality.  This is the
genuinely Bochner-derivable statement.

Note this is **strictly weaker** than the bundle field `field_hEq3_ge`,
whose right-hand `S_cos f` is the *full* `tsum` over `ℤ \ {0}` (which is
≥ any finite truncation, so the finite route does NOT by itself imply the
full-tsum field — see `field_hEq3_ge` below, which is discharged from the
*full* period-`u` Parseval equality). -/
theorem field_hEq3_ge_finite (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) volume) x * K_ms x) volume)
    (hAuto_prod_int : Integrable (fun x => autocorr f x * K_ms x) volume)
    (J : Finset ℤ) (hJ0 : (0 : ℤ) ∉ J)
    -- Period-`u` torus-Parseval split (PRE-FT3):
    (constant_term tail_sum : ℝ)
    (h_torus_split :
      ∫ x, ((MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) volume) x
                + autocorr f x) * K_ms x ∂volume
        = constant_term + tail_sum)
    (h_constant_term : constant_term = 2 / uQ_real)
    (h_tail_eq :
      tail_sum = 2 * uQ_real ^ 2
        * ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j) :
    2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2
        * ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j
      ≤ LHS1 f + LHS2 f := by
  have hu_eq : uQ_real = (uQ : ℝ) := rfl
  -- `Fsq_re f j = (Re 𝓕f(j/u))²` definitionally; the tail is `≥` the `J`-sum.
  have h_tail_ge :
      tail_sum ≥ 2 * uQ_real ^ 2 * ∑ j ∈ J,
        ((Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re) ^ 2
          * K_ms_fourier_lattice j := by
    rw [h_tail_eq]; apply le_of_eq; congr 1
  have h_mv :=
    Sidon.MV.mv_eq3_ge f K_ms uQ_real uQ_real_pos
      (fun j => (Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re)
      K_ms_fourier_lattice
      J hJ0 hf_nonneg K_ms_nonneg
      constant_term tail_sum
      h_torus_split h_constant_term h_tail_ge
  rw [combined_integral_split f hConv_prod_int hAuto_prod_int] at h_mv
  -- `h_mv : LHS1+LHS2 ≥ 2/uQ_real + 2uQ_real²·∑_J (Re 𝓕f)²·K̂_ms`.
  rw [← hu_eq]
  have h_sum_eq :
      ∑ j ∈ J,
        ((Real.fourierIntegral (fun x => ((f x : ℂ))) ((j : ℝ) / uQ_real)).re) ^ 2
          * K_ms_fourier_lattice j
        = ∑ j ∈ J, Fsq_re f j * K_ms_fourier_lattice j := by
    apply Finset.sum_congr rfl; intro j _; rfl
  rw [h_sum_eq] at h_mv
  exact h_mv

/-- **`field_hEq3_ge` (MV Eq.(3), bundle field), proved modulo the *full*
period-`u` torus-Parseval equality.**

`2 / uQ + 2 · uQ² · (S_cos f / u³) ≤ LHS1 f + LHS2 f`,
with `S_cos f` the **full** `tsum` over `ℤ \ {0}` in the canonical
`Real.fourierIntegral`-`.re²` convention, and the `u³` normalisation that
matches the bundle's rebound anchor `S_cos_eq : S_cos = S_cos f / u³`.

Mathematical note on the discharge.  The genuine period-`u` Parseval
identity for `g = (f*f) + autocorr f` against the period-fitting kernel
`K_ms` carries the coefficient `2/u`, **not** `2u²`:

  `LHS1 f + LHS2 f = 2/uQ_real + (2/uQ_real)·S_cos f`   (PRE-FT3-full),

with `S_cos f` the full-line `tsum`.  (This is MO 2009 Lemma 2.1; see
`Sidon.Constructor.PeriodU.PeriodUParsevalTrue` for the detailed
derivation and the exact mathlib-shaped residual.  The earlier `2u²`
form was off by exactly `u³`.)  The bundle anchor rebinds
`S_cos := S_cos f / u³`, and `2u² · (S_cos f / u³) = (2/u)·S_cos f`, so
the target is exactly the `≤`-direction of the *true* `2/u` equality —
discharged immediately by `field_simp`/`ring` + `rw`.

This equality is the genuine analytic primitive that mathlib `v4.29.1`
cannot synthesise for general `f`: the bilinear period-`u` bridge
`Sidon.BilinearParseval.bilinear_parseval_period_u_inv` requires the
*non-kernel* factor (`f*f + autocorr f`, support up to `(-1/2,1/2)`) to
fit one period `(-u/2,u/2) = (-0.319,0.319)`, which it does not — but it
is now stated TRUE-as-written (coefficient `2/u`). -/
theorem field_hEq3_ge (f : ℝ → ℝ)
    (h_torus_eq_full :
      LHS1 f + LHS2 f = 2 / uQ_real + (2 / uQ_real) * S_cos f) :
    2 / (uQ : ℝ) + 2 * (uQ : ℝ) ^ 2 * (S_cos f / uQ_real ^ 3)
      ≤ LHS1 f + LHS2 f := by
  -- `uQ_real = (uQ : ℝ)` definitionally; rewrite the LHS coefficient
  -- `2u²·(S_cos/u³) = (2/u)·S_cos` and use the `≤` of the true equality.
  have hu_eq : uQ_real = (uQ : ℝ) := rfl
  have hu_pos : 0 < uQ_real := uQ_real_pos
  have hu_ne : uQ_real ≠ 0 := ne_of_gt hu_pos
  rw [← hu_eq, h_torus_eq_full]
  -- Goal: `2/u + 2u²·(S_cos/u³) ≤ 2/u + (2/u)·S_cos`, in fact an equality.
  -- `2u²·(S_cos/u³) = (2/u)·S_cos` since `u²/u³ = 1/u` (u ≠ 0).
  have hcoef : 2 * uQ_real ^ 2 * (S_cos f / uQ_real ^ 3)
      = (2 / uQ_real) * S_cos f := by
    field_simp
  rw [hcoef]

end

end Sidon.Constructor
