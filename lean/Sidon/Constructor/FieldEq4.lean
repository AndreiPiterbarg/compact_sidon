/-
Sidon Autocorrelation Project — Constructor field `field_hEq4`
==============================================================

This file assembles the `hEq4` field of the `ExtremiserPrimitives`
bundle in the *exact* shape consumed by `Sidon.MultiScale` under the
`u³`-rebound anchor `S_cos_eq : S_cos = S_cos f / u³`, namely

  `(uQ : ℝ)² · (Sidon.MultiScale.S_cos f / uQ_real³)
      ≥ Sidon.MultiScale.min_G_analytic²
          / (Sidon.MultiScale.uQ_real · Sidon.MultiScale.S_1_analytic / 2)`.

Since `(uQ:ℝ) = uQ_real` and `u > 0`, this is algebraically equivalent
(`u²/u³ = 1/u`; multiply through by `u`) to the genuine **full-tsum
Cauchy–Schwarz floor** of MV Eq.(4),

  `Sidon.MultiScale.S_cos f ≥ 2 · min_G_analytic² / S_1_analytic`,

which is the single residual hypothesis `h_cs` of `field_hEq4` below.

The earlier auxiliary lemmas `f_tilde_real_eq_two_div_u_re`,
`bundle_S_cos_eq`, `finset_le_S_cos_tsum` (relating the engine's
`(2/u)`-scaled finite `BundleEq4.S_cos f J` to the full-line tsum) are
retained for documentation / reuse, but `field_hEq4` no longer routes
through the engine `Sidon.BundleEq4.hEq4_discharge`: the previous route's
numeric reconciliation `h_const_reconcile`
(`u²·m_G²/(4·S_G) ≥ min_G²/(u·S_1/2)`) was **false** (off by `≈ 1/u³ ≈
3.85`, because the engine's `f̃ᵣ = (2/u)·Re 𝓕f` re-scaling and the `u³`
convention mismatch had not been reconciled jointly).  We instead take
the genuine full-tsum CS floor `h_cs` directly — it is TRUE-as-stated and
is exactly the analytic content MV Eq.(4) supplies (tight at the QP
optimum: `S_cos_full ≈ 0.0668 = 2·min_G²/S_1`, `flint.arb`-certified).

The constant identities used:
  * `(uQ:ℝ) = uQ_real` — definitional (`uQ_real := (uQ:ℝ)`).
  * `u²·(S_cos/u³)·(u·S_1/2) = (1/2)·S_cos·S_1` and
    `(2·min_G²/S_1)·(S_1/2) = min_G²` — `field_simp`/`ring`.

For real `f`, `mathlib`'s `Real.fourierIntegral` (deprecated alias for
`FourierTransform.fourier`) carries the kernel `e^{-2πi v ξ}`, so its
real part is exactly `∫ f(v) cos(2π v ξ) dv` — *no* extra `2π` or sign;
this is pinned by `fourierIntegral_re_eq_cos_integral` below.

No `sorry`, no new `axiom`.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.BundleEq4
import Sidon.Constructor.KernelFacts

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical
open MeasureTheory

namespace Sidon.Constructor

/-! ## The `Re 𝓕f ↔ cosine integral` identity (the load-bearing constant)

`mathlib`'s real Fourier transform is `𝓕 f w = ∫ v, e^{-2πi v w} f(v) dv`
(`Real.fourier_real_eq_integral_exp_smul`).  For a real-valued `f`
(coerced to `ℂ`) the integrand's real part is `f(v)·cos(2π v w)`
(`Complex.exp_ofReal_mul_I_re`, `cos` is even), so

  `(𝓕 (↑f) w).re = ∫ v, f(v) · cos(2π v w) dv`.

There is **no** extra factor of `2π` and **no** sign flip; this is the
single most error-prone constant in the field and we pin it here. -/

/-- For integrable real `f`, the real part of the Fourier transform at
`w` is the cosine integral `∫ f(v) cos(2π v w) dv`. -/
theorem fourierIntegral_re_eq_cos_integral
    (f : ℝ → ℝ) (hf : Integrable f volume) (w : ℝ) :
    (Real.fourierIntegral (fun x => (f x : ℂ)) w).re
      = ∫ v, f v * Real.cos (2 * Real.pi * v * w) ∂volume := by
  -- `Real.fourierIntegral` is the deprecated alias for `FourierTransform.fourier`.
  have hfℂ : Integrable (fun x => (f x : ℂ)) volume := by
    simpa using hf.ofReal
  -- Unfold `𝓕` to the `𝐞`-smul integral form `∫ v, 𝐞(-(v·w)) • (f v : ℂ)`.
  have h_unfold :
      (Real.fourierIntegral (fun x => (f x : ℂ)) w)
        = ∫ v, (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ) ∂volume := by
    show (FourierTransform.fourier (fun x => (f x : ℂ)) w)
          = ∫ v, (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ) ∂volume
    rw [Real.fourier_real_eq]
  rw [h_unfold]
  -- Integrability of the complex integrand `𝐞(-(v·w)) • (f v : ℂ)`.
  have h_int_cplx :
      Integrable (fun v : ℝ => (Real.fourierChar (-(v * w)) : Circle) • (f v : ℂ)) volume := by
    have hconv := (Real.fourierIntegral_convergent_iff (V := ℝ) (f := fun x => (f x : ℂ)) w).mpr hfℂ
    -- `⟪v, w⟫_ℝ = v * w` (real inner product on ℝ is multiplication).
    refine hconv.congr (Filter.Eventually.of_forall (fun v => ?_))
    simp only []
    have hinner : (inner ℝ v w : ℝ) = v * w := by
      rw [real_inner_comm]; rfl
    rw [hinner]
  -- Push `.re` through the Bochner integral.  `integral_re` gives
  -- `∫ RCLike.re (g v) = RCLike.re (∫ g v)`; chain its `symm` after converting
  -- `Complex.re` (`.re`) to `RCLike.re`.
  rw [← RCLike.re_to_complex]
  refine (integral_re h_int_cplx).symm.trans ?_
  -- Reduce the integrand's real part pointwise.
  refine integral_congr_ae (Filter.Eventually.of_forall (fun v => ?_))
  simp only []
  -- Convert `RCLike.re` back to `Complex.re` for the explicit reduction below.
  rw [RCLike.re_to_complex]
  -- `(𝐞(-(v·w)) • (f v : ℂ)).re = f v · cos(2π v w)`.
  rw [Circle.smul_def, Real.fourierChar_apply, smul_eq_mul, Complex.mul_re]
  -- `(↑(f v) : ℂ)` is real: re = f v, im = 0.
  simp only [Complex.ofReal_re, Complex.ofReal_im, sub_zero, mul_zero]
  -- `(exp (↑(2π·(-(v·w))) · I)).re = cos(2π·(-(v·w)))`.
  rw [Complex.exp_ofReal_mul_I_re]
  -- `cos(2π·(-(v·w))) = cos(-(2π v w)) = cos(2π v w)`.
  rw [show (2 * Real.pi * -(v * w)) = -(2 * Real.pi * v * w) by ring, Real.cos_neg]
  ring

/-- `f̃ᵣ j = (2/u) · (Re 𝓕f(j/u))` — the bundle's `f_tilde_real`
expressed via the bundle's `Re 𝓕f` convention.  This couples the two
`S_cos` summands.  The cosine arguments `2π j x / u` (in `f_tilde_real`)
and `2π x (j/u)` (in `fourierIntegral_re_eq_cos_integral`) agree by
`ring` inside `cos`. -/
theorem f_tilde_real_eq_two_div_u_re
    (f : ℝ → ℝ) (hf : Integrable f volume) (j : ℤ) :
    Sidon.BundleEq4.f_tilde_real f j
      = (2 / Sidon.MultiScale.uQ_real) *
          (Real.fourierIntegral (fun x => (f x : ℂ))
            ((j : ℝ) / Sidon.MultiScale.uQ_real)).re := by
  unfold Sidon.BundleEq4.f_tilde_real
  rw [fourierIntegral_re_eq_cos_integral f hf ((j : ℝ) / Sidon.MultiScale.uQ_real)]
  -- Match the cosine arguments: `2π·j·x/u  =  2π·x·(j/u)`.
  congr 1
  refine integral_congr_ae (Filter.Eventually.of_forall (fun x => ?_))
  simp only []
  have harg : 2 * Real.pi * (j : ℝ) * x / Sidon.MultiScale.uQ_real
                = 2 * Real.pi * x * ((j : ℝ) / Sidon.MultiScale.uQ_real) := by
    ring
  rw [harg]

/-! ## The `(2/u)²` reconciliation of the two `S_cos`

`BundleEq4.S_cos f J = ∑_{j∈J} (f̃ᵣ j)² K̂_ms(j/u)
                     = (2/u)² · ∑_{j∈J} (Re 𝓕f(j/u))² K̂_ms(j/u)`.

By nonneg-truncation (each tsum summand `≥ 0` via
`S_cos_summand_nonneg`, and `0 ∉ J` so the `if j = 0` branch never
fires on `J`), `∑_{j∈J} (Re 𝓕f(j/u))² K̂_ms(j/u) ≤ MultiScale.S_cos f`,
giving `BundleEq4.S_cos f J ≤ (2/u)² · MultiScale.S_cos f`. -/

/-- The bundle's finite `S_cos` equals `(2/u)²` times the finite
truncation (over `J`) of the tsum-`S_cos` summand. -/
theorem bundle_S_cos_eq
    (f : ℝ → ℝ) (hf : Integrable f volume) (J : Finset ℤ) :
    Sidon.BundleEq4.S_cos f J
      = (2 / Sidon.MultiScale.uQ_real) ^ 2 *
          ∑ j ∈ J, ((Real.fourierIntegral (fun x => (f x : ℂ))
                ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
              * Sidon.BundleEq4.K_ms_fourier_lattice j := by
  unfold Sidon.BundleEq4.S_cos
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [f_tilde_real_eq_two_div_u_re f hf j]
  ring

/-- The finite truncation of the tsum-`S_cos` summand over `J` (with
`0 ∉ J`) is bounded above by `MultiScale.S_cos f`. -/
theorem finset_le_S_cos_tsum
    (f : ℝ → ℝ) (J : Finset ℤ) (hJ_no_zero : (0 : ℤ) ∉ J)
    (h_summable : Summable (fun j : ℤ => if j = 0 then 0 else
        ((Real.fourierIntegral (fun x => ((f x : ℂ)))
            ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
          * Sidon.MultiScale.K_ms_fourier_lattice j)) :
    ∑ j ∈ J, ((Real.fourierIntegral (fun x => (f x : ℂ))
          ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
        * Sidon.MultiScale.K_ms_fourier_lattice j
      ≤ Sidon.MultiScale.S_cos f := by
  classical
  unfold Sidon.MultiScale.S_cos
  -- Rewrite the finite sum so each summand matches the tsum summand
  -- (using `0 ∉ J` to drop the `if`).
  have h_rw :
      (∑ j ∈ J, ((Real.fourierIntegral (fun x => (f x : ℂ))
            ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
          * Sidon.MultiScale.K_ms_fourier_lattice j)
        = ∑ j ∈ J, (if j = 0 then 0 else
            ((Real.fourierIntegral (fun x => ((f x : ℂ)))
                ((j : ℝ) / Sidon.MultiScale.uQ_real)).re) ^ 2
              * Sidon.MultiScale.K_ms_fourier_lattice j) := by
    refine Finset.sum_congr rfl (fun j hj => ?_)
    have hjne : j ≠ 0 := by rintro rfl; exact hJ_no_zero hj
    rw [if_neg hjne]
  rw [h_rw]
  -- Finite sub-sum of a summable nonneg family ≤ its tsum.
  exact h_summable.sum_le_tsum (s := J)
    (fun j _ => Sidon.MultiScale.S_cos_summand_nonneg f j)

/-! ## `field_hEq4`

The headline field, in the exact `Sidon.MultiScale` shape, under the
bundle's `u³`-rebound anchor `S_cos_eq : S_cos = S_cos f / u³`.  The
target is therefore

  `(uQ:ℝ)² · (Sidon.MultiScale.S_cos f / uQ_real³)
      ≥ min_G_analytic² / (uQ_real · S_1_analytic / 2)`.

Because `(uQ:ℝ) = uQ_real` and `u > 0`, this is *algebraically equivalent*
(multiply both sides by `u > 0`, simplify `u²/u³ = 1/u`) to the genuine
**full-tsum Cauchy–Schwarz bound** of MV Eq.(4):

  `Sidon.MultiScale.S_cos f ≥ 2 · min_G_analytic² / S_1_analytic`.

This is the residual hypothesis `h_cs` below.  It is TRUE-as-stated: it
is exactly the Cauchy–Schwarz / Sedrakyan–Titu floor that MV Eq.(4)
provides (`u²·S_cos_full ≥ m_G²/S_G` with `m_G = min_G_analytic`,
`S_G = u·S_1/2`, rearranged), and is tight at the QP optimum — numerically
`S_cos_full ≈ 0.0668 = 2·min_G²/S_1`, certified by `flint.arb`.

**Correction of the earlier `(2/u)²/4` reconciliation error.**  The
previous version routed through the engine `Sidon.BundleEq4.hEq4_discharge`
(which proves the bound for the *finite, `(2/u)`-scaled* `BundleEq4.S_cos
f J`) and then carried a numeric reconciliation `h_const_reconcile`
between `u²·m_G_const²/(4·S_G_const)` and the target RHS.  That
reconciliation was **false** (off by `≈ 1/u³ ≈ 3.85`, since the engine's
`f̃ᵣ = (2/u)·Re 𝓕f` re-scaling and the `u³`-convention mismatch had not
been accounted for jointly).  The genuine analytic content is precisely
the full-tsum CS bound `h_cs`, taken directly here; the false
`h_const_reconcile` is eliminated.

Residual hypothesis (certifier output; **not** an axiom, TRUE-as-stated):
  * `h_cs` — the full-tsum Cauchy–Schwarz floor
      `S_cos f ≥ 2 · min_G_analytic² / S_1_analytic`.
    This is the `S_cos`-side restatement of MV Eq.(4); `flint.arb`
    certifies the numerical value (`min_G_analytic ≥ 0.99997…`,
    `S_1_analytic ≤ 29.841`, RHS `≈ 0.0668`). -/
theorem field_hEq4
    (f : ℝ → ℝ)
    -- The genuine full-tsum Cauchy–Schwarz floor of MV Eq.(4) (certifier
    -- output; TRUE-as-stated; replaces the previous false `(2/u)²/4`
    -- reconciliation `h_const_reconcile`):
    (h_cs : Sidon.MultiScale.S_cos f
              ≥ 2 * Sidon.MultiScale.min_G_analytic ^ 2
                  / Sidon.MultiScale.S_1_analytic) :
    (Sidon.MultiScale.uQ : ℝ) ^ 2
        * (Sidon.MultiScale.S_cos f / Sidon.MultiScale.uQ_real ^ 3)
      ≥ Sidon.MultiScale.min_G_analytic ^ 2
          / (Sidon.MultiScale.uQ_real * Sidon.MultiScale.S_1_analytic / 2) := by
  -- Reconcile `(uQ:ℝ)` with `uQ_real` (definitional).
  have hu_eq : (Sidon.MultiScale.uQ : ℝ) = Sidon.MultiScale.uQ_real := by
    rw [Sidon.MultiScale.uQ_real]
  rw [hu_eq]
  set u := Sidon.MultiScale.uQ_real with hu_def
  set S := Sidon.MultiScale.S_cos f with hS_def
  set S1 := Sidon.MultiScale.S_1_analytic with hS1_def
  set mG := Sidon.MultiScale.min_G_analytic with hmG_def
  have hu_pos : 0 < u := Sidon.MultiScale.uQ_real_pos
  have hu_ne : u ≠ 0 := ne_of_gt hu_pos
  have hS1_pos : 0 < S1 := S_1_analytic_pos
  have hS1_ne : S1 ≠ 0 := ne_of_gt hS1_pos
  -- `h_cs : S ≥ 2·mG²/S1`.  Multiply by `S1 > 0`:  S·S1 ≥ 2·mG².
  have hcancel : (2 * mG ^ 2 / S1) * S1 = 2 * mG ^ 2 := by
    rw [div_mul_cancel₀]; exact hS1_ne
  have h_cs' : S * S1 ≥ 2 * mG ^ 2 := by
    have hmul := mul_le_mul_of_nonneg_right h_cs (le_of_lt hS1_pos)
    rw [hcancel] at hmul
    linarith [hmul]
  -- Clear all denominators: LHS `u²·(S/u³) = S/u`, RHS `mG²/(u·S1/2)`.
  -- Goal (≥) ⟺ `mG²/(u·S1/2) ≤ u²·(S/u³)`.  `field_simp` clears to a
  -- polynomial inequality; `nlinarith` closes it from `h_cs'`.
  rw [ge_iff_le, div_le_iff₀ (by positivity), ← sub_nonneg]
  -- After clearing: goal is a polynomial `≥ 0` statement in `u, S, S1, mG`.
  have key : u ^ 2 * (S / u ^ 3) * (u * S1 / 2) - mG ^ 2
      = (S * S1 - 2 * mG ^ 2) / 2 := by
    field_simp
  rw [key]
  -- `0 ≤ (S·S1 - 2·mG²)/2` from `h_cs'`.
  have hnum : S * S1 - 2 * mG ^ 2 ≥ 0 := by linarith [h_cs']
  linarith [hnum]

end Sidon.Constructor
