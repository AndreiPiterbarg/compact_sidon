/-
Sidon Autocorrelation Project — val(d) Monotonicity and Convergence (Proof Stubs)

val(d) = min_{mu in Delta_d} max_W TV_W(mu)

Properties (from proof/coarse_cascade_method.md Section 5):
  1. val(d) <= C_1a for all d
  2. val(d) is non-decreasing (assuming refinement monotonicity)
  3. val(d) -> C_1a as d -> infinity
  4. val(2) = 1
-/

import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-- val(d) = min_{mu in Delta_d} max_W TV_W(mu; d). -/
noncomputable def val_d (d : ℕ) : ℝ :=
  ⨅ (μ : Fin d → ℝ) (_ : on_simplex μ),
    ⨆ (ell : ℕ) (s : ℕ) (_ : 2 ≤ ell) (_ : s + ell ≤ 2 * d),
      mass_test_value d μ ell s

-- =============================================================================
-- PART 1: val(d) <= C_1a
-- =============================================================================

/-- val(d) is a lower bound on C_1a for every d.

    Proof: For any admissible f, its bin masses mu lie on Delta_d.
    By Theorem 1: R(f) >= TV_W(mu, ell, s) for all windows.
    So R(f) >= max_W TV_W(mu) >= val(d).
    Since f is arbitrary: C_1a >= val(d). -/
theorem val_le_C1a (d : ℕ) (hd : d > 0) :
    ∀ f : ℝ → ℝ,
      (∀ x, 0 ≤ f x) →
      (Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4)) →
      (MeasureTheory.integral MeasureTheory.volume f = 1) →
      (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤
        MeasureTheory.volume ≠ ⊤) →
      autoconvolution_ratio f ≥ val_d d := by
  sorry

-- =============================================================================
-- PART 2: val(d) is Non-Decreasing (Conditional on Refinement Monotonicity)
-- =============================================================================

/-- val(2d) >= val(d), assuming refinement monotonicity.

    Proof: The minimizer mu* of val(2d) is a 2d-dimensional mass vector.
    Its parent (summing adjacent pairs) is a d-dimensional mass vector mu_par.
    By refinement monotonicity: max_W TV(mu*; 2d) >= max_W TV(mu_par; d).
    Since val(d) <= max_W TV(mu_par; d): val(2d) >= val(d). -/
theorem val_monotone (d : ℕ) (hd : d > 0) :
    val_d (2 * d) ≥ val_d d := by
  sorry

-- =============================================================================
-- PART 3: val(2) = 1
-- =============================================================================

/-- val(2) = 1. The full window (ell=4) always gives TV = 1 (= (sum mu_i)^2).
    The uniform distribution (1/2, 1/2) achieves max_W TV = 1. -/
theorem val_two_eq_one : val_d 2 = 1 := by
  sorry

-- =============================================================================
-- PART 4: val(d) Converges to C_1a
-- =============================================================================

/-- As d -> infinity, val(d) -> C_1a.
    This follows from the fact that bin masses at fine resolution
    approximate the continuous function arbitrarily well. -/
theorem val_converges_to_C1a :
    Filter.Tendsto (fun n => val_d (2 ^ n)) Filter.atTop
      (nhds (⨅ (f : ℝ → ℝ)
        (_ : ∀ x, 0 ≤ f x)
        (_ : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
        (_ : MeasureTheory.integral MeasureTheory.volume f = 1),
        autoconvolution_ratio f)) := by
  sorry

end -- noncomputable section
