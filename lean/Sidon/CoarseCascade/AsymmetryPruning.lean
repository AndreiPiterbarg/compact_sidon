/-
Sidon Autocorrelation Project — Asymmetry Pruning for Coarse Grid (Proof Stubs)

If the left-half mass fraction >= sqrt(c/2), the autoconvolution peak
is already >= c from the left half alone. This prunes compositions
with extreme left-right imbalance.

On the coarse grid this is EXACT (no correction needed), unlike the
fine grid where the asymmetry threshold must account for discretization error.

Source: run_cascade_coarse.py asymmetry_prune_mask_coarse, pruning.py asymmetry_threshold.
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

-- =============================================================================
-- Asymmetry Bound (Coarse Grid — No Correction)
-- =============================================================================

/-- Left-half mass: sum of masses in bins [0, d/2). -/
def left_mass {d : ℕ} (μ : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, if i.val < d / 2 then μ i else 0

/-- **Asymmetry pruning theorem:**
    If left_mass(mu) >= sqrt(c_target / 2), then the ell=d window
    covering the left half's self-convolution gives TV >= c_target.

    Proof sketch:
      The autoconvolution of the left half has integral >= left_mass^2.
      The left-half self-convolution is supported on [-1/2, 0], which has
      length 1/2. By averaging: max >= left_mass^2 / (1/2) = 2 * left_mass^2.
      If left_mass >= sqrt(c/2), then 2 * left_mass^2 >= c.

    On the coarse grid, this bound is EXACT — no correction needed because
    we use bin masses directly (Theorem 1), not step-function approximations.

    Source: pruning.py asymmetry_threshold (line 26). -/
theorem asymmetry_prune_sound {d : ℕ} (hd : d > 0)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target)
    (h_left : left_mass μ ≥ Real.sqrt (c_target / 2)) :
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  sorry

/-- Symmetric case: if right_mass >= sqrt(c/2), same conclusion.
    By symmetry (reversal invariance of max TV). -/
theorem asymmetry_prune_right {d : ℕ} (hd : d > 0)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target)
    (h_right : 1 - left_mass μ ≥ Real.sqrt (c_target / 2)) :
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  sorry

/-- Only compositions with left_frac in (1 - sqrt(c/2), sqrt(c/2)) need
    full window scanning. This is the "needs_check" mask in the Python code. -/
theorem asymmetry_needs_check {d : ℕ} (hd : d > 0)
    (μ : Fin d → ℝ) (hμ : on_simplex μ) (c_target : ℝ) (hct : 0 < c_target) :
    (left_mass μ ≥ Real.sqrt (c_target / 2) ∨
     left_mass μ ≤ 1 - Real.sqrt (c_target / 2)) →
    ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  sorry

end -- noncomputable section
