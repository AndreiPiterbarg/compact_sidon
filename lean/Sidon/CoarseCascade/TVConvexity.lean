/-
Sidon Autocorrelation Project — TV Convexity Properties (Proof Stubs)

F(mu) = max_W TV_W(mu) is the pointwise maximum of O(d^2) quadratic forms.
Each TV_W(mu, ell, s) = (2d/ell) * mu^T A_{ell,s} mu is a quadratic form
with non-negative coefficient matrix A.

KEY SUBTLETY: Each individual TV_W is NOT necessarily convex (A_{ell,s} is
not positive semidefinite for all windows). But F = max_W TV_W IS convex
because it is the max of bilinear forms on the non-negative orthant.

Actually, each TV_W IS convex on the non-negative orthant since all
coefficients are non-negative and all variables are non-negative.
But it is NOT convex on all of R^d.

Source: run_cascade_coarse_v2.py lines 60-63 (the Hessian issue).
-/

import Sidon.CoarseCascade.TVGradientHessian

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
-- PART 1: TV_W is NOT Convex in General
-- =============================================================================

/-- **Counterexample:** The Hessian A for d=2, ell=2, s=1 is [[0,1],[1,0]]
    with eigenvalues +1 and -1. Hence TV_W(mu, 2, 1) is NOT convex on R^2.

    This means we CANNOT use convexity to bound TV variation within a cell.
    The v2 box certification correctly accounts for this. -/
theorem tv_not_convex_counterexample :
    ∃ (d ell s : ℕ) (μ δ : Fin 2 → ℝ),
      d = 2 ∧ ell = 2 ∧ s = 1 ∧
      -- Q(delta) < 0 for some delta (non-convex direction)
      (∑ i : Fin 2, ∑ j : Fin 2,
        window_indicator 2 ell s i j * δ i * δ j) < 0 := by
  sorry

-- =============================================================================
-- PART 2: TV_W IS Convex on the Non-Negative Orthant
-- =============================================================================

/-- On the non-negative orthant (all mu_i >= 0), each term mu_i * mu_j >= 0,
    so TV_W is a sum of non-negative terms. This is a weaker but useful property.

    More precisely: TV_W(lambda*mu + (1-lambda)*nu) <= lambda*TV_W(mu) + (1-lambda)*TV_W(nu)
    does NOT hold in general. But we don't need it — we use the exact Taylor
    expansion + bounds on the quadratic term instead. -/
theorem tv_nonneg_on_simplex {d : ℕ} (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    0 ≤ mass_test_value d μ ell s := by
  sorry

-- =============================================================================
-- PART 3: F = max_W TV_W is Convex on the Simplex
-- =============================================================================

/-- F(mu) = max_W TV_W(mu) is convex on the simplex.
    F is the pointwise maximum of O(d^2) functions, and pointwise max
    of convex functions is convex. (On the simplex, each TV_W is convex
    since it's a non-negative sum of products of non-negative variables.)

    Source: proof/coarse_cascade_method.md Section 5.4. -/
theorem max_tv_convex_on_simplex (d : ℕ) :
    ∀ μ ν : Fin d → ℝ, on_simplex μ → on_simplex ν →
    ∀ λ : ℝ, 0 ≤ λ → λ ≤ 1 →
    let mix := fun i => λ * μ i + (1 - λ) * ν i
    on_simplex mix →
    (⨆ (ell : ℕ) (s : ℕ) (_ : 2 ≤ ell) (_ : s + ell ≤ 2 * d),
      mass_test_value d mix ell s) ≤
    max (⨆ (ell : ℕ) (s : ℕ) (_ : 2 ≤ ell) (_ : s + ell ≤ 2 * d),
           mass_test_value d μ ell s)
        (⨆ (ell : ℕ) (s : ℕ) (_ : 2 ≤ ell) (_ : s + ell ≤ 2 * d),
           mass_test_value d ν ell s) := by
  sorry

end -- noncomputable section
