/-
Sidon Autocorrelation Project — TV Gradient and Hessian (Proof Stubs)

TV_W(mu) is a DEGREE-2 POLYNOMIAL in mu, so its Taylor expansion is exact:

  TV_W(mu + delta) = TV_W(mu) + grad(TV_W) . delta + (2d/ell) * Q(delta)

where Q(delta) = delta^T A delta, A_{ij} = 1_{s <= i+j <= s+ell-2}.

Key facts:
  - grad_i = (4d/ell) * sum_{j: s <= i+j <= s+ell-2} mu_j
  - H_{ij} = (4d/ell) * 1_{s <= i+j <= s+ell-2}  (constant Hessian)
  - Taylor expansion is EXACT (no remainder) since TV is quadratic.

Source: run_cascade_coarse_v2.py lines 56-72, proof/coarse_cascade_method.md Section 7.
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
-- PART 1: TV is a Quadratic Form
-- =============================================================================

/-- The window indicator matrix A_{ij} = 1 if s <= i+j <= s+ell-2, else 0. -/
def window_indicator (d ell s : ℕ) (i j : Fin d) : ℝ :=
  if s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2 then 1 else 0

/-- TV_W(mu, ell, s) = (2d/ell) * mu^T A mu where A is the window indicator matrix.
    This establishes TV as a quadratic form. -/
theorem tv_as_quadratic_form (d : ℕ) (μ : Fin d → ℝ) (ell s : ℕ) (hell : 2 ≤ ell) :
    mass_test_value d μ ell s =
    (2 * (d : ℝ) / (ell : ℝ)) *
      ∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * μ i * μ j := by
  sorry

-- =============================================================================
-- PART 2: Gradient of TV
-- =============================================================================

/-- The gradient of TV_W with respect to mu_i:
    grad_i(mu) = (4d/ell) * sum_{j: s <= i+j <= s+ell-2} mu_j

    This comes from differentiating the quadratic form: d/d(mu_i) [mu^T A mu] = 2*(A*mu)_i,
    and A is symmetric. -/
def tv_gradient (d : ℕ) (μ : Fin d → ℝ) (ell s : ℕ) (i : Fin d) : ℝ :=
  (4 * (d : ℝ) / (ell : ℝ)) *
    ∑ j : Fin d, if s ≤ i.val + j.val ∧ i.val + j.val ≤ s + ell - 2
                 then μ j else 0

/-- Gradient correctness: the directional derivative of TV_W in direction delta
    equals grad . delta. -/
theorem tv_gradient_correct (d : ℕ) (μ δ : Fin d → ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell) (t : ℝ) :
    let μ_plus := fun i => μ i + t * δ i
    (mass_test_value d μ_plus ell s - mass_test_value d μ ell s) / t =
    ∑ i : Fin d, tv_gradient d μ ell s i * δ i +
    t * (2 * (d : ℝ) / (ell : ℝ)) * ∑ i : Fin d, ∑ j : Fin d,
      window_indicator d ell s i j * δ i * δ j := by
  sorry

-- =============================================================================
-- PART 3: Hessian of TV (Constant)
-- =============================================================================

/-- The Hessian of TV_W is the CONSTANT matrix:
    H_{ij} = (4d/ell) * 1_{s <= i+j <= s+ell-2}

    This is independent of mu (TV is quadratic, so the Hessian is constant). -/
def tv_hessian (d ell s : ℕ) (i j : Fin d) : ℝ :=
  (4 * (d : ℝ) / (ell : ℝ)) * window_indicator d ell s i j

/-- The Hessian is symmetric: H_{ij} = H_{ji}. -/
theorem tv_hessian_symmetric (d ell s : ℕ) (i j : Fin d) :
    tv_hessian d ell s i j = tv_hessian d ell s j i := by
  sorry

/-- The Hessian has non-negative entries (but is NOT necessarily positive semidefinite).
    Counterexample: d=2, ell=2, s=1 gives A = [[0,1],[1,0]], eigenvalues +/-1. -/
theorem tv_hessian_entries_nonneg (d ell s : ℕ) (i j : Fin d) :
    0 ≤ tv_hessian d ell s i j := by
  sorry

-- =============================================================================
-- PART 4: Exact Taylor Expansion
-- =============================================================================

/-- **Exact Taylor expansion** for TV_W (quadratic, so no remainder term):

    TV_W(mu + delta) = TV_W(mu) + grad . delta + (2d/ell) * Q(delta)

    where Q(delta) = sum_{k in window} sum_{i+j=k} delta_i * delta_j.

    This is EXACT because TV_W is a degree-2 polynomial.

    Source: run_cascade_coarse_v2.py lines 57-59. -/
theorem tv_taylor_exact (d : ℕ) (μ δ : Fin d → ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell) :
    mass_test_value d (fun i => μ i + δ i) ell s =
    mass_test_value d μ ell s +
    (∑ i : Fin d, tv_gradient d μ ell s i * δ i) +
    (2 * (d : ℝ) / (ell : ℝ)) *
      (∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * δ i * δ j) := by
  sorry

/-- Gradient entries are non-negative when mu is on the simplex.
    grad_i = (4d/ell) * sum of non-negative masses. -/
theorem tv_gradient_nonneg (d : ℕ) (μ : Fin d → ℝ) (hμ : on_simplex μ)
    (ell s : ℕ) (i : Fin d) :
    0 ≤ tv_gradient d μ ell s i := by
  sorry

end -- noncomputable section
