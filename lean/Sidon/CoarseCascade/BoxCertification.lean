/-
Sidon Autocorrelation Project — Box Certification Soundness (Proof Stubs)

The cascade verifies TV >= c at GRID POINTS (integer compositions c/S).
For continuous coverage, we need: for all mu in the Voronoi cell of c/S,
  max_W TV_W(mu) >= c_target.

Two approaches implemented in the Python code:

v1 (run_cascade_coarse.py):
  cell_var = max_{delta in cell} |grad . delta|
  UNSOUND: ignores second-order term (TV is NOT convex for all windows)

v2 (run_cascade_coarse_v2.py):
  Uses |TV(mu*+delta) - TV(mu*)| <= cell_var + quad_corr
  where quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4 S^2)
  SOUND: accounts for worst-case second-order contribution.

  Certification: margin = TV(mu*) - c_target > cell_var + quad_corr

Source: run_cascade_coarse_v2.py lines 49-101, proof/coarse_cascade_method.md Section 7.
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
-- PART 1: Voronoi Cell Definition
-- =============================================================================

/-- A point mu is in the Voronoi cell of grid point c/S on the simplex. -/
def in_voronoi_cell {d : ℕ} (c : Fin d → ℕ) (S : ℕ) (μ : Fin d → ℝ) : Prop :=
  (∀ i, |μ i - (c i : ℝ) / (S : ℝ)| < 1 / (S : ℝ)) ∧
  on_simplex μ

/-- The perturbation delta = mu - c/S satisfies |delta_i| < 1/S and sum delta_i = 0. -/
theorem voronoi_cell_delta_bound {d : ℕ} (c : Fin d → ℕ) (S : ℕ) (hS : S > 0)
    (μ : Fin d → ℝ) (h : in_voronoi_cell c S μ) (hc_sum : ∑ i, c i = S) :
    let δ := fun i => μ i - (c i : ℝ) / (S : ℝ)
    (∀ i, |δ i| < 1 / (S : ℝ)) ∧ (∑ i, δ i = 0) := by
  sorry

-- =============================================================================
-- PART 2: First-Order Cell Variation Bound
-- =============================================================================

/-- Cell variation for window (ell, s): the maximum of |grad . delta| over the
    Voronoi cell. Achieved by pairing extremes of sorted gradient.

    cell_var = (1/(2S)) * sum_{k=0..d/2-1} (grad_sorted[d-1-k] - grad_sorted[k])

    This uses the fact that |delta_i| <= h = 1/(2S) and sum delta_i = 0, so the
    maximum of a linear form over this polytope is achieved by the pairing
    of largest gradient with delta = +h and smallest with delta = -h.

    Source: run_cascade_coarse.py lines 129-145, run_cascade_coarse_v2.py. -/
theorem cell_var_bound {d : ℕ} (μ : Fin d → ℝ) (S : ℕ) (hS : S > 0)
    (ell s : ℕ) (hell : 2 ≤ ell)
    (δ : Fin d → ℝ) (hδ_bound : ∀ i, |δ i| ≤ 1 / (2 * (S : ℝ)))
    (hδ_sum : ∑ i, δ i = 0) :
    |∑ i : Fin d, tv_gradient d μ ell s i * δ i| ≤
    -- cell_var: pair sorted gradients with extremal delta
    (1 / (2 * (S : ℝ))) *
      ∑ i : Fin d, tv_gradient d μ ell s i -
      -- TODO: express as pairing of sorted gradient components
      0 := by
  sorry

-- =============================================================================
-- PART 3: Quadratic Form Decomposition
-- =============================================================================

/-- Number of ordered pairs (i,j) with i+j = k and 0 <= i,j < d. -/
def pair_count (d k : ℕ) : ℕ :=
  (Finset.filter (fun p : Fin d × Fin d => p.1.val + p.2.val = k) Finset.univ).card

/-- Number of pairs in window: N_W = sum_{k in window} pair_count(d, k). -/
def window_pair_count (d ell s : ℕ) : ℕ :=
  ∑ k ∈ Finset.Icc s (s + ell - 2), pair_count d k

/-- Number of self-terms in window: M_W = #{k in W : k even, k/2 < d}. -/
def window_self_count (d ell s : ℕ) : ℕ :=
  (Finset.filter (fun k => k % 2 = 0 ∧ k / 2 < d)
    (Finset.Icc s (s + ell - 2))).card

/-- Cross-pair count: cross_W = N_W - M_W. -/
def window_cross_count (d ell s : ℕ) : ℕ :=
  window_pair_count d ell s - window_self_count d ell s

-- =============================================================================
-- PART 4: Quadratic Correction Bound
-- =============================================================================

/-- Decomposition of Q(delta) = sum_{k in W} c_k into self-terms and cross-terms:
    Q = sum_{k in W} s_k + sum_{k in W} x_k
    where s_k = delta_{k/2}^2 >= 0 (self-term, k even, k/2 < d)
    and x_k = c_k - s_k (cross-terms).

    Source: run_cascade_coarse_v2.py lines 76-79. -/
theorem quadratic_decomposition {d : ℕ} (δ : Fin d → ℝ) (ell s : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d,
      window_indicator d ell s i j * δ i * δ j =
    -- self-terms (non-negative)
    (∑ k ∈ Finset.filter (fun k => k % 2 = 0 ∧ k / 2 < d)
      (Finset.Icc s (s + ell - 2)),
        δ ⟨k / 2, by sorry⟩ ^ 2) +
    -- cross-terms (can be negative)
    0 -- placeholder for cross-term sum
    := by
  sorry

/-- Direct bound on -Q: since self-terms are non-negative and
    |cross-terms| <= cross_W * h^2, we have -Q <= cross_W * h^2. -/
theorem quad_direct_bound {d : ℕ} (δ : Fin d → ℝ)
    (hδ : ∀ i, |δ i| ≤ h) (ell s : ℕ) :
    -(∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * δ i * δ j) ≤
    (window_cross_count d ell s : ℝ) * h ^ 2 := by
  sorry

/-- Complement bound: using total autoconvolution = (sum delta_i)^2 + terms,
    -Q <= (d^2 - N_W) * h^2 via the complement window. -/
theorem quad_complement_bound {d : ℕ} (δ : Fin d → ℝ)
    (hδ : ∀ i, |δ i| ≤ h) (ell s : ℕ) :
    -(∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * δ i * δ j) ≤
    ((d : ℝ) ^ 2 - (window_pair_count d ell s : ℝ)) * h ^ 2 := by
  sorry

/-- Tight quadratic correction: min of the two bounds.
    quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4 S^2)

    Source: run_cascade_coarse_v2.py lines 89-92. -/
theorem quad_corr_bound {d S : ℕ} (δ : Fin d → ℝ)
    (hδ : ∀ i, |δ i| ≤ 1 / (2 * (S : ℝ))) (ell s : ℕ) (hell : 2 ≤ ell) :
    |(2 * (d : ℝ) / (ell : ℝ)) *
      ∑ i : Fin d, ∑ j : Fin d,
        window_indicator d ell s i j * δ i * δ j| ≤
    (2 * (d : ℝ) / (ell : ℝ)) *
      min (window_cross_count d ell s : ℝ)
          ((d : ℝ) ^ 2 - (window_pair_count d ell s : ℝ)) /
      (4 * (S : ℝ) ^ 2) := by
  sorry

-- =============================================================================
-- PART 5: Box Certification Soundness
-- =============================================================================

/-- **Box Certification Theorem** (v2 — sound).

    If TV(mu*) - c_target > cell_var + quad_corr, then for ALL mu in the
    Voronoi cell of mu*, max_W TV_W(mu) >= c_target.

    By the exact Taylor expansion:
      TV(mu) = TV(mu*) + grad . delta + (2d/ell) * Q(delta)
    By triangle inequality:
      |TV(mu) - TV(mu*)| <= |grad . delta| + |(2d/ell) Q(delta)|
                         <= cell_var + quad_corr
    So TV(mu) >= TV(mu*) - cell_var - quad_corr > c_target.

    Source: run_cascade_coarse_v2.py line 101. -/
theorem box_certification_sound {d S : ℕ} (hS : S > 0)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S)
    (c_target : ℝ)
    (ell s : ℕ) (hell : 2 ≤ ell)
    (cell_var quad_corr : ℝ) (hcv : 0 ≤ cell_var) (hqc : 0 ≤ quad_corr)
    (h_var : ∀ δ : Fin d → ℝ,
      (∀ i, |δ i| ≤ 1 / (2 * (S : ℝ))) → (∑ i, δ i = 0) →
      |∑ i : Fin d, tv_gradient d (fun i => (c i : ℝ) / S) ell s i * δ i|
        ≤ cell_var)
    (h_quad : ∀ δ : Fin d → ℝ,
      (∀ i, |δ i| ≤ 1 / (2 * (S : ℝ))) →
      |(2 * (d : ℝ) / (ell : ℝ)) *
        ∑ i : Fin d, ∑ j : Fin d,
          window_indicator d ell s i j * δ i * δ j| ≤ quad_corr)
    (h_margin : mass_test_value d (fun i => (c i : ℝ) / S) ell s - c_target
                > cell_var + quad_corr) :
    ∀ μ : Fin d → ℝ, in_voronoi_cell c S μ →
      mass_test_value d μ ell s ≥ c_target := by
  sorry

/-- The cascade + box certification proves C_1a >= c_target.

    For every continuous mu in Delta_d:
    - mu lies in some Voronoi cell (by voronoi_coverage)
    - The grid point of that cell was handled by the cascade
    - If pruned: the TV at the grid point exceeds c_target, and by box
      certification, all mu in the cell also have TV >= c_target
    - If survived: impossible (cascade terminated with 0 survivors)

    Therefore max_W TV_W(mu) >= c_target for all mu, and by Theorem 1,
    C_1a >= c_target. -/
theorem cascade_plus_box_cert_proves_bound
    (d S : ℕ) (hd : d > 0) (hS : S > 0) (c_target : ℝ) (hct : 0 < c_target)
    (h_all_pruned : ∀ c : Fin d → ℕ, (∑ i, c i = S) →
      ∃ ell s, 2 ≤ ell ∧ mass_test_value d (fun i => (c i : ℝ) / S) ell s ≥ c_target)
    (h_box_cert : ∀ c : Fin d → ℕ, (∑ i, c i = S) →
      ∀ μ : Fin d → ℝ, in_voronoi_cell c S μ →
      ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target) :
    ∀ μ : Fin d → ℝ, on_simplex μ →
      ∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target := by
  sorry

end -- noncomputable section
