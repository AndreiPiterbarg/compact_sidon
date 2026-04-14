/-
Sidon Autocorrelation Project — Subtree Pruning Soundness (Proof Stubs)

When cursors a_0, ..., a_pos are assigned (bins 0..2*pos+1 fixed),
the partial autoconvolution of assigned bins is a LOWER BOUND on the
full autoconvolution (since all masses are non-negative).

If ws_partial > thr[ell] for any window fully within the assigned range,
then the full window sum will also exceed the threshold, and the entire
subtree below can be pruned.

This is the key optimization making the cascade tractable.

Source: proof/coarse_cascade_method.md Section 6.5.
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
-- PART 1: Partial Autoconvolution Lower Bound
-- =============================================================================

/-- Partial autoconvolution: only sum over assigned bins [0, p). -/
def partial_autoconvolution {d : ℕ} (a : Fin d → ℝ) (p : ℕ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d,
    if i.val < p ∧ j.val < p ∧ i.val + j.val = k then a i * a j else 0

/-- **Partial conv is a lower bound on full conv** when all entries are non-negative.

    For each k: partial_autoconvolution(a, p, k) <= discrete_autoconvolution(a, k)

    Proof: the full sum includes all terms from the partial sum, plus additional
    terms a_i * a_j where at least one of i,j >= p. Since a_i >= 0 for all i,
    these additional terms are non-negative. -/
theorem partial_conv_le_full {d : ℕ} (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (k : ℕ) :
    partial_autoconvolution a p k ≤ discrete_autoconvolution a k := by
  sorry

/-- Windowed partial sum is a lower bound on windowed full sum. -/
theorem partial_window_sum_le_full {d : ℕ} (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (ell s : ℕ) :
    ∑ k ∈ Finset.Icc s (s + ell - 2), partial_autoconvolution a p k ≤
    ∑ k ∈ Finset.Icc s (s + ell - 2), discrete_autoconvolution a k := by
  sorry

-- =============================================================================
-- PART 2: Subtree Pruning Soundness
-- =============================================================================

/-- **Subtree Pruning Theorem:**
    If the partial autoconvolution of assigned bins [0, p) already
    exceeds the threshold for some window, then the FULL autoconvolution
    also exceeds the threshold, regardless of how bins [p, d) are assigned.

    This justifies pruning the entire subtree of the search tree. -/
theorem subtree_pruning_sound {d : ℕ}
    (c_target : ℝ)
    (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i)
    (p : ℕ) (hp : p ≤ d)
    (ell s : ℕ) (hell : 2 ≤ ell)
    (h_partial_exceeds :
      mass_test_value d (fun i => if i.val < p then a i else 0) ell s ≥ c_target) :
    -- For ANY assignment of bins [p, d):
    ∀ b : Fin d → ℝ,
      (∀ i, 0 ≤ b i) →
      (∀ i : Fin d, i.val < p → b i = a i) →
      mass_test_value d b ell s ≥ c_target := by
  sorry

-- =============================================================================
-- PART 3: Incremental Convolution Update
-- =============================================================================

/-- When position pos is assigned (bins k1=2*pos, k2=2*pos+1 go from 0 to
    their values), the convolution changes by:

    Self-terms:  conv[2*k1] += c_{k1}^2,  conv[2*k2] += c_{k2}^2
    Mutual:      conv[k1+k2] += 2 * c_{k1} * c_{k2}
    Cross-terms: for j < k1: conv[k1+j] += 2*c_{k1}*c_j, conv[k2+j] += 2*c_{k2}*c_j

    This is the incremental update used in the DFS-based cascade.

    Source: proof/coarse_cascade_method.md Section 6.5. -/
theorem incremental_conv_update {d : ℕ} (a : Fin d → ℝ)
    (pos : ℕ) (hpos : 2 * pos + 1 < d)
    (a_old a_new : Fin d → ℝ)
    (h_same : ∀ i : Fin d, i.val ≠ 2 * pos ∧ i.val ≠ 2 * pos + 1 → a_new i = a_old i)
    (k : ℕ) :
    discrete_autoconvolution a_new k - discrete_autoconvolution a_old k =
    -- Self-terms (if k = 2*(2*pos) or k = 2*(2*pos+1))
    -- + mutual term (if k = (2*pos) + (2*pos+1))
    -- + cross-terms (sum over other bins)
    -- This decomposes the update into O(d) operations
    sorry := by
  sorry

-- =============================================================================
-- PART 4: Incremental Update Correctness (Undo on Backtrack)
-- =============================================================================

/-- Undo property: subtracting the same delta restores the old convolution.
    This justifies the backtracking in the DFS cascade. -/
theorem incremental_conv_undo {d : ℕ} (a_old a_new : Fin d → ℝ) (k : ℕ) :
    let delta := discrete_autoconvolution a_new k - discrete_autoconvolution a_old k
    discrete_autoconvolution a_new k - delta = discrete_autoconvolution a_old k := by
  simp [sub_sub_cancel]

end -- noncomputable section
