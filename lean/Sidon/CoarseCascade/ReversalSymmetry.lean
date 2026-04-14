/-
Sidon Autocorrelation Project — Reversal Symmetry for Coarse Grid (Proof Stubs)

For the coarse cascade, canonical form c <= rev(c) lexicographically halves
the search space. This is sound because TV_W is invariant under reversal
of the mass vector (with corresponding window index reflection).

Source: run_cascade_coarse.py L0 uses generate_canonical_compositions_batched.
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

/-- Reversal of a vector: rev(mu)_i = mu_{d-1-i}. -/
def vec_reverse {d : ℕ} (μ : Fin d → ℝ) : Fin d → ℝ :=
  fun i => μ ⟨d - 1 - i.val, by omega⟩

-- =============================================================================
-- PART 1: Autoconvolution Under Reversal
-- =============================================================================

/-- Discrete autoconvolution is symmetric under reversal with index reflection:
    DA(rev(mu), k) = DA(mu, 2(d-1) - k).

    This follows from the substitution i' = (d-1-i), j' = (d-1-j),
    so i'+j' = 2(d-1) - (i+j). -/
theorem autoconv_reverse {d : ℕ} (μ : Fin d → ℝ) (k : ℕ) :
    discrete_autoconvolution (vec_reverse μ) k =
    discrete_autoconvolution μ (2 * (d - 1) - k) := by
  sorry

-- =============================================================================
-- PART 2: TV Under Reversal
-- =============================================================================

/-- TV_W is invariant under reversal (with reflected window):
    TV(rev(mu), ell, s) = TV(mu, ell, 2(d-1) - (s + ell - 2))

    In particular, max_W TV_W(rev(mu)) = max_W TV_W(mu). -/
theorem tv_reverse_invariant {d : ℕ} (μ : Fin d → ℝ) (ell s : ℕ) :
    mass_test_value d (vec_reverse μ) ell s =
    mass_test_value d μ ell (2 * (d - 1) - (s + ell - 2)) := by
  sorry

/-- The maximum over all windows is the same for mu and rev(mu). -/
theorem max_tv_reverse_eq {d : ℕ} (μ : Fin d → ℝ) (c_target : ℝ) :
    (∃ ell s, 2 ≤ ell ∧ mass_test_value d μ ell s ≥ c_target) ↔
    (∃ ell s, 2 ≤ ell ∧ mass_test_value d (vec_reverse μ) ell s ≥ c_target) := by
  sorry

-- =============================================================================
-- PART 3: Canonical Form Soundness
-- =============================================================================

/-- **Canonical form:** restricting enumeration to c <= rev(c) lexicographically
    does not miss any unpruned compositions. Every composition is either
    canonical or its reversal is canonical (or both, if palindromic).

    Source: run_cascade_coarse.py uses _canonical_mask from pruning.py. -/
theorem canonical_form_complete {d : ℕ} (S : ℕ)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S) (c_target : ℝ) :
    (∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s ≥ c_target) →
    (∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (vec_reverse (fun j => (c j : ℝ) / (S : ℝ))) i) ell s ≥
        c_target) := by
  sorry

end -- noncomputable section
