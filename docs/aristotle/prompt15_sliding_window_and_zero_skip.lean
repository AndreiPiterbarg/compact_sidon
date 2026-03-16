/-
Prompt 15: Sliding-Window Scan and Zero-Bin Skipping (Claims 4.12 + 4.13)

Attach complete_proof.lean as context.

THEOREMS TO PROVE (fill in the sorry's)
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════
-- CLAIM 4.12: Sliding window equivalence
-- ═══════════════════════════════════════════════

-- Inductive step: W_{s+1} = W_s + A[s+n_cv] - A[s]
theorem sliding_window_step {N : ℕ} (A : Fin N → ℤ) (n_cv s : ℕ)
    (hs : s + n_cv < N)
    (W_s : ℤ) (hW : W_s = ∑ k ∈ Finset.Ico s (s + n_cv), A ⟨k, by omega⟩) :
    W_s + A ⟨s + n_cv, by omega⟩ - A ⟨s, by omega⟩ =
    ∑ k ∈ Finset.Ico (s + 1) (s + 1 + n_cv), A ⟨k, by omega⟩ := by
  sorry

-- Pruning decisions are identical (same ws values → same predicate)
theorem sliding_window_pruning_equiv {N : ℕ} (A : Fin N → ℤ)
    (n_cv : ℕ) (threshold : ℤ) (s : ℕ) (hs : s + n_cv ≤ N) :
    (∑ k ∈ Finset.Ico s (s + n_cv), A ⟨k, by omega⟩) > threshold ↔
    (∑ k ∈ Finset.Ico s (s + n_cv), A ⟨k, by omega⟩) > threshold :=
  Iff.rfl

-- ═══════════════════════════════════════════════
-- CLAIM 4.13: Zero-bin skipping preserves autoconvolution
-- ═══════════════════════════════════════════════

-- Zero term vanishes
theorem zero_term_vanishes (a b : ℤ) (hb : b = 0) : a * b = 0 := by
  subst hb; ring

-- Filtering out c_j = 0 terms doesn't change a sum of products
theorem sum_filter_zero {d : ℕ} (c : Fin d → ℤ) (f : Fin d → ℤ) :
    ∑ j : Fin d, c j * f j =
    ∑ j ∈ (Finset.univ.filter fun j => c j ≠ 0), c j * f j := by
  apply Finset.sum_subset (Finset.filter_subset _ _) |>.symm
  intro j _ hj
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, not_not] at hj
  simp [hj]

-- Autoconvolution with zero-skip = full autoconvolution
theorem autoconv_zero_skip {d : ℕ} (c : Fin d → ℤ) (t : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0) =
    (∑ i ∈ (Finset.univ.filter fun i => c i ≠ 0),
      ∑ j ∈ (Finset.univ.filter fun j => c j ≠ 0),
        if i.1 + j.1 = t then c i * c j else 0) := by
  sorry

-- Cross-term zero-skip: exact for unchanged-bin cross-terms
theorem cross_term_zero_skip {d : ℕ} (c : Fin d → ℤ) (delta : ℤ)
    (S : Finset (Fin d)) :
    (∑ q ∈ S, delta * c q) =
    (∑ q ∈ S.filter (fun q => c q ≠ 0), delta * c q) := by
  apply Finset.sum_subset (Finset.filter_subset _ _) |>.symm
  intro q _ hq
  simp only [Finset.mem_filter, not_not] at hq
  simp [hq.2]

-- Counterexample: changed-pair cross-terms CANNOT be zero-skipped
example : (3 : ℤ) * 0 - 1 * 5 = -5 := by norm_num
example : (-5 : ℤ) ≠ 0 := by norm_num

end
