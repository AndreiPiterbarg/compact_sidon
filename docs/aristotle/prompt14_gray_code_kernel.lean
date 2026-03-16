/-
Prompt 14: Gray Code Kernel Soundness (Claims 4.9 + 4.10 + 4.11)

Attach complete_proof.lean as context.
NOTE: Claim 4.2 (incremental autoconv) is ALREADY PROVED in complete_proof.lean.

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
-- CLAIM 4.9: Gray code is a complete bijection
-- ═══════════════════════════════════════════════

theorem gray_code_bijection {k : ℕ} (r : Fin k → ℕ) (hr : ∀ i, 0 < r i) :
    ∃ (f : Fin (∏ i, r i) → (∀ i : Fin k, Fin (r i))),
      Function.Bijective f := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 4.10: Cross-term split for arbitrary position
-- The cross-terms for bins before and after (2p, 2p+1) cover all q ∉ {2p, 2p+1}
-- ═══════════════════════════════════════════════

theorem cross_term_split {d : ℕ} (p : ℕ) (hp : 2*p+1 < d)
    (f : Fin d → ℤ) :
    (∑ q : Fin d, if q.1 ≠ 2*p ∧ q.1 ≠ 2*p+1 then f q else 0) =
    (∑ q ∈ Finset.range (2*p), f ⟨q, by omega⟩) +
    (∑ q ∈ Finset.Ico (2*p+2) d, f ⟨q, by omega⟩) := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 4.11: W_int correctness under Gray code updates
-- ═══════════════════════════════════════════════

theorem w_int_gray_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if (2*p+1) ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  sorry

end
