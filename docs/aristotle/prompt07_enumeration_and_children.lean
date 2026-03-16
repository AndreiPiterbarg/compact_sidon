/-
Prompt 07: Composition Enumeration and Child Generation (Claims 3.1 + 3.2)

Attach complete_proof.lean as context.
NOTE: Claim 3.2c (child_preserves_total_mass) and child_bin_pair_sum are ALREADY PROVED
in complete_proof.lean. Focus on Claims 3.1 and 3.2a/3.2d.

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
-- CLAIM 3.1: Stars-and-bars composition count
-- ═══════════════════════════════════════════════

theorem composition_count (m d : ℕ) (hd : d > 0) :
    Finset.card (Finset.filter (fun c : Fin d → Fin (m + 1) =>
      ∑ i, (c i : ℕ) = m) Finset.univ) = Nat.choose (m + d - 1) (d - 1) := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 3.2a: Per-bin choice count for child generation
-- ═══════════════════════════════════════════════

theorem per_bin_choices (c_i x_cap : ℕ) :
    Finset.card (Finset.Icc (Nat.max 0 (c_i - x_cap)) (Nat.min c_i x_cap)) =
    Nat.min c_i x_cap - Nat.max 0 (c_i - x_cap) + 1 := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 3.2d: Child bin sum = parent (already in complete_proof.lean as
-- child_bin_pair_sum, included here for completeness)
-- ═══════════════════════════════════════════════

theorem child_bin_sum (d : ℕ) (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i)
    (i : Fin d) :
    child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i := by
  rw [hc_even, hc_odd]; simp [ha i]

end
