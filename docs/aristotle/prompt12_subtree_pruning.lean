/-
Prompt 12: Subtree Pruning Soundness (Claim 4.4)

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
-- Inequality 1: partial conv ≤ full conv
-- (restricting sum to i,j < 2p gives a subset of nonneg terms)
-- ═══════════════════════════════════════════════

theorem partial_conv_le_full_conv {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2*p ∧ j.1 < 2*p then c i * c j else 0 ≤
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0 := by
  sorry

-- ═══════════════════════════════════════════════
-- Inequality 2: W_int(c') ≤ W_int_max for all children in subtree
-- ═══════════════════════════════════════════════

theorem w_int_bounded {d : ℕ} (child : Fin d → ℕ) (parent : Fin (d/2) → ℕ)
    (p : ℕ) (hp : 2*p ≤ d)
    (h_split : ∀ q : Fin (d/2), q.1 ≥ p →
      child ⟨2*q.1, by omega⟩ + child ⟨2*q.1+1, by omega⟩ = parent q)
    (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi < d) :
    ∑ i ∈ Finset.Icc lo hi, (child ⟨i, by omega⟩ : ℕ) ≤
    (∑ i ∈ Finset.Icc lo (min hi (2*p-1)), (child ⟨i, by omega⟩ : ℕ)) +
    (∑ q ∈ Finset.filter (fun q => 2*q ≤ hi ∧ lo ≤ 2*q+1)
      (Finset.Icc p (d/2 - 1)), (parent ⟨q, by omega⟩ : ℕ)) := by
  sorry

-- ═══════════════════════════════════════════════
-- Inequality 3: dyn_it is non-decreasing in W
-- ═══════════════════════════════════════════════

theorem dyn_it_mono (base s : ℝ) (hs : 0 < s) (W1 W2 : ℝ) (hW : W1 ≤ W2) :
    ⌊(base + 2 * W1) * s⌋ ≤ ⌊(base + 2 * W2) * s⌋ := by
  apply Int.floor_le_floor
  apply mul_le_mul_of_nonneg_right
  · linarith
  · exact le_of_lt hs

-- ═══════════════════════════════════════════════
-- Chain: subtree pruning is sound
-- ═══════════════════════════════════════════════

theorem subtree_pruning_chain (ws_partial ws_full dyn_max dyn_actual : ℤ)
    (h1 : ws_full ≥ ws_partial)
    (h2 : ws_partial > dyn_max)
    (h3 : dyn_max ≥ dyn_actual) :
    ws_full > dyn_actual := by
  omega

end
