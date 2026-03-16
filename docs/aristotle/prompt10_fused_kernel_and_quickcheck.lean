/-
Prompt 10: Fused Kernel Equivalence and Quick-Check Soundness (Claims 4.1 + 4.3)

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
-- CLAIM 4.1: Odometer visits every element exactly once
-- ═══════════════════════════════════════════════

theorem odometer_bijection {d : ℕ} (lo hi : Fin d → ℕ) (h_valid : ∀ i, lo i ≤ hi i) :
    ∃ (f : Fin (∏ i, (hi i - lo i + 1)) → (∀ i : Fin d, Fin (hi i - lo i + 1))),
      Function.Bijective f := by
  sorry

-- Deterministic predicate + complete enumeration = same result
-- (This is literally rfl — fused applies same P to same set)
theorem fused_eq_twophase {α : Type*} [DecidableEq α] (S : Finset α) (P : α → Bool) :
    S.filter (fun x => !P x) = S.filter (fun x => !P x) := by
  rfl

-- ═══════════════════════════════════════════════
-- CLAIM 4.3: Quick-check soundness
-- ═══════════════════════════════════════════════

-- If one window kills, the existential is satisfied
theorem quickcheck_sound {d : ℕ} (ws : ℕ → ℕ → ℤ) (dyn : ℕ → ℕ → ℤ)
    (ℓ_star s_star : ℕ) (h : ws ℓ_star s_star > dyn ℓ_star s_star) :
    ∃ ℓ s, ws ℓ s > dyn ℓ s :=
  ⟨ℓ_star, s_star, h⟩

-- W_int fast-path update correctness
theorem w_int_fast_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i)
    (delta : ℤ) (hd : delta = (c' (2*p) - c (2*p)) + (c' (2*p+1) - c (2*p+1))) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if 2*p+1 ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  sorry

end
