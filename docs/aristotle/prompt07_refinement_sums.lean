/-
PROMPT FOR ARISTOTLE: Prove refinement sum preservation (Claims 4.6 + 3.2c).

These are pure ℕ sum algebra about parent-child bin relationships.

Setup: Parent has d = 2n bins with masses (c₀, ..., c_{d-1}) summing to m.
Child has 2d = 4n bins: child[2i] = aᵢ, child[2i+1] = cᵢ - aᵢ, where 0 ≤ aᵢ ≤ cᵢ.

CLAIM 3.2c: Children preserve total mass.
  ∑_{j=0}^{2d-1} child[j] = ∑_{i=0}^{d-1} (aᵢ + (cᵢ - aᵢ)) = ∑ cᵢ = m.

CLAIM 4.6: Left-half mass is invariant under refinement.
  Child's left half = bins 0..2n-1 = parent bins 0..n-1 expanded.
  ∑_{j=0}^{2n-1} child[j] = ∑_{i=0}^{n-1} (aᵢ + (cᵢ - aᵢ)) = ∑_{i=0}^{n-1} cᵢ.
  This is independent of the choice of aᵢ, so asymmetry can be checked once per parent.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

section

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
HELPER: Each parent bin's mass is preserved in its two child bins.

child[2i] + child[2i+1] = parent[i]

This is immediate from the definitions:
  child[2i] = a[i], child[2i+1] = parent[i] - a[i].
-/
theorem child_bin_pair_sum (d : ℕ) (hd : d > 0)
    (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i)
    (i : Fin d) :
    child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i := by
  sorry

/-
CLAIM 3.2c: Children preserve total mass.

If ∑ parent[i] = m, then ∑ child[j] = m.

Proof:
  ∑_{j=0}^{2d-1} child[j]
  = ∑_{i=0}^{d-1} (child[2i] + child[2i+1])     (reindex: pair consecutive bins)
  = ∑_{i=0}^{d-1} (a[i] + (parent[i] - a[i]))    (by definition)
  = ∑_{i=0}^{d-1} parent[i]                       (cancellation)
  = m.
-/
theorem child_preserves_total_mass (d : ℕ) (hd : d > 0) (m : ℕ)
    (parent : Fin d → ℕ) (hp : ∑ i, parent i = m)
    (a : Fin d → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j, child j = m := by
  sorry

/-
CLAIM 4.6: Left-half mass is invariant under refinement.

Parent at resolution d = 2n. Child at resolution 2d = 4n.
The child's left half spans bins 0..2n-1, which correspond to parent bins 0..n-1.

∑_{j=0}^{2n-1} child[j] = ∑_{i=0}^{n-1} (child[2i] + child[2i+1])
                         = ∑_{i=0}^{n-1} (a[i] + parent[i] - a[i])
                         = ∑_{i=0}^{n-1} parent[i]

This is INDEPENDENT of the choice of a[i], so the asymmetry check
(which only depends on the left-half sum) can be done once per parent.
-/
theorem left_half_sum_invariant (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_even : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin (2 * n), child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) =
    ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry

/-
COROLLARY: For ANY two different refinements a₁, a₂ of the same parent,
the left-half sums of the corresponding children are equal.
-/
theorem left_half_sum_same_for_all_children (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a₁ a₂ : Fin (2 * n) → ℕ)
    (ha₁ : ∀ i, a₁ i ≤ parent i) (ha₂ : ∀ i, a₂ i ≤ parent i)
    (child₁ child₂ : Fin (4 * n) → ℕ)
    (hc₁_even : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1, by omega⟩ = a₁ i)
    (hc₁_odd : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₁ i)
    (hc₂_even : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1, by omega⟩ = a₂ i)
    (hc₂_odd : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₂ i) :
    ∑ j : Fin (2 * n), (child₁ ⟨j.1, by omega⟩ : ℕ) =
    ∑ j : Fin (2 * n), (child₂ ⟨j.1, by omega⟩ : ℕ) := by
  sorry

end
