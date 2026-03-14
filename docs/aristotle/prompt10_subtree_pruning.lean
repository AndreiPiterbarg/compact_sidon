/-
PROMPT FOR ARISTOTLE: Prove subtree pruning soundness (Claim 4.4).

Setup: In the odometer iteration, on a deep carry, we check if the ENTIRE
subtree of remaining children (sharing the same "fixed" prefix bins) can
be pruned at once.

Fixed bins: child[0], ..., child[2p-1] (same for all children in subtree).
Unfixed bins: child[2p], ..., child[d-1] (vary across subtree).

Three inequalities establish soundness:

1. ws_full(c') ≥ ws_partial  (adding nonneg terms can only increase the sum)
2. W_int(c') ≤ W_int_max     (unfixed child bins bounded by parent bin mass)
3. dyn_it(W) is non-decreasing in W  (floor of increasing linear function)

Chain: If ws_partial > dyn_it(W_int_max), then for any child c':
  ws_full(c') ≥ ws_partial > dyn_it(W_int_max) ≥ dyn_it(W_int(c'))
so the child would be pruned by the full scan.

All three inequalities and the chain are self-contained algebra/real analysis.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

section

/-! ## ============================================================
    INEQUALITY 1: Partial autoconvolution ≤ full autoconvolution
    ============================================================ -/

/-
For nonneg c and any cutoff p:
  ∑_{i+j=t, i<2p, j<2p} c_i·c_j ≤ ∑_{i+j=t} c_i·c_j

Proof: The LHS is a sub-sum of the RHS. All terms c_i·c_j ≥ 0 (since c_i ≥ 0).
The RHS includes additional terms where i ≥ 2p or j ≥ 2p, all ≥ 0.
-/
theorem partial_conv_le_full {d : ℕ} (c : Fin d → ℕ) (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0 ≤
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0 := by
  sorry

/-
Summing over a window: ws_partial ≤ ws_full.
-/
theorem partial_window_le_full {d : ℕ} (c : Fin d → ℕ) (p : ℕ) (hp : 2 * p ≤ d)
    (s_lo ℓ : ℕ) :
    ∑ t ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      (∑ i : Fin d, ∑ j : Fin d,
        if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0) ≤
    ∑ t ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      (∑ i : Fin d, ∑ j : Fin d,
        if i.1 + j.1 = t then c i * c j else 0) := by
  sorry

/-! ## ============================================================
    INEQUALITY 3: dyn_it is non-decreasing in W
    ============================================================ -/

/-
dyn_it(W) = ⌊(base + 2W) · scale⌋ where base, scale > 0.

(base + 2W) · scale is strictly increasing in W (coefficient 2·scale > 0).
⌊·⌋ is non-decreasing.
Composition of increasing and non-decreasing is non-decreasing.
-/
theorem dyn_it_monotone (base scale : ℝ) (hbase : 0 ≤ base) (hscale : 0 < scale)
    (W₁ W₂ : ℝ) (hW : W₁ ≤ W₂) :
    ⌊(base + 2 * W₁) * scale⌋ ≤ ⌊(base + 2 * W₂) * scale⌋ := by
  sorry

/-! ## ============================================================
    INEQUALITY 2: W_int bounded by W_int_max
    ============================================================ -/

/-
For each unfixed parent position q ≥ p:
  child[2q] + child[2q+1] = parent[q]   (mass preservation)
  child[2q] ≥ 0, child[2q+1] ≥ 0       (nonneg)

If bin 2q is in the contributing range [lo, hi]:
  child[2q] ≤ child[2q] + child[2q+1] = parent[q]

If both 2q and 2q+1 are in range:
  child[2q] + child[2q+1] = parent[q]    (exact)

So W_int = ∑_{i ∈ [lo,hi]} child[i] ≤ W_int_max where W_int_max uses
parent[q] for each unfixed position q with any child bin in range.
-/
theorem w_int_upper_bound (d : ℕ) (p : ℕ) (hp : p < d)
    (child : Fin (2 * d) → ℕ) (parent : Fin d → ℕ)
    (h_split : ∀ q : Fin d, q.1 ≥ p →
      child ⟨2 * q.1, by omega⟩ + child ⟨2 * q.1 + 1, by omega⟩ = parent q) :
    ∀ (q : Fin d), q.1 ≥ p →
      child ⟨2 * q.1, by omega⟩ ≤ parent q ∧
      child ⟨2 * q.1 + 1, by omega⟩ ≤ parent q := by
  sorry

/-! ## ============================================================
    CHAIN CONCLUSION: Subtree pruning is sound
    ============================================================ -/

/-
If ws_partial > dyn_it(W_max), then for any child c' in the subtree:
  ws_full(c') ≥ ws_partial > dyn_it(W_max) ≥ dyn_it(W_actual(c'))

So ws_full(c') > dyn_it(W_actual(c')), meaning the full scan would prune c'.
Therefore pruning the entire subtree is sound.
-/
theorem subtree_pruning_sound
    (ws_partial ws_full : ℤ) (dyn_it_wmax dyn_it_wactual : ℤ)
    (h1 : ws_full ≥ ws_partial)           -- Inequality 1
    (h2 : dyn_it_wmax ≥ dyn_it_wactual)   -- Inequality 3 applied to W_max ≥ W_actual
    (h3 : ws_partial > dyn_it_wmax)        -- The subtree pruning condition
    : ws_full > dyn_it_wactual := by       -- Conclusion: full scan would also prune
  omega

end
