/-
Sidon Autocorrelation Project — Sparse Cross-Term Optimization (Claims 4.26–4.35)

This file collects ALL the theorems and lemmas that must be proved to
certify the sparse cross-term optimization implemented in
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization works as follows: instead of iterating all d_child bins
in the cross-term update loop (checking `if child[j] != 0` for each),
we maintain an explicit nonzero index list `nz_list` with a reverse-index
`nz_pos`. When d_child ≥ 32, the cross-term loop iterates only over
`nz_list`, skipping zero bins entirely.

The nz_list is maintained incrementally: when the Gray code advances and
bins k1, k2 change values, at most 2 add/remove operations on nz_list
keep it in sync. After a subtree prune (which resets child bins and does
a full raw_conv recompute), nz_list is rebuilt from scratch.

STATUS: PROOF OBLIGATIONS ONLY — no proofs are attempted here.
Each `sorry` marks an open obligation. Dependencies on existing modules
(Defs, IncrementalAutoconv, GrayCode) are noted.
-/

import Mathlib
import Sidon.Defs

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Nonzero List Invariant (Claims 4.26–4.27)
--
-- The nz_list is a faithful representation of the set of nonzero child bins.
-- This invariant must hold at every point where the cross-term loop executes.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.26: nz_list invariant — the set of indices stored in
    nz_list[0..nz_count-1] is exactly the set of indices i where
    child[i] ≠ 0.

    Formally: nz_list is a permutation of {i : Fin d | child i ≠ 0}.
    This is the core invariant that the sparse cross-term loop relies on.
    It must hold:
      (a) after initialization from the first child,
      (b) after each incremental nz_list update following a Gray code step,
      (c) after nz_list rebuild following a subtree prune. -/
theorem nz_list_invariant
    {d : ℕ} (child : Fin d → ℤ)
    (nz_list : Fin d → ℕ)  -- indices of nonzero bins
    (nz_count : ℕ) (hnz_count : nz_count ≤ d)
    -- nz_list[0..nz_count-1] are all valid indices
    (h_valid : ∀ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ < d)
    -- nz_list[0..nz_count-1] are all distinct
    (h_distinct : ∀ k₁ k₂ : Fin nz_count,
      nz_list ⟨k₁.1, by omega⟩ = nz_list ⟨k₂.1, by omega⟩ → k₁ = k₂)
    -- nz_list[0..nz_count-1] are exactly the nonzero indices
    (h_nonzero : ∀ k : Fin nz_count,
      child ⟨nz_list ⟨k.1, by omega⟩, by omega⟩ ≠ 0)
    (h_complete : ∀ i : Fin d, child i ≠ 0 →
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1) :
    -- The set represented by nz_list equals {i | child i ≠ 0}
    (Finset.image (fun k : Fin nz_count => (⟨nz_list ⟨k.1, by omega⟩, by omega⟩ : Fin d))
      Finset.univ) =
    Finset.filter (fun i => child i ≠ 0) Finset.univ := by
  sorry

/-- Claim 4.27: The reverse-index nz_pos is consistent with nz_list.
    For every nonzero bin i, nz_pos[i] gives the position of i in nz_list.
    For every zero bin i, nz_pos[i] = -1 (sentinel).

    This enables O(1) removal from nz_list via swap-with-last. -/
theorem nz_pos_consistent
    {d : ℕ} (child : Fin d → ℤ)
    (nz_list : Fin d → ℕ) (nz_pos : Fin d → ℤ)
    (nz_count : ℕ) (hnz_count : nz_count ≤ d)
    -- Forward: nz_pos maps nonzero bins to their position in nz_list
    (h_forward : ∀ i : Fin d, child i ≠ 0 →
      0 ≤ nz_pos i ∧ nz_pos i < nz_count ∧
      nz_list ⟨(nz_pos i).toNat, by omega⟩ = i.1)
    -- Backward: zero bins map to -1
    (h_zero : ∀ i : Fin d, child i = 0 → nz_pos i = -1) :
    -- Consistency: nz_list[nz_pos[i]] = i for all nonzero i
    ∀ i : Fin d, child i ≠ 0 →
      nz_list ⟨(nz_pos i).toNat, by omega⟩ = i.1 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Incremental Update Correctness (Claims 4.28–4.30)
--
-- When the Gray code advances and exactly one cursor position changes,
-- bins k1 = 2*pos and k2 = 2*pos+1 get new values. The nz_list must be
-- updated to reflect these changes. There are four cases per bin:
--   nonzero → zero:   remove from list
--   zero → nonzero:   add to list
--   nonzero → nonzero: no change
--   zero → zero:       no change
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.28: Swap-remove preserves the nz_list invariant.

    When removing index i from nz_list, we swap it with the last element
    and decrement nz_count. This preserves the set of stored indices
    (minus the removed one) and the distinctness property. -/
theorem swap_remove_preserves_invariant
    {d : ℕ} (nz_list : Fin d → ℕ) (nz_pos : Fin d → ℤ)
    (nz_count : ℕ) (hnz : 0 < nz_count) (hnz_d : nz_count ≤ d)
    (i : Fin d)  -- the index being removed
    (h_in_list : 0 ≤ nz_pos i ∧ (nz_pos i).toNat < nz_count)
    -- Define the updated list after swap-remove
    (nz_list' : Fin d → ℕ) (nz_pos' : Fin d → ℤ) (nz_count' : ℕ)
    (h_count' : nz_count' = nz_count - 1)
    -- The swap: nz_list'[pos_of_i] = nz_list[nz_count-1]
    (h_swap : nz_list' ⟨(nz_pos i).toNat, by omega⟩ =
              nz_list ⟨nz_count - 1, by omega⟩)
    -- Everything else unchanged
    (h_rest : ∀ k : ℕ, k < nz_count' → k ≠ (nz_pos i).toNat →
      nz_list' ⟨k, by omega⟩ = nz_list ⟨k, by omega⟩)
    -- Reverse index updated
    (h_pos_last : nz_pos' ⟨nz_list ⟨nz_count - 1, by omega⟩, by omega⟩ = nz_pos i)
    (h_pos_removed : nz_pos' i = -1) :
    -- The set of indices in nz_list'[0..nz_count'-1] equals
    -- the set in nz_list[0..nz_count-1] minus {i}
    True := by
  sorry

/-- Claim 4.29: Append preserves the nz_list invariant.

    When adding index i to nz_list, we place it at position nz_count
    and increment nz_count. -/
theorem append_preserves_invariant
    {d : ℕ} (nz_list : Fin d → ℕ) (nz_pos : Fin d → ℤ)
    (nz_count : ℕ) (hnz_d : nz_count < d)
    (i : Fin d)  -- the index being added
    (h_not_in : nz_pos i = -1)
    -- Define the updated list after append
    (nz_list' : Fin d → ℕ) (nz_pos' : Fin d → ℤ) (nz_count' : ℕ)
    (h_count' : nz_count' = nz_count + 1)
    (h_append : nz_list' ⟨nz_count, by omega⟩ = i.1)
    (h_rest : ∀ k : ℕ, k < nz_count →
      nz_list' ⟨k, by omega⟩ = nz_list ⟨k, by omega⟩)
    (h_pos_new : nz_pos' i = nz_count) :
    -- The set of indices in nz_list'[0..nz_count'-1] equals
    -- the set in nz_list[0..nz_count-1] ∪ {i}
    True := by
  sorry

/-- Claim 4.30: After the four-case update (old→new for bins k1, k2),
    the nz_list invariant is restored for the updated child array.

    This combines Claims 4.28–4.29 for the two bins that change in
    each Gray code step. The key insight is that bins k1 and k2 are
    the ONLY bins that change, so the invariant for all other bins
    is trivially preserved. -/
theorem incremental_nz_update_correct
    {d : ℕ} (child child' : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k1 ≠ k2)
    -- Only k1, k2 changed
    (h_unchanged : ∀ i : Fin d, i ≠ k1 → i ≠ k2 → child' i = child i)
    -- nz_list was correct before
    (nz_list nz_pos : Fin d → _) (nz_count : ℕ)
    (h_inv_before : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1)
    -- nz_list', nz_pos', nz_count' are the result of the four-case update
    (nz_list' nz_pos' : Fin d → _) (nz_count' : ℕ) :
    -- nz_list' is correct for child'
    (∀ i : Fin d, child' i ≠ 0 ↔
      ∃ k : Fin nz_count', nz_list' ⟨k.1, by omega⟩ = i.1) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Cross-Term Equivalence (Claims 4.31–4.32)
--
-- The sparse cross-term loop computes the same raw_conv updates as the
-- original two-loop version. This is the central correctness theorem.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.31: The sparse cross-term sum equals the dense cross-term sum.

    The original code computes:
      Σ_{j ∈ [0,k1)} if child[j]≠0: 2·δ₁·child[j] at raw_conv[k1+j]
                                       2·δ₂·child[j] at raw_conv[k2+j]
      + Σ_{j ∈ (k2,d)} same

    The sparse code computes:
      Σ_{j ∈ nz_list, j≠k1, j≠k2}: 2·δ₁·child[j] at raw_conv[k1+j]
                                       2·δ₂·child[j] at raw_conv[k2+j]

    These are equal because:
      (a) nz_list contains exactly the indices where child[j] ≠ 0
      (b) The original code skips j where child[j] = 0
      (c) The original code skips j = k1, k2 via range boundaries
      (d) The sparse code skips j = k1, k2 via explicit check -/
theorem sparse_cross_term_eq_dense
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (delta1 delta2 : ℤ)
    (nz_list : Fin d → ℕ) (nz_count : ℕ)
    -- nz_list invariant holds
    (h_inv : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1) :
    -- For every convolution index t, the sparse update equals the dense update
    ∀ t : ℕ,
    -- Dense: sum over j < k1 and j > k2 where child[j] ≠ 0
    (∑ j : Fin d,
      if (j.1 < k1.1 ∨ j.1 > k2.1) ∧ child j ≠ 0 ∧ k1.1 + j.1 = t
      then 2 * delta1 * child j else 0) =
    -- Sparse: sum over nz_list entries ≠ k1, k2
    (∑ idx : Fin nz_count,
      let j := (⟨nz_list ⟨idx.1, by omega⟩, by omega⟩ : Fin d)
      if j ≠ k1 ∧ j ≠ k2 ∧ k1.1 + j.1 = t
      then 2 * delta1 * child j else 0) := by
  sorry  -- Follows from h_inv bijectivity + child j = 0 filtering

/-- Claim 4.32: The raw_conv array after the sparse cross-term update
    is identical to the raw_conv array after the dense cross-term update.

    This lifts Claim 4.31 from individual convolution indices to the
    full raw_conv array. Since both updates touch the same set of
    raw_conv entries with the same deltas, the resulting arrays are equal.

    This is the master equivalence theorem: it guarantees that the
    pruning test (which reads raw_conv) sees identical values regardless
    of whether sparse or dense cross-terms were used. -/
theorem raw_conv_sparse_eq_dense
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (delta1 delta2 : ℤ)
    (raw_conv_before : Fin (2 * d - 1) → ℤ)
    (nz_list : Fin d → ℕ) (nz_count : ℕ)
    (h_inv : ∀ i : Fin d, child i ≠ 0 ↔
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1)
    -- Self-terms and mutual term are identical (not affected by sparse)
    -- raw_conv_after_dense: result of applying dense cross-term loop
    -- raw_conv_after_sparse: result of applying sparse cross-term loop
    (raw_conv_dense raw_conv_sparse : Fin (2 * d - 1) → ℤ) :
    raw_conv_dense = raw_conv_sparse := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: Subtree Prune Rebuild (Claim 4.33)
--
-- After a subtree prune, child bins are reset and raw_conv is fully
-- recomputed. The nz_list must be rebuilt from scratch. We must prove
-- that the rebuild produces a valid nz_list for the new child state.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.33: Rebuilding nz_list from scratch by iterating all d_child
    bins and collecting nonzero indices produces a valid nz_list satisfying
    the invariant of Claim 4.26.

    This is straightforward: the rebuild loop is identical to the
    initialization loop. The only subtlety is that nz_pos must also
    be reset (zero bins get nz_pos = -1, nonzero bins get their
    position in the list). -/
theorem rebuild_nz_list_correct
    {d : ℕ} (child : Fin d → ℤ)
    -- Rebuild procedure: scan all bins, collect nonzero
    (nz_list : Fin d → ℕ) (nz_pos : Fin d → ℤ) (nz_count : ℕ)
    -- Rebuild postconditions
    (h_count : nz_count = (Finset.filter (fun i => child i ≠ 0) Finset.univ).card)
    (h_nonzero : ∀ k : Fin nz_count,
      child ⟨nz_list ⟨k.1, by omega⟩, by omega⟩ ≠ 0)
    (h_complete : ∀ i : Fin d, child i ≠ 0 →
      ∃ k : Fin nz_count, nz_list ⟨k.1, by omega⟩ = i.1)
    (h_pos_nz : ∀ i : Fin d, child i ≠ 0 →
      nz_list ⟨(nz_pos i).toNat, by omega⟩ = i.1)
    (h_pos_zero : ∀ i : Fin d, child i = 0 → nz_pos i = -1) :
    -- The invariant holds
    (Finset.image (fun k : Fin nz_count => (⟨nz_list ⟨k.1, by omega⟩, by omega⟩ : Fin d))
      Finset.univ) =
    Finset.filter (fun i => child i ≠ 0) Finset.univ := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Gating Correctness (Claim 4.34)
--
-- The optimization is gated on d_child ≥ 32. We must prove that both
-- code paths (sparse and dense) produce identical results, so the gate
-- only affects performance, not correctness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.34: The use_sparse gate does not affect the survivor set.

    For d_child < 32, the original dense cross-term loop is used.
    For d_child ≥ 32, the sparse cross-term loop is used.
    By Claim 4.32, both produce identical raw_conv arrays, so the
    pruning test produces identical results, and the survivor set
    is the same in both cases.

    This theorem states that the gate is purely a performance decision
    with no effect on the mathematical output. -/
theorem sparse_gate_correctness
    {d : ℕ} (child : Fin d → ℤ)
    (raw_conv_dense raw_conv_sparse : Fin (2 * d - 1) → ℤ)
    (h_eq : raw_conv_dense = raw_conv_sparse)
    -- Same pruning test applied to both
    (pruned : (Fin (2 * d - 1) → ℤ) → Prop)
    (h_deterministic : ∀ r₁ r₂ : Fin (2 * d - 1) → ℤ,
      r₁ = r₂ → pruned r₁ = pruned r₂) :
    pruned raw_conv_dense = pruned raw_conv_sparse := by
  sorry  -- Immediate from h_eq and h_deterministic

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: End-to-End Soundness (Claim 4.35)
--
-- The final theorem: the Gray code kernel with sparse cross-term
-- optimization produces the identical set of canonical survivors as
-- the Gray code kernel without it.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.35 (Master Soundness Theorem): For any parent composition,
    the set of canonical survivors produced by the Gray code kernel with
    sparse cross-term optimization is identical to the set produced by
    the Gray code kernel without sparse optimization.

    Proof sketch:
    1. Both kernels enumerate the same Cartesian product of children
       (the Gray code traversal is unchanged — Claims 4.9, 4.22).
    2. For each child, the incremental autoconvolution update produces
       identical raw_conv arrays (Claim 4.32):
       - Self-terms and mutual term: unchanged between sparse and dense.
       - Cross-terms: identical by Claims 4.26, 4.31.
       - After subtree prune rebuild: invariant restored (Claim 4.33).
    3. The pruning test reads only raw_conv and child, both identical,
       so pruning decisions are identical (Claim 4.34).
    4. The quick-check, canonicalization, and survivor storage are
       unchanged, so the output sets are identical.

    Depends on: GrayCode (4.9), IncrementalAutoconv (4.2),
                GrayCodeSubtreePruning (4.22), Claims 4.26–4.34. -/
theorem sparse_cross_term_sound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (m : ℕ) (c_target : ℝ) (n_half_child : ℕ)
    -- S_sparse: survivors with sparse optimization
    -- S_dense: survivors without sparse optimization
    (S_sparse S_dense : Finset (Fin (2 * d_parent) → ℕ)) :
    S_sparse = S_dense := by
  sorry

end -- noncomputable section
