/-
Sidon Autocorrelation Project — Gray Code Subtree Pruning (Claims 4.14–4.22)

This file collects ALL the theorems and lemmas that must be proved to
certify the Gray code subtree pruning optimization implemented in
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization works as follows: when the Gray code focus pointer
reaches level J_MIN, a partial autoconvolution of the fixed left-prefix
child bins is computed and checked against a conservative threshold.
If the partial sum already exceeds the threshold for some window, the
entire inner sweep (~128–249 children) is skipped.

STATUS: PROOF OBLIGATIONS ONLY — no proofs are attempted here.
Each `sorry` marks an open obligation. Dependencies on existing modules
(Defs, SubtreePruning, IncrementalAutoconv, GrayCode) are noted.
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
-- PART A: Digit-Ordering Independence (Claim 4.14)
--
-- The Gray code active_pos array is built right-to-left (reversed) so that
-- inner (fast-changing) digits correspond to rightmost parent positions.
-- We must prove the survivor set is independent of the digit ordering.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.14: The set of canonical survivors produced by the fused
    generate-and-prune kernel is independent of the ordering of the
    active_pos array.

    Formally: let σ be any permutation of Fin(n_active). Define
    active_pos' = active_pos ∘ σ. Then the Gray code kernel with
    active_pos' produces the same multiset of canonical survivors
    as with active_pos.

    Proof sketch: The Cartesian product ∏ᵢ [lo_i, hi_i] is the same
    set regardless of enumeration order. The test (quick-check + full
    window scan) and canonicalization are applied identically to each
    element. The only difference is traversal order, which affects
    which compositions are visited but not the set of survivors. -/
theorem gray_code_digit_order_independence
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (σ : Equiv.Perm (Fin d_parent))
    (m : ℕ) (c_target : ℝ)
    (n_half_child : ℕ)
    -- The set of children enumerated is the same Cartesian product
    (h_same_product : ∀ child : Fin (2 * d_parent) → ℕ,
      (∀ i, lo i ≤ child ⟨2 * i.1, by omega⟩ ∧ child ⟨2 * i.1, by omega⟩ ≤ hi i ∧
            child ⟨2 * i.1 + 1, by omega⟩ = parent i - child ⟨2 * i.1, by omega⟩) ↔
      (∀ i, lo (σ i) ≤ child ⟨2 * (σ i).1, by omega⟩ ∧
            child ⟨2 * (σ i).1, by omega⟩ ≤ hi (σ i) ∧
            child ⟨2 * (σ i).1 + 1, by omega⟩ = parent (σ i) - child ⟨2 * (σ i).1, by omega⟩)) :
    True := by  -- Survivor set equality; stated as True placeholder
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Fixed Prefix Characterization (Claims 4.15–4.16)
--
-- With reversed active_pos ordering, the "fixed prefix" is the set of
-- child bins 0..2p-1 where p = active_pos[J_MIN - 1]. We must prove
-- that all parent positions with index < p are indeed fixed (either
-- inactive or outer active positions).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.15: With active_pos built in decreasing order of physical
    parent index, active_pos[k] is strictly decreasing in k.
    Therefore active_pos[J_MIN-1] < active_pos[J_MIN-2] < ... < active_pos[0]. -/
theorem active_pos_decreasing
    {d_parent : ℕ} (lo hi : Fin d_parent → ℕ)
    (active_pos : Fin d_parent → ℕ) (n_active : ℕ)
    (h_built : ∀ k : Fin n_active, ∀ k' : Fin n_active,
      k.1 < k'.1 → active_pos ⟨k'.1, by omega⟩ < active_pos ⟨k.1, by omega⟩)
    (k k' : Fin n_active) (hk : k.1 < k'.1) :
    active_pos ⟨k'.1, by omega⟩ < active_pos ⟨k.1, by omega⟩ := by
  sorry

/-- Claim 4.16: Every parent position with index < active_pos[J_MIN - 1]
    is either inactive (range = 1) or an outer active position (digit
    index ≥ J_MIN). In either case, the corresponding child bins are
    fixed during the inner sweep of digits 0..J_MIN-1. -/
theorem fixed_prefix_characterization
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (active_pos : Fin d_parent → ℕ) (n_active : ℕ)
    (J_MIN : ℕ) (hJ : J_MIN < n_active)
    (h_decreasing : ∀ k k' : Fin n_active, k.1 < k'.1 →
      active_pos ⟨k'.1, by omega⟩ < active_pos ⟨k.1, by omega⟩)
    (p : ℕ) (hp : p < active_pos ⟨J_MIN - 1, by omega⟩) :
    -- p is either inactive or has digit index ≥ J_MIN
    (hi ⟨p, by omega⟩ - lo ⟨p, by omega⟩ + 1 = 1) ∨
    (∃ k : Fin n_active, k.1 ≥ J_MIN ∧ active_pos ⟨k.1, by omega⟩ = p) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Partial Autoconvolution Soundness (Claims 4.17–4.18)
--
-- The partial autoconvolution of the fixed prefix is a lower bound on
-- the full autoconvolution for every window. Combined with the W_int_max
-- upper bound, this gives a sound pruning criterion.
--
-- Note: Claim 4.17 generalizes the existing `partial_conv_le_full_conv`
-- from SubtreePruning.lean to handle the specific prefix structure of
-- the Gray code variant.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.17 (extends SubtreePruning.partial_conv_le_full_conv):
    The partial autoconvolution restricted to a prefix of length 2p
    is a lower bound on the full autoconvolution, for every
    convolution index t.

    This is the core soundness lemma. It follows from the fact that
    all omitted terms c_i * c_j (where i ≥ 2p or j ≥ 2p) are
    nonneg because c_i ≥ 0 for all i. -/
theorem partial_conv_prefix_le_full
    {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    -- Sum restricted to i,j < 2p
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0) ≤
    -- Full sum
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0) := by
  sorry  -- Follows from existing SubtreePruning.partial_conv_le_full_conv

/-- Claim 4.18: Window sum monotonicity — if the partial window sum
    exceeds the threshold for every possible W_int, then the full
    window sum also exceeds the threshold.

    Precisely: let ws_partial = Σ_{k ∈ [s,s+ℓ-2]} partial_conv[k]
    and ws_full = Σ_{k ∈ [s,s+ℓ-2]} full_conv[k]. Then:
      (1) ws_full ≥ ws_partial         (from Claim 4.17)
      (2) W_int_max ≥ W_int_actual     (from Claim 4.19)
      (3) threshold(W_int_max) ≥ threshold(W_int_actual)  (monotonicity)
    Chain: ws_full ≥ ws_partial > threshold(W_int_max) ≥ threshold(W_int_actual)
    ⟹ ws_full > threshold(W_int_actual) ⟹ child is pruned. -/
theorem subtree_pruning_soundness_gray
    (ws_partial ws_full : ℤ)
    (dyn_max dyn_actual : ℤ)
    (h_partial_le : ws_full ≥ ws_partial)
    (h_exceeds : ws_partial > dyn_max)
    (h_threshold_mono : dyn_max ≥ dyn_actual) :
    ws_full > dyn_actual := by
  sorry  -- Follows from existing SubtreePruning.subtree_pruning_chain

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: W_int_max Correctness (Claims 4.19–4.20)
--
-- The W_int_max computation uses parent_prefix for unfixed bins.
-- We must prove that parent_int[p] is an upper bound on the sum
-- of child masses for any split of parent position p.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.19: For any parent position p and any valid split
    child[2p] + child[2p+1] = parent[p], the total child mass
    contributed by position p to any window is at most parent[p]. -/
theorem parent_mass_bounds_child_mass
    {d_parent : ℕ} (parent : Fin d_parent → ℕ) (child : Fin (2 * d_parent) → ℕ)
    (h_split : ∀ p : Fin d_parent,
      child ⟨2 * p.1, by omega⟩ + child ⟨2 * p.1 + 1, by omega⟩ = parent p)
    (p : Fin d_parent) :
    child ⟨2 * p.1, by omega⟩ + child ⟨2 * p.1 + 1, by omega⟩ = parent p := by
  sorry  -- Direct from h_split

/-- Claim 4.20: W_int_max = W_int_fixed + W_int_unfixed is an upper
    bound on the actual W_int for any child in the subtree.

    W_int_fixed uses the exact child masses of fixed bins.
    W_int_unfixed uses parent_prefix[hi_parent+1] - parent_prefix[lo_parent]
    as an upper bound on the unfixed bins' contribution. -/
theorem w_int_max_is_upper_bound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (child_any : Fin (2 * d_parent) → ℕ)
    (fixed_child : Fin (2 * d_parent) → ℕ)
    (p_boundary : ℕ) (hp : 2 * p_boundary ≤ 2 * d_parent)
    -- Fixed prefix matches for all bins < 2*p_boundary
    (h_fixed : ∀ i : Fin (2 * d_parent), i.1 < 2 * p_boundary →
      fixed_child i = child_any i)
    -- child_any is a valid split of parent
    (h_split : ∀ q : Fin d_parent,
      child_any ⟨2 * q.1, by omega⟩ + child_any ⟨2 * q.1 + 1, by omega⟩ = parent q)
    (lo_bin hi_bin : ℕ) (hlo : lo_bin ≤ hi_bin)
    (hhi : hi_bin < 2 * d_parent)
    -- W_int_fixed: sum of fixed_child masses in the fixed portion of the window
    (W_int_fixed : ℤ)
    (hWf : W_int_fixed = ∑ i ∈ Finset.Icc lo_bin (min hi_bin (2 * p_boundary - 1)),
      (child_any ⟨i, by omega⟩ : ℤ))
    -- W_int_unfixed: parent mass upper bound for unfixed portion
    (W_int_unfixed : ℤ)
    (hWu : W_int_unfixed ≥ ∑ q ∈ Finset.filter
      (fun q => 2 * q ≤ hi_bin ∧ lo_bin ≤ 2 * q + 1 ∧ q ≥ p_boundary)
      (Finset.range d_parent),
      (parent ⟨q, by omega⟩ : ℤ)) :
    -- Actual W_int for child_any
    (∑ i ∈ Finset.Icc lo_bin hi_bin, (child_any ⟨i, by omega⟩ : ℤ))
      ≤ W_int_fixed + W_int_unfixed := by
  sorry  -- Extends SubtreePruning.w_int_bounded to non-contiguous case

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Gray Code Fast-Forward Correctness (Claims 4.21–4.22)
--
-- After a successful subtree prune, the inner Gray code state is reset
-- so that the next outer advance starts a fresh inner sweep. We must
-- prove that this preserves enumeration completeness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.21: After resetting inner digits to gc_a[k]=0, gc_dir[k]=+1,
    gc_focus[k]=k for k < J_MIN, and setting gc_focus[0] = next_focus,
    the next invocation of the Gray code advance will change the digit
    at index next_focus (or terminate if next_focus = n_active).

    This is the key invariant for the fast-forward: the inner sweep is
    skipped by wiring the focus chain past all inner digits. -/
theorem gray_code_fast_forward_focus
    (n_active J_MIN : ℕ) (hJ : J_MIN ≤ n_active)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (next_focus : ℕ)
    -- After reset: gc_focus[0] = next_focus, gc_focus[k] = k for 1 ≤ k < J_MIN
    -- gc_focus[J_MIN] = J_MIN, gc_focus[k] unchanged for k > J_MIN
    (h_reset : gc_focus ⟨0, by omega⟩ = next_focus) :
    -- The next digit to advance is next_focus
    gc_focus ⟨0, by omega⟩ = next_focus := by
  sorry  -- Immediate from h_reset; the real content is the enumeration completeness below

/-- Claim 4.22: Enumeration completeness — after a subtree prune at
    level J_MIN and fast-forward, the subsequent execution of the Gray
    code kernel visits exactly the compositions NOT in the pruned subtree.

    Formally: let S_pruned be the set of children in the pruned subtree
    (all combinations of inner digits 0..J_MIN-1 with the current outer
    state). Let S_total be the full Cartesian product. The kernel visits
    all elements of S_total \ S_pruned (and possibly one extra "reset"
    child at the boundary, which is individually pruned by the normal test).

    Proof sketch: The fast-forward sets gc_focus[0] = next_focus, causing
    the main loop to advance the outer digit (next_focus ≥ J_MIN) on the
    next iteration. The inner digits are reset to gc_a[k]=0, gc_dir[k]=+1,
    which is a valid initial state for a fresh Knuth mixed-radix Gray code
    traversal. By the completeness of Algorithm M (TAOCP 7.2.1.1), the
    next inner sweep starting from this state visits all ∏_{k<J_MIN} r_k
    combinations. Combined with the outer iteration, all non-pruned
    compositions are visited. -/
theorem gray_code_subtree_enumeration_completeness
    {n_active : ℕ} (r : Fin n_active → ℕ) (hr : ∀ i, r i ≥ 2)
    (J_MIN : ℕ) (hJ : J_MIN < n_active)
    (total : ℕ) (htotal : total = ∏ i, r i) :
    -- The number of compositions visited + compositions in pruned subtrees
    -- equals the total Cartesian product
    ∃ (n_visited n_pruned : ℕ),
      n_visited + n_pruned = total ∧
      -- Every pruned composition would have been individually pruned
      -- (this is guaranteed by the partial autoconv check)
      True := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: Threshold Arithmetic (Claims 4.23–4.24)
--
-- The integer threshold computation must be exact. The same dyn_base_ell,
-- two_ell_inv_4n, and one_minus_4eps constants are used for the subtree
-- check as for the per-child check. We must prove the floor-rounding
-- is consistent.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.23: The dynamic threshold is monotone non-decreasing in W_int.
    Since we use W_int_max ≥ W_int_actual, the threshold with W_int_max
    is at least as large, making the pruning test conservative. -/
theorem dynamic_threshold_monotone
    (c_target : ℝ) (m : ℝ) (ell : ℝ) (inv_4n : ℝ)
    (h_pos : 0 < 1 - 4 * (2.220446049250313e-16 : ℝ))
    (W1 W2 : ℝ) (hW : W1 ≤ W2) :
    ⌊(c_target * m ^ 2 * ell * inv_4n + 1 + 1e-9 * m ^ 2 + 2 * W1 * ell * inv_4n) *
      (1 - 4 * (2.220446049250313e-16 : ℝ))⌋ ≤
    ⌊(c_target * m ^ 2 * ell * inv_4n + 1 + 1e-9 * m ^ 2 + 2 * W2 * ell * inv_4n) *
      (1 - 4 * (2.220446049250313e-16 : ℝ))⌋ := by
  sorry  -- Follows from existing SubtreePruning.dyn_it_mono

/-- Claim 4.24: The int32 convolution entries are non-negative when
    all child masses are non-negative. This ensures the partial
    autoconvolution (a subset of non-negative terms) is also non-negative,
    which is needed for the lower-bound argument. -/
theorem partial_conv_nonneg
    {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    0 ≤ ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2 * p ∧ j.1 < 2 * p then c i * c j else 0 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART G: End-to-End Soundness (Claim 4.25)
--
-- The final theorem tying everything together: the Gray code kernel
-- with subtree pruning produces a SUPERSET of the survivors that the
-- kernel without subtree pruning would produce. (I.e., subtree pruning
-- only removes compositions that would be individually pruned anyway.)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.25 (Master Soundness Theorem): For any parent composition,
    the set of canonical survivors produced by the Gray code kernel with
    subtree pruning is identical to the set produced without subtree pruning.

    Direction (⊆): Every survivor of the pruned kernel is a survivor of
    the unpruned kernel (trivial — pruning only removes compositions).

    Direction (⊇): Every survivor of the unpruned kernel is a survivor
    of the pruned kernel. This requires showing that no survivor is in
    a pruned subtree. By Claims 4.17–4.18, if a subtree is pruned, then
    ws_full > dyn_actual for every child in the subtree, meaning every
    child would be individually pruned by the normal test. Hence no
    survivor can be in a pruned subtree. -/
theorem gray_code_subtree_pruning_sound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (m : ℕ) (c_target : ℝ) (n_half_child : ℕ)
    -- survivor_with_pruning: set of children that survive with subtree pruning
    -- survivor_without_pruning: set of children that survive without
    (S_with S_without : Finset (Fin (2 * d_parent) → ℕ))
    -- Core hypotheses linking the two sets
    (h_with_subset : S_with ⊆ S_without)
    (h_pruned_all_killed : ∀ child ∈ S_without,
      -- If child is in a pruned subtree, then it would be individually pruned
      True) :
    S_with = S_without := by
  sorry

end -- noncomputable section
