/-
Sidon Autocorrelation Project — Minimum Contribution Bounds (Claims 6.37–6.44)

This file collects the theorems and lemmas certifying the minimum contribution
bounds used in multi-level subtree pruning, implemented in the GPU kernel
(cascade_kernel.cu lines 1729-1860: min contribution computation, and
lines 814-897: partial_window_scan_max_threshold).

When subtree pruning fires at Gray code level gc_j, the child bins split into:
  (A) Fixed bins [0..fixed_len-1]: their autoconvolution is exactly known.
  (B) Inner active bins (digits 0..gc_j-1): variable, with cursor ranges.
  (C) Non-active unfixed bins (range==1, beyond fixed prefix): single value.

For the variable bins, we compute LOWER BOUNDS on their convolution
contributions to each window sum. If partial_conv(fixed) + min_contrib(unfixed)
> threshold for some window, the entire subtree is prunable.

The minimum contribution bounds are:
  - Self-terms: lo² for each variable bin (minimum possible self-contribution)
  - Cross with fixed: 2*lo*fixed_i (minimum cross-term, since all values ≥ 0)
  - Cross between unfixed: pairwise min products

Claims covered:
  6.37  Self-term minimum: lo² ≤ x² for all x ∈ [lo, hi] with lo ≥ 0
  6.38  Cross-term minimum: 2*lo*f ≤ 2*x*f for x ≥ lo ≥ 0, f ≥ 0
  6.39  Pairwise cross minimum: 2*lo_a*lo_b ≤ 2*x_a*x_b for x ≥ lo ≥ 0
  6.40  Total minimum contribution is lower bound on unfixed conv contribution
  6.41  Minimum contribution prefix sum is monotone
  6.42  W_int upper bound from parent masses (unfixed bins)
  6.43  Combined partial + min > threshold implies subtree prunable
  6.44  End-to-end: multi-level subtree pruning is sound

Cross-cutting dependencies:
  - SubtreePruning.lean (Claim 4.4): partial conv ≤ full conv
  - GrayCodeSubtreePruning.lean (Claims 4.14-4.25): Gray code subtree
  - DynamicThreshold.lean: threshold monotonicity in W_int

STATUS: All sorry stubs — proofs not yet attempted.
-/

import Mathlib
import Sidon.Defs
import Sidon.SubtreePruning

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
-- PART A: Elementary Bounds (Claims 6.37–6.39)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.37: Self-term minimum bound.
    For nonneg integers, lo² ≤ x² when 0 ≤ lo ≤ x.

    This gives the minimum self-term conv[2k] contribution from a variable bin:
    if child[k] ∈ [lo, hi], then child[k]² ≥ lo².

    Matches: cascade_kernel.cu min contribution computation — self-term:
    min_self = lo * lo. -/
theorem self_term_minimum (lo x : ℤ) (hlo : 0 ≤ lo) (hx : lo ≤ x) :
    lo * lo ≤ x * x := by
  sorry

/-- Claim 6.38: Cross-term minimum with fixed bin.
    For nonneg integers, 2*lo*f ≤ 2*x*f when 0 ≤ lo ≤ x and 0 ≤ f.

    This gives the minimum cross-term contribution conv[k+j] from a variable
    bin k with a fixed bin j: if child[k] ∈ [lo, hi] and child[j] = f (fixed),
    then 2*child[k]*child[j] ≥ 2*lo*f.

    Matches: cascade_kernel.cu min contribution — cross with fixed:
    min_cross = 2 * lo * fixed_val. -/
theorem cross_term_minimum_fixed (lo x f : ℤ) (hlo : 0 ≤ lo) (hx : lo ≤ x) (hf : 0 ≤ f) :
    2 * lo * f ≤ 2 * x * f := by
  sorry

/-- Claim 6.39: Pairwise cross-term minimum between two unfixed bins.
    For nonneg integers, 2*lo_a*lo_b ≤ 2*x_a*x_b when 0 ≤ lo ≤ x.

    Matches: cascade_kernel.cu min contribution — cross between inner actives:
    min_pairwise = 2 * lo_a * lo_b. -/
theorem cross_term_minimum_pairwise (lo_a lo_b x_a x_b : ℤ)
    (hla : 0 ≤ lo_a) (hlb : 0 ≤ lo_b) (hxa : lo_a ≤ x_a) (hxb : lo_b ≤ x_b) :
    2 * lo_a * lo_b ≤ 2 * x_a * x_b := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Total Minimum Contribution (Claim 6.40)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.40: The total minimum contribution from unfixed bins is a lower
    bound on the actual unfixed convolution contribution.

    Let min_contrib[k] = sum of minimum self/cross/pairwise terms contributing
    to conv[k] from unfixed bins. Then for any valid assignment of unfixed bins
    within their ranges, the actual contribution to conv[k] is ≥ min_contrib[k].

    Formally: for each conv index k, the sum of all (i,j) terms where at least
    one of i,j is unfixed and i+j=k, evaluated at min values, is ≤ the same
    sum evaluated at actual values.

    Matches: cascade_kernel.cu lines 1729-1860 (min contribution computation). -/
theorem min_contribution_lower_bound
    {d : ℕ} (child child_min : Fin d → ℤ)
    (fixed_len : ℕ) (hfl : fixed_len ≤ d)
    (hc_nonneg : ∀ i, 0 ≤ child i)
    (hm_nonneg : ∀ i, 0 ≤ child_min i)
    (h_fixed : ∀ i : Fin d, i.1 < fixed_len → child_min i = child i)
    (h_unfixed : ∀ i : Fin d, fixed_len ≤ i.1 → child_min i ≤ child i)
    (k : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = k ∧ (fixed_len ≤ i.1 ∨ fixed_len ≤ j.1)
      then child_min i * child_min j else 0) ≤
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = k ∧ (fixed_len ≤ i.1 ∨ fixed_len ≤ j.1)
      then child i * child j else 0) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Prefix Sum Monotonicity (Claim 6.41)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.41: The minimum contribution inclusive prefix sum is monotone.
    If min_contrib[k] ≥ 0 for all k, then the prefix sum is monotone.
    This allows efficient sliding-window queries over min_contrib.

    Matches: cascade_kernel.cu — min_contrib_smem is converted to inclusive
    prefix sum for window scanning in partial_window_scan_max_threshold. -/
theorem min_contrib_prefix_monotone
    (min_contrib : ℕ → ℤ) (h_nonneg : ∀ k, 0 ≤ min_contrib k)
    (n : ℕ) (a b : ℕ) (hab : a ≤ b) (hbn : b ≤ n) :
    ∑ k ∈ Finset.Icc 0 a, min_contrib k ≤ ∑ k ∈ Finset.Icc 0 b, min_contrib k := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: W_int Upper Bound (Claim 6.42)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.42: W_int for unfixed bins is bounded above by the sum of parent
    masses in the corresponding parent positions.

    For unfixed child bins (beyond the fixed prefix), the child pair
    child[2q] + child[2q+1] = parent[q], so W_int in any window is bounded
    by the parent masses. This gives W_int_max for threshold lookup.

    Matches: cascade_kernel.cu partial_window_scan_max_threshold lines 860-870
    (W_int_fixed + W_int_unfixed computation using parent_prefix). -/
theorem w_int_unfixed_upper_bound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (child : Fin (2 * d_parent) → ℕ)
    (h_split : ∀ q : Fin d_parent,
      child ⟨2 * q.1, by omega⟩ + child ⟨2 * q.1 + 1, by omega⟩ = parent q)
    (lo_bin hi_bin : ℕ) (fixed_child_len : ℕ) :
    (∑ i ∈ Finset.Icc (max lo_bin fixed_child_len) hi_bin,
      if h : i < 2 * d_parent then (child ⟨i, h⟩ : ℤ) else 0) ≤
    (∑ q ∈ Finset.filter (fun q => 2 * q ≤ hi_bin ∧ lo_bin ≤ 2 * q + 1 ∧
        fixed_child_len ≤ 2 * q + 1)
      (Finset.range d_parent),
      if h : q < d_parent then (parent ⟨q, h⟩ : ℤ) else 0) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Subtree Pruning Soundness (Claims 6.43, 6.44)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.43: If partial_conv(fixed) + min_contrib(unfixed) > threshold
    for some window (ell, s), then ALL children in the subtree exceed the
    threshold for that window.

    Proof sketch:
    1. partial_conv(fixed) is exact for the fixed prefix.
    2. min_contrib(unfixed) ≤ actual_unfixed_contrib (Claim 6.40).
    3. So partial + min ≤ partial + actual = total window sum.
    4. If partial + min > threshold, then total > threshold.

    Matches: cascade_kernel.cu partial_window_scan_max_threshold lines 830-890. -/
theorem partial_plus_min_implies_prune
    {d : ℕ} (child child_min : Fin d → ℤ)
    (fixed_len : ℕ) (hfl : fixed_len ≤ d)
    (hc_nonneg : ∀ i, 0 ≤ child i)
    (hm_nonneg : ∀ i, 0 ≤ child_min i)
    (h_fixed : ∀ i : Fin d, i.1 < fixed_len → child_min i = child i)
    (h_unfixed : ∀ i : Fin d, fixed_len ≤ i.1 → child_min i ≤ child i)
    (threshold : ℤ) (s_lo ell : ℕ)
    (h_partial_exceeds :
      (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
        ((∑ i : Fin d, ∑ j : Fin d,
          if i.1 + j.1 = k ∧ i.1 < fixed_len ∧ j.1 < fixed_len
          then child i * child j else 0) +
        (∑ i : Fin d, ∑ j : Fin d,
          if i.1 + j.1 = k ∧ (fixed_len ≤ i.1 ∨ fixed_len ≤ j.1)
          then child_min i * child_min j else 0))) > threshold) :
    (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
      (∑ i : Fin d, ∑ j : Fin d,
        if i.1 + j.1 = k then child i * child j else 0)) > threshold := by
  sorry

/-- Claim 6.44: End-to-end multi-level subtree pruning soundness.
    The pruning check fires at every Gray code level gc_j ≥ 2 where the
    cost/benefit ratio justifies it. At each level, if the check passes
    (subtree prunable), all children in the subtree would individually fail
    the window scan. No survivors are lost.

    Matches: cascade_kernel.cu lines 1691-1932 (multi-level subtree pruning). -/
theorem multi_level_subtree_sound
    {d : ℕ} (children : Finset (Fin d → ℤ))
    (h_nonneg : ∀ c ∈ children, ∀ i, 0 ≤ c i)
    (fixed_len : ℕ) (hfl : fixed_len ≤ d)
    (h_same_prefix : ∀ c ∈ children, ∀ i : Fin d, i.1 < fixed_len →
      c i = (children.toList.head (by sorry) : Fin d → ℤ) i)
    (threshold : ℤ)
    (h_subtree_prune : ∃ ell s_lo,
      ∀ c ∈ children,
        (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
          (∑ i : Fin d, ∑ j : Fin d,
            if i.1 + j.1 = k then c i * c j else 0)) > threshold) :
    ∀ c ∈ children,
      ∃ ell s_lo, (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
        (∑ i : Fin d, ∑ j : Fin d,
          if i.1 + j.1 = k then c i * c j else 0)) > threshold := by
  sorry

end -- noncomputable section
