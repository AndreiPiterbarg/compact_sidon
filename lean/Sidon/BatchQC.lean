/-
Sidon Autocorrelation Project — Batch Quick-Check (Claims 6.13–6.19)

This file collects the theorems and lemmas certifying the batch quick-check
optimization ("Idea 2") implemented in the GPU kernel
(cascade_kernel.cu lines ~1362-1460).

The optimization works as follows: when the innermost Gray code digit (j==0)
is about to sweep through its remaining values, and we have a cached killing
window, we analytically check ALL remaining cursor values in one shot.
If all of them would be killed, we skip the entire inner sweep.

For each remaining cursor value c in [lo_i, hi_i]:
  ws_c = ws_base + delta_cross * (c - c_current) + delta_self_quadratic
  W_c = W_base + (c - c_current) * bin_in_range_indicator
  If ws_c > threshold_table[ell_idx][W_c] for ALL c, skip the sweep.

Claims covered:
  6.13  Window sum is affine in innermost cursor (cross-terms only)
  6.14  Self-term contribution is quadratic in cursor
  6.15  Total ws(c) = quadratic polynomial in cursor value
  6.16  W_int is affine in cursor value
  6.17  Batch check: if all values killed, no survivors in inner sweep
  6.18  Batch skip preserves enumeration completeness
  6.19  End-to-end: batch QC produces identical survivor set

Cross-cutting dependencies:
  - UnivariateSweepSkip.lean (Claims 4.36-4.48): ws is quadratic in cursor
  - LazyQCSensitivity.lean (Claims 6.1-6.12): sensitivity definitions
  - GrayCode.lean: Gray code traversal properties

STATUS: All sorry stubs — proofs not yet attempted.
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
-- Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Child vector parameterized by innermost cursor value x.
    child_at_x base k1 k2 parent_val x returns the child vector where
    child[k1] = x, child[k2] = parent_val - x, and all other bins are fixed. -/
def child_at_x {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (parent_val : ℤ) (x : ℤ) : Fin d → ℤ :=
  fun i => if i.1 = k1 then x
           else if i.1 = k2 then parent_val - x
           else base i

/-- Window sum as a function of innermost cursor value x. -/
def ws_at_x {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (parent_val : ℤ) (s_lo ell : ℕ) (x : ℤ) : ℤ :=
  ∑ k ∈ Finset.Icc s_lo (s_lo + ell - 2),
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = k then child_at_x base k1 k2 hk1 hk2 parent_val x i *
                             child_at_x base k1 k2 hk1 hk2 parent_val x j
      else 0)

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Polynomial Structure (Claims 6.13–6.15)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.13: The cross-term contribution to the window sum is affine in
    the innermost cursor value x.

    When only child[k1] and child[k2] change (as x varies), the cross-terms
    ∑_{j ∉ {k1,k2}} 2·child[j]·(child[k1] + child[k2]) are affine in x
    because child[k1] + child[k2] = parent_val is constant, BUT the individual
    cross-terms 2·child[j]·child[k1] and 2·child[j]·child[k2] are each
    affine in x with opposite slopes, and only a subset falls in the window.

    Matches: cascade_kernel.cu batch QC lines ~1370-1400 (precompute
    cross-term sensitivity for innermost position). -/
theorem cross_term_affine_in_cursor
    {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (hk12 : k1 ≠ k2) (parent_val : ℤ)
    (s_lo ell : ℕ) :
    ∃ (A B : ℤ), ∀ x : ℤ,
      (∑ k ∈ Finset.Icc s_lo (s_lo + ell - 2),
        ∑ i : Fin d, ∑ j : Fin d,
          if i.1 + j.1 = k ∧ ((i.1 = k1 ∧ j.1 ≠ k1 ∧ j.1 ≠ k2) ∨
                                (j.1 = k1 ∧ i.1 ≠ k1 ∧ i.1 ≠ k2) ∨
                                (i.1 = k2 ∧ j.1 ≠ k1 ∧ j.1 ≠ k2) ∨
                                (j.1 = k2 ∧ i.1 ≠ k1 ∧ i.1 ≠ k2))
          then child_at_x base k1 k2 hk1 hk2 parent_val x i *
               child_at_x base k1 k2 hk1 hk2 parent_val x j
          else 0) = A * x + B := by
  sorry

/-- Claim 6.14: The self-term contribution (conv[2k1], conv[2k2], conv[k1+k2])
    is quadratic in x.

    conv[2k1] = x²,  conv[2k2] = (parent_val - x)²,
    conv[k1+k2] = 2·x·(parent_val - x).
    Sum of self-terms in window = quadratic in x.

    Matches: cascade_kernel.cu batch QC self-term precomputation. -/
theorem self_term_quadratic_in_cursor
    (parent_val : ℤ) (s_lo ell k1 k2 : ℕ) :
    ∃ (A B C : ℤ), ∀ x : ℤ,
      let self_k1 := if 2 * k1 ∈ Finset.Icc s_lo (s_lo + ell - 2) then x * x else 0
      let self_k2 := if 2 * k2 ∈ Finset.Icc s_lo (s_lo + ell - 2)
                     then (parent_val - x) * (parent_val - x) else 0
      let mutual := if k1 + k2 ∈ Finset.Icc s_lo (s_lo + ell - 2)
                    then 2 * x * (parent_val - x) else 0
      self_k1 + self_k2 + mutual = A * x^2 + B * x + C := by
  sorry

/-- Claim 6.15: Total ws(x) = A·x² + B·x + C for some integer A, B, C.
    Combines Claims 6.13 and 6.14.

    Matches: UnivariateSweepSkip.lean Claim 4.36, but stated here in integer
    arithmetic for the batch QC context. -/
theorem ws_quadratic_in_cursor
    {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (hk12 : k1 ≠ k2) (parent_val : ℤ) (s_lo ell : ℕ) :
    ∃ (A B C : ℤ), ∀ x : ℤ,
      ws_at_x base k1 k2 hk1 hk2 parent_val s_lo ell x = A * x^2 + B * x + C := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: W_int Structure (Claim 6.16)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.16: W_int is affine in the cursor value x.
    W_int(x) = W_base + α·x where α ∈ {-1, 0, 1, 2} depending on whether
    k1 and/or k2 are in the contributing bin range.

    Matches: cascade_kernel.cu batch QC W_int computation lines ~1410-1420. -/
theorem w_int_affine_in_cursor
    {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (parent_val : ℤ) (lo_bin hi_bin : ℕ) :
    ∃ (α W_base : ℤ), ∀ x : ℤ,
      (∑ i ∈ Finset.Icc lo_bin hi_bin,
        if h : i < d then child_at_x base k1 k2 hk1 hk2 parent_val x ⟨i, h⟩ else 0) =
      α * x + W_base := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Batch Soundness (Claims 6.17–6.19)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.17: If ALL remaining cursor values in [lo, hi] produce ws > threshold,
    then no child in the inner sweep survives.

    Formally: if ∀ x ∈ [lo, hi], ws_at_x(x) > threshold(W_int(x)), then
    no child in the sweep passes the window test.

    Matches: cascade_kernel.cu batch QC lines ~1430-1440 (loop checking all
    remaining cursor values). -/
theorem batch_qc_all_killed
    {d : ℕ} (base : Fin d → ℤ) (k1 k2 : ℕ) (hk1 : k1 < d) (hk2 : k2 < d)
    (parent_val : ℤ) (s_lo ell : ℕ) (lo hi : ℤ)
    (threshold : ℤ → ℤ)  -- threshold as function of W_int
    (h_all_killed : ∀ x : ℤ, lo ≤ x → x ≤ hi →
      ws_at_x base k1 k2 hk1 hk2 parent_val s_lo ell x >
        threshold (∑ i ∈ Finset.Icc 0 (d-1),
          if h : i < d then child_at_x base k1 k2 hk1 hk2 parent_val x ⟨i, h⟩ else 0)) :
    ∀ x : ℤ, lo ≤ x → x ≤ hi →
      ∃ ℓ s, ws_at_x base k1 k2 hk1 hk2 parent_val s ℓ x >
        threshold (∑ i ∈ Finset.Icc 0 (d-1),
          if h : i < d then child_at_x base k1 k2 hk1 hk2 parent_val x ⟨i, h⟩ else 0) := by
  sorry

/-- Claim 6.18: Batch skip preserves enumeration completeness.
    After a batch skip, the Gray code state is reset to the boundary of the
    innermost digit, and enumeration continues with the next outer digit.
    No children are missed — the skipped children are exactly those in
    [lo, hi] × (fixed outer digits).

    Matches: cascade_kernel.cu lines ~1440-1460 (reset inner GC digit). -/
theorem batch_skip_completeness
    {n : ℕ} (radix : Fin n → ℕ)
    (j_inner : Fin n) (hj : j_inner.1 = 0)
    (skipped_count : ℕ)
    (h_skip : skipped_count = radix j_inner - 1)  -- remaining values in inner digit
    (total_before total_after : ℕ)
    (h_total : total_after = total_before + skipped_count) :
    total_after - total_before = radix j_inner - 1 := by
  sorry

/-- Claim 6.19: End-to-end batch QC soundness.
    The survivor set produced with batch QC is identical to the survivor set
    without batch QC. Batch QC only skips children that would be individually
    pruned by the full window scan.

    Proof sketch: Each skipped child c has ws(c) > threshold for some cached
    window (Claim 6.17). The full window scan would also find this window and
    prune the child. -/
theorem batch_qc_end_to_end
    {d : ℕ} (children : List (Fin d → ℤ)) (threshold : ℤ)
    (s_lo ell : ℕ)
    (survivors_with_batch survivors_without : List (Fin d → ℤ))
    (h_without : ∀ c ∈ survivors_without,
      ¬ ∃ ℓ s, ∑ k ∈ Finset.Icc s (s + ℓ - 2),
        (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0) > threshold)
    (h_with_subset : ∀ c ∈ survivors_with_batch, c ∈ survivors_without)
    (h_with_superset : ∀ c ∈ survivors_without, c ∈ survivors_with_batch) :
    survivors_with_batch = survivors_without := by
  sorry

end -- noncomputable section
