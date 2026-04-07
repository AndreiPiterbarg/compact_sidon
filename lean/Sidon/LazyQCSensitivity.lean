/-
Sidon Autocorrelation Project — Lazy QC Sensitivity (Claims 6.1–6.12)

This file collects the theorems and lemmas certifying the lazy quick-check
(QC) with analytical sensitivity tracking, implemented as "Idea 1+4" in
the GPU kernel (cascade_kernel.cu: precompute_qc_sensitivity,
update_sensitivities_after_step, lazy_qc_update_and_check).

The optimization replaces O(ell) conv-read QC with O(1) analytical updates:
  - When a Gray code position `pos` advances by delta_c = ±1, the window sum
    ws changes by: delta_c * sens_cross[pos] + self-term corrections.
  - The sensitivity sens_cross[a] for each active position a and fixed window
    (ell, s) equals:
      ∑_{j ∉ {k1,k2}, k1+j ∈ [s..s+ell-2]} 2·child[j]
    − ∑_{j ∉ {k1,k2}, k2+j ∈ [s..s+ell-2]} 2·child[j]
    where k1 = 2·active_pos[a], k2 = k1+1.
  - When another position j_changed advances, each other position a's
    sensitivity is updated incrementally.

The multi-window cache (Idea 4) maintains K=5 independent cached windows,
each with its own sensitivity array, ws_lazy, and W_int.

Claims covered:
  6.1   Sensitivity definition matches actual ws delta for single-step advance
  6.2   Self-term delta formula (conv[2k1], conv[2k2], conv[k1+k2])
  6.3   Analytical ws_lazy = actual window sum (invariant)
  6.4   Sensitivity incremental update correctness
  6.5   W_int lazy update correctness (bin-range based)
  6.6   Multi-window cache: independent updates preserve per-window invariant
  6.7   Lazy QC prune implies actual QC prune (no false prunes)
  6.8   Sensitivity recompute matches initial computation after conv rebuild
  6.9   Self-term mask correctness (mask_k1, mask_k2, mask_m)
  6.10  Cache seeding: ws_lazy initialized from actual conv matches
  6.11  Batch sensitivity update: all K windows updated after each GC step
  6.12  End-to-end: lazy QC kernel produces identical survivor set

Cross-cutting dependencies:
  - IncrementalAutoconv.lean (Claim 4.2): delta decomposition into groups
  - GrayCode.lean (Claim 4.11): W_int update correctness
  - ThresholdLookupTable.lean (Claim 5.3): threshold table entry formula

STATUS: All sorry stubs — proofs not yet attempted.
-/

import Mathlib
import Sidon.Defs
import Sidon.IncrementalAutoconv

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

/-- Window sum: sum of conv[k] for k in [s_lo .. s_lo + ell - 2].
    This is the integer window sum from the conv array. -/
def window_sum_int {d : ℕ} (child : Fin d → ℤ) (s_lo ell : ℕ) : ℤ :=
  ∑ k ∈ Finset.Icc s_lo (s_lo + ell - 2),
    (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then child i * child j else 0)

/-- Cross-term sensitivity for active position a with respect to window (ell, s).
    This is the change in ws when child[k1] increases by 1 and child[k2]
    decreases by 1 (one Gray code step), counting only cross-terms. -/
def sensitivity_cross {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ) (s_lo ell : ℕ) : ℤ :=
  (∑ j : Fin d, if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k1 + j.1 ∈ Finset.Icc s_lo (s_lo + ell - 2)
    then 2 * child j else 0) -
  (∑ j : Fin d, if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k2 + j.1 ∈ Finset.Icc s_lo (s_lo + ell - 2)
    then 2 * child j else 0)

/-- Self-term mask: 1 if conv index 2*k falls in the window [s_lo, s_lo+ell-2]. -/
def self_mask (k : ℕ) (s_lo ell : ℕ) : ℤ :=
  if 2 * k ∈ Finset.Icc s_lo (s_lo + ell - 2) then 1 else 0

/-- Mutual-term mask: 1 if conv index k1+k2 falls in the window. -/
def mutual_mask (k1 k2 : ℕ) (s_lo ell : ℕ) : ℤ :=
  if k1 + k2 ∈ Finset.Icc s_lo (s_lo + ell - 2) then 1 else 0

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Single-Step Sensitivity (Claims 6.1, 6.2)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.1: When position pos advances (child[k1] += delta_c, child[k2] -= delta_c),
    the change in window sum equals:
      delta_c * sens_cross + self-term corrections.

    This is the core correctness claim for the lazy QC.
    The GPU computes ws_new = ws_old + delta_c * sens_cross[a] + self_corrections
    instead of re-reading conv.

    Matches: cascade_kernel.cu lazy_qc_update_and_check lines 374-383. -/
theorem sensitivity_single_step_correct
    {d : ℕ} (child child' : Fin d → ℤ)
    (pos : ℕ) (hpos : 2 * pos + 1 < d)
    (delta_c : ℤ)
    (h_k1 : child' ⟨2*pos, by omega⟩ = child ⟨2*pos, by omega⟩ + delta_c)
    (h_k2 : child' ⟨2*pos+1, by omega⟩ = child ⟨2*pos+1, by omega⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ 2*pos ∧ i.1 ≠ 2*pos+1 → child' i = child i)
    (s_lo ell : ℕ) :
    window_sum_int child' s_lo ell - window_sum_int child s_lo ell =
      delta_c * sensitivity_cross child (2*pos) (2*pos+1) s_lo ell
      + delta_c * (2 * child ⟨2*pos, by omega⟩ + delta_c) *
          self_mask (2*pos) s_lo ell
      + delta_c * (-(2 * child ⟨2*pos+1, by omega⟩ - delta_c)) *
          self_mask (2*pos+1) s_lo ell
      + delta_c * 2 * (child ⟨2*pos+1, by omega⟩ - child ⟨2*pos, by omega⟩ - delta_c) *
          mutual_mask (2*pos) (2*pos+1) s_lo ell := by
  sorry

/-- Claim 6.2: Self-term delta formulas.
    When child[k1] changes from c_old to c_old + delta_c:
      conv[2*k1] changes by delta_c * (2*c_old + delta_c)   [= (c_old+d)² - c_old²]
      conv[2*k2] changes by delta_c * (-(2*(a-c_old) - delta_c)) [= (a-c_old-d)² - (a-c_old)²]
      conv[k1+k2] changes by delta_c * 2*(a - 2*c_old - delta_c)

    Matches: cascade_kernel.cu lines 378-383. -/
theorem self_term_deltas
    (c_old a_pos delta_c : ℤ) :
    ((c_old + delta_c)^2 - c_old^2 = delta_c * (2 * c_old + delta_c)) ∧
    ((a_pos - c_old - delta_c)^2 - (a_pos - c_old)^2 =
      delta_c * (-(2 * (a_pos - c_old) - delta_c))) ∧
    (2 * (c_old + delta_c) * (a_pos - c_old - delta_c) - 2 * c_old * (a_pos - c_old) =
      delta_c * 2 * (a_pos - 2 * c_old - delta_c)) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Lazy Invariant (Claim 6.3)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.3: The lazy window sum ws_lazy equals the actual window sum from conv
    at every point in the Gray code traversal.

    Invariant: ws_lazy = window_sum_int child s_lo ell.

    This is maintained inductively: it holds after seeding (from actual conv),
    and each Gray code step preserves it via the sensitivity formula (Claim 6.1).

    Matches: cascade_kernel.cu lazy_qc_update_and_check — the function updates
    ws_lazy analytically and checks it against the threshold. -/
theorem ws_lazy_invariant
    {d : ℕ} (child₀ : Fin d → ℤ) (s_lo ell : ℕ)
    (ws_lazy₀ : ℤ)
    (h_init : ws_lazy₀ = window_sum_int child₀ s_lo ell)
    (child₁ : Fin d → ℤ) (pos : ℕ) (hpos : 2 * pos + 1 < d)
    (delta_c : ℤ)
    (h_k1 : child₁ ⟨2*pos, by omega⟩ = child₀ ⟨2*pos, by omega⟩ + delta_c)
    (h_k2 : child₁ ⟨2*pos+1, by omega⟩ = child₀ ⟨2*pos+1, by omega⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ 2*pos ∧ i.1 ≠ 2*pos+1 → child₁ i = child₀ i)
    (ws_lazy₁ : ℤ)
    (h_update : ws_lazy₁ = ws_lazy₀
      + delta_c * sensitivity_cross child₀ (2*pos) (2*pos+1) s_lo ell
      + delta_c * (2 * child₀ ⟨2*pos, by omega⟩ + delta_c) *
          self_mask (2*pos) s_lo ell
      + delta_c * (-(2 * child₀ ⟨2*pos+1, by omega⟩ - delta_c)) *
          self_mask (2*pos+1) s_lo ell
      + delta_c * 2 * (child₀ ⟨2*pos+1, by omega⟩ - child₀ ⟨2*pos, by omega⟩ - delta_c) *
          mutual_mask (2*pos) (2*pos+1) s_lo ell) :
    ws_lazy₁ = window_sum_int child₁ s_lo ell := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Sensitivity Incremental Update (Claim 6.4)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.4: When position j_changed advances (bins k3, k4 change by d3, d4 = -d3),
    each OTHER position a's sensitivity updates incrementally:
      sens'[a] = sens[a] + corrections from k3, k4 entering/leaving cross-term range.

    This ensures that after the GC step, sens_cross[a] still equals
    sensitivity_cross(child', k1a, k2a, s, ell) for the new child'.

    Matches: cascade_kernel.cu update_sensitivities_after_step lines 303-335. -/
theorem sensitivity_incremental_update
    {d : ℕ} (child child' : Fin d → ℤ)
    (pos_a pos_j : ℕ)
    (hpa : 2 * pos_a + 1 < d)
    (hpj : 2 * pos_j + 1 < d)
    (h_diff : pos_a ≠ pos_j)
    (delta_c : ℤ)
    (h_k3 : child' ⟨2*pos_j, by omega⟩ = child ⟨2*pos_j, by omega⟩ + delta_c)
    (h_k4 : child' ⟨2*pos_j+1, by omega⟩ = child ⟨2*pos_j+1, by omega⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ 2*pos_j ∧ i.1 ≠ 2*pos_j+1 → child' i = child i)
    (s_lo ell : ℕ) :
    let k1a := 2 * pos_a
    let k2a := k1a + 1
    let k3 := 2 * pos_j
    let k4 := k3 + 1
    sensitivity_cross child' k1a k2a s_lo ell =
      sensitivity_cross child k1a k2a s_lo ell
      + (if k3 ≠ k1a ∧ k3 ≠ k2a then
          (if k1a + k3 ∈ Finset.Icc s_lo (s_lo + ell - 2) then 2 * delta_c else 0) +
          (if k2a + k3 ∈ Finset.Icc s_lo (s_lo + ell - 2) then -(2 * delta_c) else 0)
         else 0)
      + (if k4 ≠ k1a ∧ k4 ≠ k2a then
          (if k1a + k4 ∈ Finset.Icc s_lo (s_lo + ell - 2) then 2 * (-delta_c) else 0) +
          (if k2a + k4 ∈ Finset.Icc s_lo (s_lo + ell - 2) then -(2 * (-delta_c)) else 0)
         else 0) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: W_int Lazy Update (Claim 6.5)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.5: W_int lazy update is correct.
    When child[k3] changes by +delta_c and child[k4] changes by -delta_c,
    W_int changes by delta_c if k3 is in the contributing bin range,
    and by -delta_c if k4 is in the contributing bin range.

    The contributing bin range for window (ell, s) is:
      lo_bin = max(0, s - (d-1)),  hi_bin = min(d-1, s + ell - 2)

    Matches: cascade_kernel.cu lazy_qc_update_and_check lines 386-393. -/
theorem w_int_lazy_update
    {d : ℕ} (child child' : Fin d → ℤ)
    (k3 k4 : ℕ) (hk3 : k3 < d) (hk4 : k4 < d)
    (delta_c : ℤ)
    (h_k3 : child' ⟨k3, hk3⟩ = child ⟨k3, hk3⟩ + delta_c)
    (h_k4 : child' ⟨k4, hk4⟩ = child ⟨k4, hk4⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ k3 ∧ i.1 ≠ k4 → child' i = child i)
    (lo_bin hi_bin : ℕ) :
    (∑ i ∈ Finset.Icc lo_bin hi_bin, (if h : i < d then child' ⟨i, h⟩ else 0)) =
    (∑ i ∈ Finset.Icc lo_bin hi_bin, (if h : i < d then child ⟨i, h⟩ else 0))
      + (if k3 ∈ Finset.Icc lo_bin hi_bin then delta_c else 0)
      + (if k4 ∈ Finset.Icc lo_bin hi_bin then -delta_c else 0) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Multi-Window Cache (Claims 6.6, 6.7)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.6: The multi-window cache of K windows maintains independent invariants.
    Each cached window w ∈ [0, K) satisfies the ws_lazy invariant (Claim 6.3)
    independently, because each has its own sensitivity array, ws_lazy, and W_int.

    Formally: if each window's invariant holds before a GC step, and the update
    is applied to all K windows, then each invariant holds after.

    Matches: cascade_kernel.cu lazy_qc_update_and_check lines 369-421
    (loop over cache_count windows). -/
theorem multi_window_cache_invariant
    {d : ℕ} (K : ℕ) (hK : 0 < K)
    (child₀ child₁ : Fin d → ℤ)
    (s_lo ell : Fin K → ℕ)
    (ws_lazy₀ ws_lazy₁ : Fin K → ℤ)
    (h_init : ∀ w : Fin K, ws_lazy₀ w = window_sum_int child₀ (s_lo w) (ell w))
    (pos : ℕ) (hpos : 2 * pos + 1 < d) (delta_c : ℤ)
    (h_step : child₁ ⟨2*pos, by omega⟩ = child₀ ⟨2*pos, by omega⟩ + delta_c)
    (h_step2 : child₁ ⟨2*pos+1, by omega⟩ = child₀ ⟨2*pos+1, by omega⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ 2*pos ∧ i.1 ≠ 2*pos+1 → child₁ i = child₀ i)
    (h_update : ∀ w : Fin K, ws_lazy₁ w = ws_lazy₀ w
      + delta_c * sensitivity_cross child₀ (2*pos) (2*pos+1) (s_lo w) (ell w)
      + delta_c * (2 * child₀ ⟨2*pos, by omega⟩ + delta_c) *
          self_mask (2*pos) (s_lo w) (ell w)
      + delta_c * (-(2 * child₀ ⟨2*pos+1, by omega⟩ - delta_c)) *
          self_mask (2*pos+1) (s_lo w) (ell w)
      + delta_c * 2 * (child₀ ⟨2*pos+1, by omega⟩ - child₀ ⟨2*pos, by omega⟩ - delta_c) *
          mutual_mask (2*pos) (2*pos+1) (s_lo w) (ell w)) :
    ∀ w : Fin K, ws_lazy₁ w = window_sum_int child₁ (s_lo w) (ell w) := by
  sorry

/-- Claim 6.7: Lazy QC prune implies actual prune (no false prunes).
    If ws_lazy > threshold for some cached window, then the actual window sum
    from conv also exceeds the threshold.

    This follows directly from the ws_lazy invariant (Claim 6.3): if
    ws_lazy = actual_ws and ws_lazy > threshold, then actual_ws > threshold.

    Matches: the conditional check at cascade_kernel.cu line 401. -/
theorem lazy_qc_no_false_prune
    {d : ℕ} (child : Fin d → ℤ) (s_lo ell : ℕ) (threshold : ℤ)
    (ws_lazy : ℤ)
    (h_inv : ws_lazy = window_sum_int child s_lo ell)
    (h_kill : ws_lazy > threshold) :
    window_sum_int child s_lo ell > threshold := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: Cache Seeding and Recompute (Claims 6.8, 6.9, 6.10)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.8: After a full conv recompute (e.g., after batch skip or subtree prune),
    recomputing sensitivity from scratch yields the same result as
    sensitivity_cross(child, k1, k2, s, ell).

    This is trivially true by definition — the GPU's precompute_qc_sensitivity
    computes exactly the cross-term sensitivity formula.

    Matches: cascade_kernel.cu precompute_qc_sensitivity lines 251-292. -/
theorem sensitivity_recompute_correct
    {d : ℕ} (child : Fin d → ℤ) (pos : ℕ) (hpos : 2 * pos + 1 < d) (s_lo ell : ℕ) :
    let k1 := 2 * pos
    let k2 := k1 + 1
    let sc_k1 := ∑ j : Fin d, if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k1 + j.1 ∈ Finset.Icc s_lo (s_lo + ell - 2)
                  then 2 * child j else 0
    let sc_k2 := ∑ j : Fin d, if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k2 + j.1 ∈ Finset.Icc s_lo (s_lo + ell - 2)
                  then 2 * child j else 0
    sc_k1 - sc_k2 = sensitivity_cross child k1 k2 s_lo ell := by
  sorry

/-- Claim 6.9: Self-term mask correctness.
    mask_k1[a] = 1 iff 2*k1 is in the window range [s_lo, s_lo+ell-2].

    Matches: cascade_kernel.cu precompute_qc_sensitivity lines 288-290. -/
theorem self_mask_correct (k s_lo ell : ℕ) :
    self_mask k s_lo ell = if 2 * k ∈ Finset.Icc s_lo (s_lo + ell - 2) then 1 else 0 := by
  sorry

/-- Claim 6.10: When seeding a QC cache slot from actual conv, the initial ws_lazy
    equals the actual window sum.

    This is the base case for the ws_lazy invariant.

    Matches: cascade_kernel.cu — after thread_private_window_scan finds a killing
    window, cache_ws_lazy[w] is seeded from the actual conv sum. -/
theorem cache_seeding_correct
    {d : ℕ} (child : Fin d → ℤ) (s_lo ell : ℕ)
    (ws_seed : ℤ) (h_seed : ws_seed = window_sum_int child s_lo ell) :
    ws_seed = window_sum_int child s_lo ell := by
  exact h_seed

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART G: End-to-End (Claims 6.11, 6.12)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.11: After each Gray code step, ALL K cached windows have their
    sensitivities updated. This ensures the invariant is preserved globally.

    Matches: cascade_kernel.cu lines 414-421 (loop updating all cached windows'
    sensitivities regardless of whether the child was killed). -/
theorem all_windows_updated_after_step
    {d : ℕ} (K : ℕ) (child child' : Fin d → ℤ)
    (pos_a : Fin K → ℕ) (pos_j : ℕ)
    (s_lo ell : Fin K → ℕ)
    (sens sens' : Fin K → ℤ)
    (hpj : 2 * pos_j + 1 < d)
    (delta_c : ℤ)
    (h_step : child' ⟨2*pos_j, by omega⟩ = child ⟨2*pos_j, by omega⟩ + delta_c)
    (h_step2 : child' ⟨2*pos_j+1, by omega⟩ = child ⟨2*pos_j+1, by omega⟩ - delta_c)
    (h_rest : ∀ i : Fin d, i.1 ≠ 2*pos_j ∧ i.1 ≠ 2*pos_j+1 → child' i = child i)
    (h_pre : ∀ w : Fin K, sens w = sensitivity_cross child (2 * pos_a w) (2 * pos_a w + 1)
              (s_lo w) (ell w))
    (h_update : ∀ w : Fin K, 2 * pos_a w + 1 < d → pos_a w ≠ pos_j →
      sens' w = sens w
        + (if 2*pos_j ≠ 2*(pos_a w) ∧ 2*pos_j ≠ 2*(pos_a w)+1 then
            (if 2*(pos_a w) + 2*pos_j ∈ Finset.Icc (s_lo w) (s_lo w + ell w - 2)
              then 2 * delta_c else 0) +
            (if 2*(pos_a w)+1 + 2*pos_j ∈ Finset.Icc (s_lo w) (s_lo w + ell w - 2)
              then -(2 * delta_c) else 0)
           else 0)
        + (if 2*pos_j+1 ≠ 2*(pos_a w) ∧ 2*pos_j+1 ≠ 2*(pos_a w)+1 then
            (if 2*(pos_a w) + (2*pos_j+1) ∈ Finset.Icc (s_lo w) (s_lo w + ell w - 2)
              then 2 * (-delta_c) else 0) +
            (if 2*(pos_a w)+1 + (2*pos_j+1) ∈ Finset.Icc (s_lo w) (s_lo w + ell w - 2)
              then -(2 * (-delta_c)) else 0)
           else 0)) :
    ∀ w : Fin K, 2 * pos_a w + 1 < d → pos_a w ≠ pos_j →
      sens' w = sensitivity_cross child' (2 * pos_a w) (2 * pos_a w + 1) (s_lo w) (ell w) := by
  sorry

/-- Claim 6.12: End-to-end soundness of lazy QC.
    The set of survivors produced by the lazy QC kernel is identical to the set
    produced by a kernel that recomputes the window sum from conv for every child.

    Proof sketch: By induction on Gray code steps. At each step, ws_lazy = actual_ws
    (Claim 6.3), so the pruning decision is identical. -/
theorem lazy_qc_end_to_end
    {d : ℕ} (child : Fin d → ℤ) (threshold : ℤ) (s_lo ell : ℕ)
    (ws_lazy : ℤ) (h_inv : ws_lazy = window_sum_int child s_lo ell) :
    (ws_lazy > threshold) ↔ (window_sum_int child s_lo ell > threshold) := by
  sorry

end -- noncomputable section
