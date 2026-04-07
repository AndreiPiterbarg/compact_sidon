/-
Sidon Autocorrelation Project — Arc Consistency / Range Tightening (Claims 6.30–6.36)

This file collects the theorems and lemmas certifying the arc consistency
(range tightening) optimization implemented in both CPU and GPU:
  - CPU: run_cascade.py _tighten_ranges (lines ~1780-1900)
  - GPU: cascade_host.cu tighten_ranges (lines 583-733)

The optimization pre-tightens per-bin cursor ranges [lo[i], hi[i]] before
the main enumeration loop. For each position p and edge value v, it checks:
  "If position p takes value v and all other positions take their minimum-
   contribution values, does some window already exceed the threshold?"
If yes, v is infeasible for ALL children and can be removed from the range.

This is sound because: if v causes pruning even when all other positions
are at their most favorable (minimum-contribution) values, then v causes
pruning for every combination of other positions' values.

Critical property: NO valid child is excluded. If a child would have survived
the full window scan, it must also survive the tightened-range enumeration.

Claims covered:
  6.30  Minimum-contribution child is a valid lower bound on window sum
  6.31  If min-contribution child exceeds threshold, all children with that
        edge value exceed the threshold (monotonicity in other positions)
  6.32  Range tightening from low end preserves all survivors
  6.33  Range tightening from high end preserves all survivors
  6.34  Fixed-point convergence: iterating tightening terminates
  6.35  Empty range detection: if any range empties, parent has no valid children
  6.36  End-to-end: tightened ranges produce identical survivor set

Cross-cutting dependencies:
  - SubtreePruning.lean (Claim 4.4): partial conv ≤ full conv
  - CauchySchwarz.lean: bin range computation (x_cap formula)
  - DynamicThreshold.lean: threshold formula

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

/-- Minimum-contribution child: position p takes value v, all other positions
    take their minimum-window-contribution value (lo[i] for most windows). -/
def min_contribution_child {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo : Fin d_parent → ℕ) (p : Fin d_parent) (v : ℕ) : Fin (2 * d_parent) → ℕ :=
  fun i =>
    let q := i.1 / 2
    if h : q < d_parent then
      if q = p.1 then
        if i.1 % 2 = 0 then v else parent ⟨q, h⟩ - v
      else
        if i.1 % 2 = 0 then lo ⟨q, h⟩ else parent ⟨q, h⟩ - lo ⟨q, h⟩
    else 0

/-- A value v is feasible at position p if there exists at least one child
    with cursor[p] = v that survives (is not pruned by any window). -/
def feasible_value {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ) (p : Fin d_parent) (v : ℕ)
    (threshold : ℕ → ℕ → ℤ) : Prop :=
  ∃ (cursor : Fin d_parent → ℕ),
    cursor p = v ∧
    (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) ∧
    let child : Fin (2 * d_parent) → ℕ := fun i =>
      let q := i.1 / 2
      if h : q < d_parent then
        if i.1 % 2 = 0 then cursor ⟨q, h⟩ else parent ⟨q, h⟩ - cursor ⟨q, h⟩
      else 0
    ¬ ∃ ell s_lo,
      (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
        (∑ i : Fin (2 * d_parent), ∑ j : Fin (2 * d_parent),
          if i.1 + j.1 = k then (child i : ℤ) * child j else 0)) >
      threshold ell s_lo

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Monotonicity (Claims 6.30, 6.31)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.30: The minimum-contribution child provides a lower bound on
    the window sum for any child with the same value at position p.

    When all other positions take their lowest values (lo[i]), the cross-term
    contributions are minimized. Since all child values are nonneg, the window
    sum can only increase when other positions take higher values.

    Matches: cascade_host.cu tighten_ranges lines 620-650 (build child_min). -/
theorem min_contribution_lower_bound
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ) (h_le : ∀ i, lo i ≤ hi i)
    (p : Fin d_parent) (v : ℕ) (hv_lo : lo p ≤ v) (hv_hi : v ≤ hi p)
    (cursor : Fin d_parent → ℕ)
    (h_cursor : cursor p = v)
    (h_bounds : ∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i)
    (ell s_lo : ℕ) :
    let child_min := min_contribution_child parent lo p v
    let child_actual : Fin (2 * d_parent) → ℕ := fun i =>
      let q := i.1 / 2
      if h : q < d_parent then
        if i.1 % 2 = 0 then cursor ⟨q, h⟩ else parent ⟨q, h⟩ - cursor ⟨q, h⟩
      else 0
    (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
      (∑ i : Fin (2 * d_parent), ∑ j : Fin (2 * d_parent),
        if i.1 + j.1 = k then (child_min i : ℤ) * child_min j else 0)) ≤
    (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
      (∑ i : Fin (2 * d_parent), ∑ j : Fin (2 * d_parent),
        if i.1 + j.1 = k then (child_actual i : ℤ) * child_actual j else 0)) := by
  sorry

/-- Claim 6.31: If the minimum-contribution child with cursor[p]=v exceeds the
    threshold for some window, then ALL children with cursor[p]=v exceed the
    threshold for that window.

    Follows from Claim 6.30: ws_min ≤ ws_actual for all windows. If ws_min >
    threshold, then ws_actual ≥ ws_min > threshold.

    Matches: cascade_host.cu tighten_ranges lines 655-680 (infeasibility check). -/
theorem infeasible_value_prunable
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ) (h_le : ∀ i, lo i ≤ hi i)
    (p : Fin d_parent) (v : ℕ) (hv_lo : lo p ≤ v) (hv_hi : v ≤ hi p)
    (threshold : ℤ) (ell s_lo : ℕ)
    (h_min_exceeds :
      let child_min := min_contribution_child parent lo p v
      (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
        (∑ i : Fin (2 * d_parent), ∑ j : Fin (2 * d_parent),
          if i.1 + j.1 = k then (child_min i : ℤ) * child_min j else 0)) > threshold) :
    ∀ cursor : Fin d_parent → ℕ,
      cursor p = v → (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) →
      let child_actual : Fin (2 * d_parent) → ℕ := fun i =>
        let q := i.1 / 2
        if h : q < d_parent then
          if i.1 % 2 = 0 then cursor ⟨q, h⟩ else parent ⟨q, h⟩ - cursor ⟨q, h⟩
        else 0
      (∑ k ∈ Finset.Ico s_lo (s_lo + ell - 1),
        (∑ i : Fin (2 * d_parent), ∑ j : Fin (2 * d_parent),
          if i.1 + j.1 = k then (child_actual i : ℤ) * child_actual j else 0)) > threshold := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Survivor Preservation (Claims 6.32, 6.33)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.32: Tightening lo[p] from v to v+1 preserves all survivors.
    If value v is infeasible (pruned by some window for all children), then
    no survivor uses cursor[p] = v, so removing it loses nothing.

    Matches: cascade_host.cu tighten_ranges lines 685-700 (tighten from low end). -/
theorem tighten_lo_preserves_survivors
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (p : Fin d_parent) (v : ℕ) (hv : lo p = v)
    (h_infeasible : ¬ feasible_value parent lo hi p v (fun _ _ => 0)) :
    ∀ cursor : Fin d_parent → ℕ,
      (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) →
      cursor p ≠ v →
      (∀ i, (if i = p then v + 1 else lo i) ≤ cursor i ∧ cursor i ≤ hi i) := by
  sorry

/-- Claim 6.33: Tightening hi[p] from v to v-1 preserves all survivors.
    Symmetric to Claim 6.32.

    Matches: cascade_host.cu tighten_ranges lines 700-715 (tighten from high end). -/
theorem tighten_hi_preserves_survivors
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (p : Fin d_parent) (v : ℕ) (hv : hi p = v)
    (h_infeasible : ¬ feasible_value parent lo hi p v (fun _ _ => 0)) :
    ∀ cursor : Fin d_parent → ℕ,
      (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) →
      cursor p ≠ v →
      (∀ i, lo i ≤ cursor i ∧ cursor i ≤ (if i = p then v - 1 else hi i)) := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Termination and Completeness (Claims 6.34–6.36)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.34: Fixed-point convergence. Each tightening round strictly reduces
    the total range size ∑(hi[i] - lo[i] + 1) or makes no change. Since the
    total range is bounded below by 0, the iteration terminates in at most
    ∑(hi₀[i] - lo₀[i] + 1) rounds.

    Matches: cascade_host.cu tighten_ranges lines 605-610 (iterate up to
    d_parent rounds until convergence). -/
theorem tightening_terminates
    {d_parent : ℕ} (lo hi : Fin d_parent → ℕ) :
    ∃ (n : ℕ), n ≤ ∑ i : Fin d_parent, (hi i - lo i + 1) := by
  sorry

/-- Claim 6.35: If any range becomes empty (lo[p] > hi[p]) after tightening,
    the parent has no valid children. All children are prunable.

    Matches: cascade_host.cu tighten_ranges lines 720-725 (return false if
    any range empties). -/
theorem empty_range_no_children
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi : Fin d_parent → ℕ)
    (p : Fin d_parent) (h_empty : hi p < lo p)
    (threshold : ℕ → ℕ → ℤ) :
    ¬ ∃ cursor : Fin d_parent → ℕ,
      (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) := by
  sorry

/-- Claim 6.36: End-to-end arc consistency soundness.
    The set of survivors enumerated over tightened ranges [lo', hi'] is identical
    to the set of survivors enumerated over original ranges [lo, hi].

    Proof sketch: By induction on tightening rounds. Each round either removes
    infeasible edge values (no survivors lost by Claim 6.32/6.33) or converges.
    No new survivors are created because tightened ranges are subsets of originals.

    Matches: cascade_host.cu tighten_ranges — the complete function. -/
theorem arc_consistency_end_to_end
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (lo hi lo' hi' : Fin d_parent → ℕ)
    (h_sub : ∀ i, lo i ≤ lo' i ∧ hi' i ≤ hi i)
    (h_tight : ∀ i, lo' i ≤ hi' i)
    (threshold : ℕ → ℕ → ℤ)
    (h_sound : ∀ p v, lo' p ≤ v → v ≤ hi' p →
      feasible_value parent lo hi p v threshold →
      feasible_value parent lo' hi' p v threshold) :
    ∀ cursor : Fin d_parent → ℕ,
      (∀ i, lo i ≤ cursor i ∧ cursor i ≤ hi i) →
      feasible_value parent lo hi (⟨0, by omega⟩) (cursor ⟨0, by omega⟩) threshold →
      (∀ i, lo' i ≤ cursor i ∧ cursor i ≤ hi' i) := by
  sorry

end -- noncomputable section
