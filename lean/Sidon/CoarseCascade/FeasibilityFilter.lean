/-
Sidon Autocorrelation Project — Feasibility Pre-Filter Soundness (Proof Stubs)

When transitioning from cascade level d to 2d, each parent bin with integer
mass p splits into children (c, p-c) where both c and p-c must satisfy
0 <= c <= x_cap and 0 <= p-c <= x_cap.

This means p <= 2*x_cap is NECESSARY for a valid child split to exist.
Parents with any bin p > 2*x_cap can be safely discarded.

BUG FIX (2026-04-13): Previously used p <= x_cap, which was too aggressive
and discarded valid parents. The correct bound is p <= 2*x_cap because
child splits (c, p-c) allow c up to x_cap AND p-c up to x_cap independently.

Source: run_cascade_coarse.py line 595, run_cascade_coarse_v2.py line 703.
  feasible = np.all(current <= 2 * x_cap, axis=1)
-/

import Sidon.CoarseCascade.IntegerThreshold

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- =============================================================================
-- PART 1: Child Split Feasibility
-- =============================================================================

/-- A child split (c, p-c) is feasible if both halves are in [0, x_cap]. -/
def feasible_split (p x_cap : ℕ) : Prop :=
  ∃ c : ℕ, c ≤ x_cap ∧ p - c ≤ x_cap ∧ c ≤ p

/-- **Feasibility necessary condition:** if a valid split exists, then p <= 2*x_cap.

    Proof: c <= x_cap and p - c <= x_cap, so p = c + (p-c) <= 2*x_cap. -/
theorem feasibility_necessary (p x_cap : ℕ) :
    feasible_split p x_cap → p ≤ 2 * x_cap := by
  sorry

/-- **Feasibility sufficient condition:** if p <= 2*x_cap, then a valid split exists.

    Construction: c = min(p, x_cap). Then c <= x_cap by definition.
    And p - c = p - min(p, x_cap) <= p - (p - x_cap) = x_cap when p >= x_cap,
    or p - c = 0 when p <= x_cap. Either way, p - c <= x_cap. -/
theorem feasibility_sufficient (p x_cap : ℕ) (h : p ≤ 2 * x_cap) :
    feasible_split p x_cap := by
  sorry

/-- **Feasibility iff:** p <= 2*x_cap is necessary and sufficient. -/
theorem feasibility_iff (p x_cap : ℕ) :
    feasible_split p x_cap ↔ p ≤ 2 * x_cap := by
  sorry

-- =============================================================================
-- PART 2: Pre-Filter Soundness
-- =============================================================================

/-- **Pre-filter soundness:** Discarding parents where any bin > 2*x_cap
    does not lose any parent that could produce unpruned children.

    If parent bin p_i > 2*x_cap, then no child split of bin i can have both
    halves in [0, x_cap], so no valid child exists. But since children must
    have all bins in [0, x_cap] (per_bin_mass_cap), this parent generates
    NO children at all. Discarding it is therefore safe.

    Source: run_cascade_coarse.py lines 593-599. -/
theorem prefilter_sound {d : ℕ} (S : ℕ) (c_target : ℝ)
    (parent : Fin d → ℕ) (h_sum : ∑ i, parent i = S)
    (x_cap : ℕ)
    (h_xcap_def : x_cap = Nat.floor ((S : ℝ) * Real.sqrt (c_target / (d : ℝ))))
    (i : Fin d) (h_infeasible : parent i > 2 * x_cap) :
    -- No valid child composition exists for this parent
    ¬∃ child : Fin (2 * d) → ℕ,
      (∀ j : Fin d,
        child ⟨2 * j.val, by omega⟩ + child ⟨2 * j.val + 1, by omega⟩ = parent j) ∧
      (∀ k : Fin (2 * d), child k ≤ x_cap) := by
  sorry

-- =============================================================================
-- PART 3: Old Filter Was Unsound (Counterexample Witness)
-- =============================================================================

/-- The OLD filter `p <= x_cap` was too aggressive: it discarded parents
    that DO have valid children.

    Counterexample: x_cap = 10, parent bin p = 15.
    Split c = 8, p-c = 7: both <= 10. Valid!
    But old filter rejects since 15 > 10.

    This means the old code could miss survivors, potentially invalidating
    the proof (survivors thought to be pruned were actually just not explored). -/
theorem old_filter_unsound :
    ∃ p x_cap : ℕ, p > x_cap ∧ feasible_split p x_cap := by
  sorry

-- =============================================================================
-- PART 4: Cursor Range Correctness
-- =============================================================================

/-- The cursor range for parent bin p is [max(0, p - x_cap), min(p, x_cap)].
    This is non-empty iff p <= 2*x_cap (the feasibility condition).

    Source: run_cascade_coarse.py process_parent(), lines 491-497:
      lo = max(0, p - x_cap)
      hi = min(p, x_cap)
      if lo > hi: return empty -/
theorem cursor_range_correct (p x_cap : ℕ) (h : p ≤ 2 * x_cap) :
    let lo := p - min p x_cap  -- max(0, p - x_cap) in natural number arithmetic
    let hi := min p x_cap
    lo ≤ hi ∧
    (∀ c, lo ≤ c → c ≤ hi → c ≤ x_cap ∧ p - c ≤ x_cap ∧ c ≤ p) ∧
    (∀ c, c ≤ x_cap → p - c ≤ x_cap → c ≤ p → lo ≤ c ∧ c ≤ hi) := by
  sorry

/-- The number of children per parent bin is (hi - lo + 1) = min(p, x_cap) - max(0, p-x_cap) + 1.
    Total children = product over all parent bins. -/
theorem children_count (p x_cap : ℕ) (h : p ≤ 2 * x_cap) :
    let lo := p - min p x_cap
    let hi := min p x_cap
    hi - lo + 1 = min p x_cap - (p - min p x_cap) + 1 := by
  sorry

end -- noncomputable section
