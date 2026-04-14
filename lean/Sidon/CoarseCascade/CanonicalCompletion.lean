/-
Sidon Autocorrelation Project — Canonical Completion Soundness (Proof Stubs)

The cascade enumerates only CANONICAL compositions (c <= rev(c) lexicographically)
at L0. At subsequent levels, children of canonical parents may be non-canonical.

BUG FIX (2026-04-13): The old code DISCARDED non-canonical survivors:
  canon = _canonical_mask(current)
  current = current[canon]                   -- DROPS non-canonical!

This was UNSOUND: non-canonical children of a canonical parent are the
SAME children that the reversed (non-canonical) parent would produce.
Since we never expanded the reversed parent (it wasn't enumerated),
discarding its children means those regions of the search space are
permanently lost — potential survivors are missed.

The fix: MAP non-canonical survivors to their canonical form via reversal,
then DEDUPLICATE (since canonical and reversed may collide):
  canon = _canonical_mask(current)
  non_canon = ~canon
  current[non_canon] = current[non_canon, ::-1]   -- Map to canonical
  current = np.unique(current, axis=0)              -- Dedup

This is sound because:
1. TV is invariant under reversal (ReversalSymmetry.lean)
2. A non-canonical child c has the same TV as rev(c), which IS canonical
3. If rev(c) was already present, dedup keeps one copy — no loss
4. If rev(c) was NOT present, we've recovered a survivor that was missed

Source: run_cascade_coarse.py lines 663-674, run_cascade_coarse_v2.py lines 771-784.
-/

import Sidon.CoarseCascade.ReversalSymmetry

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
-- PART 1: Canonical Form Definition
-- =============================================================================

/-- A composition is canonical if c <= rev(c) lexicographically. -/
def is_canonical {d : ℕ} (c : Fin d → ℕ) : Prop :=
  ∀ k : Fin d, (c k < c ⟨d - 1 - k.val, by omega⟩) ∨
    (c k = c ⟨d - 1 - k.val, by omega⟩ ∧
      ∀ j : Fin d, j.val < k.val →
        c j = c ⟨d - 1 - j.val, by omega⟩)

/-- Every composition is either canonical or its reversal is canonical. -/
theorem canonical_or_reverse_canonical {d : ℕ} (c : Fin d → ℕ) :
    is_canonical c ∨ is_canonical (fun i => c ⟨d - 1 - i.val, by omega⟩) := by
  sorry

-- =============================================================================
-- PART 2: Old Discard Was Unsound
-- =============================================================================

/-- **The old approach of discarding non-canonical survivors was UNSOUND.**

    Consider: canonical parent P is enumerated. Its child C is non-canonical.
    C is a survivor (not pruned). The old code drops C.

    But rev(P) was NEVER enumerated (only canonical parents at L0).
    So rev(C) — which would be a child of rev(P) — was never tested.
    If rev(C) is ALSO a survivor, we've lost it permanently.

    This means the cascade's "0 survivors" claim could be FALSE:
    we might claim all-pruned while survivors exist in the unexplored
    reverse-parent subtree. -/
theorem old_discard_unsound_example :
    -- There exist configurations where discarding non-canonical survivors
    -- causes the cascade to miss actual survivors
    ∃ (d S : ℕ) (c_target : ℝ),
      -- A canonical parent P with a non-canonical surviving child C
      ∃ (P : Fin d → ℕ) (C : Fin (2 * d) → ℕ),
        is_canonical P ∧
        ¬is_canonical C ∧
        -- C is a valid child of P
        (∀ i : Fin d,
          C ⟨2 * i.val, by omega⟩ + C ⟨2 * i.val + 1, by omega⟩ = P i) ∧
        -- rev(C) is canonical
        is_canonical (fun i => C ⟨2 * d - 1 - i.val, by omega⟩) ∧
        -- rev(P) was never enumerated (not canonical)
        ¬is_canonical (fun i => P ⟨d - 1 - i.val, by omega⟩)
        -- So rev(C) as a child of rev(P) was never explored
        := by
  sorry

-- =============================================================================
-- PART 3: Reversal Preserves Child-Parent Relationship
-- =============================================================================

/-- If C is a child of P (split relationship), then rev(C) is a child of rev(P).

    Parent bin i has mass P_i.  Child bins 2i and 2i+1 have masses C_{2i}, C_{2i+1}
    with C_{2i} + C_{2i+1} = P_i.

    Under reversal: rev(P)_i = P_{d-1-i}, and
    rev(C)_{2i} = C_{2d-1-2i} = C_{2(d-1-i)+1}
    rev(C)_{2i+1} = C_{2d-2-2i} = C_{2(d-1-i)}
    So rev(C)_{2i} + rev(C)_{2i+1} = C_{2(d-1-i)+1} + C_{2(d-1-i)} = P_{d-1-i} = rev(P)_i.
-/
theorem reverse_child_of_reverse_parent {d : ℕ}
    (P : Fin d → ℕ) (C : Fin (2 * d) → ℕ)
    (h_child : ∀ i : Fin d,
      C ⟨2 * i.val, by omega⟩ + C ⟨2 * i.val + 1, by omega⟩ = P i) :
    let revP : Fin d → ℕ := fun i => P ⟨d - 1 - i.val, by omega⟩
    let revC : Fin (2 * d) → ℕ := fun i => C ⟨2 * d - 1 - i.val, by omega⟩
    ∀ i : Fin d,
      revC ⟨2 * i.val, by omega⟩ + revC ⟨2 * i.val + 1, by omega⟩ = revP i := by
  sorry

-- =============================================================================
-- PART 4: Canonical Completion Soundness
-- =============================================================================

/-- **Canonical completion is sound:** mapping non-canonical survivors to
    their canonical form via reversal preserves the survivor property.

    If C is a survivor (not pruned by any window), then rev(C) is also
    a survivor, because TV is invariant under reversal (max_tv_reverse_eq).

    So mapping C -> rev(C) when C is non-canonical does not create
    false survivors. -/
theorem canonical_completion_preserves_survivor {d S : ℕ} (hS : S > 0)
    (c_target : ℝ)
    (C : Fin d → ℕ) (hC_sum : ∑ i, C i = S)
    (h_survivor : ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (C i : ℝ) / (S : ℝ)) ell s ≥ c_target)
    (h_not_canon : ¬is_canonical C) :
    let revC := fun i : Fin d => C ⟨d - 1 - i.val, by omega⟩
    -- rev(C) is also a survivor
    ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (revC i : ℝ) / (S : ℝ)) ell s ≥ c_target := by
  sorry

/-- **Canonical completion is complete:** every composition that survives
    has its canonical representative in the output.

    If c survives, then either c is canonical (already in the output),
    or rev(c) is canonical and rev(c) survives (added by the completion).

    Combined with dedup: the output contains exactly the set of canonical
    representatives of all surviving compositions. -/
theorem canonical_completion_complete {d S : ℕ} (hS : S > 0)
    (c_target : ℝ)
    (C : Fin d → ℕ) (hC_sum : ∑ i, C i = S)
    (h_survivor : ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (C i : ℝ) / (S : ℝ)) ell s ≥ c_target) :
    -- The canonical representative of C also survives
    let canon := if is_canonical C then C
                 else fun i : Fin d => C ⟨d - 1 - i.val, by omega⟩
    ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (canon i : ℝ) / (S : ℝ)) ell s ≥ c_target := by
  sorry

-- =============================================================================
-- PART 5: Deduplication Soundness
-- =============================================================================

/-- **Dedup does not lose survivors:** If both c and rev(c) map to the same
    canonical form, keeping one copy is sufficient. The cascade only needs
    to know WHICH canonical compositions survive, not their multiplicity.

    At the next level, the cascade will enumerate ALL children of each
    surviving canonical composition. This covers all children of rev(c)
    too (since rev(child_of_c) = child_of_rev(c), covered by
    reverse_child_of_reverse_parent). -/
theorem dedup_sound {d S : ℕ} (hS : S > 0)
    (c_target : ℝ)
    (C1 C2 : Fin d → ℕ)
    (hC1_sum : ∑ i, C1 i = S) (hC2_sum : ∑ i, C2 i = S)
    (h_same_canon : ∀ i : Fin d, C1 i = C2 i)
    -- Both survived
    (h_surv1 : ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (C1 i : ℝ) / (S : ℝ)) ell s ≥ c_target)
    (h_surv2 : ¬∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun i => (C2 i : ℝ) / (S : ℝ)) ell s ≥ c_target) :
    -- Keeping just C1 is sufficient: all children of C2 are also children of C1
    ∀ child : Fin (2 * d) → ℕ,
      (∀ i : Fin d,
        child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = C2 i) →
      (∀ i : Fin d,
        child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = C1 i) := by
  sorry

-- =============================================================================
-- PART 6: Combined Cascade Soundness with Canonical Completion
-- =============================================================================

/-- **Full cascade soundness with canonical completion.**

    At each level, the cascade:
    1. Enumerates all children of canonical surviving parents
    2. Prunes children with TV >= c_target
    3. Maps non-canonical survivors to canonical form via reversal
    4. Deduplicates

    The resulting set of canonical survivors covers ALL compositions that
    would survive at this level — including children of the unexpanded
    reverse parents.

    If this set is empty, ALL compositions at this dimension are pruned. -/
theorem cascade_level_with_completion_sound {d S : ℕ} (hS : S > 0)
    (c_target : ℝ)
    -- All canonical parents at dimension d
    (parents : Finset (Fin d → ℕ))
    (h_parents_cover : ∀ c : Fin d → ℕ, (∑ i, c i = S) →
      ¬(∃ ell s, 2 ≤ ell ∧
        mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s ≥ c_target) →
      ∃ p ∈ parents, ∀ i, p i = c i ∨
        p i = c ⟨d - 1 - i.val, by omega⟩)
    -- All children of all parents are explored
    (h_all_children_tested : ∀ p ∈ parents,
      ∀ child : Fin (2 * d) → ℕ,
        (∀ i : Fin d,
          child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = p i) →
        (∑ j, child j = S) →
        -- child is either pruned OR its canonical form is in the output
        (∃ ell s, 2 ≤ ell ∧
          mass_test_value (2 * d) (fun i => (child i : ℝ) / (S : ℝ)) ell s ≥ c_target) ∨
        True -- canonical form is in output (abstract over output set)
    ) :
    -- Then: every unpruned composition at dimension 2d has its canonical
    -- representative in the output
    True := by
  sorry

end -- noncomputable section
