# Lean Repair Plan

## 1. What We Changed in the Python Code

### Change 1: Correction constant 3 -> 1

In every W-refined pruning threshold, we changed the constant from 3 to 1.

**Files:** `run_cascade.py` (5 locations), `solvers.py` (7 locations), `cascade_host.cu` (2 locations)

**Before (TV-space):** `c_target + (3 + W_int/(2n)) / m^2`
**After (TV-space):** `c_target + (1 + W_int/(2n)) / m^2`

### Change 2: Fixed W_int coefficient bug in _prove_target_d4

**File:** `solvers.py:998`

**Before:** `corr_w = (3.0 + 2.0 * W_int) * inv_m_sq`
**After:** `corr_w = (1.0 + float(W_int) / (2.0 * n_half)) * inv_m_sq`

The old coefficient `2*W_int` was 8x too large for n_half=2 (should be `W_int/4`).

### Change 3: Updated correction_max in solvers.py

**Before:** `correction_max = 3.0 * inv_m_sq + 2.0 * inv_m`
**After:** `correction_max = 1.0 * inv_m_sq + 2.0 * inv_m`

### What we did NOT change

- The **flat threshold** (`--use_flat_threshold`) is UNCHANGED:
  `flat_corr = 2*m + 1`, i.e., C&S Lemma 3 correction `2/m + 1/m^2`.
- `pruning.py` `correction()` function is UNCHANGED.
- `test_values.py` is UNCHANGED.

---

## 2. Why We Changed It

The MATLAB code (Cloninger & Steinerberger, used to prove C_{1a} >= 1.28) uses:
```matlab
boundToBeat = (lowerBound + gridSpace^2) + 2*gridSpace * W;
% = c + 1/m^2 + 2*W_g/m   (constant 1, not 3)
```

Our constant 3 added an unnecessary `2/m^2` safety margin (bounding `W_f <= W_g + 1/m`).
At convolution knot points, `W_f = W_g` exactly, so this overhead is not needed.

**Impact (from threshold_analysis.md):**
- L1 pruning went from 25.7% to ~91.7% per parent at c=1.28
- Expansion factor improved ~8.9x, matching MATLAB

---

## 3. The Current Lean Proof Structure

The Lean proof establishes `C_{1a} >= 32/25 = 1.28`. It relies on exactly
two axioms — statements trusted without Lean-internal proof:

### Axiom 1: `cs_lemma3_per_window` (DiscretizationError.lean:167)

```
TV_discrete(c, ell, s) - TV_continuous(f, ell, s) <= 2/m + 1/m^2
```

This is C&S Lemma 3 from arXiv:1403.7988. It says the test value of the
discrete step function g (with heights c_i/m) exceeds the test value of
the underlying continuous function f by at most `2/m + 1/m^2`. This is a
published, peer-reviewed mathematical result. Formalizing it would require
~200-300 lines of piecewise integration against Mathlib's MeasureTheory,
which is why it remains an axiom.

### Axiom 2: `cascade_all_pruned` (FinalResult.lean:68)

```
forall c : Fin 4 -> N, sum c = 160 ->
  exists ell s_lo, TV(2, 20, c, ell, s_lo) > 32/25 + 2/20 + 1/400
```

This is a computational claim: every composition of 160 into 4 bins has
some window where the test value exceeds the threshold 1.3825. It is
meant to be verified by running the Python cascade with `--use_flat_threshold`.

### How the proof works

The main theorem `autoconvolution_ratio_ge_32_25` (FinalResult.lean:133)
proves R(f) >= 32/25 for any admissible f:

1. Normalize f to g with integral 1
2. Discretize g to composition c at n_half=2, m=20 (d=4 bins, sum=160)
3. Apply `cascade_all_pruned`: get window (ell, s_lo) where TV > 1.3825
4. Apply `dynamic_threshold_sound_cs` (proved from Axiom 1): since
   TV > c_target + 2/m + 1/m^2, conclude R(g) >= c_target
5. Since R is scale-invariant, R(f) = R(g) >= 32/25

The intermediate theorem `dynamic_threshold_sound_cs` (DiscretizationError.lean:194)
is fully proved from Axiom 1 — it is not an axiom.

---

## 4. The Fatal Axiom Bug (Pre-existing)

### The axiom is provably false

```lean
axiom cascade_all_pruned :
  forall c : Fin (2 * 2) -> N, sum i, c i = 4 * 2 * 20 ->
    exists ell s_lo, 2 <= ell /\
      test_value 2 20 c ell s_lo > (32/25 : R) + 2 / 20 + 1 / 20 ^ 2
```

This claims **every** d=4 composition has a window where the d=4 test value
exceeds 1.3825.

### Counterexample: c = [40, 40, 40, 40]

The uniform composition c = [40, 40, 40, 40] (sum = 160) gives heights
a = c/m = [2, 2, 2, 2].

The autoconvolution conv[k] = sum_{i+j=k} a_i * a_j:

```
k:      0    1    2    3    4    5    6
conv:   4    8   12   16   12    8    4
```

The test value TV(n, m, c, ell, s_lo) = (1/(4*n*ell)) * sum_{k=s_lo}^{s_lo+ell-2} conv[k].

Exhaustive check over all valid (ell, s_lo):

| ell | best s_lo | window sum | TV = sum/(4*2*ell) |
|-----|-----------|------------|-------------------|
| 2   | 3         | 16         | 1.000             |
| 3   | 2         | 28         | 1.167             |
| 4   | 2         | 40         | **1.250**         |
| 5   | 1         | 48         | 1.200             |
| 6   | 1         | 56         | 1.167             |
| 7   | 0         | 60         | 1.071             |
| 8   | 0         | 64         | 1.000             |

Maximum TV = **1.25** (at ell=4, s=2).

Since 1.25 < 1.3825, the axiom's claim `TV > 1.3825` is false for this
composition. No window exceeds the threshold.

### What happens to the uniform composition in the cascade

The cascade does NOT claim the uniform composition is pruned at d=4.
It survives L0 and gets refined:

- **L0 (d=4):** c = [40,40,40,40], max TV = 1.25 < 1.3825. SURVIVES.
- **L1 (d=8):** Each parent bin 40 splits into two child bins summing to 80.
  Children like [40,40,40,40,40,40,40,40] have higher TV at d=8.
  Most children get pruned. Some survive and go to L2.
- **L2 (d=16):** Further refinement. More children pruned.
- **...continues until all branches terminate.**

The cascade terminates with 0 survivors across ALL levels combined. But the
axiom claims something about a SINGLE level (d=4), which is false.

### Why this matters

In Lean, any statement can be proved from a false axiom (ex falso quodlibet).
The main theorem `autoconvolution_ratio_ge_32_25` typechecks, but since it
depends on a false axiom, it does not constitute a valid proof of C_{1a} >= 1.28.
The conclusion happens to be mathematically true, but the Lean formalization
does not establish it.

This bug predates our Python changes. Our constant 3->1 change did not
introduce it and does not affect it (the flat threshold path is unchanged).

---

## 5. What the Cascade Actually Proves

The cascade proves a **multi-level** property:

> For every d=4 composition c0, either:
> - c0 is directly pruned (TV > threshold at d=4), OR
> - ALL children of c0 at d=8 are either directly pruned or have all
>   THEIR children pruned at d=16, etc.
>
> Eventually every branch terminates.

This is an inductive/tree-shaped property, not a flat universal statement
about d=4 test values.

### Why this implies R(f) >= c_target

Take any continuous f with integral 1. Its canonical discretization at d=4
is some composition c0. The cascade proves c0 is "cascade-pruned." We need
to show R(f) >= c_target.

**Case 1:** c0 is directly pruned at d=4. Some window has TV(c0) > threshold.
By `dynamic_threshold_sound_cs`, R(f) >= c_target. Done.

**Case 2:** c0 survives d=4, but all its children are cascade-pruned. The key
fact is that f also has a canonical discretization at d=8, and this d=8
discretization is one of c0's children. So f's d=8 discretization is
cascade-pruned. Recurse: either it's directly pruned (giving R(f) >= c_target)
or we go deeper.

The recursion terminates because the cascade terminates (finite depth).

### The three pieces of the argument

1. **Pruning soundness** (already proved in Lean):
   If TV(c, ell, s) > c_target + 2/m + 1/m^2, then R(f) >= c_target
   for any f whose discretization is c.
   → `dynamic_threshold_sound_cs` in DiscretizationError.lean

2. **Enumeration completeness** (already proved in Lean):
   The cascade algorithm tests every valid child of every parent.
   → `gray_code_subtree_pruning_sound` in Algorithm/ layer

3. **Refinement preserves discretization** (MISSING from Lean):
   If f discretizes to c at d=4, then f discretizes to some valid child
   of c at d=8. This links the cascade levels together.
   → Needs to be proved (Section 6.3)

---

## 6. The Fix

### Axiom count: stays at exactly 2

We keep exactly two axioms:
1. `cs_lemma3_per_window` — unchanged (published mathematical result)
2. `cascade_all_pruned` — REFORMULATED to be true (computational result)

Everything else is PROVED, not axiomatic.

The W-refined threshold (our constant-1 change) is a Python-only optimization
for faster cascade convergence. The Lean proof uses `--use_flat_threshold`
which is unchanged. For c=1.28, the flat-threshold cascade terminates, so the
flat correction `2/m + 1/m^2` suffices.

### 6.1 Reformulate cascade_all_pruned (REPLACE, not add)

Define an inductive predicate that captures the multi-level cascade:

```lean
/-- A composition is cascade-pruned if either it directly exceeds the
    threshold, or ALL its children are cascade-pruned.
    
    This mirrors the cascade algorithm: at each level, the code either
    prunes a composition (TV > threshold) or refines it into children
    and processes each child recursively. The cascade terminating with
    0 survivors means every root composition is CascadePruned. -/
inductive CascadePruned (m : ℕ) (c_target correction : ℝ) :
    (n_half : ℕ) → (Fin (2 * n_half) → ℕ) → Prop where
  | direct {n_half} (c : Fin (2 * n_half) → ℕ)
      (h : ∃ ℓ s_lo, 2 ≤ ℓ ∧
        test_value n_half m c ℓ s_lo > c_target + correction) :
      CascadePruned m c_target correction n_half c
  | refine {n_half} (c : Fin (2 * n_half) → ℕ)
      (h : ∀ child : Fin (2 * (2 * n_half)) → ℕ,
        is_valid_child n_half m c child →
        CascadePruned m c_target correction (2 * n_half) child) :
      CascadePruned m c_target correction n_half c
```

Replace the axiom:

```lean
/-- Computational axiom (REPLACES the old cascade_all_pruned).
    
    Every composition of 160 into 4 bins is cascade-pruned with
    n_half=2, m=20, c_target=32/25, correction=2/20+1/400.
    
    Verified by: python -m cloninger-steinerberger.cpu.run_cascade \
      --n_half 2 --m 20 --c_target 1.28 --use_flat_threshold
    
    The cascade runs L0 -> L1 -> L2 -> ... until 0 survivors remain.
    Compositions pruned at level k use the `direct` constructor.
    Compositions that survive level k but whose children are all pruned
    use the `refine` constructor. -/
axiom cascade_all_pruned :
  ∀ c : Fin 4 → ℕ, ∑ i, c i = 160 →
    CascadePruned 20 (32/25) (2/20 + 1/20^2) 2 c
```

### Why the new axiom is true

For c = [40, 40, 40, 40] (which falsified the old axiom):

- max TV at d=4 is 1.25 < 1.3825, so `direct` does NOT apply
- But `refine` applies: every valid d=8 child of [40,40,40,40] is
  cascade-pruned. The cascade verifies this by testing all children at
  L1, then their children at L2, etc., until every branch terminates.
- So `CascadePruned` holds via the `refine` constructor.

For compositions that ARE pruned at d=4 (TV > 1.3825), `direct` applies.

### 6.2 Prove cascade_pruned_implies_bound (NEW THEOREM, not axiom)

```lean
/-- If a composition is cascade-pruned, then every continuous function
    whose canonical discretization matches it has R(f) >= c_target.
    
    Proof by induction on the CascadePruned derivation:
    - direct: f's discretization has high TV, so R(f) >= c_target
      by dynamic_threshold_sound_cs (already proved)
    - refine: f also discretizes at the finer grid to some child of c.
      That child is cascade-pruned (by hypothesis). Apply induction. -/
theorem cascade_pruned_implies_bound
    (m n_half : ℕ) (c_target : ℝ)
    (hm : m > 0) (hn : n_half > 0) (hct : 0 < c_target)
    (c : Fin (2 * n_half) → ℕ)
    (h_pruned : CascadePruned m c_target (2/m + 1/m^2) n_half c) :
    ∀ f : ℝ → ℝ, admissible f →
      canonical_discretization f n_half m = c →
      autoconvolution_ratio f ≥ c_target := by
  induction h_pruned with
  | direct c h_exceeds =>
      -- TV exceeds threshold directly. Use existing dynamic_threshold_sound_cs.
      intro f hf hdisc
      obtain ⟨ℓ, s_lo, hℓ, h_tv⟩ := h_exceeds
      exact dynamic_threshold_sound_cs n_half m c_target hn hm hct c ℓ s_lo hℓ
        (by rw [← hdisc]; exact h_tv) f ... hdisc.symm ▸ rfl
  | refine c h_children ih =>
      -- All children cascade-pruned. f discretizes to some child. Apply ih.
      intro f hf hdisc
      obtain ⟨child, h_valid, h_child_disc⟩ :=
        refinement_preserves_discretization f n_half m c hf hdisc
      exact ih child h_valid f hf h_child_disc
```

The `direct` case reuses `dynamic_threshold_sound_cs` directly — this theorem
is already proved in the current Lean. No new math.

The `refine` case uses the induction hypothesis plus the key lemma below.

### 6.3 Prove refinement_preserves_discretization (NEW THEOREM, not axiom)

```lean
/-- If f discretizes to c at resolution n_half, then f discretizes
    to some valid child of c at resolution 2*n_half.
    
    "Valid child" means: child[2i] + child[2i+1] = 2*c[i] for all i
    (each parent bin splits into two child bins preserving total mass).
    
    This follows from how canonical_discretization works:
    - It floors cumulative masses: D(k) = floor(cum_mass(k) * S)
    - The finer grid at 2*n_half has bin boundaries at BOTH the
      coarse boundaries AND the midpoints of coarse bins
    - At coarse boundaries, the finer D agrees with the coarser D
      (same cumulative mass, same floor target)
    - Therefore child[2i] + child[2i+1] = c_coarse[i] -/
theorem refinement_preserves_discretization
    (f : ℝ → ℝ) (n_half m : ℕ) (c : Fin (2 * n_half) → ℕ)
    (hf : admissible f) (hdisc : canonical_discretization f n_half m = c) :
    ∃ child : Fin (2 * (2 * n_half)) → ℕ,
      is_valid_child n_half m c child ∧
      canonical_discretization f (2 * n_half) m = child
```

This is the hardest piece. The proof needs to show that doubling the grid
resolution produces a valid bin-splitting. The argument is:

**Bin structure at each resolution:**
- At resolution n_half: d = 2*n_half bins, each of width delta = 1/(4*n_half).
  Bin i covers [i*delta - 1/4, (i+1)*delta - 1/4].
- At resolution 2*n_half: d' = 4*n_half bins, each of width delta/2.
  Bins 2i and 2i+1 together cover the same interval as parent bin i.

**Cumulative mass and floor-rounding:**
- canonical_discretization computes D(k) = floor(sum of first k bin masses * S)
- At the coarser grid, D_coarse(k) uses bins 0..k-1 of the coarse grid
- At the finer grid, D_fine(2k) uses bins 0..2k-1, which cover the same
  physical interval as coarse bins 0..k-1
- Therefore D_fine(2k) = D_coarse(k) for all k (same cumulative mass,
  same S = 4*n_half*m at coarse or 4*(2*n_half)*m at fine... wait, S
  changes too)

**The S parameter:** S = 4*n_half*m at resolution n_half. At resolution
2*n_half, S' = 4*(2*n_half)*m = 2*S. So the finer grid has twice as many
quanta. This means D_fine(2k) = 2*D_coarse(k) (twice the cumulative mass
target). And child[2i] + child[2i+1] = D_fine(2i+2) - D_fine(2i) =
2*(D_coarse(i+1) - D_coarse(i)) = 2*c[i]. This is the `is_valid_child`
condition (each parent bin's mass doubles and splits into two children).

The existing `RefinementMass.lean` already has partial infrastructure for
mass conservation under refinement. The key new piece is connecting
`canonical_discretization` at two resolutions through the cumulative
distribution function.

### 6.4 Update main theorem

Replace the proof body of `autoconvolution_ratio_ge_32_25` to:
1. Discretize f to c at n_half=2, m=20
2. Apply `cascade_all_pruned` to get `CascadePruned 20 (32/25) ... 2 c`
3. Apply `cascade_pruned_implies_bound` to conclude R(f) >= 32/25

### 6.5 Algorithm-layer formula updates (MINOR)

Update `dynamic_threshold_monotone` and `threshold_formula_consistency` in
`GrayCodeSubtreePruning.lean` to match the Python formula:

**Current:** `c_target * m^2 * ell * inv_4n + 1 + eps + 2*W`
(only c*m^2 scaled by ell/(4n))

**Should be:** `(c_target * m^2 + 1 + eps + W/(2n)) * 4n*ell`
(everything scaled together)

Update comments throughout to reflect the new formula.

---

## 7. Summary

| Change | Type | Why |
|--------|------|-----|
| Reformulate `cascade_all_pruned` with `CascadePruned` | Replace axiom (still 1 axiom) | Current axiom is false |
| Prove `cascade_pruned_implies_bound` | New theorem | Connect new axiom to main theorem |
| Prove `refinement_preserves_discretization` | New theorem | Required for inductive step |
| Update `autoconvolution_ratio_ge_32_25` proof body | Modify theorem | Use new axiom |
| Update algorithm-layer formulas + comments | Modify theorems | Sync with Python code |

**Total axioms before: 2. Total axioms after: 2.** No axioms added.
Everything new is proved, not assumed.
