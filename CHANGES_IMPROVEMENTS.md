# Algorithmic Changes from Original Cloninger-Steinerberger

This document lists every genuine algorithmic change made to the original
Cloninger & Steinerberger algorithm (arXiv:1403.7988) in this codebase.
Only CPU-side changes are listed. Implementation-only changes (e.g. using
Numba instead of pure Python) are excluded unless they enabled a new
algorithmic technique.

---

## Measurement Plan

Each change below needs a quantified impact. The changes fall into four
categories, each requiring a different measurement approach.

### Category A: Pruning Power (changes that affect how many children survive)

**Changes:** §1, §2, §3, §4, §12, §13, §15, §16

**How to measure:** Run a fixed cascade level (e.g. L2→L3 with n_half=2,
m=20, c_target=1.40) twice: once with the change enabled, once with it
disabled (or replaced by the baseline approach). Compare:

- **Survivor count** at the end of the level.
- **Pruning rate** = 1 − (survivors / total_children_tested).

For §1 and §2, the baseline is the flat C&S Lemma 3 threshold (already
implemented as `use_flat_threshold=True`). Run both modes on the same
parent set and compare survivor counts.

For §4 (asymmetry), count parents skipped by the asymmetry filter vs
total parents. This is a simple counter added to the cascade loop.

For §12 and §13 (subtree pruning), the code already tracks
`n_subtree_pruned`. Compare runs with `J_MIN` set to a value beyond
`n_active` (disabling subtree pruning) vs the default `J_MIN=7`.

For §15 and §16, disable the early-out check and measure whether the
survivor count changes (it should not — these are performance
optimisations that do not affect pruning power, only speed). If the
survivor count is identical, these belong in Category B instead.

### Category B: Per-Child Cost (changes that reduce work per child without affecting which children survive)

**Changes:** §5, §6, §7, §8, §9, §10, §11, §15, §16, §21

**How to measure:** Run the fused kernel on a fixed parent set (e.g.
1000 parents from L2 checkpoint). Measure **wall-clock time** and
**children processed per second**. Disable one optimisation at a time:

| Change | How to disable |
|--------|---------------|
| §5 (fused kernel) | Use `_fused_generate_and_prune` (non-Gray) vs `_fused_generate_and_prune_gray` |
| §6 (incremental conv) | Replace incremental update with full O(d²) recompute after each Gray step |
| §7 (sparse nz list) | Set `use_sparse = False` unconditionally |
| §8 (quick-check) | Set `qc_ell = 0` permanently (never cache killing window) |
| §9 (incremental QC W_int) | Recompute `qc_W_int` from scratch each step instead of incrementally |
| §10 (threshold table) | Compute threshold inline with float64 arithmetic instead of table lookup |
| §11 (ℓ scan order) | Replace profile-guided order with sequential ℓ = 2, 3, …, 2d |
| §15 (ℓ=2 shortcut) | Remove the `max_a²` early-out in `_test_values_jit` |
| §16 (d=4 pair-sum) | Remove the pair-sum continue checks in `_find_min_eff_d4` |
| §21 (staging buffer) | Write survivors directly to `out_buf` instead of staging |

Report children/sec for each variant. The ratio vs the full-optimisation
baseline gives the speedup factor for each change.

**Important:** §8 (quick-check) also affects which threshold lookups are
performed but does not change pruning outcomes — it is a pure
performance optimisation. Verify this by confirming identical survivor
sets with and without quick-check.

### Category C: Search Space Reduction (changes that reduce the number of children enumerated)

**Changes:** §14, §22

**How to measure:**

For §14 (canonical symmetry): the reduction is exactly 2× by
construction — each composition and its reversal are equivalent, and
only one is kept. Verify by counting total children enumerated with and
without the canonical filter.

For §22 (right-to-left ordering): this does not change the set of
children enumerated. It changes the effectiveness of subtree pruning
(§12/§13) by controlling which bins are in the fixed prefix. Measure by
running subtree pruning with left-to-right vs right-to-left ordering
and comparing `n_subtree_pruned`.

### Category D: Scalability (changes that affect large-scale runs but not per-child throughput)

**Changes:** §17, §18, §19, §20

**How to measure:**

For §17 (int32/int64 dispatch): time the `_prune_dynamic_int32` vs
`_prune_dynamic_int64` functions on the same batch. The int32 path
should be faster due to halved memory footprint in the conv array.

For §18 (interleaved work distribution): measure the standard deviation
of per-thread completion times with interleaved vs sequential c₀
ordering. Lower variance = better balance.

For §19 (sort-based dedup): time `_fast_dedup` vs the naive
`set(tuple(row))` approach on a large survivor array (e.g. 1M rows).

For §20 (pairwise merge-dedup): measure peak RSS memory during a merge
of two large shards using `_sorted_merge_dedup_kernel` vs
`np.vstack + _fast_dedup`.

### Running the Measurements

A benchmark script should:

1. Load a fixed checkpoint (e.g. `data/checkpoint_L2_survivors.npy`).
2. Select a representative subset (e.g. 1000 parents).
3. Run each variant, recording survivors, wall time, and children/sec.
4. Produce a table like:

```
| Change disabled  | Survivors | Children/sec | Time (s) | vs baseline |
|------------------|-----------|--------------|----------|-------------|
| None (baseline)  | 147,279   | 1,200,000    | 12.3     | 1.00×       |
| No quick-check   | 147,279   | ???          | ???      | ???×        |
| No sparse nz     | 147,279   | ???          | ???      | ???×        |
| ...              |           |              |          |             |
```

Until these measurements are run, the impact of each change is
unquantified. The descriptions below state *what* each change does,
not *how much* it helps.

### Avoiding Double-Counting: Interaction Map

Many changes are not independent — they interact, and disabling one
changes the apparent impact of another. Naively summing individual
speedup factors will overcount.

**Known interactions:**

```
§6 (incremental conv) ──enables──▶ §8 (quick-check)
    Gray code produces minimal diffs between children,
    which is why the same window re-kills. Without §6,
    children would not be visited in a locality-preserving
    order, and §8 would have a much lower hit rate.

§6 (incremental conv) ──overlaps──▶ §7 (sparse nz list)
    Both reduce cross-term cost. Disabling §7 makes §6's
    cross-term loop slower; disabling §6 makes §7 irrelevant
    (full recompute doesn't use nz_list).

§8 (quick-check) ──masks──▶ §11 (ℓ scan order)
    Quick-check kills ~85% of children before the full window
    scan runs. The ℓ scan order only matters for the ~15% that
    reach the full scan. Disabling §8 would make §11 appear
    much more impactful.

§12 (subtree pruning) ──requires──▶ §22 (R-to-L ordering)
    Subtree pruning checks the fixed left prefix. R-to-L
    ordering controls which bins are in that prefix. With
    L-to-R ordering, the fixed prefix would be the rightmost
    bins (typically sparser), making §12 less effective.

§13 (min-contribution) ──requires──▶ §12 (subtree pruning)
    Min-contribution bounds only apply during subtree pruning.
    If §12 is disabled, §13 has zero effect.

§1 (threshold scaling) ──subsumes──▶ §2 (W-refinement)
    Both affect the threshold formula. §1 is the correct
    scaling; §2 is the per-window tightening on top of §1.
    Measuring §2 in isolation requires §1 to be correct.
    The meaningful comparison is: flat threshold (§1 only)
    vs W-refined threshold (§1 + §2).
```

**Measurement protocol to avoid double-counting:**

1. **Measure dependency chains as units, not individual changes.**
   The chain {§6 → §8 → §11} should be measured as:
   - Baseline: no Gray code, no quick-check, sequential ℓ order
   - +§6 only: Gray code + incremental conv, no quick-check
   - +§6 +§8: add quick-check
   - +§6 +§8 +§11: add profile-guided ℓ order
   Each step's marginal improvement is measured on top of the
   previous, giving non-overlapping contributions.

2. **Measure §12/§13/§22 as a single subtree-pruning unit.**
   - Baseline: no subtree pruning (J_MIN > n_active)
   - +§22 +§12: subtree pruning with R-to-L ordering
   - +§22 +§12 +§13: add min-contribution bounds
   Report the total subtree-pruning impact, then the marginal
   gain from §13 within it.

3. **Measure §1/§2 as a threshold unit.**
   - Flat threshold (`use_flat_threshold=True`): §1 only
   - W-refined threshold (default): §1 + §2
   Report the marginal gain from §2 on top of §1.

4. **For pruning mechanisms, measure who actually gets the kill.**
   Multiple pruning checks can kill the same child. Only the first
   one to fire matters — the rest would have killed it too, but
   never ran. This means disabling one pruning mechanism does NOT
   necessarily increase the survivor count, because another
   mechanism catches the same child later.

   The correct measurement is **attribution**: for each pruned
   child, record *which* mechanism killed it. This requires
   instrumentation counters in the inner loop:

   ```
   n_killed_by_asymmetry     — parent skipped by §4
   n_killed_by_qc            — child killed by quick-check (§8)
   n_killed_by_window_scan   — child killed by full window scan
   n_killed_by_subtree       — children skipped by subtree pruning (§12/§13)
   n_survived                — not killed by any mechanism
   ```

   These are mutually exclusive (each child is counted exactly
   once) and sum to the total Cartesian product size. This gives
   the true attribution without double-counting.

   **But attribution ≠ necessity.** A child killed by quick-check
   would also have been killed by the full window scan (just
   slower). Quick-check's value is *speed*, not additional pruning
   power. To measure whether a mechanism provides *additional*
   pruning (kills children that nothing else would catch), disable
   it and check if the survivor count increases:

   - **§4 (asymmetry):** Provides additional pruning — these
     parents are skipped entirely, no other mechanism tests them.
     Disabling §4 will increase the number of children tested
     (though most will be caught by the window scan). Survivor
     count may or may not change.
   - **§8 (quick-check):** Does NOT provide additional pruning.
     Every child it kills would also be killed by the full window
     scan. Its value is purely speed. Survivor count is identical
     with or without §8.
   - **§11 (ℓ scan order):** Does NOT provide additional pruning.
     All ℓ values are eventually checked. Its value is reaching
     the killing ℓ sooner (speed). Survivor count is identical.
   - **§12/§13 (subtree pruning):** Does NOT provide additional
     pruning. Every child in a pruned subtree would have been
     individually killed by the window scan. Its value is
     skipping the enumeration entirely (speed).
   - **§15 (ℓ=2 shortcut):** Does NOT provide additional pruning.
     The full autoconvolution + window scan would catch the same
     configs. Speed only.
   - **§1/§2 (threshold corrections):** These DO affect pruning
     power. The W-refined threshold (§2) is strictly tighter than
     the flat threshold. Disabling §2 (switching to flat) will
     increase the survivor count.

   Summary of pruning overlaps:

   ```
   Mechanism          Provides additional   Value is
                      pruning power?        primarily
   ─────────────────  ────────────────────  ──────────
   §1 threshold fix   Yes                   Correctness
   §2 W-refinement    Yes                   Fewer survivors
   §4 asymmetry       Possibly              Speed (skip parents)
   §8 quick-check     No                    Speed
   §11 ℓ order        No                    Speed
   §12 subtree prune  No                    Speed
   §13 min-contrib    No                    Speed (tighter §12)
   §15 ℓ=2 shortcut   No                    Speed
   §16 pair-sum       No                    Speed
   ```

   For the "speed only" mechanisms, the right metric is
   children/sec, not survivor count.

5. **For independent changes, single-disable is valid.**
   Changes that do not interact (e.g. §4 asymmetry, §14 canonical,
   §17 int32/int64, §19 sort-dedup) can be measured by disabling
   one at a time. Their contributions do not overlap.

5. **Final total: build up from the naive baseline.**
   Start from the simplest correct implementation (generate all
   children, full O(d²) conv, flat threshold, sequential ℓ order,
   no quick-check, no subtree pruning, no symmetry reduction).
   Add changes cumulatively in dependency order. The total speedup
   is the ratio of the final vs the naive baseline — this number
   is correct by construction and does not double-count.

---

## 1. Corrected Threshold Scaling

**Original paper:** C&S Lemma 3 bounds `‖g*g - f*f‖_∞ ≤ 2/m + 1/m²`.
The paper does not specify how to convert this pointwise bound into an
integer-arithmetic threshold for the windowed test value.

**Our implementation:** The entire threshold expression — including the
correction — is scaled by `ℓ × 4n`:

```
threshold = floor((c_target·m² + correction + ε) × 4n·ℓ)
```

An earlier version of this code incorrectly left the correction term
unscaled by ℓ, which made narrow windows substantially weaker for
pruning.

**Reference:** `run_cascade.py:141–143`

---

## 2. Dynamic Per-Window W-Refinement

**Original paper:** C&S Lemma 3 gives a single global correction
`2/m + 1/m²` applied uniformly to all windows.

**Our implementation:** Each window `(ℓ, s_lo)` computes its own `W_int`
— the actual discrete mass in the child bins overlapping that window's
support — and uses the tighter per-window correction:

```
correction = (3 + W_int/(2n)) / m²
```

derived from C&S equation (1) with `W_f ≤ W_g + 1/m`. Narrow windows
near the edges of `[-1/4, 1/4]` have small `W_int`, yielding tighter
thresholds than the flat bound.

The per-window `W_int` is computed via a prefix sum of child masses,
enabling O(1) range queries.

**Reference:** `run_cascade.py:134–143`

---

## 3. Flat vs W-Refined Dual Mode

The `use_flat_threshold` parameter lets the prover run in two modes:

- **W-refined** (default): uses the per-window correction from §2.
  Tighter, prunes more.
- **Flat**: uses `(2m + 1)/m²` (equivalent to C&S Lemma 3). Required
  for formal verification — the Lean proof's `cascade_all_pruned` axiom
  uses this window-independent bound.

Both modes use the same integer-arithmetic pipeline; the flat mode
simply pre-computes a single threshold per ℓ instead of a 2D table
indexed by `(ℓ, W_int)`.

**Reference:** `run_cascade.py:52–92`

---

## 4. Asymmetry Early-Termination

Before any autoconvolution, check if the parent's mass distribution is
lopsided enough to guarantee pruning of all its children:

```
If left_mass_fraction ≥ √(c_target/2):
    ‖f*f‖_∞ ≥ 2·(dom_fraction)² ≥ c_target → prune all children
```

This is exact for piecewise-constant functions (no discretization margin
needed). All children of such a parent inherit the same left-sum, so
the check applies at the parent level — the entire Cartesian product is
skipped.

**Reference:** `pruning.py:26–32`, `run_cascade.py:1092–1097`

---

## 5. Fused Generate-and-Prune Kernel

**Original approach:** Generate all children of a parent, store them,
then batch-test.

**Our implementation:** Children are generated one at a time via Gray
code enumeration and pruned inline. Survivors are collected on the fly.
This avoids materialising the full Cartesian product (which can be 50M+
rows at L3) and enables all incremental optimisations below.

**Reference:** `run_cascade.py:1066–1080`

---

## 6. Incremental Convolution Update via Gray Code — O(d) per Step

Each Gray code step changes exactly one cursor position (two child
bins). The autoconvolution is updated incrementally:

```
Self-terms:   conv[2k₁] += new₁² − old₁²
              conv[2k₂] += new₂² − old₂²
Mutual term:  conv[k₁+k₂] += 2(new₁·new₂ − old₁·old₂)
Cross-terms:  for j ∉ {k₁,k₂}: conv[k₁+j] += 2·δ₁·child[j]
                                  conv[k₂+j] += 2·δ₂·child[j]
```

This is O(d_child) per step instead of O(d_child²) for a full
recompute.

**Reference:** `run_cascade.py:1357–1406`

---

## 7. Sparse Nonzero List for Cross-Term Iteration

For `d_child ≥ 32`, a dynamically-maintained list of nonzero child bin
indices (`nz_list[]` + reverse index `nz_pos[]`) is used. Cross-term
updates iterate only over nonzero bins instead of all `d_child` bins.

At m=20, d=64, the average child has ~3–4 nonzero bins out of 64.

**Reference:** `run_cascade.py:1134–1138`, `run_cascade.py:1375–1395`

---

## 8. Quick-Check Heuristic (Temporal Locality)

Caches the `(ℓ, s_lo, W_int)` of the window that killed the previous
child. On the next child, re-tries that exact window first — an O(ℓ)
check instead of scanning all ~127 windows. Adjacent Gray code children
differ minimally, so the same window frequently kills the next child.

**Reference:** `run_cascade.py:1268–1277`

---

## 9. Incremental Quick-Check W_int Maintenance

The quick-check's `W_int` is maintained incrementally in O(1). When
bins k₁, k₂ change by δ₁, δ₂ and fall within the quick-check window's
bin range, `qc_W_int` is adjusted by the corresponding delta. This
avoids an O(d) recomputation and ensures the threshold lookup uses the
correct `W_int` for the current child (required for soundness).

**Reference:** `run_cascade.py:1408–1419`

---

## 10. Precomputed 2D Threshold Table

All `(ℓ, W_int)` thresholds are precomputed into a flat int64 array
before the enumeration loop:

```
threshold_table[ell_idx × S_child_plus_1 + W_int] = floor(dyn_x)
```

The hot loop performs only integer comparisons — no runtime float64
arithmetic.

**Reference:** `run_cascade.py:1167–1188`

---

## 11. Profile-Guided Window Scan Order

Instead of scanning ℓ = 2, 3, …, 2d sequentially, uses a three-phase
order based on empirically measured kill rates at d_child=32:

1. **Phase 1:** High-kill ℓ values centered around `d_child/2`
   (ℓ = 9, 10, 11, 8, 7, 12, 13, … — accounting for 92% of prunes)
2. **Phase 2:** Wide windows around `d_child`
3. **Phase 3:** All remaining ℓ values

**Reference:** `run_cascade.py:1190–1231`

---

## 12. Subtree Pruning at J_MIN Boundary

When Gray code digit `J_MIN = 7` advances, the inner digits 0..6 are
about to sweep their full Cartesian product. Before that sweep begins,
we compute the partial autoconvolution of the fixed left prefix
(`child[0..fixed_len−1]`). If the partial conv already exceeds the
threshold for all possible inner values, the entire inner subtree is
skipped.

**Reference:** `run_cascade.py:1421–1452`

---

## 13. Guaranteed Minimum Contributions for Subtree Pruning

Extends §12: instead of checking only whether the partial conv of fixed
bins exceeds threshold, also computes the **guaranteed minimum**
autoconvolution contributions from unfixed bins. For each unfixed
position, the minimum cursor value (`lo_arr`) gives a lower bound on
self-terms, mutual terms, and cross-terms with the fixed prefix and
with each other. These are accumulated into `min_contrib[]`, prefix-
summed, and added to the partial conv in the window scan:

```
prune if: partial_conv_ws + min_contrib_ws > threshold
```

This is strictly tighter than checking `partial_conv_ws > threshold`
alone.

**Reference:** `run_cascade.py:1454–1558`

---

## 14. Canonical Symmetry Reduction

Autoconvolution is symmetric under reversal: `c(b) = c(rev(b))`. Only
the lexicographically smaller of `b` and `rev(b)` is stored. Survivors
are canonicalised inline during collection using an early-exit
lexicographic comparison.

**Impact:** 2× reduction in search space and survivor storage.

**Reference:** `pruning.py:66–79`, `run_cascade.py:1315–1323`

---

## 15. ℓ=2 Max-Element Shortcut

Before the full O(d²) autoconvolution, check if `max(child[i])²`
already exceeds the threshold. For ℓ=2 (single conv entry), the
maximum window sum is `max_a²`. This O(d) pre-check avoids the
expensive autoconvolution for strongly peaked configurations.

**Reference:** `test_values.py:29–37`

---

## 16. d=4 Pair-Sum Bound (Loop Tightening)

For d=4, the ℓ=4 window is dominated by the pair sums `(c₀+c₁)` and
`(c₂+c₃)`. If either pair's squared contribution already exceeds the
threshold, the configuration is skipped before computing the full
autoconvolution.

**Reference:** `solvers.py:84–91`

---

## 17. Int32/Int64 Dynamic Dispatch

Uses int32 autoconvolution when m ≤ 200 (max conv entry = m² = 40,000,
within int32 range). Values are widened to int64 only at the threshold
comparison. Int32 halves the memory footprint of the conv array.

**Reference:** `run_cascade.py:249–270`

---

## 18. Interleaved Work Distribution for Parallel Load Balancing

Small c₀ values produce far more children than large c₀ values. Static
`prange` scheduling would give one thread all the heavy work.
Interleaved assignment (`0, n−1, 1, n−2, 2, …`) ensures each thread's
contiguous chunk contains a balanced mix.

**Reference:** `solvers.py:27–48`

---

## 19. Sort-Based Deduplication

Replaces `set(tuple(row) for row in survivors)` with `np.lexsort` +
Numba linear scan. Avoids creating Python tuple objects.

**Reference:** `run_cascade.py:332–344`

---

## 20. Pairwise Merge-Dedup for Disk Shards

When survivors exceed available RAM, sorted shards are written to disk
and merged pairwise using a two-pointer merge-dedup kernel. Peak memory
is O(output_size) instead of O(3 × total) for vstack + re-sort.

**Reference:** `run_cascade.py:351–529`

---

## 21. L1-Resident Staging Buffer for Survivor Writes

Survivors are staged in a buffer sized to fit entirely in CPU L1 cache
(32 KB on AMD EPYC 9354):

```
d_child ≤ 32: 256 slots × 32 × 4B = 32 KB
d_child > 32: 128 slots × 64 × 4B = 32 KB
```

Flushed to the output array in bulk. An earlier version used a 64 KB
buffer that spilled into L2.

**Reference:** `run_cascade.py:1144–1153`

---

## 22. Right-to-Left Active Position Ordering

Active positions (bins where the cursor has range > 1) are built
right-to-left so that the innermost (fastest-changing) Gray code digits
correspond to the rightmost parent bins. This makes the fixed region
for subtree pruning a **left prefix** of `child[]` — the region most
likely to be concentrated and thus exceed the threshold.

**Reference:** `run_cascade.py:1246–1256`
