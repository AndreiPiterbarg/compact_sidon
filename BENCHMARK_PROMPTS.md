# Benchmark Prompts for Measuring Individual Improvements

Each prompt below is fully self-contained. Copy-paste any single prompt
to an agent and it can run independently without context from the others.

Each prompt instructs the agent to write and run a benchmark script,
then update `CHANGES_IMPROVEMENTS.md` with the measured results.

**Groupings:**

| Group | Prompt | Changes tested |
|-------|--------|----------------|
| 1 | Threshold: flat vs W-refined | §1, §2, §3 |
| 2 | Asymmetry parent filter | §4 |
| 3 | Incremental conv + sparse nz | §5, §6, §7 |
| 4 | Quick-check + ℓ scan order | §8, §9, §11 |
| 5 | Precomputed threshold table | §10 |
| 6 | Subtree pruning chain | §12, §13, §22 |
| 7 | Canonical symmetry | §14 |
| 8 | Analytic early-outs | §15, §16, §26 |
| 9 | Scalability: int32, dedup, etc. | §17, §18, §19, §20, §21 |
| 10 | Arc consistency + Cauchy-Schwarz cap | §23, §24 |
| 11 | Mass-based palindrome grid | §25 |

---

## Prompt 1: Threshold Formula — Flat vs W-Refined (§1, §2, §3)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}. The algorithm discretises mass
distributions into `d = 2n` bins with integer heights `c_i` summing to
`S = 4nm` (the "fine grid"). It then computes the autoconvolution
`conv[k] = sum_{i+j=k} c_i * c_j` and checks whether any window of the
autoconvolution exceeds a pruning threshold.

The pruning threshold has two modes:

**Flat (C&S Lemma 3):** A single global correction `(2m+1)/m²`
independent of the window:
```
threshold = floor((c_target*m² + 2m + 1 + eps) * 4n*ℓ)
```

**W-refined:** A per-window correction using the actual mass `W_int` in
the bins overlapping each window:
```
threshold = floor((c_target*m² + 3 + W_int/(2n) + eps) * 4n*ℓ)
```
The W-refined threshold is tighter (lower), so it prunes MORE children.
This is one of the few changes that actually affects the survivor count.

Both modes are already implemented. The function
`_fused_generate_and_prune_gray` in
`cloninger-steinerberger/cpu/run_cascade.py` accepts a
`use_flat_threshold` parameter (default `False` = W-refined).

### Your task

Write a Python script `tests/bench_threshold.py` that measures the
exact difference in pruning power between flat and W-refined thresholds.
The script must:

1. Add the project root and `cloninger-steinerberger/` to `sys.path`.
2. Import `_fused_generate_and_prune_gray`, `_compute_bin_ranges`,
   `_tighten_ranges` from `cloninger-steinerberger.cpu.run_cascade`.
3. Load `data/checkpoint_L1_survivors.npy` as the parent set
   (these are d=8 compositions, used as parents for L2).
4. Take a deterministic sample of 200 parents (`rng = np.random.default_rng(42); idx = rng.choice(len(parents), 200, replace=False)`).
5. For each parent, compute cursor ranges via `_compute_bin_ranges`
   with `n_half_child = 4` (since d_parent=8, d_child=16,
   n_half_child = d_child//2 = 8... WAIT — actually n_half_child for
   the cascade is d_parent. Read the code: in the cascade, when
   d_parent=2*n_half, the child has d_child=2*d_parent=4*n_half, and
   n_half_child=d_parent=2*n_half. So for L1→L2 with n_half=2:
   d_parent=8, d_child=16, n_half_child=8). Then optionally tighten
   via `_tighten_ranges`.
6. Run `_fused_generate_and_prune_gray` on each parent twice:
   - Once with `use_flat_threshold=True`
   - Once with `use_flat_threshold=False`
7. Collect and report:
   - Total survivors (flat mode)
   - Total survivors (W-refined mode)
   - Difference: how many additional children are pruned by W-refinement
   - Percentage reduction in survivors: `(flat - wref) / flat * 100`
   - Wall time for each mode
8. Assert that both modes test the exact same children (same parents,
   same cursor ranges, same enumeration order). The ONLY difference is
   the threshold values.

Parameters: `n_half=2, m=20, c_target=1.40`.

The script should complete in under 5 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §1, §2, and §3:
- Add a `**Measured impact:**` line to §2 with the exact survivor count
  difference and percentage reduction.
- If the difference is zero or negligible, state that.
- Include the sample size and parameters used.

Do NOT fabricate numbers. Only write what the script actually measures.

---

## Prompt 2: Asymmetry Parent-Level Filter (§4)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}. The algorithm uses a cascade: survivors
at dimension d become parents at dimension 2d. Each parent's bins are
split, producing a Cartesian product of children that are tested and
pruned.

We added an asymmetry filter: before enumerating any children, check if
the parent's left-mass fraction exceeds `√(c_target/2)`. If so, ALL
children are guaranteed to exceed the threshold, so the entire Cartesian
product is skipped. The check is in `_fused_generate_and_prune_gray`
(lines 1086–1097 of `cloninger-steinerberger/cpu/run_cascade.py`) and
in `pruning.py` (`asymmetry_threshold` function).

The child mass in the left half equals the parent mass in its left half
(because `child[2k] + child[2k+1] = 2*parent[k]`), so the asymmetry
check at the parent level covers all children.

### Your task

Write a Python script `tests/bench_asymmetry.py` that measures:

1. Add the project root and `cloninger-steinerberger/` to `sys.path`.
2. Import `asymmetry_threshold` from `pruning`, and
   `_compute_bin_ranges` from `cpu.run_cascade`.
3. Load checkpoint files: `data/checkpoint_L0_survivors.npy` (d=4),
   `data/checkpoint_L1_survivors.npy` (d=8),
   `data/checkpoint_L2_survivors.npy` (d=16).
4. For each level's survivors (used as parents for the next level):
   a. Compute `threshold_asym = asymmetry_threshold(c_target)`.
   b. For each parent, compute `left_frac = sum(parent[0:d//2]) / sum(parent)`.
   c. Count parents where `left_frac >= threshold_asym` or
      `left_frac <= 1 - threshold_asym` (these are "asymmetry-skipped").
   d. For the NON-skipped parents, compute cursor ranges via
      `_compute_bin_ranges` and sum up
      `total_children = product(hi[i] - lo[i] + 1)` per parent.
   e. For the skipped parents, similarly compute the total children that
      WOULD have been enumerated.
5. Report per level:
   - Total parents
   - Parents skipped by asymmetry
   - Percentage skipped
   - Total children avoided (from skipped parents)
   - Percentage of total Cartesian product avoided
6. Correctness verification: for 20 randomly-chosen "skipped" parents
   from L1, actually run `_fused_generate_and_prune_gray` with the
   asymmetry check removed (i.e., manually pass them through — the
   easiest way is to note that the function returns `(0, 0)` when
   asymmetry triggers, so instead compute cursor ranges and count what
   the total Cartesian product would be, and verify it's nonzero but
   all children would be pruned). The simplest approach: run
   `_prune_dynamic` on a batch of the skipped parents' compositions
   themselves (at dimension d_parent) and confirm all are pruned. OR:
   just verify the left_frac exceeds the threshold and trust the
   mathematical argument.

Parameters: `n_half=2, m=20, c_target=1.40`.

The script should complete in under 2 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` section §4:
- Add a `**Measured impact:**` line with: percentage of parents skipped
  and total children avoided, at each level tested.

---

## Prompt 3: Fused Kernel + Incremental Conv + Sparse Nonzero (§5, §6, §7)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

The inner loop of the cascade prover generates children of each parent
and prunes them. We made three coupled changes:

- **§5 Fused kernel:** Children are generated and pruned inline (no
  intermediate array).
- **§6 Incremental convolution:** Gray code enumeration changes 2 child
  bins per step. The autoconvolution is updated in O(d) instead of
  recomputed in O(d²).
- **§7 Sparse nonzero list:** For d_child >= 32, cross-term updates
  iterate only over nonzero bins via a maintained `nz_list[]`.

These are in `_fused_generate_and_prune_gray` in
`cloninger-steinerberger/cpu/run_cascade.py`. The incremental update is
at lines 1357–1406, sparse nz at lines 1134–1138 and 1375–1395.

These are speed-only changes — they do NOT affect the survivor set.

**Interaction:** §7 is an optimisation within §6's cross-term loop. §6
requires the fused kernel (§5). Measure as a chain.

### Your task

Write a Python script `tests/bench_incremental_conv.py` that measures
the children-per-second throughput under four configurations:

**Config A (full recompute baseline):** After each Gray code advance +
child bin update, zero `raw_conv` and recompute the full O(d²)
autoconvolution from scratch. Keep everything else the same (same
enumeration order, same threshold, same pruning). This requires making a
modified copy of `_fused_generate_and_prune_gray` where the incremental
update block (lines 1369–1406) is replaced with a full recompute loop.

**Config B (incremental conv, no sparse):** The Gray code incremental
O(d) update, but with `use_sparse = False` forced (iterate all d_child
bins in cross-term loop, not just nonzero ones).

**Config C (incremental conv + sparse nz):** The current code as-is
(incremental + sparse for d_child >= 32).

For each config:
1. Load `data/checkpoint_L2_survivors.npy` (d=16 parents → d_child=32).
2. Take 50 parents (deterministic seed 42).
3. For each parent, compute cursor ranges via `_compute_bin_ranges`
   (with `d_child=32, n_half_child=16`), then `_tighten_ranges`.
4. Run the kernel variant. Record wall time and count children tested.
5. Assert that ALL configs produce the IDENTICAL survivor set. These are
   speed-only. Any difference means a bug.

Report:
- Children/sec for each config
- Speedup of B over A (isolates incremental conv)
- Speedup of C over B (isolates sparse nz list)
- Speedup of C over A (combined)

Parameters: `n_half=2, m=20, c_target=1.40`.

The script should complete in under 10 minutes. If Config A is too slow
for 50 parents, reduce to 20 parents but keep the same set across all
configs.

**Implementation note for Config A:** The simplest approach is to copy
`_fused_generate_and_prune_gray` into the test script, decorate with
`@njit`, and replace the incremental update section (everything from
`# === INCREMENTAL UPDATE` through the cross-term loops) with:
```python
for k in range(conv_len):
    raw_conv[k] = np.int32(0)
for i in range(d_child):
    ci = np.int32(child[i])
    if ci != 0:
        raw_conv[2 * i] += ci * ci
        for j in range(i + 1, d_child):
            cj = np.int32(child[j])
            if cj != 0:
                raw_conv[i + j] += np.int32(2) * ci * cj
```
Keep the Gray code advance, quick-check, window scan, etc. all the same.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §5, §6, and §7:
- Add `**Measured impact:**` lines with the children/sec numbers and
  speedup ratios.
- Note the d_child, sample size, and parameters used.
- Note that these are speed-only (identical survivor sets).

---

## Prompt 4: Quick-Check + Incremental W_int + ℓ Scan Order (§8, §9, §11)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

The inner loop tests each child against all `(ℓ, s_lo)` windows. We
added three related speed optimisations:

- **§8 Quick-check:** Cache the `(ℓ, s_lo, W_int)` of the window that
  killed the previous child. Re-try it first — O(ℓ) instead of scanning
  all ~127 windows. In `_fused_generate_and_prune_gray`, lines
  1268–1277.
- **§9 Incremental QC W_int:** Maintain `qc_W_int` in O(1) by delta
  adjustment (lines 1408–1419) instead of recomputing from scratch.
- **§11 Profile-guided ℓ order:** When the full scan IS needed, test ℓ
  values in empirically-tuned order (lines 1190–1231) instead of
  sequentially.

These are speed-only — identical survivor sets.

**Interaction:** §8 kills ~85% of children before the full scan, so §11
only matters for the remaining ~15%. Disabling §8 would make §11 appear
much more impactful. They must be measured as a chain.

### Your task

Write a Python script `tests/bench_quickcheck.py` that measures four
configurations by creating modified copies of
`_fused_generate_and_prune_gray`:

**Config A (no QC, sequential ℓ):** Set `qc_ell = 0` permanently (never
use cached window). Replace the profile-guided ℓ order with sequential
`ℓ = 2, 3, ..., 2*d_child`. Every child gets the full window scan with
naive ordering.

**Config B (no QC, profile-guided ℓ):** `qc_ell = 0` permanently, but
use the profile-guided ℓ order from §11. Isolates §11 without §8.

**Config C (QC enabled, sequential ℓ):** Enable quick-check (§8 + §9),
but sequential ℓ order. Isolates §8 without §11.

**Config D (full — QC + profile ℓ):** Current code as-is.

For each config:
1. Load `data/checkpoint_L2_survivors.npy` (d=16 parents → d_child=32).
2. Take 200 parents (seed 42).
3. Compute cursor ranges + tighten.
4. Run the kernel variant. Record wall time, children tested.
5. Also instrument: count how many children were killed by quick-check
   vs killed by full window scan (add two counters in the modified
   kernel).
6. Assert identical survivor sets across all four configs.

Report:
- Children/sec for each config
- QC hit rate for C and D
- Speedup of B over A: marginal gain of §11 alone
- Speedup of C over A: marginal gain of §8 alone
- Speedup of D over A: combined
- Speedup of D over C: marginal gain of §11 on top of §8

Parameters: `n_half=2, m=20, c_target=1.40`.

Complete in under 10 minutes. Reduce to 100 parents if needed.

**Implementation notes:**
- To disable QC: in the modified kernel, skip the `if qc_ell > 0:`
  block and always go to the full window scan. Also skip the
  `qc_ell = np.int32(ell)` assignment that updates the cache.
- To use sequential ℓ: replace the ell_order construction with a simple
  `for oi in range(ell_count): ell_order[oi] = oi + 2`.
- §9 is coupled to §8 — it only runs when QC is active. No separate
  measurement needed.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §8, §9, and §11:
- Add `**Measured impact:**` with children/sec, QC hit rate, and
  speedup ratios.
- Report all four comparisons (B/A, C/A, D/A, D/C) to show interaction.
- Note that these are speed-only (identical survivor sets).

---

## Prompt 5: Precomputed Threshold Table vs Inline Float64 (§10)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

In the inner loop, each `(ℓ, s_lo)` window check compares
`ws > threshold`. We precompute all thresholds into a flat int64 array
(§10), so the hot path does a single table lookup instead of computing:
```python
corr_w = 3.0 + float(W_int) / (2.0 * n_half)
dyn_x = (c_target * m * m + corr_w + eps) * float(ell) * 4.0 * n_half
dyn_it = int(dyn_x)
```

The table is built in `_fused_generate_and_prune_gray` at lines
1167–1188 of `cloninger-steinerberger/cpu/run_cascade.py`. The lookup
is at line 1306: `threshold_table[ell_idx * S_child_plus_1 + W_int]`.

### Your task

Write a Python script `tests/bench_threshold_table.py` that measures
two configurations by creating modified copies of
`_fused_generate_and_prune_gray`:

**Config A (inline float64):** Remove the threshold table. In the window
scan inner loop (lines 1296–1312), replace the table lookup with inline
float64 computation:
```python
corr_w = 3.0 + np.float64(W_int) / (2.0 * n_half_d)
dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
dyn_it = np.int64(dyn_x)
if ws > dyn_it:
```
Also do the same in the quick-check block (line 1275).

**Config B (precomputed table):** Current code as-is.

For each config:
1. Load `data/checkpoint_L2_survivors.npy`.
2. Take 200 parents (seed 42).
3. Compute cursor ranges + tighten.
4. Run kernel. Record wall time, children tested, survivors.
5. Assert IDENTICAL survivor sets. If they differ, report which children
   differ and investigate — it would indicate a numerical discrepancy
   between inline and precomputed thresholds.

Report: children/sec for each config, speedup ratio.

Parameters: `n_half=2, m=20, c_target=1.40`.

Complete in under 5 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` section §10:
- Add `**Measured impact:**` with children/sec and speedup.
- Note whether survivor sets are identical (expected: yes).

---

## Prompt 6: Subtree Pruning Chain (§12, §13, §22)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

We added subtree pruning: when a slow outer Gray code digit advances,
check if the partial autoconvolution of the fixed left prefix already
exceeds the threshold. If so, skip the entire inner subtree. Three
changes form a dependency chain:

- **§22 R-to-L ordering:** Inner digits = rightmost bins, so the fixed
  region is the left prefix (concentrated, more likely to exceed
  threshold). Lines 1246–1256.
- **§12 Subtree pruning:** Partial autoconv check at `J_MIN=7`. Lines
  1421–1452.
- **§13 Min-contribution bounds:** Add guaranteed minimum contributions
  from unfixed bins to tighten the subtree check. Lines 1454–1558.

These are speed-only: every child in a pruned subtree would also have
been individually killed by the window scan. Survivor sets must be
identical.

**Interaction:** §13 only fires during §12. §12 depends on §22 for
which bins are in the fixed prefix.

### Your task

Write a Python script `tests/bench_subtree.py` that measures three
configurations by creating modified copies of
`_fused_generate_and_prune_gray`:

**Config A (no subtree pruning):** Set `J_MIN = 999` (larger than any
possible `n_active`). The `if j == J_MIN` check never fires. Every
child is individually enumerated and tested.

**Config B (subtree pruning, no min-contrib):** Set `J_MIN = 7`. Enable
the subtree pruning check, but in the window scan within the subtree
check, use ONLY the partial conv of fixed bins — skip the min_contrib
computation entirely. Replace:
```python
ws += mc_sum  # min_contrib
```
with nothing (just use `ws` from partial conv alone). Also skip the
W_int_unfixed computation and use W_int_fixed only.

**Config C (full):** Current code: §12 + §13 + §22.

For each config:
1. Load `data/checkpoint_L2_survivors.npy`.
2. Take 100 parents (seed 42).
3. Compute cursor ranges + tighten.
4. Run kernel. Record wall time, children tested, children skipped by
   subtree pruning (`n_subtree_pruned` — the second return value),
   survivors.
5. Assert IDENTICAL survivor sets across all three configs.

Report:
- Children tested per config (WILL differ — A tests more)
- `n_subtree_pruned` for B and C
- Additional children skipped by C over B (marginal gain of §13)
- Wall time and children/sec for each
- Average subtree size: `n_subtree_pruned / n_subtree_events` (how
  many children skipped per event)

Parameters: `n_half=2, m=20, c_target=1.40`.

Complete in under 10 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §12, §13, and §22:
- Add `**Measured impact:**` with children skipped, percentage of
  Cartesian product avoided, and wall time speedup.
- Report marginal gain of §13 on top of §12.
- Note identical survivor sets.

---

## Prompt 7: Canonical Symmetry Reduction (§14)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

We added canonical symmetry reduction (§14): since
`conv(b) = conv(rev(b))`, only the lexicographically smaller of `b` and
`rev(b)` is stored. This should reduce storage by exactly 2× minus
palindromes (compositions equal to their own reverse).

Canonicalization is in `_canonicalize_inplace` (run_cascade.py:277–301)
and inline in `_fused_generate_and_prune_gray` (lines 1315–1323). The
canonical mask is in `pruning.py:_canonical_mask`.

### Your task

Write a Python script `tests/bench_canonical.py` that:

1. Load checkpoint files: `data/checkpoint_L0_survivors.npy`,
   `data/checkpoint_L1_survivors.npy`,
   `data/checkpoint_L2_survivors.npy`.
2. For each checkpoint:
   a. Count total survivors (this is the canonical count).
   b. For each survivor, check if `row == rev(row)` (palindrome).
   c. Count palindromes and non-palindromes.
   d. Compute the theoretical non-canonical count:
      `non_canonical = 2 * non_palindromes + palindromes`
   e. Compute the reduction factor:
      `reduction = non_canonical / canonical_count`
3. Verify correctness: for 500 random survivors from L1 (seed 42):
   a. Confirm `row <= rev(row)` lexicographically (it IS canonical).
   b. Confirm `rev(row)` is NOT in the survivor set (unless palindrome).
      Use a set of tuples for O(1) lookup.
4. Report per level:
   - Total survivors (canonical)
   - Palindromes
   - Non-palindromes
   - Theoretical non-canonical count
   - Reduction factor (should be very close to 2.0)

Parameters: use the existing checkpoint files as-is.

Complete in under 1 minute.

### After running

Update `CHANGES_IMPROVEMENTS.md` section §14:
- Add `**Measured impact:**` with the exact reduction factor at each
  level and the palindrome count.
- Confirm the factor is ~2× as expected.

---

## Prompt 8: Analytic Early-Out Checks (§15, §16, §26)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

We added two early-out checks that avoid computing the full O(d²)
autoconvolution:

- **§15 ℓ=2 max-element shortcut:** In `_test_values_jit`
  (test_values.py:29–37), check `max(child[i])²` against the ℓ=2
  threshold before the full autoconvolution.
- **§16 d=4 pair-sum bound:** In `_find_min_eff_d4`
  (solvers.py:84–91), check `(c₀+c₁)²` and `(c₂+c₃)²` against the
  ℓ=4 threshold before computing the full autoconvolution.
- **§26 Unrolled autoconvolution for d=4 and d=6:** In `_test_values_jit`
  (test_values.py:41–70), the autoconvolution double loop is replaced
  with fully unrolled straight-line arithmetic for d=4 (7 conv entries)
  and d=6 (11 conv entries).

All are speed-only: every configuration they catch would also be caught
by the full autoconvolution + window scan. §26 does not change pruning
at all — it computes the same values faster.

### Your task

Write a Python script `tests/bench_earlyout.py` with two benchmarks:

**Benchmark A (§15 — ℓ=2 shortcut):**
1. Load `data/checkpoint_L0_survivors.npy` (d=4) and
   `data/checkpoint_L1_survivors.npy` (d=8).
2. For each checkpoint, call `_test_values_jit` (from
   `cloninger-steinerberger/test_values.py`) with
   `early_stop = c_target + correction(m)` (shortcut enabled).
3. Call again with `early_stop = 0.0` (shortcut disabled — always
   computes full autoconvolution).
4. Measure wall time for each. Verify identical output arrays
   (the returned test values should be >= the early_stop value for
   pruned configs, so they may differ in magnitude but the pruning
   decisions should be the same).
5. To count how many configs are caught by the shortcut: instrument by
   running once with early_stop and counting configs where the result
   equals `max_a² * inv_ell2` exactly (the shortcut value). Simplest:
   compare the results from early_stop=0 (true values) vs early_stop>0
   (lower bounds). Configs where the early_stop version has the value
   `max_a² / (4*n*2)` were caught by the shortcut.

**Benchmark B (§16 — d=4 pair-sum bound):**
1. This is harder to isolate because `_find_min_eff_d4` is a fused
   Numba kernel. Instead, measure it indirectly:
2. Run `_find_min_eff_d4` (from `solvers.py`) with its current code.
   Time it.
3. Create a modified copy `_find_min_eff_d4_no_pairsum` that removes
   the two `continue` checks at lines 88–91 (the `pair_left` and
   `pair_right` checks). Time it.
4. Verify identical output (same minimum effective value found).
5. Report wall time difference.

**Benchmark C (§26 — unrolled autoconvolution):**
1. Create a batch of 50,000 random d=4 compositions summing to S=160
   (seed 42).
2. Run `_test_values_jit` with d=4 (uses the unrolled path, lines
   41–57). Time it.
3. Create a modified copy of `_test_values_jit` where the `if d == 4:`
   branch is removed, forcing the generic double-loop path for d=4.
   Time it.
4. Verify identical output arrays.
5. Repeat for d=6 with S=240 (n_half=3).
6. Report speedup ratio for each.

Parameters: `n_half=2, m=20, c_target=1.40`. The `S` for d=4 is
`4 * n_half * m = 160`.

Complete in under 3 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §15, §16, and §26:
- Add `**Measured impact:**` with wall time speedup and (for §15)
  percentage of configs caught by the shortcut.
- For §26, report speedup ratio for d=4 and d=6.
- Note that these are speed-only (identical pruning outcomes).

---

## Prompt 9: Scalability — Int32, Dedup, Work Distribution, Staging (§17–§21)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}.

We made five scalability changes:

- **§17 Int32/Int64 dispatch:** `_prune_dynamic_int32` vs
  `_prune_dynamic_int64` in run_cascade.py:50–270. Uses int32 conv when
  m ≤ 200.
- **§18 Interleaved work distribution:** `_build_interleaved_order` in
  solvers.py:27–48. Pattern `[0, n-1, 1, n-2, ...]` for balanced prange.
- **§19 Sort-based dedup:** `_fast_dedup` in run_cascade.py:332–344.
  Lexsort + Numba scan instead of `set(tuple(row))`.
- **§20 Pairwise merge-dedup:** `_sorted_merge_dedup_kernel` in
  run_cascade.py:351–412. Two-pointer merge of sorted shards.
- **§21 L1-resident staging buffer:** `_STAGE_CAP` in
  run_cascade.py:1144–1153. 32KB buffer fits L1 cache.

These are mostly independent of each other and of the pruning changes.

### Your task

Write a Python script `tests/bench_scalability.py` with five separate
benchmarks:

**Benchmark A (§17 — int32 vs int64):**
1. Load `data/checkpoint_L1_survivors.npy` (d=8). Take 10,000 rows.
2. Run `_prune_dynamic_int32(batch, n_half=2, m=20, c_target=1.40)`.
   Time it (3 runs, take median).
3. Run `_prune_dynamic_int64(batch, n_half=2, m=20, c_target=1.40)`.
   Time it (3 runs, take median).
4. Assert identical boolean masks.
5. Report speedup ratio.

**Benchmark B (§18 — interleaved vs sequential order):**
1. Import `_find_min_eff_d4` and `_build_interleaved_order` from
   `solvers`.
2. Compute `S = 4 * 2 * 20 = 160`. Build interleaved order for
   `n = S // 2 + 1 = 81`.
3. Run `_find_min_eff_d4` with the interleaved order (current). Time it.
4. Run with `np.arange(81, dtype=np.int32)` (sequential). Time it.
5. Verify identical output (same min_eff value).
6. Report wall time difference.

**Benchmark C (§19 — sort-dedup vs set-dedup):**
1. Generate a synthetic array: 500,000 rows, d=32, values drawn from
   `rng.integers(0, 21, size=(500000, 32))` with seed 42. Inject ~10%
   duplicates by copying random rows.
2. Time `_fast_dedup(arr)`. (3 runs, median.)
3. Time the naive approach:
   ```python
   seen = set()
   unique = []
   for i in range(len(arr)):
       t = tuple(arr[i])
       if t not in seen:
           seen.add(t)
           unique.append(i)
   result = arr[np.array(unique)]
   ```
   (1 run — this will be slow.)
4. Verify identical output (same rows, possibly different order — sort
   both before comparing).
5. Report speedup ratio and note memory usage if measurable
   (use `tracemalloc`).

**Benchmark D (§20 — sorted merge vs vstack+dedup):**
1. Generate two sorted, deduplicated arrays: take the benchmark C output
   and split it into two halves. Sort each.
2. Time `_sorted_merge_dedup_kernel(a, b, out)` with pre-allocated
   output.
3. Time `_fast_dedup(np.vstack([a, b]))`.
4. Verify identical output.
5. Report speedup ratio.

**Benchmark E (§21 — staging buffer):**
1. This is tightly integrated into `_fused_generate_and_prune_gray` and
   hard to isolate cleanly. Instead, report the theoretical analysis:
   at d_child=32, `_STAGE_CAP=256` means the 32KB staging buffer fits
   L1 (32KB). With `_STAGE_CAP=1`, every survivor write goes directly
   to `out_buf`, which is in L2/L3. The impact depends on the survivor
   rate.
2. If you want to measure empirically: create two copies of the kernel
   with `_STAGE_CAP=256` vs `_STAGE_CAP=1`. Run on 50 parents from L2
   checkpoint. Compare wall time. This is optional — if the code
   modification is too complex, just report "not measured" and explain
   why.

Parameters: `n_half=2, m=20, c_target=1.40`.

Complete in under 5 minutes total (benchmark C's naive approach may
dominate — reduce to 100K rows if needed).

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §17, §18, §19, §20, and §21:
- Add `**Measured impact:**` to each with the speedup ratio or other
  relevant metric.
- For §21, report either the measured result or "not measured —
  requires kernel modification" with the theoretical justification.

---

## Prompt 10: Arc Consistency + Cauchy-Schwarz Cursor Cap (§23, §24)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}. The algorithm uses a cascade: survivors
at dimension d become parents at dimension 2d. Each parent's bins are
split into child bins with cursor ranges `[lo, hi]`. The Cartesian
product of all cursor ranges gives the set of children to enumerate.

We added two changes that reduce cursor ranges BEFORE enumeration:

1. **Cauchy-Schwarz cursor cap (§24):** The initial per-bin cap `x_cap`
   is tightened using `min(x_cap_energy, x_cap_cs)` where:
   ```
   x_cap_energy = floor(m * sqrt(4 * d_child * (c_target + correction)))
   x_cap_cs     = floor(m * sqrt(4 * d_child * c_target)) + 1
   ```
   The CS bound is independent of the correction term. In
   `_compute_bin_ranges` at `run_cascade.py:1726–1728`.

2. **Arc consistency (§23):** After computing initial cursor ranges,
   `_tighten_ranges` iteratively removes provably-infeasible edge
   values. For each position `p` and each edge value `v`, it computes a
   lower bound on the window sum when all OTHER positions take their
   minimum-contribution values. If this lower bound exceeds the
   threshold for ANY window, `v` is removed. Iterated until convergence.
   In `run_cascade.py:1748–1998`.

   Arc consistency is called from `process_parent_fused`
   (line 2342) on every parent before enumeration.

These changes reduce the Cartesian product size and can eliminate
parents entirely (if a range becomes empty).

### Your task

Write a Python script `tests/bench_arc_consistency.py` that measures:

1. Add the project root and `cloninger-steinerberger/` to `sys.path`.
2. Import `_compute_bin_ranges`, `_tighten_ranges` from
   `cloninger-steinerberger.cpu.run_cascade`.
3. Load `data/checkpoint_L1_survivors.npy` (d=8) and
   `data/checkpoint_L2_survivors.npy` (d=16) as parent sets.
4. For each level, take 500 parents (seed 42).

**Measurement A (§24 — Cauchy-Schwarz cap):**
5. For each parent, compute cursor ranges twice:
   a. With the CS cap: `_compute_bin_ranges` as-is.
   b. Without the CS cap: use only the energy bound (manually compute
      `x_cap = floor(m * sqrt(4 * d_child * (c_target + corr)))` and
      recompute lo/hi arrays using that larger cap).
6. Compare: total_children (product of ranges) with and without CS cap.
   Sum across all parents.

**Measurement B (§23 — arc consistency):**
7. For each parent, after `_compute_bin_ranges`, record
   `total_children_before = product(hi[i] - lo[i] + 1)`.
8. Call `_tighten_ranges`. Record `total_children_after`.
9. Count parents eliminated (total_children_after == 0).
10. Report:
    - Total Cartesian product before tightening (sum across parents)
    - Total Cartesian product after tightening
    - Percentage reduction
    - Parents eliminated
    - Average range reduction per position (sum of (hi-lo+1) deltas)

**Measurement C (combined — §23 affects survivor count?):**
11. For 100 parents from L2, run `_fused_generate_and_prune_gray` twice:
    a. With tightened ranges (current code path via `process_parent_fused`).
    b. With un-tightened ranges (call `_compute_bin_ranges` but skip
       `_tighten_ranges`).
12. Compare survivor sets. They SHOULD be identical — arc consistency
    only removes values that would have been pruned anyway. If they
    differ, it indicates a soundness bug. Assert this.
13. Compare wall time — fewer children to enumerate means faster.

Parameters: `n_half=2, m=20, c_target=1.40`.
For L1 parents: `d_parent=8, d_child=16, n_half_child=8`.
For L2 parents: `d_parent=16, d_child=32, n_half_child=16`.

Complete in under 5 minutes.

### After running

Update `CHANGES_IMPROVEMENTS.md` sections §23 and §24:
- Add `**Measured impact:**` with:
  - §24: percentage reduction in total Cartesian product from CS cap
  - §23: percentage reduction in total Cartesian product from
    tightening, number of parents eliminated, wall time speedup
- Note whether survivor sets are identical (expected: yes for §23).

---

## Prompt 11: Mass-Based Palindrome Grid (§25)

You are an expert in optimization and numerical algorithms. You are
working in the repository at
`c:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon`.

### Background

This codebase implements the Cloninger-Steinerberger branch-and-prune
algorithm (arXiv:1403.7988) for computing lower bounds on the Sidon
autocorrelation constant C_{1a}. The algorithm starts at L0 by
generating all compositions at dimension d=2*n_half summing to
S=4*n_half*m and pruning them.

We added an alternative L0 generator using a mass-based palindrome grid
(§25): instead of generating ALL compositions at d=4 summing to
S=4*n_half*m=160, it generates only palindromic compositions — vectors
where `b[i] = b[d-1-i]`. This exploits the fact that the autoconvolution
is symmetric under reversal, so if `b` survives then `rev(b)` also
survives, and for the cascade we only need to track one representative.
The palindrome grid goes further: it only GENERATES the symmetric ones.

The standard L0 is `run_level0` in `run_cascade.py:2496`. The palindrome
L0 is `run_level0_mass` at line 2199. The palindrome cascade kernel is
`_fused_mass_palindrome` at line 2075. The main cascade calls
`run_level0_mass` at line 3083.

At L0 with n_half=2, m=20: the standard grid generates compositions of
d=4 summing to S=160. The palindrome grid generates compositions of
n_half=2 elements summing to m=20 (half-domain), then mirrors. This is
a fundamentally different parameterisation.

### Your task

Write a Python script `tests/bench_palindrome.py` that measures:

1. Add the project root and `cloninger-steinerberger/` to `sys.path`.
2. Import `run_level0`, `run_level0_mass`, `count_compositions` from
   `cloninger-steinerberger.cpu.run_cascade` and
   `cloninger-steinerberger.pruning`.

**Measurement A — L0 search space:**
3. For the standard grid: compute `count_compositions(d=4, S=160)` to
   get the total L0 compositions. Then run `run_level0` and count
   survivors.
4. For the palindrome grid: compute `count_compositions(n_half=2, S_half=20)`
   to get the total palindromic L0 compositions. Then run
   `run_level0_mass` and count survivors.
5. Report:
   - Total compositions tested (standard vs palindrome)
   - Survivors (standard vs palindrome)
   - Reduction in search space
   - Wall time for each

**Measurement B — L1 branching factor:**
6. For 100 survivors from each L0 (used as parents for L1), compute
   cursor ranges and total_children per parent.
7. For palindrome survivors: note that subsequent levels use
   `process_parent_mass` which has only `d_parent/2` free cursor
   variables. Compute total_children per parent.
8. For standard survivors: use `_compute_bin_ranges` +
   `_tighten_ranges` with full `d_parent` cursor variables.
9. Compare average children per parent at L1.

Parameters: `n_half=2, m=20, c_target=1.40`.

Complete in under 3 minutes.

**Note:** The palindrome grid and the standard grid produce DIFFERENT
survivors (different parameterisations). This is not a speed-only
change — it's a fundamentally different approach. Report both survivor
counts and note whether the palindrome approach produces fewer or more
survivors.

### After running

Update `CHANGES_IMPROVEMENTS.md` section §25:
- Add `**Measured impact:**` with: L0 search space reduction, survivor
  counts for both approaches, and L1 branching factor comparison.
