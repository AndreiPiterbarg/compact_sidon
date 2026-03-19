# Part 6: Cascade Orchestration, Completeness Argument & Deduplication — Verification Report

**Scope:** Verify the cascade end-to-end: every configuration in the search space is either pruned or refined until exhausted. Verify deduplication correctness and checkpoint integrity.

**Result: ALL 9 ITEMS VERIFIED**

**Files audited:**

- `run_cascade.py:1199-1296`: `run_level0`
- `run_cascade.py:1667-2126`: `run_cascade` (main loop)
- `run_cascade.py:1303-1435`: `generate_children_uniform`, `test_children` (legacy fallback)
- `run_cascade.py:274-286`: `_fast_dedup`
- `run_cascade.py:250-271`: `_dedup_sorted`
- `run_cascade.py:293-354`: `_sorted_merge_dedup_kernel`
- `run_cascade.py:357-492`: `_merge_dedup_shards`
- `run_cascade.py:1566-1624`: `_save_checkpoint`, `_load_checkpoint`
- `run_cascade.py:1442-1560`: multiprocessing workers
- `run_cascade.py:219-243`: `_canonicalize_inplace`
- `pruning.py:32-52`: `asymmetry_prune_mask`
- `pruning.py:55-69`: `_canonical_mask`
- `compositions.py`: `generate_compositions_batched`
- `initial_baseline.m`: MATLAB reference

**MATLAB mapping:**

- Lines 49-51: Outer loop over parent bins → Python `run_cascade` level loop
- Lines 67-69: `binStore` iteration → Python processes `current_configs`
- Lines 140-153: Per-bin split options → Python `_compute_bin_ranges`
- Lines 238-251: `leftOver` survivor collection, `binStore` append → Python shard accumulation + merge-dedup
- Lines 259-263: Checkpoint save → Python `_save_checkpoint`

---

## Item 1: L0 Completeness

**Claim:** `run_level0` enumerates every non-negative integer vector of length $d = 2 \cdot n_{\text{half}}$ summing to $S = m$, and either prunes it (soundly) or includes it as a survivor.

### Proof

**Enumeration.** Line 1229 calls `generate_compositions_batched(d, S, batch_size=200_000)`. This generator yields ALL $\binom{S+d-1}{d-1}$ compositions (verified in Part 2, Item 1). Every composition is processed.

**Filter chain** (lines 1233-1253): For each batch, three filters are applied in sequence:

1. **Canonical filter** (`_canonical_mask`, pruning.py:56-69): Keeps only $b \le \text{rev}(b)$ lexicographically.
2. **Asymmetry filter** (`asymmetry_prune_mask`, pruning.py:32-52): Returns `True` for configs NOT covered by the asymmetry argument.
3. **Dynamic threshold** (`_prune_dynamic`, run_cascade.py:199-212): Returns `True` for configs that survive pruning.

**Filter independence.** The filters are independent — each is sound on its own:

- *Canonical:* For every non-canonical $b$ (where $b > \text{rev}(b)$), the reversal $\text{rev}(b)$ is also enumerated and IS canonical. The autoconvolution satisfies $\text{rev}(b * b) = \text{rev}(b) * \text{rev}(b)$, so $\max(b*b) = \max(\text{rev}(b) * \text{rev}(b))$. The asymmetry condition $\text{left\_frac}(\text{rev}(b)) = 1 - \text{left\_frac}(b)$ is symmetric under the test $(x > 1-\theta) \land (x < \theta)$. Therefore testing the canonical representative suffices.

- *Asymmetry:* If $\text{left\_frac} \ge \sqrt{c_{\text{target}}/2}$ or $\text{left\_frac} \le 1 - \sqrt{c_{\text{target}}/2}$, then $\|f*f\|_\infty \ge c_{\text{target}}$ (proved in Part 1, Verification 2). Sound.

- *Dynamic threshold:* If $\text{ws} > \text{dyn\_it}$ for any window, then $\|f*f\|_\infty \ge c_{\text{target}}$ by the window-based lower bound with per-window correction (proved in Parts 1, 3). Sound.

A config must fail ALL three filters to survive. No interaction between filters is possible: canonical delegates to the canonical twin, asymmetry proves the bound directly via mass concentration, and dynamic threshold proves it via the windowed autoconvolution.

### Verdict: VERIFIED $\square$

---

## Item 2: Refinement Level Completeness

**Claim:** At each refinement level, EVERY surviving parent from the previous level is processed, and for each parent the COMPLETE Cartesian product of cursor values is tested.

### Proof

**Parallel path** (lines 1878-1992): The parent array is written to a memory-mapped file (line 1912). Workers are initialized via `_init_worker_shm` to open this file read-only. Processing:

```python
pool.imap_unordered(_process_parent_shm, range(n_parents), chunksize=chunksize)
```

`range(n_parents)` generates indices $\{0, 1, \ldots, n_{\text{parents}} - 1\}$. Python's `multiprocessing.Pool.imap_unordered` guarantees every element of the iterable is processed exactly once (results returned in arbitrary completion order). No index is skipped.

**Sequential path** (lines 1994-2015):

```python
for p_idx in range(n_parents):
```

Explicit iteration over all parents. No index is skipped.

**Per-parent completeness.** Both paths call `process_parent_fused` (line 1550 / line 2004), which calls `_fused_generate_and_prune`. The fused kernel uses an odometer (lines 720-730):

```python
carry = d_parent - 1
while carry >= 0:
    cursor[carry] += 1
    if cursor[carry] <= hi_arr[carry]: break
    cursor[carry] = lo_arr[carry]
    carry -= 1
if carry < 0: break  # all combinations exhausted
```

Starting from `cursor = lo_arr`, this visits every point in $\prod_{i=0}^{d_p-1} [\text{lo}[i], \text{hi}[i]]$. Total iterations $= \prod_i (\text{hi}[i] - \text{lo}[i] + 1)$, verified in Part 4, Item 1.

**Subtree pruning** (lines 841-961): When a deep carry occurs ($n_{\text{changed}} > \lfloor d_p/4 \rfloor$), the kernel checks whether the partial autoconvolution from fixed child bins already exceeds the dynamic threshold. Three claims establish soundness (proved in Part 4, Additional Verification: Subtree Pruning):

1. $\text{ws\_partial} \le \text{ws\_full}$ for any completion (cross-terms $\ge 0$)
2. $W_{\text{int,max}} \ge W_{\text{int,actual}}$ (parent sums bound child sums)
3. $\text{dyn\_it}(W)$ is non-decreasing in $W$

Combined: subtree pruning prunes only configurations that the full scan would also prune. The fast-forwarded cursor positions (line 936-937) are exactly those that would have been individually pruned. Sound.

**Buffer overflow detection.** `process_parent_fused` (lines 1033-1078) allocates a survivor buffer of size `min(total_children, buf_cap)`. The kernel counts all survivors; if the count exceeds the buffer capacity, the wrapper re-allocates at exact size and re-runs the kernel (lines 1067-1076). No survivors are lost.

### Verdict: VERIFIED $\square$

---

## Item 3: Pre-filter Soundness

**Claim:** The pre-filter at lines 1796-1813 correctly identifies parents that produce zero children.

### Proof

The pre-filter computes:

$$x_{\text{cap}} = \min\!\bigl(\lfloor m\sqrt{(\text{thresh})/d_{\text{child}}} \rfloor,\; \lfloor m\sqrt{c_{\text{target}}/d_{\text{child}}} \rfloor,\; m\bigr)$$

$$\text{max\_bin\_val} = 2 \cdot x_{\text{cap}}$$

and removes parents where any bin $b_i > \text{max\_bin\_val}$.

This is sound because `_compute_bin_ranges` (lines 991-1020) uses the identical formula. If $b_i > 2 \cdot x_{\text{cap}}$:

$$\text{lo} = \max(0,\; b_i - x_{\text{cap}}) = b_i - x_{\text{cap}} \quad (\text{since } b_i > x_{\text{cap}})$$
$$\text{hi} = \min(b_i,\; x_{\text{cap}}) = x_{\text{cap}} \quad (\text{since } b_i > x_{\text{cap}})$$
$$\text{lo} = b_i - x_{\text{cap}} > x_{\text{cap}} = \text{hi} \quad (\text{since } b_i > 2 \cdot x_{\text{cap}})$$

Empty range $\Rightarrow$ zero children. Parents with zero children can be safely skipped.

The formulas in the pre-filter and `_compute_bin_ranges` are algebraically identical:
- Pre-filter: `min(x_cap_pf, x_cap_cs_pf, m)` (line 1804)
- `_compute_bin_ranges`: `max(0, min(x_cap, x_cap_cs, m))` (lines 1003-1005)

Since $\lfloor m\sqrt{\cdot}\rfloor \ge 0$, the `max(0, ...)` is redundant and both compute the same value.

### Verdict: VERIFIED $\square$

---

## Item 4: Dimension Doubling

**Claim:** At each refinement level, the child dimension and half-parameter double, matching the paper's refinement scheme.

### Proof

At each level (lines 1789-1790, 2099-2101):

```python
d_child = 2 * d_parent
n_half_child = 2 * n_half_parent
```

At level $L$ (0-indexed from L0): $d = 2^{L+1} \cdot n_{\text{half}}$, $n_{\text{half,current}} = 2^L \cdot n_{\text{half}}$.

The child-parent relationship (fused kernel, lines 590-592):

$$\text{child}[2k] + \text{child}[2k+1] = \text{parent}[k], \quad k = 0, \ldots, d_{\text{parent}} - 1$$

This splits each parent bin $[k/(4n_{\text{parent}}),\; (k+1)/(4n_{\text{parent}}))$ into two child bins of half the width: $[2k/(4n_{\text{child}}),\; (2k+1)/(4n_{\text{child}}))$ and $[(2k+1)/(4n_{\text{child}}),\; (2k+2)/(4n_{\text{child}}))$. Total mass is preserved: $\sum \text{child} = \sum \text{parent} = m$.

This matches MATLAB's refinement (initial_baseline.m, lines 86, 140-153): `numBins = 2*length(bin)`, with each parent bin split into two sub-bins via `partialBin = [subBins; weight-subBins]'`.

### Verdict: VERIFIED $\square$

---

## Item 5: Deduplication Correctness

**Claim:** Survivors from different parents may produce the same canonical child. The deduplication pipeline correctly removes all duplicates while preserving every unique canonical child.

### Proof

Deduplication proceeds in two stages.

**Stage 1: Intra-shard dedup via `_fast_dedup`** (lines 274-286).

```python
keys = tuple(arr[:, d-1-i] for i in range(d))
sort_idx = np.lexsort(keys).astype(np.int64)
unique_idx = _dedup_sorted(arr, sort_idx)
return arr[unique_idx]
```

`np.lexsort` sorts by its last key first. The keys are `[arr[:,d-1], arr[:,d-2], ..., arr[:,0]]`, so the primary sort key is column 0, secondary is column 1, etc. This is standard lexicographic row order. $\checkmark$

`_dedup_sorted` (lines 250-271) walks the sorted index array and keeps the first of each run of identical rows. Comparison is across ALL $d$ columns with early exit on first difference:

```python
for j in range(d):
    if arr[curr, j] != arr[prev, j]:
        is_same = False
        break
```

This correctly identifies duplicates across all columns. The output `arr[unique_idx]` preserves sorted order (since `unique_idx` is a subsequence of `sort_idx`), yielding a **sorted, deduplicated** array. $\checkmark$

**Stage 2: Cross-shard dedup via `_sorted_merge_dedup_kernel`** (lines 293-354).

Two-pointer merge of pre-sorted, pre-deduped arrays. The lexicographic comparison (lines 314-322):

```python
for c in range(d):
    if a[i, c] < b[j, c]: cmp = -1; break
    elif a[i, c] > b[j, c]: cmp = 1; break
```

Column 0 is the primary key, column 1 is secondary, etc. — the same order as lexsort. $\checkmark$

Merge logic:
- $\text{cmp} < 0$: emit from $a$, advance $a$ pointer
- $\text{cmp} > 0$: emit from $b$, advance $b$ pointer
- $\text{cmp} = 0$: emit one copy, advance BOTH pointers (cross-array dedup)

Remaining elements from either array are appended after one is exhausted (lines 342-352). The output is **sorted and deduplicated**, maintaining the invariant for subsequent rounds. $\checkmark$

**Stage 3: Tournament reduction via `_merge_dedup_shards`** (lines 357-492).

Pairwise merge rounds reduce shards until one remains:

```python
while len(current) > 1:
    for i in range(0, len(current), 2):
        if i + 1 < len(current):
            # merge pair → next_round
        else:
            next_round.append(current[i])  # carry odd shard
```

**No shard is lost:** Every shard in `current` appears in `next_round` either as part of a merged output or carried forward (line 464). The progress check (line 467: `if len(next_round) >= len(current): break`) prevents infinite loops when memory limits block merging. In that case, unmerged shards are reported and the cascade terminates gracefully (lines 2046-2057).

**Sorted-merge path** (lines 413-433): When RAM is tight but a two-pointer merge fits, shards are merged via `_sorted_merge_dedup_kernel` with memory-mapped inputs. This produces the same result as the load+vstack+dedup path, using $\sim$1x RAM instead of $\sim$3x. $\checkmark$

### Verdict: VERIFIED $\square$

---

## Item 6: Refinement-Level Canonical Handling

**Claim:** At refinement levels, ALL children of each canonical parent are tested (no canonical filter), survivors are canonicalized, and dedup captures every unique canonical child.

### Proof

The fused kernel `_fused_generate_and_prune` does NOT apply a canonical filter to children. It tests every child in the Cartesian product. This is correct, and the comment at line 1393 explains why: applying a canonical filter at refinement would silently drop canonical children whose parent is non-canonical (i.e., $\text{rev}(P)$ for canonical $P$), since $\text{rev}(P)$ is never in the parent list.

**Completeness argument.** For any canonical child $C$ (i.e., $C \le \text{rev}(C)$):

- $C$ is a child of parent $P_C = (C[0]+C[1],\; C[2]+C[3],\; \ldots)$
- $\text{rev}(C)$ is a child of $\text{rev}(P_C)$
- Exactly one of $\{P_C,\; \text{rev}(P_C)\}$ is canonical (both if palindrome)

**Case 1: $P_C$ is canonical.** Then $C$ is directly tested as a child of the canonical parent $P_C$. If it survives, it is canonicalized to $\min(C, \text{rev}(C)) = C$. $\checkmark$

**Case 2: $\text{rev}(P_C)$ is canonical** (and $P_C$ is not). Then $\text{rev}(C)$ is a child of the canonical parent $\text{rev}(P_C)$. If $\text{rev}(C)$ survives, it is canonicalized to $\min(\text{rev}(C), \text{rev}(\text{rev}(C))) = \min(\text{rev}(C), C) = C$. $\checkmark$

In both cases, the canonical child $C$ appears in the survivor set if and only if it is not pruned. Dedup (Item 5) then removes any duplicates arising from palindromic parents or other overlaps.

**Inline canonicalization** (fused kernel, lines 701-718): Each survivor is stored as $\min(\text{child}, \text{rev}(\text{child}))$ lexicographically, using the same comparison as `_canonicalize_inplace` (lines 219-243). Verified in Part 4, Item 5. $\checkmark$

### Verdict: VERIFIED $\square$

---

## Item 7: Checkpoint Integrity

**Claim:** `_save_checkpoint` and `_load_checkpoint` correctly preserve survivors and reject parameter mismatches.

### Proof

**Save** (`_save_checkpoint`, lines 1566-1587):
- Writes `checkpoint_L{level}_survivors.npy` via `np.save` (binary, lossless)
- Writes `checkpoint_meta.json` with `n_half`, `m`, `c_target`, `level_completed`, survivor shape, and cascade info
- Custom JSON converter handles numpy scalar types (`np.integer`, `np.floating`, `np.ndarray`)

**Load** (`_load_checkpoint`, lines 1590-1624):
- Reads `checkpoint_meta.json` and validates parameters (lines 1603-1610):

```python
if (meta['n_half'] != n_half or meta['m'] != m
        or meta['c_target'] != c_target):
    return None
```

If ANY parameter mismatches, the checkpoint is rejected (`return None`) and the cascade starts fresh. $\checkmark$

- Loads survivors via `np.load(npy_path, mmap_mode='r')` — read-only memory map
- Before any mutation (shuffle), the main loop creates a writable copy (lines 1822-1823):

```python
if not current_configs.flags.writeable:
    current_configs = np.array(current_configs)
```

This ensures the checkpoint file on disk is never modified. $\checkmark$

### Verdict: VERIFIED $\square$

---

## Item 8: Shuffle Does Not Affect Correctness

**Claim:** Shuffling the parent array before processing does not change the set of survivors.

### Proof

Lines 1824-1825:

```python
rng = np.random.RandomState(42)
rng.shuffle(current_configs)
```

Shuffling is a bijection on the parent array — it permutes elements but does not add, remove, or modify any parent. The processing loop covers ALL indices:

- Parallel: `pool.imap_unordered(_process_parent_shm, range(n_parents))` — every index in $\{0, \ldots, n_{\text{parents}}-1\}$ is dispatched exactly once
- Sequential: `for p_idx in range(n_parents)` — explicit full iteration

Since every parent is processed regardless of order, and the pruning tests are deterministic per-child, the set of survivors is invariant under shuffling. The fixed seed (42) ensures reproducibility across runs. $\checkmark$

### Verdict: VERIFIED $\square$

---

## Item 9: Final Proof Claim — Completeness Chain

**Claim:** If `n_survived == 0` at any level $L_k$, then every non-negative function $f$ with $\text{supp}(f) \subseteq (-1/4, 1/4)$ satisfies $\|f*f\|_\infty \ge c_{\text{target}}$.

### Proof

The proof proceeds by induction on the cascade levels.

**Base case (L0).** Every $f \ge 0$ with $\text{supp}(f) \subseteq (-1/4, 1/4)$ and $\int f = 1$ has a discretization $b_0$ at resolution $d_0 = 2 n_{\text{half}}$:

$$b_0[k] = \text{round}\bigl(m \cdot \text{avg}(f \text{ on bin } k)\bigr)$$

By the completeness of the composition generator (Item 1, Part 2), $b_0$ is one of the $\binom{m + d_0 - 1}{d_0 - 1}$ enumerated vectors. It is either:

- **Pruned by canonical filter:** Its canonical twin $\min(b_0, \text{rev}(b_0))$ is tested and gives the same $\|f*f\|_\infty$
- **Pruned by asymmetry:** $\|f*f\|_\infty \ge c_{\text{target}}$ by mass concentration (Part 1, Verification 2)
- **Pruned by dynamic threshold:** $\|f*f\|_\infty \ge c_{\text{target}}$ by Lemma 3 + per-window correction (Parts 1, 3)
- **Survivor:** Carried to the next level

**Inductive step (L$_{k-1}$ → L$_k$).** Every survivor $b_{k-1}$ from level $L_{k-1}$ is refined. By Item 2, every parent is processed and the complete Cartesian product is enumerated. For a continuous function $f$ whose coarse discretization is $b_{k-1}$, its finer discretization $b_k$ satisfies $b_k[2i] + b_k[2i+1] = b_{k-1}[i]$, so $b_k$ is a child of $b_{k-1}$ (Item 4). This child is either:

- **Pruned by asymmetry or dynamic threshold:** $\|f*f\|_\infty \ge c_{\text{target}}$
- **Survivor:** Canonicalized (Item 6) and carried to level $L_{k+1}$ after dedup (Item 5)

**Termination.** If at any level $L_k$, $n_{\text{survived}} = 0$ (line 2086):

```python
if n_survived == 0:
    info['proven_at'] = f'L{level_num}'
```

then every configuration at resolution $2^{k+1} \cdot n_{\text{half}}$ has been pruned. By the completeness chain above, every continuous function $f$ has a discretization at this resolution that was pruned (either at level $k$ or at an earlier level). Therefore $\|f*f\|_\infty \ge c_{\text{target}}$ for all such $f$, proving $c \ge c_{\text{target}}$.

**Key requirement.** This argument requires that pruning at finite resolution is sound for the continuous function. This is guaranteed by Lemma 3 (Part 1, Verification 1): the discretization error is bounded by $(4n/\ell) \cdot (2/m + 1/m^2)$ per window (or $2n \cdot (2/m + 1/m^2)$ globally), and the dynamic threshold incorporates the per-window correction $\varepsilon^2 + 2\varepsilon \cdot W_{\text{int}}$ (scaled by the $4n/\ell$ window normalization factor) to account for this error. $\checkmark$

### Verdict: VERIFIED $\square$

---

## MATLAB Correspondence

The Python cascade is a breadth-first reimplementation of MATLAB's depth-first approach. Both are complete:

| Aspect | MATLAB | Python | Equivalence |
|--------|--------|--------|-------------|
| Parent processing order | Depth-first: survivors appended to same `binStore`, processed later | Breadth-first: level-by-level, survivors collected and deduped | Both process every survivor eventually |
| Parallelism | SPMD over master bins | `multiprocessing.Pool` over parents within a level | Both test every parent |
| Deduplication | None (may test the same child from multiple parents) | Sort-based dedup between levels | Python is more efficient; both are sound |
| Canonical symmetry | Not explicitly used (MATLAB tests all compositions) | Canonical filter at L0; canonicalize-after-test at L1+ | Both cover all canonical vectors |
| Checkpointing | Save `binStore` periodically | Save survivor arrays + metadata per level | Both allow resumption |

The key structural difference — breadth-first vs depth-first — does not affect completeness. In MATLAB, `indicator` advances through `binStore` which grows as survivors are appended; the `while indicator <= lengthBinStore` loop (line 72) eventually processes all. In Python, `for level_num in range(...)` processes one level at a time, with `current_configs` carrying the complete survivor set forward.

---

## Summary

| Item | Description | Verdict |
|------|-------------|---------|
| 1 | L0 completeness: all compositions enumerated, filter chain sound | VERIFIED |
| 2 | Refinement completeness: all parents processed, Cartesian product complete | VERIFIED |
| 3 | Pre-filter soundness: parents with empty cursor range correctly skipped | VERIFIED |
| 4 | Dimension doubling: $d$ and $n_{\text{half}}$ double each level, matching paper | VERIFIED |
| 5 | Deduplication: lexsort order correct, merge-dedup sound, no shard lost | VERIFIED |
| 6 | Canonical handling: no filter at refinement, canonicalize after test, complete | VERIFIED |
| 7 | Checkpoint integrity: parameter validation, mmap copied before mutation | VERIFIED |
| 8 | Shuffle invariance: bijection on parent set, full index range processed | VERIFIED |
| 9 | Final proof claim: completeness chain valid, pruning sound for continuous $f$ | VERIFIED |

The cascade orchestration is **sound and complete**: if the cascade terminates with zero survivors at any level, the claimed lower bound $c \ge c_{\text{target}}$ holds for all non-negative functions $f$ with $\text{supp}(f) \subseteq (-1/4, 1/4)$.
