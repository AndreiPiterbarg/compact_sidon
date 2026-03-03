# L4 Optimization Roadmap

**Proof parameters**: c_target=1.40, m=20, n_half=2 (147M parents, d_child=64)

**Implemented**: Memory-mapped parent array (A), incremental autoconvolution (B).
**Rejected**: Tree-structured iteration (C).

**Current baseline**: ~150 hours on cloud CPU. (The original 5,300-hour estimate assumed 16 workers on a 128-core/51 GB machine with SharedMemory; mmap + incremental autoconvolution brought this down to the current ~150h.)

---

## Optimizations Ranked by Impact

| Rank | Optimization | Speedup | Effort | Risk |
|------|-------------|---------|--------|------|
| **1** | Adaptive survivor buffer | 1.5-2x | 0.5 days | Very low |
| **2** | Window scan reordering | 1.1-1.3x | 0.5 days | Zero |
| **3** | Int32 autoconvolution | 1.1-1.3x | 1 day | Very low |
| **4** | Precompute per-ell constants | 1.05-1.1x | 0.5 days | Zero |
| **5** | Parent pre-filtering + ordering | 1.05-1.15x | 0.5-1 day | Zero |
| **6** | Parent-level asymmetry skip | 1.01-1.05x | 0.5 days | Zero |
| **7** | Intra-level checkpointing | — (reliability) | 1-2 days | Zero |
| **8** | Distributed across N machines | Nx | 2-3 days | Low |
| **9** | Increase m (experimental) | Unknown | 1-2 days | Medium |

---

## Implementation Plan

### Phase 1: Low-Hanging Fruit (2-3 days)

These are all trivial to implement, zero-to-very-low risk, and compound multiplicatively.

1. **Adaptive survivor buffer** — Reduce buf_cap from 1M to 1K-10K rows at d_child=64. Per-worker memory drops from 556 MB to ~302 MB. More workers fit in memory → more cores utilized. Add a retry mechanism for the unlikely case a parent overflows the small buffer. *(~1.5-2x)*

2. **Window scan reordering** — Reverse the ell loop from `range(2, 2*d+1)` to `range(2*d, 1, -1)`. One-line change in the fused kernel. Wide windows prune spread-out survivors faster. *(~1.1-1.3x)*

3. **Int32 autoconvolution** — Change `conv = np.empty(conv_len, dtype=np.int64)` to `dtype=np.int32` in the fused kernel. Switch inner-loop casts from `np.int64` to `np.int32`. Widen to int64 only at the threshold comparison. Halves conv memory bandwidth. Safe for m ≤ 200. *(~1.1-1.3x)*

4. **Precompute per-ell constants** — Add two precomputed arrays (`dyn_base_ell_arr`, `two_ell_inv_4n_arr`) before the while loop, matching the existing pattern in `_prune_dynamic_int32`. Replace per-ell multiplications with array lookups. *(~1.05-1.1x)*

5. **Parent pre-filtering** — Before pool dispatch, scan all parents for infeasible bins (any bin ≥ 7 at x_cap=3). Filter these out. Also check parent-level asymmetry (sum of first 16 bins). Sort remaining parents by estimated Cartesian product size for load balance. Reduce chunksize to 16-32. *(~1.05-1.15x)*

**Combined Phase 1 estimate**: 1.5 × 1.2 × 1.2 × 1.07 × 1.1 ≈ **2.5-3x** speedup on current baseline.

### Phase 2: Reliability (1-2 days)

6. **Intra-level checkpointing** — Save a bitfield of completed parent indices + accumulated survivor shards every 5-10 minutes. On resume, skip completed parents. Essential for a multi-day run on cloud instances that may be preempted.

### Phase 3: Scale Out (2-3 days, if needed)

7. **Distributed computation** — Split the parent array across N machines. Each machine processes its chunk independently. Merge survivor shards at the end. Extends cpupod with multi-pod orchestration. Wall time scales as 1/N.

### Phase 4: Exploratory (1-2 days)

8. **Higher m experiment** — Run cascade at m=30 through L3 to measure survivor count. If L3 survivors drop below ~10M, restart at m=30 for a fundamentally easier L4.

---

## Projected Wall Times

Current baseline is ~150 hours on cloud CPU. The table below shows projected wall times after each phase of optimization.

| After | Cumulative speedup | Projected wall time |
|-------|-------------------|-------------------|
| Current baseline | 1x | ~150 hours (~6 days) |
| Phase 1 (items 1-5) | ~2.5-3x | ~50-60 hours (~2-3 days) |
| + Distributed (4 machines) | ~10-12x | ~13-15 hours (~0.5 day) |

---

## Validation

- All Phase 1 optimizations preserve mathematical correctness: buffer sizing is a memory policy; window reordering checks the same conditions; int32 produces identical convolution values (verified by existing `_prune_dynamic_int32`); constant precomputation is algebraically identical; parent pre-filtering only skips provably-infeasible or provably-prunable parents.
- **Test strategy**: After each change, run L1 (345 parents → 48,443 survivors) and L2 (48,443 → 7,499,382 survivors) and verify exact match against existing checkpoints.
