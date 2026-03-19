# Optimization Briefs for L4 Cascade (c=1.40, m=20, n_half=2)

Each brief is standalone and self-contained.

**Implemented**: Memory-mapped parent array, incremental autoconvolution.
**Rejected**: Tree-structured iteration.

---

## 1. Adaptive Survivor Buffer Sizing

**Expected speedup: 1.5-2x | Effort: 0.5 days | Risk: Very low**

### Problem

Each worker process pre-allocates an output buffer to hold survivors produced by one parent's Cartesian product. At d_child=64, this buffer is sized at 1 million rows of 64 int32 columns, consuming 256 MB per worker. The memory budget calculation counts this 256 MB plus roughly 300 MB of Python/Numba process overhead, totaling 556 MB per worker. Even with memory-mapped parents (which eliminated the 18.85 GB shared-memory cost), this per-worker footprint limits the number of simultaneous workers: on a 128 GB machine the budget allows roughly 60 workers, and on a 51 GB machine roughly 25. Since the fused kernel uses one core per worker, unused cores remain idle.

The 1-million-row buffer is wildly oversized for L4. After processing 29,313 parents (0.02% of 147M), zero survivors have been found. The L3-to-L4 expansion factor trend (155x at L2, 19.6x at L3, likely <1x at L4) strongly suggests the buffer will remain empty or nearly empty for the entire level. Yet every worker reserves 256 MB for it regardless.

### Solution

Reduce the default buffer capacity from 1 million rows to a small value — 1,000 to 10,000 rows — for levels where the expected survivor count is very low. At 10,000 rows, the buffer drops from 256 MB to 2.5 MB, reducing per-worker memory from 556 MB to roughly 302 MB. This allows 47 workers on a 51 GB machine (vs. 25 before) or 110 on a 128 GB machine (vs. 60).

To ensure mathematical soundness, add a safety mechanism: if the buffer fills during a parent's processing, flag that parent and re-process it with a full-size buffer after the main pass. Since the expected number of parents producing any survivors at all is zero or extremely small, this fallback path should never trigger. The flag can be as simple as having the kernel return a sentinel count (e.g., buf_cap + 1) when it runs out of space, causing the dispatcher to queue a retry.

---

## 2. Window Scan Reordering

**Expected speedup: 1.1-1.3x | Effort: 0.5 days | Risk: Zero**

### Problem

After computing a child's autoconvolution, the pruning kernel scans across all possible "windows" — contiguous intervals of the convolution domain — to check if any window's normalized value exceeds the target threshold. The scan iterates over window widths from the smallest (ell=2, covering 2 convolution bins) to the largest (ell=2d=128, covering the entire domain), and for each width iterates over all valid starting positions. Early exit occurs as soon as any window exceeds the threshold. The current ordering starts with the narrowest windows.

At the current parameters (m=20, d=64, c_target=1.40), the narrowest windows are provably useless: the ell=2 threshold requires a single child bin of value 4 or greater, but the energy cap (x_cap=3) already prevents any bin from exceeding 3. Window widths ell=2 through approximately ell=5 almost never trigger a prune. Meanwhile, children that survive the asymmetry filter tend to have mass spread relatively evenly, which means wider windows (where many bins contribute to the sum) are the ones most likely to exceed the threshold and trigger early exit. Scanning small-to-large means the kernel burns through dozens of useless narrow-window checks before reaching the wide windows that actually prune.

### Solution

Reverse the window scan to iterate from the widest window (ell=128) down to the narrowest (ell=2). For the spread-out mass distributions typical of asymmetry-surviving children, the wide-window check triggers early exit within the first few iterations, avoiding the many subsequent narrow-window checks entirely. A more refined approach is profile-guided ordering: instrument a sample run (e.g., on the first 10,000 parents) to record which specific (width, position) window combination prunes most frequently, then sort the scan order by descending prune frequency. This data-driven ordering minimizes the expected number of windows checked before early exit. Since the window scan accounts for a significant fraction of per-child cost, reducing the average windows checked from roughly 30 to 15-20 yields a measurable speedup. The optimization does not change which children are pruned or which survive — only the order in which windows are tested — so mathematical correctness is trivially preserved.

---

## 3. Int32 Autoconvolution in the Fused Kernel

**Expected speedup: 1.1-1.3x | Effort: 1 day | Risk: Very low**

### Problem

The fused generate-and-prune kernel computes the autoconvolution using int64 arithmetic throughout. The convolution array, prefix sums, and all intermediate products are 64-bit integers. At d_child=64, the convolution array has 127 entries of 8 bytes each, totaling 1,016 bytes. This array is zeroed, filled, prefix-summed, and scanned for every child that passes the asymmetry filter — it is the hottest data structure in the inner loop.

For the current parameters (m=20), the maximum possible value in any single convolution entry before prefix-summing is bounded by m² = 400 (occurring when all mass concentrates in bins i and j with i+j = k). After prefix-summing, the maximum is the sum of all entries, which equals the square of the total mass: m² = 400. Both fit comfortably in int32 (max 2,147,483,647). The int64 arithmetic doubles the memory bandwidth consumed by the convolution loop and halves the number of values that fit in a CPU cache line, for no mathematical benefit.

### Solution

Switch the convolution array and inner-loop arithmetic to int32, matching the approach already used in the batch-mode pruning function `_prune_dynamic_int32` (which dispatches to int32 when m ≤ 200). The convolution array shrinks from 1,016 bytes to 508 bytes. The inner multiply-add loop processes 4-byte integers instead of 8-byte, doubling the effective cache throughput and potentially enabling wider SIMD vectorization (8 int32 operations per 256-bit AVX register vs. 4 int64). Widen to int64 only at the threshold comparison point where the dynamic threshold computation requires float64 precision. This is the same safe-widening pattern already validated in `_prune_dynamic_int32`. The optimization applies whenever m ≤ 200, which covers all current and foreseeable parameter choices.

---

## 4. Precompute Per-Ell Window Constants

**Expected speedup: 1.05-1.1x | Effort: 0.5 days | Risk: Zero**

### Problem

Inside the fused kernel's window scan loop, two floating-point values are computed for every window width ell: the base dynamic threshold scaled by ell (`dyn_base * ell * inv_4n`), incorporating the per-window correction `(4n/ell)(2/m + 1/m^2)`, and the mass-dependent scaling factor (`2.0 * ell * inv_4n`). These values depend only on ell, not on the child composition being tested. Yet they are recomputed for every child that reaches the window scan — potentially hundreds of millions of times. While each recomputation is just two multiplications, the aggregate cost across all children and all 126 ell values (2 through 128) is non-trivial. The compiler cannot hoist these computations out of the child loop because the window scan contains an early-exit branch (`if pruned: break`) that makes the loop trip count data-dependent.

### Solution

Before entering the main Cartesian product iteration loop, precompute two arrays of 127 entries indexed by ell, storing the pre-scaled threshold base and mass factor for each window width. Inside the window scan, replace the per-ell multiplications with array lookups. This is the same pattern already used in the batch-mode `_prune_dynamic_int32` function, which precomputes `dyn_base_ell_arr` and `two_ell_inv_4n_arr`. Applying it to the fused kernel eliminates roughly 252 floating-point multiplications per child (2 multiplies × 126 ell values), replacing them with 252 array reads. The arrays are small (roughly 2 KB total), easily fitting in L1 cache for the duration of the entire parent's processing.

---

## 5. Parent-Level Asymmetry Skip

**Expected speedup: 1.01-1.05x | Effort: 0.5 days | Risk: Zero**

### Problem

The fused kernel applies an asymmetry check to every child: it sums the first d_child/2 = 32 child bins and checks whether the left-half mass fraction exceeds the asymmetry threshold. If so, the child is pruned without computing the autoconvolution. This check costs roughly 34 operations per child.

There is a mathematical property that makes this check redundant for many parents. Each child bin pair (child[2i], child[2i+1]) sums to parent[i] regardless of the cursor value, because child[2i] = cursor[i] and child[2i+1] = parent[i] - cursor[i]. Therefore the sum of the first 32 child bins equals the sum of the first 16 parent bins — a constant for all children of that parent. If this constant puts the left fraction above the asymmetry threshold (or below its complement), then every single child of that parent will be asymmetry-pruned. The kernel currently discovers this child-by-child, wastefully enumerating the entire Cartesian product only to discard every child at the asymmetry step.

### Solution

Before dispatching a parent to the fused kernel, compute the sum of its first d_parent/2 bins and check whether it exceeds the asymmetry threshold. If so, skip the parent entirely — it provably produces zero survivors. This eliminates all Cartesian product enumeration, autoconvolution, and window scanning for that parent. Additionally, for parents that DO pass the parent-level check, the per-child asymmetry check is guaranteed to pass for all children, so it could optionally be removed from the kernel for those parents (saving 34 operations per child). The fraction of parents eliminated depends on the mass distribution of L3 survivors, but even a small percentage translates to millions of skipped parents given the 147M total.

---

## 6. Parent Pre-Filtering and Ordering

**Expected speedup: 1.05-1.15x | Effort: 0.5-1 day | Risk: Zero**

### Problem

The 147 million parent compositions are dispatched to workers via a multiprocessing pool, one parent index at a time. Three sources of waste exist in this dispatch. First, some parents are infeasible: if any bin in the parent has a value of 7 or higher, the energy cap (x_cap=3) makes it impossible to split that bin into two valid child bins, so the parent produces zero children. The kernel detects this and returns immediately, but the overhead of dispatching the index through the pool, serializing an empty result, and returning it through the pipe is non-trivial when multiplied across potentially millions of infeasible parents. Second, parents vary enormously in cost: a parent with 20 non-zero bins (each offering 2 split options) produces roughly one million children, while a parent with 5 non-zero bins produces only 32 children — a 30,000x difference. With the pool's fixed chunk size, a worker that draws a sequence of million-child parents runs for minutes while others sit idle. Third, the pool's chunk size (currently 128) is tuned for the previous levels' parent counts and may not be optimal for 147M parents, reducing dynamic load-balancing responsiveness.

### Solution

Before dispatching to the pool, make a single pass over all 147 million parents to compute per-parent metadata. For each parent, compute the per-bin split ranges using the energy cap and check if any bin's range is empty (infeasible). Filter out infeasible parents entirely, reducing the dispatch count. For the remaining parents, compute the total Cartesian product size (the product of per-bin ranges) as a proxy for processing cost. Sort or shuffle parents so that expensive and cheap parents are interleaved, preventing pathological clustering of heavy parents in one worker's chunk. Optionally, reduce the chunk size from 128 to 16-32 to improve the pool's ability to dynamically balance load when individual parent costs vary by orders of magnitude. The pre-filtering pass is cheap (O(d) per parent, fully parallelizable) and the sorting is a single O(N log N) operation on 147M integers, both negligible relative to the hours of pruning computation.

---

## 7. Intra-Level Checkpointing

**Expected speedup: 0x (reliability, not speed) | Effort: 1-2 days | Risk: Zero**

### Problem

The cascade prover checkpoints after each completed level, saving the full survivor array and metadata. L4 is estimated at ~150 hours on cloud CPU. If the process crashes, is killed by the OS, or the cloud instance is preempted, all progress on L4 is lost. The run must restart from parent 0. With 147 million parents and processing rates of thousands of parents per second, losing even a few hours of progress is costly. The current architecture provides no mechanism to resume a partially-completed level.

The difficulty is that the multiprocessing pool dispatches parents via `imap_unordered`, meaning parents are processed in arbitrary order. Simply recording "we processed parents 0 through N" does not work because parent 5,000,000 might finish before parent 100,000.

### Solution

Periodically save an intra-level checkpoint consisting of: (a) the set of completed parent indices (as a bitfield or sorted index array), (b) all survivor shards flushed to disk so far, and (c) aggregate statistics (total children processed, total survivors). On resume, load the bitfield, skip already-completed parents when dispatching to the pool, and continue accumulating survivors into the existing shards.

For 147M parents, a bitfield requires 18.4 MB (147M / 8 bytes). Saving it every 5-10 minutes adds negligible I/O overhead. The pool dispatch changes from `range(n_parents)` to a filtered iterator that skips indices already marked complete. This enables resuming L4 after any interruption with minimal lost work — at most the few minutes since the last checkpoint.

---

## 8. Distributed Computation Across Machines

**Expected speedup: Nx for N machines | Effort: 2-3 days | Risk: Low**

### Problem

Even with full utilization of a single machine, the projected wall time for L4 is ~150 hours (~6 days). The fundamental bottleneck is the sheer number of parents (147 million) that must each be independently processed. A single machine's core count is the ceiling on parallelism.

### Solution

The cascade's refinement step is embarrassingly parallel at the parent level: each parent's children are generated and pruned independently, with no communication between parents during processing. This means the parent array can be split into N chunks, each chunk assigned to a separate machine, and all machines can work simultaneously. Each machine loads only its chunk of parents (e.g., 37M parents on each of 4 machines instead of 147M), reducing both memory requirements and wall time by a factor of N. After all machines finish, the survivor arrays are collected and merged with deduplication (since different parents on different machines could theoretically produce the same canonical child). The existing cpupod infrastructure for RunPod cloud instances already supports pod creation, code synchronization, job launch, and result retrieval. Extending it to multi-pod jobs requires: a parent-splitting step that saves chunks as separate files, a launch step that assigns each pod its chunk index, and a collection step that fetches all survivor shards and runs a merge-dedup pass. Total CPU-hours are unchanged regardless of how many machines are used — only wall time decreases.

---

## 9. Increase Grid Resolution m (Alternative Parameters)

**Expected speedup: Unclear — requires experiment | Effort: 1-2 days for experiment | Risk: Medium**

### Problem

The cascade prover's pruning effectiveness depends on the gap between the target constant c_target and the effective per-window threshold c_target + (4n/ℓ)(2/m + 1/m²). This gap is the discretization correction: it accounts for the error introduced by approximating continuous functions with step functions on an m-point grid. The correction factor includes (4n/ℓ) from the window normalization. At m=20 with n_half=2, the global correction (ℓ=2, worst case) is 2n(2/m + 1/m²) = 4 × 0.1025 = 0.41, making the worst-case effective threshold 1.81. Every child composition must have its maximum windowed autoconvolution value exceed the per-window threshold to be pruned. This loose threshold (especially for narrow windows) is why so many compositions survive to become parents at the next level. At m=40 the raw correction drops to 0.05025 (global = 4 × 0.05025 = 0.201), and at m=80 it drops to 0.0252 (global = 4 × 0.0252 = 0.1006). A tighter threshold means more aggressive pruning, fewer survivors per level, and potentially fewer cascade levels before convergence.

### Solution

Re-run the cascade from L0 with a higher grid resolution m (e.g., m=30 or m=40) while keeping c_target=1.40 and n_half=2. The tighter per-window correction $(4n/\ell)(2/m + 1/m^2)$ makes pruning more effective at every level, potentially reducing the L3 survivor count from 147 million to a much smaller number, which would make L4 trivially feasible. The trade-off is that higher m increases the number of compositions at L0 (C(m+3, 3) for d=4: 1,771 at m=20 vs. 12,341 at m=30 vs. 82,251 at m=40) and increases children-per-parent at every level (since each bin has more split options on a finer grid). The net effect is not analytically predictable — the tighter pruning might more than compensate for the larger search space, or it might not. A quick experiment at m=30 through L3 (which took 16 hours at m=20) would reveal whether the L3 survivor count drops enough to justify the approach. If L3 survivors at m=30 drop below roughly 10 million, then L4 at m=30 would be straightforward even without the other optimizations.
