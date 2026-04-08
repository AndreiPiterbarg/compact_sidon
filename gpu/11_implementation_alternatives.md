# 16. Implementation Alternatives for Each Algorithm Component

This section breaks the algorithm into its 10 core components and provides multiple possible CUDA implementation strategies for each, with tradeoffs.

---

## Component A: Parent Dispatch & Work Distribution

**What it does:** Assigns parent compositions to GPU execution units and balances the highly irregular workload (child counts vary 1000:1+).

### Option A1: Persistent Blocks + Global Atomic Counter

```cuda
__device__ int g_next_parent;

__global__ void kernel(/* ... */) {
    while (true) {
        int pid;
        if (threadIdx.x == 0) pid = atomicAdd(&g_next_parent, 1);
        pid = __shfl_sync(0xFFFFFFFF, pid, 0);
        if (pid >= num_parents) return;
        process_parent(pid);
    }
}
```

- **Pros:** Perfect dynamic load balancing. Blocks that finish fast immediately pick up more work. Zero idle time.
- **Cons:** Atomic contention if parents are very cheap (e.g. L1). Requires persistent kernel launch with `gridDim ≤ SM_count × blocks_per_SM`.
- **Best for:** L3/L4 where parents are expensive (thousands+ children each).

### Option A2: Grid-Stride Loop (Static Round-Robin)

```cuda
for (int pid = blockIdx.x; pid < num_parents; pid += gridDim.x) {
    process_parent(pid);
}
```

- **Pros:** Zero synchronization overhead. Simplest to implement. No atomics.
- **Cons:** Static assignment causes tail imbalance. If parents are sorted by difficulty, the last few blocks get all the heavy parents. Can mitigate by random-shuffling parents first.
- **Best for:** L2 where workload variation is moderate and parent count is large enough to absorb imbalance.

### Option A3: Two-Level Dispatch (Chunked Work-Stealing)

```cuda
__device__ int g_next_chunk;
const int CHUNK_SIZE = 64;  // parents per chunk

__global__ void kernel(/* ... */) {
    while (true) {
        int chunk;
        if (threadIdx.x == 0) chunk = atomicAdd(&g_next_chunk, 1);
        chunk = __shfl_sync(0xFFFFFFFF, chunk, 0);
        int start = chunk * CHUNK_SIZE;
        if (start >= num_parents) return;
        int end = min(start + CHUNK_SIZE, num_parents);
        for (int pid = start; pid < end; pid++)
            process_parent(pid);
    }
}
```

- **Pros:** Reduces atomic contention by 64× vs Option A1 while still balancing. Good compromise.
- **Cons:** Chunk boundaries can still cause minor imbalance (last parent in chunk might be huge).
- **Best for:** Any level. Tune CHUNK_SIZE: small (4-16) for heavy parents, large (64-256) for light parents.

### Option A4: Pre-Sorted Priority Queue with Weight-Balanced Partitioning

CPU pre-computes `total_children[pid]` for all parents. Sort descending. Greedily assign to GPU blocks: always give the next-heaviest parent to the currently-least-loaded block.

- **Pros:** Near-optimal static balance. No runtime synchronization.
- **Cons:** Requires pre-computation pass on CPU (`_compute_bin_ranges` for all 147M parents). The sort itself takes time. Doesn't adapt if estimates are wrong (subtree pruning can drastically reduce actual work).
- **Best for:** Multi-GPU distribution (assign parent shards to GPUs), not intra-GPU scheduling.

### Option A5: Dynamic Parallelism (Heavy Parent Splitting)

When a block encounters a parent with >N children, it launches child kernels to split the work:

```cuda
if (total_children > THRESHOLD) {
    // Split: launch sub-grid partitioned by cursor[0] range
    int n_slices = hi_arr[0] - lo_arr[0] + 1;
    cascade_kernel_slice<<<n_slices, BLOCK_SIZE>>>(parent, lo_arr, hi_arr, ...);
} else {
    process_parent_inline();
}
```

- **Pros:** Handles extreme outliers (parents with 10M+ children) without blocking an SM.
- **Cons:** Dynamic parallelism has significant launch overhead (~5μs per child kernel). Only worthwhile for very heavy parents. Increases code complexity. H100 supports dynamic parallelism but it's not free.
- **Best for:** Edge cases where a handful of parents have 100x more children than average. Can be combined with A1/A3 as a fallback.

---

## Component B: Child Enumeration Strategy

**What it does:** Generates all children of a parent by iterating through the Cartesian product of cursor ranges. Each parent bin `i` with mass `a_i` splits into `(c, 2*a_i - c)` for `lo_i ≤ c ≤ hi_i`.

### Option B1: Sequential Mixed-Radix Gray Code (Port CPU Directly)

One thread (lane 0) drives the Knuth TAOCP 7.2.1.1 state machine. Other threads participate in conv update and window scan.

```cuda
// Lane 0 only:
int j = gc_focus[0];
if (j == n_active) break;  // done
gc_focus[0] = 0;
int pos = active_pos[j];
gc_a[j] += gc_dir[j];
cursor_smem[pos] = lo_smem[pos] + gc_a[j];
if (gc_a[j] == 0 || gc_a[j] == radix[j] - 1) {
    gc_dir[j] = -gc_dir[j];
    gc_focus[j] = gc_focus[j + 1];
    gc_focus[j + 1] = j + 1;
}
// Broadcast pos to all lanes
pos = __shfl_sync(0xFFFFFFFF, pos, 0);
```

- **Pros:** Guaranteed single-position change per step → always O(d) incremental conv update. Preserves quick-check locality. Enables subtree pruning. Exact match to CPU for verification.
- **Cons:** Lane 0 is a serial bottleneck for state transitions (~5 ops/step). Other 31/63 lanes idle during state advance.
- **Best for:** Default choice. The serial overhead is tiny compared to the parallel conv update + window scan that follows.

### Option B2: Sequential Lexicographic Odometer

Simpler state machine: increment rightmost cursor, carry left on overflow.

```cuda
// Lane 0 only:
int carry = d_parent - 1;
while (carry >= 0) {
    cursor_smem[carry]++;
    if (cursor_smem[carry] <= hi_smem[carry]) break;
    cursor_smem[carry] = lo_smem[carry];
    carry--;
}
int n_changed = d_parent - carry;
```

- **Pros:** Simpler to implement and debug. No focus-pointer bookkeeping.
- **Cons:** ~33% of steps change multiple positions ("deep carries"), requiring O(d²) full conv recompute instead of O(d) incremental. CPU benchmarks show Gray code is 1.5-2x faster due to 100% incremental updates.
- **Best for:** Prototyping / fallback. Use to verify Gray code correctness.

### Option B3: Warp-Parallel Rank/Unrank (Abandon Incremental Updates)

Each lane independently handles a different child by unranking from a global index:

```cuda
// Each lane processes children at stride WARP_SIZE
for (int64_t idx = lane; idx < total_children; idx += WARP_SIZE) {
    unrank_composition(idx, lo_arr, hi_arr, child_local);
    compute_full_autoconv(child_local, conv_local);  // O(d²)
    prune(conv_local, child_local);
}
```

Unranking for mixed-radix: decompose `idx` by dividing by radix products (like converting an integer to mixed-radix digits).

- **Pros:** Full warp utilization (all 32 lanes active and independent). No Gray code complexity. No serial bottleneck. Trivially partitionable across multiple warps/blocks for very heavy parents.
- **Cons:** Every child requires O(d²) full autoconvolution — no incremental updates. At d=64: 2048 multiply-adds per child vs ~128 for incremental. **~16x slower per child.** Loses quick-check, subtree pruning. No temporal locality at all.
- **Tradeoff:** 32 lanes active vs 16x more work/child = **net ~2x slower** than Gray code with warp-parallel update. Worse at higher d.
- **Best for:** Only if occupancy is the binding constraint, or for L0-L1 where d is tiny (d=4: O(d²) = 16 ops, negligible).

### Option B4: Hybrid Gray Code with Partitioned Outer Digits

Split the cursor space: outer digits (slow-moving) partitioned across warps, inner digits (fast-moving) via Gray code within each warp.

```
Example for d_parent=32:
- Outer 4 digits: 2^4 = 16 partitions, one per warp (within a block of 16 warps)
- Inner 28 digits: each warp runs Gray code over its partition's subspace
```

- **Pros:** 16x more warps active per parent. Each warp runs independent Gray code with full incremental updates. Load balance within a parent across warps.
- **Cons:** Each warp needs its own conv + child arrays in shared memory (16 × 2.2KB = 35KB — tight but feasible). Outer-digit partitioning requires computing initial conv for each warp's starting point (O(d²) setup once per warp). Subtree pruning only works within each warp's inner digits.
- **Best for:** Very heavy parents (>1M children) where a single warp would take too long. Launch condition: `if (total_children > THRESHOLD) use_hybrid; else use_single_warp;`

### Option B5: Batched Odometer with Partial Incremental Updates

Like B2 but batch-process children in groups of 32: generate 32 consecutive children (lexicographic), compute all 32 autoconvolutions, prune in parallel.

For the batch: children differ only in the last few cursor positions. Precompute the "common prefix" conv and add per-child deltas.

- **Pros:** Full warp utilization during pruning. Partial incremental: the common prefix conv is shared.
- **Cons:** Complex bookkeeping for the batch boundaries. Within the batch, children still need individual conv computations for the differing suffix — not fully incremental. Memory for 32 concurrent child + conv states.
- **Best for:** Not clearly better than B1 for this problem. Consider if quick-check hit rate is low (meaning most children need full window scans anyway).

---

## Component C: Autoconvolution Update

**What it does:** After each Gray code step changes bins `k1 = 2*pos` and `k2 = 2*pos+1`, update the raw_conv array in O(d) via cross-term deltas.

### Option C1: Warp-Parallel Cross-Terms (One Thread per Child Bin)

```cuda
// All threads execute in parallel:
int j = lane;  // lane ∈ [0, d_child)
if (j < d_child && j != k1 && j != k2) {
    int cj = child_smem[j];
    if (cj != 0) {
        conv_smem[k1 + j] += 2 * delta1 * cj;
        conv_smem[k2 + j] += 2 * delta2 * cj;
    }
}
__syncwarp();  // or __syncthreads() for d_child=64

// Lane 0 handles self-terms + mutual term
if (lane == 0) {
    conv_smem[2*k1] += new1*new1 - old1*old1;
    conv_smem[2*k2] += new2*new2 - old2*old2;
    conv_smem[k1+k2] += 2*(new1*new2 - old1*old2);
}
__syncwarp();
```

- **Pros:** O(d/warp_size) per thread. At d=32: one iteration per lane. At d=64: two iterations per lane. Clean mapping. Self-terms on lane 0 avoid races.
- **Cons:** Write conflicts possible: thread `j` writes to `conv[k1+j]` and thread `j-1` writes to `conv[k2+(j-1)] = conv[k1+j]` (since k2=k1+1). Need two-phase write (even j first, odd j second) or shared memory atomics.
- **Bank conflicts:** `conv_smem[k1 + j]` for consecutive j hits consecutive banks — no conflict. `conv_smem[k2 + j]` similarly fine. But the two writes from the same thread go to `k1+j` and `k2+j = k1+j+1` — adjacent banks, no conflict.
- **Best for:** Default choice for d_child=32 (1 warp) and d_child=64 (2 warps).

### Option C2: Two-Phase Even/Odd to Avoid Write Conflicts

```cuda
// Phase 1: even-indexed j only
if (j < d_child && j != k1 && j != k2 && (j & 1) == 0 && child_smem[j] != 0) {
    conv_smem[k1 + j] += 2 * delta1 * child_smem[j];
    conv_smem[k2 + j] += 2 * delta2 * child_smem[j];
}
__syncwarp();
// Phase 2: odd-indexed j only
if (j < d_child && j != k1 && j != k2 && (j & 1) == 1 && child_smem[j] != 0) {
    conv_smem[k1 + j] += 2 * delta1 * child_smem[j];
    conv_smem[k2 + j] += 2 * delta2 * child_smem[j];
}
__syncwarp();
```

- **Pros:** Guarantees no write conflicts (thread j and thread j-1 never write simultaneously to the same conv entry). No atomics needed.
- **Cons:** Two sync barriers per update instead of one. Half the lanes idle in each phase. For d=64: effectively O(d/32) per phase × 2 phases = same total work but 2x sync overhead.
- **Best for:** If we measure that shared memory atomics are slower than the two-phase approach. Worth benchmarking.

### Option C3: Shared Memory Atomics (atomicAdd on int32)

```cuda
if (j < d_child && j != k1 && j != k2 && child_smem[j] != 0) {
    int cj = child_smem[j];
    atomicAdd(&conv_smem[k1 + j], 2 * delta1 * cj);
    atomicAdd(&conv_smem[k2 + j], 2 * delta2 * cj);
}
__syncwarp();
```

- **Pros:** Single phase, single sync. Simple code. Shared memory atomics on H100 are fast (~1 cycle when no conflict, ~5 cycles on conflict).
- **Cons:** The conflict rate is low (only threads j and j-1 can conflict, and only if both are nonzero). Average ~1 conflict per update. Slight overhead vs direct write.
- **Best for:** Simplest correct implementation. Good starting point.

### Option C4: Full Recompute Every Step (Abandon Incremental)

```cuda
// Rebuild conv from scratch after each Gray code step
// Distribute O(d²) work across warp
for (int i = lane; i < d_child; i += WARP_SIZE) {
    int ci = child_smem[i];
    if (ci == 0) continue;
    atomicAdd(&conv_smem[2*i], ci * ci);
    for (int j = i + 1; j < d_child; j++) {
        int cj = child_smem[j];
        if (cj != 0)
            atomicAdd(&conv_smem[i + j], 2 * ci * cj);
    }
}
```

- **Pros:** No incremental state to maintain. Works with any enumeration order (not just Gray code). Simpler to reason about correctness.
- **Cons:** O(d²/warp_size) per step instead of O(d/warp_size). At d=64: ~64 ops/thread vs ~2 ops/thread. **~32x slower.** Completely negates the fused kernel advantage.
- **Best for:** Never, unless d is very small (d=4: only 6 multiply-adds total, trivial). Useful only for verification.

### Option C5: Register-Cached Partial Conv

Each thread caches a slice of conv in registers and updates locally, only writing back to shared memory when needed for the window scan.

```cuda
// Thread lane owns conv[lane], conv[lane + 32], conv[lane + 64], conv[lane + 96]
int conv_reg[4];  // 4 registers per thread for d_child=64 (conv_len=127, 127/32 ≈ 4)

// Update: each thread checks if its owned conv entries are affected
// conv[k1 + j] where j varies → each thread checks if k1 + j matches any of its owned indices
```

- **Pros:** Register access is ~10x faster than shared memory. Eliminates bank conflict concerns entirely.
- **Cons:** Very complex index mapping. Each thread must determine which of its 4 conv entries are affected by the update — requires conditional logic. Also, the window scan needs access to contiguous conv values, so registers must be shuffled to shared memory anyway.
- **Best for:** Likely not worth the complexity. Shared memory is fast enough for this access pattern.

---

## Component D: Quick-Check Heuristic

**What it does:** Before the expensive full window scan, re-try the (ell, s_lo) pair that killed the previous child. Kills ~85% of children at near-zero cost.

### Option D1: Lane-0 Sequential Quick-Check + Broadcast

```cuda
bool quick_killed = false;
if (lane == 0 && qc_ell > 0) {
    int64_t ws = 0;
    for (int k = qc_s; k < qc_s + qc_ell - 1; k++)
        ws += (int64_t)conv_smem[k];
    int64_t thresh = threshold_table_smem[qc_ell_idx * (S_child+1) + qc_W_int];
    quick_killed = (ws > thresh);
}
quick_killed = __shfl_sync(0xFFFFFFFF, quick_killed, 0);
```

- **Pros:** Minimal code. Preserves CPU semantics exactly. Lane 0 is already idle during its portion of warp work, so this is "free" in terms of warp slots.
- **Cons:** Sequential sum of up to ~63 elements on lane 0 while 31 other lanes idle. At 2 cycles/load+add: ~126 cycles wasted per non-killed child.
- **Best for:** If qc_ell is typically small (2-8). The idle time is still small compared to the ~200-cycle full scan that it avoids.

### Option D2: Warp-Cooperative Quick-Check Sum

```cuda
bool quick_killed = false;
if (qc_ell > 0) {
    int n_cv_qc = qc_ell - 1;
    int partial = 0;
    for (int k = qc_s + lane; k < qc_s + n_cv_qc; k += WARP_SIZE)
        partial += conv_smem[k];
    int64_t ws = warp_reduce_sum_i64((int64_t)partial);
    if (lane == 0) {
        int64_t thresh = threshold_table_smem[qc_ell_idx * (S_child+1) + qc_W_int];
        quick_killed = (ws > thresh);
    }
    quick_killed = __shfl_sync(0xFFFFFFFF, quick_killed, 0);
}
```

- **Pros:** Reduces quick-check latency from O(ell) to O(ell/32 + log2(32)). For ell=32: from 64 cycles to 2+5=7 cycles. Significant.
- **Cons:** Warp reduction adds 5 cycles of `__shfl_down_sync`. For small ell (2-4), the cooperative approach is actually slower than lane-0 sequential due to reduction overhead.
- **Best for:** When typical qc_ell is 8-32 (which it is at L3/L4 based on profiling showing ell=9-13 as top killers).

### Option D3: Speculative Parallel Quick-Check (Multiple Windows)

Instead of tracking one (ell, s_lo) pair, track the top-K killing windows. Test all K in parallel across lanes:

```cuda
// Track top 4 killing windows in shared memory
// qc_windows[4] = { (ell, s_lo, W_int), ... }
// Each of first 4 lanes tests one window
if (lane < 4 && qc_windows[lane].ell > 0) {
    int64_t ws = compute_window_sum(conv_smem, qc_windows[lane].s_lo, qc_windows[lane].ell - 1);
    int64_t thresh = threshold_table_smem[...];
    lane_killed = (ws > thresh);
}
uint32_t any_killed = __ballot_sync(0xF, lane_killed);  // check first 4 lanes
quick_killed = (any_killed != 0);
```

- **Pros:** Higher quick-kill rate — if the primary window doesn't kill, a backup might. Could push kill rate from 85% to 90%+.
- **Cons:** More shared memory for tracking. More complexity in updating which windows to track. Marginal benefit if primary already kills 85%.
- **Best for:** L4 if profiling shows the primary quick-check misses frequently. Low implementation priority.

### Option D4: No Quick-Check (Rely on Parallel Window Scan)

Skip the quick-check entirely. Every child goes through the full parallel window scan.

- **Pros:** Simplest implementation. No stateful tracking. Full warp utilization on every step.
- **Cons:** The full window scan costs ~100-200 cycles vs ~10 cycles for quick-check. On 85% of children, we waste 90-190 cycles. Net cost: 0.85 × 190 = **~160 extra cycles/child**. At 50K children/parent: +8M cycles/parent = ~4ms at 1.83GHz. Over 147M parents: **+590,000 seconds = +6.8 days.** Completely unacceptable.
- **Best for:** Never for production. Only for initial prototyping to simplify debugging.

---

## Component E: Window Scan / Pruning

**What it does:** For each non-quick-killed child, scan all (ell, s_lo) window combinations to find one where the windowed convolution sum exceeds the dynamic threshold.

### Option E1: Prefix Sum + Parallel Range Queries

```cuda
// Step 1: Build prefix sum of conv (cooperative, one-time per child)
// Use warp-level Kogge-Stone prefix sum
__shared__ int64_t prefix_conv[128];  // conv_len ≤ 127
prefix_conv[lane] = (lane < conv_len) ? (int64_t)conv_smem[lane] : 0;
// d_child=32 → conv_len=63, fits in 2 warp passes
// Kogge-Stone scan in shared memory...

// Step 2: For each ell in ell_order, test all windows in parallel
for (int ell_oi = 0; ell_oi < ell_count; ell_oi++) {
    int ell = ell_order_smem[ell_oi];
    int n_cv = ell - 1;
    int n_windows = conv_len - n_cv + 1;

    bool lane_pruned = false;
    for (int s_lo = lane; s_lo < n_windows; s_lo += WARP_SIZE) {
        int64_t ws = prefix_conv[s_lo + n_cv - 1] - (s_lo > 0 ? prefix_conv[s_lo - 1] : 0);
        int W_int = compute_W_int(prefix_c_smem, s_lo, ell, d_child);
        int64_t thresh = threshold_table_smem[ell_idx * (S_child+1) + W_int];
        if (ws > thresh) lane_pruned = true;
    }
    uint32_t pruned_mask = __ballot_sync(0xFFFFFFFF, lane_pruned);
    if (pruned_mask) {
        // Record which (ell, s_lo) killed for quick-check update
        int killer_lane = __ffs(pruned_mask) - 1;
        // Lane killer_lane broadcasts its s_lo
        break;
    }
}
```

- **Pros:** All window positions for a given ell tested in parallel (32 at a time). Prefix sum built once, reused for all ell. Early exit at ell level via ballot.
- **Cons:** Must also compute prefix_c (for W_int) once per child — another cooperative prefix sum. Two prefix sums per non-quick-killed child.
- **Best for:** Default choice. Converts sequential O(ell_count × n_windows) to O(ell_count × ceil(n_windows/32)).

### Option E2: Parallel Across Both ell AND s_lo

Flatten the (ell, s_lo) pairs into a 1D work list. Each lane picks a unique (ell, s_lo) pair:

```cuda
// Precompute: total_pairs = sum over ell of n_windows(ell)
// At d=64: ~63 ell values × ~64 avg windows = ~4000 pairs
// Each lane handles pairs at stride 32
for (int pair_idx = lane; pair_idx < total_pairs; pair_idx += WARP_SIZE) {
    auto [ell, s_lo] = decode_pair(pair_idx);
    int64_t ws = prefix_conv[s_lo + ell - 2] - (s_lo > 0 ? prefix_conv[s_lo - 1] : 0);
    int W_int = compute_W_int(...);
    int64_t thresh = threshold_table_smem[...];
    if (ws > thresh) pruned = true;
}
uint32_t pruned_mask = __ballot_sync(0xFFFFFFFF, pruned);
```

- **Pros:** Maximum parallelism — all 32 lanes testing different (ell, s_lo) pairs simultaneously. No ell-level loop serialization.
- **Cons:** Loses the CPU's "optimized ell ordering" — we can't early-exit at the ell that has highest kill rate because all ells are tested simultaneously. Net: we test more pairs before finding a killer. Also, `decode_pair()` adds overhead.
- **Best for:** Only if the optimized ell ordering doesn't help much (i.e., kill probability is roughly uniform across ell values). Probably worse than E1 in practice because ell ordering is very effective (kills within first 5 ells 90%+ of the time).

### Option E3: Sequential ell Loop + Lane-0 Sliding Window (CPU Direct Port)

Lane 0 runs the exact CPU window scan logic sequentially:

```cuda
if (lane == 0) {
    for (int ell_oi = 0; ell_oi < ell_count; ell_oi++) {
        int ell = ell_order_smem[ell_oi];
        int n_cv = ell - 1;
        int64_t ws = 0;
        for (int k = 0; k < n_cv; k++) ws += conv_smem[k];
        for (int s_lo = 0; s_lo < n_windows; s_lo++) {
            if (s_lo > 0) ws += conv_smem[s_lo + n_cv - 1] - conv_smem[s_lo - 1];
            // ... threshold check ...
        }
    }
}
// Broadcast result
pruned = __shfl_sync(0xFFFFFFFF, pruned, 0);
```

- **Pros:** Exact CPU semantics. Simplest to verify. Preserves sliding window efficiency (O(1) per window shift). Preserves ell ordering with early exit.
- **Cons:** 31 of 32 lanes completely idle during the scan. At ~100-200 cycles for the scan, this wastes ~3000-6000 lane-cycles. The window scan is 45% of CPU time — making it serial on GPU is very costly.
- **Best for:** Initial prototype only. Use to validate E1/E2 correctness.

### Option E4: Hierarchical Scan — Narrow ell Parallel, Wide ell Sequential

Split the ell range into two groups:

```
Narrow ells (2-16):  n_windows = 48-112. Fits well in parallel (few windows per lane).
Wide ells (17-128):  n_windows = 1-47. Very few windows — sequential is fine.
```

Phase 1: Test narrow ells in parallel (E1 style). If pruned, done.
Phase 2 (rare): Test wide ells sequentially on lane 0 (E3 style). Only ~10% of children reach here.

- **Pros:** Gets 90% of the parallel benefit with simpler code for the wide ell case. Wide ells have so few windows that parallelizing them gives minimal benefit.
- **Cons:** Two code paths, slightly more complex.
- **Best for:** Good practical compromise. Avoids the prefix sum overhead for the 90% that are killed by narrow ells in the quick-check or first parallel scan pass.

---

## Component F: Subtree Pruning

**What it does:** When a "slow" outer Gray code digit advances, checks whether the partial autoconvolution of the fixed prefix already exceeds all possible thresholds — if so, skips the entire inner sweep.

### Option F1: Lane-0 Sequential Subtree Check (CPU Port)

When `j == J_MIN` in the Gray code, lane 0 computes the partial conv of the fixed prefix and runs the window scan on it:

```cuda
if (lane == 0 && j == J_MIN && n_active > J_MIN) {
    compute_partial_conv(child_smem, fixed_len, partial_conv_smem);
    // Run window scan on partial_conv with W_int_max thresholds
    subtree_pruned = partial_window_scan(...);
    if (subtree_pruned) {
        // Reset inner Gray code digits
        reset_inner_gc_state(...);
    }
}
subtree_pruned = __shfl_sync(0xFFFFFFFF, subtree_pruned, 0);
```

- **Pros:** Preserves the CPU optimization that skips millions of children. Simple.
- **Cons:** The partial conv computation is O(fixed_len²) on a single lane. At fixed_len=14 (J_MIN=7, 2 child bins per parent): 91 multiply-adds. The window scan is another ~100 ops. Total ~200 ops on lane 0 while others idle. But this only triggers when an outer digit changes (~1 per N_inner children).
- **Best for:** Direct port. The subtree pruning amortizes well: 200 cycles to potentially skip thousands of children.

### Option F2: Warp-Cooperative Partial Conv + Window Scan

Distribute the partial conv computation across the warp:

```cuda
if (j == J_MIN && n_active > J_MIN) {
    // Cooperative partial conv: each lane handles part of O(fixed_len²)
    cooperative_autoconv(child_smem, fixed_len, partial_conv_smem);
    __syncwarp();

    // Cooperative window scan on partial conv
    cooperative_window_scan(partial_conv_smem, ...);
    // Result via ballot
}
```

- **Pros:** Faster than F1 when fixed_len is large (L4: fixed_len could be ~14, so O(196)/32 = 6 ops/lane).
- **Cons:** More complex. The partial conv computation doesn't map cleanly to 32 lanes because the work is triangular (i < j pairs). Need careful work distribution.
- **Best for:** L4 where fixed_len is larger. Moderate priority.

### Option F3: Skip Subtree Pruning Entirely

Don't implement subtree pruning on GPU. Enumerate all children individually.

- **Pros:** Massively simplifies the kernel. No partial conv, no Gray code reset, no J_MIN logic.
- **Cons:** At L4, subtree pruning skips ~5-20% of the Cartesian product. Losing it adds 5-20% more children to process. At 7.4T children: +370B-1.5T extra children. At 30 cycles/child: +3-12 hours on single GPU.
- **Best for:** If the subtree pruning hit rate is low (<5%). Profile on CPU first — if `n_subtree_pruned` is small relative to total children, skip it.

### Option F4: Outer-Digit Partitioning as Implicit Subtree Pruning

If using Option B4 (hybrid partitioned Gray code), the outer digits define the subtree partitions. Compute the partial conv for each outer-digit combination on CPU or in a setup kernel, and skip entire partitions that are prunable.

- **Pros:** Moves the subtree check out of the inner loop entirely. Each warp only processes non-pruned subtrees.
- **Cons:** Requires a pre-pass to evaluate all outer-digit combinations. At L4 with 4 outer digits and radix ~10 each: ~10K combinations per parent — manageable. But 147M parents × 10K = 1.47T evaluations in the pre-pass.
- **Best for:** Only if subtree pruning eliminates a large fraction of work. Likely not practical.

---

## Component G: Canonicalization

**What it does:** Converts each survivor to canonical form: `min(child, reverse(child))` lexicographically.

### Option G1: Warp-Ballot Parallel Comparison (Recommended)

```cuda
int fwd = child_smem[lane];
int rev = child_smem[d_child - 1 - lane];
int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
uint32_t lt_mask = __ballot_sync(0xFFFFFFFF, cmp < 0);
uint32_t gt_mask = __ballot_sync(0xFFFFFFFF, cmp > 0);
bool use_rev = (__ffs(lt_mask) < __ffs(gt_mask));  // 0 means no bit set → 0 < any → false
```

- **Pros:** O(1) warp time. All 32 lanes participate in the comparison simultaneously. `__ffs` finds the leftmost differing position in a single instruction. Elegant.
- **Cons:** For d_child=64 with 2 warps: need `__ballot_sync` across both warps (use `__syncthreads` + shared memory flag). Slightly more complex but still O(1).
- **Best for:** Default. Runs only on survivors (~0.001% of children) so even the simplest approach is fine.

### Option G2: Lane-0 Sequential (CPU Port)

```cuda
if (lane == 0) {
    bool use_rev = false;
    for (int i = 0; i < d_child / 2; i++) {
        if (child_smem[d_child - 1 - i] < child_smem[i]) { use_rev = true; break; }
        if (child_smem[d_child - 1 - i] > child_smem[i]) { break; }
    }
}
```

- **Pros:** Dead simple.
- **Cons:** O(d/2) on lane 0. At d=64: up to 32 iterations. ~64 cycles worst case. But this runs for <0.001% of children — total impact: nanoseconds across the entire run.
- **Best for:** If G1 is overkill. Both are fine. G1 is more "GPU-idiomatic."

---

## Component H: Survivor Collection / Stream Compaction

**What it does:** Collects the ~76K survivors (at L4) from 147M parents into a contiguous output array.

### Option H1: Shared Memory Staging + Global Atomic (Recommended)

```cuda
__shared__ int32_t surv_buf[SURV_CAP * D_CHILD];
__shared__ int surv_count;

// On survivor:
int slot = atomicAdd_block(&surv_count, 1);
write_canonical(surv_buf + slot * D_CHILD, child_smem);

// On buffer full or parent done:
if (surv_count > 0) {
    int base;
    if (threadIdx.x == 0) base = atomicAdd(d_global_count, surv_count);
    base = __shfl_sync(FULL_MASK, base, 0);
    cooperative_copy(surv_buf, d_survivors + base * D_CHILD, surv_count * D_CHILD);
    if (threadIdx.x == 0) surv_count = 0;
}
```

- **Pros:** Batches global atomics (one per flush, not per survivor). With SURV_CAP=64: ~1 flush per 64 survivors. At 76K total: ~1200 global atomics across 147M parents — zero contention.
- **Cons:** 16KB of shared memory for staging at d=64. Reduces occupancy slightly.
- **Best for:** Default choice. Battle-tested pattern in GPU programming.

### Option H2: Per-Block Global Memory Buffer (No Staging)

Pre-allocate `global_surv[num_blocks * PER_BLOCK_CAP * d_child]`. Each block writes to its own region:

```cuda
int my_region = blockIdx.x * PER_BLOCK_CAP;
// On survivor:
int slot = atomicAdd_block(&block_surv_count, 1);
write_to_global(d_survivors + (my_region + slot) * D_CHILD, child_smem);
```

After kernel: compact the sparse output by scanning all block regions.

- **Pros:** No staging buffer — saves 16KB shared memory per block. No global atomics during kernel.
- **Cons:** Wastes global memory (num_blocks × PER_BLOCK_CAP × 256B). With 4224 blocks × 1000 cap × 256B = 1.08GB. Post-kernel compaction pass needed. Scattered writes to global memory (non-coalesced if each block writes to a different region).
- **Best for:** If shared memory is the occupancy bottleneck and survivors are rare enough that the sparse output doesn't waste too much memory.

### Option H3: Warp-Vote Compaction (No Atomics)

Use `__ballot_sync` to count survivors within batches of 32 children, then `__popc` + `__ffs` for prefix sums:

```cuda
// After testing 32 children (one per lane):
uint32_t survive_mask = __ballot_sync(0xFFFFFFFF, is_survivor);
int count = __popc(survive_mask);
int my_offset = __popc(survive_mask & ((1u << lane) - 1));
```

- **Pros:** Zero atomics, zero synchronization for counting. Elegant.
- **Cons:** Only works if children are processed in parallel across lanes (Option B3). With Gray code (sequential within a warp), only one child is active at a time, making ballot meaningless.
- **Best for:** Only with Option B3 (rank/unrank parallel enumeration). Not compatible with Gray code.

### Option H4: Host-Initiated Flush via Mapped Memory

Map a host-visible counter. When GPU survivor count exceeds a threshold, the host copies survivors to disk asynchronously:

```cuda
// GPU writes to mapped (pinned) memory
__managed__ int survivor_count;
// Host monitors survivor_count, async copies when >THRESHOLD
```

- **Pros:** Infinite effective output buffer — survivors flow to host continuously.
- **Cons:** Mapped memory writes are slow (PCIe/NVLink latency). At L4 with 76K survivors: only 76K × 256B = 19.5MB total — trivially fits in GPU memory. Mapped memory is overkill.
- **Best for:** L2/L3 where survivor counts are in the hundreds of millions (147M at L3). At 147M × 128B = 18.8GB, this approaches GPU memory limits. Host-side paging becomes necessary.

---

## Component I: Global Deduplication

**What it does:** After collecting all survivors across all parents, removes duplicates (same canonical composition from different parents).

### Option I1: Multi-Pass Radix Sort on Bit-Packed Keys

Pack each d-element composition into a compact key and sort:

```cuda
// d=32, m=20: 5 bits/element × 32 = 160 bits → 3 × uint64
// Sort pass 1: sort by key[2] (least significant)
// Sort pass 2: stable sort by key[1]
// Sort pass 3: stable sort by key[0] (most significant)
cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys2, d_sorted2, n);
cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys1, d_sorted1, n);
cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys0, d_sorted0, n);
// Then unique scan
```

- **Pros:** CUB radix sort is extremely fast on H100 (billions of keys/sec for 64-bit). 3-pass for 160-bit keys. Bit packing reduces memory 5.3x.
- **Cons:** Packing/unpacking adds a kernel pass. 3 sort passes instead of 1. Still, total sort time for 147M × 24B = 3.5GB: ~1-2 seconds on H100.
- **Best for:** L3 (147M survivors, d=32). Good balance of speed and memory.

### Option I2: Column-by-Column Stable Sort (Lexsort Port)

Sort survivors column by column, last column first (like numpy lexsort):

```cuda
for (int col = d_child - 1; col >= 0; col--) {
    // Extract column into key array
    extract_column<<<...>>>(d_survivors, d_keys, n, col, d_child);
    // Stable sort key+values
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_sorted_keys,
                                     d_values, d_sorted_values, n);
    // Use sorted order for next column
}
// Unique scan on sorted array
```

- **Pros:** Conceptually simple (direct GPU analog of CPU lexsort). No bit packing needed.
- **Cons:** d_child sort passes (32 or 64). Each pass sorts N items by a single int32 key. At d=64: 64 sort passes × ~100ms each = ~6.4 seconds. Much slower than I1.
- **Best for:** Prototyping only. Too many passes for production.

### Option I3: GPU Hash Table Dedup

Insert survivors into a GPU hash table (e.g., cuCollections or custom Robin Hood):

```cuda
// Hash: FNV-1a or xxHash on the d-element composition
uint64_t hash = hash_composition(child, d_child);
// Insert into hash table
bool inserted = hash_table.insert(child, d_child, hash);
```

- **Pros:** O(n) expected time. No sorting. Works well for high duplicate rates.
- **Cons:** Hash table needs ~2x memory for load factor. At 147M × 256B: needs ~75GB — exceeds single GPU memory. Hash collisions require full d-element comparison. For low duplicate rates (L4: 76K unique out of ~100K total), hashing overhead is not justified.
- **Best for:** Only if duplicate rate is very high (>50%). Not the case at L4 (low survivor count, most unique).

### Option I4: CPU-Side Dedup After Transfer

Transfer survivors to host, use the existing numpy lexsort + Numba scan:

```python
survivors_gpu = cuda_fetch_survivors()  # GPU → CPU transfer
survivors_cpu = _fast_dedup(survivors_gpu)  # existing Numba function
```

- **Pros:** Zero GPU development effort. Reuses proven CPU code. At L4: 76K × 256B = 19.5MB transfer → <1ms over PCIe.
- **Cons:** At L3: 147M × 128B = 18.8GB transfer → ~5 seconds over PCIe Gen5. CPU dedup of 147M rows takes ~30-60 seconds with lexsort. Not terrible but adds a minute.
- **Best for:** L4 (tiny survivor count). Acceptable for L3 if GPU sort implementation isn't ready.

### Option I5: Distributed Sort Across 64 GPUs

Partition survivors by hash into 64 buckets (one per GPU). Each GPU deduplicates its bucket locally:

```
1. Each GPU: hash all local survivors → assign bucket ID (0-63)
2. All-to-all shuffle: send each survivor to its target GPU (via NCCL)
3. Each GPU: sort + unique its bucket locally
```

- **Pros:** Scales linearly. Load balanced by hash. Each GPU handles 1/64th of the data.
- **Cons:** All-to-all communication at L3: 147M × 128B = 18.8GB → ~300MB per GPU pair → ~1-2 seconds over NVSwitch. Complex orchestration.
- **Best for:** L3 when single-GPU memory can't hold all survivors during sort. At L4: overkill (76K survivors fit on one GPU).

---

## Component J: Multi-GPU Work Distribution

**What it does:** Distributes the 147M parents across 64 H100 GPUs.

### Option J1: Static Pre-Sorted Round-Robin

```python
# CPU pre-pass:
est_work = compute_estimated_children(all_parents)  # per-parent
order = np.argsort(-est_work)  # descending by work
gpu_assignments = order % 64   # round-robin heaviest first

# Transfer each GPU's shard
for gpu in range(64):
    mask = gpu_assignments == gpu
    transfer_to_gpu(gpu, parents[mask])
```

- **Pros:** Near-optimal static balance. Heavy parents distributed evenly. Simple. One-time CPU pre-pass.
- **Cons:** Work estimates may be inaccurate (subtree pruning, variable survival rates). If estimates are off by 2x, GPUs can idle. The pre-pass computes bin ranges for 147M parents — takes ~30s on CPU.
- **Best for:** Default choice. Robust for large parent counts where statistical averaging smooths imbalance.

### Option J2: Dynamic Host-Coordinated Work-Stealing

```python
# Host maintains global work queue
work_queue = deque(sorted_parents)  # sorted by estimated difficulty

# Each GPU:
while work_queue:
    batch = work_queue.pop_n(BATCH_SIZE)  # next N parents
    transfer_to_gpu(gpu_id, batch)
    launch_kernel(gpu_id, batch)
    results = wait_and_fetch(gpu_id)
    collect_survivors(results)
```

- **Pros:** Perfect dynamic balance — no GPU ever idles if work remains. Adapts to actual (not estimated) runtime.
- **Cons:** Host↔GPU round-trips add latency (~100μs per batch). Need to overlap: GPU N processes batch K while host prepares batch K+1. Complex async orchestration. With BATCH_SIZE=10K and 147M parents: 14,700 batches × 100μs = 1.47s total coordination overhead — acceptable.
- **Best for:** If workload imbalance is extreme (>10x between lightest and heaviest GPU shard). Insurance against bad static partitioning.

### Option J3: MPI + CUDA-Aware MPI (Multi-Node)

For clusters with multiple DGX nodes:

```cpp
MPI_Init(&argc, &argv);
int rank = MPI_Comm_rank(MPI_COMM_WORLD);
int nprocs = MPI_Comm_size(MPI_COMM_WORLD);

// Each rank gets parents[rank::nprocs]
int my_start = rank * (num_parents / nprocs);
int my_end = (rank + 1) * (num_parents / nprocs);
launch_kernel(parents + my_start, my_end - my_start);

// Gather survivors
MPI_Gatherv(my_survivors, my_count, MPI_INT32_T, all_survivors, ...);
```

- **Pros:** Industry-standard for multi-node GPU clusters. Efficient with CUDA-aware MPI (GPU-direct RDMA, no host staging). Well-tested with NCCL backend.
- **Cons:** Requires MPI environment setup. Adds build complexity. For single-node (8 GPUs), overkill — use CUDA multi-device API instead.
- **Best for:** 8-node × 8 GPU = 64 GPU deployments on clusters.

### Option J4: Single-GPU with Host Paging (No Multi-GPU)

If only one GPU is available, page parents in and out:

```python
for batch_start in range(0, num_parents, GPU_BATCH):
    batch = parents[batch_start : batch_start + GPU_BATCH]
    transfer_to_gpu(batch)
    launch_kernel(batch)
    fetch_survivors()
```

- **Pros:** Simplest. No multi-GPU coordination.
- **Cons:** No parallelism across GPUs. At L4: ~10 hours on single H100 vs ~10 minutes on 64.
- **Best for:** Development and testing. Production needs multi-GPU.

---

## Decision Matrix: Recommended Combinations

| Component | L2 (d=16) | L3 (d=32) | L4 (d=64) |
|-----------|-----------|-----------|-----------|
| **A: Dispatch** | A2 (grid-stride) | A3 (chunked steal) | A1 (atomic counter) |
| **B: Enumeration** | B1 (Gray code) | B1 (Gray code) | B1 (Gray code), consider B4 for heavy parents |
| **C: Conv Update** | C1 (warp-parallel) | C1 (warp-parallel) | C1 or C3 (atomics) |
| **D: Quick-Check** | D1 (lane-0) | D2 (warp-coop) | D2 (warp-coop) |
| **E: Window Scan** | E1 (prefix+parallel) | E1 (prefix+parallel) | E4 (hierarchical) |
| **F: Subtree Prune** | F3 (skip) | F1 (lane-0) | F1 or F2 (warp-coop) |
| **G: Canonicalize** | G1 (ballot) | G1 (ballot) | G1 (ballot, 2-warp) |
| **H: Collection** | H4 (host flush) | H1 (staging) | H1 (staging) |
| **I: Dedup** | I4 (CPU-side) | I1 (bit-packed sort) | I4 (CPU-side) |
| **J: Multi-GPU** | J4 (single) | J1 (pre-sort RR) | J1 (pre-sort RR) |
