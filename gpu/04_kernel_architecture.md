# 4. Kernel Architecture

## 4.1 Parent-Level Parallelism (Outer Loop)

**Strategy: Persistent Thread Blocks with Grid-Stride Loop**

Each thread block processes one parent at a time, then moves to the next via a global atomic counter. This naturally balances load — blocks that finish easy parents (few children) pick up new work immediately.

```
Grid:  min(num_parents, 132 * occupancy_factor) blocks
Block: 32 threads (d_child=32) or 64 threads (d_child=64)

for (parent_idx = blockIdx.x; parent_idx < num_parents; parent_idx += gridDim.x) {
    OR (better):
    while ((parent_idx = atomicAdd(&global_counter, 1)) < num_parents) {
        process_parent(parent_idx);
    }
}
```

**Why persistent + atomic counter > grid-stride:**
- Grid-stride assigns parents in round-robin, which doesn't adapt to variable workload
- Atomic counter gives true work-stealing: blocks finishing cheap parents instantly get new work
- At 147M parents, atomic contention is negligible (~1 atomic per ~50K children processed)

## 4.2 Intra-Parent Computation (Inner Loop)

**The Gray code enumeration within a single parent is inherently sequential** — each step depends on the previous child's convolution state. This means we cannot parallelize across children of the same parent.

Instead, we parallelize **within each child's processing**:

### Warp-Parallel Incremental Convolution Update (d_child=32)

Each of 32 threads "owns" one child bin index. When position `pos` changes (Gray code step), the update is:

```cuda
// Thread lane owns child[lane] and helps update conv
int k1 = 2 * pos, k2 = k1 + 1;

// Self-terms and mutual term: single thread (lane 0 or designated)
if (lane == 0) {
    conv[2*k1] += new1*new1 - old1*old1;
    conv[2*k2] += new2*new2 - old2*old2;
    conv[k1+k2] += 2*(new1*new2 - old1*old2);
}

// Cross-terms: distributed across warp
// Each thread updates conv entries for its owned child bin
int j = lane;
if (j != k1 && j != k2 && child_smem[j] != 0) {
    int cj = child_smem[j];
    atomicAdd_block(&conv_smem[k1 + j], 2 * delta1 * cj);  // shared mem atomic
    atomicAdd_block(&conv_smem[k2 + j], 2 * delta2 * cj);
}
__syncwarp();
```

**For d_child=64:** Same pattern but with 64 threads (2 warps). Use `__syncthreads()` instead of `__syncwarp()`.

**IMPORTANT: Shared memory atomics vs. direct write.** Since each thread writes to a unique set of conv indices (k1+j and k2+j for its unique j), there are no conflicts — we can use direct writes, not atomics! The only conflict would be if two different j values map to the same conv index, which happens when `k1 + j1 == k2 + j2`, i.e., `j2 = j1 + k1 - k2 = j1 - 1`. So threads j and j-1 can conflict on one entry. **Resolution: process even-indexed j first, then odd-indexed j, with a `__syncwarp()` between.**

### Warp-Parallel Window Scan

The window scan for a given `ell` slides a window of size `n_cv = ell - 1` across the conv array. Each window position `s_lo` requires:
1. Compute window sum `ws` (sliding: add one, subtract one)
2. Compute `W_int` from `prefix_c` (2 lookups)
3. Look up threshold from `threshold_table`
4. Compare `ws > threshold`

**Parallelization strategy:** Distribute window positions across warp lanes.

```cuda
// For each ell in ell_order:
int n_windows = conv_len - (ell - 1) + 1;

// Each lane handles windows at stride 32
for (int s_lo = lane; s_lo < n_windows; s_lo += 32) {
    int64_t ws = compute_window_sum(conv_smem, s_lo, ell - 1);
    int W_int = compute_W_int(prefix_c_smem, s_lo, ell, d_child);
    int64_t thresh = threshold_table[ell_idx * (S_child+1) + W_int];
    if (ws > thresh) {
        pruned = true;  // use warp vote to propagate
    }
}
// Reduce across warp: any lane pruned?
uint32_t pruned_mask = __ballot_sync(0xFFFFFFFF, pruned);
if (pruned_mask != 0) {
    // Record killing (ell, s_lo) for quick-check
    ...
}
```

**But there's a subtlety:** The sliding window sum requires sequential updates (`ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]`). If we parallelize across s_lo, each thread must compute its window sum independently (prefix sum approach).

**Better approach: Prefix sum on conv, then parallel range queries.**

```cuda
// One-time prefix sum of conv (cooperative across warp)
// prefix_conv[k] = sum(conv[0..k])
// Window sum = prefix_conv[s_lo + n_cv - 1] - prefix_conv[s_lo - 1]

// Build prefix sum cooperatively
for (int stride = 1; stride < conv_len; stride *= 2) {
    if (lane < conv_len && lane >= stride)
        prefix_conv[lane] += prefix_conv[lane - stride];
    __syncwarp();
}

// Now each lane independently queries any window
for (int s_lo = lane; s_lo < n_windows; s_lo += WARP_SIZE) {
    int64_t ws = prefix_conv[s_lo + n_cv - 1];
    if (s_lo > 0) ws -= prefix_conv[s_lo - 1];
    // ... threshold check ...
}
```

**Problem: prefix sum must be rebuilt for each ell.** No — the prefix sum of the *raw* conv is independent of ell. The window sum for any (ell, s_lo) is just `prefix_conv[s_lo + ell - 2] - prefix_conv[s_lo - 1]`. **Build prefix sum once per child, query in parallel for all (ell, s_lo).**

This is a major optimization for GPU: instead of the CPU's sequential sliding window, we do **one prefix sum + parallel range queries**.

## 4.3 Quick-Check Optimization (GPU Adaptation)

The CPU quick-check re-tries the previous child's killing (ell, s_lo) before full scan. On GPU:
- Store `qc_ell, qc_s, qc_W_int` in registers (single thread, lane 0)
- Lane 0 performs the quick-check on the raw conv (not prefix sum — it's O(ell) sequential adds)
- If quick-killed: broadcast result via `__shfl_sync`, skip full scan
- **This saves ~85% of children from needing the full prefix sum + parallel scan**

## 4.4 Threshold Table Placement

The threshold table for d_child=64, m=20 has dimensions: `ell_count` ell values × `(S_child+1)` W_int values × 8 bytes, where `S_child = 4 * n_child * m`. At d_child=64, n_child=32, m=20: `S_child = 2560`, so 127 × 2561 × 8 = **~2.5 MB**.

Options:
1. **Shared memory** — too large at ~2.5MB; does not fit (228KB limit per SM)
2. **Constant memory** — 64KB limit, broadcast to all threads, but limited to read-only; also too small
3. **L2 cache** — automatic, fits in 50MB L2; slightly slower but the only viable option at this table size

**Recommendation:** Place in L2 cache (global memory with caching). At 2.5MB the table is too large for shared memory or constant memory. Cooperatively load frequently-accessed slices into shared memory if profiling shows L2 latency is a bottleneck.

For d_child=32, m=20: n_child=16, `S_child = 4*16*20 = 1280`, so 63 × 1281 × 8 = **~630 KB** — also exceeds shared memory; use L2 cache.

## 4.5 Survivor Collection (Stream Compaction)

**Challenge:** At L4, ~7.4 trillion children produce ~76K survivors. The survival rate is ~10^-8. We need zero-overhead collection that doesn't slow down the hot path.

**Three-tier buffering strategy:**

### Tier 1: Per-Block Shared Memory Buffer
```cuda
__shared__ int32_t surv_buf[SURV_CAP * d_child];  // e.g., 64 × 64 × 4 = 16KB
__shared__ int surv_count;

// When a survivor is found (extremely rare):
if (is_survivor) {
    int slot = atomicAdd_block(&surv_count, 1);  // block-level atomic
    for (int i = 0; i < d_child; i++)
        surv_buf[slot * d_child + i] = canonical_child[i];
}

// When buffer full or parent done:
if (surv_count >= SURV_CAP) {
    flush_to_global();
}
```

### Tier 2: Global Memory with Block-Level Atomic Reservation
```cuda
void flush_to_global() {
    __syncthreads();
    int base;
    if (threadIdx.x == 0)
        base = atomicAdd(d_global_surv_count, surv_count);
    base = __shfl_sync(0xFFFFFFFF, base, 0);  // broadcast

    // Cooperative copy from shared to global
    for (int i = threadIdx.x; i < surv_count * d_child; i += blockDim.x)
        d_survivors[base * d_child + i] = surv_buf[i];

    __syncthreads();
    if (threadIdx.x == 0) surv_count = 0;
}
```

### Tier 3: Host-Side Ring Buffer with Pinned Memory
For very large survivor counts (L2, L3), use CUDA streams and pinned memory to async-transfer survivors to host while GPU continues computing.

**Atomic contention analysis:** At L4, 76K survivors from 147M parents means ~1 flush per ~2000 parents. With 4224 concurrent blocks, that's ~2 global atomics per second — completely negligible.
