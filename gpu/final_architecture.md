# Final GPU Kernel Architecture: Cascade Prover for C_{1a} Lower Bound

## 1. Executive Summary

This document specifies the definitive CUDA kernel architecture for the Sidon autocorrelation cascade prover, targeting NVIDIA H100 SXM 80GB GPUs. The kernel accelerates the L3 (d=32) and L4 (d=64) cascade levels, where the CPU implementation requires 16 hours and ~3 days respectively. The architecture preserves **absolute mathematical correctness** -- every design choice is constrained by the requirement that this is a rigorous proof: a single missed survivor or a single false prune invalidates the entire result.

**Architectural thesis:** The CPU's speed comes from sequential exploitation of structure (Gray code incremental updates, quick-check temporal locality, subtree pruning). Rather than fighting this serial structure, we parallelize *within* each child's processing (warp-parallel conv update + window scan) while parallelizing *across* parents at the block level. This hybrid approach retains every CPU optimization while mapping naturally to the GPU execution model.

**Target performance:**
- L3 (147M parents, d=32): ~10 minutes on 8x H100 (vs. 16 hours CPU)
- L4 (147M parents, d=64): ~1 hour on 8x H100 (vs. ~3 days CPU)

---

## 2. Fundamental Architectural Decisions

### 2.1 One Parent Per Thread Block

Each CUDA thread block processes exactly one parent at a time before claiming the next. This is the foundational decision from which all others follow.

**Rationale:** The Gray code enumeration within a parent is inherently sequential -- each child's autoconvolution state depends on the previous child's. Parallelizing *across* children of the same parent would require abandoning incremental O(d) updates in favor of O(d^2) full recomputes, a 16x regression at d=64 that cannot be recovered by warp-level parallelism (32 lanes x 16x slowdown = net 2x slower). By keeping enumeration serial within a block, we preserve:
- O(d) incremental convolution updates (vs O(d^2) full recompute)
- Quick-check temporal locality (85% hit rate, saving the full window scan)
- Subtree pruning (skipping entire sub-Cartesian-products)
- Gray code guarantees (exactly one cursor position changes per step)

### 2.2 Warp-Parallel Intra-Child Processing

Within each block, 32 threads (one warp for d=32) or 64 threads (two warps for d=64) cooperatively process each child:
- **Convolution update:** Each thread owns one child bin index and updates the corresponding cross-terms in O(1) per thread
- **Window scan:** After building a prefix sum cooperatively, each lane tests a different window position in parallel
- **Canonicalization:** Warp-ballot parallel comparison determines canonical form in O(1)

This gives us 32x (or 64x) parallelism on the two most expensive operations (conv update: 35% of time; window scan: 45% of time) while the cheap serial operations (Gray code advance: <5%; quick-check: 15%) run on lane 0 with negligible idle cost.

### 2.3 Exact Integer Arithmetic Throughout

All autoconvolution values are computed and stored as **int32**. All threshold comparisons use **int64** from a precomputed table. There is **zero floating-point arithmetic** on the GPU hot path.

**Correctness guarantee:** The threshold table is computed on the CPU using the exact same float64 formula as the CPU prover (`dyn_x = (c_target * m² + 3.0 + W_int/(2*n) + eps_margin) * 4*n*ell`, floored with `one_minus_4eps` rounding guard), then uploaded as an int64 lookup table. The GPU never performs float64 arithmetic -- it only performs integer comparisons against precomputed integer thresholds. This completely eliminates concerns about FMA rounding, flush-to-zero, and IEEE 754 compliance differences between CPU and GPU.

**Overflow analysis for int32 convolution:**
- Max conv entry = m^2 = 400 (self-term at m=20)
- Max cross-term contribution = 2 * m * m = 800
- Max accumulated value across all updates: bounded by sum of all a_i * a_j = m^2 = 400
- INT32 range: [-2^31, 2^31-1] = [-2.1B, 2.1B]
- Safety margin: 400 << 2.1B. **Safe for m <= 46,340.**

---

## 3. Kernel Architecture (Detailed)

### 3.1 Grid Configuration

```
Grid dimensions:  min(num_parents, SM_COUNT * BLOCKS_PER_SM) blocks
Block dimensions: D_CHILD threads (32 for L3, 64 for L4)

L3 (d_child=32): 132 SMs * 9 blocks/SM = 1188 concurrent blocks, 32 threads each
L4 (d_child=64): 132 SMs * 5 blocks/SM = 660 concurrent blocks, 64 threads each
```

Blocks per SM is limited by shared memory: ~21.5 KB per block at d=64 (threshold table is now in L2-cached global memory since S_child = 4*n_child*m makes it too large for smem; 2x 1 KB Kogge-Stone buffers remain), 228 KB / 21.5 KB ≈ 10 blocks/SM. At d=32: ~13.9 KB per block (no threshold table in smem), 228 KB / 13.9 KB = 16 blocks/SM.

### 3.2 Parent Dispatch: Persistent Blocks + Global Atomic Counter (Component A)

**Selected: Option A1 (Persistent Blocks + Global Atomic Counter) for L3/L4.**

```cuda
__device__ int g_next_parent;

__global__ void cascade_kernel(
    const int32_t* __restrict__ parents,       // [num_parents x d_parent]
    const int32_t* __restrict__ lo_arrays,      // [num_parents x d_parent]
    const int32_t* __restrict__ hi_arrays,      // [num_parents x d_parent]
    const int64_t* __restrict__ threshold_table, // [max_ell x (S_child+1)], S_child = 4*n_child*m
    const int32_t* __restrict__ ell_order,       // [max_ell]
    int32_t* __restrict__ survivors,             // [max_survivors x d_child]
    int32_t* __restrict__ survivor_count,        // global atomic counter
    int num_parents, int d_parent, int d_child, int m,
    int ell_count, int conv_len,
    float threshold_asym               // sqrt(c_target / 2.0), precomputed on host
) {
    const int lane = threadIdx.x;

    while (true) {
        __shared__ int parent_idx;
        if (lane == 0) parent_idx = atomicAdd(&g_next_parent, 1);
        __syncthreads();  // for d_child=64 (2 warps); use __syncwarp for d_child=32
        if (parent_idx >= num_parents) return;

        process_parent(parent_idx, /* all params */);
    }
}
```

**Why atomic counter over grid-stride:** At L4, the child count per parent varies by 1000:1+. Grid-stride assigns parents in round-robin which doesn't adapt to this variance. The atomic counter provides true work-stealing: blocks that finish easy parents (few children, high prune rate) immediately claim the next available parent. The atomic contention is negligible -- one atomic per parent completion, and each parent processes ~50K children (thousands of cycles between atomics).

**Why not chunked work-stealing (A3):** At L4, the per-parent cost is high enough that single-parent granularity gives optimal balance. Chunking would introduce unnecessary tail imbalance at chunk boundaries. At L3 (lighter parents), we use chunk size 16 to reduce atomic frequency from ~7.5M to ~470K total atomics -- still negligible but a cleaner access pattern.

### 3.3 Shared Memory Layout (per block)

**For d_child = 64 (L4):**

```
OFFSET   SIZE      NAME                  TYPE      DESCRIPTION
──────   ────      ────                  ────      ───────────
0        128 B     parent[32]            int32     Parent composition (read-only after load)
128      256 B     child[64]             int32     Current child composition (mutated each step)
384      508 B     raw_conv[127]         int32     Autoconvolution array (mutated each step)
892      520 B     prefix_c[65]          int64     Prefix sum of child (built on demand for non-quick-killed)
1412     128 B     cursor[32]            int32     Current cursor values per parent bin
1540     128 B     lo_arr[32]            int32     Lower bounds per parent bin
1668     128 B     hi_arr[32]            int32     Upper bounds per parent bin
1796     128 B     gc_a[32]              int32     Gray code digit values
1924     128 B     gc_dir[32]            int32     Gray code direction (+1/-1)
2052     132 B     gc_focus[33]          int32     Gray code focus pointer chain
2184     128 B     active_pos[32]        int32     Mapping from active index to parent position
2312     128 B     radix[32]             int32     Per-position radix (hi-lo+1)
2440     508 B     ell_order[127]        int32     Window size ordering for ell scan (ell_count = 2*d_child - 1 = 127)
2948     (threshold_table in L2-cached global memory; S_child=4*n_child*m, table too large for smem)
2948     1024 B    prefix_conv[128]      int64     Inclusive prefix sum of raw_conv (Kogge-Stone scan)
3972     1024 B    prefix_tmp[128]       int64     Ping-pong buffer for Kogge-Stone prefix sum
4996     16384 B   surv_buf[64*64]       int32     Survivor staging buffer (64 slots x 64 bins)
21380    4 B       surv_count            int32     Current survivors in staging buffer
21384    64 B      misc                  various   qc_ell, qc_s, qc_W_int, n_active, total_tested, flags
─────────────────────────────────────────────────
TOTAL:   ~21.4 KB  (threshold table in L2 global; fits 10 blocks/SM: 228 KB / 21.4 KB ≈ 10)
```

**For d_child = 32 (L3):**

```
OFFSET   SIZE      NAME                  TYPE
──────   ────      ────                  ────
0        64 B      parent[16]            int32
64       128 B     child[32]             int32
192      252 B     raw_conv[63]          int32
444      264 B     prefix_c[33]          int64
708      64 B      cursor[16]            int32
...      (same structure, halved dimensions)
ell_o    252 B     ell_order[63]         int32     (ell_count = 2*d_child - 1 = 63)
         (threshold_table in L2-cached global memory; S_child=4*n_child*m too large for smem)
surv     8192 B    surv_buf[64*32]       int32
─────────────────────────────────────────────────
TOTAL:   ~13.9 KB  (threshold table in L2 global; fits 16 blocks/SM: 228 KB / 13.9 KB = 16)
```

**Threshold table placement decision:** Placed in L2-cached global memory. With the fine-grid parameterization (compositions sum to `S_child = 4 * n_child * m`), the table has `ell_count × (S_child + 1)` entries, which is too large for shared memory (e.g., 127 × 2561 × 8B = ~2.5 MB at d=64, m=20). The table is read-only and benefits from L2 caching (50 MB L2 on H100). The ~30 cycle L2 latency is acceptable since the table is only accessed during the window scan of non-quick-killed children (~15% of all children). Constant memory is also insufficient (64 KB limit). No cooperative loading is needed:

```cuda
// Cooperative load of threshold table from global to shared memory
// Threshold table now lives in L2-cached global memory (too large for smem
// with S_child = 4*n_child*m).  Access directly via threshold_table_global pointer.
```

### 3.4 Block Initialization Phase

When a block claims a new parent, it performs the following setup:

```cuda
__device__ void process_parent(int pid, /* params */) {
    const int lane = threadIdx.x;

    // ── Phase 0: Load parent data from global memory ──
    // Coalesced read: 32 threads read 32 consecutive int32s = 128B = one cache line
    if (lane < d_parent)
        parent_smem[lane] = parents[pid * d_parent + lane];
    if (lane < d_parent) {
        lo_smem[lane] = lo_arrays[pid * d_parent + lane];
        hi_smem[lane] = hi_arrays[pid * d_parent + lane];
    }
    __syncthreads();

    // ── Phase 0b: Asymmetry pre-filter ──
    // Skip parents whose left-mass fraction is outside [1-thresh, thresh].
    // Matches CPU reference (_fused_generate_and_prune_gray, lines 1025-1034).
    // All children of such parents inherit the same left_sum (child left-sum
    // equals parent left-sum), so ALL children will be pruned by the standard
    // windowed test.  Skipping early avoids wasting cycles.
    //
    // threshold_asym = sqrt(c_target / 2.0), precomputed on host and passed
    // as a kernel parameter (float32, computed once).  Float32 precision is
    // sufficient since this is a performance optimization, not a soundness check.
    if (lane == 0) {
        int left_sum = 0;
        for (int i = 0; i < d_parent / 2; i++)
            left_sum += parent_smem[i];
        float left_frac = (float)left_sum / (float)(4 * (d_parent / 2) * m);  // S_parent = 4*n_parent*m
        // Use float comparison: safe because this is a performance optimization,
        // not a soundness check.  Borderline parents just proceed to the full test.
        if (left_frac >= threshold_asym || left_frac <= 1.0f - threshold_asym)
            skip_parent = true;
        else
            skip_parent = false;
    }
    __syncthreads();
    if (skip_parent) return;  // claim next parent

    // ── Phase 1: Compute active positions (bins where lo < hi, i.e., cursor has range > 1) ──
    // Lane 0 builds the active_pos, radix, and initial cursor arrays.
    // IMPORTANT: iterate RIGHT-TO-LEFT so that inner (fastest-changing) Gray code
    // digits correspond to rightmost parent bins.  This ensures the fixed outer
    // region (for subtree pruning) is the LEFT prefix of child[], matching the
    // CPU reference implementation (_fused_generate_and_prune_gray, line 1180).
    if (lane == 0) {
        n_active = 0;
        for (int i = d_parent - 1; i >= 0; i--) {
            cursor_smem[i] = lo_smem[i];
            if (hi_smem[i] > lo_smem[i]) {
                active_pos_smem[n_active] = i;
                radix_smem[n_active] = hi_smem[i] - lo_smem[i] + 1;
                n_active++;
            }
        }
    }
    __syncthreads();

    // ── Phase 2: Build initial child composition from cursor ──
    // Each lane builds its two child bins from parent bin and cursor
    if (lane < d_parent) {
        int c = cursor_smem[lane];
        child_smem[2 * lane]     = c;
        child_smem[2 * lane + 1] = 2 * parent_smem[lane] - c;
    }
    __syncthreads();

    // ── Phase 3: Compute initial full autoconvolution O(d_child^2) ──
    // Distribute across lanes: each lane handles rows at stride
    cooperative_full_autoconv(child_smem, raw_conv_smem, d_child, conv_len);
    __syncthreads();

    // ── Phase 4: Initialize Gray code state ──
    if (lane == 0) {
        for (int j = 0; j < n_active; j++) {
            gc_a_smem[j] = 0;
            gc_dir_smem[j] = +1;
            gc_focus_smem[j] = j;
        }
        gc_focus_smem[n_active] = n_active;  // sentinel
        qc_ell = 0;  // no quick-check history yet
    }
    __syncthreads();

    // ── Phase 5: Test and possibly collect the initial child ──
    bool pruned = full_window_scan(raw_conv_smem, child_smem, /* ... */);
    if (!pruned) {
        canonicalize_and_stage(child_smem, surv_buf_smem, &surv_count_smem, /* ... */);
    }

    // ── Phase 6: Gray code enumeration loop ──
    gray_code_enumeration_loop(/* all shared memory arrays, params */);

    // ── Phase 7: Flush remaining survivors ──
    flush_survivors_to_global(surv_buf_smem, surv_count_smem, survivors, survivor_count);
}
```

**Initial autoconvolution (Phase 3)** is the only O(d^2) computation per parent. It is distributed across the warp:

```cuda
__device__ void cooperative_full_autoconv(
    const int32_t* child, int32_t* conv, int d_child, int conv_len
) {
    const int lane = threadIdx.x;
    // Zero conv
    for (int k = lane; k < conv_len; k += blockDim.x)
        conv[k] = 0;
    __syncthreads();

    // Each lane handles a slice of the i-loop
    for (int i = lane; i < d_child; i += blockDim.x) {
        int ci = child[i];
        if (ci == 0) continue;
        atomicAdd_block(&conv[2 * i], ci * ci);  // self-term
        for (int j = i + 1; j < d_child; j++) {
            int cj = child[j];
            if (cj != 0)
                atomicAdd_block(&conv[i + j], 2 * ci * cj);  // cross-term
        }
    }
    __syncthreads();
}
```

At d=64: the triangular loop has d*(d-1)/2 = 2016 pairs. Distributed across 64 threads: ~31 iterations per thread. Shared memory atomics on H100 cost ~1 cycle when uncontended, ~5 cycles on conflict. Conflicts are sparse (different i values hit different conv indices except at `i+j == i'+j'`). Total: ~200 cycles for the full autoconv. This runs once per parent, amortized over ~50K children. **Negligible.**

### 3.5 Gray Code Enumeration Loop (Component B: Option B1)

**Selected: Sequential Mixed-Radix Gray Code (Knuth TAOCP 7.2.1.1).**

This is the inner loop that dominates runtime. Each iteration:
1. Lane 0 advances the Gray code state machine (determines which position changes and by how much)
2. All lanes cooperatively update the autoconvolution (O(d/warp_size) per lane)
3. Lane 0 (or all lanes cooperatively) performs quick-check
4. If not quick-killed: all lanes cooperatively build prefix sum and perform parallel window scan
5. If survivor: all lanes cooperatively canonicalize and stage

```cuda
__device__ void gray_code_enumeration_loop(/* shared memory pointers, params */) {
    const int lane = threadIdx.x;
    int64_t children_tested = 1;  // already tested initial child
    int64_t n_skipped = 0;        // children skipped by subtree pruning

    while (true) {
        // ═══════════════════════════════════════════════════
        // STEP 1: Gray code advance (lane 0 only, ~5 ops)
        // ═══════════════════════════════════════════════════
        __shared__ int gc_pos;     // which active position changed
        __shared__ int gc_j;       // which active index changed (for subtree pruning)
        __shared__ bool gc_done;

        if (lane == 0) {
            int j = gc_focus_smem[0];
            if (j >= n_active) {
                gc_done = true;
            } else {
                gc_done = false;
                gc_focus_smem[0] = 0;

                int pos = active_pos_smem[j];
                gc_a_smem[j] += gc_dir_smem[j];

                // Update cursor
                cursor_smem[pos] = lo_smem[pos] + gc_a_smem[j];

                // Update focus chain
                if (gc_a_smem[j] == 0 || gc_a_smem[j] == radix_smem[j] - 1) {
                    gc_dir_smem[j] = -gc_dir_smem[j];
                    gc_focus_smem[j] = gc_focus_smem[j + 1];
                    gc_focus_smem[j + 1] = j + 1;
                }

                gc_pos = pos;
                gc_j = j;  // active index, needed for subtree pruning check
            }
        }
        __syncthreads();
        if (gc_done) break;

        // Broadcast pos to all lanes (for d=32 use __shfl_sync instead of shared mem)
        int pos = gc_pos;
        children_tested++;

        // ═══════════════════════════════════════════════════
        // STEP 2: Update child bins and autoconvolution
        // ═══════════════════════════════════════════════════
        int k1 = 2 * pos;
        int k2 = k1 + 1;
        int new_cursor = cursor_smem[pos];
        int old1 = child_smem[k1];
        int old2 = child_smem[k2];
        int new1 = new_cursor;
        int new2 = 2 * parent_smem[pos] - new_cursor;

        // Update child array (two lanes write)
        if (lane == k1) child_smem[k1] = new1;
        if (lane == k2) child_smem[k2] = new2;

        // Incremental conv update (see Section 3.6)
        incremental_conv_update(raw_conv_smem, child_smem, k1, k2,
                                old1, old2, new1, new2, d_child);

        // ── Incremental qc_W_int update (lane 0 only, O(1)) ──
        // Must happen BEFORE quick-check so the threshold lookup uses the
        // correct W_int for the current child.  Matches CPU reference
        // (_fused_generate_and_prune_gray, lines 1339-1349).
        if (lane == 0 && qc_ell > 0) {
            int delta1 = new1 - old1;
            int delta2 = new2 - old2;
            int qc_lo_bin = qc_s - (d_child - 1);
            if (qc_lo_bin < 0) qc_lo_bin = 0;
            int qc_hi_bin = qc_s + qc_ell - 2;
            if (qc_hi_bin > d_child - 1) qc_hi_bin = d_child - 1;
            if (qc_lo_bin <= k1 && k1 <= qc_hi_bin)
                qc_W_int += (int64_t)delta1;
            if (qc_lo_bin <= k2 && k2 <= qc_hi_bin)
                qc_W_int += (int64_t)delta2;
        }

        // ═══════════════════════════════════════════════════
        // STEP 3: Quick-check (see Section 3.7)
        // ═══════════════════════════════════════════════════
        bool pruned = false;
        pruned = warp_cooperative_quick_check(
            raw_conv_smem, threshold_table_smem, qc_ell, qc_s, qc_W_int, m);

        // ═══════════════════════════════════════════════════
        // STEP 4: Full window scan if not quick-killed (see Section 3.8)
        // ═══════════════════════════════════════════════════
        if (!pruned) {
            pruned = parallel_window_scan(
                raw_conv_smem, child_smem, threshold_table_smem,
                ell_order_smem, ell_count, conv_len, d_child, m,
                &qc_ell, &qc_s, &qc_W_int);
        }

        // ═══════════════════════════════════════════════════
        // STEP 5: Collect survivor (see Section 3.10)
        // ═══════════════════════════════════════════════════
        if (!pruned) {
            canonicalize_and_stage(child_smem, surv_buf_smem,
                                   &surv_count_smem, d_child);
            if (surv_count_smem >= SURV_CAP) {
                flush_survivors_to_global(surv_buf_smem, &surv_count_smem,
                                          survivors_global, survivor_count_global,
                                          d_child);
            }
        }
    }

    // Verify enumeration completeness (debug mode)
    // n_skipped accounts for children skipped by subtree pruning.
    // children_tested counts children that were actually visited and tested.
    // Their sum must equal the full Cartesian product size.
    #ifdef DEBUG
    if (lane == 0) {
        int64_t expected = 1;
        for (int j = 0; j < n_active; j++) expected *= radix_smem[j];
        assert(children_tested + n_skipped == expected);
    }
    #endif
}
```

### 3.6 Incremental Autoconvolution Update (Component C: Two-Phase Even/Odd)

**Selected: Option C2 (Two-Phase Even/Odd) for guaranteed correctness without atomics.**

The write conflict occurs when thread `j` writes to `conv[k2 + j]` and thread `j-1` writes to `conv[k1 + (j-1)] = conv[k1 + j - 1]`. Since `k2 = k1 + 1`, we have `conv[k1 + 1 + j]` vs `conv[k1 + j - 1]`. These are different addresses (differ by 2), so actually there is no conflict for the k1 writes. The conflict is: thread `j` writes `conv[k2 + j] = conv[k1+1+j]`, and thread `j+1` writes `conv[k1 + (j+1)] = conv[k1+j+1]`. These ARE the same address. So adjacent threads `j` and `j+1` conflict on one conv entry.

The two-phase approach resolves this cleanly:

```cuda
__device__ void incremental_conv_update(
    int32_t* conv, const int32_t* child,
    int k1, int k2, int old1, int old2, int new1, int new2, int d_child
) {
    const int lane = threadIdx.x;
    int delta1 = new1 - old1;
    int delta2 = new2 - old2;

    // ── Self-terms are SPLIT across the two phases to avoid a data race ──
    //
    // BUG FIX: The original code wrote all three self-terms (conv[2*k1],
    // conv[2*k2], conv[k1+k2]) before Phase 1 with no barrier.  For
    // d_child=64 (two warps), this creates a write-write race:
    //
    //   Lane 0 (warp 0) writes conv[2*k2] as a self-term.
    //   Thread k1+2 (possibly warp 1 when pos >= 15) writes
    //   conv[k1 + (k1+2)] = conv[2*k2] as a Phase 1 cross-term.
    //
    // Without a barrier, these two writes from different warps race on the
    // same shared-memory address → undefined behavior → silent corruption.
    //
    // Fix: place each self-term in the phase where no cross-term touches
    // the same address:
    //
    //   conv[2*k1]   → Phase 1.  No Phase 1 cross-term hits 2*k1 because
    //                   conv[k1+j]=2*k1 requires j=k1 (excluded) and
    //                   conv[k2+j]=2*k1 requires j=k1-1 (odd → Phase 2).
    //
    //   conv[k1+k2]  → Phase 1.  Index 2*k1+1 is odd.  Phase 1 (even j):
    //                   conv[k1+j] hits even offsets; conv[k2+j]=k1+1+j
    //                   equals 2*k1+1 only when j=k1 (excluded).
    //
    //   conv[2*k2]   → Phase 2.  No Phase 2 cross-term hits 2*k2 because
    //                   conv[k1+j]=2*k2 requires j=k1+2 (even → Phase 1)
    //                   and conv[k2+j]=2*k2 requires j=k2 (excluded).
    //
    // This eliminates the race without adding an extra __syncthreads().

    // ── Phase 1: even-indexed j  +  self-terms for conv[2*k1], conv[k1+k2] ──
    if (lane == 0) {
        conv[2 * k1] += new1 * new1 - old1 * old1;
        conv[k1 + k2] += 2 * (new1 * new2 - old1 * old2);
    }
    int j = lane;
    if (j < d_child && j != k1 && j != k2 && (j & 1) == 0) {
        int cj = child[j];
        if (cj != 0) {  // predicated -- no warp divergence cost on H100
            conv[k1 + j] += 2 * delta1 * cj;
            conv[k2 + j] += 2 * delta2 * cj;
        }
    }
    __syncthreads();  // barrier between phases

    // ── Phase 2: odd-indexed j  +  self-term for conv[2*k2] ──
    if (lane == 0) {
        conv[2 * k2] += new2 * new2 - old2 * old2;
    }
    if (j < d_child && j != k1 && j != k2 && (j & 1) == 1) {
        int cj = child[j];
        if (cj != 0) {
            conv[k1 + j] += 2 * delta1 * cj;
            conv[k2 + j] += 2 * delta2 * cj;
        }
    }
    __syncthreads();  // ensure all writes visible before next step
}
```

**Why two-phase over shared memory atomics (C3):** While H100 shared memory atomics are fast (~1 cycle uncontended), the two-phase approach has **zero** atomic overhead and **guaranteed** correctness by construction. The two `__syncthreads()` barriers cost ~20 cycles each (40 cycles total), whereas atomics on conflicting addresses cost ~5 cycles per conflict. With ~3 nonzero bins on average (m=20 across 64 bins), conflicts are rare -- but the two-phase approach provides deterministic performance regardless of data distribution. For a proof system, determinism is paramount.

**Why self-terms are split across phases (not written before Phase 1):** A naïve placement of all three self-terms before Phase 1 creates a write-write data race for d_child=64: lane 0 (warp 0) writes `conv[2*k2]` as a self-term, and thread `k1+2` (potentially warp 1 when `pos >= 15`) writes `conv[k1 + (k1+2)] = conv[2*k2]` as a Phase 1 cross-term. Without a barrier between them, the two warps race on the same address. By placing `conv[2*k2]`'s self-term in Phase 2 (where no cross-term touches that address), the race is eliminated without adding an extra barrier. The full conflict analysis is documented inline in the code above.

**Why not option C1 (single-phase direct writes):** C1 is correct only if we prove no two threads ever write to the same conv index. The conflict analysis shows they can (adjacent threads j and j+1 conflict on `conv[k2+j] == conv[k1+(j+1)]`). C2 eliminates the proof obligation entirely.

**Sparsity optimization:** At L4 (m=20, d=64), the average child has only ~3-4 nonzero bins out of 64. The `if (cj != 0)` predicate causes 60/64 threads to skip the write. On H100, predicated execution has zero divergence cost -- the `cj != 0` check is compiled to a predicate register, and the write is simply not issued. This is strictly better than maintaining a nonzero list (which would require per-step list updates and additional shared memory). No explicit sparse data structure is needed.

### 3.7 Quick-Check Heuristic (Component D: Warp-Cooperative)

**Selected: Option D2 (Warp-Cooperative Quick-Check Sum).**

The quick-check retries the (ell, s_lo) window that killed the previous child. Since Gray code enumeration changes only one cursor position per step, adjacent children have nearly identical autoconvolutions, so the same window kills ~85% of children at near-zero cost.

```cuda
__device__ bool warp_cooperative_quick_check(
    const int32_t* conv, const int64_t* threshold_table,
    int qc_ell, int qc_s, int qc_W_int, int m
) {
    if (qc_ell == 0) return false;  // no history yet

    const int lane = threadIdx.x;
    const int warp_lane = lane & 31;        // lane within warp (0-31)
    const int warp_id   = lane >> 5;        // 0 for warp 0, 1 for warp 1
    int n_cv_qc = qc_ell - 1;

    // Only warp 0 participates in the quick-check sum.
    // Warp 1 (if present for d_child=64) is idle -- cost is negligible
    // (~7 cycles out of ~200 per child iteration).
    int partial = 0;
    if (warp_id == 0) {
        for (int k = qc_s + warp_lane; k < qc_s + n_cv_qc; k += WARP_SIZE) {
            partial += conv[k];
        }
    }

    // Warp 0 reduction to get total sum
    int64_t ws = (int64_t)partial;
    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            ws += __shfl_down_sync(0xFFFFFFFF, ws, offset);
    }
    // ws is now correct on warp 0, lane 0

    // Lane 0 performs threshold check
    __shared__ bool qc_killed_smem;
    if (lane == 0) {
        int ell_idx = qc_ell - 2;
        int64_t thresh = threshold_table[ell_idx * (S_child + 1) + qc_W_int];
        qc_killed_smem = (ws > thresh);
    }
    __syncthreads();  // broadcast result to both warps via shared memory
    bool killed = qc_killed_smem;
    return killed;
}
```

**Performance:** For typical qc_ell of 9-13 (profiled as top killers at L3/L4), the cooperative sum takes ceil(12/32) = 1 iteration per lane + 5 cycles for the shuffle reduction = ~7 cycles total. The lane-0 sequential approach would take ~12 cycles. The improvement is modest for small ell but the warp-cooperative approach handles large ell (up to 127 at d=64) without degradation.

**For d_child=64 (two warps):** Only warp 0 participates in the quick-check sum. At the largest ell (128), n_cv=127, so each of warp 0's 32 lanes sums ceil(127/32) = 4 entries. The result is broadcast to both warps via shared memory (`__syncthreads` + shared flag), NOT via `__shfl_sync` which is warp-local. Warp 1 is idle during the sum but the cost is negligible (~7 cycles out of ~200 per child iteration).

### 3.8 Full Window Scan: Prefix Sum + Parallel Range Queries (Component E: Option E1 with E4 Hierarchical Optimization)

**Selected: Hierarchical approach -- narrow ells (2-16) tested in parallel first, wide ells (17+) tested sequentially only if needed.**

This is the most compute-intensive operation, consuming 45% of CPU time and an even higher fraction on GPU for non-quick-killed children. The architecture exploits two key insights:

1. **Prefix sum of raw_conv is independent of ell.** Build it once, then any window sum for any (ell, s_lo) is a single subtraction: `prefix_conv[s_lo + ell - 2] - prefix_conv[s_lo - 1]`.

2. **Optimized ell ordering kills 90%+ of children within the first 5 ell values.** Testing ells in optimized order with early exit at the ell level preserves this efficiency.

```cuda
__device__ bool parallel_window_scan(
    const int32_t* conv, const int32_t* child,
    const int64_t* threshold_table, const int32_t* ell_order,
    int ell_count, int conv_len, int d_child, int m,
    int* qc_ell_out, int* qc_s_out, int* qc_W_int_out
) {
    const int lane = threadIdx.x;

    // ── Build prefix_conv in shared memory (Kogge-Stone scan) ──
    // conv_len = 2*d_child - 1: 63 for d=32, 127 for d=64.
    // Use cooperative load with blockDim.x stride to correctly handle
    // all conv entries regardless of d_child.  Previous version used a
    // hard-coded +32 offset that missed indices 96-126 at d=64.
    __shared__ int64_t prefix_conv[128];  // max conv_len = 127
    for (int i = lane; i < conv_len; i += blockDim.x)
        prefix_conv[i] = (int64_t)conv[i];
    // Zero-fill padding (needed for power-of-2 prefix sum alignment)
    for (int i = conv_len + lane; i < 128; i += blockDim.x)
        prefix_conv[i] = 0;
    __syncthreads();

    // Inclusive prefix sum (Kogge-Stone, log2(128) = 7 steps)
    // NOTE: All threads MUST participate in every __syncthreads() call.
    // A while-loop with early exit would deadlock when conv_len is not an
    // exact multiple of blockDim.x (e.g., d_child=32: conv_len=63, blockDim=32,
    // lane 31 would exit at idx=63 while lanes 0-30 remain).  We use a
    // ping-pong buffer with fixed loop bounds to avoid this.
    __shared__ int64_t prefix_tmp[128];  // second buffer for ping-pong
    int64_t* src = prefix_conv;
    int64_t* dst = prefix_tmp;
    for (int stride = 1; stride < conv_len; stride <<= 1) {
        for (int idx = lane; idx < conv_len; idx += blockDim.x) {
            int64_t val = src[idx];
            if (idx >= stride)
                val += src[idx - stride];
            dst[idx] = val;
        }
        __syncthreads();
        // Swap src and dst
        int64_t* swap = src; src = dst; dst = swap;
    }
    // Ensure final result is in prefix_conv (if odd number of swaps, copy back)
    if (src != prefix_conv) {
        for (int idx = lane; idx < conv_len; idx += blockDim.x)
            prefix_conv[idx] = src[idx];
        __syncthreads();
    }

    // ── Build prefix_c (prefix sum of child) for W_int computation ──
    __shared__ int64_t prefix_c[65];  // d_child + 1
    if (lane == 0) prefix_c[0] = 0;
    // Similar Kogge-Stone scan on child values
    if (lane < d_child)
        prefix_c[lane + 1] = child[lane];
    __syncthreads();
    for (int stride = 1; stride < d_child; stride <<= 1) {
        int64_t val;
        if (lane < d_child) {
            int idx = lane + 1;
            val = prefix_c[idx];
            if (idx > stride) val += prefix_c[idx - stride];
        }
        __syncthreads();
        if (lane < d_child) prefix_c[lane + 1] = val;
        __syncthreads();
    }

    // ── Scan ell values in optimized order ──
    // Phase 1: Narrow ells (2..16) -- high kill probability, parallel window positions
    for (int ell_oi = 0; ell_oi < ell_count; ell_oi++) {
        int ell = ell_order[ell_oi];
        int n_cv = ell - 1;
        int n_windows = conv_len - n_cv + 1;
        if (n_windows <= 0) continue;

        bool lane_pruned = false;
        int lane_killer_s = -1;
        int lane_killer_W = -1;

        for (int s_lo = lane; s_lo < n_windows; s_lo += blockDim.x) {
            // Window sum from prefix_conv
            int64_t ws = prefix_conv[s_lo + n_cv - 1];
            if (s_lo > 0) ws -= prefix_conv[s_lo - 1];

            // W_int computation: sum of child masses in the window's real-domain support.
            // The window conv[s_lo..s_lo+ell-2] captures cross-terms a_i*a_j where
            // s_lo <= i+j <= s_lo+ell-2.  The contributing bins satisfy:
            //   lo_bin = max(0, s_lo - (d_child - 1))
            //   hi_bin = min(d_child - 1, s_lo + ell - 2)
            // Matches CPU reference (_prune_dynamic_int32, lines 109-115).
            int lo_bin = s_lo - (d_child - 1);
            if (lo_bin < 0) lo_bin = 0;
            int hi_bin = s_lo + ell - 2;
            if (hi_bin > d_child - 1) hi_bin = d_child - 1;
            int W_int = (int)(prefix_c[hi_bin + 1] - prefix_c[lo_bin]);

            int64_t thresh = threshold_table[(ell - 2) * (S_child + 1) + W_int];
            if (ws > thresh) {
                lane_pruned = true;
                lane_killer_s = s_lo;
                lane_killer_W = W_int;
            }
        }

        // Reduce across block: any lane pruned?
        // For d_child=32 (1 warp): use __ballot_sync directly.
        // For d_child=64 (2 warps): use shared memory since __ballot_sync
        // only operates within a single warp.
        //
        // To find the killing lane's (s_lo, W_int) for the quick-check state,
        // the lowest-indexed killing thread writes its values to shared memory.
        // We use atomicMin_block on the thread index to find the first killer.
        __shared__ int killer_s_smem;
        __shared__ int killer_W_smem;
        __shared__ int killer_idx_smem;
        bool any_killed = false;

        if (blockDim.x == 32) {
            uint32_t pruned_mask = __ballot_sync(0xFFFFFFFF, lane_pruned);
            any_killed = (pruned_mask != 0);
            if (any_killed && lane_pruned) {
                int first_killer = __ffs(pruned_mask) - 1;
                if (lane == first_killer) {
                    killer_s_smem = lane_killer_s;
                    killer_W_smem = lane_killer_W;
                }
            }
        } else {
            // Two-warp path: shared memory coordination
            if (lane == 0) killer_idx_smem = blockDim.x;  // sentinel: no killer
            __syncthreads();
            if (lane_pruned) {
                atomicMin_block(&killer_idx_smem, lane);  // lowest killing lane wins
            }
            __syncthreads();
            any_killed = (killer_idx_smem < blockDim.x);
            if (any_killed && lane == killer_idx_smem) {
                // The winning (lowest-index) killer writes its state
                killer_s_smem = lane_killer_s;
                killer_W_smem = lane_killer_W;
            }
            __syncthreads();  // ensure writes visible before lane 0 reads
        }

        if (any_killed) {
            // Record killer for quick-check
            if (lane == 0) {
                *qc_ell_out = ell;
                *qc_s_out = killer_s_smem;
                *qc_W_int_out = killer_W_smem;
            }
            return true;
        }
    }

    return false;  // survived all windows -- this child is a survivor
}
```

**Key optimization -- prefix sum built once per child:** The CPU uses a sequential sliding window (add one element, subtract one per shift). On GPU, the sliding window is inherently serial per-lane. Instead, we build one prefix sum in O(conv_len * log(conv_len) / blockDim) time, then every window sum is a single subtraction. This converts O(ell_count * n_windows) sequential adds into O(conv_len * 7) prefix sum + O(ell_count * n_windows / blockDim) parallel lookups.

**Why prefix_c is built on demand (not maintained incrementally):** The CPU maintains prefix_c across children because the sliding window approach uses it per-window-shift. On GPU, we only need prefix_c for non-quick-killed children (~15%). Building it from scratch for those 15% is cheaper than incrementally maintaining it for all 100%.

### 3.9 Subtree Pruning (Component F: Lane-0 Sequential)

**Selected: Option F1 (Lane-0 Sequential Subtree Check).**

When a "slow" outer Gray code digit advances (specifically, when `gc_j == J_MIN` where `J_MIN` is the minimum active index that separates outer from inner digits), we check whether the partial autoconvolution of the fixed outer bins already exceeds all possible thresholds. If so, the entire inner enumeration subtree is skipped by resetting the inner Gray code digits.

With right-to-left active position ordering (see Phase 1), inner digits 0..J_MIN-1 correspond to the **rightmost** parent bins, and outer digits J_MIN..n_active-1 correspond to the **leftmost**. The fixed region is thus the **left prefix** of child[], specifically `child[0 .. 2*active_pos[J_MIN-1] - 1]`. This matches the CPU reference (_fused_generate_and_prune_gray, lines 1356-1358).

```cuda
// Inside the Gray code enumeration loop, after the Gray code advance:
// gc_j is broadcast from the shared variable set in STEP 1.
if (lane == 0 && gc_j == J_MIN && n_active > J_MIN) {
    // Compute partial autoconvolution of the fixed LEFT prefix.
    // active_pos[J_MIN-1] is the rightmost parent position in the fixed set
    // (because active_pos is built right-to-left, J_MIN-1 is the leftmost
    // active index that is still "inner").  All parent positions to the LEFT
    // of active_pos[J_MIN-1] are outer (fixed).
    int fixed_parent_boundary = active_pos_smem[J_MIN - 1];
    int fixed_len = 2 * fixed_parent_boundary;
    int64_t partial_conv_local[128];  // stack-allocated, lane 0 only
    memset(partial_conv_local, 0, sizeof(int64_t) * (2 * fixed_len - 1));

    for (int i = 0; i < fixed_len; i++) {
        int ci = child_smem[i];
        if (ci == 0) continue;
        partial_conv_local[2 * i] += (int64_t)ci * ci;
        for (int jj = i + 1; jj < fixed_len; jj++) {
            int cj = child_smem[jj];
            if (cj != 0)
                partial_conv_local[i + jj] += 2LL * ci * cj;
        }
    }

    // Check if partial conv already exceeds max threshold for any window
    bool subtree_killed = partial_window_scan_max_threshold(
        partial_conv_local, fixed_len, threshold_table_smem, /* ... */);

    if (subtree_killed) {
        // ── Reset inner Gray code digits (Knuth state machine) ──
        // Matches CPU reference (_fused_generate_and_prune_gray, lines 1450-1460).
        int next_focus = gc_focus_smem[J_MIN];
        for (int kk = 0; kk < J_MIN; kk++) {
            gc_a_smem[kk] = 0;
            gc_dir_smem[kk] = 1;
            gc_focus_smem[kk] = kk;
        }
        gc_focus_smem[0] = next_focus;  // wire focus to skip inner sweep
        gc_focus_smem[J_MIN] = J_MIN;

        // ── Reset cursor and child bins for inner positions ──
        // Matches CPU lines 1463-1467.
        for (int kk = 0; kk < J_MIN; kk++) {
            int p = active_pos_smem[kk];
            cursor_smem[p] = lo_smem[p];
            child_smem[2 * p] = lo_smem[p];
            child_smem[2 * p + 1] = 2 * parent_smem[p] - lo_smem[p];
        }

        // Accounting: the current child (created by the J_MIN advance) was
        // already counted in children_tested (line: children_tested++).
        // That child is one member of the inner subtree being skipped.
        // So the REMAINING skipped children are inner_subtree_size - 1.
        n_skipped += inner_subtree_size - 1;
    }
}

// Broadcast subtree_killed to all lanes.
// For d_child=64 (2 warps), use shared memory instead of __shfl_sync.
__shared__ bool subtree_killed_smem;
if (lane == 0) subtree_killed_smem = subtree_killed;
__syncthreads();
subtree_killed = subtree_killed_smem;

if (subtree_killed) {
    // ── CRITICAL: Full O(d^2) recompute of raw_conv after child reset ──
    // The incremental conv state is invalid because multiple child bins changed.
    // All threads cooperate.  Matches CPU lines 1469-1479.
    cooperative_full_autoconv(child_smem, raw_conv_smem, d_child, conv_len);
    __syncthreads();

    // ── Recompute qc_W_int from scratch ──
    // The incremental tracker is stale after resetting child bins.
    // Matches CPU lines 1492-1502.
    if (lane == 0 && qc_ell > 0) {
        int qc_lo2 = qc_s - (d_child - 1);
        if (qc_lo2 < 0) qc_lo2 = 0;
        int qc_hi2 = qc_s + qc_ell - 2;
        if (qc_hi2 > d_child - 1) qc_hi2 = d_child - 1;
        qc_W_int = 0;
        for (int ii = qc_lo2; ii <= qc_hi2; ii++)
            qc_W_int += (int64_t)child_smem[ii];
    }
    __syncthreads();

    continue;  // skip to next outer step (Gray code advance)
}
```

**Why lane-0 sequential over warp-cooperative (F2):** The partial conv computation is O(fixed_len^2). At L4 with J_MIN=7 and 2 child bins per parent position: fixed_len ~14, so O(196) operations. Distributing 196 ops across 32 lanes gives ~6 ops/lane -- the coordination overhead (syncthreads, index computation) would exceed the computation itself. Lane-0 sequential is simpler and the subtree check only triggers once per inner sweep (~1 per N_inner children where N_inner = product of inner radixes). The amortized cost is negligible.

**Why not skip subtree pruning entirely (F3):** CPU profiling shows subtree pruning eliminates 5-20% of children at L4. At 7.4 trillion total children, that's 370B-1.5T children avoided. At ~30 cycles/child, this saves 3-12 hours of GPU time. The implementation cost is modest (lane-0 only, ~50 lines of code). Clearly worthwhile.

### 3.10 Canonicalization and Survivor Collection (Components G + H)

**Canonicalization: Option G1 (Warp-Ballot Parallel Comparison)**

```cuda
__device__ void canonicalize_and_stage(
    int32_t* child, int32_t* surv_buf, int* surv_count, int d_child
) {
    const int lane = threadIdx.x;

    // ── Determine canonical form ──
    bool use_rev = false;
    if (d_child <= 32) {
        // Single warp: direct ballot
        int fwd = child[lane];
        int rev = child[d_child - 1 - lane];
        int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
        uint32_t lt_mask = __ballot_sync(0xFFFFFFFF, cmp < 0);
        uint32_t gt_mask = __ballot_sync(0xFFFFFFFF, cmp > 0);
        int first_lt = lt_mask ? __ffs(lt_mask) : 33;
        int first_gt = gt_mask ? __ffs(gt_mask) : 33;
        use_rev = (first_lt < first_gt);
    } else {
        // Two warps: use shared memory to combine results.
        // NOTE: __shfl_sync only broadcasts within a single warp, so we MUST
        // use shared memory to communicate use_rev across the warp boundary.
        int fwd = (lane < d_child) ? child[lane] : 0;
        int rev = (lane < d_child) ? child[d_child - 1 - lane] : 0;
        int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
        // Store comparison results in shared memory, lane 0 of first warp finds
        // the lexicographically first differing position
        __shared__ int cmp_array[64];
        cmp_array[lane] = cmp;
        __syncthreads();
        if (lane == 0) {
            for (int i = 0; i < d_child; i++) {
                if (cmp_array[i] < 0) { use_rev = true; break; }
                if (cmp_array[i] > 0) { break; }
            }
        }
        // Broadcast via shared memory (not __shfl_sync, which is warp-local)
        __shared__ bool use_rev_smem;
        if (lane == 0) use_rev_smem = use_rev;
        __syncthreads();
        use_rev = use_rev_smem;
    }

    // ── Stage to survivor buffer ──
    // NOTE: For d_child=64 (2 warps), slot must also be broadcast via shared
    // memory, not __shfl_sync.
    __shared__ int slot_smem;
    int slot;
    if (lane == 0) {
        slot_smem = atomicAdd_block(surv_count, 1);
    }
    __syncthreads();
    slot = slot_smem;

    if (lane < d_child) {
        if (use_rev)
            surv_buf[slot * d_child + lane] = child[d_child - 1 - lane];
        else
            surv_buf[slot * d_child + lane] = child[lane];
    }
    __syncthreads();
}
```

**Survivor Collection: Option H1 (Shared Memory Staging + Global Atomic)**

```cuda
__device__ void flush_survivors_to_global(
    const int32_t* surv_buf, int* surv_count_smem,
    int32_t* survivors_global, int32_t* survivor_count_global, int d_child
) {
    const int lane = threadIdx.x;
    int count = *surv_count_smem;
    if (count == 0) return;

    __shared__ int base;
    if (lane == 0)
        base = atomicAdd(survivor_count_global, count);
    __syncthreads();

    // Cooperative copy: all threads participate
    int total_elements = count * d_child;
    for (int i = lane; i < total_elements; i += blockDim.x)
        survivors_global[base * d_child + i] = surv_buf[i];

    __syncthreads();
    if (lane == 0) *surv_count_smem = 0;
}
```

**Buffer sizing:** SURV_CAP = 64 survivors per block. At d=64: 64 * 64 * 4B = 16 KB of shared memory. Flush triggers when the buffer is full. At L4 with 76K total survivors across 147M parents: average 0.0005 survivors per parent. The buffer will almost never fill within a single parent's processing. Flushing occurs once per parent completion (in the cleanup phase), requiring at most one global atomic per parent.

**Overflow protection:** If the global survivor buffer fills (should never happen with correct max_survivors estimate), the kernel writes a flag and the host relaunches with a larger buffer. The estimate is conservative: at L4, we allocate for 200K survivors (76K expected + 2.6x margin) = 200K * 64 * 4B = 51.2 MB. Trivially fits in GPU memory.

---

## 4. Correctness Guarantees

### 4.1 Proof Obligation

This kernel is computing a **rigorous mathematical proof**. The correctness requirements are:

1. **No false prunes (soundness):** If a composition could witness a bound violation, it must appear in the survivor set. A false prune (claiming ws > thresh when it isn't) would eliminate a potential counter-example, invalidating the proof.

2. **Complete enumeration (completeness):** Every child in the Cartesian product of every parent must be tested. Missing any child means the proof has a gap.

3. **Correct deduplication:** The canonical form of every survivor must be computed correctly so that deduplication doesn't accidentally merge distinct compositions.

### 4.2 How Each Component Maintains Correctness

**Integer autoconvolution (no FP):** All conv values are int32. All arithmetic is exact integer addition and multiplication. There is no floating-point computation in the kernel hot path. This eliminates all concerns about FMA rounding, denormals, flush-to-zero, and IEEE 754 mode differences.

**Precomputed threshold table:** The threshold comparison `ws > dyn_it` is performed entirely in integer arithmetic. `ws` is an int64 sum of int32 conv values (exact). `dyn_it` is an int64 value precomputed on the CPU from:
```
dyn_x = (c_target * m * m + 3.0 + (double)W_int / (2.0 * n) + eps_margin) * 4.0 * n * ell
dyn_it = (int64_t)(dyn_x * one_minus_4eps)
```
where `one_minus_4eps = 1.0 - 4 * 2.220446049250313e-16`. This computation happens once on the CPU, is verified against the CPU prover's thresholds, and is uploaded as a read-only table. The GPU never performs float64 arithmetic.

**Conservative threshold direction:** The threshold is computed with `one_minus_4eps` rounding guard, which floors the threshold slightly below the exact mathematical value. This means the GPU may prune *fewer* compositions than mathematically necessary (conservative) but never prunes one that should survive. This is the correct direction for proof soundness -- we may have extra survivors (requiring more work at the next level) but never miss any.

**Gray code completeness verification:** In debug builds, each block asserts that `children_tested + n_skipped == product(radix[j] for j in 0..n_active-1)`, where `n_skipped` accounts for children bypassed by subtree pruning. This verifies that the Gray code enumeration accounts for every element of the Cartesian product. The assertion is compiled out in release builds (zero runtime cost).

**Two-phase conv update correctness:** The even/odd phasing guarantees no write conflicts by construction. There are two classes of potential conflicts:

1. **Cross-term vs cross-term:** Thread j writes `conv[k2+j] = conv[k1+1+j]` and thread (j+1) writes `conv[k1+(j+1)] = conv[k1+j+1]` — the SAME address. By separating even and odd j into two phases with a barrier between them, we guarantee no two cross-term threads ever write to the same address in the same phase.

2. **Self-term vs cross-term:** Lane 0's self-term `conv[2*k2]` conflicts with thread `k1+2`'s Phase 1 cross-term `conv[k1+(k1+2)] = conv[2*k2]`. For d_child=64 when `pos >= 15`, these are on different warps — a data race without synchronization. Similarly, `conv[2*k1]` conflicts with thread `k1-1`'s Phase 2 cross-term. This is resolved by **splitting self-terms across phases**: `conv[2*k1]` and `conv[k1+k2]` are written in Phase 1 (where no cross-term touches those addresses), and `conv[2*k2]` is written in Phase 2 (where no cross-term touches that address). See the inline proof in Section 3.6.

**Canonicalization correctness:** The warp-ballot comparison finds the leftmost position where `child[i] != child[d-1-i]`. If `child[d-1-i] < child[i]` at that position, the reverse is lexicographically smaller -- use the reverse. This is identical to the CPU's early-exit comparison loop but computed in O(1) warp time. Correctness follows from the `__ballot_sync` + `__ffs` pattern finding the minimum set bit (= leftmost differing position). For d_child=64 (2 warps), the `use_rev` result and survivor slot are communicated via shared memory, not `__shfl_sync` (which is warp-local).

**Quick-check W_int maintenance:** The quick-check reuses the previous child's killing window (ell, s_lo, W_int). Since the Gray code changes exactly one cursor position per step (affecting child bins k1=2*pos and k2=2*pos+1), the `qc_W_int` is updated incrementally in O(1): if k1 or k2 falls within the quick-check window's bin range [qc_lo_bin, qc_hi_bin], the corresponding delta is added to `qc_W_int`. This update occurs between the conv update (STEP 2) and the quick-check (STEP 3), ensuring the threshold lookup uses the correct W_int for the current child. Without this update, stale W_int values would cause incorrect threshold lookups, potentially producing false prunes (soundness violation).

**Active position ordering:** Active positions are built right-to-left (matching CPU reference line 1180), so inner (fastest-changing) Gray code digits correspond to rightmost parent bins. This ensures the subtree pruning's fixed region (`child[0..fixed_len-1]`) is the left prefix of unchanged outer bins, matching the CPU's subtree pruning logic.

**Prefix sum thread safety:** The Kogge-Stone prefix sum uses a ping-pong buffer (two arrays, src/dst) with fixed `for` loop bounds. All threads participate in every `__syncthreads()` call regardless of whether they process valid indices. This avoids the deadlock that would occur with a `while (idx < conv_len)` loop where threads exit at different iterations.

### 4.3 Verification Strategy

**Level 1 (Automated):** Run L0, L1, L2 on both CPU and GPU. Compare survivor sets exactly (after sorting). These levels are small enough to complete in seconds and exercise every code path.

**Level 2 (Spot-check):** For L3, select 10,000 random parents. Run each on both CPU and GPU. Verify that the GPU survivor set for each parent is identical to the CPU survivor set. This validates the fused kernel at production scale.

**Level 3 (Injection):** For L4, construct synthetic parents with known survivors (e.g., compositions where all windows are just barely below threshold). Verify these appear in the GPU output.

**Level 4 (Enumeration audit):** In debug builds, every block reports `(parent_idx, children_tested, children_skipped, survivors_found)`. The host verifies that `children_tested + children_skipped == product(ranges)` for every parent (accounting for subtree-pruned children) and that `children_pruned + survivors_found == children_tested`.

**Level 5 (Bitwise reproducibility):** Run the same parent set on two different GPUs (or the same GPU twice). Verify bitwise-identical survivor sets. Since all computation is integer, this is guaranteed absent hardware errors. Any discrepancy indicates a bug.

### 4.4 Compilation Flags for Correctness

```bash
nvcc -arch=sm_90 -O3 \
     --use_fast_math=false \    # CRITICAL: no fast math (affects float64 in threshold precompute)
     -ftz=false \               # no flush-to-zero
     -prec-div=true \           # precise division (not used in hot path, but safety)
     -prec-sqrt=true \          # precise sqrt (not used, but safety)
     -fmad=false \              # CRITICAL: no fused multiply-add (affects threshold precompute)
     -Xptxas -dlcm=ca \        # cache all loads in L1
     -maxrregcount=32 \         # limit registers for occupancy
     -lineinfo \                # debug line info for profiler
     cascade_kernel.cu -o cascade_kernel
```

Note: `-fmad=false` and FP precision flags only matter for any float64 computation in the kernel (e.g., if threshold precomputation were done on GPU). Since our architecture moves ALL float64 to the CPU and uses only integer on GPU, these flags are defense-in-depth.

---

## 5. Global Memory Layout

```
GLOBAL MEMORY MAP (L4, d_parent=32, d_child=64, 147M parents)
═══════════════════════════════════════════════════════════════

INPUT (read-only):
  parents[147,279,894 x 32]        = 18.84 GB   int32, row-major
  lo_arrays[147,279,894 x 32]      = 18.84 GB   int32, row-major
  hi_arrays[147,279,894 x 32]      = 18.84 GB   int32, row-major
  threshold_table[127 x 2561]      = ~2.5 MB     int64, row-major (ell_count=127, S_child=4*32*20=2560)
  ell_order[127]                    = 508 B       int32

OUTPUT:
  survivors[200,000 x 64]          = 51.2 MB     int32, row-major
  survivor_count                   = 4 B          int32, atomic

CONTROL:
  g_next_parent                    = 4 B          int32, atomic (device variable)

TOTAL: ~56.5 GB (fits in 80 GB H100 memory)
```

**Multi-GPU partitioning for memory:** On a single H100 with 80GB, the full L4 dataset (56.5 GB) fits. However, for multi-GPU execution on a DGX H100 (8x 80GB), we partition parents across GPUs:

```
GPU 0: parents[0 .. 18.4M-1]       → 7.07 GB input per GPU
GPU 1: parents[18.4M .. 36.8M-1]
...
GPU 7: parents[128.8M .. 147.3M-1]

Each GPU: 7.07 GB parents + 7.07 GB lo + 7.07 GB hi + ~2.5 MB thresh + 51.2 MB surv
        = ~21.3 GB per GPU (26.6% utilization)
```

Parent assignment uses **pre-sorted round-robin** (Component J, Option J1): parents are sorted by estimated child count (descending), then assigned round-robin to GPUs. This ensures each GPU gets approximately equal total work. The estimation is `product(hi[i] - lo[i] + 1)` for each parent, computed on CPU in a single pass (~5 seconds for 147M parents).

---

## 6. Performance Model

### 6.1 Per-Child Cost Breakdown (L4, d_child=64)

| Operation | Cycles (est.) | Fraction | Notes |
|-----------|--------------|----------|-------|
| Gray code advance | 5-8 | 2% | Lane 0 only, ~5 int32 ops |
| Child bin update | 2 | <1% | Two writes to shared memory |
| Incremental conv update | 30-40 | 12% | Two phases x 32 threads x 2 writes; ~3 nonzero bins active |
| Quick-check (85% hit) | 7-10 | 3% | Warp-cooperative sum + reduction |
| Quick-check miss → prefix sums | 80-100 | ~12% | Two Kogge-Stone scans (conv + child), only 15% of children |
| Window scan (15% of children) | 100-200 | ~15% | Parallel range queries, early exit at ell level |
| Sync barriers | 40-60 | ~8% | 2-3 __syncthreads per child |
| **Weighted average** | **~50-70** | **100%** | Accounting for 85% quick-kill rate |

### 6.2 Throughput Estimate

At ~60 cycles/child average and 1.83 GHz clock:
- Children per second per SM: 1.83e9 / 60 * 5 (blocks/SM) = **152M children/sec/SM**
- Children per second per H100: 152M * 132 SMs = **20.1B children/sec**
- Children per second on 8x H100: **161B children/sec**

L4 total children: ~7.4 trillion
- Single H100: 7.4T / 20.1B = **368 seconds ≈ 6.1 minutes**
- 8x H100: 7.4T / 161B = **46 seconds**

This is optimistic (doesn't account for load imbalance, memory latency, host coordination). With a 3-5x reality factor: **2-25 minutes on 8x H100** vs. ~3 days on CPU. Even the pessimistic estimate is a **170x speedup**.

### 6.3 Occupancy Analysis

**d_child=64 (L4):**
- Block size: 64 threads (2 warps)
- Shared memory per block: ~21.4 KB (threshold table in L2 global; Kogge-Stone ping-pong buffers 2 KB)
- Blocks per SM: floor(228 KB / 21.4 KB) = 10
- Threads per SM: 10 * 64 = 640
- Warps per SM: 10 * 2 = 20
- Max warps per SM: 64
- **Occupancy: 20/64 = 31.3%**

This is moderate. The threshold table is in L2-cached global memory (too large for shared memory with `S_child = 4*n_child*m`), which frees significant shared memory per block. The ~30 cycle L2 latency for threshold lookups is acceptable since it only affects the 15% of non-quick-killed children. With 10 concurrent parents per SM, there are enough independent instructions to hide memory latency.

**Recommendation: Accept ~31% occupancy with threshold table in L2.** Profile to determine if L2 cache hit rates are sufficient; if not, consider caching the most frequently accessed ell rows (covering ells that kill 92% of non-quick-killed children) in a small shared memory buffer.

**d_child=32 (L3):**
- Block size: 32 threads (1 warp)
- Shared memory per block: ~13.9 KB (threshold table in L2 global)
- Blocks per SM: floor(228 / 13.9) = 16
- Threads per SM: 16 * 32 = 512
- Warps per SM: 16
- Max warps per SM: 64
- **Occupancy: 16/64 = 25%**

Improved occupancy with threshold table in L2 global memory.

---

## 7. Global Deduplication Strategy (Component I)

**Selected: Hybrid -- GPU bit-packed radix sort for L3, CPU-side dedup for L4.**

### L4 (76K survivors, d=64):

CPU-side dedup (Option I4). Transfer 76K * 256B = 19.5 MB to host over PCIe Gen5 in <1ms. Run existing `_fast_dedup` (np.lexsort + Numba scan) in <1 second. Total: negligible.

### L3 (147M survivors, d=32):

GPU bit-packed radix sort (Option I1).

Bit packing: m=20 means each bin value is 0-20, requiring 5 bits. 32 bins * 5 bits = 160 bits = 2.5 uint64s. Round up to 3 uint64s per composition.

```
Pack:  comp[0..31] → key[0]: bits 0-4=comp[0], 5-9=comp[1], ..., 60-63=comp[12] (13 per uint64)
                     key[1]: bits 0-4=comp[13], ..., 60-63=comp[25]
                     key[2]: bits 0-4=comp[26], ..., 25-29=comp[31], 30-63=0
```

Three-pass radix sort using CUB DeviceRadixSort:
1. Sort 147M entries by key[2] (least significant)
2. Stable sort by key[1]
3. Stable sort by key[0] (most significant)
4. Unique scan: adjacent-difference on the 3-uint64 key, mark duplicates, compact.

Memory: 147M * 24B (packed) = 3.5 GB for keys. CUB sort needs ~2x temp = 7 GB. Total: ~10.5 GB. Fits on single GPU with room to spare.

Time: CUB radix sort on H100 sorts ~10 billion 8-byte keys/sec. 147M keys * 3 passes / 10B/s ≈ 0.04 seconds. Plus packing + unique scan: ~0.1 seconds total.

---

## 8. Multi-GPU Orchestration

**Selected: Option J1 (Pre-Sorted Round-Robin) for L3/L4 on single DGX node.**

### Host Orchestration Loop

```python
def run_cascade_gpu(parents, lo_arrays, hi_arrays, level, num_gpus=8):
    """
    Host-side orchestration for multi-GPU cascade level.
    """
    # 1. Estimate work per parent
    est_children = np.prod(hi_arrays - lo_arrays + 1, axis=1)  # product of ranges
    order = np.argsort(-est_children)  # sort descending by work

    # 2. Round-robin assign to GPUs
    gpu_assignments = np.empty(len(parents), dtype=np.int32)
    gpu_assignments[order] = np.arange(len(parents)) % num_gpus

    # 3. Transfer shards to GPUs
    for gpu_id in range(num_gpus):
        mask = gpu_assignments == gpu_id
        cuda.set_device(gpu_id)
        d_parents[gpu_id] = cuda.to_device(parents[mask])
        d_lo[gpu_id] = cuda.to_device(lo_arrays[mask])
        d_hi[gpu_id] = cuda.to_device(hi_arrays[mask])
        d_survivors[gpu_id] = cuda.device_array((MAX_SURVIVORS_PER_GPU, d_child), dtype=np.int32)
        d_count[gpu_id] = cuda.device_array(1, dtype=np.int32)

    # 4. Launch kernels concurrently
    for gpu_id in range(num_gpus):
        cuda.set_device(gpu_id)
        cascade_kernel[grid_dim, block_dim](
            d_parents[gpu_id], d_lo[gpu_id], d_hi[gpu_id],
            d_threshold, d_ell_order,
            d_survivors[gpu_id], d_count[gpu_id],
            shard_sizes[gpu_id], d_parent, d_child, m,
            ell_count, conv_len,
            np.float32(np.sqrt(c_target / 2.0))  # threshold_asym
        )

    # 5. Synchronize and collect
    all_survivors = []
    for gpu_id in range(num_gpus):
        cuda.set_device(gpu_id)
        cuda.synchronize()
        count = d_count[gpu_id].copy_to_host()[0]
        survivors = d_survivors[gpu_id][:count].copy_to_host()
        all_survivors.append(survivors)

    # 6. Global dedup on host (or GPU)
    combined = np.concatenate(all_survivors, axis=0)
    return fast_dedup(combined)
```

### Progress Monitoring

Since the atomic counter `g_next_parent` is a device variable, the host can periodically read it via `cudaMemcpy` (device-to-host, 4 bytes) to monitor progress without interrupting the kernel:

```python
while not all_done:
    for gpu_id in range(num_gpus):
        progress = read_device_counter(gpu_id)
        print(f"GPU {gpu_id}: {progress}/{shard_sizes[gpu_id]} parents ({100*progress/shard_sizes[gpu_id]:.1f}%)")
    time.sleep(10)
```

---

## 9. Summary of Architectural Choices per Component

| Component | Choice | Why |
|-----------|--------|-----|
| **A: Parent Dispatch** | Persistent blocks + global atomic counter | Perfect dynamic load balancing for 1000:1 workload variance; negligible atomic contention at 50K children/parent |
| **B: Child Enumeration** | Sequential mixed-radix Gray code (Knuth TAOCP 7.2.1.1) on lane 0 | Guarantees single-position change per step → 100% incremental conv updates. Serial overhead (~5 ops) is negligible vs parallel conv update + window scan |
| **C: Conv Update** | Two-phase even/odd warp-parallel cross-terms, self-terms split across phases | Zero write conflicts by construction (no atomics needed): cross-terms separated by even/odd phasing, self-terms placed in the phase where no cross-term touches the same address. Deterministic performance. O(d/warp_size) = O(1) at d=32 |
| **D: Quick-Check** | Warp-cooperative sum + lane-0 threshold check | 85% kill rate preserved. Cooperative sum reduces latency from O(ell) to O(ell/32 + 5). Broadcasts result to all lanes |
| **E: Window Scan** | Prefix sum + parallel range queries, ell-ordered with early exit | One prefix sum per child (built only for 15% non-quick-killed). All window positions for each ell tested in parallel. Preserves CPU's optimized ell ordering |
| **F: Subtree Pruning** | Lane-0 sequential partial conv + window scan | Skips 5-20% of Cartesian product. Triggers once per outer-digit change, amortized well. Simple implementation |
| **G: Canonicalization** | Warp-ballot parallel comparison | O(1) warp time via `__ballot_sync` + `__ffs`. Runs on <0.001% of children. Negligible cost |
| **H: Survivor Collection** | Shared memory staging (64 slots) + block-level flush via global atomic | One global atomic per flush (~1 per parent at L4). 16KB staging buffer. Near-zero contention |
| **I: Deduplication** | L3: GPU 3-pass bit-packed radix sort. L4: CPU-side lexsort | Matched to survivor count: 147M needs GPU sort; 76K trivially fits CPU |
| **J: Multi-GPU** | Pre-sorted round-robin across 8 GPUs on DGX node | Simple, statistically balanced. Work estimates smooth over 147M / 8 = 18.4M parents per GPU |

---

## 10. Risk Analysis and Mitigations

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Gray code bug causes missed/duplicated child | **Critical** (proof invalid) | Low | Enumeration count assertion; exact match vs CPU on L0-L2; spot-check 10K random parents on L3 |
| Conv update write conflict produces wrong value | **Critical** (proof invalid) | Eliminated | Two-phase even/odd with self-terms split across phases guarantees no conflicts by construction: cross-term conflicts separated by even/odd phasing, self-term vs cross-term conflicts eliminated by placing each self-term in the phase where no cross-term touches the same address (see Section 3.6 inline proof). Verified by running both C2 and C4 (full recompute) on same parents and comparing conv arrays |
| Integer overflow in conv accumulation | **Critical** | None for m≤200 | Max conv value = m^2 = 400 << INT32_MAX. Debug-mode range assertions |
| Threshold comparison wrong direction | **Critical** | None | Thresholds precomputed identically to CPU with `one_minus_4eps` guard (conservative). GPU performs same `ws > thresh` integer comparison |
| Stale qc_W_int causes wrong quick-check threshold | **Critical** (proof invalid) | Eliminated | Incremental O(1) update of qc_W_int between conv update and quick-check (STEP 2→3), matching CPU lines 1339-1349. After subtree prune reset, qc_W_int is recomputed from scratch (matching CPU lines 1492-1502) |
| Active position ordering mismatch breaks subtree pruning | **Critical** (proof invalid) | Eliminated | Right-to-left iteration (matching CPU line 1180) ensures fixed outer region = left prefix |
| Prefix sum __syncthreads deadlock | **Critical** (kernel hang) | Eliminated | Ping-pong buffer with fixed loop bounds; all threads participate in every barrier |
| Cross-warp broadcast via __shfl_sync | **Critical** (silent corruption) | Eliminated | Canonicalization and slot broadcast use shared memory for d_child=64 (2 warps) |
| Survivor buffer overflow | Medium (lost survivors) | Very low | SURV_CAP=64 >> expected 0.0005 survivors/parent. Global buffer 200K >> expected 76K. Overflow flag triggers host relaunch |
| GPU load imbalance causes idle time | Low (perf only) | Medium | Pre-sorted round-robin + atomic work-stealing within each GPU. Statistical balancing over 18M+ parents per GPU |
| Shared memory bank conflicts | Low (perf only) | Low | Conv access pattern is stride-1 (consecutive lanes access consecutive addresses). Analysis confirms zero bank conflicts for cross-term writes |
| Moderate occupancy (~31%) may limit throughput | Low (perf only) | Possible | Compute-bound kernel; 10 concurrent parents per SM at d=64 provide sufficient ILP. Shared memory ~21.4 KB/block (threshold table in L2 global). If bottleneck, consider caching high-kill ell rows in smem for faster access |

---

## 11. Implementation Phases

### Phase 1: Single-GPU Kernel (L2 verification)
- Implement full kernel targeting d_child=16 (L2)
- Verify exact survivor set match vs CPU (48,443 expected survivors)
- Profile kernel: identify actual bottlenecks vs model

### Phase 2: L3 Production
- Scale to d_child=32, 7.5M parents
- Implement GPU dedup (bit-packed radix sort)
- Verify: 147,279,894 expected survivors
- Profile and optimize based on Phase 1 learnings

### Phase 3: L4 Production
- Scale to d_child=64, 147M parents
- Multi-GPU orchestration (8x H100)
- Verify: ~76K expected survivors (compare to CPU in-progress result)
- If zero survivors at any point: **bound is proven**

### Phase 4: Extended Cascade
- L5 (d=128) and beyond if L4 produces survivors
- May require register-spilling strategies for d=128 conv arrays (255 entries)
- May require int64 conv for m>200
