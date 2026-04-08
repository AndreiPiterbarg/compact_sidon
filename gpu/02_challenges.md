# 2. Challenges of Converting This CPU Cascade Prover to CUDA

## Challenge 1: Fundamentally Serial Per-Parent Enumeration (The Core Problem)

The fused kernel (`_fused_generate_and_prune` / `_fused_generate_and_prune_gray`) processes one parent at a time, iterating through potentially millions of children per parent in a tight sequential loop. The Gray code / odometer advances are inherently sequential — each step's state depends on the previous step's cursor position, autoconvolution array, and quick-check state. You cannot parallelize within a single parent's child enumeration without losing the O(d) incremental update that makes the whole thing fast.

**The dilemma:** If you give each GPU thread one parent, you get parallelism across parents — but each thread runs a long, variable-length serial loop. If you try to parallelize across children of one parent, you lose the incremental convolution update and must do O(d²) full recomputes per child, negating the entire 10-18x speedup of the fused kernel.

## Challenge 2: Massive Work Imbalance Across Parents

Different parents produce wildly different numbers of children. A parent with bins `[10, 5, 3, 2]` has `11 × 6 × 4 × 3 = 792` children. A parent with bins `[1, 1, 1, 17]` has `2 × 2 × 2 × 18 = 144`. At L3→L4 (147M parents, d=32→64), the ratio between the lightest and heaviest parent can be **1000:1 or more**. GPU warps (32 threads) execute in lockstep — if one thread in a warp has 10M children and another has 100, the fast thread idles for 99.999% of the time. This is the **warp divergence / tail latency problem** and it's devastating for GPU utilization.

## Challenge 3: Per-Thread Memory Footprint vs. GPU Register/Shared Memory Limits

Each thread needs its own working state:

- `child[d_child]` — 64 × 4B = 256B at L4
- `raw_conv[conv_len]` — 127 × 4B = 508B at L4
- `prefix_c[d_child+1]` — 65 × 8B = 520B at L4
- `cursor[d_parent]`, `prev_child[d_child]`, `gc_a/gc_dir/gc_focus`, `stage_buf`, etc.

**Total per-thread: ~2-3 KB minimum at L4.** An NVIDIA GPU SM has 256KB of registers and 48-100KB of shared memory. At 2KB per thread, you can only fit ~128 threads per SM (4 warps), far below the 1024-2048 threads needed for full occupancy. Low occupancy means the GPU can't hide memory latency, killing throughput.

## Challenge 4: The Incremental Convolution Update Is Branch-Heavy and Sequential

The O(d) cross-term update loop:

```python
for j in range(k1):
    cj = child[j]
    if cj != 0:
        raw_conv[k1 + j] += 2 * delta1 * cj
        raw_conv[k2 + j] += 2 * delta2 * cj
```

This has **data-dependent branching** (`if cj != 0`), **scattered writes** to `raw_conv` at computed indices (`k1+j`, `k2+j`), and the **loop length depends on `pos`** (which position changed). GPUs hate all three of these things — branches cause warp divergence, scattered writes prevent coalescing, and variable loop lengths cause more divergence.

## Challenge 5: The Quick-Check Heuristic Is Inherently Sequential and Stateful

The quick-check (`qc_ell`, `qc_s`, `qc_W_int`) exploits temporal locality: "the window that killed child N probably kills child N+1." This only works because children are visited in a specific order (Gray code) where adjacent children differ minimally. On a GPU, if you parallelize across children, there is no "previous child" — each thread works independently, so you lose the quick-check entirely. Given that quick-check avoids the full O(d × ell_count × n_windows) scan for the majority of children, losing it is a **major performance regression**.

## Challenge 6: Subtree Pruning Requires Sequential Cursor State

The subtree pruning optimization (lines 846-971) detects that an entire subtree of the Cartesian product can be skipped, fast-forwarding multiple cursor positions at once. This depends on knowing the current cursor state and the partial autoconvolution of the "fixed" prefix. On a GPU, if you pre-enumerate children independently (to parallelize), you can't do subtree pruning — you'd have to enumerate every child individually, including millions that the CPU version skips entirely.

## Challenge 7: Gray Code State Machine Cannot Be Parallelized

The Knuth mixed-radix Gray code (TAOCP 7.2.1.1) maintains three coupled state arrays (`gc_a`, `gc_dir`, `gc_focus`). The focus-pointer chain is what enables O(1) next-position lookup, but it's inherently sequential — you can't compute the state at step N without running through steps 0..N-1. There's no closed-form "jump to step K" for the focus-pointer variant. This means you can't partition the enumeration across threads without either:

1. Abandoning Gray code entirely (losing the guaranteed single-position change per step)
2. Pre-computing the full Gray code sequence on CPU (defeats the purpose)

## Challenge 8: Dynamic Threshold Table Lookup Pattern

The threshold table `threshold_table[ell_idx * (S_child+1) + W_int]` is accessed with data-dependent indices (where `S_child = 4 * n_child * m`). At L4 with d_child=64 and m=20, `S_child = 4*32*20 = 2560`, so this table is 127 × 2561 int64 entries (~2.5MB). This is far too large for shared memory (228KB) and must reside in L2-cached global memory. The access pattern across threads in a warp is non-uniform (different threads have different `ell` and `W_int` values), causing **cache pressure and serialized accesses**.

## Challenge 9: The Deduplication Pipeline Doesn't Map to GPU

Survivors need canonicalization (`min(comp, reverse(comp))`) and global deduplication across all parents. The CPU uses `np.lexsort` + a Numba scan. On GPU:

- Canonicalization per-thread is straightforward
- Global deduplication of potentially billions of variable-length survivor rows requires a GPU sort (CUB/Thrust radix sort), but the key width is `d_child × 4 bytes = 256 bytes` at L4. **Sorting 256-byte keys is extremely expensive on GPU** — radix sort works best on 4-8 byte keys. You'd need a custom multi-pass sort or hash-based dedup.

## Challenge 10: Memory Capacity — Survivor Output is Unpredictable

The number of survivors per parent is unknowable in advance. The CPU uses a pre-allocated `out_buf` with a guess at max survivors, plus disk-shard spilling when RAM is exceeded. On GPU:

- You can't dynamically allocate per-thread output buffers
- Atomic counters for a global output buffer create contention
- You need a compact/stream-compaction pattern, but the output rate is highly variable (some parents produce 0 survivors, others produce thousands)
- GPU global memory is limited (24-80GB), vs. the CPU's ability to spill to disk transparently

## Challenge 11: The Cascade Structure Creates Pipeline Dependencies

Each level depends on the previous level's survivors. L0→L1→L2→L3→L4 is fundamentally sequential across levels. Within a level, parents are independent — this is where GPU parallelism could help — but the CPU already parallelizes across parents with `prange`. The GPU advantage only materializes if the per-parent work can be efficiently mapped to the GPU execution model, which (per challenges 1-7) is extremely difficult.

## Challenge 12: Numerical Precision Requirements for Correctness

This is a rigorous mathematical proof — any floating-point error that causes a false prune invalidates the entire result. The code uses careful `one_minus_4eps` rounding guards, `eps_margin`, and integer arithmetic specifically to avoid FP issues. CUDA's default is to use fused multiply-add (FMA) which produces different rounding than separate multiply + add. You'd need `--fmad=false` or explicit `__fmul_rn`/`__fadd_rn` intrinsics everywhere, and verify that the int64↔float64 conversion semantics match exactly. **A single bit flip in the threshold comparison can silently produce wrong results.**

## Challenge 13: Sparse Cross-Term Optimization Depends on Per-Thread Bookkeeping

The `use_sparse` path (d_child >= 32) maintains a nonzero list (`nz_list`, `nz_pos`) that's updated incrementally as child bins change. This avoids iterating over zero bins in the cross-term loop. On GPU, maintaining a per-thread sparse index adds register pressure and branches, and the sparsity pattern varies per thread, causing more warp divergence.

## The Fundamental Architectural Tension

The CPU code's speed comes from **sequential exploitation of structure**: incremental updates, quick-check temporal locality, subtree pruning, and Gray code ordering. These are all serial optimizations that trade independence for efficiency. A GPU needs the **opposite**: massive independent parallelism with uniform work per thread.

The most viable GPU architecture would likely be:

1. **Parallelize across parents** (one parent per thread-block or warp), accepting low occupancy
2. **Keep the serial fused kernel within each thread/block**, essentially running the Numba kernel as a CUDA device function
3. **Use dynamic parallelism or cooperative groups** for load balancing across parents with very different child counts
4. **Accept losing some CPU-specific optimizations** (like quick-check) that don't translate

But even this approach faces the **register pressure** (challenge 3) and **work imbalance** (challenge 2) problems, which are structural to the GPU hardware model.
