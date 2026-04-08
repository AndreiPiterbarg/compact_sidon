# 3. H100 Hardware Specifications & Constraints

## H100 SXM 80GB Key Specs

| Feature | Value | Implication |
|---------|-------|-------------|
| SMs | 132 | Min 132 blocks for full occupancy; target 132×32 = 4224+ concurrent warps |
| FP32 cores/SM | 128 | 16,896 total FP32 cores |
| INT32 cores/SM | 64 | **Half FP32 throughput** — critical since our hot path is int32 |
| Shared memory/SM | 228 KB (configurable up to 228KB) | Enough for ~228 concurrent parents' working sets per SM |
| L2 cache | 50 MB | Fits ~1.5M parent compositions at d=32 (4B×32 = 128B each) |
| HBM3 bandwidth | 3.35 TB/s | Not the bottleneck — we're compute-bound |
| Warp size | 32 | d=32 maps perfectly; d=64 needs 2 warps |
| Max threads/SM | 2048 | 64 warps/SM or 32 blocks of 64 threads |
| Max blocks/SM | 32 | |
| Clock (boost) | 1.83 GHz | |
| TMA units | 1 per SM | Async bulk copy for parent loading |
| Thread Block Clusters | Up to 16 blocks | Hopper-specific; distributed shared memory |

## INT32 Throughput Concern

**CRITICAL:** H100 has **half** the INT32 throughput compared to FP32. Our autoconvolution uses int32 multiply-add exclusively. Options:
1. **Stay int32** — simpler, correct, but only ~8.4 TIPS peak
2. **Use FP32 for convolution** — double throughput, but risk precision loss. For m=20: max conv value = 400, max cross-term = 800. FP32 has 24-bit mantissa (16M precision) — safe for m ≤ ~4000. **Recommended: FP32 arithmetic with int32 storage for correctness verification.**
3. **Mixed approach** — compute in FP32, verify survivors in int64

## Memory Budget Per SM (228KB shared memory)

For d_child=64:
- `raw_conv[127]` = 508B (int32) or 508B (fp32)
- `child[64]` = 256B
- `parent[32]` = 128B
- `prefix_c[65]` = 520B (int64)
- `threshold_table[ell_count × (S_child+1)]` where `S_child = 4*n_child*m` — ~2.5 MB at d_child=64, m=20 (ell_count=127, S_child=2560; L2 cached, too large for shared memory)
- `cursor[32]` = 128B
- `gc_state` (gc_a, gc_dir, gc_focus) = ~300B
- `ell_order[63]` = 252B
- `staging buffer` = variable
- **Total per-parent working set ≈ 2.2KB** (excluding threshold table)

At 228KB shared memory / SM, we can fit **~103 concurrent parents per SM** if we exclude the threshold table. With threshold table in L2 cache or constant memory, this works.

For d_child=32:
- Working set ≈ 1.1KB per parent
- ~207 concurrent parents per SM

## Occupancy Targets

| d_child | Threads/parent | Parents/SM | Warps/SM | Occupancy |
|---------|---------------|------------|----------|-----------|
| 32 | 32 (1 warp) | 64 | 64 | 100% |
| 64 | 64 (2 warps) | 32 | 64 | 100% |
