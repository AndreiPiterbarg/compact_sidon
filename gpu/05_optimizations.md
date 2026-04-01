# 5. Critical Optimizations

## 5.1 INT32 vs FP32 Arithmetic

**Analysis of INT32 throughput on H100:**
- H100 has 64 INT32 ALUs per SM vs 128 FP32 ALUs
- Our hot path (incremental conv update + window sum) is all integer multiply-add
- Using FP32 would double peak throughput

**Safety analysis for FP32:**
- Max conv value at m=20: `m^2 = 400` (self-term) or `2 × m × m = 800` (cross-term)
- Max prefix_conv value: `d_child × m^2 = 64 × 400 = 25,600`
- FP32 mantissa: 24 bits = exact to 16,777,216
- **All values fit exactly in FP32 for m ≤ 200** (max intermediate: `2 × 200 × 200 × 64 = 5,120,000 < 16M`)

**HOWEVER:** The threshold comparison uses int64: `ws (int64) > dyn_it (int64)`. The window sum can be up to `63 × 400 = 25,200` which fits int32. But `dyn_x = c_target × m² × ell / (4n) + 1 + eps + 2 × W_int` is float64.

**Recommendation:**
- Compute autoconvolution cross-terms in **int32** (they must be exact for proof correctness)
- Compute window sums in **int32** (max 25,200 at m=20, d=64)
- Threshold comparison in **int64** (precomputed table)
- The conv update is mostly addition, not multiplication — INT32 throughput for addition is same as FP32

## 5.2 Shared Memory Bank Conflicts

Shared memory on H100 has 32 banks, 4-byte stride. Accessing `conv_smem[k1 + lane]` where `k1` is the same for all lanes creates a sequential access pattern with **zero bank conflicts** (consecutive addresses map to consecutive banks).

However, `conv_smem[k2 + lane]` where `k2 = k1 + 1` also has zero bank conflicts. The two writes per thread (one to `conv[k1+j]`, one to `conv[k2+j]`) can be interleaved without issue.

**Potential conflict:** When building prefix sum via warp shuffle or shared memory reduction, ensure stride patterns avoid bank conflicts. Use `__shfl_up_sync` instead of shared memory for warp-level prefix sums.

## 5.3 Register Pressure

Per-thread register usage estimate (d_child=64, 64 threads/block):
- `child` values: loaded from shared memory, not kept in registers
- `conv` values: loaded from shared memory per-access
- Loop counters, temporaries: ~10 registers
- Gray code state: ~5 registers (lane 0 only, others don't need it)
- Quick-check state: 3 registers (lane 0 only)

**Estimated: ~20 registers/thread.** H100 has 65,536 registers/SM. At 64 threads/block × 32 blocks/SM = 2048 threads × 20 regs = 40,960 registers — fits with headroom. **No register pressure issue.**

## 5.4 Warp Divergence

**Critical concern:** Gray code advance, quick-check, and pruning decisions are evaluated by lane 0 and broadcast. Only lane 0 does the sequential Gray code state machine. All other lanes idle during:
- Gray code advance (~5 ops): **Negligible**
- Quick-check evaluation (~10-30 ops): **Worth optimizing** — can we distribute?

**Quick-check parallelization:** The quick-check computes `ws_qc = sum(raw_conv[qc_s .. qc_s + n_cv_qc - 1])`. With shared memory access, multiple lanes can cooperatively sum this range:
```cuda
int partial = 0;
for (int k = qc_s + lane; k < qc_s + n_cv_qc; k += WARP_SIZE)
    partial += conv_smem[k];
int ws_qc = warp_reduce_sum(partial);  // __shfl_down_sync reduction
```
This reduces quick-check from O(ell) sequential to O(ell/32) — significant when ell ≈ 16-32.

## 5.5 Prefix Sum Strategy

**One prefix sum per child is expensive if most children are quick-killed.** CPU avoids prefix sum entirely for quick-killed children.

**GPU strategy:**
1. Lane 0 evaluates quick-check (with warp-cooperative sum)
2. If quick-killed (~85%): broadcast skip, advance to next child
3. If not quick-killed: build prefix sum cooperatively, then parallel window scan

This preserves the CPU's "skip prefix_c for quick-killed children" optimization.

## 5.6 Memory Access Coalescing

**Parent array reads:** Each block reads one parent (32 × 4B = 128B at d_parent=32). With warp-coalesced reads, this is one 128B transaction — perfectly coalesced.

**Survivor writes:** The staging buffer flush writes `surv_count × d_child × 4B` to global memory. With cooperative writes (`threadIdx.x` stride), this is coalesced.

**Threshold table:** Read-only, accessed by `[ell_idx × (m+1) + W_int]`. Different lanes may access different W_int values (different windows), causing **non-coalesced** access. Mitigation: place in shared memory (loaded once at block start, then fast random access).

## 5.7 Sparse Cross-Term Optimization (from CPU)

The CPU uses a nonzero list for d_child ≥ 32 to skip zero bins in cross-term updates. On GPU:

**Option A: Predicated execution** — Each thread checks `if (child_smem[lane] != 0)` before writing. No branch divergence (predication), and at L4 most bins are zero (m=20 spread across 64 bins → ~3 nonzero bins on average).

This is actually better than a nonzero list on GPU because:
- No extra shared memory for the list
- No maintenance overhead when bins change
- Predicated execution has zero divergence cost
- With ~3/64 bins nonzero, 61/64 threads skip the write — but they still execute the predicate check (1 cycle)

**Verdict: Skip the nonzero list. Use predicated execution. Much simpler, equally fast on GPU.**

## 5.8 Canonicalization on GPU

Survivors need canonicalization: `min(child, reverse(child))` lexicographically. On GPU:

```cuda
// Warp-cooperative canonicalization (d_child=32)
int i = lane;
int j = d_child - 1 - lane;
int fwd = child_smem[i];
int rev = child_smem[j];

// Parallel comparison: find first differing position
int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
// Find leftmost nonzero comparison using __ffs on ballot
uint32_t lt_mask = __ballot_sync(0xFFFFFFFF, cmp < 0);
uint32_t gt_mask = __ballot_sync(0xFFFFFFFF, cmp > 0);
int first_lt = lt_mask ? __ffs(lt_mask) : 33;
int first_gt = gt_mask ? __ffs(gt_mask) : 33;
bool use_rev = (first_lt < first_gt);

// Write canonical form to staging buffer
if (use_rev) {
    surv_buf[slot * d_child + lane] = child_smem[j];
} else {
    surv_buf[slot * d_child + lane] = child_smem[i];
}
```

This is **O(1) parallel** using warp ballots — much faster than CPU's O(d/2) sequential scan.
