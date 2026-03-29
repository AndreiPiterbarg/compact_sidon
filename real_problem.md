# Real Computational Bottlenecks in the Gray Code Kernel

## Why Ideas 1-3 Failed

Ideas 1-3 targeted the **O(d_child) prefix_c recomputation** and the **O(ell) quick-check summation** -- operations that account for <5% of total per-child cost. Numba compiles these simple sequential loops into tight vectorized machine code that runs in ~50ns. The "O(1) replacements" introduced branchy control flow (range checks, prefix_c lookups, conditional subtractions) that defeated branch prediction and instruction pipelining, producing net regressions.

The real bottlenecks are elsewhere.

## Measured Cost Breakdown (d_child=32, L3 level)

| Component | Share | Notes |
|-----------|-------|-------|
| **Output buffer writes** | 44% | 32 int32 stores to a multi-MB buffer that far exceeds L2 cache. Every write is a cache miss. |
| **Cache interference** | 30% | The cross-term update (14 multiply-adds at d=32) should take 50ns but costs 1.9us -- a 40x slowdown from L1/L2 thrashing caused by the output buffer polluting the cache. |
| **Full window scan** | 26% | 63 ell values x ~32 windows each = 2016 evaluations per survivor. Each eval: sliding-window update + prefix_c lookup + float64 threshold + int64 cast + compare. |

## The Five Real Problems

### 1. Subtree Pruning Has a 0% Hit Rate (Pure Overhead)

Across all tested parents at L1 through L4:
- Subtree checks fired: thousands per level
- **Subtree successes: ZERO at every level**
- Children skipped: ZERO

The J_MIN=7 check fires whenever the 7th Gray code digit advances, computing a partial autoconvolution of the fixed left prefix (O(fixed_len^2)) and scanning windows. It **never succeeds** because the conservative W_int_max estimate (using full parent mass for unfixed bins) inflates the threshold beyond what the partial convolution can exceed.

**Impact:** Every subtree check is wasted work -- O(fixed_len^2 + ell_count * windows) ops for nothing. At L4 with hundreds of checks per parent, this is significant overhead.

### 2. 48-63% of Children Survive All Pruning (The Survival Rate Problem)

| Level | d_child | Survival Rate | x_cap |
|-------|---------|---------------|-------|
| L1 | 8 | 62.6% | 8 |
| L2 | 16 | 59.9% | 5 |
| L3 | 32 | 57.7% | 4 |
| L4 | 64 | 63.1% | 2 |

The survival rate is stuck at 57-63% across all levels. This is because the correction term `correction(m, n_half) = n_half * (2/m + 1/m^2)` grows linearly with dimension. At L4: correction = 32 * (0.1 + 0.0025) = 6.56, so the effective pruning target is c_target + 6.56 = 7.96, nearly 6x the actual c_target of 1.4. The window sums peak at ~83% of this inflated threshold -- not enough to prune.

Every surviving child must scan **all** ell values and **all** windows before being declared a survivor. At d=32 that's 2,016 window evaluations; at d=64 it's 8,128. This O(d^2) work per survivor is the dominant cost.

### 3. Output Buffer Cache Pressure (44% of Per-Survivor Cost)

At d=32 with millions of survivors, the output buffer is ~150MB+. Writing each survivor (32 int32 values = 128 bytes) evicts L1/L2 cache lines that hold the hot working set (raw_conv, child, prefix_c arrays). The next child's cross-term update then suffers ~40x slowdown from cache misses reloading these arrays.

The cross-term update itself is only 14 multiply-adds (~50ns in isolation), but in context costs 1.9us due to this cache pollution.

### 4. Quick-Check Only Catches 33-49% of Prunable Children

| Level | Quick-check hit rate | Full-scan needed |
|-------|---------------------|-----------------|
| L1 | 32.7% | 67.3% |
| L2 | 35.9% | 64.1% |
| L3 | 38.7% | 61.3% |
| L4 | 33.3% | 66.7% |

The quick-check caches **one** (ell, s_lo) pair from the previous prune. When the killing window changes between adjacent Gray code children (which happens often since pos can be anywhere in the array), the cached window fails and the full O(d^2) scan is needed. The 33-39% hit rate means ~2/3 of prunable children pay full scan cost.

### 5. Ell Scan Ordering Misses 40% of Kills

The profile-guided ell ordering starts at hc+1=17 for d=32, but empirical kill distribution peaks at ell=17-21 (accounting for 40% of kills) which are 4th-8th in the scan order. When the first 3 ell values fail, the full scan continues through all 63 values. Pruned children scan an average of 11.3 ell values and 503 windows instead of the theoretical minimum of ~3 ell values and ~100 windows.

## What Would Actually Help

### A. Disable Subtree Pruning (Immediate, Free Speedup)

Remove or gate the J_MIN subtree check -- it has 0% success rate and adds pure overhead. Every cycle spent computing partial autoconvolutions for subtree checks is wasted. This alone should give a measurable speedup at L3/L4 by eliminating thousands of O(fixed_len^2) computations per parent.

### B. Output Buffer Staging (Cache-Friendly Writes)

Replace the direct-to-output-buffer writes with a small L1-resident staging buffer (~64KB = 500 rows x 32 x 4B). Flush to the main buffer when full. This keeps the hot working set (raw_conv, child, prefix_c -- ~2KB total at d=32) in L1 cache, eliminating the 40x cache-miss penalty on cross-term updates.

### C. Threshold Lookup Table (Eliminate Per-Window Float Math)

The per-window threshold computation:
```
dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * float64(W_int)
dyn_it = int64(dyn_x * one_minus_4eps)
```
involves float64 multiply-add and float-to-int conversion **per window**. Since W_int is an integer in [0, m], precompute `threshold[ell_idx][W_int]` as an int64 lookup table (63 x 21 = 1,323 entries, ~10KB -- fits in L1). Replace the per-window computation with a single `threshold_table[ell_idx * (m+1) + W_int]` load.

### D. Multi-Window Quick-Check Cache

Instead of caching one (ell, s_lo) pair, cache the top 3-5 killing windows ordered by kill rate. Try all cached windows before falling back to the full scan. Since adjacent Gray code children share most bins, a window that killed a nearby child often works. This could raise the quick-check hit rate from 33% to 60-70%, cutting full-scan invocations in half.

### E. Increase m (Fundamental Fix)

At m=20, the correction term at L4 is 6.56, inflating the pruning target to 7.96. At m=50, correction = 32 * (0.04 + 0.0004) = 1.293, giving a pruning target of 2.693 -- much tighter. The tradeoff is larger Cartesian products at lower levels (O(m^d) at L0), but L0-L2 are already cheap. The real computation is at L3-L4 where tighter bounds would dramatically increase the prune rate.

### F. Specialized L4 Kernel (x_cap=2 Structure)

At L4 with x_cap=2, each parent bin splits into at most 3 choices (0, 1, or 2 for the left child). The compositions are extremely sparse (~73% zero bins). A kernel specialized for this ternary structure could:
- Enumerate children as ternary codewords rather than general compositions
- Use bitwise operations for the sparse cross-term updates
- Exploit the fact that at most 17 of 64 bins are nonzero

## Priority Ranking

| Priority | Fix | Expected Impact | Effort |
|----------|-----|----------------|--------|
| 1 | Disable subtree pruning | 3-8% speedup (removes pure overhead) | Trivial |
| 2 | Output buffer staging | 15-30% speedup (fixes cache pressure) | Low |
| 3 | Threshold lookup table | 10-15% speedup (eliminates per-window float math) | Low |
| 4 | Multi-window quick-check cache | 15-25% speedup (reduces full scans) | Medium |
| 5 | Increase m to 50 | Potentially orders of magnitude at L4 (tighter bounds) | Requires re-running cascade |
| 6 | Specialized L4 kernel | 2-4x speedup at L4 | High |
