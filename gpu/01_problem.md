# 1. Problem Characterization

## What We're Computing

We are exhaustively proving that **no** nonneg function on [-1/4, 1/4] with unit integral can have autoconvolution peak below a target threshold $C_{1a}$. The computation reduces to:

1. **Enumerate** all integer compositions of $S = m$ into $d$ parts (mass distributions over $d$ bins)
2. **Compute autoconvolution** for each composition — a discrete self-convolution yielding $2d - 1$ values
3. **Test** whether any windowed sum of the autoconvolution exceeds a per-window dynamic threshold
4. **Collect survivors** (compositions that pass all windows) for refinement at the next cascade level
5. **Cascade**: survivors at dimension $d$ become parents at dimension $2d$, with each parent generating a Cartesian product of children

## Scale of the Problem

| Level | d_child | Parents | Avg Children/Parent | Total Children | Survivors |
|-------|---------|---------|--------------------:|---------------:|----------:|
| L0 | 4 | — | — | 1,771 | 345 |
| L1 | 8 | 345 | ~15K | ~5M | 48,443 |
| L2 | 16 | 48,443 | ~200K | ~10B | 7,499,382 |
| L3 | 32 | 7,499,382 | ~7K | ~50B | 147,279,894 |
| L4 | 64 | 147,279,894 | ~50K | **~7.4 trillion** | ~76K (in progress) |

**Key insight:** At L4, we process **7.4 trillion children** with a **survival rate of ~0.000001%**. This is a massive parallel filter operation — generate, test, discard 99.999999%, collect the survivors.

## Computational Bottleneck Analysis

Per child at L4 (d_child=64, conv_len=127):

| Operation | FLOPs/IOP | Memory | % of time (CPU profile) |
|-----------|-----------|--------|------------------------|
| Incremental conv update (Gray code) | ~128 int32 MADs | 127×4 = 508B conv + 64×4 = 256B child | 35% |
| Window scan (ell loop × sliding window) | ~500-2000 int64 adds + comparisons | threshold_table lookup | 45% |
| Quick-check (re-try previous killing window) | ~10-30 int64 adds | — | 15% |
| Canonicalization (survivors only) | ~32 comparisons | 64×4 = 256B | <1% |
| Gray code advance | ~5 int32 ops | ~20B state | <5% |

**The bottleneck is the window scan.** For each child that isn't quick-killed (~10-20%), we must scan up to ~63 different window sizes (ell=2..128), each with up to ~127 window positions. Worst case: ~8000 threshold comparisons per child. However, with the optimized ell ordering, >90% of non-quick-killed children are pruned within the first 5-10 ell values.

## Problem Category

This maps to a well-studied GPU pattern: **massively parallel branch-and-bound with lightweight bounding function**. Key characteristics:
- **Embarrassingly parallel across parents** (147M independent work units at L4)
- **Sequential within each parent** (Gray code enumeration is inherently sequential)
- **Compute-bound** (autoconv update + window scan, not memory-bound)
- **Extremely low survival rate** (stream compaction with ~0.001% hit rate)
- **Irregular workload** (different parents have vastly different child counts)
