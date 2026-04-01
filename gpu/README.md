# CUDA Kernel Specification: Cascade Branch-and-Prune Prover

> **Target:** 64x NVIDIA H100 80GB SXM (Hopper architecture)
> **Goal:** Port the CPU cascade prover to CUDA, achieving 100-500x speedup over 32-core CPU Numba.
> **Critical constraint:** Mathematical correctness is non-negotiable. This is a rigorous proof — a single missed survivor invalidates the result.

## Document Index

| File | Contents |
|------|----------|
| [01_problem.md](01_problem.md) | Problem characterization, scale, bottleneck analysis |
| [02_challenges.md](02_challenges.md) | 13 challenges of converting the CPU algorithm to CUDA |
| [03_h100_hardware.md](03_h100_hardware.md) | H100 specs, INT32 throughput, memory budgets, occupancy |
| [04_kernel_architecture.md](04_kernel_architecture.md) | Kernel design: dispatch, conv update, window scan, quick-check, compaction |
| [05_optimizations.md](05_optimizations.md) | INT32/FP32 analysis, bank conflicts, register pressure, divergence, coalescing |
| [08_correctness.md](08_correctness.md) | Mathematical rigor, verification strategy, FP reproducibility |
| [10_memory_layout.md](10_memory_layout.md) | Global/shared memory maps, register estimates, pitfalls, build flags |
| [11_implementation_alternatives.md](11_implementation_alternatives.md) | 10 algorithm components × 3-5 options each, decision matrix |

## Priority Summary

| Priority | Optimization | Impact | Difficulty |
|----------|-------------|--------|------------|
| **P0** | Persistent blocks with atomic work-stealing | Enables full GPU utilization | Low |
| **P0** | Warp-parallel incremental conv update | Core throughput | Medium |
| **P0** | Quick-check with warp-cooperative sum | Skips 85% of full scans | Low |
| **P0** | Precomputed threshold table in shared memory | Eliminates runtime float64 | Low |
| **P0** | Correct Gray code implementation | Mathematical correctness | High (must verify) |
| **P1** | Prefix sum + parallel window scan | Parallelizes remaining 15% | Medium |
| **P1** | Warp-ballot canonicalization | O(1) vs O(d/2) | Low |
| **P1** | Block-level survivor staging + flush | Minimizes global atomics | Low |
| **P1** | TMA parent loading with double buffering | Hides memory latency | Medium |
| **P2** | Bit-packed sort keys for dedup | 5x less sort bandwidth | Medium |
| **P2** | Multi-GPU round-robin with pre-sort | Linear scaling to 64 GPUs | Low |
| **P2** | Thread block clusters for shared threshold table | Saves shared memory | Medium |
| **P3** | Subtree pruning on GPU | Reduces children at L4 | High (complex control flow) |
| **P3** | Profile-guided ell ordering per level | Marginal improvement | Low |
