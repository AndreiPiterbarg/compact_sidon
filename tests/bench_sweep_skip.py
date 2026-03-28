"""Benchmark: univariate sweep skip performance impact.

Measures wall-clock time for process_parent_fused on diverse parents at
L0->L1 and L1->L2 transitions.  Runs multiple iterations to get stable
timings, reporting median and IQR.

Usage:
    python tests/bench_sweep_skip.py
"""
import sys
import os
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import process_parent_fused, _compute_bin_ranges

M = 20
C_TARGET = 1.4


def warmup():
    """JIT warmup on a small parent."""
    p = np.array([5, 5, 5, 5], dtype=np.int32)
    process_parent_fused(p, M, C_TARGET, len(p))


def bench_parents(parents, label, n_half_child, n_iters=5):
    """Benchmark a set of parents, returning median time."""
    times = []
    total_surv = 0
    total_children = 0

    for _ in range(n_iters):
        t0 = time.perf_counter()
        for p in parents:
            surv, tc = process_parent_fused(p, M, C_TARGET, n_half_child)
            if _ == 0:
                total_surv += len(surv)
                total_children += tc
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times) // 2]
    lo = times[max(0, len(times) // 4)]
    hi = times[min(len(times) - 1, 3 * len(times) // 4)]

    print(f"  {label}: {median:.4f}s (IQR [{lo:.4f}, {hi:.4f}])")
    print(f"    parents={len(parents)}, children={total_children:,}, "
          f"survivors={total_surv:,}")
    return median


def main():
    print("Warming up JIT...", flush=True)
    warmup()
    print("JIT warm.\n")

    # --- L0 parents (d_parent=4 -> d_child=8) ---
    l0_parents = [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 5, 3, 2], dtype=np.int32),
        np.array([4, 6, 4, 6], dtype=np.int32),
        np.array([0, 10, 10, 0], dtype=np.int32),
        np.array([3, 3, 7, 7], dtype=np.int32),
        np.array([7, 3, 7, 3], dtype=np.int32),
        np.array([0, 5, 0, 15], dtype=np.int32),
        np.array([2, 8, 8, 2], dtype=np.int32),
        np.array([6, 4, 6, 4], dtype=np.int32),
        np.array([1, 9, 9, 1], dtype=np.int32),
    ]

    print("=== L0 -> L1 (d_child=8) ===")
    bench_parents(l0_parents, "L0 parents", n_half_child=4, n_iters=10)

    # --- L1 parents: generate from L0 ---
    print("\nGenerating L1 parents...", flush=True)
    l1_parents = []
    for p in l0_parents:
        surv, _ = process_parent_fused(p, M, C_TARGET, len(p))
        l1_parents.extend(surv[:20])
    print(f"  {len(l1_parents)} L1 parents collected.\n")

    print("=== L1 -> L2 (d_child=16) ===")
    bench_parents(l1_parents, "L1 parents", n_half_child=8, n_iters=5)

    # --- Focused L1 timing (more iterations for stability) ---
    print("\n=== Focused L1 -> L2 benchmark (20 iterations) ===")
    bench_parents(l1_parents, "L1 focused", n_half_child=8, n_iters=20)


if __name__ == "__main__":
    main()
