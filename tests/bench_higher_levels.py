"""Benchmark conv optimization at L2->L3 and L3->L4 using actual checkpoint data.

Uses the real L2 survivors (116K parents at d=16) from the c_target=1.3 run
to measure wall-clock impact at the production bottleneck level.

Usage:
    python tests/bench_higher_levels.py
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
C_TARGET = 1.3  # matches checkpoint data


def warmup():
    p = np.array([5, 5, 5, 5], dtype=np.int32)
    process_parent_fused(p, M, C_TARGET, len(p))


def bench_parents(parents, label, n_half_child, n_iters=5):
    times = []
    total_surv = 0
    total_children = 0

    for it in range(n_iters):
        t0 = time.perf_counter()
        for p in parents:
            surv, tc = process_parent_fused(p, M, C_TARGET, n_half_child)
            if it == 0:
                total_surv += len(surv)
                total_children += tc
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times) // 2]
    lo = times[max(0, len(times) // 4)]
    hi = times[min(len(times) - 1, 3 * len(times) // 4)]

    rate = total_children / median if median > 0 else 0
    print(f"  {label}: {median:.4f}s (IQR [{lo:.4f}, {hi:.4f}])")
    print(f"    parents={len(parents)}, children={total_children:,}, "
          f"survivors={total_surv:,}, rate={rate/1e6:.1f}M children/s")
    return median


def main():
    print("Warming up JIT...", flush=True)
    warmup()
    print("JIT warm.\n")

    # --- L2 -> L3 using actual checkpoint data ---
    l2_path = os.path.join(_project_dir, 'data', 'checkpoint_L2_survivors.npy')
    if os.path.exists(l2_path):
        l2_all = np.load(l2_path)
        print(f"Loaded L2 survivors: {l2_all.shape}")

        # Sample 200 parents for tractable benchmarking
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(42)
        idx = rng.choice(len(l2_all), size=min(200, len(l2_all)), replace=False)
        l2_sample = l2_all[idx]

        print(f"\n=== L2 -> L3 (d_child=32, {len(l2_sample)} parents) ===")
        bench_parents(l2_sample, "L2->L3", n_half_child=16, n_iters=5)
    else:
        print(f"No L2 checkpoint at {l2_path}")

    # --- L3 -> L4 using actual checkpoint data ---
    l3_path = os.path.join(_project_dir, 'data', 'checkpoint_L3_survivors.npy')
    if os.path.exists(l3_path):
        l3_all = np.load(l3_path)
        print(f"\nLoaded L3 survivors: {l3_all.shape}")

        # Sample 50 parents (L4 is expensive per parent)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(l3_all), size=min(50, len(l3_all)), replace=False)
        l3_sample = l3_all[idx]

        print(f"\n=== L3 -> L4 (d_child=64, {len(l3_sample)} parents) ===")
        bench_parents(l3_sample, "L3->L4", n_half_child=32, n_iters=3)
    else:
        print(f"\nNo L3 checkpoint at {l3_path}")


if __name__ == "__main__":
    main()
