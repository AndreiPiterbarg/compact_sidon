"""Benchmark for _fused_generate_and_prune_gray kernel.

Generates L1 survivors (d=8) and uses them as L2 parents (d_child=16),
and loads L2 shard data for L3 benchmarks (d_child=32).

Reports throughput in children/sec for median of 3 runs.
"""
import sys
import os
import time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    _fused_generate_and_prune_gray,
    _fused_generate_and_prune,
    _compute_bin_ranges,
    run_level0,
    process_parent_fused,
)

M = 20
C_TARGET = 1.4


def generate_l1_survivors():
    """Run L0 -> L1 to get d=8 survivors (L2 parents)."""
    l0 = run_level0(n_half=2, m=M, c_target=C_TARGET, verbose=False)
    l0_surv = l0['survivors']

    all_l1 = []
    n_half_child = 4
    for parent in l0_surv:
        surv, tc = process_parent_fused(parent, M, C_TARGET, n_half_child)
        if len(surv) > 0:
            all_l1.append(surv)

    if all_l1:
        l1_surv = np.vstack(all_l1)
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        _canonicalize_inplace(l1_surv)
        l1_surv = _fast_dedup(l1_surv)
    else:
        l1_surv = np.empty((0, 8), dtype=np.int32)

    print(f"L0: {len(l0_surv)} survivors, L1: {len(l1_surv)} survivors")
    return l0_surv, l1_surv


def benchmark_kernel(parents, n_half_child, label, n_parents=200, n_runs=3,
                     min_children_per_parent=10):
    """Benchmark _fused_generate_and_prune_gray on a sample of parents.

    Returns dict with median throughput.
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent

    # Pre-compute bin ranges and filter valid parents with enough children
    all_valid = []
    for i in range(len(parents)):
        parent = parents[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is not None:
            lo_arr, hi_arr, total = result
            if total >= min_children_per_parent:
                all_valid.append((parent, lo_arr, hi_arr, total))

    if not all_valid:
        print(f"  [{label}] No valid parents found!")
        return None

    # Sort by total_children descending, take top n_parents for interesting workload
    all_valid.sort(key=lambda x: -x[3])
    valid_parents = all_valid[:n_parents]

    total_children_all = sum(tc for _, _, _, tc in valid_parents)
    avg_tc = total_children_all / len(valid_parents)
    print(f"  [{label}] {len(valid_parents)} parents, "
          f"{total_children_all:,} total children "
          f"(avg {avg_tc:,.0f}/parent)")

    # Warmup JIT
    p0, lo0, hi0, tc0 = valid_parents[0]
    buf = np.empty((min(tc0, 100000), d_child), dtype=np.int32)
    _fused_generate_and_prune_gray(p0, n_half_child, M, C_TARGET, lo0, hi0, buf)

    # Benchmark runs
    timings = []
    for run in range(n_runs):
        total_children = 0
        total_survivors = 0
        total_subtree = 0
        t0 = time.perf_counter()
        for parent, lo_arr, hi_arr, tc in valid_parents:
            buf_cap = min(tc, 5_000_000)
            out_buf = np.empty((buf_cap, d_child), dtype=np.int32)
            n_surv, n_sub = _fused_generate_and_prune_gray(
                parent, n_half_child, M, C_TARGET, lo_arr, hi_arr, out_buf)
            total_children += tc
            total_survivors += n_surv
            total_subtree += n_sub
        elapsed = time.perf_counter() - t0
        throughput = total_children / elapsed
        timings.append((elapsed, throughput, total_children, total_survivors,
                        total_subtree))
        print(f"    Run {run+1}: {elapsed:.3f}s, "
              f"{throughput/1e6:.2f}M children/sec, "
              f"{total_survivors:,} surv, {total_subtree:,} subtree prunes")

    timings.sort(key=lambda x: x[0])
    median = timings[len(timings) // 2]
    print(f"  [{label}] MEDIAN: {median[0]:.3f}s, "
          f"{median[1]/1e6:.2f}M children/sec")
    return {
        'label': label,
        'elapsed': median[0],
        'throughput': median[1],
        'total_children': median[2],
        'total_survivors': median[3],
        'total_subtree': median[4],
        'n_parents': len(valid_parents),
    }


def run_benchmarks():
    print("=" * 70)
    print("Gray Code Kernel Benchmark")
    print("=" * 70)

    # Generate parents
    l0_surv, l1_surv = generate_l1_survivors()

    results = {}

    # Benchmark L2 (d_child=16, parents are d=8)
    if len(l1_surv) > 0:
        print(f"\n--- L2 benchmark: d_parent=8 -> d_child=16 ---")
        r = benchmark_kernel(l1_surv, n_half_child=8, label="L2_d16",
                            n_parents=300)
        if r:
            results['L2'] = r

    # Try to load L2 shard data for L3 benchmark (d_child=32)
    shard_dir = os.path.join(_project_dir, 'data', '_shards_L2')
    if os.path.exists(shard_dir):
        shard_files = sorted([f for f in os.listdir(shard_dir)
                             if f.startswith('shard_') and f.endswith('.npy')
                             and '.m' not in f])
        if shard_files:
            shard_path = os.path.join(shard_dir, shard_files[0])
            try:
                l2_surv = np.load(shard_path, mmap_mode='r')[:2000]
                l2_surv = np.array(l2_surv)
                print(f"\n--- L3 benchmark: d_parent=16 -> d_child=32 ---")
                print(f"  Loaded {len(l2_surv)} L2 survivors from {shard_files[0]}")
                r = benchmark_kernel(l2_surv, n_half_child=16,
                                    label="L3_d32", n_parents=300,
                                    min_children_per_parent=100)
                if r:
                    results['L3'] = r
            except Exception as e:
                print(f"  Could not load L2 shard: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {v['label']}: {v['throughput']/1e6:.2f}M children/sec "
              f"({v['total_children']:,} children, "
              f"{v['total_survivors']:,} surv, "
              f"{v['total_subtree']:,} subtree prunes, "
              f"{v['elapsed']:.3f}s)")

    return results


if __name__ == '__main__':
    run_benchmarks()
