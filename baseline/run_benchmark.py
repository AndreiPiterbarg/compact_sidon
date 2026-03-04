"""L4 baseline benchmark — measure per-parent throughput for cloud projections.

Samples N parents from L3 survivors, processes them through the L4 fused
kernel (d=32 -> 64), and reports:

  1. Per-parent timing statistics (mean, median, p95)
  2. Single-core throughput (parents/sec)
  3. Projected wall-clock time at various core counts (local, 48-core, 196-core)

The key metric is single-core throughput — cloud scaling is nearly linear
because each parent is processed independently with no cross-parent
communication.

Usage:
    python -m baseline.run_benchmark                  # default: 100 parents
    python -m baseline.run_benchmark --n_sample 500   # more parents for tighter estimate
    python -m baseline.run_benchmark --parallel        # also run parallel scaling test
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

# -- Path setup: import from cloninger-steinerberger/ --
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

# Import the cascade machinery (triggers JIT warmup on load)
print("Loading modules + JIT warmup...", flush=True)
t_load = time.time()
from cpu.run_cascade import (
    process_parent_fused,
    _fused_generate_and_prune,
    _compute_bin_ranges,
    _prune_dynamic_int32,
    _canonicalize_inplace,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

# =====================================================================
# Constants — must match the actual proof parameters
# =====================================================================
N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
# At L4, n_half has doubled 4 times: 2 -> 4 -> 8 -> 16 -> 32
N_HALF_CHILD = N_HALF * (2 ** 4)  # = 32
TOTAL_L3_SURVIVORS = 147_279_894

CHECKPOINT_PATH = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results.json')


def load_sample(n_sample, seed=42):
    """Load a stratified sample of L3 parents.

    Samples uniformly across the sorted survivor array to get a
    representative mix of light parents (few children) and heavy
    parents (many children). This matters because parent weight
    varies by 10-100x.
    """
    print(f"Loading L3 survivors from {CHECKPOINT_PATH}...")
    parents = np.load(CHECKPOINT_PATH)
    n_total = len(parents)
    print(f"  {n_total:,} parents, shape={parents.shape}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    indices.sort()
    sample = parents[indices]
    del parents  # free ~2.2 GB
    print(f"  Sampled {len(sample)} parents (seed={seed}, "
          f"same indices every run)")
    return sample


def _warmup_l4():
    """Warm up JIT for L4-sized arrays (d_child=64) so first parent isn't slow."""
    dummy = np.zeros((1, D_CHILD), dtype=np.int32)
    dummy[0, 0] = M  # valid composition summing to m
    _prune_dynamic_int32(dummy, N_HALF_CHILD, M, C_TARGET)
    _canonicalize_inplace(dummy.copy())
    # Also warm the fused kernel at L4 dimensions
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)  # returns tuple, ignored


def benchmark_sequential(sample):
    """Time each parent individually — gives per-parent distribution."""
    n = len(sample)
    times = np.empty(n)
    children_counts = np.empty(n, dtype=np.int64)
    survivor_counts = np.empty(n, dtype=np.int64)

    # Warm up JIT for L4 dimensions before timing
    print("Warming up JIT for d=64...", flush=True)
    _warmup_l4()

    print(f"\nRunning {n} parents sequentially (single-core)...")
    t_total = time.time()

    for i in range(n):
        t0 = time.time()
        survivors, n_children = process_parent_fused(
            sample[i], M, C_TARGET, N_HALF_CHILD)
        elapsed = time.time() - t0

        times[i] = elapsed
        children_counts[i] = n_children
        survivor_counts[i] = len(survivors)

        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            pct = (i + 1) / n * 100
            wall = time.time() - t_total
            print(f"  [{i+1}/{n}] ({pct:.0f}%) "
                  f"wall={wall:.1f}s, last={elapsed:.3f}s, "
                  f"children={n_children:,}, surv={len(survivors):,}",
                  flush=True)

    wall_total = time.time() - t_total
    return times, children_counts, survivor_counts, wall_total


def benchmark_parallel(sample, n_workers):
    """Time parallel processing to measure scaling efficiency."""
    from cpu.run_cascade import _process_single_parent_fused
    import numba

    n = len(sample)
    numba_threads = max(1, mp.cpu_count() // n_workers)

    def init_worker(nt):
        numba.set_num_threads(nt)

    args_list = [(sample[i], M, C_TARGET, N_HALF_CHILD, 500_000)
                 for i in range(n)]

    print(f"\nRunning {n} parents with {n_workers} workers "
          f"(numba_threads={numba_threads})...")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=init_worker,
                  initargs=(numba_threads,)) as pool:
        results = list(pool.imap_unordered(
            _process_single_parent_fused, args_list, chunksize=1))

    wall = time.time() - t0
    total_survived = sum(r[1]['survived'] for r in results)
    total_children = sum(r[1]['children'] for r in results)
    return wall, total_children, total_survived


def compute_stats(times, children_counts, survivor_counts, wall_total,
                  parallel_result=None):
    """Compute and format benchmark statistics."""
    n = len(times)
    parents_per_sec = n / wall_total

    stats = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'n_half': N_HALF, 'm': M, 'c_target': C_TARGET,
            'd_parent': D_PARENT, 'd_child': D_CHILD,
            'n_half_child': N_HALF_CHILD,
            'x_cap': int(np.floor(M * np.sqrt(
                (C_TARGET + 2.0/M + 1.0/(M*M) + 1e-9) / D_CHILD))),
        },
        'sample': {
            'n_sample': n,
            'seed': 42,
            'total_l3_survivors': TOTAL_L3_SURVIVORS,
        },
        'sequential': {
            'wall_seconds': round(wall_total, 3),
            'parents_per_sec': round(parents_per_sec, 4),
            'per_parent_mean_ms': round(np.mean(times) * 1000, 3),
            'per_parent_median_ms': round(np.median(times) * 1000, 3),
            'per_parent_p5_ms': round(np.percentile(times, 5) * 1000, 3),
            'per_parent_p95_ms': round(np.percentile(times, 95) * 1000, 3),
            'per_parent_max_ms': round(np.max(times) * 1000, 3),
            'children_per_parent_mean': round(float(np.mean(children_counts)), 1),
            'children_per_parent_median': round(float(np.median(children_counts)), 1),
            'survivor_rate': round(float(np.sum(survivor_counts)) /
                                   max(1, float(np.sum(children_counts))), 6),
        },
        'projections': {},
    }

    # Project total L4 time at various core counts
    for label, cores in [('local', mp.cpu_count()),
                         ('cloud_48', 48),
                         ('cloud_96', 96),
                         ('cloud_196', 196)]:
        # Each core processes parents at single-core rate
        total_sec = TOTAL_L3_SURVIVORS / (parents_per_sec * cores)
        stats['projections'][label] = {
            'cores': cores,
            'projected_hours': round(total_sec / 3600, 2),
            'projected_days': round(total_sec / 86400, 2),
        }

    if parallel_result is not None:
        par_wall, par_children, par_survived = parallel_result
        par_rate = n / par_wall
        efficiency = par_rate / (parents_per_sec * stats['projections']['local']['cores'])
        stats['parallel'] = {
            'n_workers': mp.cpu_count(),
            'wall_seconds': round(par_wall, 3),
            'parents_per_sec': round(par_rate, 4),
            'scaling_efficiency': round(efficiency, 4),
        }

    return stats


def print_report(stats):
    """Pretty-print the benchmark results."""
    seq = stats['sequential']
    print(f"\n{'='*60}")
    print(f"  L4 BASELINE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Sample: {stats['sample']['n_sample']} parents "
          f"(of {stats['sample']['total_l3_survivors']:,})")
    print(f"  Parameters: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"  d: {D_PARENT} -> {D_CHILD}")
    print()
    print(f"  Sequential (1 core):")
    print(f"    Wall time:          {seq['wall_seconds']:.1f}s")
    print(f"    Throughput:         {seq['parents_per_sec']:.2f} parents/sec")
    print(f"    Per-parent mean:    {seq['per_parent_mean_ms']:.1f} ms")
    print(f"    Per-parent median:  {seq['per_parent_median_ms']:.1f} ms")
    print(f"    Per-parent p5/p95:  {seq['per_parent_p5_ms']:.1f} / "
          f"{seq['per_parent_p95_ms']:.1f} ms")
    print(f"    Children/parent:    {seq['children_per_parent_mean']:.0f} (mean)")
    print(f"    Survivor rate:      {seq['survivor_rate']:.4%}")
    print()
    print(f"  Projections for full L4 ({TOTAL_L3_SURVIVORS:,} parents):")
    for label, proj in stats['projections'].items():
        print(f"    {label:>12} ({proj['cores']:>3} cores): "
              f"{proj['projected_hours']:>8.1f} hours "
              f"({proj['projected_days']:.1f} days)")

    if 'parallel' in stats:
        par = stats['parallel']
        print()
        print(f"  Parallel ({par['n_workers']} workers):")
        print(f"    Wall time:          {par['wall_seconds']:.1f}s")
        print(f"    Throughput:         {par['parents_per_sec']:.2f} parents/sec")
        print(f"    Scaling efficiency: {par['scaling_efficiency']:.1%}")

    print(f"{'='*60}")


def save_results(stats):
    """Append results to the baseline results file."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(stats)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description='L4 baseline benchmark')
    parser.add_argument('--n_sample', type=int, default=100,
                        help='Number of L3 parents to sample (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--parallel', action='store_true',
                        help='Also run parallel scaling test')
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: L3 checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    sample = load_sample(args.n_sample, seed=args.seed)

    # -- Sequential benchmark (the key measurement) --
    times, children_counts, survivor_counts, wall_total = \
        benchmark_sequential(sample)

    # -- Optional parallel test --
    parallel_result = None
    if args.parallel:
        parallel_result = benchmark_parallel(sample, mp.cpu_count())

    # -- Report --
    stats = compute_stats(times, children_counts, survivor_counts,
                          wall_total, parallel_result)
    print_report(stats)
    save_results(stats)


if __name__ == '__main__':
    main()
