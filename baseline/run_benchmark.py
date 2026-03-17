"""L4 baseline benchmark — measure per-parent throughput for cloud projections.

Samples N parents from L3 survivors, processes them through the L4 fused
kernel (d=32 -> 64), and reports:

  1. Per-parent timing statistics (mean, median, p95)
  2. Single-core throughput (parents/sec)
  3. Projected wall-clock time at various core counts (local, 48-core, 196-core)

Reliability features:
  - CPU pinned to single core + high priority (eliminates migration/scheduling noise)
  - time.perf_counter() for sub-microsecond resolution (vs 10-16ms for time.time())
  - Two warmup passes before timed passes to stabilize JIT + caches + branch predictors
  - 5 timed passes by default, reports median
  - children/sec as primary metric (stable across parent weight variation)
  - CV% across passes to quantify reliability

Usage:
    python -m baseline.run_benchmark                  # default: 100 parents, 5 passes
    python -m baseline.run_benchmark --n_sample 500   # more parents for tighter estimate
    python -m baseline.run_benchmark --passes 7       # more passes for tighter variance
    python -m baseline.run_benchmark --parallel        # also run parallel scaling test
"""
import argparse
import ctypes
import gc
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
    _fused_generate_and_prune_gray,
    _compute_bin_ranges,
    _prune_dynamic_int32,
    _canonicalize_inplace,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

# Module-level flag: set by --gray CLI arg
_USE_GRAY = False

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


# =====================================================================
# OS-level noise reduction
# =====================================================================

def _pin_and_boost():
    """Pin process to one core + raise priority. Returns cleanup function."""
    old_affinity = None
    old_priority = None
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()

        # Save and set affinity — pick core 1 (avoid core 0 which handles
        # most OS interrupts on Windows)
        mask = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(mask), ctypes.byref(ctypes.c_ulonglong(0)))
        old_affinity = mask.value
        # Pin to core 1 if available, else core 0
        target_core = 2 if (old_affinity & 2) else 1  # bitmask: core1=0b10
        kernel32.SetProcessAffinityMask(handle, target_core)

        # Save and set priority to HIGH_PRIORITY_CLASS (0x80)
        old_priority = kernel32.GetPriorityClass(handle)
        kernel32.SetPriorityClass(handle, 0x00000080)

        print(f"  Pinned to core {target_core.bit_length()-1 if isinstance(target_core, int) else 1}, "
              f"priority HIGH", flush=True)
    except Exception as e:
        print(f"  (could not pin/boost: {e})", flush=True)

    def _restore():
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            if old_affinity is not None:
                kernel32.SetProcessAffinityMask(handle, old_affinity)
            if old_priority is not None:
                kernel32.SetPriorityClass(handle, old_priority)
        except Exception:
            pass

    return _restore


def load_sample(n_sample, seed=42):
    """Load a stratified sample of L3 parents."""
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


def _warmup_jit():
    """Warm up JIT for L4-sized arrays (d_child=64) so first parent isn't slow."""
    dummy = np.zeros((1, D_CHILD), dtype=np.int32)
    dummy[0, 0] = M
    _prune_dynamic_int32(dummy, N_HALF_CHILD, M, C_TARGET)
    _canonicalize_inplace(dummy.copy())
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)
    _fused_generate_and_prune_gray(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)


def _process_parent_gray(parent_int, m, c_target, n_half_child):
    """Like process_parent_fused but uses the Gray code kernel."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    result = _compute_bin_ranges(parent_int, m, c_target, d_child)
    if result is None:
        return np.empty((0, d_child), dtype=np.int32), 0
    lo_arr, hi_arr, total_children = result
    if total_children == 0:
        return np.empty((0, d_child), dtype=np.int32), 0
    max_buf = min(total_children, 500_000)
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)
    n_survivors, _ = _fused_generate_and_prune_gray(
        parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf)
    if n_survivors > max_buf:
        max_buf = n_survivors
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)
        n2, _ = _fused_generate_and_prune_gray(
            parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf)
        n_survivors = n2
    return out_buf[:n_survivors].copy(), total_children


def run_one_pass(sample):
    """Run all parents once, return (times, children_counts, survivor_counts, wall)."""
    n = len(sample)
    times = np.empty(n)
    children_counts = np.empty(n, dtype=np.int64)
    survivor_counts = np.empty(n, dtype=np.int64)

    process_fn = _process_parent_gray if _USE_GRAY else process_parent_fused

    gc.disable()
    t_total = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        survivors, n_children = process_fn(sample[i], M, C_TARGET, N_HALF_CHILD)
        times[i] = time.perf_counter() - t0
        children_counts[i] = n_children
        survivor_counts[i] = len(survivors)
    wall = time.perf_counter() - t_total
    gc.enable()
    return times, children_counts, survivor_counts, wall


def benchmark_sequential(sample, n_passes=5):
    """Time each parent over multiple passes — gives reliable throughput.

    Two warmup passes (results discarded), then n_passes timed.
    Returns the median pass results.
    """
    n = len(sample)

    # JIT warmup on dummy data
    print("Warming up JIT...", flush=True)
    _warmup_jit()

    # Two warmup passes on real data (stabilizes caches + branch predictors;
    # second pass catches anything the first didn't fully warm)
    for w in range(2):
        print(f"  Warmup pass {w+1}/2 ({n} parents)...", flush=True)
        t0 = time.perf_counter()
        run_one_pass(sample)
        print(f"    {time.perf_counter() - t0:.2f}s", flush=True)

    # Timed passes
    pass_walls = []
    pass_times = []
    pass_children = None
    pass_survivors = None

    for p in range(n_passes):
        # Force GC between passes so it doesn't fire mid-pass
        gc.collect()
        print(f"\n  Pass {p+1}/{n_passes}...", flush=True)
        times, children_counts, survivor_counts, wall = run_one_pass(sample)
        pass_walls.append(wall)
        pass_times.append(times)
        # Children/survivors are deterministic — same every pass
        if pass_children is None:
            pass_children = children_counts
            pass_survivors = survivor_counts

        total_children = int(np.sum(children_counts))
        children_per_sec = total_children / wall
        print(f"    {wall:.3f}s  |  {n/wall:.2f} parents/sec  |  "
              f"{children_per_sec/1e6:.2f}M children/sec", flush=True)

    # Select median pass by wall time
    median_idx = int(np.argsort(pass_walls)[len(pass_walls) // 2])
    return (pass_times[median_idx], pass_children, pass_survivors,
            pass_walls[median_idx], pass_walls)


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
                  pass_walls, parallel_result=None):
    """Compute and format benchmark statistics."""
    n = len(times)
    parents_per_sec = n / wall_total
    total_children = int(np.sum(children_counts))
    children_per_sec = total_children / wall_total

    # Variance across passes
    walls_arr = np.array(pass_walls)
    rates = n / walls_arr
    child_rates = total_children / walls_arr
    cv_pct = float(np.std(rates) / np.mean(rates) * 100) if len(rates) > 1 else 0.0

    stats = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'variant': 'gray' if _USE_GRAY else 'odometer',
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
            'children_per_sec': round(children_per_sec, 0),
            'per_parent_mean_ms': round(np.mean(times) * 1000, 3),
            'per_parent_median_ms': round(np.median(times) * 1000, 3),
            'per_parent_p5_ms': round(np.percentile(times, 5) * 1000, 3),
            'per_parent_p95_ms': round(np.percentile(times, 95) * 1000, 3),
            'per_parent_max_ms': round(np.max(times) * 1000, 3),
            'children_per_parent_mean': round(float(np.mean(children_counts)), 1),
            'children_per_parent_median': round(float(np.median(children_counts)), 1),
            'total_children': total_children,
            'survivor_rate': round(float(np.sum(survivor_counts)) /
                                   max(1, float(np.sum(children_counts))), 6),
        },
        'reliability': {
            'n_passes': len(pass_walls),
            'pass_walls': [round(w, 3) for w in pass_walls],
            'pass_rates': [round(n / w, 4) for w in pass_walls],
            'pass_child_rates': [round(total_children / w, 0) for w in pass_walls],
            'rate_cv_pct': round(cv_pct, 2),
            'min_rate': round(float(np.min(rates)), 4),
            'max_rate': round(float(np.max(rates)), 4),
            'median_rate': round(float(np.median(rates)), 4),
        },
        'projections': {},
    }

    # Project total L4 time at various core counts
    for label, cores in [('local', mp.cpu_count()),
                         ('cloud_48', 48),
                         ('cloud_96', 96),
                         ('cloud_196', 196)]:
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
    rel = stats['reliability']
    print(f"\n{'='*60}")
    print(f"  L4 BASELINE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Sample: {stats['sample']['n_sample']} parents "
          f"(of {stats['sample']['total_l3_survivors']:,})")
    print(f"  Parameters: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"  d: {D_PARENT} -> {D_CHILD}")
    print()
    print(f"  Sequential (1 core, median of {rel['n_passes']} passes):")
    print(f"    Wall time:          {seq['wall_seconds']:.3f}s")
    print(f"    Throughput:         {seq['parents_per_sec']:.2f} parents/sec")
    print(f"    Children/sec:       {seq['children_per_sec']/1e6:.2f}M")
    print(f"    Per-parent mean:    {seq['per_parent_mean_ms']:.1f} ms")
    print(f"    Per-parent median:  {seq['per_parent_median_ms']:.1f} ms")
    print(f"    Per-parent p5/p95:  {seq['per_parent_p5_ms']:.1f} / "
          f"{seq['per_parent_p95_ms']:.1f} ms")
    print(f"    Children/parent:    {seq['children_per_parent_mean']:.0f} (mean)")
    print(f"    Survivor rate:      {seq['survivor_rate']:.4%}")
    print()
    print(f"  Reliability ({rel['n_passes']} passes):")
    print(f"    Pass rates:         "
          f"{', '.join(f'{r:.2f}' for r in rel['pass_rates'])} parents/sec")
    print(f"    CV:                 {rel['rate_cv_pct']:.1f}%")
    print(f"    Range:              {rel['min_rate']:.2f} - "
          f"{rel['max_rate']:.2f} parents/sec")
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
    """Save results, keeping only the last 5 entries."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(stats)

    # Keep only last 5 results to avoid unbounded growth
    if len(all_results) > 5:
        all_results = all_results[-5:]

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description='L4 baseline benchmark')
    parser.add_argument('--n_sample', type=int, default=100,
                        help='Number of L3 parents to sample (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--passes', type=int, default=5,
                        help='Number of timed passes (default: 5)')
    parser.add_argument('--parallel', action='store_true',
                        help='Also run parallel scaling test')
    parser.add_argument('--gray', action='store_true',
                        help='Use Gray code kernel instead of lexicographic odometer')
    args = parser.parse_args()

    global _USE_GRAY
    _USE_GRAY = args.gray
    if _USE_GRAY:
        print(">>> Using GRAY CODE kernel <<<", flush=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: L3 checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    sample = load_sample(args.n_sample, seed=args.seed)

    # Pin to single core + raise priority for stable measurements
    restore = _pin_and_boost()

    try:
        # -- Sequential benchmark (the key measurement) --
        times, children_counts, survivor_counts, wall_total, pass_walls = \
            benchmark_sequential(sample, n_passes=args.passes)
    finally:
        restore()

    # -- Optional parallel test (after restoring affinity!) --
    parallel_result = None
    if args.parallel:
        parallel_result = benchmark_parallel(sample, mp.cpu_count())

    # -- Report --
    stats = compute_stats(times, children_counts, survivor_counts,
                          wall_total, pass_walls, parallel_result)
    print_report(stats)
    save_results(stats)


if __name__ == '__main__':
    main()
