"""Definitive pod benchmark — ONE-SHOT comprehensive L4 performance test.

Designed to run ON THE POD to answer:
  1. What is the true per-parent time on this pod's CPUs?
  2. What is the optimal worker count?
  3. What is the projected L4 runtime at optimal?

Uses the EXACT same dispatch path as run_cascade.py:
  - mmap parent array to temp file
  - spawn Pool with _init_worker_shm
  - imap_unordered with _process_parent_shm
  - Same chunksize formula

Usage:
    python3.12 -u baseline/pod_benchmark.py
    python3.12 -u baseline/pod_benchmark.py --n_single 200 --n_scaling 2000
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

# -- Path setup --
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

print("Loading modules + JIT warmup...", flush=True)
t_load = time.time()
from cpu.run_cascade import (
    process_parent_fused,
    _fused_generate_and_prune,
    _compute_bin_ranges,
    _prune_dynamic_int32,
    _canonicalize_inplace,
    _init_worker_shm,
    _process_parent_shm,
    _default_buf_cap,
)
import numba
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

# =====================================================================
# Constants — must match the actual proof parameters
# =====================================================================
N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
N_HALF_CHILD = N_HALF * (2 ** 4)  # = 32
TOTAL_L3 = 147_279_894

CHECKPOINT = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'pod_benchmark_results.json')


# =====================================================================
# Phase 0: Hardware fingerprint
# =====================================================================

def hardware_fingerprint():
    """Collect comprehensive hardware info."""
    info = {
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu_count_logical': mp.cpu_count(),
        'numpy_version': np.__version__,
        'numba_version': numba.__version__,
        'numba_threading_layer': numba.config.THREADING_LAYER,
    }

    # lscpu
    try:
        out = subprocess.check_output(['lscpu'], text=True, timeout=5)
        info['lscpu_raw'] = out
        for line in out.split('\n'):
            k, _, v = line.partition(':')
            v = v.strip()
            if not v:
                continue
            if 'Model name' in k:
                info['cpu_model'] = v
            elif k.strip() == 'CPU(s)':
                info['cpu_logical'] = int(v)
            elif 'Thread(s) per core' in k:
                info['threads_per_core'] = int(v)
            elif 'Core(s) per socket' in k:
                info['cores_per_socket'] = int(v)
            elif k.strip() == 'Socket(s)':
                info['sockets'] = int(v)
            elif 'CPU max MHz' in k:
                info['cpu_max_mhz'] = float(v)
            elif k.strip() == 'CPU MHz' :
                try:
                    info['cpu_mhz'] = float(v)
                except ValueError:
                    pass
            elif 'L1d cache' in k:
                info['l1d_cache'] = v
            elif 'L1i cache' in k:
                info['l1i_cache'] = v
            elif 'L2 cache' in k:
                info['l2_cache'] = v
            elif 'L3 cache' in k:
                info['l3_cache'] = v
            elif 'NUMA node(s)' in k:
                info['numa_nodes'] = int(v)
    except Exception as e:
        info['lscpu_error'] = str(e)

    # Compute physical cores
    if 'cores_per_socket' in info and 'sockets' in info:
        info['physical_cores'] = info['cores_per_socket'] * info['sockets']

    # RAM via /proc/meminfo (no psutil dependency)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    info['ram_total_gb'] = round(int(line.split()[1]) / 1e6, 2)
                elif line.startswith('MemAvailable'):
                    info['ram_available_gb'] = round(int(line.split()[1]) / 1e6, 2)
    except Exception:
        pass

    # cgroup CPU quota (container vCPUs)
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read().strip())
        if quota > 0:
            info['cgroup_vcpus'] = round(quota / period, 1)
    except Exception:
        pass
    # cgroup v2
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            parts = f.read().strip().split()
            if parts[0] != 'max':
                quota = int(parts[0])
                period = int(parts[1])
                info['cgroup_vcpus'] = round(quota / period, 1)
    except Exception:
        pass

    # Check what's actually running (competing processes)
    try:
        out = subprocess.check_output(
            ['ps', 'aux', '--sort=-pcpu'], text=True, timeout=5)
        lines = out.strip().split('\n')[:8]  # top 7 processes
        info['top_processes'] = '\n'.join(lines)
    except Exception:
        pass

    return info


# =====================================================================
# Sample loading + JIT warmup
# =====================================================================

def load_sample(n_sample, seed=42):
    """Stratified sample from L3 survivors."""
    print(f"Loading L3 survivors from {CHECKPOINT}...", flush=True)
    parents = np.load(CHECKPOINT, mmap_mode='r')
    n_total = len(parents)
    print(f"  {n_total:,} parents, shape={parents.shape}", flush=True)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=min(n_sample, n_total),
                                 replace=False))
    sample = np.array(parents[indices])  # copy out of mmap
    print(f"  Sampled {len(sample)} parents (seed={seed})", flush=True)
    return sample


def warmup_jit():
    """Full JIT warmup for L4 dimensions so timing is clean."""
    print("Warming up JIT for d=64...", flush=True)
    t0 = time.time()
    dummy = np.zeros((1, D_CHILD), dtype=np.int32)
    dummy[0, 0] = M
    _prune_dynamic_int32(dummy, N_HALF_CHILD, M, C_TARGET)
    _canonicalize_inplace(dummy.copy())
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    buf = np.empty((1, D_CHILD), dtype=np.int32)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _fused_generate_and_prune(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, buf)  # returns tuple, ignored
    # Also do a second call to ensure all code paths are compiled
    real_parent = np.zeros(D_PARENT, dtype=np.int32)
    real_parent[0] = 3
    real_parent[1] = 3
    real_parent[D_PARENT // 2] = 3
    real_parent[D_PARENT - 1] = M - 9
    process_parent_fused(real_parent, M, C_TARGET, N_HALF_CHILD)
    print(f"  JIT warm in {time.time() - t0:.1f}s", flush=True)


# =====================================================================
# Phase 1: Single-core per-parent timing
# =====================================================================

def single_core_benchmark(sample):
    """Detailed per-parent timing — gives the true compute cost."""
    n = len(sample)
    times = np.empty(n)
    children_counts = np.empty(n, dtype=np.int64)
    survivor_counts = np.empty(n, dtype=np.int64)

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 1: Single-core per-parent timing ({n} parents)", flush=True)
    print(f"{'='*70}", flush=True)

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
            wall = time.time() - t_total
            rate = (i + 1) / wall
            print(f"  [{i+1}/{n}] wall={wall:.1f}s, "
                  f"last={elapsed*1000:.1f}ms, "
                  f"children={n_children:,}, "
                  f"rate={rate:.2f}/s", flush=True)

    wall_total = time.time() - t_total

    stats = {
        'n_parents': n,
        'wall_seconds': round(wall_total, 3),
        'parents_per_sec': round(n / wall_total, 4),
        'per_parent_mean_ms': round(float(np.mean(times)) * 1000, 3),
        'per_parent_median_ms': round(float(np.median(times)) * 1000, 3),
        'per_parent_p5_ms': round(float(np.percentile(times, 5)) * 1000, 3),
        'per_parent_p25_ms': round(float(np.percentile(times, 25)) * 1000, 3),
        'per_parent_p75_ms': round(float(np.percentile(times, 75)) * 1000, 3),
        'per_parent_p95_ms': round(float(np.percentile(times, 95)) * 1000, 3),
        'per_parent_min_ms': round(float(np.min(times)) * 1000, 3),
        'per_parent_max_ms': round(float(np.max(times)) * 1000, 3),
        'children_per_parent_mean': round(float(np.mean(children_counts)), 1),
        'children_per_parent_median': round(float(np.median(children_counts)), 1),
        'children_per_parent_min': int(np.min(children_counts)),
        'children_per_parent_max': int(np.max(children_counts)),
        'survivor_rate': round(
            float(np.sum(survivor_counts)) /
            max(1, float(np.sum(children_counts))), 8),
        'total_survivors': int(np.sum(survivor_counts)),
    }

    ideal_196 = TOTAL_L3 / stats['parents_per_sec'] / 196 / 3600
    ideal_32 = TOTAL_L3 / stats['parents_per_sec'] / 32 / 3600

    print(f"\n  Single-core throughput: {stats['parents_per_sec']:.2f} parents/sec",
          flush=True)
    print(f"  Per-parent: {stats['per_parent_mean_ms']:.1f}ms mean, "
          f"{stats['per_parent_median_ms']:.1f}ms median", flush=True)
    print(f"  Per-parent p5/p95: {stats['per_parent_p5_ms']:.1f} / "
          f"{stats['per_parent_p95_ms']:.1f} ms", flush=True)
    print(f"  Children/parent: {stats['children_per_parent_mean']:.0f} mean "
          f"({stats['children_per_parent_min']:,} - "
          f"{stats['children_per_parent_max']:,})", flush=True)
    print(f"  Ideal projection @32 vCPU:  {ideal_32:.1f}h "
          f"({ideal_32/24:.1f}d)", flush=True)
    print(f"  Ideal projection @196 vCPU: {ideal_196:.1f}h "
          f"({ideal_196/24:.1f}d)", flush=True)

    return stats, times, children_counts


# =====================================================================
# Phase 2: Parallel scaling curve (EXACT cascade code path)
# =====================================================================

def scaling_benchmark(sample, worker_counts, output_dir):
    """Test parallel throughput at various worker counts.

    Uses the EXACT same mechanism as run_cascade's parallel path:
      1. Write parents to raw binary temp file
      2. spawn Pool with _init_worker_shm (workers mmap the file)
      3. imap_unordered with _process_parent_shm
      4. Same chunksize formula
    """
    n = len(sample)

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2: Parallel scaling ({n} parents x "
          f"{len(worker_counts)} worker configs)", flush=True)
    print(f"{'='*70}", flush=True)

    # Write parents to temp file (same as cascade line 1597-1601)
    fd, mmap_path = tempfile.mkstemp(suffix='_bench_parents.dat',
                                     dir=output_dir)
    os.close(fd)
    sample.tofile(mmap_path)
    parents_shape = sample.shape
    parents_dtype_str = sample.dtype.str

    print(f"  Parent mmap: {mmap_path} "
          f"({sample.nbytes / 1e6:.1f} MB)", flush=True)

    results = []
    ctx = mp.get_context("spawn")

    for nw in worker_counts:
        if nw > n:
            print(f"\n  Skipping {nw} workers (> {n} parents)", flush=True)
            continue

        numba_threads = min(max(1, mp.cpu_count() // nw),
                            numba.config.NUMBA_NUM_THREADS)
        chunksize = max(1, min(n // (nw * 20), 128))

        print(f"\n  --- {nw} workers "
              f"(numba_threads={numba_threads}, chunksize={chunksize}) ---",
              flush=True)

        # Measure pool spawn time separately
        t_spawn = time.time()
        try:
            pool = ctx.Pool(
                nw,
                initializer=_init_worker_shm,
                initargs=(mmap_path, parents_shape, parents_dtype_str,
                          M, C_TARGET, N_HALF_CHILD, numba_threads))
        except Exception as e:
            print(f"    FAILED to create pool: {e}", flush=True)
            results.append({'n_workers': nw, 'error': str(e)})
            continue
        spawn_time = time.time() - t_spawn
        print(f"    Pool spawned in {spawn_time:.1f}s", flush=True)

        # Measure processing time (what matters for real run)
        t_work = time.time()
        total_children = 0
        total_survived = 0
        completed = 0
        first_result_time = None

        try:
            for surv, stats in pool.imap_unordered(
                    _process_parent_shm, range(n), chunksize=chunksize):
                total_children += stats['children']
                total_survived += stats['survived']
                completed += 1
                if first_result_time is None:
                    first_result_time = time.time() - t_work

                # Progress at 25%, 50%, 75%, 100%
                if completed in (n // 4, n // 2, 3 * n // 4, n):
                    elapsed = time.time() - t_work
                    rate = completed / elapsed
                    print(f"    [{completed}/{n}] {rate:.1f} parents/sec, "
                          f"elapsed={elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"    FAILED during processing: {e}", flush=True)
            results.append({'n_workers': nw, 'error': str(e)})
            pool.terminate()
            pool.join()
            continue

        work_time = time.time() - t_work

        pool.close()
        pool.join()

        throughput = completed / work_time if work_time > 0 else 0
        per_parent_eff = (work_time * nw / completed) if completed > 0 else 0
        projected_hours = TOTAL_L3 / throughput / 3600 if throughput > 0 else 0

        result = {
            'n_workers': nw,
            'numba_threads': numba_threads,
            'chunksize': chunksize,
            'spawn_seconds': round(spawn_time, 3),
            'first_result_seconds': round(first_result_time, 3)
                if first_result_time else None,
            'work_seconds': round(work_time, 3),
            'throughput': round(throughput, 4),
            'per_parent_effective_ms': round(per_parent_eff * 1000, 3),
            'total_children': total_children,
            'total_survived': total_survived,
            'projected_l4_hours': round(projected_hours, 2),
            'projected_l4_days': round(projected_hours / 24, 2),
        }
        results.append(result)

        print(f"    Throughput: {throughput:.1f} parents/sec | "
              f"Effective: {per_parent_eff*1000:.1f}ms/parent | "
              f"L4 projection: {projected_hours:.1f}h "
              f"({projected_hours/24:.1f}d)", flush=True)

    # Clean up
    try:
        os.remove(mmap_path)
    except OSError:
        pass

    # Compute scaling efficiency relative to single-core throughput
    # from Phase 1 (filled in by caller)
    return results


# =====================================================================
# Phase 3: Sustained throughput at optimal (verify steady-state)
# =====================================================================

def sustained_benchmark(sample, n_workers, output_dir, duration_target=90):
    """Run for a fixed duration at the optimal worker count.

    Cycles through the sample repeatedly to accumulate enough data
    for a stable throughput estimate. This verifies there's no
    degradation over time (thermal throttling, memory leaks, etc).
    """
    n = len(sample)

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 3: Sustained throughput ({n_workers} workers, "
          f"~{duration_target}s target)", flush=True)
    print(f"{'='*70}", flush=True)

    fd, mmap_path = tempfile.mkstemp(suffix='_bench_sustained.dat',
                                     dir=output_dir)
    os.close(fd)
    sample.tofile(mmap_path)
    parents_shape = sample.shape
    parents_dtype_str = sample.dtype.str

    numba_threads = min(max(1, mp.cpu_count() // n_workers),
                        numba.config.NUMBA_NUM_THREADS)
    chunksize = max(1, min(n // (n_workers * 20), 128))
    # For sustained test: repeat indices to fill duration
    # Estimate repeats needed
    estimated_rate = n_workers * 0.5  # conservative: 0.5 parents/sec/worker
    total_parents_needed = int(estimated_rate * duration_target * 1.5)
    total_parents_needed = max(total_parents_needed, n * 3)
    # Create repeated index sequence
    repeats = (total_parents_needed // n) + 1
    indices = list(range(n)) * repeats

    ctx = mp.get_context("spawn")

    print(f"  Will process up to {len(indices):,} parent-tasks "
          f"(sample repeated {repeats}x)", flush=True)
    print(f"  Target duration: {duration_target}s", flush=True)

    t_spawn = time.time()
    pool = ctx.Pool(
        n_workers,
        initializer=_init_worker_shm,
        initargs=(mmap_path, parents_shape, parents_dtype_str,
                  M, C_TARGET, N_HALF_CHILD, numba_threads))
    print(f"  Pool spawned in {time.time() - t_spawn:.1f}s", flush=True)

    t_start = time.time()
    completed = 0
    checkpoints = []  # (elapsed, completed) for throughput-over-time

    try:
        for surv, stats in pool.imap_unordered(
                _process_parent_shm, indices, chunksize=chunksize):
            completed += 1
            elapsed = time.time() - t_start

            # Record checkpoint every 10s
            if not checkpoints or elapsed - checkpoints[-1][0] >= 10.0:
                rate = completed / elapsed
                checkpoints.append((round(elapsed, 1), completed,
                                    round(rate, 2)))
                print(f"    t={elapsed:.0f}s: {completed:,} done, "
                      f"{rate:.1f} parents/sec", flush=True)

            if elapsed >= duration_target:
                break
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
    finally:
        pool.terminate()
        pool.join()

    total_elapsed = time.time() - t_start

    try:
        os.remove(mmap_path)
    except OSError:
        pass

    throughput = completed / total_elapsed if total_elapsed > 0 else 0
    projected = TOTAL_L3 / throughput / 3600 if throughput > 0 else 0

    # Analyze throughput stability
    rates = [cp[2] for cp in checkpoints]
    rate_drift = None
    if len(rates) >= 3:
        first_half = np.mean(rates[:len(rates)//2])
        second_half = np.mean(rates[len(rates)//2:])
        rate_drift = round((second_half - first_half) / first_half * 100, 1)

    result = {
        'n_workers': n_workers,
        'duration_seconds': round(total_elapsed, 3),
        'parents_processed': completed,
        'throughput': round(throughput, 4),
        'projected_l4_hours': round(projected, 2),
        'projected_l4_days': round(projected / 24, 2),
        'checkpoints': checkpoints,
        'rate_drift_pct': rate_drift,
    }

    print(f"\n  Sustained throughput: {throughput:.1f} parents/sec "
          f"over {total_elapsed:.0f}s", flush=True)
    print(f"  Rate drift: {rate_drift:+.1f}% "
          f"(first half vs second half)" if rate_drift is not None
          else "  (not enough data for drift analysis)", flush=True)
    print(f"  Projected L4: {projected:.1f}h ({projected/24:.1f}d)",
          flush=True)

    return result


# =====================================================================
# Final report
# =====================================================================

def print_final_report(hw, single_stats, scaling_results, sustained):
    """Print the definitive analysis and recommendation."""

    print(f"\n{'='*70}", flush=True)
    print(f"  DEFINITIVE L4 PERFORMANCE REPORT", flush=True)
    print(f"{'='*70}", flush=True)

    # Hardware summary
    model = hw.get('cpu_model', 'unknown')
    phys = hw.get('physical_cores', '?')
    logical = hw.get('cpu_count_logical', '?')
    cgroup = hw.get('cgroup_vcpus', None)
    l3 = hw.get('l3_cache', '?')
    numa = hw.get('numa_nodes', '?')
    ram = hw.get('ram_total_gb', '?')

    print(f"\n  Hardware:", flush=True)
    print(f"    CPU model:      {model}", flush=True)
    print(f"    Physical cores: {phys}", flush=True)
    print(f"    Logical CPUs:   {logical} (what mp.cpu_count() sees)",
          flush=True)
    if cgroup is not None:
        print(f"    cgroup vCPUs:   {cgroup} (ACTUAL container limit)",
              flush=True)
    print(f"    L3 cache:       {l3}", flush=True)
    print(f"    NUMA nodes:     {numa}", flush=True)
    print(f"    RAM:            {ram} GB", flush=True)

    # Single-core baseline
    sc = single_stats
    print(f"\n  Single-core baseline:", flush=True)
    print(f"    Throughput:     {sc['parents_per_sec']:.2f} parents/sec",
          flush=True)
    print(f"    Per-parent:     {sc['per_parent_mean_ms']:.0f}ms mean, "
          f"{sc['per_parent_median_ms']:.0f}ms median", flush=True)
    print(f"    Children/parent: {sc['children_per_parent_mean']:.0f} mean",
          flush=True)

    # Scaling curve
    print(f"\n  Scaling curve:", flush=True)
    print(f"    {'Workers':>8}  {'Throughput':>14}  {'Eff/parent':>12}  "
          f"{'L4 hours':>10}  {'L4 days':>8}  {'Efficiency':>10}",
          flush=True)
    print(f"    {'-'*8}  {'-'*14}  {'-'*12}  "
          f"{'-'*10}  {'-'*8}  {'-'*10}", flush=True)

    base_rate = single_stats['parents_per_sec']
    best = None
    for r in scaling_results:
        if 'error' in r:
            print(f"    {r['n_workers']:>8}  {'FAILED':>14}", flush=True)
            continue

        t = r['throughput']
        ideal = base_rate * r['n_workers']
        eff = t / ideal if ideal > 0 else 0

        marker = ''
        if best is None or t > best['throughput']:
            best = r

        print(f"    {r['n_workers']:>8}  {t:>11.1f}/s  "
              f"{r['per_parent_effective_ms']:>9.0f}ms  "
              f"{r['projected_l4_hours']:>8.1f}h  "
              f"{r['projected_l4_days']:>6.1f}d  "
              f"{eff:>9.1%}", flush=True)

    # Sustained result
    if sustained:
        print(f"\n  Sustained throughput ({sustained['n_workers']} workers, "
              f"{sustained['duration_seconds']:.0f}s):", flush=True)
        print(f"    Throughput:     {sustained['throughput']:.1f} parents/sec",
              flush=True)
        print(f"    Rate drift:     {sustained['rate_drift_pct']:+.1f}%"
              if sustained.get('rate_drift_pct') is not None
              else "    Rate drift:     N/A", flush=True)
        print(f"    Projected L4:   {sustained['projected_l4_hours']:.1f}h "
              f"({sustained['projected_l4_days']:.1f}d)", flush=True)

    # Optimal recommendation
    if best:
        print(f"\n  {'='*50}", flush=True)
        print(f"  RECOMMENDATION:", flush=True)
        print(f"  {'='*50}", flush=True)
        print(f"  Optimal workers: {best['n_workers']}", flush=True)
        print(f"  Throughput:      {best['throughput']:.1f} parents/sec",
              flush=True)
        print(f"  L4 completion:   {best['projected_l4_hours']:.1f} hours "
              f"({best['projected_l4_days']:.1f} days)", flush=True)
        print(f"\n  Restart command:", flush=True)
        print(f"    python3.12 -u cloninger-steinerberger/cpu/run_cascade.py "
              f"--n_half 2 --m 20 --c_target 1.40 "
              f"--workers {best['n_workers']} --resume", flush=True)
        print(f"  {'='*50}", flush=True)

    return best


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Definitive pod L4 performance benchmark')
    parser.add_argument('--n_single', type=int, default=100,
                        help='Parents for single-core test (default: 100)')
    parser.add_argument('--n_scaling', type=int, default=1000,
                        help='Parents for scaling tests (default: 1000)')
    parser.add_argument('--sustained_secs', type=int, default=90,
                        help='Duration for sustained test (default: 90)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()

    t_bench_start = time.time()

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: {CHECKPOINT} not found", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Phase 0: Hardware ----
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 0: Hardware fingerprint", flush=True)
    print(f"{'='*70}", flush=True)
    hw = hardware_fingerprint()
    for k, v in sorted(hw.items()):
        if k not in ('lscpu_raw', 'top_processes'):
            print(f"  {k}: {v}", flush=True)
    if 'top_processes' in hw:
        print(f"\n  Top processes by CPU:", flush=True)
        for line in hw['top_processes'].split('\n'):
            print(f"    {line}", flush=True)

    logical_cpus = hw.get('cpu_count_logical', mp.cpu_count())
    physical_cores = hw.get('physical_cores', logical_cpus)
    cgroup_vcpus = hw.get('cgroup_vcpus', None)

    # Determine effective CPU count for scaling tests
    if cgroup_vcpus and cgroup_vcpus < logical_cpus:
        effective_cpus = int(cgroup_vcpus)
        print(f"\n  *** CONTAINER LIMIT DETECTED: {effective_cpus} vCPUs "
              f"(host has {logical_cpus}) ***", flush=True)
    else:
        effective_cpus = physical_cores
        print(f"\n  Using physical core count: {effective_cpus}", flush=True)

    # ---- Load samples ----
    sample_single = load_sample(args.n_single, seed=args.seed)
    if args.n_scaling > args.n_single:
        sample_scaling = load_sample(args.n_scaling, seed=args.seed)
    else:
        sample_scaling = sample_single

    # ---- JIT warmup ----
    warmup_jit()

    # ---- Phase 1: Single-core ----
    single_stats, times, children = single_core_benchmark(sample_single)

    # ---- Phase 2: Scaling curve ----
    # Build worker counts: dense around expected optimal, sparse at extremes
    wc = set()
    # Always test these
    for w in [4, 8, 16, 24, 32, 48, 64, 96, 128]:
        if w <= logical_cpus:
            wc.add(w)
    # Test around effective CPU count
    for frac in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        w = max(4, int(effective_cpus * frac))
        if w <= logical_cpus:
            wc.add(w)
    # Test at logical max
    wc.add(logical_cpus)
    worker_counts = sorted(wc)

    print(f"\n  Worker counts to test: {worker_counts}", flush=True)

    scaling_results = scaling_benchmark(sample_scaling, worker_counts,
                                        args.output_dir)

    # Find optimal from scaling results
    best_throughput = 0
    optimal_workers = effective_cpus
    for r in scaling_results:
        if 'error' not in r and r['throughput'] > best_throughput:
            best_throughput = r['throughput']
            optimal_workers = r['n_workers']

    # ---- Phase 3: Sustained throughput at optimal ----
    sustained = sustained_benchmark(sample_scaling, optimal_workers,
                                     args.output_dir,
                                     duration_target=args.sustained_secs)

    # ---- Final report ----
    best = print_final_report(hw, single_stats, scaling_results, sustained)

    total_bench_time = time.time() - t_bench_start

    # ---- Save everything ----
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmark_duration_seconds': round(total_bench_time, 1),
        'hardware': {k: v for k, v in hw.items()
                     if k not in ('lscpu_raw', 'top_processes')},
        'lscpu_raw': hw.get('lscpu_raw', ''),
        'top_processes': hw.get('top_processes', ''),
        'parameters': {
            'n_half': N_HALF, 'm': M, 'c_target': C_TARGET,
            'd_parent': D_PARENT, 'd_child': D_CHILD,
            'n_half_child': N_HALF_CHILD,
            'total_l3_survivors': TOTAL_L3,
        },
        'single_core': single_stats,
        'scaling': scaling_results,
        'sustained': sustained,
        'optimal': best if best and 'error' not in best else None,
        'optimal_workers': optimal_workers,
        'per_parent_times_ms': [round(t * 1000, 3) for t in times.tolist()],
        'children_per_parent': children.tolist(),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)
    print(f"Total benchmark time: {total_bench_time/60:.1f} minutes",
          flush=True)


if __name__ == '__main__':
    main()
