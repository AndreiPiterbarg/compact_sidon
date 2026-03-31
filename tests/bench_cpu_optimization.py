#!/usr/bin/env python3
"""Comprehensive CPU optimization benchmark for RunPod cloud pods.

Designed for the cpu3c-32-64 instance (32 vCPU, 64 GB RAM).
Tests every tunable parameter to find the optimal configuration
for the cascade prover's production L4 run.

Usage (on pod):
    python tests/bench_cpu_optimization.py              # Full suite
    python tests/bench_cpu_optimization.py --quick       # Quick subset (~10 min)
    python tests/bench_cpu_optimization.py --section 3   # Run only section 3

Sections:
    1. Hardware detection & baseline
    2. Worker count vs Numba thread ratio
    3. Chunk size for imap_unordered
    4. Buffer capacity tuning
    5. L1/L2 cache profiling (staging buffer size)
    6. Single-parent kernel throughput at each level
    7. Memory budget & shard threshold
    8. NUMA / thread affinity effects
    9. End-to-end cascade timing (L0-L2 fast, L3 sample)
   10. Idea 1: Cursor-range tightening effectiveness
   11. Parameter sweep summary & recommendations
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import subprocess
import tempfile

import numpy as np
import numba
from numba import njit, prange

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_proj_dir, "cloninger-steinerberger")
_cpu_dir = os.path.join(_cs_dir, "cpu")
if _cs_dir not in sys.path:
    sys.path.insert(0, _cs_dir)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)

from run_cascade import (
    _compute_bin_ranges,
    _fused_generate_and_prune_gray,
    _prune_dynamic_int32,
    _prune_dynamic_int64,
    _default_buf_cap,
    _effective_cpu_count,
    process_parent_fused,
    run_level0,
    run_cascade,
    _fast_dedup,
    _canonicalize_inplace,
)
from pruning import correction, asymmetry_threshold, count_compositions
from compositions import generate_canonical_compositions_batched

# Also import bench_common helpers if available
try:
    sys.path.insert(0, _this_dir)
    from bench_common import (
        build_threshold_table, build_ell_order, build_parent_prefix,
        build_floor_child, compute_autoconv, tighten_cursor_ranges,
        compute_product, warmup_jit as warmup_bench_jit,
    )
    HAS_BENCH_COMMON = True
except ImportError:
    HAS_BENCH_COMMON = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M = 20
C_TARGET = 1.4
N_HALF = 2
DATA_DIR = os.path.join(_proj_dir, "data")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg):
    print(msg, flush=True)


def _fmt_time(s):
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def _separator(title, char="=", width=78):
    _log(f"\n{char * width}")
    _log(f"  {title}")
    _log(f"{char * width}")


def _subsep(title, char="-", width=60):
    _log(f"\n  {char * width}")
    _log(f"  {title}")
    _log(f"  {char * width}")


def _table_header(cols, widths):
    header = "  "
    for c, w in zip(cols, widths):
        header += f"{c:>{w}} | "
    _log(header)
    _log("  " + "-" * (sum(widths) + 3 * len(widths)))


def _table_row(vals, widths, formats=None):
    row = "  "
    for i, (v, w) in enumerate(zip(vals, widths)):
        fmt = formats[i] if formats else ""
        if isinstance(v, float):
            row += f"{v:>{w}.{fmt}f} | " if fmt else f"{v:>{w}.3f} | "
        elif isinstance(v, int):
            row += f"{v:>{w},} | "
        else:
            row += f"{str(v):>{w}} | "
    _log(row)


def load_checkpoint(level):
    """Load checkpoint survivors for a given level."""
    path = os.path.join(DATA_DIR, f"checkpoint_L{level}_survivors.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def sample_parents(parents, n, seed=42):
    """Random sample of parents for benchmarking."""
    if len(parents) <= n:
        return parents
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(parents), n, replace=False)
    return parents[np.sort(idx)]


# =====================================================================
# Section 1: Hardware Detection & Baseline
# =====================================================================

def section1_hardware():
    _separator("SECTION 1: Hardware Detection & Baseline")

    # CPU info
    logical_cpus = mp.cpu_count()
    effective_cpus = _effective_cpu_count()
    _log(f"  Logical CPUs:    {logical_cpus}")
    _log(f"  Effective CPUs:  {effective_cpus}")
    _log(f"  Platform:        {platform.platform()}")
    _log(f"  Architecture:    {platform.machine()}")
    _log(f"  Python:          {platform.python_version()}")
    _log(f"  Numba:           {numba.__version__}")
    _log(f"  NumPy:           {np.__version__}")
    _log(f"  Numba threads:   {numba.config.NUMBA_NUM_THREADS}")

    # Memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        _log(f"  Total RAM:       {vm.total / 1e9:.1f} GB")
        _log(f"  Available RAM:   {vm.available / 1e9:.1f} GB")
        _log(f"  RAM usage:       {vm.percent:.1f}%")
    except ImportError:
        _log("  psutil not available - cannot detect RAM")

    # CPU model (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    _log(f"  CPU model:       {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        pass

    # Cache sizes (Linux)
    for level in (1, 2, 3):
        for suffix in ("d", ""):
            path = f"/sys/devices/system/cpu/cpu0/cache/index{level}/size"
            try:
                with open(path) as f:
                    _log(f"  L{level} cache:      {f.read().strip()}")
                    break
            except FileNotFoundError:
                continue

    # NUMA topology
    try:
        result = subprocess.run(
            ["numactl", "--hardware"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split("\n")[:8]:
                if line.strip():
                    _log(f"  NUMA: {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _log("  NUMA: numactl not available")

    # Numba JIT warmup timing
    _log("\n  JIT warmup timing:")
    for d in (4, 8, 16, 32, 64):
        dummy = np.zeros((1, d), dtype=np.int32)
        t0 = time.time()
        _prune_dynamic_int32(dummy, d // 2, 20, 1.3)
        t1 = time.time()
        _log(f"    d={d:>3}: {(t1-t0)*1000:.0f}ms (first call / compilation)")
        t0 = time.time()
        for _ in range(100):
            _prune_dynamic_int32(dummy, d // 2, 20, 1.3)
        t1 = time.time()
        _log(f"    d={d:>3}: {(t1-t0)/100*1e6:.1f}us (warm, per call)")

    return {"logical_cpus": logical_cpus, "effective_cpus": effective_cpus}


# =====================================================================
# Section 2: Worker Count vs Numba Thread Ratio
# =====================================================================

def section2_worker_threads(hw_info):
    _separator("SECTION 2: Worker Count vs Numba Thread Ratio")
    _log("  Goal: Find optimal (workers, numba_threads) split for 32 vCPU")
    _log("  Test: Process 200 L2 parents at level 3 (d_child=32)")

    # Load L2 checkpoint
    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found. Run cascade through L2 first.")
        return {}

    n_sample = min(200, len(parents))
    test_parents = sample_parents(parents, n_sample)
    n_half_child = 16  # L3: d_parent=16 -> d_child=32
    effective = hw_info.get("effective_cpus", mp.cpu_count())

    # Test configurations: (workers, numba_threads_per_worker)
    # Total threads = workers * numba_threads should not exceed effective CPUs
    configs = []
    for w in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]:
        if w > effective:
            continue
        nt = max(1, effective // w)
        configs.append((w, nt))
    # Also test oversubscription
    if effective >= 16:
        configs.append((effective, 1))        # max workers, 1 thread each
        configs.append((effective // 2, 2))   # half workers, 2 threads
        configs.append((4, effective // 4))   # 4 workers, many threads

    # Remove duplicates, sort by workers
    configs = sorted(set(configs), key=lambda x: x[0])

    results = []
    _table_header(
        ["Workers", "NThreads", "Total", "Time(s)", "Parents/s", "Speedup"],
        [8, 8, 6, 8, 10, 8])

    baseline_time = None

    for n_workers, n_threads in configs:
        total_threads = n_workers * n_threads

        # Write parents to temp mmap
        fd, mmap_path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        test_parents.tofile(mmap_path)

        t0 = time.time()
        try:
            if n_workers == 1:
                # Sequential path
                numba.set_num_threads(n_threads)
                total_surv = 0
                for i in range(n_sample):
                    s, _ = process_parent_fused(
                        test_parents[i], M, C_TARGET, n_half_child)
                    total_surv += len(s)
            else:
                ctx = mp.get_context("spawn")
                chunksize = max(1, min(n_sample // (n_workers * 20), 128))

                from run_cascade import _init_worker_shm, _process_parent_shm
                with ctx.Pool(
                        n_workers,
                        initializer=_init_worker_shm,
                        initargs=(mmap_path, test_parents.shape,
                                  test_parents.dtype.str,
                                  M, C_TARGET, n_half_child,
                                  n_threads)) as pool:
                    total_surv = 0
                    for surv, stats in pool.imap_unordered(
                            _process_parent_shm, range(n_sample),
                            chunksize=chunksize):
                        total_surv += stats['survived']
            elapsed = time.time() - t0
        except Exception as e:
            _log(f"  ERROR with workers={n_workers}, threads={n_threads}: {e}")
            elapsed = float('inf')
        finally:
            try:
                os.remove(mmap_path)
            except OSError:
                pass

        if baseline_time is None:
            baseline_time = elapsed
        speedup = baseline_time / elapsed if elapsed > 0 else 0

        rate = n_sample / elapsed if elapsed > 0 else 0
        results.append({
            "workers": n_workers, "threads": n_threads,
            "total_threads": total_threads,
            "elapsed": elapsed, "rate": rate, "speedup": speedup,
        })
        _table_row(
            [n_workers, n_threads, total_threads, elapsed, rate, speedup],
            [8, 8, 6, 8, 10, 8],
            ["", "", "", "2", "1", "2"])

    # Reset numba threads
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    if results:
        best = min(results, key=lambda r: r["elapsed"])
        _log(f"\n  BEST: {best['workers']} workers x {best['threads']} threads "
             f"= {best['elapsed']:.2f}s ({best['rate']:.1f} parents/s)")

    return {"worker_thread_results": results}


# =====================================================================
# Section 3: Chunk Size Tuning
# =====================================================================

def section3_chunksize(hw_info):
    _separator("SECTION 3: imap_unordered Chunk Size")
    _log("  Goal: Find optimal chunksize for work distribution")

    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found.")
        return {}

    n_sample = min(500, len(parents))
    test_parents = sample_parents(parents, n_sample)
    n_half_child = 16
    effective = hw_info.get("effective_cpus", mp.cpu_count())
    n_workers = max(1, effective)

    # Chunksizes to test
    chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Also test the auto-computed default
    auto_chunk = max(1, min(n_sample // (n_workers * 20), 128))
    if auto_chunk not in chunk_sizes:
        chunk_sizes.append(auto_chunk)
    chunk_sizes = sorted(set(chunk_sizes))

    fd, mmap_path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    test_parents.tofile(mmap_path)

    results = []
    _table_header(
        ["Chunksize", "Time(s)", "Parents/s", "Note"],
        [10, 8, 10, 15])

    for cs in chunk_sizes:
        note = "(auto)" if cs == auto_chunk else ""
        t0 = time.time()
        try:
            ctx = mp.get_context("spawn")
            n_threads = max(1, effective // n_workers)
            from run_cascade import _init_worker_shm, _process_parent_shm
            with ctx.Pool(
                    n_workers,
                    initializer=_init_worker_shm,
                    initargs=(mmap_path, test_parents.shape,
                              test_parents.dtype.str,
                              M, C_TARGET, n_half_child,
                              n_threads)) as pool:
                for _ in pool.imap_unordered(
                        _process_parent_shm, range(n_sample),
                        chunksize=cs):
                    pass
            elapsed = time.time() - t0
        except Exception as e:
            _log(f"  ERROR chunksize={cs}: {e}")
            elapsed = float('inf')

        rate = n_sample / elapsed if elapsed > 0 else 0
        results.append({"chunksize": cs, "elapsed": elapsed, "rate": rate})
        _table_row([cs, elapsed, rate, note], [10, 8, 10, 15], ["", "2", "1", ""])

    try:
        os.remove(mmap_path)
    except OSError:
        pass

    if results:
        best = min(results, key=lambda r: r["elapsed"])
        _log(f"\n  BEST chunksize: {best['chunksize']} "
             f"({best['elapsed']:.2f}s, {best['rate']:.1f} parents/s)")

    return {"chunksize_results": results}


# =====================================================================
# Section 4: Buffer Capacity Tuning
# =====================================================================

def section4_buffer_capacity():
    _separator("SECTION 4: Survivor Buffer Capacity")
    _log("  Goal: Find buffer cap that avoids re-runs without wasting RAM")

    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found.")
        return {}

    # Sample parents with varying child counts
    n_sample = min(50, len(parents))
    test_parents = sample_parents(parents, n_sample)
    n_half_child = 16

    buf_caps = [1000, 5000, 10_000, 50_000, 100_000, 500_000,
                1_000_000, 5_000_000, 10_000_000]

    results = []
    _table_header(
        ["BufCap", "Time(s)", "ReRuns", "MaxSurv", "AvgSurv", "MB/worker"],
        [12, 8, 7, 10, 10, 10])

    for cap in buf_caps:
        t0 = time.time()
        n_reruns = 0
        max_surv = 0
        total_surv = 0

        for i in range(n_sample):
            parent = test_parents[i]
            result = _compute_bin_ranges(parent, M, C_TARGET, 32, n_half_child)
            if result is None:
                continue
            lo_arr, hi_arr, total_children = result
            if total_children == 0:
                continue

            max_buf = min(total_children, cap)
            out_buf = np.empty((max_buf, 32), dtype=np.int32)
            n_surv, _ = _fused_generate_and_prune_gray(
                parent, n_half_child, M, C_TARGET, lo_arr, hi_arr, out_buf)

            if n_surv > max_buf:
                n_reruns += 1
                # Would need rerun in production
            max_surv = max(max_surv, min(n_surv, cap))
            total_surv += min(n_surv, cap)

        elapsed = time.time() - t0
        avg_surv = total_surv / n_sample if n_sample > 0 else 0
        mem_mb = cap * 32 * 4 / 1e6

        results.append({
            "buf_cap": cap, "elapsed": elapsed,
            "reruns": n_reruns, "max_surv": max_surv,
            "avg_surv": avg_surv, "mem_mb": mem_mb,
        })
        _table_row(
            [cap, elapsed, n_reruns, max_surv, avg_surv, mem_mb],
            [12, 8, 7, 10, 10, 10],
            ["", "2", "", "", "0", "1"])

    _log(f"\n  Current defaults: d<=16: 10M, d<=32: 5M, d>32: 100K")
    if results:
        # Find smallest cap with zero reruns
        zero_rerun = [r for r in results if r["reruns"] == 0]
        if zero_rerun:
            best = min(zero_rerun, key=lambda r: r["mem_mb"])
            _log(f"  RECOMMENDED: buf_cap={best['buf_cap']:,} "
                 f"(0 reruns, {best['mem_mb']:.1f} MB/worker)")

    return {"buffer_results": results}


# =====================================================================
# Section 5: Staging Buffer Size (L1 Cache Fit)
# =====================================================================

def section5_staging_buffer():
    _separator("SECTION 5: Staging Buffer Size (L1 Cache Fit)")
    _log("  Goal: Find staging cap that maximizes cache-friendly writes")
    _log("  Current: d<=32 -> 512 rows, d>32 -> 256 rows (fits 64KB)")

    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found.")
        return {}

    # Test single parents at L3 to measure kernel timing
    n_sample = min(20, len(parents))
    test_parents = sample_parents(parents, n_sample)
    n_half_child = 16
    d_child = 32

    # The staging buffer size is compiled into the kernel, so we can't
    # change it at runtime. Instead, measure the kernel's performance
    # as-is and report the current setting's effectiveness.
    _log("\n  Measuring kernel throughput with current staging settings...")

    total_children = 0
    total_survivors = 0
    t0 = time.time()
    for i in range(n_sample):
        s, nc = process_parent_fused(test_parents[i], M, C_TARGET, n_half_child)
        total_children += nc
        total_survivors += len(s)
    elapsed = time.time() - t0

    children_per_sec = total_children / elapsed if elapsed > 0 else 0
    surv_rate = total_survivors / total_children * 100 if total_children > 0 else 0

    _log(f"  Parents processed:  {n_sample}")
    _log(f"  Total children:     {total_children:,}")
    _log(f"  Total survivors:    {total_survivors:,}")
    _log(f"  Survival rate:      {surv_rate:.1f}%")
    _log(f"  Throughput:         {children_per_sec:,.0f} children/s")
    _log(f"  Time per parent:    {elapsed/n_sample*1000:.1f}ms")
    _log(f"  Time per child:     {elapsed/total_children*1e6:.2f}us")

    # Theoretical staging buffer analysis
    _log("\n  Staging buffer size analysis (compile-time constant):")
    for stage_cap in [64, 128, 256, 512, 1024, 2048]:
        mem_bytes = stage_cap * d_child * 4
        _log(f"    stage_cap={stage_cap:>5}: {mem_bytes/1024:.1f}KB "
             f"({'fits L1' if mem_bytes <= 32768 else 'EXCEEDS L1 (32KB)'})")

    return {
        "children_per_sec": children_per_sec,
        "surv_rate": surv_rate,
        "time_per_child_us": elapsed / total_children * 1e6 if total_children else 0,
    }


# =====================================================================
# Section 6: Single-Parent Kernel Throughput per Level
# =====================================================================

def section6_kernel_throughput():
    _separator("SECTION 6: Single-Parent Kernel Throughput per Level")
    _log("  Goal: Measure per-child cost at each cascade level")

    level_configs = [
        # (level, d_parent, d_child, n_half_child, checkpoint_level)
        (1, 4, 8, 4, 0),
        (2, 8, 16, 8, 1),
        (3, 16, 32, 16, 2),
    ]

    # Try to add L4 if checkpoint exists
    l3_ckpt = load_checkpoint(3)
    if l3_ckpt is not None:
        level_configs.append((4, 32, 64, 32, 3))

    results = []
    _table_header(
        ["Level", "d_child", "Parents", "Children", "Survivors",
         "Surv%", "us/child", "children/s"],
        [6, 8, 8, 12, 10, 6, 9, 12])

    for level, d_par, d_child, nhc, ckpt_level in level_configs:
        parents = load_checkpoint(ckpt_level)
        if parents is None:
            _log(f"  L{level}: SKIP (no L{ckpt_level} checkpoint)")
            continue

        # Sample size depends on level
        n_sample = {1: 100, 2: 50, 3: 20, 4: 5}.get(level, 5)
        n_sample = min(n_sample, len(parents))
        test_parents = sample_parents(parents, n_sample)

        total_children = 0
        total_survivors = 0
        t0 = time.time()
        for i in range(n_sample):
            s, nc = process_parent_fused(test_parents[i], M, C_TARGET, nhc)
            total_children += nc
            total_survivors += len(s)
        elapsed = time.time() - t0

        surv_pct = total_survivors / total_children * 100 if total_children > 0 else 0
        us_per_child = elapsed / total_children * 1e6 if total_children > 0 else 0
        cps = total_children / elapsed if elapsed > 0 else 0

        results.append({
            "level": level, "d_child": d_child,
            "parents": n_sample, "children": total_children,
            "survivors": total_survivors, "surv_pct": surv_pct,
            "us_per_child": us_per_child, "children_per_sec": cps,
        })
        _table_row(
            [f"L{level}", d_child, n_sample, total_children,
             total_survivors, surv_pct, us_per_child, cps],
            [6, 8, 8, 12, 10, 6, 9, 12],
            ["", "", "", "", "", "1", "2", "0"])

    # Cost scaling analysis
    if len(results) >= 2:
        _log("\n  Cost scaling (us/child ratio between levels):")
        for i in range(1, len(results)):
            ratio = results[i]["us_per_child"] / results[i-1]["us_per_child"]
            _log(f"    L{results[i]['level']} / L{results[i-1]['level']}: "
                 f"{ratio:.2f}x (expected ~4x for d^2 scaling)")

    return {"kernel_throughput": results}


# =====================================================================
# Section 7: Memory Budget & Shard Threshold
# =====================================================================

def section7_memory_budget():
    _separator("SECTION 7: Memory Budget & Shard Threshold")
    _log("  Goal: Validate memory management for 64GB RAM pod")

    try:
        import psutil
        vm = psutil.virtual_memory()
        avail = vm.available
        total = vm.total
    except ImportError:
        avail = int(64e9 * 0.80)
        total = int(64e9)

    _log(f"\n  Total RAM:        {total/1e9:.1f} GB")
    _log(f"  Available RAM:    {avail/1e9:.1f} GB")

    # Simulate budget calculations for each level
    _log(f"\n  Budget analysis per level:")

    for level, d_child in [(1, 8), (2, 16), (3, 32), (4, 64)]:
        bytes_per_row = d_child * 4
        buf_cap = _default_buf_cap(d_child)

        # Compute memory layout
        survivor_mem_budget = max(int(1e9), (avail - int(10e9)) // 4)
        shard_threshold = max(100_000, survivor_mem_budget // bytes_per_row)

        # Per-worker memory
        per_worker = buf_cap * d_child * 4 + 150 * 1024 * 1024
        worker_budget = max(int(1e9), avail - survivor_mem_budget - int(4e9))
        max_workers_by_mem = max(1, int(worker_budget / per_worker))

        _log(f"\n  L{level} (d_child={d_child}):")
        _log(f"    Bytes/row:          {bytes_per_row} B")
        _log(f"    Default buf_cap:    {buf_cap:,} rows ({buf_cap*bytes_per_row/1e6:.0f} MB)")
        _log(f"    Survivor budget:    {survivor_mem_budget/1e9:.1f} GB")
        _log(f"    Shard threshold:    {shard_threshold:,} rows ({shard_threshold*bytes_per_row/1e9:.2f} GB)")
        _log(f"    Per-worker mem:     {per_worker/1e9:.2f} GB")
        _log(f"    Max workers (mem):  {max_workers_by_mem}")

    # Dedup memory analysis
    _log(f"\n  Deduplication overhead:")
    for n_rows, d in [(1_000_000, 32), (10_000_000, 32), (100_000_000, 32),
                       (1_000_000, 64), (10_000_000, 64)]:
        arr_bytes = n_rows * d * 4
        # lexsort needs: input + sort indices + output = ~3x
        peak = arr_bytes * 3
        _log(f"    {n_rows:>12,} rows x d={d}: "
             f"arr={arr_bytes/1e9:.1f}GB, peak_dedup={peak/1e9:.1f}GB "
             f"{'OK' if peak < avail * 0.8 else 'NEEDS SHARDING'}")

    return {"survivor_mem_budget": survivor_mem_budget}


# =====================================================================
# Section 8: NUMA / Thread Affinity Effects
# =====================================================================

def section8_numa():
    _separator("SECTION 8: NUMA / Thread Affinity Effects")

    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found.")
        return {}

    n_sample = min(100, len(parents))
    test_parents = sample_parents(parents, n_sample)
    n_half_child = 16

    # Check if numactl is available
    has_numactl = False
    try:
        result = subprocess.run(["numactl", "--show"],
                                capture_output=True, text=True, timeout=5)
        has_numactl = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not has_numactl:
        _log("  numactl not available - testing thread count only")

    # Test different Numba thread counts (single process)
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    max_threads = numba.config.NUMBA_NUM_THREADS
    thread_counts = [t for t in thread_counts if t <= max_threads]

    results = []
    _table_header(
        ["Threads", "Time(s)", "Parents/s", "Speedup"],
        [8, 8, 10, 8])

    baseline = None
    for nt in thread_counts:
        numba.set_num_threads(nt)
        t0 = time.time()
        for i in range(n_sample):
            process_parent_fused(test_parents[i], M, C_TARGET, n_half_child)
        elapsed = time.time() - t0

        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed

        rate = n_sample / elapsed
        results.append({
            "threads": nt, "elapsed": elapsed,
            "rate": rate, "speedup": speedup,
        })
        _table_row(
            [nt, elapsed, rate, speedup],
            [8, 8, 10, 8],
            ["", "2", "1", "2"])

    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    _log("\n  NOTE: The fused kernel is NOT prange-parallel (sequential Gray code).")
    _log("  Numba thread count mainly affects _prune_dynamic and _canonicalize.")
    _log("  Parallelism comes from multiprocessing (workers), not Numba threads.")
    _log("  RECOMMENDATION: Use 1 Numba thread per worker, maximize workers.")

    return {"numa_results": results}


# =====================================================================
# Section 9: End-to-End Cascade Timing
# =====================================================================

def section9_cascade_timing():
    _separator("SECTION 9: End-to-End Cascade Timing")
    _log("  Goal: Measure actual cascade performance L0-L2 + L3 estimate")

    # Quick L0-L1 run
    _log("\n  Running L0...")
    t0 = time.time()
    l0 = run_level0(N_HALF, M, C_TARGET, verbose=False)
    t_l0 = time.time() - t0
    _log(f"    L0: {l0['n_survivors']:,} survivors in {t_l0:.2f}s")

    # L1 timing
    _log("  Running L1...")
    l0_surv = l0['survivors']
    n_half_child = 4
    t0 = time.time()
    l1_survivors = []
    for i in range(len(l0_surv)):
        s, _ = process_parent_fused(l0_surv[i], M, C_TARGET, n_half_child)
        if len(s) > 0:
            l1_survivors.append(s)
    if l1_survivors:
        l1_all = np.vstack(l1_survivors)
        l1_all = _fast_dedup(l1_all)
    else:
        l1_all = np.empty((0, 8), dtype=np.int32)
    t_l1 = time.time() - t0
    _log(f"    L1: {len(l1_all):,} survivors in {t_l1:.2f}s")

    # L2 timing
    _log("  Running L2...")
    n_half_child = 8
    t0 = time.time()
    l2_survivors = []
    for i in range(len(l1_all)):
        s, _ = process_parent_fused(l1_all[i], M, C_TARGET, n_half_child)
        if len(s) > 0:
            l2_survivors.append(s)
    if l2_survivors:
        l2_all = np.vstack(l2_survivors)
        l2_all = _fast_dedup(l2_all)
    else:
        l2_all = np.empty((0, 16), dtype=np.int32)
    t_l2 = time.time() - t0
    _log(f"    L2: {len(l2_all):,} survivors in {t_l2:.2f}s")

    # L3 estimate from sample
    _log("  Estimating L3 (sample of 10 parents)...")
    n_half_child = 16
    n_sample_l3 = min(10, len(l2_all))
    test_l3 = sample_parents(l2_all, n_sample_l3)
    t0 = time.time()
    l3_children = 0
    l3_survived = 0
    for i in range(n_sample_l3):
        s, nc = process_parent_fused(test_l3[i], M, C_TARGET, n_half_child)
        l3_children += nc
        l3_survived += len(s)
    t_l3_sample = time.time() - t0

    if n_sample_l3 > 0:
        per_parent = t_l3_sample / n_sample_l3
        est_l3_total = per_parent * len(l2_all)
        children_per_parent = l3_children / n_sample_l3
        surv_per_parent = l3_survived / n_sample_l3
    else:
        per_parent = 0
        est_l3_total = 0
        children_per_parent = 0
        surv_per_parent = 0

    _log(f"    L3 sample: {l3_children:,} children, {l3_survived:,} survivors")
    _log(f"    L3 per parent: {per_parent*1000:.1f}ms")
    _log(f"    L3 est. total (sequential): {_fmt_time(est_l3_total)}")
    effective = _effective_cpu_count()
    _log(f"    L3 est. total ({effective} workers): "
         f"{_fmt_time(est_l3_total / effective)}")

    _log(f"\n  Summary:")
    _log(f"    L0:           {_fmt_time(t_l0)}")
    _log(f"    L1:           {_fmt_time(t_l1)}")
    _log(f"    L2:           {_fmt_time(t_l2)}")
    _log(f"    L3 (est):     {_fmt_time(est_l3_total / effective)} "
         f"({effective} workers)")

    return {
        "l0_time": t_l0, "l0_surv": l0['n_survivors'],
        "l1_time": t_l1, "l1_surv": len(l1_all),
        "l2_time": t_l2, "l2_surv": len(l2_all),
        "l3_per_parent_ms": per_parent * 1000,
        "l3_children_per_parent": children_per_parent,
        "l3_est_parallel_s": est_l3_total / effective,
    }


# =====================================================================
# Section 10: Cursor-Range Tightening (Idea 1 Effectiveness)
# =====================================================================

def section10_tightening():
    _separator("SECTION 10: Cursor-Range Tightening (Idea 1)")

    if not HAS_BENCH_COMMON:
        _log("  SKIP: bench_common not importable")
        return {}

    _log("  Goal: Measure tightening effectiveness on real parents")

    # Test at L3 (d_child=32)
    parents = load_checkpoint(2)
    if parents is None:
        _log("  SKIP: No L2 checkpoint found.")
        return {}

    n_sample = min(200, len(parents))
    test_parents = sample_parents(parents, n_sample)
    d_child = 32
    n_half_child = 16

    _log(f"  Testing {n_sample} L2 parents -> L3 children (d_child={d_child})")

    # Warm up JIT
    warmup_bench_jit(verbose=False)

    results = {
        "tightened_count": 0,
        "total_original_product": 0,
        "total_tightened_product": 0,
        "eliminated_count": 0,
        "total_values_removed": 0,
        "total_iterations": 0,
        "tighten_time_us": [],
    }

    for i in range(n_sample):
        parent = test_parents[i]
        br = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if br is None:
            continue
        lo, hi, orig_product = br

        tt = build_threshold_table(d_child, M, C_TARGET, n_half_child)
        eo = build_ell_order(d_child)
        pp = build_parent_prefix(parent)

        t0 = time.time()
        new_lo, new_hi, n_removed, n_iters = tighten_cursor_ranges(
            parent, lo, hi, d_child, M, C_TARGET, n_half_child, tt, pp, eo)
        t1 = time.time()

        new_product = compute_product(new_lo, new_hi)

        results["total_original_product"] += orig_product
        results["total_tightened_product"] += new_product
        results["tighten_time_us"].append((t1 - t0) * 1e6)
        results["total_values_removed"] += n_removed
        results["total_iterations"] += n_iters

        if n_removed > 0:
            results["tightened_count"] += 1
        if new_product == 0:
            results["eliminated_count"] += 1

    tightened_pct = results["tightened_count"] / n_sample * 100
    if results["total_original_product"] > 0:
        reduction_pct = (1 - results["total_tightened_product"] /
                          results["total_original_product"]) * 100
    else:
        reduction_pct = 0
    avg_us = np.mean(results["tighten_time_us"]) if results["tighten_time_us"] else 0
    p50_us = np.median(results["tighten_time_us"]) if results["tighten_time_us"] else 0
    p99_us = np.percentile(results["tighten_time_us"], 99) if results["tighten_time_us"] else 0

    _log(f"\n  Results ({n_sample} parents):")
    _log(f"    Parents tightened:   {results['tightened_count']:,} "
         f"({tightened_pct:.1f}%)")
    _log(f"    Parents eliminated:  {results['eliminated_count']:,}")
    _log(f"    Values removed:      {results['total_values_removed']:,}")
    _log(f"    Product reduction:   {reduction_pct:.1f}%")
    _log(f"    Avg tighten time:    {avg_us:.1f}us")
    _log(f"    P50 tighten time:    {p50_us:.1f}us")
    _log(f"    P99 tighten time:    {p99_us:.1f}us")
    _log(f"    Avg iterations:      "
         f"{results['total_iterations']/n_sample:.1f}")

    return {
        "tightened_pct": tightened_pct,
        "product_reduction_pct": reduction_pct,
        "avg_tighten_us": avg_us,
    }


# =====================================================================
# Section 11: Summary & Recommendations
# =====================================================================

def section11_summary(all_results):
    _separator("SECTION 11: Summary & Recommendations", char="*")

    _log("\n  HARDWARE:")
    hw = all_results.get("section1", {})
    _log(f"    CPUs: {hw.get('effective_cpus', '?')} effective / "
         f"{hw.get('logical_cpus', '?')} logical")

    _log("\n  OPTIMAL PARAMETERS:")

    # Worker/thread ratio
    wt = all_results.get("section2", {}).get("worker_thread_results", [])
    if wt:
        best = min(wt, key=lambda r: r["elapsed"])
        _log(f"    Workers:          {best['workers']}")
        _log(f"    Numba threads:    {best['threads']}")
        _log(f"    Speedup vs 1w:    {best['speedup']:.1f}x")

    # Chunksize
    cs = all_results.get("section3", {}).get("chunksize_results", [])
    if cs:
        best = min(cs, key=lambda r: r["elapsed"])
        _log(f"    Chunksize:        {best['chunksize']}")

    # Buffer cap
    bc = all_results.get("section4", {}).get("buffer_results", [])
    if bc:
        zero_rerun = [r for r in bc if r["reruns"] == 0]
        if zero_rerun:
            best = min(zero_rerun, key=lambda r: r["mem_mb"])
            _log(f"    Buffer cap:       {best['buf_cap']:,} "
                 f"({best['mem_mb']:.0f} MB)")

    # Kernel throughput
    kt = all_results.get("section6", {}).get("kernel_throughput", [])
    if kt:
        _log("\n  KERNEL THROUGHPUT:")
        for r in kt:
            _log(f"    L{r['level']}: {r['us_per_child']:.2f}us/child, "
                 f"{r['children_per_sec']:,.0f} children/s, "
                 f"{r['surv_pct']:.1f}% survival")

    # Tightening
    ti = all_results.get("section10", {})
    if ti:
        _log(f"\n  IDEA 1 (Cursor Tightening):")
        _log(f"    Tightened:        {ti.get('tightened_pct', 0):.1f}% of parents")
        _log(f"    Product cut:      {ti.get('product_reduction_pct', 0):.1f}%")
        _log(f"    Cost:             {ti.get('avg_tighten_us', 0):.0f}us/parent")

    # Cascade estimate
    ce = all_results.get("section9", {})
    if ce:
        _log(f"\n  CASCADE TIMING:")
        _log(f"    L0: {_fmt_time(ce.get('l0_time', 0))}, "
             f"{ce.get('l0_surv', 0):,} survivors")
        _log(f"    L1: {_fmt_time(ce.get('l1_time', 0))}, "
             f"{ce.get('l1_surv', 0):,} survivors")
        _log(f"    L2: {_fmt_time(ce.get('l2_time', 0))}, "
             f"{ce.get('l2_surv', 0):,} survivors")
        _log(f"    L3 (est, parallel): "
             f"{_fmt_time(ce.get('l3_est_parallel_s', 0))}")

    _log(f"\n  RECOMMENDED COMMAND:")
    w = best['workers'] if wt else 32
    _log(f"    python -m cloninger-steinerberger.cpu.run_cascade \\")
    _log(f"        --n_half {N_HALF} --m {M} --c_target {C_TARGET} \\")
    _log(f"        --workers {w} --max_levels 10 --resume")

    _log(f"\n  OPTIMIZATION PRIORITIES:")
    _log(f"    1. Implement Idea 1 (cursor tightening) - ~29% fewer children")
    _log(f"    2. Implement Idea 3 (dynamic inner tightening) - ~57% fewer children")
    _log(f"    3. Implement Idea 2 (batched scan) - ~34% faster per-child")
    _log(f"    4. Combined estimate: ~3.5x total speedup")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPU optimization benchmark for RunPod cascade prover")
    parser.add_argument("--quick", action="store_true",
                        help="Quick subset: sections 1, 5, 6, 7 only (~5 min)")
    parser.add_argument("--section", type=int, default=None,
                        help="Run only a specific section (1-11)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    _log(f"{'='*78}")
    _log(f"  CPU OPTIMIZATION BENCHMARK")
    _log(f"  Target: RunPod cpu3c-32-64 (32 vCPU, 64 GB RAM)")
    _log(f"  Parameters: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    _log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"{'='*78}")

    all_results = {}
    t_total = time.time()

    # Define section runners
    sections = {
        1: ("Hardware Detection", lambda: section1_hardware()),
        2: ("Worker/Thread Ratio", lambda: section2_worker_threads(
            all_results.get("section1", {}))),
        3: ("Chunk Size", lambda: section3_chunksize(
            all_results.get("section1", {}))),
        4: ("Buffer Capacity", lambda: section4_buffer_capacity()),
        5: ("Staging Buffer", lambda: section5_staging_buffer()),
        6: ("Kernel Throughput", lambda: section6_kernel_throughput()),
        7: ("Memory Budget", lambda: section7_memory_budget()),
        8: ("NUMA/Threading", lambda: section8_numa()),
        9: ("Cascade Timing", lambda: section9_cascade_timing()),
        10: ("Cursor Tightening", lambda: section10_tightening()),
        11: ("Summary", lambda: section11_summary(all_results)),
    }

    if args.section is not None:
        run_sections = [args.section]
    elif args.quick:
        run_sections = [1, 5, 6, 7, 11]
    else:
        run_sections = list(range(1, 12))

    for s in run_sections:
        if s not in sections:
            _log(f"  Unknown section {s}, skipping")
            continue
        name, runner = sections[s]
        try:
            result = runner()
            if isinstance(result, dict):
                all_results[f"section{s}"] = result
        except Exception as e:
            _log(f"\n  ERROR in section {s} ({name}): {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - t_total
    _log(f"\n{'='*78}")
    _log(f"  BENCHMARK COMPLETE - Total time: {_fmt_time(total_time)}")
    _log(f"{'='*78}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DATA_DIR, f"bench_cpu_opt_{ts}.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        _log(f"  Results saved to {output_path}")
    except Exception as e:
        _log(f"  Could not save results: {e}")


if __name__ == "__main__":
    main()
