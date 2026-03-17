"""Comprehensive algorithm comparison: MATLAB/Octave [CS14] vs Python Naive vs Optimized.

Measures per-parent throughput at cascade levels L2-L4, comparing up to four tiers:
  OCTAVE:       Original CS14 algorithm run in GNU Octave (loaded from .mat files)
  MATLAB-FAITHFUL: Python reimplementation of CS14 dense matmul approach (optional)
  NAIVE:        Materialize all children -> batch-prune (full O(d^2) autoconv each)
  OPTIMIZED:    Fused generate+prune (incremental autoconv, quick-check, subtree pruning)

Produces comparison_results.json consumed by plot_comparison.py.

Estimated times (default settings, 100 L4 parents, 3 passes):
  JIT warmup:              ~3-5 seconds
  L2 (all ~170 parents):   ~1-2 seconds per pass
  L3 (all ~30 parents):    ~1 second per pass
  L4 (100 parents):        ~30s naive + ~3s optimized per pass
  L4 instrumentation:      ~5 seconds
  Total (L2+L3+L4):        ~5-8 minutes
  --quick mode:            ~2-3 minutes
  --levels L4 only:        ~3-5 minutes
  --matlab-faithful adds:  ~30-40 minutes (L4, very slow)

Usage:
    python -m baseline.run_comparison                   # full run (~5-8 min)
    python -m baseline.run_comparison --quick            # fast check (~2-3 min)
    python -m baseline.run_comparison --matlab-faithful  # include Tier 0 (~40 min)
    python -m baseline.run_comparison --levels L4        # single level (~3-5 min)
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import numba
from numba import njit

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

print("Loading modules + JIT warmup...", flush=True)
t_load = time.time()
from cpu.run_cascade import (
    _fused_generate_and_prune,
    _fused_generate_and_prune_gray,
    _fused_generate_and_prune_instrumented,
    _compute_bin_ranges,
)
print(f"  Done in {time.time() - t_load:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M = 20
C_TARGET = 1.4
N_HALF = 2
SEED = 42
DATA_DIR = os.path.join(_root, 'data')
BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASELINE_DIR, 'comparison_results.json')

# Level configurations: input checkpoint -> process to next level
LEVELS = [
    {'name': 'L2', 'checkpoint': 'checkpoint_L1_survivors.npy',
     'd_parent': 8, 'n_half_child': 8, 'full_parents': 48443,
     'sample_size': None},
    {'name': 'L3', 'checkpoint': 'checkpoint_L2_survivors.npy',
     'd_parent': 16, 'n_half_child': 16, 'full_parents': 7499382,
     'sample_size': None},
    {'name': 'L4', 'checkpoint': 'checkpoint_L3_survivors.npy',
     'd_parent': 32, 'n_half_child': 32, 'full_parents': 147279894,
     'sample_size': 100},
]

# Published MATLAB baseline (Cloninger & Steinerberger, 2014)
MATLAB_CPU_HOURS = 20_000
MATLAB_C_TARGET = 1.28

# Measured L0-L3 cascade time (seconds) from checkpoint_meta.json
OUR_L0_L3_ELAPSED = [0.36, 12.04, 28.71, 56768.66]


# =====================================================================
# Naive baseline kernels (same as before — per-child autoconv)
# =====================================================================

@njit(cache=True)
def _generate_all_children(parent_int, lo_arr, hi_arr):
    """Materialize all Cartesian-product children as a dense (N, d_child) array."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    total = 1
    for i in range(d_parent):
        total *= (hi_arr[i] - lo_arr[i] + 1)
    children = np.empty((total, d_child), dtype=np.int32)
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]
    for idx in range(total):
        for i in range(d_parent):
            children[idx, 2 * i] = cursor[i]
            children[idx, 2 * i + 1] = parent_int[i] - cursor[i]
        for i in range(d_parent - 1, -1, -1):
            cursor[i] += 1
            if cursor[i] <= hi_arr[i]:
                break
            cursor[i] = lo_arr[i]
    return children


@njit(cache=True)
def _prune_batch_naive(children, n_half, m, c_target):
    """Batch pruning: full O(d^2) autoconvolution from scratch per child."""
    B = children.shape[0]
    d = children.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    d_minus_1 = d - 1

    for b in range(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(children[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(children[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int32)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int32(children[b, i])

        pruned = False
        for ell in range(2, 2 * d + 1):
            if pruned:
                break
            n_cv = ell - 1
            ell_f = np.float64(ell)
            dbl_ell = dyn_base * ell_f * inv_4n
            two_li = 2.0 * ell_f * inv_4n
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = np.int64(prefix_c[hi_bin + 1]) - np.int64(prefix_c[lo_bin])
                dyn_x = dbl_ell + two_li * np.float64(W_int)
                dyn_it = np.int64(dyn_x * one_minus_4eps)
                if ws > dyn_it:
                    pruned = True
                    break
        if pruned:
            survived[b] = False
    return survived


# =====================================================================
# Per-parent wrappers
# =====================================================================

def _asymmetry_skip(parent_int, m, c_target):
    """Return True if parent is trivially pruned by asymmetry argument."""
    d_parent = len(parent_int)
    threshold = math.sqrt(c_target / 2.0)
    left_sum = sum(int(parent_int[i]) for i in range(d_parent // 2))
    left_frac = left_sum / m
    return left_frac >= threshold or left_frac <= 1.0 - threshold


def process_parent_naive(parent_int, m, c_target, n_half_child):
    """Naive: materialize ALL children -> batch-prune."""
    d_child = 2 * len(parent_int)
    if _asymmetry_skip(parent_int, m, c_target):
        return 0, -1
    result = _compute_bin_ranges(parent_int, m, c_target, d_child)
    if result is None:
        return 0, 0
    lo_arr, hi_arr, total_children = result
    if total_children == 0:
        return 0, 0
    children = _generate_all_children(parent_int, lo_arr, hi_arr)
    survived = _prune_batch_naive(children, n_half_child, m, c_target)
    return int(np.sum(survived)), total_children


def process_parent_optimized(parent_int, m, c_target, n_half_child, out_buf):
    """Optimized: fused generate+prune kernel."""
    d_child = 2 * len(parent_int)
    if _asymmetry_skip(parent_int, m, c_target):
        return 0, -1
    result = _compute_bin_ranges(parent_int, m, c_target, d_child)
    if result is None:
        return 0, 0
    lo_arr, hi_arr, total_children = result
    if total_children == 0:
        return 0, 0
    n_surv, _ = _fused_generate_and_prune_gray(
        parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf)
    return int(n_surv), total_children


# =====================================================================
# Octave results loading
# =====================================================================

def load_octave_results(level_name):
    """Load Octave benchmark results for a given level, if available."""
    try:
        import scipy.io
    except ImportError:
        return None
    path = os.path.join(BASELINE_DIR, f'octave_results_{level_name}.mat')
    if not os.path.exists(path):
        return None
    data = scipy.io.loadmat(path)
    return {
        'per_parent_times_s': data['per_parent_times'].flatten().astype(float),
        'per_parent_survivors': data['per_parent_survivors'].flatten().astype(int),
        'per_parent_children': data['per_parent_children'].flatten().astype(int),
        'n_parents': int(data['n_parents'].flat[0]),
        'd_parent': int(data['d_parent'].flat[0]),
        'total_elapsed': float(data['total_elapsed'].flat[0]),
    }


# =====================================================================
# JIT warmup
# =====================================================================

def warmup_jit():
    """Compile all Numba kernels on small data before timed runs."""
    print("  Warming up naive + optimized kernels...", flush=True)
    for dp in [4, 8, 16, 32]:
        dc = 2 * dp
        nhc = dp
        parent = np.zeros(dp, dtype=np.int32)
        parent[0] = M
        lo = np.zeros(dp, dtype=np.int32)
        hi = np.zeros(dp, dtype=np.int32)
        hi[0] = min(M, 3)
        children = _generate_all_children(parent, lo, hi)
        _prune_batch_naive(children, nhc, M, C_TARGET)
        buf = np.empty((100, dc), dtype=np.int32)
        _fused_generate_and_prune_gray(parent, nhc, M, C_TARGET, lo, hi, buf)
        _fused_generate_and_prune(parent, nhc, M, C_TARGET, lo, hi, buf)
        _fused_generate_and_prune_instrumented(parent, nhc, M, C_TARGET, lo, hi, buf)
    print("  JIT warmup complete.", flush=True)


# =====================================================================
# Experiment 1: Per-level throughput comparison
# =====================================================================

def load_sample(level_cfg, seed=SEED):
    """Load and sample parents for a given level."""
    path = os.path.join(DATA_DIR, level_cfg['checkpoint'])
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.", flush=True)
        return None
    parents = np.load(path, mmap_mode='r')
    n_total = len(parents)
    sample_size = level_cfg['sample_size']
    if sample_size is None or sample_size >= n_total:
        result = np.array(parents)
        del parents
        print(f"  Using all {n_total} parents (d={level_cfg['d_parent']})", flush=True)
        return result
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=sample_size, replace=False)
    indices.sort()
    sample = np.array(parents[indices])
    del parents
    print(f"  Sampled {len(sample)} of {n_total:,} parents (d={level_cfg['d_parent']})",
          flush=True)
    return sample


def run_level_experiment(level_cfg, sample, n_passes=3,
                         run_matlab_faithful=False):
    """Run naive, optimized (and optionally MATLAB-faithful) on same parents."""
    name = level_cfg['name']
    nhc = level_cfg['n_half_child']
    d_child = 2 * level_cfg['d_parent']
    n = len(sample)

    # Time estimates per parent (single-core, rough):
    #   L2 (d=16): ~0.5ms naive, ~0.4ms optimized
    #   L3 (d=32): ~5ms naive, ~4ms optimized
    #   L4 (d=64): ~300ms naive, ~30ms optimized
    est_naive_s = {16: 0.001, 32: 0.01, 64: 0.3}.get(d_child, 0.1)
    est_opt_s = {16: 0.001, 32: 0.01, 64: 0.03}.get(d_child, 0.01)
    est_total = n * (est_naive_s + est_opt_s) * n_passes + n * 0.5  # +verify
    if run_matlab_faithful and d_child >= 64:
        est_total += n * 35  # ~35s per L4 parent for MATLAB-faithful
    est_min = est_total / 60
    if est_min >= 1:
        print(f"  {name}: estimated time ~{est_min:.0f} min "
              f"({n} parents, {n_passes} passes)", flush=True)
    else:
        print(f"  {name}: estimated time ~{est_total:.0f}s "
              f"({n} parents, {n_passes} passes)", flush=True)

    buf_cap = 500_000 if d_child >= 64 else 5_000_000
    out_buf = np.empty((buf_cap, d_child), dtype=np.int32)

    # --- Warmup pass ---
    print(f"  {name}: warmup pass...", end='', flush=True)
    for i in range(min(n, 20)):
        process_parent_naive(sample[i], M, C_TARGET, nhc)
        process_parent_optimized(sample[i], M, C_TARGET, nhc, out_buf)
    print(" done", flush=True)

    # --- Correctness verification ---
    print(f"  {name}: verifying correctness...", end='', flush=True)
    mismatches = 0
    for i in range(n):
        sn, cn = process_parent_naive(sample[i], M, C_TARGET, nhc)
        so, co = process_parent_optimized(sample[i], M, C_TARGET, nhc, out_buf)
        if sn != so:
            mismatches += 1
            if mismatches <= 3:
                print(f"\n    MISMATCH parent {i}: naive={sn}, opt={so}", flush=True)
    if mismatches == 0:
        print(f" OK (all {n} match)", flush=True)
    else:
        print(f" FAIL ({mismatches}/{n} mismatches!)", flush=True)

    # --- Timed passes: Naive ---
    naive_pass_walls = []
    naive_per_parent = None
    naive_children = None
    naive_survivors = None

    for p in range(n_passes):
        times = np.empty(n)
        children_arr = np.empty(n, dtype=np.int64)
        survivors_arr = np.empty(n, dtype=np.int64)
        gc.collect()
        gc.disable()
        t_wall = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            ns, nc = process_parent_naive(sample[i], M, C_TARGET, nhc)
            times[i] = time.perf_counter() - t0
            children_arr[i] = nc
            survivors_arr[i] = ns
        wall = time.perf_counter() - t_wall
        gc.enable()
        naive_pass_walls.append(wall)
        if naive_per_parent is None:
            naive_per_parent = times.copy()
            naive_children = children_arr.copy()
            naive_survivors = survivors_arr.copy()
        print(f"    Naive pass {p+1}/{n_passes}: {wall:.3f}s "
              f"({n/wall:.1f} parents/sec)", flush=True)

    naive_median_idx = int(np.argsort(naive_pass_walls)[len(naive_pass_walls) // 2])
    naive_wall = naive_pass_walls[naive_median_idx]

    # --- Timed passes: Optimized ---
    opt_pass_walls = []
    opt_per_parent = None

    for p in range(n_passes):
        times = np.empty(n)
        gc.collect()
        gc.disable()
        t_wall = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            process_parent_optimized(sample[i], M, C_TARGET, nhc, out_buf)
            times[i] = time.perf_counter() - t0
        wall = time.perf_counter() - t_wall
        gc.enable()
        opt_pass_walls.append(wall)
        if opt_per_parent is None:
            opt_per_parent = times.copy()
        print(f"    Optimized pass {p+1}/{n_passes}: {wall:.3f}s "
              f"({n/wall:.1f} parents/sec)", flush=True)

    opt_median_idx = int(np.argsort(opt_pass_walls)[len(opt_pass_walls) // 2])
    opt_wall = opt_pass_walls[opt_median_idx]

    # --- Optional: MATLAB-faithful tier ---
    mf_data = None
    if run_matlab_faithful:
        try:
            from baseline.matlab_faithful import CS14Faithful
        except ImportError:
            sys.path.insert(0, BASELINE_DIR)
            from matlab_faithful import CS14Faithful

        mf_est_min = n * 35 / 60  # ~35s per L4 parent
        print(f"  {name}: MATLAB-faithful tier (~{mf_est_min:.0f} min for {n} parents)...",
              flush=True)
        cs14 = CS14Faithful(d_child, C_TARGET, M)
        mf_times = np.empty(n)
        mf_survivors = np.empty(n, dtype=np.int64)
        mf_children_arr = np.empty(n, dtype=np.int64)

        # Convert to continuous weights for MATLAB-faithful
        mf_t0 = time.perf_counter()
        for i in range(n):
            bin_weights = sample[i].astype(np.float64) / M
            ns, nc, et = cs14.process_parent(bin_weights)
            mf_times[i] = et
            mf_survivors[i] = ns
            mf_children_arr[i] = nc
            elapsed_mf = time.perf_counter() - mf_t0
            avg_mf = elapsed_mf / (i + 1)
            eta_mf = avg_mf * (n - i - 1) / 60
            if (i + 1) % 5 == 0 or i == 0:
                print(f"    Parent {i+1}/{n}: {et:.1f}s  "
                      f"(avg {avg_mf:.1f}s, ETA {eta_mf:.1f} min)", flush=True)

        mf_wall = float(np.sum(mf_times))
        mf_data = {
            'wall_seconds': round(mf_wall, 4),
            'parents_per_sec': round(n / mf_wall, 2),
            'per_parent_mean_ms': round(float(np.mean(mf_times)) * 1000, 3),
            'per_parent_median_ms': round(float(np.median(mf_times)) * 1000, 3),
            'per_parent_p5_ms': round(float(np.percentile(mf_times, 5)) * 1000, 3),
            'per_parent_p95_ms': round(float(np.percentile(mf_times, 95)) * 1000, 3),
            'per_parent_times_ms': [round(t * 1000, 3) for t in mf_times.tolist()],
            'pass_walls': [round(mf_wall, 4)],
        }
        print(f"    MATLAB-faithful: {mf_wall:.3f}s "
              f"({n/mf_wall:.1f} parents/sec)", flush=True)

    # --- Load Octave results ---
    octave_data = load_octave_results(name)
    octave_result = None
    if octave_data is not None:
        ot = octave_data['per_parent_times_s']
        n_oct = len(ot)
        oct_wall = float(np.sum(ot))
        octave_result = {
            'wall_seconds': round(oct_wall, 4),
            'parents_per_sec': round(n_oct / oct_wall, 2),
            'per_parent_mean_ms': round(float(np.mean(ot)) * 1000, 3),
            'per_parent_median_ms': round(float(np.median(ot)) * 1000, 3),
            'per_parent_p5_ms': round(float(np.percentile(ot, 5)) * 1000, 3),
            'per_parent_p95_ms': round(float(np.percentile(ot, 95)) * 1000, 3),
            'per_parent_times_ms': [round(t * 1000, 3) for t in ot.tolist()],
            'n_parents': n_oct,
            'total_survivors': int(np.sum(octave_data['per_parent_survivors'])),
            'total_children': int(np.sum(octave_data['per_parent_children'])),
        }
        print(f"    Octave [CS14] loaded: {n_oct} parents, "
              f"{oct_wall:.1f}s total, "
              f"{n_oct/oct_wall:.2f} parents/sec", flush=True)

    # --- Assemble result ---
    total_children = int(np.sum(naive_children))
    total_survivors = int(np.sum(naive_survivors))

    nontrivial = (naive_per_parent > 1e-6) & (opt_per_parent > 1e-7)
    if np.any(nontrivial):
        per_parent_speedup = (naive_per_parent[nontrivial] /
                              np.maximum(opt_per_parent[nontrivial], 1e-9))
    else:
        per_parent_speedup = np.array([1.0])

    result = {
        'name': name,
        'd_parent': level_cfg['d_parent'],
        'd_child': d_child,
        'n_half_child': nhc,
        'n_parents_sampled': n,
        'n_parents_total': level_cfg['full_parents'],
        'total_children': total_children,
        'total_survivors': total_survivors,
        'correctness_verified': mismatches == 0,
        'naive': {
            'wall_seconds': round(naive_wall, 4),
            'parents_per_sec': round(n / naive_wall, 2),
            'children_per_sec': round(total_children / naive_wall, 0),
            'per_parent_mean_ms': round(float(np.mean(naive_per_parent)) * 1000, 3),
            'per_parent_median_ms': round(float(np.median(naive_per_parent)) * 1000, 3),
            'per_parent_p5_ms': round(float(np.percentile(naive_per_parent, 5)) * 1000, 3),
            'per_parent_p95_ms': round(float(np.percentile(naive_per_parent, 95)) * 1000, 3),
            'pass_walls': [round(w, 4) for w in naive_pass_walls],
            'per_parent_times_ms': [round(t * 1000, 3) for t in naive_per_parent.tolist()],
        },
        'optimized': {
            'wall_seconds': round(opt_wall, 4),
            'parents_per_sec': round(n / opt_wall, 2),
            'children_per_sec': round(total_children / opt_wall, 0),
            'per_parent_mean_ms': round(float(np.mean(opt_per_parent)) * 1000, 3),
            'per_parent_median_ms': round(float(np.median(opt_per_parent)) * 1000, 3),
            'per_parent_p5_ms': round(float(np.percentile(opt_per_parent, 5)) * 1000, 3),
            'per_parent_p95_ms': round(float(np.percentile(opt_per_parent, 95)) * 1000, 3),
            'pass_walls': [round(w, 4) for w in opt_pass_walls],
            'per_parent_times_ms': [round(t * 1000, 3) for t in opt_per_parent.tolist()],
        },
        'speedup': {
            'wall_time': round(naive_wall / opt_wall, 2),
            'per_parent_median': round(float(np.median(per_parent_speedup)), 2),
            'per_parent_mean': round(float(np.mean(per_parent_speedup)), 2),
            'per_parent_p25': round(float(np.percentile(per_parent_speedup, 25)), 2),
            'per_parent_p75': round(float(np.percentile(per_parent_speedup, 75)), 2),
        },
    }

    if octave_result is not None:
        result['octave'] = octave_result
        # Compute Octave -> Optimized speedup
        oct_median_ms = octave_result['per_parent_median_ms']
        opt_median_ms = result['optimized']['per_parent_median_ms']
        if opt_median_ms > 0:
            result['speedup']['octave_vs_optimized'] = round(oct_median_ms / opt_median_ms, 1)

    if mf_data is not None:
        result['matlab_faithful'] = mf_data

    return result


# =====================================================================
# Experiment 2: Instrumentation analysis (L4 only)
# =====================================================================

def run_instrumentation(sample, n_half_child=32):
    """Run instrumented kernel on L4 parents."""
    d_parent = sample.shape[1]
    d_child = 2 * d_parent
    n = len(sample)

    totals = {
        'n_surv': 0, 'n_fast': 0, 'n_short': 0, 'n_deep': 0,
        'n_subtree_success': 0, 'n_subtree_children_skipped': 0,
        'n_qc_hit': 0, 'n_full_scan': 0, 'n_visited': 0,
        'n_cartesian': 0, 'n_asym_skipped': 0,
    }

    print(f"\n  Instrumented run on {n} L4 parents...", flush=True)
    t0 = time.time()
    for i in range(n):
        result = _compute_bin_ranges(sample[i], M, C_TARGET, d_child)
        if result is None:
            continue
        lo_arr, hi_arr, total_children = result
        if total_children == 0:
            continue
        totals['n_cartesian'] += total_children
        buf = np.empty((min(total_children, 500_000), d_child), dtype=np.int32)
        stats = _fused_generate_and_prune_instrumented(
            sample[i], n_half_child, M, C_TARGET, lo_arr, hi_arr, buf)
        n_surv = stats[0]
        if stats[8] == 0:
            totals['n_asym_skipped'] += 1
        totals['n_surv'] += int(n_surv)
        totals['n_fast'] += int(stats[1])
        totals['n_short'] += int(stats[2])
        totals['n_deep'] += int(stats[3])
        totals['n_subtree_success'] += int(stats[4])
        totals['n_subtree_children_skipped'] += int(stats[5])
        totals['n_qc_hit'] += int(stats[6])
        totals['n_full_scan'] += int(stats[7])
        totals['n_visited'] += int(stats[8])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)

    cart = totals['n_cartesian']
    vis = totals['n_visited']
    n_advances = totals['n_fast'] + totals['n_short'] + totals['n_deep']

    return {
        'n_parents': n,
        'elapsed_seconds': round(elapsed, 1),
        'total_cartesian': cart,
        'total_visited': vis,
        'total_advances': n_advances,
        'survivors': totals['n_surv'],
        'asym_skipped_parents': totals['n_asym_skipped'],
        'carry_paths': {
            'fast_n1': totals['n_fast'],
            'fast_n1_pct': round(100 * totals['n_fast'] / max(1, n_advances), 1),
            'short_carry': totals['n_short'],
            'short_carry_pct': round(100 * totals['n_short'] / max(1, n_advances), 1),
            'deep_carry': totals['n_deep'],
            'deep_carry_pct': round(100 * totals['n_deep'] / max(1, n_advances), 1),
        },
        'subtree_pruning': {
            'prunes_succeeded': totals['n_subtree_success'],
            'children_skipped': totals['n_subtree_children_skipped'],
            'skip_pct_of_cartesian': round(
                100 * totals['n_subtree_children_skipped'] / max(1, cart), 1),
        },
        'quick_check': {
            'hits': totals['n_qc_hit'],
            'hit_pct_of_visited': round(100 * totals['n_qc_hit'] / max(1, vis), 1),
            'full_scans': totals['n_full_scan'],
            'full_scan_pct': round(100 * totals['n_full_scan'] / max(1, vis), 1),
        },
    }


# =====================================================================
# Experiment 3: End-to-end projection
# =====================================================================

def compute_end_to_end(level_results):
    """Compute projected end-to-end proof times."""
    our_l0_l3_hours = sum(OUR_L0_L3_ELAPSED) / 3600

    l4 = next((r for r in level_results if r['name'] == 'L4'), None)
    if l4 is None:
        return None

    opt_rate = l4['optimized']['parents_per_sec']
    naive_rate = l4['naive']['parents_per_sec']
    total_l4_parents = l4['n_parents_total']

    projections = {}
    for label, cores in [('single_core', 1), ('cloud_48', 48),
                         ('cloud_96', 96), ('cloud_196', 196)]:
        opt_hours = total_l4_parents / (opt_rate * cores * 3600)
        naive_hours = total_l4_parents / (naive_rate * cores * 3600)
        projections[label] = {
            'cores': cores,
            'optimized_l4_hours': round(opt_hours, 1),
            'naive_l4_hours': round(naive_hours, 1),
        }

    ee = {
        'matlab': {
            'cpu_hours': MATLAB_CPU_HOURS,
            'c_target': MATLAB_C_TARGET,
            'note': 'Published figure from Cloninger & Steinerberger (2014); '
                    'GPU-accelerated, 3 parallel workers.',
        },
        'ours': {
            'c_target': C_TARGET,
            'l0_l3_hours': round(our_l0_l3_hours, 2),
            'l0_l3_source': 'measured (single machine, multiprocessing)',
            'l4_optimized_single_core_hours': projections['single_core']['optimized_l4_hours'],
            'l4_naive_single_core_hours': projections['single_core']['naive_l4_hours'],
        },
        'projections': projections,
        'speedup_vs_naive_single_core':
            round(projections['single_core']['naive_l4_hours'] /
                  max(0.01, projections['single_core']['optimized_l4_hours']), 1),
    }

    # Add Octave projection if available
    if 'octave' in l4:
        oct = l4['octave']
        oct_rate = oct['parents_per_sec']
        for label, cores in [('single_core', 1), ('cloud_48', 48),
                             ('cloud_96', 96), ('cloud_196', 196)]:
            oct_hours = total_l4_parents / (oct_rate * cores * 3600)
            projections[label]['octave_l4_hours'] = round(oct_hours, 1)
        ee['ours']['l4_octave_single_core_hours'] = \
            projections['single_core']['octave_l4_hours']
        ee['speedup_vs_octave_single_core'] = \
            round(projections['single_core'].get('octave_l4_hours', 1) /
                  max(0.01, projections['single_core']['optimized_l4_hours']), 1)

    return ee


# =====================================================================
# Console report
# =====================================================================

def print_report(level_results, instrumentation, end_to_end):
    """Print a formatted summary."""
    print(f"\n{'='*72}")
    print(f"  ALGORITHM COMPARISON RESULTS")
    print(f"{'='*72}")
    print(f"  Parameters: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"  Tiers: Octave [CS14] | Python Naive | Python Optimized\n")

    # Header
    has_octave = any('octave' in r for r in level_results)
    has_mf = any('matlab_faithful' in r for r in level_results)

    hdr = f"  {'Level':<6} {'d':<10}"
    if has_octave:
        hdr += f" {'Octave':>14}"
    hdr += f" {'Naive':>14} {'Optimized':>14} {'Speedup':>10}"
    if has_octave:
        hdr += f" {'vs Octave':>10}"
    print(hdr)

    units = f"  {'':6} {'':10}"
    if has_octave:
        units += f" {'(parents/s)':>14}"
    units += f" {'(parents/s)':>14} {'(parents/s)':>14} {'(N/O)':>10}"
    if has_octave:
        units += f" {'(Oct/O)':>10}"
    print(units)
    print(f"  {'-'*68}")

    for r in level_results:
        line = (f"  {r['name']:<6} {r['d_parent']}->{r['d_child']:<6} ")
        if has_octave:
            if 'octave' in r:
                line += f"{r['octave']['parents_per_sec']:>13.1f} "
            else:
                line += f"{'—':>13} "
        line += (f"{r['naive']['parents_per_sec']:>13.1f} "
                 f"{r['optimized']['parents_per_sec']:>13.1f} "
                 f"{r['speedup']['wall_time']:>9.1f}x")
        if has_octave and 'octave' in r:
            sp = r['speedup'].get('octave_vs_optimized', 0)
            line += f" {sp:>9.0f}x"
        print(line)

    if has_mf:
        print(f"\n  MATLAB-faithful tier (cross-check):")
        for r in level_results:
            if 'matlab_faithful' in r:
                mf = r['matlab_faithful']
                print(f"    {r['name']}: {mf['parents_per_sec']:.1f} parents/sec "
                      f"({mf['per_parent_median_ms']:.1f} ms median)")

    if instrumentation:
        inst = instrumentation
        print(f"\n  Optimization Breakdown (L4, {inst['n_parents']} parents):")
        cp = inst['carry_paths']
        print(f"    Fast-path carries (n_changed=1):  {cp['fast_n1_pct']:>5.1f}%")
        print(f"    Short carries (2<=n<=thr):        {cp['short_carry_pct']:>5.1f}%")
        print(f"    Deep carries (n>thr):             {cp['deep_carry_pct']:>5.1f}%")
        sp = inst['subtree_pruning']
        print(f"    Subtree prune skip rate:          {sp['skip_pct_of_cartesian']:>5.1f}% of Cartesian")
        qc = inst['quick_check']
        print(f"    Quick-check hit rate:             {qc['hit_pct_of_visited']:>5.1f}% of visited")

    if end_to_end:
        ee = end_to_end
        print(f"\n  End-to-End Projection:")
        print(f"    MATLAB [CS14]:  ~{ee['matlab']['cpu_hours']:,} CPU-hours  "
              f"(c >= {ee['matlab']['c_target']})")
        proj = ee['projections']
        if 'octave_l4_hours' in proj.get('cloud_196', {}):
            print(f"    Octave (196 cores): "
                  f"{proj['cloud_196']['octave_l4_hours']:.1f}h L4 + "
                  f"{ee['ours']['l0_l3_hours']:.1f}h L0-L3  (c >= {ee['ours']['c_target']})")
        print(f"    Ours (196 cores, optimized): "
              f"{proj['cloud_196']['optimized_l4_hours']:.1f}h L4 + "
              f"{ee['ours']['l0_l3_hours']:.1f}h L0-L3  (c >= {ee['ours']['c_target']})")
        print(f"    Ours (196 cores, naive):     "
              f"{proj['cloud_196']['naive_l4_hours']:.1f}h L4 + "
              f"{ee['ours']['l0_l3_hours']:.1f}h L0-L3")
        if 'speedup_vs_octave_single_core' in ee:
            print(f"    Speedup vs Octave (L4):     "
                  f"{ee['speedup_vs_octave_single_core']:.1f}x")
        print(f"    Speedup vs naive  (L4):     "
              f"{ee['speedup_vs_naive_single_core']:.1f}x")

    print(f"{'='*72}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Algorithm comparison experiment')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer parents, fewer passes')
    parser.add_argument('--passes', type=int, default=3,
                        help='Number of timed passes per approach (default: 3)')
    parser.add_argument('--levels', type=str, default='L2,L3,L4',
                        help='Comma-separated levels to benchmark (default: L2,L3,L4)')
    parser.add_argument('--matlab-faithful', action='store_true',
                        help='Include MATLAB-faithful Python tier (slow)')
    args = parser.parse_args()

    if args.quick:
        args.passes = 2
        for lev in LEVELS:
            if lev['sample_size'] is not None:
                lev['sample_size'] = min(lev['sample_size'], 30)
            else:
                lev['sample_size'] = 50
        print(">>> QUICK MODE: reduced sample sizes + passes <<<", flush=True)
        print(">>> Estimated time: ~2-3 minutes <<<\n", flush=True)
    else:
        if args.matlab_faithful:
            print(">>> Estimated time: ~40 minutes (MATLAB-faithful is slow) <<<\n",
                  flush=True)
        else:
            print(f">>> Estimated time: ~5-8 minutes "
                  f"(levels={args.levels}, passes={args.passes}) <<<\n", flush=True)

    requested = set(args.levels.split(','))

    numba.set_num_threads(1)
    print(f"Numba threads: {numba.get_num_threads()}", flush=True)

    warmup_jit()

    # --- Experiment 1: per-level comparison ---
    level_results = []
    l4_sample = None
    for lev in LEVELS:
        if lev['name'] not in requested:
            continue
        print(f"\n--- {lev['name']}: d={lev['d_parent']} -> {2*lev['d_parent']} ---",
              flush=True)
        sample = load_sample(lev)
        if sample is None:
            continue
        if lev['name'] == 'L4':
            l4_sample = sample
        result = run_level_experiment(lev, sample, n_passes=args.passes,
                                      run_matlab_faithful=args.matlab_faithful)
        level_results.append(result)

    # --- Experiment 2: instrumentation (L4 only) ---
    instrumentation = None
    if l4_sample is not None:
        instrumentation = run_instrumentation(l4_sample)

    # --- Experiment 3: end-to-end ---
    end_to_end = compute_end_to_end(level_results)

    # --- Report ---
    print_report(level_results, instrumentation, end_to_end)

    # --- Save ---
    output = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {'n_half': N_HALF, 'm': M, 'c_target': C_TARGET},
            'numba_threads': 1,
            'n_passes': args.passes,
            'seed': SEED,
        },
        'levels': level_results,
        'instrumentation': instrumentation,
        'end_to_end': end_to_end,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
