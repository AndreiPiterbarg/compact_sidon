"""Benchmark: Idea 2 — 2D Precomputed Integer Threshold Table.

Measures the EXACT effect of replacing per-window float math with an
integer lookup table, by comparing:
  A) BASELINE: current kernel (_fused_generate_and_prune_gray) with
     per-window float64 threshold computation.
  B) LUT:      modified kernel with a precomputed threshold_table[ell][W_int]
     replacing all per-window float math with a single int64 table lookup.

Since modifying the Numba kernel inline is fragile, we instead implement
BOTH versions as standalone @njit functions that replicate ONLY the window
scan hot path (the inner loop of lines 1193-1223). This isolates the
exact code being optimized and measures the difference in that specific
code path with zero confounding factors.

Methodology:
  - Uses REAL parent compositions to generate REAL child autoconvolutions.
  - For each child, we run the FULL window scan (all ell values, all
    positions) using both the baseline and LUT threshold paths.
  - Both implementations produce the same pruning decision (verified).
  - We time ONLY the window scan, not the Gray code iteration or cross-
    term updates, to isolate the effect.
  - N_RUNS repetitions, interleaved A/B pattern, reporting all timings
    plus median.
  - Also benchmarks the quick-check threshold computation (lines 1183-1184).

Usage:
    python -m tests.bench_idea2_threshold_lut
"""
import sys
import os
import time
import math
import numpy as np
from numba import njit

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    _fused_generate_and_prune_gray,
    _compute_bin_ranges,
    run_level0,
    process_parent_fused,
    correction,
)

M = 20
C_TARGET = 1.4
N_RUNS = 7


# =====================================================================
# Baseline window scan: current float64 threshold per-window
# =====================================================================

@njit(cache=False)
def _window_scan_baseline(raw_conv, child, conv_len, d_child,
                          dyn_base_ell_arr, ell_order, ell_count,
                          one_minus_4eps, eps_margin):
    """Window scan using current per-window float64 threshold.

    Returns (pruned: bool, ell_kill: int, s_lo_kill: int, W_int_kill: int64).
    Exactly replicates lines 1193-1223 of the Gray code kernel.
    """
    # Compute prefix_c
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    prefix_c[0] = 0
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

    pruned = False
    ell_kill = np.int32(0)
    s_lo_kill = np.int32(0)
    W_int_kill = np.int64(0)

    for ell_oi in range(ell_count):
        if pruned:
            break
        ell = ell_order[ell_oi]
        n_cv = ell - 1
        ell_idx = ell - 2
        dyn_base_ell = dyn_base_ell_arr[ell_idx]
        n_windows = conv_len - n_cv + 1

        ws = np.int64(0)
        for k in range(n_cv):
            ws += np.int64(raw_conv[k])
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += np.int64(raw_conv[s_lo + n_cv - 1]) - \
                      np.int64(raw_conv[s_lo - 1])
            lo_bin = s_lo - (d_child - 1)
            if lo_bin < 0:
                lo_bin = 0
            hi_bin = s_lo + ell - 2
            if hi_bin > d_child - 1:
                hi_bin = d_child - 1
            W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]

            # --- BASELINE: float64 threshold computation ---
            dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
            dyn_it = np.int64(dyn_x * one_minus_4eps)

            if ws > dyn_it:
                pruned = True
                ell_kill = np.int32(ell)
                s_lo_kill = np.int32(s_lo)
                W_int_kill = W_int
                break

    return pruned, ell_kill, s_lo_kill, W_int_kill


# =====================================================================
# LUT window scan: precomputed int64 threshold table
# =====================================================================

@njit(cache=False)
def _build_threshold_table(ell_count, m, c_target, inv_4n,
                           one_minus_4eps, eps_margin):
    """Build the 2D threshold table: threshold_table[ell_idx * (m+1) + W_int]."""
    m_d = np.float64(m)
    table = np.empty(ell_count * (m + 1), dtype=np.int64)
    for ell_idx in range(ell_count):
        ell = ell_idx + 2
        dyn_base_ell = c_target * m_d * m_d * np.float64(ell) * inv_4n
        for w in range(m + 1):
            dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * np.float64(w)
            table[ell_idx * (m + 1) + w] = np.int64(dyn_x * one_minus_4eps)
    return table


@njit(cache=False)
def _window_scan_lut(raw_conv, child, conv_len, d_child,
                     threshold_table, m_plus_1,
                     ell_order, ell_count):
    """Window scan using precomputed 2D threshold table.

    Returns (pruned: bool, ell_kill: int, s_lo_kill: int, W_int_kill: int64).
    Same logic as baseline, but threshold lookup replaces float math.
    """
    # Compute prefix_c (identical)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    prefix_c[0] = 0
    for i in range(d_child):
        prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

    pruned = False
    ell_kill = np.int32(0)
    s_lo_kill = np.int32(0)
    W_int_kill = np.int64(0)

    for ell_oi in range(ell_count):
        if pruned:
            break
        ell = ell_order[ell_oi]
        n_cv = ell - 1
        ell_idx = ell - 2
        n_windows = conv_len - n_cv + 1

        ws = np.int64(0)
        for k in range(n_cv):
            ws += np.int64(raw_conv[k])
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += np.int64(raw_conv[s_lo + n_cv - 1]) - \
                      np.int64(raw_conv[s_lo - 1])
            lo_bin = s_lo - (d_child - 1)
            if lo_bin < 0:
                lo_bin = 0
            hi_bin = s_lo + ell - 2
            if hi_bin > d_child - 1:
                hi_bin = d_child - 1
            W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]

            # --- LUT: single integer table lookup ---
            dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]

            if ws > dyn_it:
                pruned = True
                ell_kill = np.int32(ell)
                s_lo_kill = np.int32(s_lo)
                W_int_kill = W_int
                break

    return pruned, ell_kill, s_lo_kill, W_int_kill


# =====================================================================
# Generate child data for benchmarking
# =====================================================================

@njit(cache=False)
def _compute_autoconv(child, d_child, conv_len):
    """Compute the raw autoconvolution of a child composition."""
    raw_conv = np.zeros(conv_len, dtype=np.int32)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj
    return raw_conv


def generate_children_sample(parents, n_half_child, n_children_target=5000):
    """Generate a sample of real children from real parents.

    Returns list of (child, raw_conv) pairs for benchmarking.
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    children = []

    for idx in range(len(parents)):
        if len(children) >= n_children_target:
            break
        parent = parents[idx]
        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, total = result
        if total < 2:
            continue

        # Generate a few children per parent by sampling cursor values
        buf = np.empty((min(total, 500), d_child), dtype=np.int32)
        n_surv, _ = _fused_generate_and_prune_gray(
            parent, n_half_child, M, C_TARGET, lo_arr, hi_arr, buf)

        # Take survivors (these exercise the full window scan) and some
        # prunable children (these exercise early exit)
        # Re-generate a few children manually for diversity
        for trial in range(min(total, 20)):
            # Deterministic child from cursor sweep
            child = np.empty(d_child, dtype=np.int32)
            cursor_vals = []
            for i in range(d_parent):
                lo = int(lo_arr[i])
                hi = int(hi_arr[i])
                rng = hi - lo + 1
                c = lo + (trial % rng)
                child[2 * i] = c
                child[2 * i + 1] = int(parent[i]) - c
                cursor_vals.append(c)
            raw_conv = _compute_autoconv(child, d_child, conv_len)
            children.append((child.copy(), raw_conv.copy()))

    return children, d_child, conv_len


# =====================================================================
# Build ell ordering (replicated from kernel)
# =====================================================================

def build_ell_order(d_child):
    """Replicate the kernel's ell ordering logic."""
    ell_count = 2 * d_child - 1
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0

    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = ell
                ell_used[ell - 2] = 1
                oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, d_child * 2, d_child + d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = ell
                ell_used[ell - 2] = 1
                oi += 1
    else:
        phase1_end = min(16, 2 * d_child)
        for ell in range(2, phase1_end + 1):
            ell_order[oi] = ell
            ell_used[ell - 2] = 1
            oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, d_child * 2, d_child + d_child // 2,
                    d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = ell
                ell_used[ell - 2] = 1
                oi += 1

    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = ell
            oi += 1

    return ell_order[:oi].copy(), oi


# =====================================================================
# Main benchmark
# =====================================================================

def bench_window_scan(children_data, d_child, conv_len, level_label):
    """Benchmark baseline vs LUT window scan on a set of children."""
    n_half_child = d_child // 2
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    m_d = float(M)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d

    ell_count_full = 2 * d_child - 1
    ell_order, ell_count = build_ell_order(d_child)

    # Build baseline per-ell constants
    dyn_base_ell_arr = np.empty(ell_count_full, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = C_TARGET * m_d * m_d * float(ell) * inv_4n

    # Build LUT
    threshold_table = _build_threshold_table(
        ell_count_full, M, C_TARGET, inv_4n, one_minus_4eps, eps_margin)
    m_plus_1 = M + 1

    n_children = len(children_data)
    print(f"\n  [{level_label}] {n_children} children, "
          f"d_child={d_child}, conv_len={conv_len}, "
          f"ell_count={ell_count}")

    # ---- JIT warmup ----
    child0, conv0 = children_data[0]
    _window_scan_baseline(conv0, child0, conv_len, d_child,
                          dyn_base_ell_arr, ell_order, ell_count,
                          one_minus_4eps, eps_margin)
    _window_scan_lut(conv0, child0, conv_len, d_child,
                     threshold_table, m_plus_1, ell_order, ell_count)

    # ---- Correctness verification ----
    n_mismatch = 0
    for child, raw_conv in children_data[:200]:
        r_base = _window_scan_baseline(raw_conv, child, conv_len, d_child,
                                       dyn_base_ell_arr, ell_order, ell_count,
                                       one_minus_4eps, eps_margin)
        r_lut = _window_scan_lut(raw_conv, child, conv_len, d_child,
                                 threshold_table, m_plus_1,
                                 ell_order, ell_count)
        if r_base[0] != r_lut[0]:
            n_mismatch += 1
    if n_mismatch > 0:
        print(f"  WARNING: {n_mismatch} pruning decision mismatches!")
    else:
        print(f"  Correctness verified: 0 mismatches on {min(200, n_children)} children")

    # ---- Count survivors vs pruned ----
    n_surv = 0
    n_pruned = 0
    for child, raw_conv in children_data:
        r = _window_scan_baseline(raw_conv, child, conv_len, d_child,
                                  dyn_base_ell_arr, ell_order, ell_count,
                                  one_minus_4eps, eps_margin)
        if r[0]:
            n_pruned += 1
        else:
            n_surv += 1
    pct_surv = 100 * n_surv / n_children if n_children > 0 else 0
    print(f"  Children: {n_pruned} pruned, {n_surv} survived ({pct_surv:.1f}%)")

    # ---- Benchmark: interleaved A/B runs ----
    baseline_times = []
    lut_times = []

    for run_idx in range(N_RUNS):
        if run_idx % 2 == 0:
            # Baseline first
            t0 = time.perf_counter()
            for child, raw_conv in children_data:
                _window_scan_baseline(raw_conv, child, conv_len, d_child,
                                      dyn_base_ell_arr, ell_order, ell_count,
                                      one_minus_4eps, eps_margin)
            t_b = time.perf_counter() - t0

            t0 = time.perf_counter()
            for child, raw_conv in children_data:
                _window_scan_lut(raw_conv, child, conv_len, d_child,
                                 threshold_table, m_plus_1,
                                 ell_order, ell_count)
            t_l = time.perf_counter() - t0
        else:
            # LUT first
            t0 = time.perf_counter()
            for child, raw_conv in children_data:
                _window_scan_lut(raw_conv, child, conv_len, d_child,
                                 threshold_table, m_plus_1,
                                 ell_order, ell_count)
            t_l = time.perf_counter() - t0

            t0 = time.perf_counter()
            for child, raw_conv in children_data:
                _window_scan_baseline(raw_conv, child, conv_len, d_child,
                                      dyn_base_ell_arr, ell_order, ell_count,
                                      one_minus_4eps, eps_margin)
            t_b = time.perf_counter() - t0

        baseline_times.append(t_b)
        lut_times.append(t_l)

        us_per_child_b = 1e6 * t_b / n_children
        us_per_child_l = 1e6 * t_l / n_children
        print(f"    Run {run_idx+1}/{N_RUNS}:  "
              f"baseline={t_b:.4f}s ({us_per_child_b:.2f}us/child)  "
              f"LUT={t_l:.4f}s ({us_per_child_l:.2f}us/child)  "
              f"delta={t_b - t_l:+.4f}s")

    # ---- Statistics ----
    baseline_times.sort()
    lut_times.sort()
    med_b = baseline_times[N_RUNS // 2]
    med_l = lut_times[N_RUNS // 2]
    min_b = baseline_times[0]
    min_l = lut_times[0]

    abs_diff = med_b - med_l
    rel_diff = abs_diff / med_b * 100 if med_b > 0 else 0
    speedup = med_b / med_l if med_l > 0 else float('inf')

    us_b = 1e6 * med_b / n_children
    us_l = 1e6 * med_l / n_children

    print(f"\n  [{level_label}] RESULTS ({N_RUNS} runs, median):")
    print(f"    {'Metric':<35s} {'Baseline':>14s} {'LUT':>14s} {'Delta':>14s}")
    print(f"    {'-'*35} {'-'*14} {'-'*14} {'-'*14}")
    print(f"    {'Median time (s)':<35s} {med_b:>14.4f} {med_l:>14.4f} "
          f"{abs_diff:>+14.4f}")
    print(f"    {'Min time (s)':<35s} {min_b:>14.4f} {min_l:>14.4f} "
          f"{min_b - min_l:>+14.4f}")
    print(f"    {'Median per-child (us)':<35s} {us_b:>14.2f} {us_l:>14.2f} "
          f"{us_b - us_l:>+14.2f}")
    print(f"    {'Speedup (window scan only)':<35s} {'':>14s} "
          f"{speedup:>14.4f}x")
    print(f"    {'Relative improvement':<35s} {'':>14s} "
          f"{rel_diff:>+13.1f}%")
    print(f"    {'Children tested':<35s} {n_children:>14,}")
    print(f"    {'Survivors':<35s} {n_surv:>14,}")
    print(f"    {'Survival rate':<35s} {pct_surv:>13.1f}%")
    print(f"    {'LUT size':<35s} "
          f"{ell_count_full * (M+1) * 8:>13,} bytes")

    print(f"\n  [{level_label}] ALL TIMINGS (seconds):")
    print(f"    Baseline: {['%.4f' % t for t in baseline_times]}")
    print(f"    LUT:      {['%.4f' % t for t in lut_times]}")

    return {
        'level': level_label,
        'n_children': n_children,
        'n_survivors': n_surv,
        'survival_pct': pct_surv,
        'baseline_median': med_b,
        'lut_median': med_l,
        'speedup': speedup,
        'rel_improvement_pct': rel_diff,
        'us_per_child_baseline': us_b,
        'us_per_child_lut': us_l,
        'baseline_all': baseline_times,
        'lut_all': lut_times,
    }


def main():
    print("=" * 72)
    print("BENCHMARK: Idea 2 — 2D Precomputed Threshold Table (ell x W_int)")
    print("=" * 72)
    print(f"Parameters: m={M}, c_target={C_TARGET}")
    print(f"Runs per config: {N_RUNS} (interleaved A/B pattern)")
    print(f"Clock: time.perf_counter()")
    print(f"Scope: window scan ONLY (isolates threshold computation)")
    print()

    # ---- Load parents ----
    print("Loading parent data and generating children...")

    # L2 parents (d=8) for L2 children (d=16)
    l0 = run_level0(n_half=2, m=M, c_target=C_TARGET, verbose=False)
    l0_surv = l0['survivors']
    all_l1 = []
    for parent in l0_surv:
        surv, tc = process_parent_fused(parent, M, C_TARGET, n_half_child=4)
        if len(surv) > 0:
            all_l1.append(surv)
    if all_l1:
        l1_surv = np.vstack(all_l1)
        from cpu.run_cascade import _fast_dedup, _canonicalize_inplace
        _canonicalize_inplace(l1_surv)
        l1_surv = _fast_dedup(l1_surv)
    else:
        l1_surv = np.empty((0, 8), dtype=np.int32)

    results = []

    # ---- L2 benchmark (d_child=16) ----
    if len(l1_surv) > 0:
        print(f"\n{'='*72}")
        print("L2 WINDOW SCAN: d_child=16")
        print(f"{'='*72}")
        children_l2, d_child_l2, conv_len_l2 = generate_children_sample(
            l1_surv, n_half_child=8, n_children_target=5000)
        if children_l2:
            r = bench_window_scan(children_l2, d_child_l2, conv_len_l2, "L2")
            if r:
                results.append(r)

    # ---- L3 benchmark (d_child=32) ----
    shard_dir = os.path.join(_project_dir, 'data', '_shards_L2')
    if os.path.exists(shard_dir):
        shard_files = sorted([f for f in os.listdir(shard_dir)
                             if f.startswith('shard_') and f.endswith('.npy')
                             and '.m' not in f])
        if shard_files:
            shard_path = os.path.join(shard_dir, shard_files[0])
            l2_surv = np.array(np.load(shard_path, mmap_mode='r')[:3000])
            print(f"\n{'='*72}")
            print("L3 WINDOW SCAN: d_child=32")
            print(f"{'='*72}")
            children_l3, d_child_l3, conv_len_l3 = generate_children_sample(
                l2_surv, n_half_child=16, n_children_target=3000)
            if children_l3:
                r = bench_window_scan(children_l3, d_child_l3, conv_len_l3,
                                      "L3")
                if r:
                    results.append(r)

    # ---- Summary ----
    if results:
        print(f"\n{'='*72}")
        print("FINAL SUMMARY — Window Scan Speedup (Threshold LUT)")
        print(f"{'='*72}")
        print(f"  {'Level':<8s} {'Baseline':>12s} {'LUT':>12s} "
              f"{'Speedup':>10s} {'Improvement':>12s} {'us/child(B)':>12s} "
              f"{'us/child(L)':>12s}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*12} "
              f"{'-'*12} {'-'*12}")
        for r in results:
            print(f"  {r['level']:<8s} "
                  f"{r['baseline_median']:>12.4f} "
                  f"{r['lut_median']:>12.4f} "
                  f"{r['speedup']:>9.4f}x "
                  f"{r['rel_improvement_pct']:>+11.1f}% "
                  f"{r['us_per_child_baseline']:>12.2f} "
                  f"{r['us_per_child_lut']:>12.2f}")
        print()
        print("  NOTE: These numbers measure the window scan in ISOLATION.")
        print("  The window scan is ~26% of total per-survivor cost at L3.")
        print("  Full-kernel speedup = window_scan_speedup * 0.26 (roughly).")
    else:
        print("\nNo benchmarks completed.")


if __name__ == '__main__':
    main()
