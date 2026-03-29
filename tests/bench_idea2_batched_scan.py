"""Benchmark: Idea 2 — Block-Batched Transposed Window Scan.

Measures the speedup from processing B children per batch with a
transposed (window-major) scan order vs the current sequential
(child-major) scan.

For each level (L1, L2, L3):
  1. Generate realistic children (mix of survivors + non-survivors)
  2. Precompute autoconvolutions and prefix sums (shared by both modes)
  3. Time the window scan in sequential mode
  4. Time the window scan in batched mode (B=4, 8, 16)
  5. Verify identical pruning decisions (soundness)
  6. Report: speedup, window evaluations, prune rate

NOTE: This benchmarks the SCAN ONLY, not the full kernel. The cache
phase-separation benefit (conv-update vs scan interleaving) is not
captured here — it requires the full kernel context. The numbers
below represent a LOWER BOUND on the real improvement.

Usage:
    python tests/bench_idea2_batched_scan.py
"""
import os, sys, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from bench_common import (
    M, C_TARGET, level_params, generate_parents_for_level,
    build_threshold_table, build_ell_order, build_parent_prefix,
    prepare_scan_data, scan_sequential, scan_batched,
    sample_children, warmup_jit,
)

_proj_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_proj_dir, "cloninger-steinerberger")
_cpu_dir = os.path.join(_cs_dir, "cpu")
if _cs_dir not in sys.path:
    sys.path.insert(0, _cs_dir)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)
from run_cascade import _compute_bin_ranges


def generate_children_for_level(level, n_children, rng_seed=42):
    """Generate a mix of children from real parents at *level*."""
    d_parent, d_child, n_half_child = level_params(level)

    # Use fewer parents at higher levels (generation is more expensive)
    n_source = {1: 30, 2: 15, 3: 5}[level]
    parents = generate_parents_for_level(level, max_parents=n_source,
                                          rng_seed=rng_seed)
    if len(parents) == 0:
        return np.empty((0, d_child), dtype=np.int32), d_child, n_half_child

    # Sample children from each parent
    per_parent = max(1, n_children // len(parents))
    all_children = []
    for i in range(len(parents)):
        result = _compute_bin_ranges(parents[i], M, C_TARGET, d_child,
                                      n_half_child)
        if result is None:
            continue
        lo, hi, total = result
        n_samp = min(per_parent, total)
        ch = sample_children(parents[i], lo, hi, n_samp,
                              rng_seed=rng_seed + i)
        all_children.append(ch)

    if not all_children:
        return np.empty((0, d_child), dtype=np.int32), d_child, n_half_child

    children = np.vstack(all_children)
    # Deduplicate (optional, for cleaner stats)
    if len(children) > n_children:
        rng = np.random.default_rng(rng_seed + 999)
        idx = rng.choice(len(children), n_children, replace=False)
        children = children[idx]

    return children, d_child, n_half_child


def run_scan_benchmark(level, children, d_child, n_half_child, n_trials=5):
    """Run Idea 2 benchmark for one level."""
    tt = build_threshold_table(d_child, M, C_TARGET, n_half_child)
    eo = build_ell_order(d_child)
    B = len(children)

    print(f"\n{'='*70}")
    print(f"  IDEA 2 — BATCHED TRANSPOSED SCAN  |  Level {level}  "
          f"(d_child={d_child})")
    print(f"  Children: {B}  |  m={M}, c_target={C_TARGET}")
    print(f"{'='*70}")

    # Precompute (shared cost — not part of the timed comparison)
    print("\n  Precomputing autoconvolutions ...", end="", flush=True)
    t0 = time.time()
    conv_prefix, child_prefix = prepare_scan_data(children, d_child)
    print(f" done ({time.time()-t0:.2f}s)")

    # Warmup both scan paths
    scan_sequential(conv_prefix[:2], child_prefix[:2], d_child, M, tt, eo)
    for bs in (4, 8, 16):
        scan_batched(conv_prefix[:min(bs, B)], child_prefix[:min(bs, B)],
                      d_child, M, tt, eo, bs)

    # --- Sequential timing ---
    seq_times = []
    seq_pruned = None
    seq_evals = 0
    for trial in range(n_trials):
        t0 = time.perf_counter()
        pruned, evals = scan_sequential(conv_prefix, child_prefix, d_child,
                                         M, tt, eo)
        t1 = time.perf_counter()
        seq_times.append(t1 - t0)
        if trial == 0:
            seq_pruned = pruned.copy()
            seq_evals = int(evals)

    n_pruned = int(seq_pruned.sum())
    n_survived = B - n_pruned
    seq_med = np.median(seq_times)

    print(f"\n  --- Sequential Scan ---")
    print(f"  Pruned: {n_pruned}/{B} ({100*n_pruned/B:.1f}%)  "
          f"Survived: {n_survived}/{B} ({100*n_survived/B:.1f}%)")
    print(f"  Window evaluations: {seq_evals:,}  "
          f"({seq_evals/B:.0f} per child avg)")
    print(f"  Time (median of {n_trials}): {seq_med*1000:.3f} ms  "
          f"({seq_med/B*1e6:.2f} us/child)")

    # --- Batched timing ---
    print(f"\n  --- Batched Scan (B=4, 8, 16) ---")
    print(f"  {'B':>4s}  {'Time(ms)':>10s}  {'Speedup':>8s}  "
          f"{'Evals':>12s}  {'Evals/child':>12s}  {'Pruned':>8s}  {'Sound':>6s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*6}")

    for batch_size in (4, 8, 16):
        batch_times = []
        batch_pruned = None
        batch_evals = 0
        for trial in range(n_trials):
            t0 = time.perf_counter()
            pruned, evals = scan_batched(conv_prefix, child_prefix, d_child,
                                          M, tt, eo, batch_size)
            t1 = time.perf_counter()
            batch_times.append(t1 - t0)
            if trial == 0:
                batch_pruned = pruned.copy()
                batch_evals = int(evals)

        batch_med = np.median(batch_times)
        speedup = seq_med / batch_med if batch_med > 0 else float('inf')
        sound = np.array_equal(seq_pruned, batch_pruned)

        print(f"  {batch_size:4d}  {batch_med*1000:10.3f}  {speedup:8.3f}x  "
              f"{batch_evals:12,}  {batch_evals/B:12.0f}  "
              f"{int(batch_pruned.sum()):8d}  {'OK' if sound else 'FAIL'}")

    # --- Per-child cost breakdown ---
    total_windows = (2 * d_child - 1)  # ell count
    max_evals_per_child = sum(
        (2 * d_child - 1) - (ell - 1) + 1
        for ell in range(2, 2 * d_child + 1)
    )
    print(f"\n  --- Cost Analysis ---")
    print(f"  Window sizes (ell): 2..{2*d_child}  ({total_windows} values)")
    print(f"  Max evals per child (full scan): {max_evals_per_child:,}")
    print(f"  Avg evals per pruned child:  "
          f"{seq_evals/max(1,B):.0f} / {max_evals_per_child} "
          f"({100*seq_evals/max(1,B)/max_evals_per_child:.1f}% of max)")
    print(f"  Survivors scan ALL windows ({max_evals_per_child:,} evals each)")
    print()
    print(f"  NOTE: This scan-only benchmark does NOT capture the cache")
    print(f"  phase-separation benefit from separating conv updates and")
    print(f"  window scans. The full-kernel improvement is expected to be")
    print(f"  HIGHER than the speedup shown above.")
    print()


def main():
    print("=" * 70)
    print("  BENCHMARK: Idea 2 — Block-Batched Transposed Window Scan")
    print("  Cache phase separation + amortized per-ell overhead")
    print("=" * 70)

    warmup_jit()

    # --- L1 ---
    print("\nGenerating L1 children ...")
    ch1, dc1, nhc1 = generate_children_for_level(1, n_children=50000)
    if len(ch1) > 0:
        run_scan_benchmark(1, ch1, dc1, nhc1)

    # --- L2 ---
    print("\nGenerating L2 children ...")
    ch2, dc2, nhc2 = generate_children_for_level(2, n_children=10000)
    if len(ch2) > 0:
        run_scan_benchmark(2, ch2, dc2, nhc2)

    # --- L3 ---
    print("\nGenerating L3 children ...")
    ch3, dc3, nhc3 = generate_children_for_level(3, n_children=5000)
    if len(ch3) > 0:
        run_scan_benchmark(3, ch3, dc3, nhc3)


if __name__ == "__main__":
    main()
