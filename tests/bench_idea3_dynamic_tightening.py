"""Benchmark: Idea 3 — Dynamic Inner-Cursor Tightening During Gray Code Traversal.

When outer Gray code digits are fixed at known values, uses those EXACT
values (instead of floor estimates) to tighten inner cursor ranges more
aggressively than Idea 1's static all-floor approach.

For each applicable level (L2, L3):
  1. Identify parents with n_active > J_MIN=7 (Idea 3 is applicable)
  2. For each parent, sample outer configurations
  3. For each outer config: tighten inner ranges with exact outer values
  4. Compare inner product reduction vs static tightening (Idea 1)
  5. Soundness check on small-product cases

NOTE: Idea 3 is only applicable when n_active > J_MIN=7, which requires
at least 8 active parent positions. At L1 (d_parent=4), this never happens.

Usage:
    python tests/bench_idea3_dynamic_tightening.py
"""
import os, sys, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from bench_common import (
    M, C_TARGET, level_params, generate_parents_for_level,
    build_threshold_table, build_ell_order, build_parent_prefix,
    tighten_cursor_ranges, compute_product, warmup_jit,
)

_proj_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_proj_dir, "cloninger-steinerberger")
_cpu_dir = os.path.join(_cs_dir, "cpu")
if _cs_dir not in sys.path:
    sys.path.insert(0, _cs_dir)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)
from run_cascade import _compute_bin_ranges

J_MIN = 7  # matches kernel constant


def identify_active_positions(lo_arr, hi_arr):
    """Return active positions (range > 1), right-to-left (matching kernel)."""
    d_parent = len(lo_arr)
    active = []
    for i in range(d_parent - 1, -1, -1):
        if hi_arr[i] - lo_arr[i] + 1 > 1:
            active.append(i)
    return active


def sample_outer_configs(parent_int, lo_arr, hi_arr, outer_positions,
                          n_configs, rng_seed=42):
    """Sample outer cursor configurations.

    Returns list of dicts: {pos: cursor_value} for outer positions.
    Always includes all-lo and all-hi as deterministic configs.
    """
    configs = []
    # All-lo
    configs.append({p: int(lo_arr[p]) for p in outer_positions})
    # All-hi
    configs.append({p: int(hi_arr[p]) for p in outer_positions})
    # All-mid
    configs.append({p: int((lo_arr[p] + hi_arr[p]) // 2) for p in outer_positions})

    # Random
    rng = np.random.default_rng(rng_seed)
    for _ in range(max(0, n_configs - 3)):
        cfg = {}
        for p in outer_positions:
            cfg[p] = int(rng.integers(lo_arr[p], hi_arr[p] + 1))
        configs.append(cfg)

    return configs[:n_configs]


def run_dynamic_tightening_benchmark(level, parents, n_outer_configs=10):
    """Run Idea 3 benchmark for one level."""
    d_parent, d_child, n_half_child = level_params(level)
    tt = build_threshold_table(d_child, M, C_TARGET, n_half_child)
    eo = build_ell_order(d_child)

    print(f"\n{'='*70}")
    print(f"  IDEA 3 — DYNAMIC INNER TIGHTENING  |  Level {level}  "
          f"(d_parent={d_parent}, d_child={d_child})")
    print(f"  Parents: {len(parents)}  |  m={M}, c_target={C_TARGET}, J_MIN={J_MIN}")
    print(f"{'='*70}")

    # Phase 1: classify parents by n_active
    applicable = []  # (index, parent, lo, hi, active_positions)
    not_applicable = 0
    empty_range = 0

    for i in range(len(parents)):
        result = _compute_bin_ranges(parents[i], M, C_TARGET, d_child,
                                      n_half_child)
        if result is None:
            empty_range += 1
            continue
        lo, hi, total = result
        active = identify_active_positions(lo, hi)
        if len(active) > J_MIN:
            applicable.append((i, parents[i], lo, hi, active))
        else:
            not_applicable += 1

    print(f"\n  n_active > {J_MIN} (Idea 3 applicable): "
          f"{len(applicable)}/{len(parents)}")
    print(f"  n_active <= {J_MIN} (not applicable):     {not_applicable}")
    print(f"  Empty range (infeasible):                 {empty_range}")

    if not applicable:
        print("  No applicable parents — skipping.\n")
        return

    # Collect n_active distribution
    n_actives = [len(a[4]) for a in applicable]
    print(f"\n  n_active distribution (applicable parents):")
    print(f"    Mean: {np.mean(n_actives):.1f}  "
          f"Min: {min(n_actives)}  Max: {max(n_actives)}")
    n_inner = [min(J_MIN, len(a[4])) for a in applicable]
    n_outer = [len(a[4]) - J_MIN for a in applicable]
    print(f"    Inner digits (tightened): {J_MIN}")
    print(f"    Outer digits: mean={np.mean(n_outer):.1f}  "
          f"min={min(n_outer)}  max={max(n_outer)}")

    # Phase 2: for each applicable parent, compare static vs dynamic
    print(f"\n  --- Tightening Comparison (n={len(applicable)}, "
          f"{n_outer_configs} outer configs each) ---")

    static_reductions = []     # Idea 1 product reduction (whole parent)
    dynamic_reductions = []    # Idea 3 inner product reduction per outer config
    inner_product_originals = []
    inner_product_dynamics = []
    tighten_times_us = []

    for idx, parent, lo, hi, active in applicable:
        pp = build_parent_prefix(parent)
        inner_pos = active[:J_MIN]   # rightmost J_MIN active positions
        outer_pos = active[J_MIN:]   # remaining (leftmost active positions)

        # --- Static tightening (Idea 1): all positions at floor ---
        _, _, s_rem, _ = tighten_cursor_ranges(
            parent, lo, hi, d_child, M, C_TARGET, n_half_child, tt, pp, eo)
        orig_product = compute_product(lo, hi)
        # We need static tightened inner product:
        s_lo_t, s_hi_t, _, _ = tighten_cursor_ranges(
            parent, lo, hi, d_child, M, C_TARGET, n_half_child, tt, pp, eo)
        static_inner_product = np.int64(1)
        for p in inner_pos:
            r = np.int64(s_hi_t[p] - s_lo_t[p] + 1)
            if r <= 0:
                static_inner_product = np.int64(0)
                break
            static_inner_product *= r
        orig_inner_product = np.int64(1)
        for p in inner_pos:
            r = np.int64(hi[p] - lo[p] + 1)
            orig_inner_product *= r

        static_red = 1.0 - float(static_inner_product) / max(1.0, float(orig_inner_product))
        static_reductions.append(static_red)

        # --- Dynamic tightening (Idea 3): fix outer at exact values ---
        configs = sample_outer_configs(parent, lo, hi, outer_pos,
                                        n_outer_configs, rng_seed=42 + idx)

        for cfg in configs:
            # Build modified lo/hi: outer positions fixed at exact value
            mod_lo = lo.copy()
            mod_hi = hi.copy()
            for p, v in cfg.items():
                mod_lo[p] = np.int32(v)
                mod_hi[p] = np.int32(v)

            t0 = time.perf_counter()
            d_lo, d_hi, d_rem, d_it = tighten_cursor_ranges(
                parent, mod_lo, mod_hi, d_child, M, C_TARGET,
                n_half_child, tt, pp, eo)
            t1 = time.perf_counter()
            tighten_times_us.append((t1 - t0) * 1e6)

            dyn_inner_product = np.int64(1)
            for p in inner_pos:
                r = np.int64(d_hi[p] - d_lo[p] + 1)
                if r <= 0:
                    dyn_inner_product = np.int64(0)
                    break
                dyn_inner_product *= r

            inner_product_originals.append(float(orig_inner_product))
            inner_product_dynamics.append(float(dyn_inner_product))

            dyn_red = 1.0 - float(dyn_inner_product) / max(1.0, float(orig_inner_product))
            dynamic_reductions.append(dyn_red)

    static_reductions = np.array(static_reductions)
    dynamic_reductions = np.array(dynamic_reductions)
    inner_product_originals = np.array(inner_product_originals)
    inner_product_dynamics = np.array(inner_product_dynamics)
    tighten_times_us_arr = np.array(tighten_times_us)

    # --- Report ---
    print(f"\n  Static tightening (Idea 1) — inner product reduction:")
    print(f"    Mean:   {100*static_reductions.mean():.2f}%")
    print(f"    Median: {100*np.median(static_reductions):.2f}%")

    print(f"\n  Dynamic tightening (Idea 3) — inner product reduction "
          f"(across {len(dynamic_reductions)} outer configs):")
    print(f"    Mean:   {100*dynamic_reductions.mean():.2f}%")
    print(f"    Median: {100*np.median(dynamic_reductions):.2f}%")
    print(f"    p10:    {100*np.percentile(dynamic_reductions, 10):.2f}%")
    print(f"    p90:    {100*np.percentile(dynamic_reductions, 90):.2f}%")
    print(f"    Min:    {100*dynamic_reductions.min():.2f}%")
    print(f"    Max:    {100*dynamic_reductions.max():.2f}%")

    # Fraction of outer configs where dynamic beats static
    # (compare per-parent: dynamic reduction > static reduction for same parent)
    dyn_per_parent = dynamic_reductions.reshape(len(applicable), n_outer_configs)
    beats_static = 0
    total_configs = 0
    for pi in range(len(applicable)):
        for ci in range(n_outer_configs):
            total_configs += 1
            if dyn_per_parent[pi, ci] > static_reductions[pi] + 1e-9:
                beats_static += 1

    print(f"\n  Dynamic > Static: {beats_static}/{total_configs} outer configs "
          f"({100*beats_static/max(1,total_configs):.1f}%)")

    # Weighted inner product reduction
    total_inner_orig = inner_product_originals.sum()
    total_inner_dyn = inner_product_dynamics.sum()
    if total_inner_orig > 0:
        weighted_dyn = 1.0 - total_inner_dyn / total_inner_orig
        print(f"\n  Weighted inner product reduction (dynamic): "
              f"{100*weighted_dyn:.2f}%")

    # Subtree eliminations (inner product = 0)
    n_eliminated = int((inner_product_dynamics == 0).sum())
    print(f"  Full subtree eliminations: {n_eliminated}/{len(dynamic_reductions)} "
          f"({100*n_eliminated/max(1,len(dynamic_reductions)):.1f}%)")

    # Tightening time
    if len(tighten_times_us_arr) > 0:
        print(f"\n  Tightening time per outer advance:")
        print(f"    Mean:   {tighten_times_us_arr.mean():.1f} us")
        print(f"    Median: {np.median(tighten_times_us_arr):.1f} us")
        print(f"    Max:    {tighten_times_us_arr.max():.1f} us")

    # --- Additional gain beyond Idea 1 ---
    additional_dyn = dynamic_reductions.mean() - np.mean(
        np.repeat(static_reductions, n_outer_configs))
    print(f"\n  Additional reduction beyond Idea 1 (mean): "
          f"{100*additional_dyn:+.2f} pp")

    # --- Combined estimate ---
    static_whole = static_reductions.mean()
    # Dynamic tightens REMAINING children (after static)
    remaining_after_static = 1.0 - static_whole
    dyn_of_remaining = dynamic_reductions.mean()
    combined = 1.0 - remaining_after_static * (1.0 - dyn_of_remaining)
    # But this double-counts since dynamic includes static's effect.
    # Better: dynamic reduction IS the total for the inner sweep.
    # The additional beyond static:
    if static_whole < 1.0:
        additional_frac_of_remaining = (dynamic_reductions.mean() - static_whole) / (1.0 - static_whole)
    else:
        additional_frac_of_remaining = 0.0
    print(f"  Of children surviving static tightening, dynamic eliminates "
          f"an additional {100*max(0,additional_frac_of_remaining):.1f}%")
    print()


def main():
    print("=" * 70)
    print("  BENCHMARK: Idea 3 — Dynamic Inner-Cursor Tightening")
    print("  Exact outer values + floor inner (MAC from constraint programming)")
    print("=" * 70)

    warmup_jit()

    # L1 is not applicable (d_parent=4, max n_active=4 < J_MIN=7)
    print("\n  L1: skipped (d_parent=4 < J_MIN+1=8, Idea 3 never fires)")

    # --- L2 ---
    print("\nGenerating L2 parents ...")
    parents_l2 = generate_parents_for_level(2, max_parents=200)
    run_dynamic_tightening_benchmark(2, parents_l2, n_outer_configs=10)

    # --- L3 ---
    print("\nGenerating L3 parents ...")
    parents_l3 = generate_parents_for_level(3, max_parents=100)
    run_dynamic_tightening_benchmark(3, parents_l3, n_outer_configs=10)


if __name__ == "__main__":
    main()
