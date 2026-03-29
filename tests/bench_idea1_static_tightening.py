"""Benchmark: Idea 1 — Iterative Cursor-Range Tightening via Floor-Convolution Bound.

For each cascade level (L1, L2, L3), measures:
  1. Fraction of parents where at least one cursor value is tightened
  2. Cartesian product reduction (original vs tightened)
  3. Tightening overhead (time per parent)
  4. Parents fully eliminated (product -> 0)
  5. Soundness verification: kernel with tightened ranges produces identical survivors
  6. Gold-standard verification: on small-product parents, enumerate all children and
     confirm every removed cursor value produces ZERO surviving children.

Usage:
    python tests/bench_idea1_static_tightening.py
"""
import os, sys, time
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from bench_common import (
    M, C_TARGET, level_params, generate_parents_for_level,
    build_threshold_table, build_ell_order, build_parent_prefix,
    tighten_cursor_ranges, compute_product, child_survives_check,
    sort_rows, warmup_jit,
)

# Import kernel for soundness check
_proj_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_proj_dir, "cloninger-steinerberger")
_cpu_dir = os.path.join(_cs_dir, "cpu")
if _cs_dir not in sys.path:
    sys.path.insert(0, _cs_dir)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)
from run_cascade import _compute_bin_ranges, _fused_generate_and_prune_gray


def run_tightening_benchmark(level, parents, max_soundness=10,
                              max_goldcheck=5, gold_max_product=50000):
    """Run Idea 1 benchmark for one level."""
    d_parent, d_child, n_half_child = level_params(level)
    tt = build_threshold_table(d_child, M, C_TARGET, n_half_child)
    eo = build_ell_order(d_child)
    ell_count = 2 * d_child - 1

    print(f"\n{'='*70}")
    print(f"  IDEA 1 — STATIC TIGHTENING  |  Level {level}  "
          f"(d_parent={d_parent}, d_child={d_child})")
    print(f"  Parents: {len(parents)}  |  m={M}, c_target={C_TARGET}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Phase 1: tightening statistics
    # ------------------------------------------------------------------
    n_parents = len(parents)
    tightened_count = 0
    eliminated_count = 0
    orig_products = np.zeros(n_parents, dtype=np.float64)
    tight_products = np.zeros(n_parents, dtype=np.float64)
    reductions = np.zeros(n_parents, dtype=np.float64)
    values_removed = np.zeros(n_parents, dtype=np.int64)
    iterations = np.zeros(n_parents, dtype=np.int64)
    times_us = np.zeros(n_parents, dtype=np.float64)

    # Store tightened ranges for soundness check
    all_lo_orig = []
    all_hi_orig = []
    all_lo_tight = []
    all_hi_tight = []

    for i in range(n_parents):
        parent = parents[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, d_child, n_half_child)
        if result is None:
            orig_products[i] = 0
            tight_products[i] = 0
            continue
        lo, hi, total = result
        pp = build_parent_prefix(parent)
        orig_products[i] = float(total)

        t0 = time.perf_counter()
        new_lo, new_hi, n_rem, n_it = tighten_cursor_ranges(
            parent, lo, hi, d_child, M, C_TARGET, n_half_child, tt, pp, eo)
        t1 = time.perf_counter()

        new_total = compute_product(new_lo, new_hi)
        tight_products[i] = float(new_total)
        values_removed[i] = n_rem
        iterations[i] = n_it
        times_us[i] = (t1 - t0) * 1e6

        if new_total == 0:
            reductions[i] = 1.0
            eliminated_count += 1
            tightened_count += 1
        elif new_total < total:
            reductions[i] = 1.0 - float(new_total) / float(total)
            tightened_count += 1
        else:
            reductions[i] = 0.0

        all_lo_orig.append(lo.copy())
        all_hi_orig.append(hi.copy())
        all_lo_tight.append(new_lo.copy())
        all_hi_tight.append(new_hi.copy())

    # Filter to parents with non-zero product
    valid = orig_products > 0
    n_valid = int(valid.sum())

    print(f"\n  --- Tightening Statistics (n={n_valid} valid parents) ---")
    print(f"  Parents tightened:    {tightened_count}/{n_valid} "
          f"({100*tightened_count/max(1,n_valid):.1f}%)")
    print(f"  Parents eliminated:   {eliminated_count}/{n_valid} "
          f"({100*eliminated_count/max(1,n_valid):.1f}%)")

    valid_reductions = reductions[valid]
    if len(valid_reductions) > 0:
        print(f"\n  Product reduction (among all valid parents):")
        print(f"    Mean:   {100*valid_reductions.mean():.2f}%")
        print(f"    Median: {100*np.median(valid_reductions):.2f}%")
        print(f"    p10:    {100*np.percentile(valid_reductions, 10):.2f}%")
        print(f"    p90:    {100*np.percentile(valid_reductions, 90):.2f}%")
        print(f"    Min:    {100*valid_reductions.min():.2f}%")
        print(f"    Max:    {100*valid_reductions.max():.2f}%")

        # Weighted average (by original product size)
        total_orig = orig_products[valid].sum()
        total_tight = tight_products[valid].sum()
        if total_orig > 0:
            weighted_reduction = 1.0 - total_tight / total_orig
            print(f"\n  Weighted product reduction (by original size): "
                  f"{100*weighted_reduction:.2f}%")

    tightened_mask = valid & (values_removed > 0)
    if tightened_mask.sum() > 0:
        tight_only = reductions[tightened_mask]
        print(f"\n  Among tightened parents only (n={tightened_mask.sum()}):")
        print(f"    Mean reduction:   {100*tight_only.mean():.2f}%")
        print(f"    Median reduction: {100*np.median(tight_only):.2f}%")

    valid_times = times_us[valid]
    if len(valid_times) > 0:
        print(f"\n  Tightening time per parent:")
        print(f"    Mean:   {valid_times.mean():.1f} us")
        print(f"    Median: {np.median(valid_times):.1f} us")
        print(f"    Max:    {valid_times.max():.1f} us")

    valid_iters = iterations[valid]
    if len(valid_iters) > 0:
        print(f"\n  Iterations to convergence:")
        print(f"    Mean: {valid_iters.mean():.2f}  Max: {valid_iters.max()}")

    # ------------------------------------------------------------------
    # Phase 2: kernel soundness check
    # ------------------------------------------------------------------
    n_sound = min(max_soundness, n_valid)
    sound_indices = []
    for idx in range(n_parents):
        if orig_products[idx] > 0 and tight_products[idx] > 0:
            sound_indices.append(idx)
            if len(sound_indices) >= n_sound:
                break

    if sound_indices:
        print(f"\n  --- Kernel Soundness Check (n={len(sound_indices)}) ---")
        all_pass = True
        for si in sound_indices:
            parent = parents[si]
            lo_o = all_lo_orig[si]
            hi_o = all_hi_orig[si]
            lo_t = all_lo_tight[si]
            hi_t = all_hi_tight[si]
            total_o = int(orig_products[si])
            total_t = int(tight_products[si])

            buf1 = np.empty((min(total_o, 5_000_000), d_child), dtype=np.int32)
            n1, _ = _fused_generate_and_prune_gray(
                parent, n_half_child, M, C_TARGET, lo_o, hi_o, buf1)
            s1 = sort_rows(buf1[:n1].copy())

            buf2 = np.empty((min(total_t, 5_000_000), d_child), dtype=np.int32)
            n2, _ = _fused_generate_and_prune_gray(
                parent, n_half_child, M, C_TARGET, lo_t, hi_t, buf2)
            s2 = sort_rows(buf2[:n2].copy())

            ok = (n1 == n2) and (len(s1) == 0 or np.array_equal(s1, s2))
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    Parent {si}: orig={n1} surv, tight={n2} surv -> {status}")

        print(f"  Kernel soundness: {'ALL PASS' if all_pass else '*** FAILURES ***'}")

    # ------------------------------------------------------------------
    # Phase 3: gold-standard enumeration check (small products only)
    # ------------------------------------------------------------------
    gold_indices = []
    for idx in range(n_parents):
        if 0 < orig_products[idx] <= gold_max_product and values_removed[idx] > 0:
            gold_indices.append(idx)
            if len(gold_indices) >= max_goldcheck:
                break

    if gold_indices:
        print(f"\n  --- Gold-Standard Enumeration Check (n={len(gold_indices)}) ---")
        gold_all_pass = True
        for gi in gold_indices:
            parent = parents[gi]
            lo_o = all_lo_orig[gi]
            hi_o = all_hi_orig[gi]
            lo_t = all_lo_tight[gi]
            hi_t = all_hi_tight[gi]
            pp = build_parent_prefix(parent)

            # Find which (position, cursor_value) pairs were removed
            removed_pairs = []
            for p in range(d_parent):
                for c in range(lo_o[p], lo_t[p]):
                    removed_pairs.append((p, c))
                for c in range(hi_t[p] + 1, hi_o[p] + 1):
                    removed_pairs.append((p, c))

            # For each removed pair, enumerate all children with that cursor
            # and verify NONE survive
            parent_ok = True
            for (pos, cval) in removed_pairs:
                # Build ranges for other positions
                found_survivor = False
                # Enumerate all children with cursor[pos]=cval
                # (other positions vary over original ranges)
                ranges = []
                for p2 in range(d_parent):
                    if p2 == pos:
                        ranges.append((cval, cval))
                    else:
                        ranges.append((int(lo_o[p2]), int(hi_o[p2])))

                # Enumerate via nested iteration (small product guaranteed)
                child = np.empty(d_child, dtype=np.int32)
                cursors = np.array([r[0] for r in ranges], dtype=np.int32)
                while True:
                    for p2 in range(d_parent):
                        child[2 * p2] = cursors[p2]
                        child[2 * p2 + 1] = parent[p2] - cursors[p2]
                    if child_survives_check(child, d_child, M, tt, eo, ell_count):
                        found_survivor = True
                        break
                    # Increment (odometer)
                    carry = True
                    for p2 in range(d_parent - 1, -1, -1):
                        if p2 == pos:
                            continue
                        if carry:
                            cursors[p2] += 1
                            if cursors[p2] > ranges[p2][1]:
                                cursors[p2] = ranges[p2][0]
                            else:
                                carry = False
                    if carry:
                        break

                if found_survivor:
                    print(f"    Parent {gi}: FAIL — cursor[{pos}]={cval} "
                          f"has a surviving child!")
                    parent_ok = False
                    gold_all_pass = False

            if parent_ok:
                print(f"    Parent {gi}: PASS — {len(removed_pairs)} removed "
                      f"cursor values verified, 0 survivors in excluded region")

        print(f"  Gold-standard: {'ALL PASS' if gold_all_pass else '*** FAILURES ***'}")

    print()


def main():
    print("=" * 70)
    print("  BENCHMARK: Idea 1 — Static Cursor-Range Tightening")
    print("  Floor-convolution bound propagation (arc-consistency)")
    print("=" * 70)

    warmup_jit()

    # --- L1 ---
    print("\nGenerating L1 parents ...")
    parents_l1 = generate_parents_for_level(1)
    run_tightening_benchmark(1, parents_l1, max_soundness=20, max_goldcheck=10,
                              gold_max_product=100000)

    # --- L2 ---
    print("\nGenerating L2 parents ...")
    parents_l2 = generate_parents_for_level(2, max_parents=200)
    run_tightening_benchmark(2, parents_l2, max_soundness=10, max_goldcheck=5,
                              gold_max_product=50000)

    # --- L3 ---
    print("\nGenerating L3 parents ...")
    parents_l3 = generate_parents_for_level(3, max_parents=100)
    run_tightening_benchmark(3, parents_l3, max_soundness=3, max_goldcheck=3,
                              gold_max_product=20000)


if __name__ == "__main__":
    main()
