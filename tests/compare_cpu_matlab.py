"""Compare CPU (integer) vs MATLAB-faithful (float) survivor counts.

Runs both methods on small samples at each cascade level for
c_target = 1.30, 1.35, 1.40 and reports discrepancies.

Usage:
    python tests/compare_cpu_matlab.py
"""
import math
import os
import sys
import time

import numpy as np

# Path setup
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_dir, 'cpu')
sys.path.insert(0, _cs_dir)
sys.path.insert(0, _cs_cpu)
sys.path.insert(0, _root)

print("Loading CPU modules + JIT warmup...", flush=True)
from pruning import correction, count_compositions, _canonical_mask, asymmetry_prune_mask
from compositions import generate_compositions_batched
from run_cascade import (
    _prune_dynamic, process_parent_fused, run_level0,
)
from baseline.matlab_faithful import CS14Faithful
print("Done.\n", flush=True)

N_HALF = 2
M = 20
C_TARGETS = [1.30, 1.35, 1.40]
MAX_LEVEL = 4  # go up to L4 (d=64)

# Sample sizes per level — MATLAB-faithful gets expensive at high d
# because it precomputes O(d^2 x d^2) indicator matrices and does
# dense matmul per batch.  Scale down at higher levels.
SAMPLE_PARENTS_BY_LEVEL = {
    'L1': 200,
    'L2': 100,
    'L3': 20,
    'L4': 5,
}
# How many parents to process for generating survivors to feed next level
GENERATE_PARENTS_BY_LEVEL = {
    'L1': 500,
    'L2': 200,
    'L3': 50,
    'L4': 10,
}


# ---------------------------------------------------------------
# MATLAB-faithful L0: prune all compositions using float threshold
# ---------------------------------------------------------------

def matlab_prune_l0(compositions, n_half, m, c_target):
    """Prune L0 compositions using MATLAB-faithful float threshold.

    Reproduces the MATLAB's test-value + threshold logic on raw compositions.
    """
    d = compositions.shape[1]
    gs = 1.0 / m
    numBins = d  # d_child at L0 = d

    # Convert integer compositions to continuous weights
    weights = compositions.astype(np.float64) / m  # shape (N, d)

    survived = np.ones(len(weights), dtype=bool)

    for b in range(len(weights)):
        w = weights[b]
        pruned = False

        # Compute autoconvolution (2d-1 values)
        conv = np.zeros(2 * d - 1, dtype=np.float64)
        for i in range(d):
            if w[i] != 0:
                conv[2 * i] += w[i] * w[i]
                for j in range(i + 1, d):
                    if w[j] != 0:
                        conv[i + j] += 2.0 * w[i] * w[j]

        # Window scan matching MATLAB logic
        for ell in range(2, 2 * d + 1):
            if pruned:
                break
            n_cv = ell - 1  # number of conv bins in window

            for s_lo in range(2 * d - 1 - n_cv + 1):
                # Window sum of conv
                ws = sum(conv[s_lo:s_lo + n_cv])

                # Test value: scale by (2d) / ell
                test_val = ws * (2 * d) / ell

                # Dynamic threshold (MATLAB style):
                # boundToBeat = (lb + gs^2) + 2*gs * W
                # where W = sum of weights of bins contributing to this window
                lo_bin = s_lo - (d - 1)
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d - 1:
                    hi_bin = d - 1
                W = sum(w[lo_bin:hi_bin + 1])
                bound = (c_target + gs * gs) + 2.0 * gs * W

                if test_val >= bound:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


def cpu_prune_l0(compositions, n_half, m, c_target):
    """Prune using the CPU integer threshold (what run_cascade uses)."""
    return _prune_dynamic(compositions, n_half, m, c_target)


# ---------------------------------------------------------------
# Level 0 comparison
# ---------------------------------------------------------------

def compare_l0(n_half, m, c_target):
    """Compare L0 survivors between CPU and MATLAB-faithful."""
    d = 2 * n_half
    n_total = count_compositions(d, m)

    print(f"  L0: d={d}, m={m}, {n_total:,} compositions")

    # Generate all compositions
    all_comps = []
    for batch in generate_compositions_batched(d, m, batch_size=500_000):
        all_comps.append(batch)
    all_comps = np.vstack(all_comps)

    # Canonical filter (same for both)
    canon = _canonical_mask(all_comps)
    canonical_comps = all_comps[canon]

    # Asymmetry filter (same for both)
    needs_check = asymmetry_prune_mask(canonical_comps, n_half, m, c_target)
    candidates = canonical_comps[needs_check]

    print(f"      {len(all_comps):,} total -> {len(canonical_comps):,} canonical "
          f"-> {len(candidates):,} after asym filter")

    # CPU pruning
    t0 = time.time()
    cpu_mask = cpu_prune_l0(candidates, n_half, m, c_target)
    cpu_time = time.time() - t0
    cpu_survivors = candidates[cpu_mask]

    # MATLAB-faithful pruning
    t0 = time.time()
    matlab_mask = matlab_prune_l0(candidates, n_half, m, c_target)
    matlab_time = time.time() - t0
    matlab_survivors = candidates[matlab_mask]

    n_cpu = len(cpu_survivors)
    n_matlab = len(matlab_survivors)

    print(f"      CPU:    {n_cpu:,} survivors ({cpu_time:.2f}s)")
    print(f"      MATLAB: {n_matlab:,} survivors ({matlab_time:.2f}s)")

    # Find discrepancies
    cpu_set = set(map(tuple, cpu_survivors))
    matlab_set = set(map(tuple, matlab_survivors))

    only_cpu = cpu_set - matlab_set
    only_matlab = matlab_set - cpu_set

    if only_cpu:
        print(f"      WARNING: {len(only_cpu)} survivors in CPU only (not in MATLAB)")
        for s in list(only_cpu)[:5]:
            print(f"        {s}")
    if only_matlab:
        print(f"      WARNING: {len(only_matlab)} survivors in MATLAB only (not in CPU)")
        for s in list(only_matlab)[:5]:
            print(f"        {s}")
    if not only_cpu and not only_matlab:
        print(f"      MATCH: identical survivor sets")

    return cpu_survivors, matlab_survivors


# ---------------------------------------------------------------
# Level 1+ comparison (refinement)
# ---------------------------------------------------------------

def compare_refinement_level(parents_int, m, c_target, level_name,
                             max_parents=200):
    """Compare one refinement level between CPU and MATLAB-faithful.

    Takes a sample of parents and compares per-parent survivor counts.
    """
    d_parent = parents_int.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    # Sample parents
    if len(parents_int) > max_parents:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(parents_int), max_parents, replace=False)
        sample = parents_int[idx]
    else:
        sample = parents_int
        max_parents = len(sample)

    print(f"  {level_name}: d_parent={d_parent} -> d_child={d_child}, "
          f"sampling {len(sample)}/{len(parents_int):,} parents")

    # Setup MATLAB-faithful
    gs = 1.0 / m
    cs14 = CS14Faithful(d_child, c_target, m, mem_buffer_rows=100_000)

    cpu_total = 0
    matlab_total = 0
    cpu_children_total = 0
    matlab_children_total = 0
    discrepant_parents = []

    t0 = time.time()

    for i, parent in enumerate(sample):
        # CPU
        cpu_surv, cpu_children = process_parent_fused(
            parent, m, c_target, n_half_child)
        n_cpu = len(cpu_surv)
        cpu_total += n_cpu
        cpu_children_total += cpu_children

        # MATLAB-faithful: convert integer parent to continuous weights
        parent_float = parent.astype(np.float64) / m
        n_matlab, matlab_children, _ = cs14.process_parent(parent_float)
        matlab_total += n_matlab
        matlab_children_total += matlab_children

        if n_cpu != n_matlab:
            discrepant_parents.append({
                'idx': i,
                'parent': tuple(parent),
                'cpu': n_cpu,
                'matlab': n_matlab,
                'cpu_children': cpu_children,
                'matlab_children': matlab_children,
            })

        if (i + 1) % max(1, len(sample) // 5) == 0:
            elapsed = time.time() - t0
            print(f"      {i+1}/{len(sample)} parents processed ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Scale estimates
    scale = len(parents_int) / len(sample) if len(sample) > 0 else 1
    est_cpu = int(cpu_total * scale)
    est_matlab = int(matlab_total * scale)

    print(f"      CPU:    {cpu_total:,} survivors from {len(sample)} parents "
          f"({cpu_children_total:,} children)")
    print(f"      MATLAB: {matlab_total:,} survivors from {len(sample)} parents "
          f"({matlab_children_total:,} children)")
    print(f"      Projected full: CPU ~{est_cpu:,}, MATLAB ~{est_matlab:,}")
    print(f"      Time: {elapsed:.1f}s")

    if discrepant_parents:
        print(f"      WARNING: {len(discrepant_parents)}/{len(sample)} parents "
              f"have different survivor counts!")
        for d in discrepant_parents[:10]:
            print(f"        parent {d['idx']}: CPU={d['cpu']}, MATLAB={d['matlab']} "
                  f"(children: CPU={d['cpu_children']}, MATLAB={d['matlab_children']})")
            print(f"          bins: {d['parent']}")
    else:
        print(f"      MATCH: all {len(sample)} parents agree on survivor count")

    return cpu_total, matlab_total, discrepant_parents


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def generate_survivors(parents_int, m, c_target, level_name, max_parents):
    """Run CPU on a subset of parents to produce survivors for the next level."""
    d_parent = parents_int.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    n_use = min(max_parents, len(parents_int))
    subset = parents_int[:n_use]

    print(f"\n  Generating {level_name} survivors for next level "
          f"(CPU, {n_use}/{len(parents_int):,} parents)...", flush=True)

    t0 = time.time()
    all_survivors = []
    for i, parent in enumerate(subset):
        surv, _ = process_parent_fused(parent, m, c_target, n_half_child)
        if len(surv) > 0:
            all_survivors.append(surv)
        if (i + 1) % max(1, n_use // 5) == 0:
            elapsed = time.time() - t0
            n_surv = sum(len(s) for s in all_survivors)
            print(f"      {i+1}/{n_use} parents -> {n_surv:,} survivors "
                  f"({elapsed:.1f}s)", flush=True)

    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    elapsed = time.time() - t0
    print(f"  Got {len(survivors):,} {level_name} survivors from "
          f"{n_use} parents ({elapsed:.1f}s)")
    return survivors


def main():
    print("=" * 70)
    print("CPU vs MATLAB-faithful survivor comparison")
    print(f"n_half={N_HALF}, m={M}, levels=L0-L{MAX_LEVEL}")
    print("=" * 70)

    for c_target in C_TARGETS:
        print(f"\n{'='*70}")
        print(f"c_target = {c_target}")
        print(f"{'='*70}")

        # L0 — exact comparison on all compositions
        cpu_l0, matlab_l0 = compare_l0(N_HALF, M, c_target)
        current_survivors = cpu_l0

        if len(current_survivors) == 0:
            print(f"  L0 has 0 survivors -- cascade complete at L0")
            continue

        # Cascade L1 through L{MAX_LEVEL}
        for level_num in range(1, MAX_LEVEL + 1):
            level_name = f"L{level_num}"
            sample_size = SAMPLE_PARENTS_BY_LEVEL.get(level_name, 5)
            gen_size = GENERATE_PARENTS_BY_LEVEL.get(level_name, 10)

            # Compare CPU vs MATLAB on a sample
            cpu_total, matlab_total, disc = compare_refinement_level(
                current_survivors, M, c_target, level_name,
                max_parents=sample_size)

            if cpu_total == 0 and matlab_total == 0:
                print(f"  {level_name} has 0 survivors -- cascade complete")
                break

            if disc:
                print(f"  STOPPING cascade due to discrepancies at {level_name}")
                break

            # Generate actual survivors for next level
            if level_num < MAX_LEVEL:
                current_survivors = generate_survivors(
                    current_survivors, M, c_target, level_name, gen_size)
                if len(current_survivors) == 0:
                    print(f"  No {level_name} survivors -- cascade complete")
                    break

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == '__main__':
    main()
