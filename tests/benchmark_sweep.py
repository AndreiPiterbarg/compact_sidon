"""Quick cascade feasibility probe.

Samples ~100 parents per level and estimates expansion factor.
Stops when expansion is clearly futile or survivors hit zero.

Usage:
    python tests/benchmark_sweep.py
    python tests/benchmark_sweep.py --m 20 --n_half 2 --c_target 1.40
    python tests/benchmark_sweep.py --sample 100 --use_flat_threshold
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused

C_UPPER = 1.5029


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--c_target', type=float, default=1.40)
    ap.add_argument('--sample', type=int, default=100,
                    help='parents to sample per level')
    ap.add_argument('--use_flat_threshold', action='store_true')
    ap.add_argument('--use_F', action='store_true',
                    help='Use variant F pruning (LP-tight Δ_BB).  Sound, '
                         '25-65%% additional pruning over W-refined.  '
                         'Mutually exclusive with --use_flat_threshold.')
    args = ap.parse_args()

    if args.use_F and args.use_flat_threshold:
        ap.error('--use_F and --use_flat_threshold are mutually exclusive.')

    m, n_half, c_target = args.m, args.n_half, args.c_target
    sample_n = args.sample
    flat = args.use_flat_threshold
    use_F = args.use_F

    # Vacuity check
    corr = correction(m, n_half)
    if c_target + corr >= C_UPPER:
        print(f"VACUOUS: c_target={c_target} + correction={corr:.4f} "
              f"= {c_target+corr:.4f} >= {C_UPPER}")
        return

    d0 = 2 * n_half
    S0 = 4 * n_half * m
    n_compositions = count_compositions(d0, S0)
    print(f"Config: m={m}, n_half={n_half}, c_target={c_target}, "
          f"flat={flat}, use_F={use_F}")
    print(f"L0: d={d0}, S={S0}, compositions={n_compositions:,}")
    print(f"    correction={corr:.6f}, threshold={c_target+corr:.6f}")
    print()

    # --- L0: run fully ---
    t0 = time.time()
    result = run_level0(n_half, m, c_target, verbose=True,
                        use_flat_threshold=flat, use_F=use_F)
    survivors = result['survivors']
    n_surv = result['n_survivors']
    print(f"\nL0 done: {n_surv:,} survivors in {time.time()-t0:.1f}s")

    if n_surv == 0:
        print("PROVEN at L0!")
        return

    # --- Cascade levels ---
    level = 1
    while True:
        d_parent = survivors.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2
        n_parents = len(survivors)

        # Sample
        if n_parents > sample_n:
            idx = np.random.default_rng().choice(n_parents, sample_n,
                                                    replace=False)
            sample = survivors[idx]
        else:
            sample = survivors
            sample_n_actual = n_parents

        sample_n_actual = len(sample)

        # Pre-check: estimate children per parent from first few samples
        from pruning import correction as _corr
        _c = _corr(m, n_half_child)
        _thresh = c_target + _c + 1e-9
        _x_cap = int(math.floor(m * math.sqrt(4 * d_child * _thresh)))
        _x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
        _x_cap = min(_x_cap, _x_cap_cs)
        # Estimate avg children/parent from the sample
        _B = sample.astype(np.int64)
        _lo = np.maximum(0, 2 * _B - _x_cap)
        _hi = np.minimum(2 * _B, _x_cap)
        _eff = np.maximum(_hi - _lo + 1, 0)
        _counts = np.prod(_eff, axis=1)
        _median_cpp = int(np.median(_counts))
        _est_total = float(_median_cpp) * n_parents

        print(f"\nL{level}: d_parent={d_parent} -> d_child={d_child}, "
              f"{n_parents:,} parents, sampling {sample_n_actual}")
        print(f"    x_cap={_x_cap}, median children/parent={_median_cpp:,}")
        print(f"    estimated total children: {_est_total:.2e}")

        BUDGET = 160e12
        if _est_total > BUDGET:
            print(f"\n*** EXCEEDS BUDGET before processing: "
                  f"{_est_total:.2e} > {BUDGET:.0e} "
                  f"({_est_total/BUDGET:.0f}x over) ***")
            return
        print(f"    budget usage: {_est_total/BUDGET*100:.1f}%")

        # Sort by children count, only process parents under 500M children
        next_level_surv = []
        MAX_CHILDREN_PER_PARENT = 500_000_000
        _order = np.argsort(_counts)
        sample = sample[_order]
        _counts_sorted = _counts[_order]

        # Filter to processable parents
        processable = _counts_sorted <= MAX_CHILDREN_PER_PARENT
        n_processable = int(processable.sum())
        if n_processable == 0:
            lightest = int(_counts_sorted[0])
            print(f"    No parents under {MAX_CHILDREN_PER_PARENT:,} "
                  f"(lightest={lightest:,}), skipping level")
            # Still pass survivors through for next level estimate
            if len(next_level_surv) == 0 and level > 1:
                return
            break
        sample = sample[processable][:sample_n]
        sample_n_actual = len(sample)
        print(f"    {n_processable} processable parents "
              f"(< {MAX_CHILDREN_PER_PARENT:,} children)", flush=True)

        total_children_sampled = 0
        total_survivors_sampled = 0
        next_level_surv = []
        need_more = True
        t0 = time.time()
        n_completed = 0

        for i, parent in enumerate(sample):
            surv_i, n_children_i = process_parent_fused(
                parent, m, c_target, n_half_child,
                use_flat_threshold=flat, use_F=use_F)
            total_children_sampled += n_children_i
            n_surv_i = len(surv_i)
            total_survivors_sampled += n_surv_i
            n_completed += 1
            if need_more and n_surv_i > 0:
                next_level_surv.append(surv_i)
                if sum(len(s) for s in next_level_surv) >= sample_n:
                    need_more = False

            print(f"    [{i+1}/{sample_n_actual}] "
                  f"{n_children_i:,} children, {n_surv_i:,} surv",
                  flush=True)

        elapsed = time.time() - t0
        if n_completed == 0:
            print(f"  L{level}: no parents completed, stopping")
            return
        avg_children = total_children_sampled / n_completed
        avg_survivors = total_survivors_sampled / n_completed
        expansion = avg_survivors  # survivors per parent

        print(f"  L{level} summary: {elapsed:.1f}s")
        print(f"    avg children/parent:  {avg_children:,.0f}")
        print(f"    avg survivors/parent: {avg_survivors:,.1f}")
        print(f"    expansion factor:     {expansion:.2f}x")

        est_total_survivors = int(expansion * n_parents)
        est_total_children = int(avg_children * n_parents)
        print(f"    estimated total survivors: {est_total_survivors:,}")
        print(f"    estimated total children:  {est_total_children:,}")

        # --- Futility checks ---
        if total_survivors_sampled == 0:
            print(f"\n*** ALL PRUNED at L{level}! "
                  f"Cascade converges. ***")
            return

        # Budget check: 160T children total across all remaining levels
        BUDGET = 160e12
        if est_total_children > BUDGET:
            print(f"\n*** EXCEEDS BUDGET: {est_total_children:.2e} total children "
                  f"at L{level} > {BUDGET:.0e} budget. "
                  f"({est_total_children/BUDGET:.0f}x over) ***")
            return

        print(f"    budget usage: {est_total_children/BUDGET*100:.1f}% "
              f"of {BUDGET:.0e}")

        # Only keep enough survivors to seed next level's sample
        survivors = np.vstack(next_level_surv)
        if len(survivors) > sample_n:
            survivors = survivors[:sample_n]

        level += 1


if __name__ == '__main__':
    main()
