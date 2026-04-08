"""Quick cascade feasibility probe.

Samples ~100 parents per level and estimates expansion factor.
Stops when expansion is clearly futile or survivors hit zero.

Usage:
    python tests/benchmark_sweep.py
    python tests/benchmark_sweep.py --m 20 --n_half 2 --c_target 1.40
    python tests/benchmark_sweep.py --sample 100 --use_flat_threshold
"""
import argparse
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
    args = ap.parse_args()

    m, n_half, c_target = args.m, args.n_half, args.c_target
    sample_n = args.sample
    flat = args.use_flat_threshold

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
          f"flat={flat}")
    print(f"L0: d={d0}, S={S0}, compositions={n_compositions:,}")
    print(f"    correction={corr:.6f}, threshold={c_target+corr:.6f}")
    print()

    # --- L0: run fully ---
    t0 = time.time()
    result = run_level0(n_half, m, c_target, verbose=True,
                        use_flat_threshold=flat)
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
            idx = np.random.default_rng(42).choice(n_parents, sample_n,
                                                    replace=False)
            sample = survivors[idx]
        else:
            sample = survivors
            sample_n_actual = n_parents

        sample_n_actual = len(sample)
        print(f"\nL{level}: d_parent={d_parent} -> d_child={d_child}, "
              f"{n_parents:,} parents, sampling {sample_n_actual}")

        total_children_sampled = 0
        total_survivors_sampled = 0
        next_level_surv = []  # only keep up to sample_n for next level
        need_more = True
        t0 = time.time()

        for i, parent in enumerate(sample):
            surv_i, n_children_i = process_parent_fused(
                parent, m, c_target, n_half_child,
                use_flat_threshold=flat)
            total_children_sampled += n_children_i
            n_surv_i = len(surv_i)
            total_survivors_sampled += n_surv_i
            if need_more and n_surv_i > 0:
                next_level_surv.append(surv_i)
                if sum(len(s) for s in next_level_surv) >= sample_n:
                    need_more = False

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                avg_children = total_children_sampled / (i + 1)
                avg_surv = total_survivors_sampled / (i + 1)
                print(f"    [{i+1}/{sample_n_actual}] "
                      f"avg children/parent={avg_children:,.0f}, "
                      f"avg survivors/parent={avg_surv:,.1f}, "
                      f"{elapsed:.1f}s")

        elapsed = time.time() - t0
        avg_children = total_children_sampled / sample_n_actual
        avg_survivors = total_survivors_sampled / sample_n_actual
        expansion = avg_survivors  # survivors per parent

        print(f"  L{level} summary: {elapsed:.1f}s")
        print(f"    avg children/parent:  {avg_children:,.0f}")
        print(f"    avg survivors/parent: {avg_survivors:,.1f}")
        print(f"    expansion factor:     {expansion:.2f}x")

        est_total_survivors = int(expansion * n_parents)
        print(f"    estimated total survivors: {est_total_survivors:,}")

        # --- Futility checks ---
        if total_survivors_sampled == 0:
            print(f"\n*** ALL PRUNED at L{level}! "
                  f"Cascade converges. ***")
            return

        if expansion > 50:
            print(f"\n*** FUTILE: expansion {expansion:.0f}x at L{level}. "
                  f"Children per parent ~{avg_children:,.0f}, "
                  f"survivors ~{avg_survivors:,.0f}. "
                  f"This config won't converge. ***")
            return

        if avg_children > 1e12:
            print(f"\n*** FUTILE: {avg_children:.1e} children/parent at L{level}. "
                  f"Enumeration alone is intractable. ***")
            return

        # Only keep enough survivors to seed next level's sample
        survivors = np.vstack(next_level_surv)
        if len(survivors) > sample_n:
            survivors = survivors[:sample_n]

        level += 1


if __name__ == '__main__':
    main()
