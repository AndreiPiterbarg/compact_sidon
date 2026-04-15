#!/usr/bin/env python
"""Benchmark SCS CPU vs GPU on a single SCS solve (no CG, no bisection).

Builds the base Lasserre SDP at a given (d, order, bw) and solves once
with SCS on CPU and (if available) GPU. Reports wall time and iterations.

Usage:
    python tests/bench_scs_cpu_vs_gpu.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
import scs


CONFIGS = [
    (4, 2, 3),
    (4, 3, 3),
    (8, 2, 4),
    (8, 3, 4),
]

MAX_ITERS = 20000
EPS = 1e-5


def bench_one(d, order, bw):
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=False)
    A, b, c, cone, meta = build_base_problem(P, add_upper_loc=True)

    print(f"  Problem: {A.shape[0]} rows x {A.shape[1]} cols, "
          f"nnz={A.nnz}, PSD cones={cone['s']}")

    data = {'A': A, 'b': b, 'c': c}
    results = {}

    # CPU direct
    try:
        solver = scs.SCS(data, cone, max_iters=MAX_ITERS,
                         eps_abs=EPS, eps_rel=EPS, verbose=False)
        t0 = time.time()
        sol = solver.solve()
        dt = time.time() - t0
        results['cpu_direct'] = {
            'time': dt, 'iters': sol['info']['iter'],
            'status': sol['info']['status'],
            'obj': float(sol['x'][meta['t_col']]) if sol['x'] is not None else None,
        }
        print(f"  CPU direct:   {dt:7.2f}s  {sol['info']['iter']:>6} iters  "
              f"obj={results['cpu_direct']['obj']:.6f}  "
              f"status={sol['info']['status']}")
    except Exception as e:
        print(f"  CPU direct:   FAILED ({e})")

    # CPU indirect
    try:
        solver = scs.SCS(data, cone, max_iters=MAX_ITERS,
                         eps_abs=EPS, eps_rel=EPS, verbose=False,
                         use_indirect=True)
        t0 = time.time()
        sol = solver.solve()
        dt = time.time() - t0
        results['cpu_indirect'] = {
            'time': dt, 'iters': sol['info']['iter'],
            'status': sol['info']['status'],
            'obj': float(sol['x'][meta['t_col']]) if sol['x'] is not None else None,
        }
        print(f"  CPU indirect: {dt:7.2f}s  {sol['info']['iter']:>6} iters  "
              f"obj={results['cpu_indirect']['obj']:.6f}  "
              f"status={sol['info']['status']}")
    except Exception as e:
        print(f"  CPU indirect: FAILED ({e})")

    # GPU (if compiled with CUDA)
    try:
        solver = scs.SCS(data, cone, max_iters=MAX_ITERS,
                         eps_abs=EPS, eps_rel=EPS, verbose=False,
                         gpu=True)
        t0 = time.time()
        sol = solver.solve()
        dt = time.time() - t0
        results['gpu'] = {
            'time': dt, 'iters': sol['info']['iter'],
            'status': sol['info']['status'],
            'obj': float(sol['x'][meta['t_col']]) if sol['x'] is not None else None,
        }
        print(f"  GPU:          {dt:7.2f}s  {sol['info']['iter']:>6} iters  "
              f"obj={results['gpu']['obj']:.6f}  "
              f"status={sol['info']['status']}")
    except Exception as e:
        print(f"  GPU:          NOT AVAILABLE ({e})")

    return results


def main():
    print(f"SCS {scs.__version__} — CPU vs GPU benchmark")
    print(f"max_iters={MAX_ITERS}, eps={EPS}")
    print()

    for d, order, bw in CONFIGS:
        print(f"d={d} O{order} bw={bw}:")
        bench_one(d, order, bw)
        print()


if __name__ == '__main__':
    main()
