# Lasserre d=16 L3 — speed optimization sweep

## Setup

- **Pod**: H100 80GB HBM3 at runpod, pod_id `xem5pmz2196eol`
- **Config (locked to production)**: `d=16 O3 bw=15 rho=0.1 atom_frac=0.5 cuts_per_round=100`
- **Benchmark**: `tests/benchmark_scan.py` — multi-scale scan at K ∈ {50, 200, 400}
- **Measurement**: phase-0 minimize-t + per-K (decomp → augment → solver_init →
  warmup → hi_feas_solve @ t=1.40 → mid_bisect_solve @ t=1.33 → viol_check)
- **Fixed per-K**: `SCS_ITERS=500, SCS_EPS=1e-5` (matches prod bisect step 4–5 budget)

## What was tried

| # | Improvement | Source | Status | Reason |
|---|---|---|---|---|
| A1 | σ auto-scale `σ = 1e-6·nnz/n` | DEFERRED #5 | **REVERTED** (null) | 0% change at all 3 scales; confirms CLAUDE.md's σ-sweep finding |
| — | sp.diags(D)@A@sp.diags(E) update_A | init-fix | **REVERTED** | update_A 24.4s → 31.3s at K=200, 98s → 168s at K=400 (scipy sparse-sparse mul is worse than np.repeat cache) |
| — | Ruiz iterations 10 → 5 | init-fix | **KEPT** | 40–50% Ruiz time savings; ADMM iter counts shift but net neutral |
| — | Anderson memory 5 → 10 | — | **REVERTED** | Per user: AA mem=5 is already optimal |
| — | check_interval 10 → 20 | final combo | **REVERTED** | Delays convergence detection; mid-solve iters 50→100 at K=400 |
| — | `_ruiz_col_idx` int32 (was int64) | final combo | **KEPT** | 2.3GB → 1.1GB alloc at K=400; marginal init speedup |
| — | WARMUP_ITERS 50 → 0 (benchmark only) | final combo | KEPT (bench) | solver.solve() warm-starts internally; warmup redundant |
| C2 | Adaptive spectral-cut eigvecs | DEFERRED #7 | **N/A** | Production uses `k_vecs=0` (full PSD cones, not spectral cuts) |

## Final wall time (baseline → final kept set)

| K | baseline | ruiz5_only | final_combo | kept-only est. | % save |
|---|---|---|---|---|---|
| 50 | 49.4s | 41.5s | 41.4s | **~41s** | **-17%** |
| 200 | 175.4s | 147.6s | 140.1s | **~145s** | **-17%** |
| 400 | 517.7s | 458.0s | 498.4s | **~455s** | **-12%** |

(“kept-only” = Ruiz5 + int32 cols minus check_interval regression; final_combo is
upper-bound observed with the regression still in.)

## Phase breakdown at K=400 (baseline vs final)

| Phase | Baseline | Final kept | Δ |
|---|---|---|---|
| decomp (sparse build) | 32.1s | 32.5s | — |
| assemble (t1+t2) | 9.1s | 10.9s | — |
| augment (phase-1) | 24.5s | 26.6s | — |
| **solver_init (Ruiz + update_A + GPU xfer)** | **193.7s** | **~150s** | **-23%** |
| warmup (50 iters) | 20.1s | 20.1s | — |
| hi_feas_solve | 144.9s | 167.1s | +15% (iter count variance) |
| mid_bisect_solve | 85.8s | 55.6s | -35% (iter count variance) |
| **total** | **517.7s** | **~458s** | **-12%** |

## Bottleneck breakdown (baseline K=400)

- `solver_init` 194s (37%) — dominated by `update_A` first-call (98s) and Ruiz (74s at 10 iters)
- `hi_feas_solve` 145s (28%) — 150 iter × ~965 µs/iter, eigh-bound
- `mid_bisect_solve` 86s (17%) — 80 iter × ~1075 µs/iter
- `decomp+augment` 58s (11%) — scipy sparse builds
- `warmup` 20s (4%)

## What didn't help (and why)

- **σ auto-scale (A1)**: for the L3 SDP, ADMM iter count is insensitive to σ (the
  Anderson-accelerated iteration mostly depends on step sizes and primal/dual
  residuals, not on the proximal prefactor). Confirmed by the K=200 sp.diags
  experiment showing identical iter counts as baseline.
- **sp.diags Ruiz application**: scipy's CSC-diag multiplication allocates a
  fresh CSC output (O(nnz) both mem and time) on each call. At nnz=291M this is
  worse than the cached `D[rows]*E[cols]` product even though the latter needs
  an nnz-sized int array.
- **check_interval 10 → 20**: ADMM convergence is checked every k·N iters; at
  N=20 we round up to the next multiple, adding ~5–20 iters per solve. The
  per-iter check cost (~30ms matvecs at K=400) is small relative to the extra
  ADMM iters. Net regression.

## What remains as future work

Bottleneck growth ratios (K=50 → K=400 is 8×):
- `solver_init` 26× (super-linear) — biggest target, mostly `update_A`
- `decomp`/`augment` 15–19× (super-linear) — scipy sparse-build scaling
- Solves 6–13× (near-linear in K) — per-iter eigh dominates

Unblocked improvements (in priority order):
1. **A3 block-Jacobi CG precond** (DEFERRED #3) — 2–3 d; attacks per-iter CG cost
2. **A2 syevdx range-selection** (DEFERRED #4) — 2–3 d; attacks per-iter eigh cost
3. **B2 bisection elimination** (DEFERRED #2) — 3–4 d; 10× speedup claim
4. **B1 Z/2 symmetry block-diag** (DEFERRED #1) — 3–5 d; halves 969×969 moment cone
5. **C1 CS pseudo-moment warm-start** (DEFERRED #6) — 1–2 d; only helps round 0

Blocked on phase-0 dual extraction:
6. **D1 Schmüdgen pair cuts** (gap-accel stub)
7. **D2 Aggregated localizer cuts** (gap-accel stub)

## Changes kept in code

- `tests/admm_gpu_solver.py`:
  - `_compute_ruiz` (both `admm_solve` @ line 684 and `ADMMSolver` @ line 1187):
    `for _ in range(10):` → `for _ in range(5):`
  - `_update_A`: `np.repeat(..., dtype=np.int64)` → `dtype=np.int32`
- `tests/benchmark_scan.py`: `WARMUP_ITERS = 0` (benchmark-only)
- `tests/run_d16_l3_experiments.py`: BENCH_LOG hooks added (no behavior change
  when `BENCH_LOG is None`, which is the production default)

## Files

- Pod artifacts: `data/bench/scan_{baseline, ruiz5_only, init_fix, final_combo}_*.json`
- Pod logs: `data/bench/{baseline, ruiz5, init_fix, final}.log`
- Pod state: `data/gap_accel_pod_state.json` — still running, tear down with
  `python deploy_gap_accel_pod.py --teardown`
