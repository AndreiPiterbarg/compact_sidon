# Proposed improvements — grouped

Source: `DEFERRED_IMPROVEMENTS.md` + stubs in `lasserre/gap_accelerator.py`
(referenced from the live SDP in `tests/run_scs_direct.py` / `tests/run_d16_l3.py`).

Working copy of production: `tests/run_d16_l3_experiments.py`.
Two metrics to attack:
  **(T)** time per round (wall-clock, worsens when #windows grows)
  **(C)** gap-closure % per round (stalls after R2–R7)

## Group A — ADMM solver-internal speedups (attack T)
No change to the Lasserre relaxation itself. Each item only changes how
fast one ADMM solve runs. Soundness is preserved because the KKT system
and verdict classification are untouched.

| # | Name | Source | Effort | Attack |
|---|---|---|---|---|
| A1 | σ auto-scaling `σ = max(1e-10, 1e-6·nnz/n)` | DEF #5 | 30 min | T |
| A2 | Range-selection eigendecomposition (`syevdx` or eigh+truncate) | DEF #4 | 2–3 d | T |
| A3 | Block-Jacobi CG preconditioner on banded clique blocks | DEF #3 | 2–3 d | T |

## Group B — Algorithmic restructuring (attack T, maybe C)
Change the outer loop structure. Correctness requires a separate proof.

| # | Name | Source | Effort | Attack |
|---|---|---|---|---|
| B1 | Z/2 time-reversal symmetry block-diag (969 → 2×485) | DEF #1 | 3–5 d | T |
| B2 | Bisection elimination — single min-t solve with lazy cuts | DEF #2 | 3–4 d | T |

## Group C — Warm-start / cut-quality (attack C)
Improve gap-closure per round without changing the relaxation.

| # | Name | Source | Effort | Attack |
|---|---|---|---|---|
| C1 | Cloninger-Steinerberger pseudo-moment warm-start for R0 | DEF #6 | 1–2 d | C |
| C2 | Adaptive restart of spectral cut eigenvectors | DEF #7 | 1 d | C |
| C3 | Sub-clique precision cuts (wire `subclique_precision_cuts`) | gap_accel stub | 1 d | C |
| C4 | Pataki-Alizadeh facial reduction of active cones | gap_accel helper | 2 d | T+C |

## Group D — Advanced dual-guided cuts (blocked on phase-0 duals)
Both require extracting ADMM dual values at the phase-0 (minimize-t) solve
and feeding them into the cut generator; this scaffolding does not yet
exist (`extract_phase0_duals=False` in `gap_accelerator.DEFAULTS`).

| # | Name | Source | Effort | Attack |
|---|---|---|---|---|
| D1 | Schmüdgen-pair cuts at fixed t | gap_accel stub | 3–4 d | C |
| D2 | Aggregated localizer cuts | gap_accel stub | 2–3 d | C |

## Sequencing plan
Attack order = cheapest-verified-first.
1. **A1** (σ auto-scale): 30-min change, isolated in ADMM. Run baseline diff.
2. **C2** (adaptive eigenvectors): 1-day change, pure cut refresh. Safest gap lift.
3. **A2** simple variant (eigh + discard positives): low-risk speed win.
4. **C3** (sub-clique cuts): if R3+ gap stalls, add these before D-group.
5. **B2** (bisection elimination): only if T still dominates.
6. **B1 / D-group**: last, biggest structural change.

Each step: verify math → implement on the experiment file → diff against
the pinned baseline → keep if improved, revert otherwise.
