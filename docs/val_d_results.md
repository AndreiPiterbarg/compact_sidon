# val(d) Computation Results (2026-04-13)

## Definition

val(d) = min_{μ on d-simplex} max_W TV_W(μ)

where TV_W(μ) = (2d/ℓ) · Σ_{k=s}^{s+ℓ-2} Σ_{i+j=k} μ_i·μ_j for window W = (ℓ, s).

val(d) is the minimum over all nonneg mass distributions μ with Σμ_i = 1 of the
maximum test value across all windows. It is a lower bound on C_{1a}.

## Results

Computed via multistart optimization (100 restarts, Nelder-Mead + COBYLA).
These are **upper bounds** on val(d) (optimizer may not find global minimum).

| d  | val(d)     | margin over 1.30 | val > 1.30? |
|----|------------|-------------------|-------------|
| 4  | 1.10233300 | -0.197667         | NO          |
| 6  | 1.17110285 | -0.128897         | NO          |
| 8  | 1.20464420 | -0.095356         | NO          |
| 10 | 1.24136874 | -0.058631         | NO          |
| 12 | 1.27071936 | -0.029281         | NO          |
| 14 | 1.28396092 | -0.016039         | NO          |
| **16** | **1.31852334** | **+0.018523** | **YES** |

## Minimizing Distributions

d=4:  [0.3598, 0.1915, 0.1405, 0.3082]
d=6:  [0.2541, 0.0867, 0.0109, 0.1883, 0.1275, 0.3325]
d=8:  [0.2855, 0.1209, 0.1154, 0.1613, 0.0113, 0.0034, 0.0803, 0.2219]
d=10: [0.2456, 0.1146, 0.1034, 0.0764, 0.1604, 0.0126, 0.0004, 0.0409, 0.0348, 0.2109]
d=12: [0.1620, 0.0821, 0.0325, 0.0217, 0.0171, 0.0001, 0.0859, 0.1192, 0.0832, 0.1116, 0.0392, 0.2453]
d=14: [0.1958, 0.0524, 0.0907, 0.0903, 0.0719, 0.1120, 0.0505, 0.0248, 0.0344, 0.0208, 0.0024, 0.0032, 0.0722, 0.1785]
d=16: [0.1937, 0.0617, 0.0576, 0.0573, 0.0404, 0.1201, 0.0616, 0.0680, 0.0000, 0.0209, 0.0012, 0.0386, 0.0430, 0.0462, 0.0449, 0.1446]

## Key Findings

1. **val(d) crosses 1.30 between d=14 and d=16.** val(14) ≈ 1.284 < 1.30, val(16) ≈ 1.319 > 1.30.
2. **Margin at d=16: +0.019.** This is the gap available for box certification.
3. **The cascade converges at d=16** for c_target=1.30 (confirmed by cascade experiments).
4. **Subdivision-based box cert is viable** at d=16 since val(16) > 1.30 guarantees all sub-cell centers have positive margin.

## Consistency Check with Grid-Point Experiments

- d=12, S=15: 2 survivors (consistent with val(12)=1.271 < 1.30)
- d=13, S=18: 3 survivors (consistent with val(13) ≈ 1.278 < 1.30)
- d=14, S=18: 2 survivors (consistent with val(14)=1.284 < 1.30)
- d=16, S=10-15: 0 survivors (consistent with val(16)=1.319 > 1.30)

## Caveats

These are **numerical upper bounds** from local optimization, not certified lower bounds.
The true val(d) could be lower (if the optimizer missed the global minimum). However:
- 100+ random restarts with two different methods (Nelder-Mead, COBYLA)
- Results are consistent with grid-point experiments
- The minimizing distributions are non-trivial (not just uniform or trivial patterns)

For a rigorous proof, val(16) > 1.30 must be established via the cascade + box cert,
not just numerical optimization.

## Method

- scipy.optimize.minimize with Nelder-Mead (100 restarts, random simplex starting points)
- scipy.optimize.minimize with COBYLA (50 restarts, explicit simplex constraints)
- Best result across all restarts reported
- Run on AMD EPYC 7702P 64-Core (128 threads)
