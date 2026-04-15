# Improving the Lower Bound on the Sidon Autocorrelation Constant ($C_{1a}$)

> **Current bounds:** $1.2802 \leq C_{1a} \leq 1.5029$
>
> **Goal:** Push the lower bound above 1.2802.
>
> **Current focus:** Lasserre SDP hierarchy (`lasserre/`) — proving lower bounds on val(d) via semidefinite programming with clique-restricted correlative sparsity for d=64-128.

## Problem Statement

For any nonneg $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported on $[-1/4, 1/4]$ with $\int f = 1$:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}$$

**Two approaches to proving lower bounds:**

1. **Cloninger-Steinerberger cascade** (`cloninger-steinerberger/`): Exhaustive branch-and-prune over discretized mass distributions. Complete at small d, but exponential in d.

2. **Lasserre SDP hierarchy** (`lasserre/`): Polynomial optimization via semidefinite relaxation. Produces rigorous lower bounds on val(d) = min_{μ ∈ Δ_d} max_W μ^T M_W μ. Scales polynomially in d with correlative sparsity.

## Lasserre SDP Hierarchy (Current Focus)

**The discrete problem:**
$$\text{val}(d) = \min_{\mu \in \Delta_d} \max_W \mu^T M_W \mu$$

where $\Delta_d$ is the standard simplex and $M_W$ are window matrices encoding the autoconvolution test values.

**Lasserre order-k relaxation** replaces $\mu \in \Delta_d$ with pseudo-moment conditions on $y_\alpha = E[x^\alpha]$:

- $y_0 = 1$, $y_\alpha \geq 0$ (normalization + nonnegativity)
- $M_k(y) \succeq 0$ (moment matrix PSD)
- $M_{k-1}(\mu_i \cdot y) \succeq 0$ (localizing for $\mu_i \geq 0$)
- $y_\alpha = \sum_i y_{\alpha+e_i}$ (consistency from $\sum \mu_i = 1$)
- $t \cdot M_{k-1}(y) - \sum M_W[i,j] \cdot M_{k-1}(\mu_i\mu_j \cdot y) \succeq 0$ (window PSD)

**Key challenge at d=128:** The full L2 SDP has C(132,4) ≈ 12M moment variables. MOSEK cannot solve this directly.

**Solution: Clique-restricted sparsity** (Waki et al. 2006). Replace the full moment PSD with overlapping clique-restricted PSD cones. With bandwidth 16: ~500K variables instead of 12M.

**Critical mathematical constraints on the high-d solver:**
- Window PSD constraints can only be added for windows whose active bins fit in a single clique. Partial-Q PSD for uncovered windows is UNSOUND (the deficit matrix Q_true - Q_partial is entrywise ≥ 0 but NOT PSD).
- Partial consistency must use INEQUALITY (y_α ≥ Σ_{i∈S'} y_{α+e_i}), not equality. Equality with missing children forces unmapped moments to zero → lb > val(d).
- Must include all degree ≤ 2k-1 moments globally (not just within cliques) for full consistency on the critical degree-0 through degree-(2k-2) chain.

## Repository Structure

```
compact_sidon/
├── lasserre/                       # Lasserre SDP package (CURRENT FOCUS)
│   ├── core.py                     # Hash utils, monomials, windows, val_d_known
│   ├── precompute.py               # _precompute, base constraints, window PSD
│   ├── cliques.py                  # Banded cliques, sparse PSD constraints
│   └── solvers.py                  # solve_highd_sparse, solve_cg, solve_enhanced
├── cloninger-steinerberger/        # CPU cascade prover
│   └── cpu/run_cascade.py          # Multi-level cascade
├── tests/                          # Sweep scripts + benchmarks
│   ├── lasserre_highd.py           # High-d solver implementation
│   ├── lasserre_enhanced.py        # Enhanced solver (sparse/DSOS/BM)
│   ├── lasserre_scalable.py        # CG solver
│   └── lasserre_fusion.py          # Original monolithic solver
├── proof/                          # Formal proof documents
├── lean/                           # Lean 4 formalization
├── data/                           # Checkpoints and run logs
└── requirements.txt
```

## Running

```bash
# High-d sparse Lasserre (current focus)
python tests/lasserre_highd.py --d 128 --bw 16
python tests/lasserre_highd.py --d 64 --bw 12
python tests/lasserre_highd.py --d 8 --bw 6  # quick test

# Using the package
python -c "from lasserre.solvers import solve_highd_sparse; solve_highd_sparse(d=8, bandwidth=6)"

# Standard CG solver (d ≤ 32)
python tests/lasserre_scalable.py --d 16 --order 2 --mode cg

# Enhanced sparse solver (d ≤ 64)
python tests/lasserre_enhanced.py --d 32 --order 2 --psd sparse --bw 8

# Tests
pytest tests/ -v
```

## References

- [Cloninger & Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [Waki, Kim, Kojima, Muramatsu (2006)](https://doi.org/10.1007/s10957-006-9030-5) — Correlative sparsity for SDP
- [Tao et al., Optimization Constants Repo](https://github.com/teorth/optimizationproblems)
- [Matolcsi & Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [White (2022), arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Boyer & Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
