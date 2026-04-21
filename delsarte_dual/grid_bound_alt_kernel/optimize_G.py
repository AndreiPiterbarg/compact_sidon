"""Per-kernel QP optimizer.

For each kernel K, the MV bound is optimised over the trig polynomial
    G(x) = sum_{j=1}^n a_j cos(2 pi j x / u)
by solving the semi-infinite QP (MV eq. 7 / p. 4):

    minimise    S_1(a; K) = sum_{j=1}^n a_j^2 / w_j(K)
    subject to  G(x) >= 1 for all x in [0, 1/4],

where the per-kernel QP weight is the period-u Fourier coefficient times u:
    w_j(K) = u * tilde_K_u(j) = hat_K_R(j / u).

For K1 (MV arcsine-auto-convolution) this recovers MV's classical weights
|J_0(pi j delta / u)|^2 and the MV solution at n=119.  For other kernels
we re-solve the QP from scratch.

The semi-infinite constraint G >= 1 on [0, 1/4] is discretised to a fine
grid of n_grid points; we additionally verify rigorously via Taylor B&B
that min_{x in [0,1/4]} G >= some positive constant (see
``grid_bound/G_min.py``) and use THAT as the effective min_G in the
gain formula (soundness).

Solver: prefers MOSEK, falls back to CLARABEL, then SCS, then ECOS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from flint import arb, fmpq, ctx

from .kernels import Kernel


def _wj_float(kernel: Kernel, j: int, delta_f: float, u_f: float,
              prec_bits: int = 128) -> float:
    """w_j(K) = hat_K_R(j/u) as a float, for QP."""
    xi = arb(fmpq(j, 1)) / arb(fmpq(u_f.as_integer_ratio()[0],
                                     u_f.as_integer_ratio()[1]))
    # Use kernel's FT at real-line frequency j/u.
    val = kernel.K_tilde_real(xi, prec_bits=prec_bits)
    return float(val.mid())


def _wj_arb(kernel: Kernel, j: int, u: fmpq, prec_bits: int = 256) -> arb:
    """w_j(K) = hat_K_R(j/u) as an arb."""
    xi = arb(fmpq(j, 1)) / arb(u)
    return kernel.K_tilde_real(xi, prec_bits=prec_bits)


@dataclass
class QPResult:
    a_opt_float: np.ndarray           # length n
    a_opt_fmpq: list                  # fmpq rounded to rational
    S1_float: float
    min_G_grid_float: float           # minimum on the discretisation grid
    solver: str
    status: str
    n: int
    n_grid: int


def solve_qp_for_kernel(
    kernel: Kernel,
    n: int = 119,
    u: fmpq = fmpq(638, 1000),
    n_grid: int = 5001,
    prec_bits_weights: int = 128,
    fmpq_denom: int = 10**8,
    verbose: bool = False,
) -> QPResult:
    """Solve the semi-infinite QP for kernel K; discretised on n_grid points.

    Returns QPResult; the coefficients are stored both as float np.ndarray
    and as fmpq with denominator ``fmpq_denom``.
    """
    u_f = float(u.p) / float(u.q)

    # Compute QP weights w_j = hat_K_R(j/u) via kernel.K_tilde_real
    w = np.zeros(n)
    for j in range(1, n + 1):
        xi = arb(fmpq(j)) / arb(u)
        w_arb = kernel.K_tilde_real(xi, prec_bits=prec_bits_weights)
        w[j - 1] = float(w_arb.mid())

    # Guard: weights must be positive for the QP to be well-posed.
    min_w = float(w.min())
    if min_w <= 0:
        # Clamp tiny / negative weights to epsilon to keep QP feasible.
        # Negative weight means Bochner fails at this frequency.  We
        # replace with a tiny positive value (which makes that term very
        # penalised in the objective, effectively forcing a_j = 0).
        tiny = 1e-12
        w = np.where(w > tiny, w, tiny)
        if verbose:
            print(f"  WARNING: min weight <= 0 (clamped); Bochner may fail")

    # Grid on [0, 1/4]
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, n))
    for j in range(1, n + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)

    # Solve with cvxpy; try MOSEK -> CLARABEL -> SCS -> ECOS
    import cvxpy as cp
    a_var = cp.Variable(n)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a_var))))
    cons = [B @ a_var >= 1.0]
    prob = cp.Problem(obj, cons)

    solver_tried = []
    solvers = []
    try:
        import mosek  # noqa
        solvers.append(("MOSEK", {}))
    except ImportError:
        pass
    solvers += [("CLARABEL", {}), ("SCS", {}), ("ECOS", {})]

    final_solver = None
    final_status = None
    for solver_name, opts in solvers:
        try:
            prob.solve(solver=solver_name, verbose=False, **opts)
            if a_var.value is not None and prob.status in (
                "optimal", "optimal_inaccurate",
            ):
                final_solver = solver_name
                final_status = prob.status
                break
        except Exception as e:
            solver_tried.append((solver_name, str(e)[:60]))
            continue

    if a_var.value is None:
        raise RuntimeError(
            f"QP failed to solve for kernel {kernel.name}; "
            f"tried {[s for s, _ in solvers]}; errors: {solver_tried}"
        )

    a_opt_float = np.asarray(a_var.value).flatten()
    S1_float = float(np.sum((a_opt_float ** 2) / w))
    min_G_grid = float((B @ a_opt_float).min())

    # Round coefficients to exact fmpq with fixed denominator
    a_fmpq = []
    for a_val in a_opt_float:
        num = int(round(a_val * fmpq_denom))
        a_fmpq.append(fmpq(num, fmpq_denom))

    if verbose:
        print(f"  {kernel.name}: solver={final_solver} status={final_status} "
              f"S1={S1_float:.5f} min_G_grid={min_G_grid:.6f}")

    return QPResult(
        a_opt_float=a_opt_float,
        a_opt_fmpq=a_fmpq,
        S1_float=S1_float,
        min_G_grid_float=min_G_grid,
        solver=final_solver,
        status=final_status,
        n=n,
        n_grid=n_grid,
    )


__all__ = ["solve_qp_for_kernel", "QPResult"]
