"""Degree-1 window-multiplier LP certificates on the simplex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog

from simplex_window_dual.core import (
    Monomial,
    evaluate_polynomial,
    up_to_degree_monomials,
    window_quadratic_coefficients,
)


@dataclass(frozen=True)
class Degree1CertificateProblem:
    d: int
    windows: Tuple[Tuple[int, int], ...]
    monomials_deg3: Tuple[Monomial, ...]
    monomials_deg2: Tuple[Monomial, ...]
    a_slice: slice
    b_slice: slice
    n_slice: slice
    h_slice: slice
    n_vars: int
    a_eq_base: sp.csc_matrix
    a_eq_alpha: sp.csc_matrix
    b_eq: np.ndarray
    bounds: Tuple[Tuple[float | None, float | None], ...]


@dataclass(frozen=True)
class Degree1CertificateResult:
    success: bool
    alpha: float
    status: int
    message: str
    x: np.ndarray | None

    @property
    def constant_window_weights(self) -> np.ndarray | None:
        return None if self.x is None else self.x


def _unit_monomials(d: int) -> Tuple[Monomial, ...]:
    out = []
    for i in range(d):
        mono = [0] * d
        mono[i] = 1
        out.append(tuple(mono))
    return tuple(out)


def build_degree1_problem(d: int) -> Degree1CertificateProblem:
    """Build the fixed-structure LP for degree-1 dual certificates.

    The feasibility problem at target alpha is

        sum_W (a_W + sum_i b_{W,i} x_i) (f_W(x) - alpha)
          = N(x) + (1 - sum_i x_i) H(x)

    with:
      - a_W, b_{W,i} >= 0
      - N having nonnegative coefficients in all monomials of degree <= 3
      - H free, degree <= 2
      - sum_W a_W = 1 normalization

    For x in the simplex, the RHS reduces to N(x) >= 0. Therefore if the LP is
    feasible, max_W f_W(x) >= alpha for every simplex point x.
    """
    windows, f_coeffs = window_quadratic_coefficients(d)
    monomials_deg3 = up_to_degree_monomials(d, 3)
    monomials_deg2 = up_to_degree_monomials(d, 2)
    idx_deg3 = {mono: idx for idx, mono in enumerate(monomials_deg3)}
    idx_deg2 = {mono: idx for idx, mono in enumerate(monomials_deg2)}

    n_windows = len(windows)
    k = 0
    a_slice = slice(k, k + n_windows)
    k += n_windows
    b_slice = slice(k, k + n_windows * d)
    k += n_windows * d
    n_slice = slice(k, k + len(monomials_deg3))
    k += len(monomials_deg3)
    h_slice = slice(k, k + len(monomials_deg2))
    k += len(monomials_deg2)
    n_vars = k

    rows_base = []
    cols_base = []
    vals_base = []
    rows_alpha = []
    cols_alpha = []
    vals_alpha = []
    b_eq = np.zeros(len(monomials_deg3) + 1, dtype=np.float64)

    zero = tuple(0 for _ in range(d))
    unit_monos = _unit_monomials(d)

    for row_idx, mono in enumerate(monomials_deg3):
        # Window quadratic coefficients from a_W * f_W(x)
        for w_idx, coeff in enumerate(f_coeffs):
            value = coeff.get(mono, 0.0)
            if value:
                rows_base.append(row_idx)
                cols_base.append(a_slice.start + w_idx)
                vals_base.append(value)

        # Window quadratic coefficients from b_{W,i} * x_i * f_W(x)
        for w_idx, coeff in enumerate(f_coeffs):
            for i in range(d):
                shifted = list(mono)
                if shifted[i] >= 1:
                    shifted[i] -= 1
                    value = coeff.get(tuple(shifted), 0.0)
                    if value:
                        rows_base.append(row_idx)
                        cols_base.append(b_slice.start + w_idx * d + i)
                        vals_base.append(value)

        # Alpha contributions: -alpha * a_W to the constant monomial
        if mono == zero:
            for w_idx in range(n_windows):
                rows_alpha.append(row_idx)
                cols_alpha.append(a_slice.start + w_idx)
                vals_alpha.append(-1.0)

        # Alpha contributions: -alpha * b_{W,i} to the degree-1 monomial x_i
        for i, unit in enumerate(unit_monos):
            if mono == unit:
                for w_idx in range(n_windows):
                    rows_alpha.append(row_idx)
                    cols_alpha.append(b_slice.start + w_idx * d + i)
                    vals_alpha.append(-1.0)

        # -N(x)
        rows_base.append(row_idx)
        cols_base.append(n_slice.start + idx_deg3[mono])
        vals_base.append(-1.0)

        # -(1 - sum x_i) H(x) = -H(x) + sum_i x_i H(x)
        if mono in idx_deg2:
            rows_base.append(row_idx)
            cols_base.append(h_slice.start + idx_deg2[mono])
            vals_base.append(-1.0)
        for i in range(d):
            shifted = list(mono)
            if shifted[i] >= 1:
                shifted[i] -= 1
                shifted_mono = tuple(shifted)
                if shifted_mono in idx_deg2:
                    rows_base.append(row_idx)
                    cols_base.append(h_slice.start + idx_deg2[shifted_mono])
                    vals_base.append(1.0)

    # Normalization: sum_W a_W = 1
    norm_row = len(monomials_deg3)
    for w_idx in range(n_windows):
        rows_base.append(norm_row)
        cols_base.append(a_slice.start + w_idx)
        vals_base.append(1.0)
    b_eq[norm_row] = 1.0

    a_eq_base = sp.csc_matrix(
        (vals_base, (rows_base, cols_base)),
        shape=(len(b_eq), n_vars),
        dtype=np.float64,
    )
    a_eq_alpha = sp.csc_matrix(
        (vals_alpha, (rows_alpha, cols_alpha)),
        shape=(len(b_eq), n_vars),
        dtype=np.float64,
    )

    bounds = (
        ((0.0, None),) * n_windows
        + ((0.0, None),) * (n_windows * d)
        + ((0.0, None),) * len(monomials_deg3)
        + ((None, None),) * len(monomials_deg2)
    )

    return Degree1CertificateProblem(
        d=d,
        windows=windows,
        monomials_deg3=monomials_deg3,
        monomials_deg2=monomials_deg2,
        a_slice=a_slice,
        b_slice=b_slice,
        n_slice=n_slice,
        h_slice=h_slice,
        n_vars=n_vars,
        a_eq_base=a_eq_base,
        a_eq_alpha=a_eq_alpha,
        b_eq=b_eq,
        bounds=bounds,
    )


def solve_degree1_feasibility(
    problem: Degree1CertificateProblem,
    alpha: float,
) -> Degree1CertificateResult:
    """Solve the degree-1 LP feasibility problem at fixed alpha."""
    a_eq = problem.a_eq_base + alpha * problem.a_eq_alpha
    objective = np.zeros(problem.n_vars, dtype=np.float64)
    res = linprog(
        objective,
        A_eq=a_eq,
        b_eq=problem.b_eq,
        bounds=problem.bounds,
        method="highs",
    )
    return Degree1CertificateResult(
        success=bool(res.success),
        alpha=alpha,
        status=int(res.status),
        message=str(res.message),
        x=None if not res.success else np.array(res.x, copy=True),
    )


def degree1_identity_coefficients(
    problem: Degree1CertificateProblem,
    result: Degree1CertificateResult,
) -> Dict[Monomial, float]:
    """Return the coefficient residuals of the certificate identity.

    A successful certificate should return residual coefficients that are all
    very close to zero.
    """
    if result.x is None:
        raise ValueError("certificate has no primal vector")

    coeff_vector = (problem.a_eq_base + result.alpha * problem.a_eq_alpha) @ result.x
    coeff_vector -= problem.b_eq
    return {
        mono: float(coeff_vector[idx])
        for idx, mono in enumerate(problem.monomials_deg3)
    }


def degree1_rhs_polynomial(
    problem: Degree1CertificateProblem,
    result: Degree1CertificateResult,
) -> Dict[Monomial, float]:
    """Return the nonnegative slack polynomial N(x)."""
    if result.x is None:
        raise ValueError("certificate has no primal vector")
    start = problem.n_slice.start
    return {
        mono: float(result.x[start + idx])
        for idx, mono in enumerate(problem.monomials_deg3)
        if result.x[start + idx] > 1e-12
    }


def evaluate_degree1_rhs_on_simplex(
    problem: Degree1CertificateProblem,
    result: Degree1CertificateResult,
    x: np.ndarray,
) -> float:
    """Evaluate the certified RHS N(x) on a simplex point."""
    return evaluate_polynomial(degree1_rhs_polynomial(problem, result), x)
