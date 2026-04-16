"""Dual certificate experiments for the discrete window problem.

This package builds lower-bound certificates for

    val(d) = min_{mu in Delta_d} max_W mu^T M_W mu

using dual objects that live on the window side.

Two families are currently implemented:

1. Constant window mixtures.
2. Degree-1 nonnegative window multipliers with a simplex equality multiplier.
"""

from simplex_window_dual.core import (
    exact_degree_monomials,
    up_to_degree_monomials,
    window_quadratic_coefficients,
)
from simplex_window_dual.simplex_qp import solve_simplex_quadratic

__all__ = [
    "exact_degree_monomials",
    "solve_simplex_quadratic",
    "up_to_degree_monomials",
    "window_quadratic_coefficients",
]
