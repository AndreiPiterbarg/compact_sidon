import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simplex_window_dual.degree1 import (
    build_degree1_problem,
    degree1_identity_coefficients,
    evaluate_degree1_rhs_on_simplex,
    solve_degree1_feasibility,
)
from simplex_window_dual.simplex_qp import solve_simplex_quadratic


def _grid_simplex_points(d: int, steps: int) -> np.ndarray:
    if d == 1:
        return np.array([[1.0]])
    points = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            points.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], steps, d)
    return np.array(points, dtype=np.float64) / steps


def test_solve_simplex_quadratic_matches_grid_search_on_small_case():
    q = np.array(
        [
            [1.20, -0.10, 0.25, 0.00],
            [-0.10, 1.05, 0.15, 0.30],
            [0.25, 0.15, 1.40, -0.20],
            [0.00, 0.30, -0.20, 1.10],
        ],
        dtype=np.float64,
    )
    q = 0.5 * (q + q.T)

    result = solve_simplex_quadratic(q)
    grid = _grid_simplex_points(d=4, steps=40)
    grid_values = np.einsum("bi,ij,bj->b", grid, q, grid)

    assert abs(result.minimizer.sum() - 1.0) < 1e-10
    assert np.all(result.minimizer >= -1e-10)
    assert result.value <= float(grid_values.min()) + 5e-3


def test_solve_simplex_quadratic_handles_vertex_optimum():
    q = np.array(
        [
            [0.75, 2.0, 2.0],
            [2.0, 1.10, 2.0],
            [2.0, 2.0, 1.30],
        ],
        dtype=np.float64,
    )
    result = solve_simplex_quadratic(q)

    assert abs(result.value - 0.75) < 1e-12
    assert np.allclose(result.minimizer, np.array([1.0, 0.0, 0.0]))


def test_degree1_certificate_d4_has_expected_small_gap():
    problem = build_degree1_problem(4)
    yes = solve_degree1_feasibility(problem, 1.01)
    no = solve_degree1_feasibility(problem, 1.05)

    assert yes.success
    assert not no.success


def test_degree1_certificate_identity_residual_is_small():
    problem = build_degree1_problem(4)
    result = solve_degree1_feasibility(problem, 1.01)
    assert result.success

    residuals = degree1_identity_coefficients(problem, result)
    worst = max(abs(v) for v in residuals.values())
    assert worst < 1e-8

    rng = np.random.default_rng(123)
    for _ in range(20):
        x = rng.dirichlet(np.ones(problem.d))
        rhs = evaluate_degree1_rhs_on_simplex(problem, result, x)
        assert rhs >= -1e-9
