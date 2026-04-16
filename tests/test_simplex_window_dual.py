import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
