"""Search helpers for simplex-window dual certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from simplex_window_dual.degree1 import (
    MultiplierCertificateProblem,
    MultiplierCertificateResult,
    build_multiplier_problem,
    solve_multiplier_feasibility,
)


@dataclass(frozen=True)
class SearchOutcome:
    problem: MultiplierCertificateProblem
    best_result: MultiplierCertificateResult | None
    attempted_alphas: List[float]


def search_multiplier_grid(
    d: int,
    alphas: Iterable[float],
    multiplier_degree: int = 1,
) -> SearchOutcome:
    """Try a list of alpha values and keep the best feasible certificate."""
    problem = build_multiplier_problem(d, multiplier_degree=multiplier_degree)
    best = None
    attempted = []
    for alpha in alphas:
        attempted.append(alpha)
        result = solve_multiplier_feasibility(problem, float(alpha))
        if result.success:
            best = result
    return SearchOutcome(problem=problem, best_result=best, attempted_alphas=attempted)


def search_degree1_grid(
    d: int,
    alphas: Iterable[float],
    multiplier_degree: int = 1,
) -> SearchOutcome:
    """Backward-compatible alias for multiplier-certificate grid search."""
    return search_multiplier_grid(
        d=d,
        alphas=alphas,
        multiplier_degree=multiplier_degree,
    )
