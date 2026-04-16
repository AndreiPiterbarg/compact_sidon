"""Search helpers for degree-1 simplex-window dual certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from simplex_window_dual.degree1 import (
    Degree1CertificateProblem,
    Degree1CertificateResult,
    build_degree1_problem,
    solve_degree1_feasibility,
)


@dataclass(frozen=True)
class SearchOutcome:
    problem: Degree1CertificateProblem
    best_result: Degree1CertificateResult | None
    attempted_alphas: List[float]


def search_degree1_grid(
    d: int,
    alphas: Iterable[float],
) -> SearchOutcome:
    """Try a list of alpha values and keep the best feasible certificate."""
    problem = build_degree1_problem(d)
    best = None
    attempted = []
    for alpha in alphas:
        attempted.append(alpha)
        result = solve_degree1_feasibility(problem, float(alpha))
        if result.success:
            best = result
    return SearchOutcome(problem=problem, best_result=best, attempted_alphas=attempted)
