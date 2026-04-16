"""CLI for simplex-window dual certificate search."""

from __future__ import annotations

import argparse

from simplex_window_dual.search import search_degree1_grid


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-12:
        yield round(value, 10)
        value += step


def main() -> None:
    parser = argparse.ArgumentParser(description="Search degree-1 simplex-window dual certificates")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--alpha-start", type=float, default=1.00)
    parser.add_argument("--alpha-stop", type=float, default=1.10)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args()

    outcome = search_degree1_grid(
        d=args.d,
        alphas=_frange(args.alpha_start, args.alpha_stop, args.alpha_step),
    )
    best = outcome.best_result
    if best is None:
        print(f"No feasible degree-1 certificate found for d={args.d}.")
        return

    print(f"Best feasible alpha for d={args.d}: {best.alpha:.10f}")


if __name__ == "__main__":
    main()
