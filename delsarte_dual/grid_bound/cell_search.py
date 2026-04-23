"""Adaptive 1-D interval B&B on y = z_1^2 for the Phi < 0 certificate.

At N = 1, Phi is a smooth function of y on [0, mu(M)].  We certify that
sup_{y in [0, mu(M)]} Phi(M, y) < 0 by subdividing the admissible y-interval
into cells and checking that every cell's arb-enclosure of Phi has
``.upper() < 0``.

Adaptive strategy
-----------------
Priority queue over "live" cells sorted by Phi.upper() descending.  Pop the
worst-case cell; if its Phi.upper() is already negative, all others are too
(queue invariant) and we're done.  Otherwise bisect the cell and re-queue.

Output on success is the complete list of terminal cells with their
arb-enclosed Phi.upper() (all < 0); this serves as the witness in the
certificate and is re-checkable by the independent verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional

from flint import arb, fmpq, ctx

from .phi import PhiParams, phi_N1, mu_of_M


@dataclass
class Cell:
    """Rational-endpoint cell [lo, hi] for y = z_1^2.

    Stored with exact fmpq endpoints so certificates remain replayable.
    """
    lo: fmpq
    hi: fmpq

    @property
    def width(self) -> fmpq:
        return self.hi - self.lo

    @property
    def center_q(self) -> fmpq:
        return (self.lo + self.hi) / fmpq(2)

    @property
    def half_width_q(self) -> fmpq:
        return (self.hi - self.lo) / fmpq(2)

    def as_arb(self) -> arb:
        """Rigorous arb enclosure of the closed cell."""
        return arb(self.center_q, self.half_width_q)

    def bisect(self) -> tuple["Cell", "Cell"]:
        mid = self.center_q
        return Cell(self.lo, mid), Cell(mid, self.hi)

    def to_dict(self) -> dict:
        return {
            "lo": f"{self.lo.p}/{self.lo.q}",
            "hi": f"{self.hi.p}/{self.hi.q}",
        }


@dataclass
class CellResult:
    cell: Cell
    phi_upper_float: float     # arb.upper() converted to float, for prioritising
    phi_arb_str: str           # arb's str(), kept for the certificate


@dataclass
class CellSearchResult:
    verdict: str                        # "CERTIFIED_FORBIDDEN" or "NOT_CERTIFIED"
    terminal_cells: List[CellResult]    # all cells that proved Phi.upper() < 0
    worst_cell: Optional[CellResult]    # cell with largest Phi.upper() (if any)
    cells_processed: int


def _mu_upper_rational(M: arb, extra_cushion: fmpq = fmpq(1, 10**10)) -> fmpq:
    """A conservative rational upper bound on mu(M) := M sin(pi/M)/pi.

    We take the arb-enclosed mu(M).upper(), then add a tiny rational cushion
    so the cell domain [0, mu_rat] strictly covers [0, mu(M)_true].
    """
    mu = mu_of_M(M)
    mu_up_arb = mu.upper()                      # arb, ball around true upper
    # Extract a float then convert to fmpq (exact binary to rational)
    f = float(mu_up_arb)
    # Round up in rationals: take a fmpq greater than f.
    # fmpq from float is exact-binary; add cushion.
    num, den = f.as_integer_ratio()
    q = fmpq(num, den) + extra_cushion
    return q


def certify_phi_negative(
    M: arb,
    params: PhiParams,
    max_cells: int = 20000,
    initial_splits: int = 16,
    prec_bits: int = 256,
) -> CellSearchResult:
    """Certify Phi(M, y) < 0 for all y in [0, mu(M)] by adaptive 1-D B&B.

    Args:
        M:              arb enclosing the claimed ||f*f||_inf.
        params:         precompiled Phi parameters.
        max_cells:      total cell budget; if exceeded without certifying,
                        returns verdict NOT_CERTIFIED.
        initial_splits: how many equal cells to start with on [0, mu_upper].
        prec_bits:      arb precision.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        mu_q = _mu_upper_rational(M)
        # Initial cells
        live: list[tuple[float, int, Cell, CellResult]] = []
        cells_processed = 0
        terminal: list[CellResult] = []
        worst_overall: Optional[CellResult] = None

        def eval_cell(cell: Cell) -> CellResult:
            y_arb = cell.as_arb()
            phi_v = phi_N1(M, y_arb, params)
            u_val = float(phi_v.upper())
            return CellResult(
                cell=cell,
                phi_upper_float=u_val,
                phi_arb_str=str(phi_v),
            )

        # Seed cells
        w = mu_q / fmpq(initial_splits)
        for k in range(initial_splits):
            c = Cell(fmpq(k) * w, fmpq(k + 1) * w)
            r = eval_cell(c)
            cells_processed += 1
            # Heap: max-heap on Phi.upper() => push negative for min-heap.
            heappush(live, (-r.phi_upper_float, cells_processed, c, r))

        # Adaptive B&B
        while live:
            neg_up, _serial, cell, res = heappop(live)
            up_val = -neg_up
            if up_val < 0:
                # Popped cell is the worst remaining; all others are <=;
                # certified.  Include it and all remaining as terminal.
                terminal.append(res)
                while live:
                    _n, _s, _c, _r = heappop(live)
                    terminal.append(_r)
                # Worst terminal == this just-popped cell (heap max).
                return CellSearchResult(
                    verdict="CERTIFIED_FORBIDDEN",
                    terminal_cells=terminal,
                    worst_cell=res,
                    cells_processed=cells_processed,
                )

            # Need to refine this cell.
            if cells_processed >= max_cells:
                # Record the worst unrefined cell, then fail.
                heappush(live, (neg_up, _serial, cell, res))
                return CellSearchResult(
                    verdict="NOT_CERTIFIED",
                    terminal_cells=terminal,
                    worst_cell=res,
                    cells_processed=cells_processed,
                )
            left, right = cell.bisect()
            r_left  = eval_cell(left)
            r_right = eval_cell(right)
            cells_processed += 2
            heappush(live, (-r_left.phi_upper_float,  cells_processed - 1, left,  r_left))
            heappush(live, (-r_right.phi_upper_float, cells_processed,     right, r_right))

        # Empty queue means all cells already < 0 (handled above), safety net:
        return CellSearchResult(
            verdict="CERTIFIED_FORBIDDEN",
            terminal_cells=terminal,
            worst_cell=worst_overall,
            cells_processed=cells_processed,
        )
    finally:
        ctx.prec = old


__all__ = [
    "Cell",
    "CellResult",
    "CellSearchResult",
    "certify_phi_negative",
]
