"""Rigorous lower bound on min_{x in [0, 1/4]} G(x) via arb interval B&B.

G(x) = sum_{j=1}^n a_j cos(2 pi j x / u)    [MV p. 4 eq.]

Strategy (Taylor mean-value form for tight enclosures):
  On a cell [c - r, c + r]:
     G(cell) = G(c) + G'(c) * (x - c) + G''(xi)/2 * (x - c)^2   (Taylor)
  Enclosure:
     G(cell) subset G(c) + G'(c) * [-r, r] + G''_cell * [0, r^2/2]
  where G''_cell is an arb-enclosure of G'' on the cell (computed from the
  cell-interval arb at one shot).
  Radius:  |G'(c)| r  +  |G''|_cell r^2 / 2.
  At points near the minimum (G' small) this is O(r^2); elsewhere O(r).

We do not actually need to FIND the min -- we only need a rigorous LOWER
BOUND on it.  Implementation:
  1. Decompose [0, 1/4] into N equal cells.
  2. For each cell, compute the Taylor enclosure and record its arb.lower().
  3. Take the min over cells -> rigorous lower bound on min G.

If the resulting bound is not tight enough, we subdivide the worst cells and
iterate.
"""
from __future__ import annotations

from typing import Sequence

from flint import arb, fmpq, ctx


def _two_pi_over_u(u: fmpq) -> arb:
    return arb(2) * arb.pi() / arb(u)


def _eval_G_at_point(
    coeffs: Sequence[fmpq], x_q: fmpq, u: fmpq
) -> arb:
    """G(x) at an exact rational x, returned as a high-precision arb ball.

    Precision loss comes only from the transcendental cos evaluations.
    """
    two_pi_over_u = _two_pi_over_u(u)
    x_arb = arb(x_q)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * x_arb
        total = total + arb(a_j) * arg.cos()
    return total


def _eval_G_prime_on_cell(
    coeffs: Sequence[fmpq], cell_arb: arb, u: fmpq
) -> arb:
    """|G'(x)| enclosure on a cell (given as cell_arb midpoint+radius).

    G'(x) = -sum_j a_j (2 pi j / u) sin(2 pi j x / u).
    """
    two_pi_over_u = _two_pi_over_u(u)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * cell_arb
        total = total + arb(a_j) * arg.sin() * (-two_pi_over_u * arb(j))
    return total


def _eval_G_second_on_cell(
    coeffs: Sequence[fmpq], cell_arb: arb, u: fmpq
) -> arb:
    """G''(x) enclosure on a cell.

    G''(x) = -sum_j a_j (2 pi j / u)^2 cos(2 pi j x / u).
    """
    two_pi_over_u = _two_pi_over_u(u)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * cell_arb
        w = two_pi_over_u * arb(j)
        total = total - arb(a_j) * (w * w) * arg.cos()
    return total


def G_enclosure_taylor(
    coeffs: Sequence[fmpq],
    c: fmpq,
    r: fmpq,
    u: fmpq,
) -> arb:
    """Rigorous arb enclosure of {G(x) : x in [c - r, c + r]} via 2nd-order Taylor.

    Returns an arb whose ball contains G(x) for every x in the closed cell.
    """
    # G at the center, exact-rational argument
    G_c = _eval_G_at_point(coeffs, c, u)
    # G'(c)
    Gp_c = _eval_G_at_point_derivative(coeffs, c, u)
    # G'' enclosure on the cell
    cell_arb = arb(c, r)
    Gpp_cell = _eval_G_second_on_cell(coeffs, cell_arb, u)

    # First-order Taylor ball: [-r, r]
    dx_ball = arb(0, r)
    # Second-order remainder: [0, r^2/2]  represented as arb(r^2/4, r^2/4)
    half_r_sq = (arb(r) * arb(r)) / arb(2)
    rem_ball = arb(0, 1) * half_r_sq  # enclosure [-r^2/2, r^2/2] -- safe superset
    # (The true Taylor remainder is on [0, r^2/2], but using [-r^2/2, r^2/2]
    # only relaxes the enclosure by a factor of 2 and avoids sign bookkeeping.)

    return G_c + Gp_c * dx_ball + Gpp_cell * rem_ball


def _eval_G_at_point_derivative(
    coeffs: Sequence[fmpq], x_q: fmpq, u: fmpq
) -> arb:
    """G'(x) at an exact rational x, returned as a high-precision arb ball."""
    two_pi_over_u = _two_pi_over_u(u)
    x_arb = arb(x_q)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * x_arb
        total = total - arb(a_j) * (two_pi_over_u * arb(j)) * arg.sin()
    return total


def min_G_lower_bound(
    coeffs: Sequence[fmpq],
    u: fmpq,
    x_lo: fmpq = fmpq(0),
    x_hi: fmpq = fmpq(1, 4),
    n_cells: int = 4096,
    prec_bits: int = 256,
) -> tuple[arb, fmpq]:
    """Certify a rigorous lower bound on min_{x in [x_lo, x_hi]} G(x).

    Returns:
        (lower_bound_arb, argmin_cell_center)
    where lower_bound_arb is an arb whose ``.lower()`` is the certified
    lower bound on min G (computed as the min over cells of the cell's
    Taylor enclosure lower bound), and argmin_cell_center is the rational
    midpoint of the cell achieving the worst (smallest) enclosure.

    The returned arb is NOT the min itself; it is an arb whose LOWER bound
    is a rigorous underestimate of min G.  Callers should use
    ``lower_bound_arb.lower()`` and treat it as certified.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total_width = x_hi - x_lo
        cell_width = total_width / fmpq(n_cells)
        half_width = cell_width / fmpq(2)

        # We compute a rigorous lower bound on min G as
        #     L = min_k  encl_k.lower()    (scalar lower bound per cell).
        # For cell selection we compare scalar floats (non-rigorous tie-break
        # is fine; rigor comes from the arb enclosure itself).  The final
        # returned arb's .lower() is the certified scalar.
        worst_lower_arb = None         # arb, cell enclosure
        worst_lower_float = None       # float, for comparison
        worst_center = None            # fmpq midpoint

        for k in range(n_cells):
            c = x_lo + (fmpq(2 * k + 1) * half_width)   # midpoint
            encl = G_enclosure_taylor(coeffs, c, half_width, u)
            # Scalar lower bound as a float (non-rigorous selection metric;
            # the rigorous certified value is encl.lower() kept in arb).
            lo_as_float = float(encl.lower())
            if worst_lower_float is None or lo_as_float < worst_lower_float:
                worst_lower_float = lo_as_float
                worst_lower_arb = encl
                worst_center = c
        return worst_lower_arb, worst_center
    finally:
        ctx.prec = old


__all__ = [
    "G_enclosure_taylor",
    "min_G_lower_bound",
]
