"""Rigorous Bessel-function quantities for the MV dual bound.

All transcendental outputs are returned as python-flint ``arb`` midpoint-radius
intervals.  Rational inputs (``delta``, ``u``) stay as ``fmpq`` so no rounding
is introduced before the transcendental step.

Quantities
----------
 - ``J0(x)`` ............. Bessel function J_0, returned as ``arb``.
 - ``j0_pi_j_delta_over_u(j, delta, u, prec_bits)``
     ................... J_0(pi * j * delta / u) as ``arb``.
 - ``K_tilde_period_u(j, delta, u, prec_bits)``
     ................... (1/u) * |J_0(pi j delta/u)|^2   [MV p. 4 eq. (5)].
 - ``k1_period_one(delta, prec_bits)``
     ................... |J_0(pi * delta)|^2  [MV p. 7 line 348].

References
----------
MV = Matolcsi & Vinuesa, arXiv:0907.1379 (2010).
Detailed derivation: ``delsarte_dual/mv_construction_detailed.md``.
"""
from __future__ import annotations

from flint import arb, fmpq, ctx


def _arb_pi_j_delta_over_u(j: int, delta: fmpq, u: fmpq) -> arb:
    """Return pi * j * delta / u as an arb ball at the current precision.

    The rational factor j * delta / u is computed exactly in fmpq, then
    multiplied by arb.pi() to produce a tight arb ball.
    """
    if j < 0:
        raise ValueError("j must be non-negative")
    if u <= 0:
        raise ValueError("u must be positive")
    q = fmpq(j) * delta / u  # exact rational
    return arb.pi() * arb(q)


def j0_pi_j_delta_over_u(
    j: int, delta: fmpq, u: fmpq, prec_bits: int = 256
) -> arb:
    """J_0(pi * j * delta / u) as a rigorous arb ball."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        return _arb_pi_j_delta_over_u(j, delta, u).bessel_j(0)
    finally:
        ctx.prec = old


def K_tilde_period_u(
    j: int, delta: fmpq, u: fmpq, prec_bits: int = 256
) -> arb:
    """Period-u Fourier coefficient (1/u) * |J_0(pi j delta / u)|^2.

    By MV eq. (5) this is the j-th Fourier coefficient of the kernel
    K(x) = (1/delta) * eta(x/delta) on the period-u torus.  Non-negative
    by the square, as required by MV Lemma 3.1(4).
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        j0 = _arb_pi_j_delta_over_u(j, delta, u).bessel_j(0)
        return (j0 * j0) / arb(u)
    finally:
        ctx.prec = old


def k1_period_one(delta: fmpq, prec_bits: int = 256) -> arb:
    """k_1 = hat_K(1) = |J_0(pi * delta)|^2  (period-1 Fourier coefficient).

    Used in the MV Lemma-3.3 z_1-refinement (MV eq. (10)).  Note the period-1
    vs period-u normalisation mismatch documented in
    ``mv_construction_detailed.md`` Section 2 Remark: both statements are
    taken from MV's paper.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        arg = arb.pi() * arb(delta)
        j0 = arg.bessel_j(0)
        return j0 * j0
    finally:
        ctx.prec = old


__all__ = [
    "j0_pi_j_delta_over_u",
    "K_tilde_period_u",
    "k1_period_one",
]
