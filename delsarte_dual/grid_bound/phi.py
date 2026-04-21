"""Forbidden-region function Phi(M, y) at N = 1 for the MV dual bound.

Spec sign convention (this package)
-----------------------------------
    Phi(M, y) >= 0    =>   (M, y) CONSISTENT with admissibility (MV ineq. holds)
    Phi(M, y) <  0    =>   (M, y) is FORBIDDEN (MV ineq. violated)

At N = 1 the variable is y := z_1^2 = |hat f(1)|^2.  MV's inequality (eq. (10)
of arXiv:0907.1379; see ``delsarte_dual/mv_construction_detailed.md`` Section 1)
reads, in our sign convention:

    Phi(M, y) := M + 1 + 2 y k_1 + sqrt((M - 1 - 2 y^2)_+) * sqrt((K2 - 1 - 2 k_1^2)_+)
                 - (2/u + a).

Quantities:
  * M   — the claim ||f*f||_inf = M.
  * y   — z_1^2 in [0, mu(M)] where mu(M) := M sin(pi/M)/pi  (Lemma 3.4 bathtub).
  * k_1 := |J_0(pi delta)|^2                      [period-1 hat_K(1)]
  * K2  := MV-declared upper bound on ||K||_2^2, taken as 0.5747/delta.
  * u   := 1/2 + delta
  * a   := (4/u) * m^2 / S_1   with  m = min_{[0,1/4]} G,   S_1 = sum a_j^2 / |J_0(pi j delta/u)|^2.

Soundness (spec sign)
---------------------
If an admissible f has ||f*f||_inf = M, then Phi(M, |hat f(1)|^2) >= 0
(MV eq. (10)).  Contrapositive: if Phi < 0 for every y in the admissible box,
no such f exists at that M => M is a lower bound on C_{1a}.

Implementation notes
--------------------
All algebraic inputs (delta, u, K2, G-coeffs) are exact fmpq.  Bessel values
are arb balls with rigorous enclosures.  The returned Phi is an arb
enclosure; for cell-search, ``Phi(...).upper()`` is a certified upper bound
on max_{y in cell} Phi (rejection test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flint import arb, fmpq, ctx

from .bessel import j0_pi_j_delta_over_u, k1_period_one
from .coeffs import mv_coeffs_fmpq, MV_DELTA, MV_U, MV_K2_NUMERATOR
from .G_min import min_G_lower_bound


def safe_sqrt(x: arb) -> arb:
    """Rigorous arb enclosure of sqrt on the non-negative part of x's ball.

    Semantics:
      - If x.upper() < 0   : raise ValueError (nothing to take sqrt of).
      - If x.lower() >= 0  : return x.sqrt() (exact arb).
      - Else               : return hull of [0, sqrt(x.upper())].
        This is a conservative super-set of {sqrt(v) : v in x AND v >= 0}.
    """
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(
            f"safe_sqrt: x.upper() = {x_up} < 0; no non-negative sub-interval."
        )
    x_lo = x.lower()
    if x_lo >= 0:
        return x.sqrt()
    # Ambiguous or mixed-sign: hull of [0, sqrt(x.upper())]
    upper = x_up.sqrt()
    return arb(0).union(upper)


@dataclass(frozen=True)
class PhiParams:
    """Compiled, rigorous inputs to Phi(M, y) at N = 1.

    All fields are arb (intervals) or fmpq (exact rationals).  Construct via
    ``PhiParams.from_mv(...)`` or pass explicit values.
    """
    delta:    fmpq
    u:        fmpq
    K2:       arb          # regularised MV bound on ||K||_2^2; arb for arithmetic
    k1:       arb          # |J_0(pi delta)|^2 (period-1)
    gain_a:   arb          # a = (4/u) m^2 / S_1
    min_G:    arb          # arb whose .lower() is the certified rigorous lower bound on m
    S1:       arb          # sum_{j=1}^n a_j^2 / |J_0(pi j delta/u)|^2
    n_coeffs: int
    # Diagnostic:
    min_G_center: fmpq     # cell-center where min-cert was attained

    @classmethod
    def from_mv(
        cls,
        delta: fmpq = MV_DELTA,
        u: fmpq = MV_U,
        coeffs: Sequence[fmpq] = None,
        K2_times_delta: fmpq = MV_K2_NUMERATOR,
        n_cells_min_G: int = 8192,
        prec_bits: int = 256,
    ) -> "PhiParams":
        """Compile Phi inputs from MV's parameters; uses arb interval arith."""
        if coeffs is None:
            coeffs = mv_coeffs_fmpq()
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            # k_1 (period-1)
            k1_arb = k1_period_one(delta, prec_bits=prec_bits)
            # K2 bound: K2_times_delta / delta  (exact fmpq, then arb)
            K2_arb = arb(K2_times_delta) / arb(delta)
            # S_1 = sum a_j^2 / |J_0(pi j delta/u)|^2
            S1 = arb(0)
            for j, a_j in enumerate(coeffs, start=1):
                j0 = j0_pi_j_delta_over_u(j, delta, u, prec_bits=prec_bits)
                S1 = S1 + (arb(a_j) * arb(a_j)) / (j0 * j0)
            # min G via rigorous Taylor B&B
            min_G_encl, min_G_center = min_G_lower_bound(
                coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
            )
            # For the gain formula, use an UNDERESTIMATE of min_G (= min_G.lower())
            # so that gain is rigorously UNDERESTIMATED (a smaller a gives a
            # conservative, i.e. weaker, lower bound on M -- still sound).
            min_G_cert_arb = min_G_encl.lower()
            # But if .lower() is negative, gain formula breaks; guard.
            if min_G_cert_arb.upper() <= 0:
                raise ValueError(
                    f"min G certified lower bound is non-positive: {min_G_cert_arb}"
                )
            gain_a = (arb(4) / arb(u)) * (min_G_cert_arb * min_G_cert_arb) / S1
            return cls(
                delta=delta,
                u=u,
                K2=K2_arb,
                k1=k1_arb,
                gain_a=gain_a,
                min_G=min_G_cert_arb,
                S1=S1,
                n_coeffs=len(coeffs),
                min_G_center=min_G_center,
            )
        finally:
            ctx.prec = old


def mu_of_M(M: arb) -> arb:
    """mu(M) := M * sin(pi / M) / pi    (Lemma 3.4 bathtub bound on |hat h(1)|).

    For M in (1, 2], mu(M) is in (0, 1) and monotone in M.
    """
    return M * (arb.pi() / M).sin() / arb.pi()


def phi_N1(
    M: arb,
    y: arb,
    params: PhiParams,
) -> arb:
    """Rigorous arb enclosure of Phi(M, y) at N = 1, spec sign.

    Args:
        M:      arb enclosing ||f*f||_inf (typically a point or a small ball).
        y:      arb enclosing z_1^2 = |hat f(1)|^2 (may be a cell).
        params: precompiled PhiParams.

    Returns:
        arb enclosure of Phi(M, y).  For certification, ``.upper() < 0`` proves
        the cell is forbidden.

    Raises:
        ValueError if either radicand has upper < 0 (non-physical regime).
    """
    two = arb(2)
    # Radicand 1: M - 1 - 2 y^2
    y_sq = y * y
    rad1 = M - arb(1) - two * y_sq
    # Radicand 2: K2 - 1 - 2 k_1^2
    rad2 = params.K2 - arb(1) - two * params.k1 * params.k1
    # sqrt (clamped to non-negative)
    s1 = safe_sqrt(rad1)
    s2 = safe_sqrt(rad2)
    # Phi
    rhs = M + arb(1) + two * y * params.k1 + s1 * s2
    lhs = arb(2) / arb(params.u) + params.gain_a
    return rhs - lhs


__all__ = [
    "PhiParams",
    "phi_N1",
    "mu_of_M",
    "safe_sqrt",
]
