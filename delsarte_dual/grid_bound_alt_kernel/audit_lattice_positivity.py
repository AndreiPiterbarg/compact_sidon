#!/usr/bin/env python3
"""Rigorous ``flint.arb`` audit of active-set lattice positivity.

This standalone certifier backs the sanctioned verifiable-by-computation
axiom

    ``Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active``

namely

    ``K_ms_fourier_lattice j > 0   for all  1 <= j <= 200``,

where the closed-form period-``u`` Fourier coefficient of the three-scale
arcsine kernel is

    ``K_ms_fourier_lattice(j) = sum_i lambda_i * J0(pi * delta_i * (j/u))^2``

with ``(delta_1, delta_2, delta_3) = (138, 55, 25)/1000``,
``(lambda_1, lambda_2, lambda_3) = (85, 10, 5)/100`` and ``u = 638/1000``.

These are *exactly* the QP active-set denominators ``w_j = hat K(j/u)``
that the production driver ``bisect_alt_kernel.py`` already divides by to
form ``S_1 = sum_j a_j^2 / w_j``; a zero (or sign-ambiguous) ``w_j`` would
make ``S_1`` ill-defined.  Here we verify positivity *directly and
explicitly*: each ``K_ms_fourier_lattice(j)`` is computed as an ``arb``
interval at 256-bit precision and we assert ``.lower() > 0`` (so the true
value is rigorously bounded away from zero by the certified lower
endpoint).

The check is a *logically decidable* finite conjunction of 200 strict
inequalities about specific real numbers; ``flint.arb`` decides it with a
proven interval enclosure (an analogue in role of MV 2010's Mathematica
``J0`` citations, but strictly more rigorous).

Run directly::

    python audit_lattice_positivity.py

or as a module::

    python -m delsarte_dual.grid_bound_alt_kernel.audit_lattice_positivity

Exit code is 0 iff every index is certified strictly positive.
"""

from __future__ import annotations

import os
import sys

from flint import arb, ctx, fmpq

# --- import bootstrap so the script runs both as a module and directly ---
try:  # normal package-relative import
    from .bisect_alt_kernel import (
        PROD_DELTAS,
        PROD_LAMBDAS,
        PROD_N_COEFFS,
        PROD_U,
        production_kernel,
    )
except ImportError:  # run as a bare script: add the repo root to sys.path
    _REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (  # type: ignore
        PROD_DELTAS,
        PROD_LAMBDAS,
        PROD_N_COEFFS,
        PROD_U,
        production_kernel,
    )


def k_ms_fourier_lattice(kernel, j: int, u: fmpq, prec_bits: int = 256) -> arb:
    """``K_ms_fourier_lattice(j) = sum_i lambda_i * J0(pi*delta_i*(j/u))^2``.

    Computed as an ``arb`` interval.  This matches the Lean definition
    ``Sidon.MultiScale.K_ms_fourier_lattice`` and the driver's
    ``w_j = kernel.K_tilde_real(j/u)`` exactly (frequency ``xi = j/u``).
    """
    xi = arb(fmpq(j)) / arb(u)
    return kernel.K_tilde_real(xi, prec_bits=prec_bits)


def audit(
    n_max: int = PROD_N_COEFFS,
    u: fmpq = PROD_U,
    prec_bits: int = 256,
    verbose: bool = True,
) -> tuple[bool, int, arb]:
    """Certify ``K_ms_fourier_lattice(j) > 0`` for ``1 <= j <= n_max``.

    Returns ``(all_positive, argmin_j, min_lower_bound)`` where
    ``min_lower_bound`` is the smallest *certified lower endpoint*
    ``min_{1<=j<=n_max} K_ms_fourier_lattice(j).lower()``.
    """
    kernel = production_kernel()
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        all_positive = True
        argmin_j = 1
        min_lower: arb | None = None
        failures: list[int] = []

        for j in range(1, n_max + 1):
            val = k_ms_fourier_lattice(kernel, j, u, prec_bits=prec_bits)
            low = val.lower()  # rigorous lower endpoint of the enclosure
            if not (low > 0):
                all_positive = False
                failures.append(j)
            if min_lower is None or low < min_lower:
                min_lower = low
                argmin_j = j

        if verbose:
            print(
                f"[audit_lattice_positivity] kernel = {kernel.name}\n"
                f"  range            : 1 <= j <= {n_max}\n"
                f"  precision        : {prec_bits}-bit arb\n"
                f"  deltas           : {[str(d) for d in PROD_DELTAS]}\n"
                f"  lambdas          : {[str(l) for l in PROD_LAMBDAS]}\n"
                f"  u                : {u}"
            )
            print(
                f"  min_j K_tilde(j) : {float(min_lower):.10g} "
                f"(certified lower endpoint, attained near j={argmin_j})"
            )
            if all_positive:
                print(
                    f"  RESULT           : CERTIFIED  "
                    f"K_ms_fourier_lattice(j) > 0  for all 1 <= j <= {n_max}"
                )
            else:
                print(
                    f"  RESULT           : FAILED at indices {failures}"
                )

        assert min_lower is not None
        return all_positive, argmin_j, min_lower
    finally:
        ctx.prec = old


def main() -> int:
    all_positive, argmin_j, min_lower = audit()
    # Hard assertion so the exit code reflects the rigorous result.
    assert all_positive, (
        "active-set lattice positivity FAILED: "
        f"min lower endpoint {float(min_lower):.6g} at j={argmin_j} is not > 0"
    )
    print(
        f"\nmin_{{1<=j<=200}} K_tilde(j) = {float(min_lower):.10g}  (> 0, certified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
