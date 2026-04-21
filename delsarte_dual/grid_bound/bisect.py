"""Top-level driver: bisect on M and emit a rational certificate.

Given compiled Phi parameters (from the MV setup), bisect on M in the
bracket [M_lo, M_hi] -- M_lo is always infeasible (certified Phi < 0
everywhere admissible), M_hi is always (possibly) feasible.  The largest
certifiable M_lo becomes M_cert, our rigorous lower bound on C_{1a}.

The certificate emitted is a self-contained JSON file:
  - M_cert as exact fmpq (numerator / denominator)
  - All MV input parameters (delta, u, K2 bound, coeffs provenance)
  - Compiled quantities: k_1, S_1, gain_a, min_G cell, as arb midpoint-radius
  - The terminal cell list for the certifying run at M = M_cert
  - The arb-interval precision used
  - SHA-256 of the certificate body (excluding the hash field itself)

The independent verifier (``certify.py``) consumes this JSON and re-checks
every quantitative claim using only python-flint primitives, with NO
imports from this package or its dependencies.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from flint import arb, fmpq, ctx

from .phi import PhiParams
from .cell_search import certify_phi_negative, CellSearchResult, Cell
from .coeffs import MV_DELTA, MV_U, MV_K2_NUMERATOR


def _fmpq_to_str(q: fmpq) -> str:
    return f"{q.p}/{q.q}"


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def _arb_to_str(a: arb) -> str:
    """Serialise an arb as a midpoint-radius decimal pair (for re-reading).

    We record ``str(a)`` which is Arb's own canonical ``[mid +/- rad]`` form
    plus the float approximations of mid and upper/lower for quick reading.
    """
    mid = a.mid()
    rad = a.rad()
    return json.dumps({
        "repr": str(a),
        "mid_float": float(mid),
        "rad_float": float(rad),
        "lower_float": float(a.lower()),
        "upper_float": float(a.upper()),
    })


@dataclass
class CertifiedBound:
    M_cert_q: fmpq
    cell_search: CellSearchResult
    params: PhiParams
    prec_bits: int
    bisection_history: list


def bisect_M_cert(
    params: PhiParams,
    M_lo_init: fmpq = fmpq(127, 100),     # 1.27 -- MV's provable starting bracket
    M_hi_init: fmpq = fmpq(1276, 1000),   # 1.276 -- MV's theoretical ceiling
    tol_q: fmpq = fmpq(1, 10**5),         # bisection resolution on M (= 1e-5)
    max_cells_per_M: int = 100000,
    initial_splits: int = 32,
    prec_bits: int = 256,
    verbose: bool = True,
) -> CertifiedBound:
    """Bisect to find the largest M with a certified Phi < 0 witness.

    The bracket invariant:
        M_lo was certified forbidden  (Phi < 0 on all admissible y).
        M_hi was NOT certified        (either truly feasible, or refinement budget hit).
    Start: M_lo_init must be certifiable.  If not, we raise.
    """
    history = []
    # Confirm M_lo_init is certifiable.
    M_lo = M_lo_init
    M_hi = M_hi_init
    if verbose:
        print(f"Initial bracket: [{_fmpq_to_float(M_lo):.6f}, {_fmpq_to_float(M_hi):.6f}]")
    first = certify_phi_negative(
        arb(M_lo), params,
        max_cells=max_cells_per_M,
        initial_splits=initial_splits,
        prec_bits=prec_bits,
    )
    history.append({
        "M_q": _fmpq_to_str(M_lo),
        "M_float": _fmpq_to_float(M_lo),
        "verdict": first.verdict,
        "cells_processed": first.cells_processed,
    })
    if first.verdict != "CERTIFIED_FORBIDDEN":
        raise RuntimeError(
            f"M_lo_init = {_fmpq_to_float(M_lo):.6f} could not be certified "
            f"forbidden (verdict: {first.verdict}); widen the bracket."
        )
    last_good_result = first

    # Bisect
    while M_hi - M_lo > tol_q:
        M_mid = (M_lo + M_hi) / fmpq(2)
        M_mid_arb = arb(M_mid)
        res = certify_phi_negative(
            M_mid_arb, params,
            max_cells=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
        )
        history.append({
            "M_q": _fmpq_to_str(M_mid),
            "M_float": _fmpq_to_float(M_mid),
            "verdict": res.verdict,
            "cells_processed": res.cells_processed,
        })
        if verbose:
            print(
                f"  mid = {_fmpq_to_float(M_mid):.6f} -> {res.verdict:22s}  "
                f"(cells={res.cells_processed})"
            )
        if res.verdict == "CERTIFIED_FORBIDDEN":
            M_lo = M_mid
            last_good_result = res
        else:
            M_hi = M_mid

    return CertifiedBound(
        M_cert_q=M_lo,
        cell_search=last_good_result,
        params=params,
        prec_bits=prec_bits,
        bisection_history=history,
    )


def emit_certificate(bound: CertifiedBound, filepath: str) -> str:
    """Write a replayable JSON certificate and return its SHA-256 hex."""
    p = bound.params
    body = {
        "format_version": 1,
        "kind": "grid_bound_N1_MV_reproduction",
        "note": (
            "Phase 1 certificate: MV 1.2748 reproduction through rigorous "
            "arb-interval cell-interval refinement at N=1. "
            "Sign convention: Phi >= 0  =>  admissible (spec sign)."
        ),
        "inputs": {
            "delta_q": _fmpq_to_str(p.delta),
            "u_q": _fmpq_to_str(p.u),
            "K2_times_delta_q": _fmpq_to_str(MV_K2_NUMERATOR),
            "n_coeffs": p.n_coeffs,
            "coeffs_source": (
                "arXiv:0907.1379 Appendix (verbatim 8-digit decimals, "
                "treated as exact rationals)"
            ),
        },
        "input_assumptions": {
            "K2_upper_bound": (
                "MV state ||K||_2^2 < 0.5747/delta (p.3 line 141); K is not "
                "in L^2 so this is a regularised surrogate inherited from "
                "Martin-O'Bryant arXiv:0807.5121 [MO Lemma 3.2].  Phase 1 "
                "takes this as a named input assumption; a Phase-2+ task "
                "is to re-derive it self-containedly."
            ),
        },
        "compiled": {
            "k1_period1":  _arb_to_str(p.k1),
            "K2_arb":      _arb_to_str(p.K2),
            "S1_arb":      _arb_to_str(p.S1),
            "min_G_cert":  _arb_to_str(p.min_G),
            "min_G_cell_center_q": _fmpq_to_str(p.min_G_center),
            "gain_a":      _arb_to_str(p.gain_a),
        },
        "M_cert": {
            "rational": _fmpq_to_str(bound.M_cert_q),
            "float":    _fmpq_to_float(bound.M_cert_q),
        },
        "cell_search_at_M_cert": {
            "verdict":     bound.cell_search.verdict,
            "n_terminal":  len(bound.cell_search.terminal_cells),
            "worst_terminal_phi_upper": (
                bound.cell_search.worst_cell.phi_upper_float
                if bound.cell_search.worst_cell else None
            ),
            "cells_processed": bound.cell_search.cells_processed,
            "terminal_cells":  [
                {
                    "cell":             r.cell.to_dict(),
                    "phi_upper_float":  r.phi_upper_float,
                    "phi_arb":          r.phi_arb_str,
                }
                for r in bound.cell_search.terminal_cells
            ],
        },
        "bisection_history": bound.bisection_history,
        "prec_bits": bound.prec_bits,
    }

    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)

    return digest


def main(argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(description=(
        "Grid-bound MV reproduction (Phase 1): bisect on M and emit a "
        "rational certificate.  Target M_cert close to MV's 1.2748."
    ))
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--n-cells-min-G", type=int, default=8192)
    parser.add_argument("--max-cells-per-M", type=int, default=100000)
    parser.add_argument("--initial-splits", type=int, default=32)
    parser.add_argument("--tol", type=str, default="1/100000")
    parser.add_argument(
        "--out",
        default="delsarte_dual/grid_bound/certificates/phase1_mv.json",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Phase 1 -- rigorous reproduction of MV's 1.2748 lower bound on C_{1a}")
    print("=" * 70)
    print()
    print(f"Compiling Phi parameters (prec_bits={args.prec_bits}, n_cells_min_G={args.n_cells_min_G})...")
    params = PhiParams.from_mv(
        n_cells_min_G=args.n_cells_min_G,
        prec_bits=args.prec_bits,
    )
    print(f"  delta     = {params.delta}")
    print(f"  u         = {params.u}")
    print(f"  min_G cert= {params.min_G}")
    print(f"  gain_a    = {params.gain_a}")
    print(f"  k_1       = {params.k1}")
    print(f"  K_2       = {params.K2}")
    print()

    # Parse tol
    tol_parts = args.tol.split("/")
    if len(tol_parts) == 2:
        tol_q = fmpq(int(tol_parts[0]), int(tol_parts[1]))
    else:
        tol_q = fmpq(int(tol_parts[0]))

    bound = bisect_M_cert(
        params,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M,
        initial_splits=args.initial_splits,
        prec_bits=args.prec_bits,
        verbose=True,
    )

    print()
    print(f"Certified M_cert = {bound.M_cert_q}  (float: {_fmpq_to_float(bound.M_cert_q):.6f})")
    print(f"MV's published:    1.2748")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    digest = emit_certificate(bound, args.out)
    print(f"Certificate written: {args.out}")
    print(f"SHA-256: {digest}")

    return bound


if __name__ == "__main__":
    main()
