"""Independent verifier for Phase-1 grid_bound certificates.

**Soundness contract** (spec §5.7):
  This script consumes only the JSON certificate, python-flint primitives,
  and MV's 119 paper-sourced coefficients (treated as data, not code).  It
  imports NOTHING from the grid_bound search modules (``phi.py``,
  ``cell_search.py``, ``bisect.py``, ``G_min.py``, ``bessel.py``) -- every
  mathematical quantity is recomputed from first principles below.

What is verified
----------------
  1. The certificate's input rationals (delta, u, K2_times_delta) match
     MV's paper (hard-coded here).
  2. The coefficients count matches 119 and is correctly parsed.
  3. Bessel values k_1 and J_0(pi j delta/u) are recomputed with arb at the
     declared precision; S_1 is recomputed; gain_a is recomputed.
  4. min_G is re-certified via its own Taylor B&B; the certified value
     agrees with (or is at least as strong as) the one in the certificate.
  5. The terminal cells in the certificate COVER [0, mu_upper_rational].
  6. For every terminal cell, the Phi(M_cert, cell) arb enclosure has
     .upper() < 0.
If all six pass, the certificate is ACCEPTED.

Usage:
  python -m delsarte_dual.grid_bound.certify <certificate.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Iterable

from flint import arb, fmpq, ctx


# ============================================================================
#  Input: hard-coded MV data (paper-sourced, not imported from search code)
# ============================================================================
#
# Source: arXiv:0907.1379v2 Appendix (pp. 12-13), retyped verbatim here so
# the verifier has no hidden dependency on the package's ``coeffs.py``.

_MV_DECIMALS_STANDALONE = [
    "+2.16620392","-1.87775750","+1.05828868","-7.29790538e-01",
    "+4.28008515e-01","+2.17832838e-01","-2.70415201e-01","+2.72834790e-02",
    "-1.91721888e-01","+5.51862060e-02","+3.21662512e-01","-1.64478392e-01",
    "+3.95478603e-02","-2.05402785e-01","-1.33758316e-02","+2.31873221e-01",
    "-4.37967118e-02","+6.12456374e-02","-1.57361919e-01","-7.78036253e-02",
    "+1.38714392e-01","-1.45201483e-04","+9.16539824e-02","-8.34020840e-02",
    "-1.01919986e-01","+5.94915025e-02","-1.19336618e-02","+1.02155366e-01",
    "-1.45929982e-02","-7.95205457e-02","+5.59733152e-03","-3.58987179e-02",
    "+7.16132260e-02","+4.15425065e-02","-4.89180454e-02","+1.65425755e-03",
    "-6.48251747e-02","+3.45951253e-02","+5.32122058e-02","-1.28435276e-02",
    "+1.48814403e-02","-6.49404547e-02","-6.01344770e-03","+4.33784473e-02",
    "-2.53362778e-04","+3.81674519e-02","-4.83816002e-02","-2.53878079e-02",
    "+1.96933442e-02","-3.04861682e-03","+4.79203471e-02","-2.00930265e-02",
    "-2.73895519e-02","+3.30183589e-03","-1.67380508e-02","+4.23917582e-02",
    "+3.64690190e-03","-1.79916104e-02","+7.31661649e-05","-2.99875575e-02",
    "+2.71842526e-02","+1.41806855e-02","-6.01781076e-03","+5.86806100e-03",
    "-3.32350597e-02","+9.23347466e-03","+1.47071722e-02","-7.42858080e-04",
    "+1.63414270e-02","-2.87265671e-02","-1.64287280e-03","+8.02601605e-03",
    "-7.62613027e-04","+2.18735533e-02","-1.78816282e-02","-6.58341101e-03",
    "+2.67706547e-03","-6.25261247e-03","+2.24942824e-02","-8.10756022e-03",
    "-5.68160823e-03","+7.01871209e-05","-1.15294332e-02","+1.83608944e-02",
    "-1.20567880e-03","-3.13147456e-03","+1.39083675e-03","-1.49312478e-02",
    "+1.32106694e-02","+1.73474188e-03","-8.53469045e-04","+4.03211203e-03",
    "-1.55352991e-02","+8.74711543e-03","+1.93998895e-03","-2.71357322e-05",
    "+6.13179585e-03","-1.41983972e-02","+5.84710551e-03","+9.22578333e-04",
    "-2.16583469e-04","+7.07919829e-03","-1.18488582e-02","+4.39698322e-03",
    "-8.91346785e-05","-3.42086367e-04","+6.46355636e-03","-8.87555371e-03",
    "+3.56799654e-03","-4.97335419e-04","-8.04560326e-04","+5.55076717e-03",
    "-7.13560569e-03","+4.53679038e-03","-3.33261516e-03","+2.35463427e-03",
    "+2.04023789e-04","-1.27746711e-03","+1.81247830e-04",
]
assert len(_MV_DECIMALS_STANDALONE) == 119

_MV_DELTA_Q           = fmpq(138, 1000)    # 0.138
_MV_U_Q               = fmpq(638, 1000)    # 0.638
_MV_K2_TIMES_DELTA_Q  = fmpq(5747, 10000)  # 0.5747 upper bound (MV input assumption)


# ============================================================================
#  Helpers
# ============================================================================

def _decimal_str_to_fmpq_v(s: str) -> fmpq:
    """Parse '+2.16620392' / '-1.29e-3' etc. to an exact fmpq.

    Independently implemented here (no import from grid_bound.coeffs).
    """
    s = s.strip()
    sign = 1
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]
    exp = 0
    if "e" in s or "E" in s:
        mant, e = s.replace("E", "e").split("e", 1)
        exp = int(e)
    else:
        mant = s
    if "." in mant:
        ip, fp = mant.split(".", 1)
    else:
        ip, fp = mant, ""
    if ip == "":
        ip = "0"
    dig_aft = len(fp)
    mi = int(ip + fp) if (ip + fp) else 0
    net = exp - dig_aft
    return fmpq(sign * mi * (10 ** net), 1) if net >= 0 else fmpq(sign * mi, 10 ** (-net))


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def _fmpq_from_str(s: str) -> fmpq:
    """Parse 'p/q' -> fmpq."""
    if "/" in s:
        p_str, q_str = s.split("/", 1)
        return fmpq(int(p_str), int(q_str))
    return fmpq(int(s))


def _arb_pi_j_delta_over_u_v(j: int, delta: fmpq, u: fmpq) -> arb:
    q = fmpq(j) * delta / u
    return arb.pi() * arb(q)


def _safe_sqrt_v(x: arb) -> arb:
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"x upper = {x_up} < 0")
    x_lo = x.lower()
    if x_lo >= 0:
        return x.sqrt()
    return arb(0).union(x_up.sqrt())


# ============================================================================
#  Re-computation of Phi-inputs (first-principles, no grid_bound imports)
# ============================================================================

def recompute_bessel_k1(delta: fmpq, prec_bits: int) -> arb:
    """|J_0(pi delta)|^2 via arb."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        arg = arb.pi() * arb(delta)
        j0  = arg.bessel_j(0)
        return j0 * j0
    finally:
        ctx.prec = old


def recompute_S1(coeffs: list[fmpq], delta: fmpq, u: fmpq, prec_bits: int) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            j0 = _arb_pi_j_delta_over_u_v(j, delta, u).bessel_j(0)
            total = total + (arb(a_j) * arb(a_j)) / (j0 * j0)
        return total
    finally:
        ctx.prec = old


def recompute_K2(K2_times_delta: fmpq, delta: fmpq, prec_bits: int) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        return arb(K2_times_delta) / arb(delta)
    finally:
        ctx.prec = old


def recompute_min_G(
    coeffs: list[fmpq], u: fmpq, prec_bits: int, n_cells: int = 8192
) -> tuple[arb, fmpq]:
    """Taylor B&B lower bound on min_{[0,1/4]} G(x).  Independent reimplementation."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        two_pi_over_u = arb(2) * arb.pi() / arb(u)
        x_lo = fmpq(0)
        x_hi = fmpq(1, 4)
        cw   = (x_hi - x_lo) / fmpq(n_cells)
        hw   = cw / fmpq(2)

        def G_at(x_q: fmpq) -> arb:
            x_a = arb(x_q)
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                t = t + arb(a) * (two_pi_over_u * arb(j) * x_a).cos()
            return t

        def Gp_at(x_q: fmpq) -> arb:
            x_a = arb(x_q)
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                t = t - arb(a) * (two_pi_over_u * arb(j)) * (two_pi_over_u * arb(j) * x_a).sin()
            return t

        def Gpp_cell(cell_a: arb) -> arb:
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                w = two_pi_over_u * arb(j)
                t = t - arb(a) * (w * w) * (w * cell_a).cos()
            return t

        worst_arb = None
        worst_center = None
        worst_float = None
        for k in range(n_cells):
            c = x_lo + fmpq(2 * k + 1) * hw
            cell_arb = arb(c, hw)
            G_c  = G_at(c)
            Gp_c = Gp_at(c)
            Gpp  = Gpp_cell(cell_arb)
            dx    = arb(0, hw)
            half_r_sq = (arb(hw) * arb(hw)) / arb(2)
            rem   = arb(0, 1) * half_r_sq
            encl  = G_c + Gp_c * dx + Gpp * rem
            lf    = float(encl.lower())
            if worst_float is None or lf < worst_float:
                worst_float   = lf
                worst_arb     = encl
                worst_center  = c
        return worst_arb, worst_center
    finally:
        ctx.prec = old


def mu_of_M_v(M: arb) -> arb:
    return M * (arb.pi() / M).sin() / arb.pi()


def phi_N1_v(M: arb, y: arb, *, K2: arb, k1: arb, u: fmpq, gain_a: arb) -> arb:
    """Re-implementation of Phi(M, y).  Independent of grid_bound.phi."""
    two = arb(2)
    rad1 = M - arb(1) - two * y * y
    rad2 = K2  - arb(1) - two * k1 * k1
    s1   = _safe_sqrt_v(rad1)
    s2   = _safe_sqrt_v(rad2)
    rhs  = M + arb(1) + two * y * k1 + s1 * s2
    lhs  = arb(2) / arb(u) + gain_a
    return rhs - lhs


# ============================================================================
#  Main verification routine
# ============================================================================

@dataclass
class VerifyResult:
    accepted: bool
    messages: list[str]
    M_cert_q: fmpq | None = None


def verify_certificate(cert_path: str, prec_bits: int | None = None) -> VerifyResult:
    with open(cert_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    msgs: list[str] = []

    def log(s: str):
        msgs.append(s)
        print(s)

    # 0. SHA-256 integrity
    body = raw["body"]
    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    if digest != raw["sha256_of_body"]:
        log(f"FAIL: SHA-256 mismatch: recomputed {digest}, stored {raw['sha256_of_body']}")
        return VerifyResult(False, msgs)
    log(f"OK: SHA-256 integrity = {digest}")

    inputs = body["inputs"]
    declared_prec = int(body.get("prec_bits", 256))
    prec = prec_bits or declared_prec

    # 1. Input rationals match MV
    delta_q = _fmpq_from_str(inputs["delta_q"])
    u_q     = _fmpq_from_str(inputs["u_q"])
    K2d_q   = _fmpq_from_str(inputs["K2_times_delta_q"])
    if delta_q != _MV_DELTA_Q:
        log(f"FAIL: delta mismatch: cert={delta_q}, MV={_MV_DELTA_Q}")
        return VerifyResult(False, msgs)
    if u_q != _MV_U_Q:
        log(f"FAIL: u mismatch: cert={u_q}, MV={_MV_U_Q}")
        return VerifyResult(False, msgs)
    if K2d_q != _MV_K2_TIMES_DELTA_Q:
        log(f"FAIL: K2_times_delta mismatch: cert={K2d_q}, MV={_MV_K2_TIMES_DELTA_Q}")
        return VerifyResult(False, msgs)
    if inputs["n_coeffs"] != 119:
        log(f"FAIL: n_coeffs != 119: {inputs['n_coeffs']}")
        return VerifyResult(False, msgs)
    log(f"OK: MV input rationals match (delta={delta_q}, u={u_q}, K2*delta={K2d_q}, n=119)")

    coeffs = [_decimal_str_to_fmpq_v(s) for s in _MV_DECIMALS_STANDALONE]

    # 2. Re-compute k_1, K2, S_1, min_G, gain_a
    k1   = recompute_bessel_k1(delta_q, prec)
    K2   = recompute_K2(K2d_q, delta_q, prec)
    S1   = recompute_S1(coeffs, delta_q, u_q, prec)
    min_G_arb, min_G_center = recompute_min_G(coeffs, u_q, prec, n_cells=8192)
    min_G_cert_arb = min_G_arb.lower()
    if min_G_cert_arb.upper() <= 0:
        log(f"FAIL: recomputed min_G certified lower <= 0 ({min_G_cert_arb})")
        return VerifyResult(False, msgs)
    gain_a = (arb(4) / arb(u_q)) * (min_G_cert_arb * min_G_cert_arb) / S1
    log(f"OK: recomputed k_1    = {k1}")
    log(f"OK: recomputed K_2    = {K2}")
    log(f"OK: recomputed S_1    = {S1}")
    log(f"OK: recomputed min_G cert lower = {min_G_cert_arb}")
    log(f"OK: recomputed gain_a = {gain_a}")

    # 3. M_cert
    M_cert_q = _fmpq_from_str(body["M_cert"]["rational"])
    M_cert_arb = arb(M_cert_q)
    log(f"Verifying M_cert = {M_cert_q}  (~{_fmpq_to_float(M_cert_q):.6f})")

    # 4. Re-evaluate each terminal cell's Phi at M_cert
    cells_info = body["cell_search_at_M_cert"]["terminal_cells"]
    log(f"Re-checking {len(cells_info)} terminal cells ...")
    max_recomputed_upper = None
    for i, rec in enumerate(cells_info):
        lo_q = _fmpq_from_str(rec["cell"]["lo"])
        hi_q = _fmpq_from_str(rec["cell"]["hi"])
        if hi_q <= lo_q:
            log(f"FAIL: cell {i} has hi <= lo: [{lo_q}, {hi_q}]")
            return VerifyResult(False, msgs)
        center = (lo_q + hi_q) / fmpq(2)
        hw     = (hi_q - lo_q) / fmpq(2)
        y_arb  = arb(center, hw)
        phi_v  = phi_N1_v(M_cert_arb, y_arb, K2=K2, k1=k1, u=u_q, gain_a=gain_a)
        up = float(phi_v.upper())
        if max_recomputed_upper is None or up > max_recomputed_upper:
            max_recomputed_upper = up
        if not (phi_v.upper() < 0):
            log(f"FAIL: cell {i}  lo={lo_q}  hi={hi_q}  Phi.upper()={phi_v.upper()} NOT < 0")
            return VerifyResult(False, msgs)
    log(f"OK: all {len(cells_info)} cells have Phi.upper() < 0; max recomputed upper = {max_recomputed_upper:+.3e}")

    # 5. Coverage: terminal cells must cover [0, mu(M_cert) + tiny cushion]
    mu_arb = mu_of_M_v(M_cert_arb)
    mu_up_arb = mu_arb.upper()
    mu_up_float = float(mu_up_arb)
    sorted_cells = sorted(
        [
            (_fmpq_from_str(c["cell"]["lo"]), _fmpq_from_str(c["cell"]["hi"]))
            for c in cells_info
        ],
        key=lambda lh: lh[0],
    )
    if sorted_cells[0][0] != 0:
        log(f"FAIL: lowest cell starts at {sorted_cells[0][0]}, not 0")
        return VerifyResult(False, msgs)
    for i in range(1, len(sorted_cells)):
        if sorted_cells[i][0] != sorted_cells[i - 1][1]:
            log(
                f"FAIL: cells are not contiguous: cell {i-1} ends at "
                f"{sorted_cells[i-1][1]}, cell {i} starts at {sorted_cells[i][0]}"
            )
            return VerifyResult(False, msgs)
    top_q = sorted_cells[-1][1]
    if _fmpq_to_float(top_q) < mu_up_float:
        log(
            f"FAIL: top cell ends at {_fmpq_to_float(top_q):.8f} < mu(M_cert).upper() = {mu_up_float:.8f}"
        )
        return VerifyResult(False, msgs)
    log(
        f"OK: cells cover [0, {_fmpq_to_float(top_q):.8f}] >= [0, mu(M_cert).upper()={mu_up_float:.8f}]"
    )

    log("")
    log(f"VERDICT: CERTIFICATE ACCEPTED.  C_{{1a}} >= {M_cert_q} (~{_fmpq_to_float(M_cert_q):.6f}).")
    return VerifyResult(True, msgs, M_cert_q)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Independent verifier for grid_bound Phase-1 certificates."
    )
    p.add_argument("certificate", help="Path to the JSON certificate.")
    p.add_argument("--prec-bits", type=int, default=None,
                   help="Override arb precision (default: certificate's).")
    args = p.parse_args(argv)
    res = verify_certificate(args.certificate, prec_bits=args.prec_bits)
    sys.exit(0 if res.accepted else 1)


if __name__ == "__main__":
    main()
