"""Per-kernel N=1 Phi bisection sweep.

For each kernel in the sweep, we:
  (1) Verify admissibility (supp, K>=0, int K=1 by construction; tilde K >= 0
      via rigorous Bochner check up to n_max).
  (2) Re-optimise the trig polynomial G(x) via the per-kernel QP (MOSEK/
      CLARABEL), with weights w_j = hat_K_R(j/u).
  (3) Compile rigorous PhiParams with the kernel's k_1 = hat_K_R(1),
      K_2 = ||K||_2^2 (arb), and re-optimised coefficients.  A rigorous
      lower bound on min_{[0,1/4]} G is certified by Taylor B&B
      (``grid_bound/G_min.py``).
  (4) Run the N=1 cell-search bisection (``grid_bound/cell_search.py``) on
      M in a bracket; record the certified M_cert.

Output: ``delsarte_dual/grid_bound_alt_kernel/kernel_sweep_results.json``.

The output also includes non-admissible kernels with ``admissible: false``
and ``M_cert: null`` entries so the full sweep is transparent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.phi import PhiParams, phi_N1
from delsarte_dual.grid_bound.bessel import j0_pi_j_delta_over_u
from delsarte_dual.grid_bound.cell_search import certify_phi_negative
from delsarte_dual.grid_bound.G_min import min_G_lower_bound
from delsarte_dual.grid_bound.bisect import bisect_M_cert, _fmpq_to_float, _fmpq_to_str

from .kernels import (
    Kernel,
    ArcsineKernel,
    default_kernel_registry,
)
from .optimize_G import solve_qp_for_kernel


def compile_phi_params_for_kernel(
    kernel: Kernel,
    coeffs: list,                 # list of fmpq from QP
    u: fmpq = fmpq(638, 1000),
    n_cells_min_G: int = 4096,
    prec_bits: int = 256,
) -> PhiParams:
    """Build a PhiParams object using kernel-specific k_1, K_2, and S_1.

    The G-coefficients ``coeffs`` come from ``solve_qp_for_kernel(kernel)``;
    they are the a_j values of G(x) = sum a_j cos(2 pi j x / u).

    Rigorous quantities:
      k_1 = hat_K_R(1)                 via kernel.K_tilde(1)
      K_2 = ||K||_2^2                  via kernel.K_norm_sq()
      S_1 = sum a_j^2 / hat_K_R(j/u)   via kernel.K_tilde_real(j/u)
      min_G = rigorous Taylor B&B bound on min_{x in [0,1/4]} G
      gain_a = (4/u) * min_G^2 / S_1
    """
    delta = kernel.supp_halfwidth
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        K2_arb = kernel.K_norm_sq(prec_bits=prec_bits)
        k1_arb = kernel.K_tilde(1, prec_bits=prec_bits)

        # S_1 (per-kernel weighted objective)
        S1 = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            xi = arb(fmpq(j)) / arb(u)
            w_j = kernel.K_tilde_real(xi, prec_bits=prec_bits)
            # Guard: if w_j <= 0 (Bochner fails), term is ill-defined; skip
            # but flag in outputs.  (Safe: setting a_j = 0 at that j doesn't
            # change S_1 by construction when Bochner-admissible.)
            if w_j.lower() <= 0:
                a_j_f = float(a_j.p) / float(a_j.q)
                if a_j_f != 0.0:
                    raise ValueError(
                        f"kernel {kernel.name}: Bochner violated at j={j} "
                        f"(w_j={w_j}), but coeff a_{j} != 0; cannot compile"
                    )
                continue
            S1 = S1 + (arb(a_j) * arb(a_j)) / w_j

        # min_G (Taylor B&B lower bound)
        min_G_encl, min_G_center = min_G_lower_bound(
            coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
        )
        min_G_cert_arb = min_G_encl.lower()
        if min_G_cert_arb.upper() <= 0:
            raise ValueError(
                f"{kernel.name}: min_G certified lower bound is non-positive: "
                f"{min_G_cert_arb}"
            )
        gain_a = (arb(4) / arb(u)) * (min_G_cert_arb * min_G_cert_arb) / S1

        return PhiParams(
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


@dataclass
class KernelSweepEntry:
    kernel_name: str
    admissible: bool
    bochner_max_j_checked: int
    bochner_ok: bool
    tilde_K_1: float
    K_norm_sq: float
    S1: Optional[float]
    min_G_cert: Optional[float]
    gain_a: Optional[float]
    M_cert: Optional[float]
    M_cert_q: Optional[str]
    bisect_history: Optional[list]
    note: str
    wall_time_sec: float


def run_single_kernel(
    kernel: Kernel,
    u: fmpq = fmpq(638, 1000),
    n_coeffs: int = 119,
    n_grid_qp: int = 5001,
    n_cells_min_G: int = 4096,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 50000,
    initial_splits: int = 32,
    bochner_max: int = 100,
    prec_bits: int = 192,
    verbose: bool = True,
) -> KernelSweepEntry:
    """Run the full N=1 bisection pipeline for a single kernel.

    On admissibility failure (Bochner) or numerical issue, returns an entry
    with ``admissible=False`` and ``M_cert=None``.
    """
    t0 = time.time()
    note_parts = []

    # Step 1: Bochner check
    bochner_ok = True
    for j in range(1, bochner_max + 1):
        try:
            v = kernel.K_tilde(j, prec_bits=prec_bits)
        except Exception as e:
            bochner_ok = False
            note_parts.append(f"K_tilde({j}) raised: {e}")
            break
        if v.lower() < 0:
            bochner_ok = False
            note_parts.append(f"Bochner fails at j={j}: tilde_K(j) = {v}")
            break

    tilde_K_1 = float(kernel.K_tilde(1, prec_bits=prec_bits).mid())
    try:
        K2_f = float(kernel.K_norm_sq(prec_bits=prec_bits).mid())
    except Exception as e:
        K2_f = float('nan')
        note_parts.append(f"K_norm_sq raised: {e}")

    if not bochner_ok:
        return KernelSweepEntry(
            kernel_name=kernel.name,
            admissible=False,
            bochner_max_j_checked=bochner_max,
            bochner_ok=False,
            tilde_K_1=tilde_K_1,
            K_norm_sq=K2_f,
            S1=None, min_G_cert=None, gain_a=None,
            M_cert=None, M_cert_q=None,
            bisect_history=None,
            note=" | ".join(note_parts) or "Bochner violation",
            wall_time_sec=time.time() - t0,
        )

    # Step 2: QP re-optimisation
    try:
        qp_res = solve_qp_for_kernel(
            kernel, n=n_coeffs, u=u, n_grid=n_grid_qp,
            verbose=verbose,
        )
    except Exception as e:
        return KernelSweepEntry(
            kernel_name=kernel.name,
            admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
            tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
            S1=None, min_G_cert=None, gain_a=None,
            M_cert=None, M_cert_q=None,
            bisect_history=None,
            note=f"QP solve failed: {e}",
            wall_time_sec=time.time() - t0,
        )

    # Step 3: PhiParams compile
    try:
        params = compile_phi_params_for_kernel(
            kernel, qp_res.a_opt_fmpq, u=u,
            n_cells_min_G=n_cells_min_G, prec_bits=prec_bits,
        )
    except Exception as e:
        return KernelSweepEntry(
            kernel_name=kernel.name,
            admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
            tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
            S1=qp_res.S1_float, min_G_cert=None, gain_a=None,
            M_cert=None, M_cert_q=None,
            bisect_history=None,
            note=f"PhiParams compile failed: {e}",
            wall_time_sec=time.time() - t0,
        )

    S1_f = float(params.S1.mid())
    min_G_f = float(params.min_G.mid())
    gain_f = float(params.gain_a.mid())
    if verbose:
        print(f"  {kernel.name}: k1={tilde_K_1:.5f} K2={K2_f:.4f} "
              f"S1={S1_f:.4f} minG={min_G_f:.5f} gain={gain_f:.5f}")

    # Step 4: Bisection
    # First confirm lower bracket is certifiable at M_lo_init; if not, retreat.
    try:
        check = certify_phi_negative(
            arb(M_lo_init), params,
            max_cells=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
        )
        if check.verdict != "CERTIFIED_FORBIDDEN":
            # Retreat: try lower starting bracket
            for M_retry in [fmpq(125, 100), fmpq(12, 10), fmpq(115, 100), fmpq(110, 100)]:
                check2 = certify_phi_negative(
                    arb(M_retry), params,
                    max_cells=max_cells_per_M,
                    initial_splits=initial_splits,
                    prec_bits=prec_bits,
                )
                if check2.verdict == "CERTIFIED_FORBIDDEN":
                    M_lo_init = M_retry
                    break
            else:
                return KernelSweepEntry(
                    kernel_name=kernel.name,
                    admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
                    tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
                    S1=S1_f, min_G_cert=min_G_f, gain_a=gain_f,
                    M_cert=None, M_cert_q=None,
                    bisect_history=None,
                    note=f"no certifiable M_lo bracket (tried 1.10-1.27)",
                    wall_time_sec=time.time() - t0,
                )
    except Exception as e:
        return KernelSweepEntry(
            kernel_name=kernel.name,
            admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
            tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
            S1=S1_f, min_G_cert=min_G_f, gain_a=gain_f,
            M_cert=None, M_cert_q=None,
            bisect_history=None,
            note=f"bracket check raised: {e}",
            wall_time_sec=time.time() - t0,
        )

    try:
        bound = bisect_M_cert(
            params,
            M_lo_init=M_lo_init,
            M_hi_init=M_hi_init,
            tol_q=tol_q,
            max_cells_per_M=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
            verbose=False,
        )
    except Exception as e:
        return KernelSweepEntry(
            kernel_name=kernel.name,
            admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
            tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
            S1=S1_f, min_G_cert=min_G_f, gain_a=gain_f,
            M_cert=None, M_cert_q=None,
            bisect_history=None,
            note=f"bisect raised: {e}",
            wall_time_sec=time.time() - t0,
        )

    return KernelSweepEntry(
        kernel_name=kernel.name,
        admissible=True, bochner_max_j_checked=bochner_max, bochner_ok=True,
        tilde_K_1=tilde_K_1, K_norm_sq=K2_f,
        S1=S1_f, min_G_cert=min_G_f, gain_a=gain_f,
        M_cert=_fmpq_to_float(bound.M_cert_q),
        M_cert_q=_fmpq_to_str(bound.M_cert_q),
        bisect_history=bound.bisection_history,
        note=" | ".join(note_parts) or "ok",
        wall_time_sec=time.time() - t0,
    )


def run_sweep(
    kernels: Optional[list] = None,
    out_path: str = "delsarte_dual/grid_bound_alt_kernel/kernel_sweep_results.json",
    verbose: bool = True,
    **kwargs,
) -> list[KernelSweepEntry]:
    if kernels is None:
        kernels = default_kernel_registry()
    results = []
    for K in kernels:
        if verbose:
            print("=" * 60)
            print(f"KERNEL: {K.name}")
            print("=" * 60)
        entry = run_single_kernel(K, verbose=verbose, **kwargs)
        results.append(entry)
        if verbose:
            if entry.M_cert is not None:
                print(f"  --> M_cert = {entry.M_cert:.6f}  ({entry.wall_time_sec:.1f}s)")
            else:
                print(f"  --> SKIPPED: {entry.note}  ({entry.wall_time_sec:.1f}s)")

    # Persist results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serialisable = []
    for r in results:
        serialisable.append({
            "kernel_name": r.kernel_name,
            "admissible": r.admissible,
            "bochner_max_j_checked": r.bochner_max_j_checked,
            "bochner_ok": r.bochner_ok,
            "tilde_K_1": r.tilde_K_1,
            "K_norm_sq": r.K_norm_sq,
            "S1": r.S1,
            "min_G_cert": r.min_G_cert,
            "gain_a": r.gain_a,
            "M_cert": r.M_cert,
            "M_cert_q": r.M_cert_q,
            "bisect_history": r.bisect_history,
            "note": r.note,
            "wall_time_sec": r.wall_time_sec,
        })

    best = None
    best_nonK1 = None
    for r in results:
        if r.M_cert is None:
            continue
        if best is None or r.M_cert > best["M_cert"]:
            best = {"kernel_name": r.kernel_name, "M_cert": r.M_cert}
        if r.kernel_name != "K1_arcsine":
            if best_nonK1 is None or r.M_cert > best_nonK1["M_cert"]:
                best_nonK1 = {"kernel_name": r.kernel_name, "M_cert": r.M_cert}

    summary = {
        "best_overall": best,
        "best_nonK1": best_nonK1,
        "beats_MV_1_2748": (best is not None and best["M_cert"] > 1.2748),
        "breaks_1_28": (best is not None and best["M_cert"] > 1.28),
    }
    body = {
        "kind": "grid_bound_alt_kernel_sweep",
        "results": serialisable,
        "summary": summary,
    }
    body_json = json.dumps(body, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True, default=str)
    if verbose:
        print(f"\nResults: {out_path}")
        print(f"SHA-256: {digest}")
        print(f"Best: {best}")
        print(f"Best non-K1: {best_nonK1}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Reduced tolerance + fewer cells for quick run")
    p.add_argument("--only-K1", action="store_true",
                   help="Sanity: only run K1 arcsine")
    p.add_argument("--out", default="delsarte_dual/grid_bound_alt_kernel/kernel_sweep_results.json")
    args = p.parse_args()

    ks = default_kernel_registry()
    if args.only_K1:
        ks = [ks[0]]

    kwargs = dict(
        u=fmpq(638, 1000),
        n_coeffs=119,
        n_grid_qp=3001 if args.quick else 5001,
        n_cells_min_G=2048 if args.quick else 4096,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=fmpq(1, 10**3) if args.quick else fmpq(1, 10**4),
        max_cells_per_M=20000 if args.quick else 50000,
        initial_splits=32,
        bochner_max=50 if args.quick else 100,
        prec_bits=128 if args.quick else 192,
    )
    run_sweep(ks, out_path=args.out, verbose=True, **kwargs)
