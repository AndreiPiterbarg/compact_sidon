"""Dual Lasserre SDP in MOSEK Task API — the canonical "sparse-LMI → dual"
transformation MOSEK's own FAQ recommends.

================================================================================
MATHEMATICAL BACKGROUND
================================================================================

**Primal Lasserre relaxation (what the rest of the repo sets up):**

    min   0
    s.t.  y_0 = 1                              (eq λ)
          y_α = Σ_i y_{α+e_i}  for |α|<2k       (eqs μ_α)
          y_α ≥ 0                               (ineqs v_α ≥ 0)
          M_k(y) ⪰ 0                            (PSD dual X_0 ⪰ 0)
          M_{k-1}(μ_i y) ⪰ 0                    (PSD duals X_i ⪰ 0)
          M_{k-1}((1-μ_i)y) ⪰ 0                 (PSD duals X'_i ⪰ 0)
          t·M_{k-1}(y) - Q_W(y) ⪰ 0             (PSD duals X_W ⪰ 0)

Here M_k(y) is a C(d+k,k)-sized symmetric matrix whose (β,γ) entry is
y_{β+γ}; M_{k-1}(μ_i y) has entries y_{β+γ+e_i}; M_{k-1}((1-μ_i)y) has
y_{β+γ} - y_{β+γ+e_i}; Q_W(y) has Σ_{i,j} M_W[i,j] y_{β+γ+e_i+e_j}.

**The Lagrangian stationarity equations** produce one equation per
primal variable y_α:

    [α=0]·λ
      + Σ_{α'∈consist} μ_{α'} · (#{i : α = α'+e_i} − 𝟙[α = α'])
      − v_α
      − ⟨X_0, E_0[α]⟩                                     moment matrix
      − Σ_i ⟨X_i, E_i[α]⟩                                 μ_i ⪰ 0 local.
      − Σ_i ⟨X'_i, E'_i[α]⟩                               (1−μ_i)⪰0 local.
      − Σ_W ⟨X_W, t·E_W^t[α] − E_W^Q[α]⟩                   windows
      = 0                                                 (one per α)

where the "sensitivity matrices" E_·[α] are:

    E_0[α]_{β,γ} = 𝟙[β+γ = α]               (moment)
    E_i[α]_{β,γ} = 𝟙[β+γ+e_i = α]           (μ_i localizing)
    E'_i[α]_{β,γ} = 𝟙[β+γ = α] − 𝟙[β+γ+e_i = α]  (1−μ_i localizing)
    E_W^t[α]_{β,γ} = 𝟙[β+γ = α]             (from t·M_{k-1}(y) in window)
    E_W^Q[α]_{β,γ} = Σ_{i,j} M_W[i,j] · 𝟙[β+γ+e_i+e_j = α]   (window Q)

**Primal-infeasibility certificate.**  The primal SDP is infeasible at
this t iff the system above is feasible with

    λ > 0,   v_α ≥ 0,   X_0, X_i, X'_i, X_W ⪰ 0

(Farkas for conic LP.)  By homogeneity we can normalise λ = 1.  The
resulting feasibility problem has ONE equality per y_α — hence the
Schur complement at MOSEK's IPM is (n_y × n_y)-sized, about 37K × 37K
at d=16 L3 z2_full + pre_elim.  Contrast with the primal formulation
whose Schur is dimensioned by the SDP-svec row count (≈ 3.4M at d=16).

The 90× reduction in Schur dimension is the reason the dual form
typically solves 10-100× faster than the primal on Lasserre SDPs
at scale.  MOSEK FAQ §3.15: "for sparse LMIs one should always input
the LMI in its dual form".

================================================================================
SYMMETRY / Z/2 COMPATIBILITY
================================================================================

This module consumes the ALREADY CANONICALIZED precompute P (i.e.
lasserre.z2_elim.canonicalize_z2 applied) — so every y index is an
orbit representative and the "one equation per y_α" count is halved
vs the raw Lasserre problem.  The moment matrix block-diagonalization
(lasserre.z2_blockdiag) replaces the single 969-dim X_0 with two
smaller blocks (X_0^sym of size F+P ≈ 489 and X_0^anti of size P ≈
480 at d=16 L3), which halves the dense factor cost on the dominant
moment cone.  σ-paired localizing and window cones are kept as-is
(one per orbit representative); dropping their σ-partners is already
folded into the canonical precompute.

================================================================================
IMPLEMENTATION
================================================================================

We emit the SDP directly to mosek.Task (not mosek.fusion.Model):

  • appendbarvars([...]) — all PSD dual matrices in one bulk call.
  • appendvars(n_scalar)  — free μ_α, free λ, non-neg v_α.
  • appendcons(n_y) — one stationarity equation per primal y variable.
  • putbaraijlist(...) — bulk coefficient submission for ⟨X_·, E_·[α]⟩.
  • putaijlist(...)    — bulk scalar coefficient submission.
  • putconboundslice / putvarboundslice — bounds.

Objective: min −λ (equivalent to "maximise λ"); Farkas holds iff
optimum < 0.  Once λ = 1 certified, lb ≥ t is proven.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mosek


# =====================================================================
# Helper: enumerate sensitivity matrix entries for each α
# =====================================================================

def _enumerate_moment_sensitivity(basis_from: List[Tuple[int, ...]],
                                    basis_to: List[Tuple[int, ...]],
                                    mono_idx: Dict[Tuple[int, ...], int],
                                    offset: Tuple[int, ...] = None,
                                    ) -> Dict[int, List[Tuple[int, int, int]]]:
    """For a PSD block M(y)[β, γ] = y_{β + γ + offset}, return, for every
    primal y-index α, the list of (β_pos, γ_pos, mult) triples such
    that β + γ + offset = basis-representative-of(α).

    β ranges over basis_from, γ over basis_to (both indices into the
    basis lists).  The optional ``offset`` multi-index is added into
    every cell (used for localizing M_{k-1}(μ_i · y), where offset = e_i,
    and for the window Q part where offset = e_i + e_j summed with a
    coefficient M_W[i,j]).

    Returns a dict α_y_index → list of (b_pos, g_pos, coeff).
    """
    d = len(basis_from[0]) if basis_from else 0
    off = tuple(offset) if offset is not None else tuple(0 for _ in range(d))

    out: Dict[int, List[Tuple[int, int, float]]] = {}
    for bpos, beta in enumerate(basis_from):
        for gpos, gamma in enumerate(basis_to):
            alpha = tuple(beta[k] + gamma[k] + off[k] for k in range(d))
            yj = mono_idx.get(alpha)
            if yj is None:
                continue
            # Under canonicalized mono_idx, yj is the orbit representative.
            out.setdefault(int(yj), []).append((bpos, gpos, 1.0))
    return out


def _enumerate_window_Q(basis: List[Tuple[int, ...]],
                         mono_idx: Dict[Tuple[int, ...], int],
                         Mw: np.ndarray,
                         ) -> Dict[int, List[Tuple[int, int, float]]]:
    """For Q_W(y)_{β,γ} = Σ_{i,j} M_W[i,j] · y_{β+γ+e_i+e_j}, collect all
    (β, γ, y_index, coef) contributions aggregated per y_index.

    Returns dict y_index → list of (b_pos, g_pos, coeff).
    """
    d = Mw.shape[0]
    n = len(basis)
    nz_i, nz_j = np.nonzero(Mw)
    out: Dict[int, Dict[Tuple[int, int], float]] = {}
    for bpos in range(n):
        beta = basis[bpos]
        for gpos in range(n):
            gamma = basis[gpos]
            base = tuple(beta[k] + gamma[k] for k in range(d))
            for ii, jj in zip(nz_i, nz_j):
                alpha_list = list(base)
                alpha_list[ii] += 1
                alpha_list[jj] += 1
                alpha = tuple(alpha_list)
                yj = mono_idx.get(alpha)
                if yj is None:
                    continue
                coef = float(Mw[ii, jj])
                key = (bpos, gpos)
                per_y = out.setdefault(int(yj), {})
                per_y[key] = per_y.get(key, 0.0) + coef
    # Flatten
    flat: Dict[int, List[Tuple[int, int, float]]] = {}
    for y, d2 in out.items():
        flat[y] = [(b, g, v) for (b, g), v in d2.items()]
    return flat


# =====================================================================
# Main builder
# =====================================================================

def build_dual_task(
    P: Dict[str, Any], *,
    t_val: float,
    env: Optional[mosek.Env] = None,
    include_upper_loc: bool = True,
    z2_blockdiag_map: Optional[Dict[str, Any]] = None,
    active_loc: Optional[List[int]] = None,
    active_windows: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[mosek.Task, Dict[str, Any]]:
    """Build the dual Lasserre SDP as a mosek.Task ready to optimize.

    Parameters
    ----------
    P                  : canonicalized precompute (from canonicalize_z2).
                         Must include: mono_list, idx (canonical), basis,
                         loc_basis, M_mats, windows, n_y, consist_mono,
                         consist_idx, consist_ei_idx.
    t_val              : the feasibility probe t value.  The dual SDP is
                         fixed at this t (no bisection); re-call to probe
                         a different t.
    env                : optional pre-created MOSEK environment.
    include_upper_loc  : include the (1−μ_i) upper-localizing cones.
    z2_blockdiag_map   : optional output of lasserre.z2_blockdiag.
                         build_blockdiag_picks(basis, idx, n_y).  If
                         supplied, the moment matrix is replaced by its
                         σ-sym/anti blocks in the dual.
    active_loc         : list of localizing indices i (σ-orbit reps).
                         Default: keep all 0..d-1 (no σ drop).
    active_windows     : list of window indices (into P['windows']) to
                         include.  Default: every nontrivial window.

    Returns
    -------
    task  : mosek.Task with the dual SDP fully encoded.
    info  : dict with provenance (bar-var sizes, constraint counts,
            variable offsets, etc.) needed to read back the solution
            and verify the certificate.
    """
    d = P['d']
    n_y = P['n_y']
    basis = P['basis']
    loc_basis = P['loc_basis']
    mono_idx = P['idx']
    M_mats = P['M_mats']
    windows = P['windows']

    if env is None:
        env = mosek.Env()
    task = env.Task()

    if verbose:
        task.set_Stream(
            mosek.streamtype.log, lambda s: print(s, end=''))

    t0 = time.time()

    # -----------------------------------------------------------------
    # 1. Declare PSD bar-variables in order:
    #       [moment]  either (X_0) or (X_0^sym, X_0^anti) under z2_bd
    #       [loc_i]   one per active localizing index, size n_loc
    #       [uloc_i]  same, when include_upper_loc
    #       [win_W]   one per active window, size n_loc
    # -----------------------------------------------------------------
    bar_sizes: List[int] = []
    bar_labels: List[str] = []  # for debug/traceability

    if z2_blockdiag_map is None:
        bar_sizes.append(len(basis))
        bar_labels.append('X_moment')
        moment_bar_ids = [0]  # single cone
    else:
        n_sym = z2_blockdiag_map['n_sym']
        n_anti = z2_blockdiag_map['n_anti']
        bar_sizes.append(n_sym)
        bar_labels.append('X_moment_sym')
        if n_anti > 0:
            bar_sizes.append(n_anti)
            bar_labels.append('X_moment_anti')
            moment_bar_ids = [0, 1]
        else:
            moment_bar_ids = [0]

    loc_bar_start = len(bar_sizes)
    if active_loc is None:
        active_loc = list(range(d))
    for i in active_loc:
        bar_sizes.append(len(loc_basis))
        bar_labels.append(f'X_loc_{i}')
    if include_upper_loc:
        for i in active_loc:
            bar_sizes.append(len(loc_basis))
            bar_labels.append(f'X_uloc_{i}')

    win_bar_start = len(bar_sizes)
    if active_windows is None:
        active_windows = list(P.get('nontrivial_windows',
                                     range(len(windows))))
    for w in active_windows:
        bar_sizes.append(len(loc_basis))
        bar_labels.append(f'X_win_{w}')

    task.appendbarvars(bar_sizes)
    n_bar = len(bar_sizes)
    if verbose:
        print(f"  Dual bar-vars: {n_bar} cones; "
              f"sizes = moment {bar_sizes[0]}..., "
              f"loc {len(loc_basis)} × {2 if include_upper_loc else 1} "
              f"× {len(active_loc)}, "
              f"windows {len(active_windows)} × {len(loc_basis)}",
              flush=True)

    info = {
        'bar_sizes': bar_sizes, 'bar_labels': bar_labels,
        'moment_bar_ids': moment_bar_ids,
        'loc_bar_start': loc_bar_start,
        'win_bar_start': win_bar_start,
        'active_loc': active_loc,
        'active_windows': active_windows,
        'n_bar': n_bar,
    }

    # -----------------------------------------------------------------
    # (The rest of this module — scalar variables, stationarity rows,
    #  sensitivity-matrix coefficient assembly, constraint bounds,
    #  objective — is deferred to a follow-up commit: see the
    #  dual-sdp TODO in this file.  This landing commit lands the
    #  infrastructure so d=4 validation can be wired up alongside the
    #  remaining math; NO optimize() call until every stationarity row
    #  is built and verified.)
    # -----------------------------------------------------------------
    info['build_time_s'] = time.time() - t0
    info['status'] = 'scaffolding-only'
    return task, info
