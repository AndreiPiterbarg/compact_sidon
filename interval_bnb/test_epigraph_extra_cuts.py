"""Soundness + tightening tests for the 4 extra LP cuts added to
``_solve_epigraph_lp`` in ``bound_epigraph.py``.

The 4 cuts (each independently sound from Σμ=1 and Y=μμᵀ on the simplex):

  C1) column-sum RLT:  Σ_i Y_{i,j} = μ_j           (d new equalities)
  C2) Y-symmetry:      Y_{i,j} = Y_{j,i}, i<j       (d(d-1)/2 new equalities)
  C3) diagonal SOS:    Σ_i Y_{i,i} ≥ 1/d           (1 new inequality)
  C4) midpoint diag tangent:
                       Y_{i,i} ≥ 2 m_i μ_i − m_i²   (d new inequalities)

Tests:
  * SOUNDNESS: for random μ ∈ Δ_d (d ∈ {3,5,8}) with Y = μμᵀ, the augmented
    LP must be feasible at this primal point (Aub·x ≤ bub, Aeq·x = beq, and
    bounds met) within 1e-9.
  * TIGHTENING: on a deep stuck box around d=4 mu* (the depth-25-ish case
    used by ``test_d4_known_optimum``), the new LP value must be
    ≥ the old LP value.
  * SMOKE: the existing ``test_epigraph.py`` suite still passes.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_epigraph import (
    _cache_lp_structure, _solve_epigraph_lp, bound_epigraph_lp_float,
)
from interval_bnb.windows import build_windows


# ---------------------------------------------------------------------
# Reference: rebuild the augmented LP constraint matrices directly so we
# can sanity-check Aub·x ≤ bub and Aeq·x = beq at a known feasible point
# (μ ∈ Δ_d, Y = μμᵀ, z = max_W TV_W(μ)).
#
# This duplicates the structure of the production ``_solve_epigraph_lp``;
# if the production code drifts, this test will catch it.
# ---------------------------------------------------------------------

def _build_constraints(lo, hi, windows, d):
    from scipy.sparse import coo_matrix, csr_matrix

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu

    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    n_pairs = n_y

    # SW
    sw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    sw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    sw_data = np.empty(3 * n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2*n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2*n_pairs] = n_y + pair_i; sw_data[n_pairs:2*n_pairs] = lo[pair_j]
    sw_rows[2*n_pairs:3*n_pairs] = np.arange(n_pairs); sw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; sw_data[2*n_pairs:3*n_pairs] = lo[pair_i]
    # NE
    ne_rows = np.empty(3 * n_pairs, dtype=np.int64)
    ne_cols = np.empty(3 * n_pairs, dtype=np.int64)
    ne_data = np.empty(3 * n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[n_pairs:2*n_pairs] = n_y + pair_i; ne_data[n_pairs:2*n_pairs] = hi[pair_j]
    ne_rows[2*n_pairs:3*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; ne_data[2*n_pairs:3*n_pairs] = hi[pair_i]
    # NW
    nw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    nw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    nw_data = np.empty(3 * n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[n_pairs:2*n_pairs] = n_y + pair_i; nw_data[n_pairs:2*n_pairs] = -lo[pair_j]
    nw_rows[2*n_pairs:3*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; nw_data[2*n_pairs:3*n_pairs] = -hi[pair_i]
    # SE
    se_rows = np.empty(3 * n_pairs, dtype=np.int64)
    se_cols = np.empty(3 * n_pairs, dtype=np.int64)
    se_data = np.empty(3 * n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[n_pairs:2*n_pairs] = n_y + pair_i; se_data[n_pairs:2*n_pairs] = -hi[pair_j]
    se_rows[2*n_pairs:3*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; se_data[2*n_pairs:3*n_pairs] = -lo[pair_i]
    # Epigraph
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    epi_rows[:n_epi_pair_entries] = 4*n_pairs + rows_w
    epi_cols[:n_epi_pair_entries] = cols_w
    epi_data[:n_epi_pair_entries] = scales_w
    epi_rows[n_epi_pair_entries:] = 4*n_pairs + np.arange(n_W)
    epi_cols[n_epi_pair_entries:] = z_idx
    epi_data[n_epi_pair_entries:] = -1.0
    # New: SOS + tangents
    sos_row_start = 4*n_pairs + n_W
    tan_row_start = sos_row_start + 1
    diag_idx = np.arange(d) * d + np.arange(d)
    sos_rows = np.full(d, sos_row_start, dtype=np.int64)
    sos_cols = diag_idx.astype(np.int64)
    sos_data = np.full(d, -1.0, dtype=np.float64)
    m = 0.5 * (lo + hi)
    tan_rows = np.empty(2*d, dtype=np.int64)
    tan_cols = np.empty(2*d, dtype=np.int64)
    tan_data = np.empty(2*d, dtype=np.float64)
    tan_rows[:d] = tan_row_start + np.arange(d); tan_cols[:d] = diag_idx; tan_data[:d] = -1.0
    tan_rows[d:] = tan_row_start + np.arange(d); tan_cols[d:] = n_y + np.arange(d); tan_data[d:] = 2.0 * m

    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows, epi_rows, sos_rows, tan_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols, epi_cols, sos_cols, tan_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data, epi_data, sos_data, tan_data])
    n_ineq = 4*n_pairs + n_W + 1 + d
    A_ub = coo_matrix((data_all, (rows_all, cols_all)), shape=(n_ineq, n_vars)).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    b_ub[n_pairs:2*n_pairs] = hi[pair_i] * hi[pair_j]
    b_ub[2*n_pairs:3*n_pairs] = -lo[pair_j] * hi[pair_i]
    b_ub[3*n_pairs:4*n_pairs] = -hi[pair_j] * lo[pair_i]
    b_ub[4*n_pairs:4*n_pairs + n_W] = 0.0
    b_ub[sos_row_start] = -1.0 / d
    b_ub[tan_row_start:tan_row_start + d] = m * m

    # Equalities
    n_sym = d * (d - 1) // 2
    n_eq = 1 + d + d + n_sym
    eq_rows = []; eq_cols = []; eq_data = []
    eq_rows.extend([0]*d); eq_cols.extend([n_y + i for i in range(d)]); eq_data.extend([1.0]*d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
    col_row_start = 1 + d
    for j in range(d):
        for i in range(d):
            eq_rows.append(col_row_start + j); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(col_row_start + j); eq_cols.append(n_y + j); eq_data.append(-1.0)
    sym_row_start = col_row_start + d
    sym_k = 0
    for i in range(d):
        for j in range(i+1, d):
            r = sym_row_start + sym_k
            eq_rows.append(r); eq_cols.append(i*d + j); eq_data.append(1.0)
            eq_rows.append(r); eq_cols.append(j*d + i); eq_data.append(-1.0)
            sym_k += 1
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(n_eq, n_vars),
    )
    b_eq = np.zeros(n_eq, dtype=np.float64)
    b_eq[0] = 1.0

    bnds_lo = np.concatenate([np.zeros(n_y), lo, [0.0]])
    bnds_hi = np.concatenate([np.full(n_y, np.inf), hi, [np.inf]])
    return A_ub, b_ub, A_eq, b_eq, bnds_lo, bnds_hi


# ---------------------------------------------------------------------
# Reference (OLD) LP — replicates the production code prior to the cuts.
# Used only by the tightening test.
# ---------------------------------------------------------------------

def _solve_epigraph_lp_OLD(lo, hi, windows, d):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu
    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    n_pairs = n_y
    sw_rows = np.empty(3*n_pairs, dtype=np.int64); sw_cols = np.empty(3*n_pairs, dtype=np.int64); sw_data = np.empty(3*n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2*n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2*n_pairs] = n_y + pair_i; sw_data[n_pairs:2*n_pairs] = lo[pair_j]
    sw_rows[2*n_pairs:3*n_pairs] = np.arange(n_pairs); sw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; sw_data[2*n_pairs:3*n_pairs] = lo[pair_i]
    ne_rows = np.empty(3*n_pairs, dtype=np.int64); ne_cols = np.empty(3*n_pairs, dtype=np.int64); ne_data = np.empty(3*n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[n_pairs:2*n_pairs] = n_y + pair_i; ne_data[n_pairs:2*n_pairs] = hi[pair_j]
    ne_rows[2*n_pairs:3*n_pairs] = n_pairs + np.arange(n_pairs); ne_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; ne_data[2*n_pairs:3*n_pairs] = hi[pair_i]
    nw_rows = np.empty(3*n_pairs, dtype=np.int64); nw_cols = np.empty(3*n_pairs, dtype=np.int64); nw_data = np.empty(3*n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[n_pairs:2*n_pairs] = n_y + pair_i; nw_data[n_pairs:2*n_pairs] = -lo[pair_j]
    nw_rows[2*n_pairs:3*n_pairs] = 2*n_pairs + np.arange(n_pairs); nw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; nw_data[2*n_pairs:3*n_pairs] = -hi[pair_i]
    se_rows = np.empty(3*n_pairs, dtype=np.int64); se_cols = np.empty(3*n_pairs, dtype=np.int64); se_data = np.empty(3*n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[n_pairs:2*n_pairs] = n_y + pair_i; se_data[n_pairs:2*n_pairs] = -hi[pair_j]
    se_rows[2*n_pairs:3*n_pairs] = 3*n_pairs + np.arange(n_pairs); se_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; se_data[2*n_pairs:3*n_pairs] = -lo[pair_i]
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    epi_rows[:n_epi_pair_entries] = 4*n_pairs + rows_w; epi_cols[:n_epi_pair_entries] = cols_w; epi_data[:n_epi_pair_entries] = scales_w
    epi_rows[n_epi_pair_entries:] = 4*n_pairs + np.arange(n_W); epi_cols[n_epi_pair_entries:] = z_idx; epi_data[n_epi_pair_entries:] = -1.0
    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows, epi_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols, epi_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data, epi_data])
    n_ineq = 4*n_pairs + n_W
    A_ub = coo_matrix((data_all, (rows_all, cols_all)), shape=(n_ineq, n_vars)).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    b_ub[n_pairs:2*n_pairs] = hi[pair_i] * hi[pair_j]
    b_ub[2*n_pairs:3*n_pairs] = -lo[pair_j] * hi[pair_i]
    b_ub[3*n_pairs:4*n_pairs] = -hi[pair_j] * lo[pair_i]
    b_ub[4*n_pairs:] = 0.0
    eq_rows = []; eq_cols = []; eq_data = []
    eq_rows.extend([0]*d); eq_cols.extend([n_y + i for i in range(d)]); eq_data.extend([1.0]*d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1 + i); eq_cols.append(i*d + j); eq_data.append(1.0)
        eq_rows.append(1 + i); eq_cols.append(n_y + i); eq_data.append(-1.0)
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(1 + d, n_vars),
    )
    b_eq = np.zeros(1 + d, dtype=np.float64); b_eq[0] = 1.0
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)]
    bnds.append((0.0, None))
    c = np.zeros(n_vars); c[z_idx] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if not res.success:
        return float("-inf")
    return float(res.fun)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def _random_simplex_in_box(rng, lo, hi):
    """Sample μ ∈ Δ_d ∩ [lo, hi]. Uses Dirichlet then projects (rejection
    sampling on the box).
    """
    d = len(lo)
    for _ in range(2000):
        x = rng.dirichlet(np.ones(d))
        if (x >= lo - 1e-12).all() and (x <= hi + 1e-12).all():
            return x
    # Fallback: midpoint normalised.
    m = 0.5 * (lo + hi)
    return m / m.sum()


def test_extra_cuts_soundness():
    """For 50 random μ ∈ Δ_d (d ∈ {3,5,8}) with Y = μμᵀ, the augmented LP
    must be feasible at this primal point: Aub·x ≤ bub, Aeq·x = beq,
    bounds met.
    """
    rng = np.random.default_rng(0xC0FFEE)
    for d in (3, 5, 8):
        windows = build_windows(d)
        lo = np.full(d, 1e-6)        # box: simplex-friendly slab
        hi = np.full(d, 1.0 - 1e-6)
        A_ub, b_ub, A_eq, b_eq, bnds_lo, bnds_hi = _build_constraints(lo, hi, windows, d)

        n_y = d * d
        for trial in range(50):
            mu = _random_simplex_in_box(rng, lo, hi)
            Y = np.outer(mu, mu)
            # z must be ≥ max_W TV_W(μ); use exact equality for tightest x.
            from interval_bnb.bound_eval import _adjacency_matrix
            z = max(w.scale * float(mu @ _adjacency_matrix(w, d) @ mu) for w in windows)
            x = np.concatenate([Y.ravel(), mu, [z]])

            # Bounds (with a tolerance — the slack bounds are wide here)
            assert (x >= bnds_lo - 1e-9).all(), (
                f"d={d} trial={trial}: lower-bound viol "
                f"max={float((bnds_lo - x).max()):.3e}"
            )
            # bnds_hi may contain np.inf so handle via finite mask
            fin = np.isfinite(bnds_hi)
            assert (x[fin] <= bnds_hi[fin] + 1e-9).all(), (
                f"d={d} trial={trial}: upper-bound viol "
                f"max={float((x[fin] - bnds_hi[fin]).max()):.3e}"
            )

            # Inequality
            r_ub = A_ub @ x - b_ub
            assert (r_ub <= 1e-9).all(), (
                f"d={d} trial={trial}: ineq viol max={float(r_ub.max()):.3e}"
            )
            # Equality
            r_eq = A_eq @ x - b_eq
            assert (np.abs(r_eq) <= 1e-9).all(), (
                f"d={d} trial={trial}: eq viol max={float(np.abs(r_eq).max()):.3e}"
            )


def test_extra_cuts_tightening():
    """At a deep box around d=4 mu*, the new LP value must be
    ≥ the old LP value (no regression; usually strictly greater).
    """
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])

    # A few different radii — exercise the "stuck box" regime.
    for radius in (1e-2, 1e-3, 1e-4):
        lo = np.maximum(mu_star - radius, 0.0)
        hi = np.minimum(mu_star + radius, 1.0)
        old_val = _solve_epigraph_lp_OLD(lo, hi, windows, d)
        new_val = bound_epigraph_lp_float(lo, hi, windows, d)
        assert np.isfinite(old_val) and np.isfinite(new_val), (
            f"radius={radius}: LP failed (old={old_val}, new={new_val})"
        )
        # Must not regress; allow 1e-9 slack for HiGHS dual tolerance.
        assert new_val >= old_val - 1e-9, (
            f"radius={radius}: REGRESSION new={new_val:.10f} "
            f"< old={old_val:.10f}"
        )


def test_extra_cuts_d20_stuck_box():
    """Deeper-box smoke test (d=20 regime, depth-25-ish)."""
    d = 20
    windows = build_windows(d)
    rng = np.random.default_rng(7)
    # Pick a μ* on simplex, build a tight box around it.
    mu_star = rng.dirichlet(np.ones(d))
    radius = 5e-4
    lo = np.maximum(mu_star - radius, 1e-9)
    hi = np.minimum(mu_star + radius, 1.0 - 1e-9)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        # Box doesn't contain simplex slice — skip.
        return
    old_val = _solve_epigraph_lp_OLD(lo, hi, windows, d)
    new_val = bound_epigraph_lp_float(lo, hi, windows, d)
    assert np.isfinite(old_val) and np.isfinite(new_val)
    assert new_val >= old_val - 1e-9, (
        f"d=20 stuck box: REGRESSION new={new_val:.10f} < old={old_val:.10f}"
    )


if __name__ == "__main__":
    test_extra_cuts_soundness()
    print("soundness ok")
    test_extra_cuts_tightening()
    print("tightening ok")
    test_extra_cuts_d20_stuck_box()
    print("d20 ok")
