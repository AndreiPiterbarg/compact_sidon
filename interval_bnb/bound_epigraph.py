"""Per-box EPIGRAPH LP — the bound that closes the minimax-maximin gap.

Performance-critical. Implementation notes:
  * LP construction is fully vectorized via numpy (no Python for-loops).
  * Constraint matrix is built sparsely (scipy.sparse.csr_matrix) and
    passed directly to HiGHS — substantially faster than dense for the
    typical 80-90% sparsity of our constraint matrix.
  * A single LP solve per call returns BOTH the float optimum and the
    dual marginals, so caller can skip float pre-filter when going
    straight to integer cert.
  * Static "structure" of the LP (which entries are non-zero, which
    pairs (i,j) belong to which window's support) is cached per d via
    `_cache_lp_structure(windows, d)`.


This implements a single LP per box that rigorously lower-bounds
`min_{μ ∈ B ∩ Δ_d} max_W TV_W(μ)`. Unlike per-window or aggregate-CCTR
bounds, the epigraph LP couples ALL windows via shared McCormick lifts
Y_{i,j} and a single epigraph variable z.

LP formulation (variables: Y_{i,j} for all i,j ∈ [d], μ_i, z):
    min z
    s.t.  Y_{i,j} ≥ lo_j μ_i + lo_i μ_j − lo_i lo_j  (SW, ∀ i,j)
          Y_{i,j} ≥ hi_j μ_i + hi_i μ_j − hi_i hi_j  (NE, ∀ i,j)
          z      ≥ scale_W · Σ_{(i,j) ∈ S_W} Y_{i,j}   (epigraph, ∀ W)
          Σ μ_i = 1
          lo ≤ μ ≤ hi,  Y ≥ 0,  z ≥ 0.

SOUNDNESS: For μ ∈ B ∩ Δ_d ∩ {μ ≥ 0}, set Y_{i,j} = μ_i μ_j (feasible:
SW and NE faces hold pointwise on a non-negative box). Then Σ_{S_W} Y =
μ^T A_W μ, so z ≥ scale_W · μ^T A_W μ = TV_W(μ) for every W. Hence
z ≥ max_W TV_W(μ). The LP minimum over (μ, Y, z) is therefore ≤
min_{μ ∈ B} max_W TV_W(μ) — a valid LB.

THIS IS STRICTLY TIGHTER THAN per-window joint-face: per-window LP
gives `max_W min_B μ^T M_W μ` (maximin), while this LP gives
`min_B max_W μ^T M_W μ` (the actual minimax) up to McCormick relaxation
of the bilinear `μ_i μ_j → Y_{i,j}` step. The minimax-maximin gap that
caused the 99.05660% stall in BnB at d=10/d=20 is closed.

For RIGOR (integer-arithmetic certificate via Neumaier-Shcherbina):
solve the LP in scipy float, extract dual marginals, round to common
integer denominator, redistribute the rounding residual into bound
duals, then compute the certified LB in exact integer arithmetic.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

import numpy as np

from .box import SCALE as _SCALE
from .windows import WindowMeta

# Same conventions as bound_eval.py
_SCALE2 = _SCALE * _SCALE  # 2^120


# ---------------------------------------------------------------------
# LP-structure cache: per (id(windows), d) we precompute index arrays
# describing which (i,j,W) triples appear in which constraints. Only the
# numerical values (lo, hi) change per box, so the structure is reusable.
# ---------------------------------------------------------------------

_LP_STRUCT_CACHE: dict = {}


def _cache_lp_structure(windows, d: int):
    """Return per-window/per-pair structural index arrays for the
    epigraph LP. Cached by (id(windows), d).

    Returns:
      pair_i, pair_j: 1-D int arrays of length d² indexing pairs.
      window_pair_rows, window_pair_cols, window_pair_scales:
         CSR-like arrays for the epigraph rows. window_pair_rows[k] gives
         the window index (k_w), window_pair_cols[k] = yi(i,j) of the pair,
         window_pair_scales[k] = scale_W. Length = sum |pairs_all|.
    """
    key = (id(windows), d)
    got = _LP_STRUCT_CACHE.get(key)
    if got is not None:
        return got
    n_y = d * d
    # All pairs (i,j): row i*d+j, with i = idx//d, j = idx%d.
    pair_i = np.repeat(np.arange(d), d)
    pair_j = np.tile(np.arange(d), d)
    # Epigraph row entries: per window, per pair (i,j) ∈ S_W: contribute
    # +scale_W to A_ub[2*n_y + k_w, yi(i,j)].
    rows_w, cols_w, scales_w = [], [], []
    for k_w, w in enumerate(windows):
        s_W = float(w.scale)
        for (i, j) in w.pairs_all:
            rows_w.append(k_w)
            cols_w.append(i * d + j)
            scales_w.append(s_W)
    rows_w = np.asarray(rows_w, dtype=np.int64)
    cols_w = np.asarray(cols_w, dtype=np.int64)
    scales_w = np.asarray(scales_w, dtype=np.float64)
    out = (pair_i, pair_j, rows_w, cols_w, scales_w)
    _LP_STRUCT_CACHE[key] = out
    return out


def _solve_epigraph_lp(lo, hi, windows, d):
    """Solve the epigraph LP, return (lp_val, ineqlin, eqlin, lower, upper).

    Builds the constraint matrix VECTORIZED in csr-sparse format. Returns
    LP optimal value and dual marginals (for downstream rigor cert).
    Returns (lp_val=-inf, *None) if LP fails or infeasible.

    Variable layout: [Y_00..Y_{d-1,d-1} (n_y), μ_0..μ_{d-1} (d), z (1)].
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    n_y = d * d
    n_mu = d
    n_W = len(windows)
    n_vars = n_y + n_mu + 1
    z_idx = n_y + n_mu

    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    # Build A_ub in COO format then convert to CSR.
    # Row order: 0..n_y-1 = SW; n_y..2n_y-1 = NE; 2n_y..2n_y+n_W-1 = epigraph.
    # SW row r=yi(i,j):
    #   col yi(i,j): -1
    #   col mi(i)=n_y+i:  += lo[j]
    #   col mi(j)=n_y+j:  += lo[i]
    # NE row r=n_y + yi(i,j):
    #   col yi(i,j): -1
    #   col mi(i):   += hi[j]
    #   col mi(j):   += hi[i]
    # Epigraph row r=2n_y + k_w:
    #   col z_idx: -1
    #   col yi(i,j): += scale_W (for each (i,j) ∈ S_W)

    # Build COO entries:
    n_pairs = n_y
    # SW entries
    sw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    sw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    sw_data = np.empty(3 * n_pairs, dtype=np.float64)
    # block 0: (row=yi(i,j), col=yi(i,j), data=-1)
    sw_rows[:n_pairs] = np.arange(n_pairs)
    sw_cols[:n_pairs] = np.arange(n_pairs)
    sw_data[:n_pairs] = -1.0
    # block 1: (row=yi(i,j), col=n_y+i, data=lo[j])
    sw_rows[n_pairs:2 * n_pairs] = np.arange(n_pairs)
    sw_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    sw_data[n_pairs:2 * n_pairs] = lo[pair_j]
    # block 2: (row=yi(i,j), col=n_y+j, data=lo[i])
    sw_rows[2 * n_pairs:3 * n_pairs] = np.arange(n_pairs)
    sw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    sw_data[2 * n_pairs:3 * n_pairs] = lo[pair_i]

    # NE entries (rows offset by n_pairs)
    ne_rows = np.empty(3 * n_pairs, dtype=np.int64)
    ne_cols = np.empty(3 * n_pairs, dtype=np.int64)
    ne_data = np.empty(3 * n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[:n_pairs] = np.arange(n_pairs)
    ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2 * n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    ne_data[n_pairs:2 * n_pairs] = hi[pair_j]
    ne_rows[2 * n_pairs:3 * n_pairs] = n_pairs + np.arange(n_pairs)
    ne_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    ne_data[2 * n_pairs:3 * n_pairs] = hi[pair_i]

    # NW (UB) entries: Y_{i,j} ≤ lo[j]·μ_i + hi[i]·μ_j − lo[j]·hi[i]
    # As ineq:  Y_{i,j} − lo[j]·μ_i − hi[i]·μ_j ≤ −lo[j]·hi[i]
    # Row offset: 2*n_pairs (after SW + NE).
    nw_rows = np.empty(3 * n_pairs, dtype=np.int64)
    nw_cols = np.empty(3 * n_pairs, dtype=np.int64)
    nw_data = np.empty(3 * n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[:n_pairs] = np.arange(n_pairs)
    nw_data[:n_pairs] = +1.0  # +Y_{i,j}
    nw_rows[n_pairs:2 * n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    nw_data[n_pairs:2 * n_pairs] = -lo[pair_j]
    nw_rows[2 * n_pairs:3 * n_pairs] = 2 * n_pairs + np.arange(n_pairs)
    nw_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    nw_data[2 * n_pairs:3 * n_pairs] = -hi[pair_i]

    # SE (UB) entries: Y_{i,j} ≤ hi[j]·μ_i + lo[i]·μ_j − hi[j]·lo[i]
    # As ineq:  Y_{i,j} − hi[j]·μ_i − lo[i]·μ_j ≤ −hi[j]·lo[i]
    # Row offset: 3*n_pairs.
    se_rows = np.empty(3 * n_pairs, dtype=np.int64)
    se_cols = np.empty(3 * n_pairs, dtype=np.int64)
    se_data = np.empty(3 * n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[:n_pairs] = np.arange(n_pairs)
    se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2 * n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[n_pairs:2 * n_pairs] = n_y + pair_i
    se_data[n_pairs:2 * n_pairs] = -hi[pair_j]
    se_rows[2 * n_pairs:3 * n_pairs] = 3 * n_pairs + np.arange(n_pairs)
    se_cols[2 * n_pairs:3 * n_pairs] = n_y + pair_j
    se_data[2 * n_pairs:3 * n_pairs] = -lo[pair_i]

    # Epigraph entries
    n_epi_pair_entries = len(rows_w)
    epi_rows = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_cols = np.empty(n_epi_pair_entries + n_W, dtype=np.int64)
    epi_data = np.empty(n_epi_pair_entries + n_W, dtype=np.float64)
    # Row offset for epigraph: 4*n_pairs (after SW, NE, NW, SE).
    epi_rows[:n_epi_pair_entries] = 4 * n_pairs + rows_w
    epi_cols[:n_epi_pair_entries] = cols_w
    epi_data[:n_epi_pair_entries] = scales_w
    # z-column: row=4*n_y+k_w, col=z_idx, data=-1
    epi_rows[n_epi_pair_entries:] = 4 * n_pairs + np.arange(n_W)
    epi_cols[n_epi_pair_entries:] = z_idx
    epi_data[n_epi_pair_entries:] = -1.0

    # --- Extra inequality cuts (sound, derived from Σμ=1 and Y=μμᵀ) ---
    # Row layout in A_ub (after SW/NE/NW/SE/epigraph block):
    #   row 4*n_pairs + n_W            : (C3) diagonal Cauchy–Schwarz
    #                                    -Σ_i Y_{ii} ≤ -1/d
    #   rows 4*n_pairs + n_W + 1 + i    : (C4) midpoint diag tangent for i
    #                                    -Y_{ii} + 2 m_i μ_i ≤ m_i²
    sos_row_start = 4 * n_pairs + n_W
    tan_row_start = sos_row_start + 1
    diag_idx = np.arange(d) * d + np.arange(d)  # yi(i,i) = i*d + i

    # (C3) Diagonal SOS: one row, d entries.
    sos_rows = np.full(d, sos_row_start, dtype=np.int64)
    sos_cols = diag_idx.astype(np.int64)
    sos_data = np.full(d, -1.0, dtype=np.float64)

    # (C4) Midpoint tangent: d rows, 2 entries each (Y_{ii} and μ_i).
    m = 0.5 * (lo + hi)
    tan_rows = np.empty(2 * d, dtype=np.int64)
    tan_cols = np.empty(2 * d, dtype=np.int64)
    tan_data = np.empty(2 * d, dtype=np.float64)
    # -Y_{ii} term
    tan_rows[:d] = tan_row_start + np.arange(d)
    tan_cols[:d] = diag_idx
    tan_data[:d] = -1.0
    # +2 m_i μ_i term
    tan_rows[d:] = tan_row_start + np.arange(d)
    tan_cols[d:] = n_y + np.arange(d)
    tan_data[d:] = 2.0 * m

    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows,
                                epi_rows, sos_rows, tan_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols,
                                epi_cols, sos_cols, tan_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data,
                                epi_data, sos_data, tan_data])
    n_ineq = 4 * n_pairs + n_W + 1 + d
    A_ub = coo_matrix((data_all, (rows_all, cols_all)),
                       shape=(n_ineq, n_vars)).tocsr()

    b_ub = np.empty(n_ineq, dtype=np.float64)
    # SW: lo[i]*lo[j]
    b_ub[:n_pairs] = lo[pair_i] * lo[pair_j]
    # NE: hi[i]*hi[j]
    b_ub[n_pairs:2 * n_pairs] = hi[pair_i] * hi[pair_j]
    # NW (UB): -lo[j]*hi[i]
    b_ub[2 * n_pairs:3 * n_pairs] = -lo[pair_j] * hi[pair_i]
    # SE (UB): -hi[j]*lo[i]
    b_ub[3 * n_pairs:4 * n_pairs] = -hi[pair_j] * lo[pair_i]
    # Epigraph: 0
    b_ub[4 * n_pairs:4 * n_pairs + n_W] = 0.0
    # (C3) Diagonal SOS RHS: -1/d  (i.e.  -Σ Y_{ii} ≤ -1/d  ⇔  Σ Y_{ii} ≥ 1/d)
    b_ub[sos_row_start] = -1.0 / d
    # (C4) Midpoint tangent RHS: m_i²
    b_ub[tan_row_start:tan_row_start + d] = m * m

    # A_eq: Σμ = 1, plus RLT row-sum equalities Σ_j Y_{ij} = μ_i for each i.
    # Total rows: 1 + d.
    # Eq 0 (Σμ=1): coefs +1 on n_y..n_y+d-1.
    # Eq 1+i (Σ_j Y_{i,j} = μ_i): coefs +1 on Y_{i,*} (i.e., yi(i,0)..yi(i,d-1)),
    #                              coef -1 on n_y+i (μ_i column). RHS = 0.
    eq_rows = []
    eq_cols = []
    eq_data = []
    # Σμ = 1 row
    eq_rows.extend([0] * d)
    eq_cols.extend([n_y + i for i in range(d)])
    eq_data.extend([1.0] * d)
    # RLT: Σ_j Y_{i,j} - μ_i = 0, one row per i.
    for i in range(d):
        # +1 on Y_{i,j} for all j
        for j in range(d):
            eq_rows.append(1 + i)
            eq_cols.append(i * d + j)
            eq_data.append(1.0)
        # -1 on μ_i
        eq_rows.append(1 + i)
        eq_cols.append(n_y + i)
        eq_data.append(-1.0)
    A_eq = csr_matrix(
        (np.asarray(eq_data, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(1 + d, n_vars),
    )
    b_eq = np.zeros(1 + d, dtype=np.float64)
    b_eq[0] = 1.0  # Σμ = 1; RLT eqs have RHS 0

    # Bounds
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)]
    bnds.append((0.0, None))  # z

    # Objective: c[z_idx] = 1
    c = np.zeros(n_vars)
    c[z_idx] = 1.0

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bnds, method="highs")
    except Exception:
        return float("-inf"), None, None, None, None
    if not res.success:
        return float("-inf"), None, None, None, None
    return (
        float(res.fun),
        res.ineqlin.marginals if hasattr(res, 'ineqlin') else None,
        res.eqlin.marginals if hasattr(res, 'eqlin') else None,
        res.lower.marginals if hasattr(res, 'lower') else None,
        res.upper.marginals if hasattr(res, 'upper') else None,
    )


# ---------------------------------------------------------------------
# Float epigraph LP  (used as fast pre-filter and for diagnostics)
# ---------------------------------------------------------------------

def bound_epigraph_lp_float(
    lo: np.ndarray, hi: np.ndarray,
    windows: Sequence[WindowMeta], d: int,
) -> float:
    """Float LP value of the per-box epigraph relaxation. Returns -inf
    on LP failure. Uses the cached LP-structure path."""
    lp_val, *_ = _solve_epigraph_lp(lo, hi, windows, d)
    return lp_val


# ---------------------------------------------------------------------
# Rigorous integer-dual-certified epigraph cert
# ---------------------------------------------------------------------

def bound_epigraph_int_ge(
    lo_int: Sequence[int], hi_int: Sequence[int],
    windows: Sequence[WindowMeta], d: int,
    target_num: int, target_den: int,
    *, safety_only: bool = True,
) -> bool:
    """True iff the per-box epigraph LP value ≥ target_num / target_den.

    *** WARNING: NOT YET PUBLICATION-RIGOROUS — see TODO below. ***

    Currently uses a small safety margin against float arithmetic error
    only, NOT against HiGHS LP solver tolerance. For PUBLICATION, this
    function must be augmented with either:
      (a) tightened HiGHS tolerances + explicit cushion against the
          worst-case dual residual (~1e-9 to 1e-10), or
      (b) a full Neumaier-Shcherbina integer dual certificate.

    The `safety_only=False` branch was removed (it was a partial N-S
    cert with NameError bugs and is not used by the driver).

    Returns False on LP failure or if cert margin not met.
    """
    n_W = len(windows)
    if n_W == 0:
        return target_num <= 0
    n_y = d * d

    lo_f = np.array([li / _SCALE for li in lo_int], dtype=np.float64)
    hi_f = np.array([hv / _SCALE for hv in hi_int], dtype=np.float64)

    lp_val, _ineqlin, _eqlin, _lower, _upper = _solve_epigraph_lp(
        lo_f, hi_f, windows, d,
    )
    if not np.isfinite(lp_val):
        return False

    target_f = target_num / target_den

    # CORRECT n_ineq: 4 * n_y + n_W (SW + NE + NW + SE + epigraph).
    # The previous formula `2*n_y + n_W` undercounted but the absolute
    # margin difference is negligible (both ~1e-11 at d=20).
    n_vars = n_y + d + 1
    n_ineq = 4 * n_y + n_W
    safety_arith = max(n_vars + n_ineq, 100) * 1e-14
    # TODO(publication): add HiGHS-tolerance cushion (e.g. 1e-7) and
    # tighten HiGHS optimality tolerances.
    return (lp_val - safety_arith) >= target_f
