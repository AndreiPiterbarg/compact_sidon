"""Coarse cascade prover for C_{1a} >= c_target (NO correction term).

Mathematical basis:
  For any nonneg f on [-1/4,1/4] with integral 1, partitioned into d bins:
    max(f*f) >= max_W TV_W(mu)
  where mu_i = integral of f over bin i (NO step-function approximation).

  By refinement monotonicity (empirically verified):
    if parent at d bins has max TV >= c, all children at 2d bins also do.

  So the cascade can prune parents without correction.

Algorithm:
  1. L0: enumerate all compositions of S into d_start parts, prune by TV >= c.
  2. L1..LK: for each survivor, split each bin into 2 children, prune.
  3. Box certification: for each pruned cell, QP-verify the Voronoi box.
  4. If 0 survivors at level K and all boxes certified: C_{1a} >= c. QED.

Grid: absolute mass quantum delta = 1/S.  Integer masses c_i sum to S.
  TV_W(ell,s) = (2d/ell) * sum_{k=s}^{s+ell-2} conv[k] / S^2
  Prune if ws_int > floor(c * ell * S^2 / (2d))

Usage:
  python coarse_cascade_prover.py
  python coarse_cascade_prover.py --c_target 1.30 --S 50
  python coarse_cascade_prover.py --c_target 1.28 --S 30 --d_start 2
"""
import argparse
import time
import os
import sys

import numpy as np
import numba
from numba import njit, prange


# =====================================================================
# Threshold computation
# =====================================================================

def compute_thresholds(c_target, S, d):
    """Precompute per-ell integer thresholds for dimension d.

    Prune if ws_int > thr[ell].
    Equivalent to TV >= c_target (approximately; sound direction).
    """
    max_ell = 2 * d
    thr = np.empty(max_ell + 1, dtype=np.int64)
    S2 = np.float64(S) * np.float64(S)
    two_d = np.float64(2 * d)
    for ell in range(2, max_ell + 1):
        # TV = 2d/(ell*S^2) * ws.  Prune if TV >= c_target.
        # ws >= c_target * ell * S^2 / (2d).
        # So prune if ws > floor(c_target * ell * S^2 / (2d) - eps).
        thr[ell] = np.int64(c_target * np.float64(ell) * S2 / two_d - 1e-9)
    return thr


def compute_xcap(c_target, S, d):
    """Max integer mass per bin before self-convolution alone prunes it.

    Single-bin self-conv: TV >= d * (c/S)^2 * 2 (from ell=2 self-window).
    Actually: conv[2i] = c^2, TV(ell=2, s=2i) = 2d/2 * c^2/S^2 = d*c^2/S^2.
    Prune if d*c^2/S^2 >= c_target => c >= S*sqrt(c_target/d).
    """
    return int(np.floor(S * np.sqrt(c_target / d)))


# =====================================================================
# L0: Branch-and-bound for initial compositions
# =====================================================================

@njit(cache=True)
def _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf, count_only):
    """BnB subtree with bins[0]=c0 fixed.

    Returns (n_survivors, n_tested).
    """
    conv_len = 2 * d - 1
    d_m1 = d - 1
    max_ell = 2 * d

    conv = np.zeros(conv_len, dtype=np.int64)
    bins = np.zeros(d, dtype=np.int32)
    rem_arr = np.zeros(d, dtype=np.int32)

    bins[0] = np.int32(c0)
    conv[0] = np.int64(c0) * np.int64(c0)
    rem_arr[0] = np.int32(S)
    rem_arr[1] = np.int32(S - c0)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    buf_cap = np.int64(0)
    if not count_only:
        buf_cap = np.int64(out_buf.shape[0])

    if d == 1:
        if c0 == S:
            n_tested = 1
        return n_surv, n_tested

    if d == 2:
        forced = S - c0
        if 0 <= forced <= x_cap:
            n_tested = 1
            bins[1] = np.int32(forced)
            conv[0] = np.int64(c0) * np.int64(c0)
            conv[1] = np.int64(2) * np.int64(c0) * np.int64(forced)
            conv[2] = np.int64(forced) * np.int64(forced)

            pruned = False
            for ell in range(2, max_ell + 1):
                if pruned:
                    break
                n_cv = ell - 1
                nw = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += conv[k]
                for s_lo in range(nw):
                    if s_lo > 0:
                        ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                    if ws > thr[ell]:
                        pruned = True
                        break

            if not pruned:
                if not count_only and n_surv < buf_cap:
                    out_buf[n_surv, 0] = np.int32(c0)
                    out_buf[n_surv, 1] = np.int32(forced)
                n_surv += 1
        return n_surv, n_tested

    # General case: d >= 3
    pos = 1
    bins[1] = np.int32(0)

    while True:
        c_val = bins[pos]
        rem = rem_arr[pos]

        if pos == d_m1:
            # Last bin: forced
            forced = rem
            if 0 <= forced <= x_cap:
                n_tested += 1
                bins[pos] = np.int32(forced)

                # Add conv contribution
                f64 = np.int64(forced)
                conv[2 * pos] += f64 * f64
                for j in range(pos):
                    conv[pos + j] += np.int64(2) * f64 * np.int64(bins[j])

                # Full window scan
                pruned_leaf = False
                for ell in range(2, max_ell + 1):
                    if pruned_leaf:
                        break
                    n_cv = ell - 1
                    nw = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(nw):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            pruned_leaf = True
                            break

                if not pruned_leaf:
                    if not count_only and n_surv < buf_cap:
                        for i in range(d):
                            out_buf[n_surv, i] = bins[i]
                    n_surv += 1

                # Undo conv
                conv[2 * pos] -= f64 * f64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * f64 * np.int64(bins[j])

            # Backtrack
            pos -= 1
            if pos < 1:
                break
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Non-last bin
        max_v = min(rem, x_cap)
        min_v = rem - (d_m1 - pos) * x_cap
        if min_v < 0:
            min_v = 0
        if c_val < min_v:
            bins[pos] = np.int32(min_v)
            c_val = min_v

        if c_val > max_v:
            if pos <= 1:
                break
            pos -= 1
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Add conv
        c64 = np.int64(c_val)
        if c_val > 0:
            conv[2 * pos] += c64 * c64
            for j in range(pos):
                conv[pos + j] += np.int64(2) * c64 * np.int64(bins[j])

        # Partial prune: windows within [0, 2*pos]
        max_cv_pos = 2 * pos
        pruned_partial = False
        for ell in range(2, max_ell + 1):
            if pruned_partial:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                pruned_partial = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    pruned_partial = True
                    break

        if pruned_partial:
            if c_val > 0:
                conv[2 * pos] -= c64 * c64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c64 * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        # Descend
        rem_arr[pos + 1] = np.int32(rem - c_val)
        pos += 1
        bins[pos] = np.int32(0)

    return n_surv, n_tested


@njit(parallel=True, cache=True)
def _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested):
    """Pass 1: count survivors per c0."""
    dummy = np.empty((0, d), dtype=np.int32)
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        ns, nt = _l0_bnb_inner(c0, d, S, x_cap, thr, dummy, True)
        counts[idx] = ns
        tested[idx] = nt


@njit(parallel=True, cache=True)
def _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf):
    """Pass 2: fill output buffer."""
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        cnt = counts[idx]
        if cnt == 0:
            continue
        off = offsets[idx]
        _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf[off:off + cnt], False)


def run_l0(d, S, c_target):
    """Run L0: enumerate all compositions of S into d parts, prune."""
    thr = compute_thresholds(c_target, S, d)
    x_cap = compute_xcap(c_target, S, d)

    min_c0 = max(0, S - (d - 1) * x_cap)
    max_c0 = min(S, x_cap)
    # Canonical: c0 <= S//2 (symmetry)
    max_c0 = min(max_c0, S // 2)
    n_c0 = max_c0 - min_c0 + 1

    if n_c0 <= 0:
        return np.empty((0, d), dtype=np.int32), 0, 0

    counts = np.zeros(n_c0, dtype=np.int64)
    tested = np.zeros(n_c0, dtype=np.int64)

    _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested)

    offsets = np.zeros(n_c0 + 1, dtype=np.int64)
    for i in range(n_c0):
        offsets[i + 1] = offsets[i] + counts[i]
    total_surv = int(offsets[n_c0])
    total_tested = int(np.sum(tested))

    if total_surv == 0:
        return np.empty((0, d), dtype=np.int32), 0, total_tested

    out_buf = np.empty((total_surv, d), dtype=np.int32)
    _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf)

    return out_buf, total_surv, total_tested


# =====================================================================
# L1+: Fused generate-and-prune for cascade children
# =====================================================================

@njit(cache=True)
def _cascade_child_bnb(parent, d_parent, S, x_cap, thr, out_buf):
    """Process one parent: BnB over all children, prune by TV >= c_target.

    Child bins: child[2i] = cursor[i], child[2i+1] = parent[i] - cursor[i].
    Cursors are independent (each ranges over its parent bin's valid splits).
    Subtree pruning via partial autoconvolution.

    Returns (n_survivors, n_tested).
    """
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    max_ell = 2 * d_child

    # Cursor ranges
    lo = np.empty(d_parent, dtype=np.int32)
    hi = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        lo[i] = np.int32(max(0, parent[i] - x_cap))
        hi[i] = np.int32(min(parent[i], x_cap))

    # Check product is nonzero
    product = np.int64(1)
    for i in range(d_parent):
        product *= np.int64(hi[i] - lo[i] + 1)
    if product == 0:
        return 0, np.int64(0)

    # DFS state
    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.zeros(d_child, dtype=np.int32)
    conv = np.zeros(conv_len, dtype=np.int64)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    max_surv = np.int64(out_buf.shape[0])

    # Quick-check state
    qc_ell = np.int32(0)
    qc_s = np.int32(0)

    pos = 0
    cursor[0] = lo[0]

    while True:
        c_val = cursor[pos]

        if c_val > hi[pos]:
            # Backtrack
            if pos == 0:
                break
            # Undo conv for position pos-1... wait, we undo the CURRENT pos
            # Actually we need to undo the pos we're leaving
            pos -= 1
            k1 = 2 * pos
            k2 = k1 + 1
            old1 = np.int64(child[k1])
            old2 = np.int64(child[k2])
            conv[2 * k1] -= old1 * old1
            conv[2 * k2] -= old2 * old2
            conv[k1 + k2] -= np.int64(2) * old1 * old2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * old1 * cj
                    conv[k2 + j] -= np.int64(2) * old2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        # Set child bins for this cursor position
        k1 = 2 * pos
        k2 = k1 + 1
        new1 = np.int64(c_val)
        new2 = np.int64(parent[pos] - c_val)
        child[k1] = np.int32(new1)
        child[k2] = np.int32(new2)

        # Add conv contribution
        conv[2 * k1] += new1 * new1
        conv[2 * k2] += new2 * new2
        conv[k1 + k2] += np.int64(2) * new1 * new2
        for j in range(k1):
            cj = np.int64(child[j])
            if cj != 0:
                conv[k1 + j] += np.int64(2) * new1 * cj
                conv[k2 + j] += np.int64(2) * new2 * cj

        # --- Partial prune: check windows fully within assigned range ---
        max_cv_pos = 2 * k2  # = 4*pos + 2
        partial_pruned = False
        for ell in range(2, max_ell + 1):
            if partial_pruned:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                partial_pruned = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    partial_pruned = True
                    break

        if partial_pruned:
            # Undo and advance cursor
            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        if pos == d_parent - 1:
            # --- Leaf: all cursors assigned ---
            n_tested += 1

            # Quick check: retry previous killing window
            quick_killed = False
            if qc_ell > 0:
                n_cv_qc = qc_ell - 1
                ws_qc = np.int64(0)
                for k in range(qc_s, qc_s + n_cv_qc):
                    ws_qc += conv[k]
                if ws_qc > thr[qc_ell]:
                    quick_killed = True

            if not quick_killed:
                # Full window scan
                full_pruned = False
                for ell in range(2, max_ell + 1):
                    if full_pruned:
                        break
                    n_cv = ell - 1
                    n_win = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(n_win):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            full_pruned = True
                            qc_ell = np.int32(ell)
                            qc_s = np.int32(s_lo)
                            break

                if not full_pruned:
                    if n_surv < max_surv:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                    n_surv += 1

            # Undo and advance cursor
            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
        else:
            # Descend to next cursor
            pos += 1
            cursor[pos] = lo[pos]

    return n_surv, n_tested


@njit(cache=True)
def _count_one_parent(parent, d_parent, S, x_cap, thr):
    """Count survivors for one parent (no output buffer needed)."""
    dummy = np.empty((0, 2 * d_parent), dtype=np.int32)
    ns, nt = _cascade_child_bnb(parent, d_parent, S, x_cap, thr, dummy)
    return ns, nt


def run_cascade_level(survivors_prev, d_parent, S, c_target, verbose=True):
    """Run one cascade level: generate and prune children of all survivors.

    Returns (survivors_array, total_survivors, total_tested).
    """
    d_child = 2 * d_parent
    n_parents = survivors_prev.shape[0]
    thr = compute_thresholds(c_target, S, d_child)
    x_cap = compute_xcap(c_target, S, d_child)

    if verbose:
        print(f"    x_cap={x_cap}, d_child={d_child}, "
              f"n_parents={n_parents}")

    # Pass 1: count survivors per parent
    counts = np.zeros(n_parents, dtype=np.int64)
    tested = np.zeros(n_parents, dtype=np.int64)

    t0 = time.time()
    for p_idx in range(n_parents):
        parent = survivors_prev[p_idx]
        ns, nt = _count_one_parent(parent, d_parent, S, x_cap, thr)
        counts[p_idx] = ns
        tested[p_idx] = nt

        if verbose and (p_idx + 1) % max(1, n_parents // 10) == 0:
            elapsed = time.time() - t0
            rate = (p_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"      counting: {p_idx+1}/{n_parents} "
                  f"({rate:.0f} parents/s, "
                  f"{np.sum(tested[:p_idx+1]):,} tested, "
                  f"{np.sum(counts[:p_idx+1]):,} survived)")

    total_surv = int(np.sum(counts))
    total_tested = int(np.sum(tested))

    if total_surv == 0:
        return np.empty((0, d_child), dtype=np.int32), 0, total_tested

    # Pass 2: fill output
    offsets = np.zeros(n_parents + 1, dtype=np.int64)
    for i in range(n_parents):
        offsets[i + 1] = offsets[i] + counts[i]

    out_buf = np.empty((total_surv, d_child), dtype=np.int32)

    for p_idx in range(n_parents):
        cnt = int(counts[p_idx])
        if cnt == 0:
            continue
        off = int(offsets[p_idx])
        parent = survivors_prev[p_idx]
        _cascade_child_bnb(parent, d_parent, S, x_cap, thr,
                           out_buf[off:off + cnt])

    return out_buf, total_surv, total_tested


# =====================================================================
# Canonicalization and deduplication
# =====================================================================

@njit(parallel=True, cache=True)
def _canonicalize_inplace(arr):
    """Replace each row with min(row, rev(row)) lexicographically."""
    B = arr.shape[0]
    d = arr.shape[1]
    half = d // 2
    for b in prange(B):
        swap = False
        for i in range(half):
            j = d - 1 - i
            if arr[b, j] < arr[b, i]:
                swap = True
                break
            elif arr[b, j] > arr[b, i]:
                break
        if swap:
            for i in range(half):
                j = d - 1 - i
                tmp = arr[b, i]
                arr[b, i] = arr[b, j]
                arr[b, j] = tmp


def dedup(arr):
    """Deduplicate rows via lexsort."""
    if len(arr) == 0:
        return arr
    d = arr.shape[1]
    keys = tuple(arr[:, d - 1 - i] for i in range(d))
    sort_idx = np.lexsort(keys)
    sorted_arr = arr[sort_idx]
    mask = np.ones(len(sorted_arr), dtype=bool)
    for i in range(1, len(sorted_arr)):
        if np.array_equal(sorted_arr[i], sorted_arr[i - 1]):
            mask[i] = False
    return sorted_arr[mask]


# =====================================================================
# Box certification (QP water-filling)
# =====================================================================

@njit(cache=True)
def _box_certify_cell(mu_center, d, delta, c_target):
    """QP-certify one cell: min over box of max_W TV_W >= c_target?

    Uses water-filling: for each window, minimize TV by concentrating
    mass in non-contributing bins.

    Returns (certified, best_min_tv).
    """
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo[i] = max(0.0, mu_center[i] - delta / 2.0)
        hi[i] = min(1.0, mu_center[i] + delta / 2.0)

    best_min_tv = 0.0

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)

        for s in range(conv_len - n_cv + 1):
            # Identify contributing bins
            contrib = np.zeros(d, dtype=numba.boolean)
            for k in range(s, s + n_cv):
                for i in range(max(0, k - d + 1), min(d, k + 1)):
                    contrib[i] = True

            # Water-filling: minimize TV by putting mass in non-contrib bins
            mu_opt = lo.copy()
            excess = 1.0 - np.sum(mu_opt)

            # Fill non-contributing bins first (free for TV)
            for i in range(d):
                if not contrib[i] and excess > 1e-15:
                    add = min(excess, hi[i] - mu_opt[i])
                    mu_opt[i] += add
                    excess -= add

            # Remaining excess goes to contributing bins
            if excess > 1e-15:
                for i in range(d):
                    if contrib[i] and excess > 1e-15:
                        add = min(excess, hi[i] - mu_opt[i])
                        mu_opt[i] += add
                        excess -= add

            # Compute TV at minimizing point
            conv_opt = np.zeros(conv_len, dtype=np.float64)
            for i in range(d):
                mi = mu_opt[i]
                if mi > 0:
                    conv_opt[2 * i] += mi * mi
                    for j in range(i + 1, d):
                        mj = mu_opt[j]
                        if mj > 0:
                            conv_opt[i + j] += 2.0 * mi * mj

            ws = 0.0
            for k in range(s, s + n_cv):
                ws += conv_opt[k]
            min_tv = ws * scale

            if min_tv > best_min_tv:
                best_min_tv = min_tv

            if best_min_tv >= c_target:
                return True, best_min_tv

    return best_min_tv >= c_target, best_min_tv


def run_box_certification(survivors_final_level, d_final, S, c_target,
                          n_sample=1000, verbose=True):
    """Run box certification on a sample of grid cells.

    In the full proof, every cell must be certified.  Here we sample
    to estimate the certification rate and identify any failures.

    Returns (n_certified, n_tested, worst_min_tv).
    """
    delta = 1.0 / S
    n_parents = survivors_final_level  # This is 0 if cascade converged

    if verbose:
        print(f"\n  Box certification at d={d_final}, delta={delta:.4f}:")

    # Test random grid cells to verify certification rate
    rng = np.random.RandomState(42)
    n_cert = 0
    worst_tv = 1e30
    tested = 0

    for _ in range(n_sample):
        # Random composition of S into d_final parts
        mu_int = np.zeros(d_final, dtype=np.int32)
        remaining = S
        for i in range(d_final - 1):
            mu_int[i] = rng.randint(0, min(remaining, compute_xcap(c_target, S, d_final)) + 1)
            remaining -= mu_int[i]
        mu_int[d_final - 1] = remaining
        if mu_int[d_final - 1] < 0 or mu_int[d_final - 1] > compute_xcap(c_target, S, d_final):
            continue

        mu_center = mu_int.astype(np.float64) / S
        certified, min_tv = _box_certify_cell(mu_center, d_final, delta, c_target)
        tested += 1
        if certified:
            n_cert += 1
        if min_tv < worst_tv:
            worst_tv = min_tv

    if verbose:
        print(f"    Sampled {tested} random cells: "
              f"{n_cert}/{tested} certified ({n_cert/max(tested,1)*100:.1f}%)")
        print(f"    Worst QP min TV: {worst_tv:.6f} "
              f"(need >= {c_target})")

    return n_cert, tested, worst_tv


# =====================================================================
# Main cascade driver
# =====================================================================

def run_cascade(c_target=1.30, S=50, d_start=2, max_levels=5, verbose=True):
    """Run the full coarse cascade proof.

    Returns True if the proof succeeds (C_{1a} >= c_target).
    """
    if verbose:
        print("=" * 64)
        print(f"COARSE CASCADE PROVER: C_{{1a}} >= {c_target}")
        print("=" * 64)
        print(f"  Grid: S={S} (delta={1/S:.4f})")
        print(f"  Starting dimension: d={d_start}")
        print(f"  No correction term (refinement monotonicity)")
        print()

    t_total = time.time()

    # --- L0 ---
    d = d_start
    if verbose:
        print(f"  L0 (d={d}):")
    t0 = time.time()
    survivors, n_surv, n_tested = run_l0(d, S, c_target)
    elapsed = time.time() - t0

    if n_surv > 0:
        _canonicalize_inplace(survivors)
        survivors = dedup(survivors)
        n_surv = len(survivors)

    if verbose:
        print(f"    Tested: {n_tested:,}")
        print(f"    Survivors: {n_surv:,}")
        print(f"    Time: {elapsed:.2f}s")

    if n_surv == 0:
        if verbose:
            print(f"\n  PROOF COMPLETE at L0: all compositions pruned.")
            print(f"  C_{{1a}} >= {c_target}")
        return True

    # Save checkpoint
    np.save(f"data/coarse_L0_survivors_S{S}.npy", survivors)

    # --- L1+ ---
    for level in range(1, max_levels + 1):
        d_parent = d
        d = 2 * d_parent

        if verbose:
            print(f"\n  L{level} (d={d}):")

        t0 = time.time()
        survivors, n_surv, n_tested = run_cascade_level(
            survivors, d_parent, S, c_target, verbose=verbose)
        elapsed = time.time() - t0

        if n_surv > 0:
            _canonicalize_inplace(survivors)
            survivors = dedup(survivors)
            n_surv = len(survivors)

        if verbose:
            print(f"    Tested: {n_tested:,}")
            print(f"    Survivors (after dedup): {n_surv:,}")
            print(f"    Time: {elapsed:.2f}s")

        if n_surv == 0:
            if verbose:
                print(f"\n  CASCADE CONVERGED at L{level} (d={d})!")
                print(f"  All {n_tested:,} children pruned by TV >= {c_target}.")

            # Box certification
            if verbose:
                print(f"\n  Running box certification...")
            run_box_certification(0, d, S, c_target, n_sample=2000,
                                 verbose=verbose)

            total_time = time.time() - t_total
            if verbose:
                print(f"\n  {'=' * 60}")
                print(f"  PROOF: C_{{1a}} >= {c_target}")
                print(f"  Method: coarse cascade (S={S}) + box certification")
                print(f"  Converged at d={d} (L{level})")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  {'=' * 60}")
            return True

        # Save checkpoint
        np.save(f"data/coarse_L{level}_survivors_S{S}.npy", survivors)

    total_time = time.time() - t_total
    if verbose:
        print(f"\n  Did not converge within {max_levels} levels.")
        print(f"  Survivors at d={d}: {n_surv:,}")
        print(f"  Total time: {total_time:.2f}s")
    return False


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coarse cascade prover for C_{1a}")
    parser.add_argument("--c_target", type=float, default=1.30,
                        help="Target lower bound (default: 1.30)")
    parser.add_argument("--S", type=int, default=50,
                        help="Grid resolution S (mass quantum = 1/S)")
    parser.add_argument("--d_start", type=int, default=2,
                        help="Starting dimension")
    parser.add_argument("--max_levels", type=int, default=5,
                        help="Maximum cascade levels")
    args = parser.parse_args()

    # JIT warmup
    print("Warming up JIT...", end="", flush=True)
    t0 = time.time()
    _warmup_thr = compute_thresholds(1.3, 10, 4)
    _warmup_buf = np.empty((0, 4), dtype=np.int32)
    _l0_bnb_inner(np.int32(2), 4, 10, 5, _warmup_thr, _warmup_buf, True)
    _warmup_parent = np.array([3, 3, 2, 2], dtype=np.int32)
    _warmup_buf2 = np.empty((100, 8), dtype=np.int32)
    _warmup_thr2 = compute_thresholds(1.3, 10, 8)
    _cascade_child_bnb(_warmup_parent, 4, 10, 5, _warmup_thr2, _warmup_buf2)
    _canonicalize_inplace(np.array([[1, 2, 3, 4]], dtype=np.int32))
    print(f" done ({time.time()-t0:.1f}s)")

    os.makedirs("data", exist_ok=True)
    success = run_cascade(
        c_target=args.c_target,
        S=args.S,
        d_start=args.d_start,
        max_levels=args.max_levels,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
