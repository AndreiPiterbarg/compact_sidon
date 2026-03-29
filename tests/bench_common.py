"""Shared utilities for optimization idea benchmarks.

Provides data loading, JIT helpers, threshold table construction,
floor-convolution tightening, and standalone window-scan kernels.
All JIT functions replicate the kernel's exact arithmetic to ensure
fair, apples-to-apples comparison.
"""
import os
import sys
import time
import math

import numpy as np
import numba
from numba import njit

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_proj_dir, "cloninger-steinerberger")
_cpu_dir = os.path.join(_cs_dir, "cpu")
if _cs_dir not in sys.path:
    sys.path.insert(0, _cs_dir)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)

from run_cascade import (
    _compute_bin_ranges,
    _fused_generate_and_prune_gray,
    process_parent_fused,
)
from pruning import correction

# ---------------------------------------------------------------------------
# Constants (matching production run: n_half=2, m=20, c_target=1.4)
# ---------------------------------------------------------------------------
M = 20
C_TARGET = 1.4
DATA_DIR = os.path.join(_proj_dir, "data")


# ---------------------------------------------------------------------------
# Data loading / generation
# ---------------------------------------------------------------------------

def load_l0_parents():
    """Load L0 checkpoint survivors (parents for L1)."""
    return np.load(os.path.join(DATA_DIR, "checkpoint_L0_survivors.npy"))


def level_params(level):
    """Return (d_parent, d_child, n_half_child) for a cascade level.

    Level 1: d_parent=4  -> d_child=8,  n_half_child=4
    Level 2: d_parent=8  -> d_child=16, n_half_child=8
    Level 3: d_parent=16 -> d_child=32, n_half_child=16
    """
    d_parent = 4 * (2 ** (level - 1))
    d_child = 2 * d_parent
    n_half_child = d_parent
    return d_parent, d_child, n_half_child


def generate_parents_for_level(level, max_parents=None, rng_seed=42,
                                verbose=True):
    """Generate real parents for *level* by cascading from L0.

    Returns ndarray of shape (N, d_parent) with dtype int32.
    """
    d_parent, d_child, n_half_child = level_params(level)

    if level == 1:
        parents = load_l0_parents()
        if verbose:
            print(f"  L1 parents: loaded {len(parents)} L0 survivors (d={parents.shape[1]})")
    elif level == 2:
        l0 = load_l0_parents()
        _, _, nhc1 = level_params(1)
        survivors = []
        if verbose:
            print(f"  Generating L1 survivors from {len(l0)} L0 parents ...", end="", flush=True)
        for i in range(len(l0)):
            s, _ = process_parent_fused(l0[i], M, C_TARGET, nhc1)
            if len(s) > 0:
                survivors.append(s)
        parents = np.vstack(survivors) if survivors else np.empty((0, d_parent), np.int32)
        if verbose:
            print(f" {len(parents)} L1 survivors")
    elif level == 3:
        l1 = generate_parents_for_level(2, max_parents=200, rng_seed=rng_seed, verbose=verbose)
        _, _, nhc2 = level_params(2)
        survivors = []
        if verbose:
            print(f"  Generating L2 survivors from {len(l1)} L1 parents ...", end="", flush=True)
        for i in range(len(l1)):
            s, _ = process_parent_fused(l1[i], M, C_TARGET, nhc2)
            if len(s) > 0:
                survivors.append(s)
        parents = np.vstack(survivors) if survivors else np.empty((0, d_parent), np.int32)
        if verbose:
            print(f" {len(parents)} L2 survivors")
    else:
        raise ValueError(f"level must be 1, 2 or 3, got {level}")

    if max_parents is not None and len(parents) > max_parents:
        rng = np.random.default_rng(rng_seed + level)
        idx = rng.choice(len(parents), max_parents, replace=False)
        parents = parents[np.sort(idx)]
        if verbose:
            print(f"  Sampled {max_parents} parents")

    return parents


# ---------------------------------------------------------------------------
# JIT helpers — replicate kernel arithmetic exactly
# ---------------------------------------------------------------------------

@njit(cache=True)
def build_threshold_table(d_child, m, c_target, n_half_child):
    """2-D threshold lookup: threshold_table[ell_idx * (m+1) + W_int]."""
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    m_d = np.float64(m)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_d * m_d

    ell_count = 2 * d_child - 1
    m_plus_1 = m + 1
    table = np.empty(ell_count * m_plus_1, dtype=np.int64)

    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        base = c_target * m_d * m_d * np.float64(ell) * inv_4n
        for w in range(m_plus_1):
            dyn_x = base + 1.0 + eps_margin + 2.0 * np.float64(w)
            table[idx * m_plus_1 + w] = np.int64(dyn_x * one_minus_4eps)
    return table


@njit(cache=True)
def build_ell_order(d_child):
    """Profile-guided ell scan order (matches kernel exactly)."""
    ell_count = 2 * d_child - 1
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0

    if d_child >= 20:
        hc = d_child // 2
        for ell in (hc + 1, hc + 2, hc + 3, hc, hc - 1, hc + 4, hc + 5,
                    hc - 2, hc + 6, hc - 3, hc + 7, hc + 8):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = 1
                oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, d_child * 2, d_child + d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = 1
                oi += 1
    else:
        phase1_end = min(16, 2 * d_child)
        for ell in range(2, phase1_end + 1):
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = 1
            oi += 1
        for ell in (d_child, d_child + 1, d_child - 1, d_child + 2,
                    d_child - 2, d_child * 2, d_child + d_child // 2,
                    d_child // 2):
            if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
                ell_order[oi] = np.int32(ell)
                ell_used[ell - 2] = 1
                oi += 1

    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1
    return ell_order


@njit(cache=True)
def build_parent_prefix(parent_int):
    d = len(parent_int)
    pp = np.empty(d + 1, dtype=np.int64)
    pp[0] = 0
    for i in range(d):
        pp[i + 1] = pp[i] + np.int64(parent_int[i])
    return pp


@njit(cache=True)
def compute_autoconv(child, d_child):
    """Autoconvolution (int32, matching kernel)."""
    conv_len = 2 * d_child - 1
    conv = np.zeros(conv_len, dtype=np.int32)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    conv[i + j] += np.int32(2) * ci * cj
    return conv


@njit(cache=True)
def build_floor_child(parent_int, lo_arr, hi_arr):
    """Floor child: position i gets (lo[i], parent[i]-hi[i])."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    fc = np.empty(d_child, dtype=np.int32)
    for i in range(d_parent):
        fc[2 * i] = lo_arr[i]
        fc[2 * i + 1] = parent_int[i] - hi_arr[i]
    return fc


# ---------------------------------------------------------------------------
# Floor-convolution infeasibility check
# ---------------------------------------------------------------------------

@njit(cache=True)
def _check_cursor_infeasible(parent_int, lo_arr, hi_arr, pos, cursor_val,
                              d_child, m, threshold_table, parent_prefix,
                              ell_order, ell_count):
    """True if ALL children with cursor[pos]=cursor_val are provably pruned.

    Builds floor child with position *pos* fixed at cursor_val, all others
    at their floor values.  Computes autoconvolution and checks if any
    window sum exceeds threshold (using W_int_max from parent_prefix).

    Soundness: product monotonicity (child >= floor componentwise) plus
    threshold monotonicity (W_int_actual <= W_int_max).
    """
    d_parent = len(parent_int)
    m_plus_1 = m + 1
    conv_len = 2 * d_child - 1

    # Build floor child with pos fixed
    floor_child = np.empty(d_child, dtype=np.int32)
    for i in range(d_parent):
        if i == pos:
            floor_child[2 * i] = np.int32(cursor_val)
            floor_child[2 * i + 1] = parent_int[i] - np.int32(cursor_val)
        else:
            floor_child[2 * i] = lo_arr[i]
            floor_child[2 * i + 1] = parent_int[i] - hi_arr[i]

    # Autoconvolution
    conv = np.zeros(conv_len, dtype=np.int32)
    for i in range(d_child):
        ci = np.int32(floor_child[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(floor_child[j])
                if cj != 0:
                    conv[i + j] += np.int32(2) * ci * cj

    # Conv prefix sum
    prefix = np.empty(conv_len + 1, dtype=np.int64)
    prefix[0] = 0
    for k in range(conv_len):
        prefix[k + 1] = prefix[k] + np.int64(conv[k])

    # Window scan with W_int_max from parent_prefix
    for ell_oi in range(ell_count):
        ell = ell_order[ell_oi]
        n_cv = ell - 1
        ell_idx = ell - 2
        n_windows = conv_len - n_cv + 1

        for s_lo in range(n_windows):
            ws = prefix[s_lo + n_cv] - prefix[s_lo]
            lo_bin = s_lo - (d_child - 1)
            if lo_bin < 0:
                lo_bin = 0
            hi_bin = s_lo + ell - 2
            if hi_bin > d_child - 1:
                hi_bin = d_child - 1
            p_lo = lo_bin // 2
            p_hi = hi_bin // 2
            if p_hi >= d_parent:
                p_hi = d_parent - 1
            W_int = parent_prefix[p_hi + 1] - parent_prefix[p_lo]
            if W_int > np.int64(m):
                W_int = np.int64(m)
            dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
            if ws > dyn_it:
                return True
    return False


# ---------------------------------------------------------------------------
# Iterative cursor-range tightening (Idea 1 core)
# ---------------------------------------------------------------------------

@njit(cache=True)
def tighten_cursor_ranges(parent_int, lo_arr, hi_arr, d_child, m, c_target,
                           n_half_child, threshold_table, parent_prefix,
                           ell_order):
    """Iteratively tighten cursor ranges via floor-convolution bound.

    Returns (new_lo, new_hi, n_values_removed, n_iterations).
    """
    d_parent = len(parent_int)
    ell_count = 2 * d_child - 1

    new_lo = lo_arr.copy()
    new_hi = hi_arr.copy()
    n_removed = 0
    n_iters = 0

    changed = True
    while changed:
        changed = False
        n_iters += 1
        if n_iters > 100:
            break

        for p in range(d_parent):
            if new_hi[p] <= new_lo[p]:
                continue

            # Check lo endpoint
            if _check_cursor_infeasible(
                    parent_int, new_lo, new_hi, p, new_lo[p],
                    d_child, m, threshold_table, parent_prefix,
                    ell_order, ell_count):
                new_lo[p] += 1
                n_removed += 1
                changed = True

            # Check hi endpoint (re-check: lo may have moved)
            if new_hi[p] > new_lo[p]:
                if _check_cursor_infeasible(
                        parent_int, new_lo, new_hi, p, new_hi[p],
                        d_child, m, threshold_table, parent_prefix,
                        ell_order, ell_count):
                    new_hi[p] -= 1
                    n_removed += 1
                    changed = True
    return new_lo, new_hi, n_removed, n_iters


# ---------------------------------------------------------------------------
# Product computation
# ---------------------------------------------------------------------------

@njit(cache=True)
def compute_product(lo_arr, hi_arr):
    """Cartesian product size from cursor ranges."""
    product = np.int64(1)
    for i in range(len(lo_arr)):
        r = np.int64(hi_arr[i] - lo_arr[i] + 1)
        if r <= 0:
            return np.int64(0)
        product *= r
    return product


# ---------------------------------------------------------------------------
# Standalone child-level pruning check (for verification)
# ---------------------------------------------------------------------------

@njit(cache=True)
def child_survives_check(child, d_child, m, threshold_table, ell_order,
                          ell_count):
    """True if child passes the full window scan (no window exceeds threshold).

    Uses exact child W_int — matches the kernel's own pruning logic.
    """
    m_plus_1 = m + 1
    conv_len = 2 * d_child - 1

    conv = np.zeros(conv_len, dtype=np.int32)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    conv[i + j] += np.int32(2) * ci * cj

    child_prefix = np.empty(d_child + 1, dtype=np.int64)
    child_prefix[0] = 0
    for j in range(d_child):
        child_prefix[j + 1] = child_prefix[j] + np.int64(child[j])

    conv_prefix = np.empty(conv_len + 1, dtype=np.int64)
    conv_prefix[0] = 0
    for k in range(conv_len):
        conv_prefix[k + 1] = conv_prefix[k] + np.int64(conv[k])

    for ell_oi in range(ell_count):
        ell = ell_order[ell_oi]
        n_cv = ell - 1
        ell_idx = ell - 2
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ws = conv_prefix[s_lo + n_cv] - conv_prefix[s_lo]
            lo_bin = s_lo - (d_child - 1)
            if lo_bin < 0:
                lo_bin = 0
            hi_bin = s_lo + ell - 2
            if hi_bin > d_child - 1:
                hi_bin = d_child - 1
            W_int = child_prefix[hi_bin + 1] - child_prefix[lo_bin]
            if W_int > np.int64(m):
                W_int = np.int64(m)
            dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
            if ws > dyn_it:
                return False
    return True


# ---------------------------------------------------------------------------
# Window-scan kernels for Idea 2 benchmarking
# ---------------------------------------------------------------------------

@njit(cache=True)
def prepare_scan_data(children, d_child):
    """Precompute conv_prefix and child_prefix for a batch of children."""
    B = children.shape[0]
    conv_len = 2 * d_child - 1
    conv_prefix = np.empty((B, conv_len + 1), dtype=np.int64)
    child_prefix = np.empty((B, d_child + 1), dtype=np.int64)

    for b in range(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d_child):
            ci = np.int32(children[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d_child):
                    cj = np.int32(children[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj
        conv_prefix[b, 0] = 0
        for k in range(conv_len):
            conv_prefix[b, k + 1] = conv_prefix[b, k] + np.int64(conv[k])
        child_prefix[b, 0] = 0
        for j in range(d_child):
            child_prefix[b, j + 1] = child_prefix[b, j] + np.int64(children[b, j])
    return conv_prefix, child_prefix


@njit(cache=True)
def scan_sequential(conv_prefix, child_prefix, d_child, m,
                     threshold_table, ell_order):
    """Sequential window scan: one child at a time (current approach)."""
    B = conv_prefix.shape[0]
    ell_count = 2 * d_child - 1
    m_plus_1 = m + 1
    conv_len = 2 * d_child - 1
    pruned = np.zeros(B, dtype=numba.boolean)
    total_evals = np.int64(0)

    for b in range(B):
        for ell_oi in range(ell_count):
            if pruned[b]:
                break
            ell = ell_order[ell_oi]
            n_cv = ell - 1
            ell_idx = ell - 2
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                total_evals += 1
                ws = conv_prefix[b, s_lo + n_cv] - conv_prefix[b, s_lo]
                lo_bin = s_lo - (d_child - 1)
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_child - 1:
                    hi_bin = d_child - 1
                W_int = child_prefix[b, hi_bin + 1] - child_prefix[b, lo_bin]
                if W_int > np.int64(m):
                    W_int = np.int64(m)
                dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
                if ws > dyn_it:
                    pruned[b] = True
                    break
    return pruned, total_evals


@njit(cache=True)
def scan_batched(conv_prefix, child_prefix, d_child, m,
                  threshold_table, ell_order, BATCH_SIZE):
    """Batched transposed window scan (Idea 2): B children per batch."""
    B = conv_prefix.shape[0]
    ell_count = 2 * d_child - 1
    m_plus_1 = m + 1
    conv_len = 2 * d_child - 1
    pruned = np.zeros(B, dtype=numba.boolean)
    total_evals = np.int64(0)

    for batch_start in range(0, B, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, B)
        bs = batch_end - batch_start

        for ell_oi in range(ell_count):
            # Early exit: check if all in batch are resolved
            all_done = True
            for bi in range(bs):
                if not pruned[batch_start + bi]:
                    all_done = False
                    break
            if all_done:
                break

            ell = ell_order[ell_oi]
            n_cv = ell - 1
            ell_idx = ell - 2
            n_windows = conv_len - n_cv + 1

            for s_lo in range(n_windows):
                lo_bin = s_lo - (d_child - 1)
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_child - 1:
                    hi_bin = d_child - 1

                for bi in range(bs):
                    b = batch_start + bi
                    if pruned[b]:
                        continue
                    total_evals += 1
                    ws = conv_prefix[b, s_lo + n_cv] - conv_prefix[b, s_lo]
                    W_int = child_prefix[b, hi_bin + 1] - child_prefix[b, lo_bin]
                    if W_int > np.int64(m):
                        W_int = np.int64(m)
                    dyn_it = threshold_table[ell_idx * m_plus_1 + W_int]
                    if ws > dyn_it:
                        pruned[b] = True
    return pruned, total_evals


# ---------------------------------------------------------------------------
# Sampling random children from a parent's Cartesian product
# ---------------------------------------------------------------------------

def sample_children(parent_int, lo_arr, hi_arr, n_samples, rng_seed=123):
    """Sample random children from the Cartesian product."""
    rng = np.random.default_rng(rng_seed)
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    children = np.empty((n_samples, d_child), dtype=np.int32)
    for s in range(n_samples):
        for i in range(d_parent):
            c = rng.integers(int(lo_arr[i]), int(hi_arr[i]) + 1)
            children[s, 2 * i] = c
            children[s, 2 * i + 1] = parent_int[i] - c
    return children


# ---------------------------------------------------------------------------
# Sort rows lexicographically (for comparing survivor sets)
# ---------------------------------------------------------------------------

def sort_rows(arr):
    """Sort rows of a 2-D int array lexicographically."""
    if len(arr) == 0:
        return arr
    keys = [arr[:, i] for i in range(arr.shape[1] - 1, -1, -1)]
    idx = np.lexsort(keys)
    return arr[idx]


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit(verbose=True):
    """Warm up all JIT functions with small inputs."""
    if verbose:
        print("Warming up JIT ...", end=" ", flush=True)
    t0 = time.time()

    parent = np.array([5, 5, 5, 5], dtype=np.int32)
    d_child = 8
    nhc = 4
    tt = build_threshold_table(d_child, M, C_TARGET, nhc)
    eo = build_ell_order(d_child)
    pp = build_parent_prefix(parent)
    lo = np.array([0, 0, 0, 0], dtype=np.int32)
    hi = np.array([5, 5, 5, 5], dtype=np.int32)
    fc = build_floor_child(parent, lo, hi)
    cv = compute_autoconv(fc, d_child)
    _check_cursor_infeasible(parent, lo, hi, 0, 0, d_child, M, tt, pp, eo, len(eo))
    tighten_cursor_ranges(parent, lo, hi, d_child, M, C_TARGET, nhc, tt, pp, eo)
    compute_product(lo, hi)
    child_survives_check(fc, d_child, M, tt, eo, len(eo))

    # scan kernels
    ch = np.zeros((2, d_child), dtype=np.int32)
    ch[0] = fc
    ch[1] = fc
    cp, cpx = prepare_scan_data(ch, d_child)
    scan_sequential(cp, cpx, d_child, M, tt, eo)
    scan_batched(cp, cpx, d_child, M, tt, eo, 2)

    # kernel (triggers Numba compilation of _fused_generate_and_prune_gray)
    process_parent_fused(parent, M, C_TARGET, nhc)

    if verbose:
        print(f"done ({time.time() - t0:.1f}s)")
