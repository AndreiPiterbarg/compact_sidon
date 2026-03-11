"""CPU-only cascade prover — no GPU, no dimension limits.

Runs L0 (composition generation + pruning) then cascades through
refinement levels until all survivors are eliminated or a max
dimension is reached.

Optimizations (integrated from parallel agent work):
  - Fused generate+prune kernel: generates children on-the-fly and prunes
    inline, avoiding 50M+ row intermediate arrays (10-18x speedup on L1-L3)
  - _prune_dynamic with int32/int64 dispatch, pre-computed per-ell constants
  - Numba-parallel canonicalization (replaces Python tuple comparison)
  - Sort-based deduplication (replaces set-of-tuples)
  - JIT warmup at module load

Usage:
    python -m cloninger-steinerberger.cpu.run_cascade
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 2 --m 20 --c_target 1.30
    python -m cloninger-steinerberger.cpu.run_cascade --n_half 3 --m 50 --c_target 1.30 --max_levels 5
"""
import argparse
import json
import math
import multiprocessing as mp
import tempfile
import os
import sys
import time
import itertools

import numpy as np
import numba
from numba import njit, prange

# Path setup — import from parent cloninger-steinerberger/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _cs_dir)

from compositions import (generate_canonical_compositions_batched,
                         generate_compositions_batched)
from pruning import (correction, asymmetry_threshold, count_compositions,
                     asymmetry_prune_mask, _canonical_mask)
from test_values import compute_test_values_batch


# =====================================================================
# Dynamic per-window threshold — int32 path (m <= 200)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_dynamic_int32(batch_int, n_half, m, c_target):
    """int32 path: halves memory bandwidth in autoconvolution inner loop.

    Safe when m <= 200 because max prefix sum of conv = m^2 = 40000,
    which fits comfortably in int32 (max 2,147,483,647).
    Values are widened to int64 only at the threshold comparison point.
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    d_minus_1 = d - 1

    # Pre-compute per-ell constants ONCE (shared read-only across threads)
    max_ell = 2 * d
    dyn_base_ell_arr = np.empty(max_ell + 1, dtype=np.float64)
    two_ell_inv_4n_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        ell_f = np.float64(ell)
        dyn_base_ell_arr[ell] = dyn_base * ell_f * inv_4n
        two_ell_inv_4n_arr[ell] = 2.0 * ell_f * inv_4n

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                conv[i + j] += np.int32(2) * ci * np.int32(batch_int[b, j])
        for k in range(1, conv_len):
            conv[k] += conv[k - 1]

        prefix_c = np.zeros(d + 1, dtype=np.int32)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int32(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            dyn_base_ell = dyn_base_ell_arr[ell]
            two_ell_inv_4n = two_ell_inv_4n_arr[ell]
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                # Widen to int64 for the threshold comparison only
                ws = np.int64(conv[s_hi])
                if s_lo > 0:
                    ws -= np.int64(conv[s_lo - 1])
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = np.int64(prefix_c[hi_bin + 1]) - np.int64(prefix_c[lo_bin])
                dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                dyn_it = np.int64(dyn_x * one_minus_4eps)
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


# =====================================================================
# Dynamic per-window threshold — int64 path (m > 200)
# =====================================================================

@njit(parallel=True, cache=True)
def _prune_dynamic_int64(batch_int, n_half, m, c_target):
    """int64 path for large m values where int32 conv may overflow."""
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    d_minus_1 = d - 1

    max_ell = 2 * d
    dyn_base_ell_arr = np.empty(max_ell + 1, dtype=np.float64)
    two_ell_inv_4n_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        ell_f = np.float64(ell)
        dyn_base_ell_arr[ell] = dyn_base * ell_f * inv_4n
        two_ell_inv_4n_arr[ell] = 2.0 * ell_f * inv_4n

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                conv[i + j] += np.int64(2) * ci * np.int64(batch_int[b, j])
        for k in range(1, conv_len):
            conv[k] += conv[k - 1]

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            dyn_base_ell = dyn_base_ell_arr[ell]
            two_ell_inv_4n = two_ell_inv_4n_arr[ell]
            n_windows = conv_len - n_cv + 1
            for s_lo in range(n_windows):
                s_hi = s_lo + n_cv - 1
                ws = conv[s_hi]
                if s_lo > 0:
                    ws -= conv[s_lo - 1]
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                dyn_it = np.int64(dyn_x * one_minus_4eps)
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


def _prune_dynamic(batch_int, n_half, m, c_target):
    """Per-window dynamic threshold — dispatches int32/int64 based on m.

    Works in integer convolution space.  For each window (ell, s_lo),
    computes W_int and the dynamic integer threshold:
        dyn_it = floor((dyn_base + 2*W_int) * ell / (4*n) * (1 - 4*eps))
    where dyn_base = c_target*m^2 + 1 + 1e-9*m^2.

    Returns boolean mask: True = survived (not pruned).
    """
    if m <= 200:
        return _prune_dynamic_int32(batch_int, n_half, m, c_target)
    else:
        return _prune_dynamic_int64(batch_int, n_half, m, c_target)


# =====================================================================
# Numba-parallel canonicalization
# =====================================================================

@njit(parallel=True, cache=True)
def _canonicalize_inplace(arr):
    """Replace each row with min(row, rev(row)) lexicographically, in-place.

    Much faster than Python-level tuple comparisons: uses Numba prange
    over survivors and an early-exit lexicographic comparison.
    """
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


# =====================================================================
# Sort-based deduplication (Numba)
# =====================================================================

@njit(cache=True)
def _dedup_sorted(arr, sort_idx):
    """Given a sorted array (via sort_idx), return indices of unique rows."""
    n = len(sort_idx)
    d = arr.shape[1]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    keep = np.empty(n, dtype=np.int64)
    keep[0] = sort_idx[0]
    count = 1
    for i in range(1, n):
        curr = sort_idx[i]
        prev = sort_idx[i - 1]
        is_same = True
        for j in range(d):
            if arr[curr, j] != arr[prev, j]:
                is_same = False
                break
        if not is_same:
            keep[count] = curr
            count += 1
    return keep[:count]


def _fast_dedup(arr):
    """Deduplicate rows using lexsort + Numba scan.

    Much faster than set-of-tuples for large arrays because it avoids
    creating Python tuple objects for each row.
    """
    if len(arr) == 0:
        return arr
    d = arr.shape[1]
    keys = tuple(arr[:, d - 1 - i] for i in range(d))
    sort_idx = np.lexsort(keys).astype(np.int64)
    unique_idx = _dedup_sorted(arr, sort_idx)
    return arr[unique_idx]


# =====================================================================
# Sorted merge for large shards (avoids 3x RAM of load+lexsort+dedup)
# =====================================================================

@njit(cache=True)
def _sorted_merge_dedup_kernel(a, b, out):
    """Two-pointer merge of two sorted, deduped 2D int32 arrays.

    Both inputs must be in lexicographic order with no duplicate rows
    (as produced by _fast_dedup).  Output is sorted and deduplicated
    (cross-shard duplicates removed).

    Uses memory-mapped inputs so peak RAM is only the output buffer,
    not 3x total like load+vstack+lexsort.

    Returns number of rows written to out.
    """
    na = a.shape[0]
    nb = b.shape[0]
    d = a.shape[1]
    i = 0
    j = 0
    k = 0

    while i < na and j < nb:
        # Lexicographic compare a[i] vs b[j]
        cmp = 0
        for c in range(d):
            if a[i, c] < b[j, c]:
                cmp = -1
                break
            elif a[i, c] > b[j, c]:
                cmp = 1
                break

        if cmp < 0:
            for c in range(d):
                out[k, c] = a[i, c]
            k += 1
            i += 1
        elif cmp > 0:
            for c in range(d):
                out[k, c] = b[j, c]
            k += 1
            j += 1
        else:
            # Equal row — take one copy, advance both
            for c in range(d):
                out[k, c] = a[i, c]
            k += 1
            i += 1
            j += 1

    while i < na:
        for c in range(d):
            out[k, c] = a[i, c]
        k += 1
        i += 1

    while j < nb:
        for c in range(d):
            out[k, c] = b[j, c]
        k += 1
        j += 1

    return k


def _merge_dedup_shards(shard_paths, d, verbose=False):
    """Merge and deduplicate disk shards using pairwise reduction.

    Uses a tournament-style merge: pairs of shards are merged and
    deduped, results written back to disk.  Peak memory is ~3x the
    size of the two shards being merged.

    Returns (array_or_None, remaining_shard_paths).
    - If everything merges into one shard that fits in RAM:
      returns (array, [])
    - If shards are too large to merge in RAM:
      returns (None, [list of shard file paths])
    """
    if not shard_paths:
        return np.empty((0, d), dtype=np.int32), []

    if len(shard_paths) == 1:
        arr = np.load(shard_paths[0])
        try:
            os.remove(shard_paths[0])
        except OSError:
            pass
        return arr, []

    current = list(shard_paths)
    merge_round = 0

    while len(current) > 1:
        merge_round += 1
        next_round = []
        hit_mem_limit = False
        if verbose:
            _log(f"       Merge round {merge_round}: {len(current)} shards")

        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                if hit_mem_limit:
                    # Can't merge any more — carry remaining shards forward
                    next_round.append(current[i])
                    next_round.append(current[i + 1])
                    continue

                # Check RAM before attempting merge
                a_size = os.path.getsize(current[i])
                b_size = os.path.getsize(current[i + 1])
                need_bytes = (a_size + b_size) * 3
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                except ImportError:
                    avail = int(50e9)
                if need_bytes > avail * 0.80:
                    # 3x RAM too expensive — try sorted merge (1x RAM).
                    # Shards are already lexicographically sorted by
                    # _fast_dedup, so a two-pointer merge works.
                    out_max_bytes = a_size + b_size
                    if out_max_bytes <= avail * 0.85:
                        a_mm = np.load(current[i], mmap_mode='r')
                        b_mm = np.load(current[i + 1], mmap_mode='r')
                        out = np.empty((len(a_mm) + len(b_mm), d),
                                       dtype=np.int32)
                        n_out = _sorted_merge_dedup_kernel(a_mm, b_mm, out)
                        del a_mm, b_mm
                        out_path = current[i] + f'.m{merge_round}.npy'
                        np.save(out_path, out[:n_out])
                        if verbose:
                            _log(f"         Sorted merge ({i},{i+1}): "
                                 f"{n_out:,} unique rows "
                                 f"({n_out * d * 4 / 1e9:.2f} GB)")
                        del out
                        next_round.append(out_path)
                        for p in [current[i], current[i + 1]]:
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        continue

                    if verbose:
                        _log(f"       Memory limit: merge needs "
                             f"{need_bytes/1e9:.1f} GB, "
                             f"available {avail/1e9:.1f} GB")
                    hit_mem_limit = True
                    next_round.append(current[i])
                    next_round.append(current[i + 1])
                    continue

                a = np.load(current[i])
                b = np.load(current[i + 1])
                combined = np.vstack([a, b])
                del a, b
                merged = _fast_dedup(combined)
                del combined
                out_path = current[i] + f'.m{merge_round}.npy'
                np.save(out_path, merged)
                if verbose:
                    _log(f"         Pair ({i},{i+1}): "
                         f"{len(merged):,} unique rows "
                         f"({merged.nbytes/1e9:.2f} GB)")
                del merged
                next_round.append(out_path)
                for p in [current[i], current[i + 1]]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            else:
                next_round.append(current[i])

        # If no progress was made (all pairs hit mem limit), stop
        if len(next_round) >= len(current):
            if verbose:
                _log(f"       Cannot reduce further — "
                     f"{len(current)} shards remain on disk")
            break
        current = next_round

    if len(current) == 1:
        result = np.load(current[0])
        try:
            os.remove(current[0])
        except OSError:
            pass
        return result, []
    else:
        # Multiple shards remain — too large for RAM
        # Count total rows across shards
        total = 0
        for p in current:
            # Quick row count from file size: file has 128-byte npy header
            # then rows * d * 4 bytes
            sz = os.path.getsize(p) - 128
            total += sz // (d * 4)
        if verbose:
            _log(f"       {len(current)} unmerged shards, ~{total:,} rows total")
        return None, current


# =====================================================================
# Fused generate + prune kernel (highest-impact optimization)
# =====================================================================

@njit(cache=True)
def _fused_generate_and_prune(parent_int, n_half_child, m, c_target,
                               lo_arr, hi_arr, out_buf):
    """Generate children of one parent and immediately prune each one.

    Replaces the pipeline of generate_children_uniform() + test_children()
    by never materializing the full children array.  Each child is built
    on-the-fly via a stack-based Cartesian-product iterator, subjected to
    asymmetry + autoconvolution pruning, and only stored if it survives.

    Optimization: maintains the autoconvolution incrementally.  Consecutive
    children in the odometer differ in only 2 bins (~67% of steps), so we
    update raw_conv in O(d) instead of recomputing in O(d^2).

    Parameters
    ----------
    parent_int : (d_parent,) int32 array
    n_half_child : int  (= 2 * n_half_parent = d_parent in the cascade)
    m : int
    c_target : float
    lo_arr : (d_parent,) int32 — per-bin cursor lower bounds
    hi_arr : (d_parent,) int32 — per-bin cursor upper bounds
    out_buf : (max_survivors, d_child) int32 array (pre-allocated)

    Returns
    -------
    (n_survivors, n_subtree_pruned) : (int, int)
        n_survivors: number of rows written to out_buf
        n_subtree_pruned: number of subtrees skipped by partial-autoconv check
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    # --- Safety check: int32 conv values require m <= 200 ---
    # Max raw_conv entry: m^2 = 40000 for m=200; max mutual cross-term 2*m*m = 80000.
    # Incremental deltas bounded by ±2*m^2.  All well within int32 range (2^31-1).
    assert m <= 200, f"int32 conv requires m <= 200, got m={m}"

    # --- Asymmetry filter constants ---
    # No discretization margin needed: left_frac is exact for step functions
    # and preserved under refinement. See docs/verification_part1_framework.md §8.
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    # --- Hoisted asymmetry check (constant across all children) ---
    # sum(child[0:n_half_child]) = sum(parent_int[0:d_parent//2])
    # because child[2k]+child[2k+1] = parent_int[k] and n_half_child = d_parent
    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0

    # --- Dynamic pruning constants ---
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv = 0
    conv_len = 2 * d_child - 1
    carry_threshold = d_parent // 4

    # --- Prefix sum of parent bins (for W_int_max in subtree pruning) ---
    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    # --- Allocate arrays ---
    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    prev_child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    n_subtree_pruned = 0

    # Quick-check state: track the (ell, s_lo) that killed the previous child.
    # When qc_ell > 0, we try that same window first on the next child,
    # computing the window sum directly from raw_conv (O(ell) instead of O(conv_len)).
    qc_ell = np.int32(0)       # 0 = not yet tracking (first child)
    qc_s = np.int32(0)         # s_lo of tracked window
    qc_W_int = np.int64(0)     # W_int for tracked window on current child

    # --- Build initial child ---
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    # --- Precompute per-ell constants (constant across all children) ---
    ell_count = 2 * d_child - 1  # ell ranges 2..2*d_child, count = 2*d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx] = 2.0 * np.float64(ell) * inv_4n

    # --- Optimized ell scan order ---
    # Most children are pruned by narrow windows (ell=2..16) or wide windows
    # (ell near d_child). Scanning these first reduces the average number of
    # ell values checked before pruning.
    # Phase 1: ell=2..min(16, 2*d_child)  (narrow windows catch peaked configs)
    # Phase 2: ell=d_child, d_child+1, d_child-1, ...  (wide windows catch spread)
    # Phase 3: remaining values
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)  # boolean flags
    oi = 0
    # Phase 1: narrow (ell=2..16)
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell)
        ell_used[ell - 2] = np.int32(1)
        oi += 1
    # Phase 2: wide windows around d_child
    for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                d_child * 2, d_child + d_child // 2, d_child // 2):
        if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
    # Phase 3: everything else in order
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    # --- Compute full raw_conv for initial child ---
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        raw_conv[2 * i] += ci * ci
        for j in range(i + 1, d_child):
            raw_conv[i + j] += np.int32(2) * ci * np.int32(child[j])

    while True:
        # --- Quick check: re-try previous killing window on raw_conv ---
        quick_killed = False
        if qc_ell > 0:
            n_cv_qc = qc_ell - 1
            ws_qc = np.int64(0)
            for k in range(qc_s, qc_s + n_cv_qc):
                ws_qc += np.int64(raw_conv[k])
            ell_idx_qc = qc_ell - 2
            dyn_x_qc = dyn_base_ell_arr[ell_idx_qc] + two_ell_arr[ell_idx_qc] * np.float64(qc_W_int)
            dyn_it_qc = np.int64(dyn_x_qc * one_minus_4eps)
            if ws_qc > dyn_it_qc:
                quick_killed = True

        if not quick_killed:
            # --- Copy raw_conv to conv and prefix-sum ---
            for k in range(conv_len):
                conv[k] = raw_conv[k]
            for k in range(1, conv_len):
                conv[k] += conv[k - 1]

            # --- Compute prefix_c ---
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

            # --- Window scan (dynamic pruning, optimized ell order) ---
            pruned = False
            for ell_oi in range(ell_count):
                if pruned:
                    break
                ell = ell_order[ell_oi]
                n_cv = ell - 1
                ell_idx = ell - 2
                dyn_base_ell = dyn_base_ell_arr[ell_idx]
                two_ell_inv_4n = two_ell_arr[ell_idx]
                n_windows = conv_len - n_cv + 1
                for s_lo in range(n_windows):
                    s_hi = s_lo + n_cv - 1
                    # Widen to int64 for the comparison to avoid int32 overflow
                    # on the subtraction (conv values can be up to m^2 * d_child)
                    ws = np.int64(conv[s_hi])
                    if s_lo > 0:
                        ws -= np.int64(conv[s_lo - 1])
                    lo_bin = s_lo - (d_child - 1)
                    if lo_bin < 0:
                        lo_bin = 0
                    hi_bin = s_lo + ell - 2
                    if hi_bin > d_child - 1:
                        hi_bin = d_child - 1
                    W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                    dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int)
                    dyn_it = np.int64(dyn_x * one_minus_4eps)
                    if ws > dyn_it:
                        pruned = True
                        qc_ell = np.int32(ell)
                        qc_s = np.int32(s_lo)
                        break

            if not pruned:
                # --- Survivor! Canonicalize: min(child, rev(child)) lex ---
                use_rev = False
                for i in range(d_child):
                    j = d_child - 1 - i
                    if child[j] < child[i]:
                        use_rev = True
                        break
                    elif child[j] > child[i]:
                        break

                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # --- Advance cursor (odometer increment) ---
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1

        if carry < 0:
            break

        # --- Build new child for changed positions ---
        n_changed = d_parent - carry

        if n_changed == 1:
            # === FAST PATH: only last position changed (~67% of steps) ===
            pos = d_parent - 1
            k1 = 2 * pos
            k2 = k1 + 1
            old1 = np.int32(child[k1])
            old2 = np.int32(child[k2])
            child[k1] = cursor[pos]
            child[k2] = parent_int[pos] - cursor[pos]
            new1 = np.int32(child[k1])
            new2 = np.int32(child[k2])
            delta1 = new1 - old1
            delta2 = new2 - old2

            # Self-terms
            raw_conv[2 * k1] += new1 * new1 - old1 * old1
            raw_conv[2 * k2] += new2 * new2 - old2 * old2
            # Mutual term
            raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
            # Cross-terms with all unchanged bins (j < k1)
            for j in range(k1):
                cj = np.int32(child[j])
                raw_conv[k1 + j] += np.int32(2) * delta1 * cj
                raw_conv[k2 + j] += np.int32(2) * delta2 * cj

            # Quick-check: O(1) W_int update (only bins k1, k2 changed)
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                if qc_lo <= k1 and k1 <= qc_hi:
                    qc_W_int += np.int64(delta1)
                if qc_lo <= k2 and k2 <= qc_hi:
                    qc_W_int += np.int64(delta2)

        elif n_changed <= carry_threshold:
            # === SHORT CARRY: incremental update for 2..threshold positions ===
            first_changed_bin = 2 * carry

            # Save prev_child (only needed for incremental path)
            for i in range(d_child):
                prev_child[i] = child[i]

            # Rebuild changed child bins
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = parent_int[pos] - cursor[pos]

            # Self + mutual terms for each changed position pair
            for pos in range(carry, d_parent):
                k1 = 2 * pos
                k2 = k1 + 1
                old1 = np.int32(prev_child[k1])
                old2 = np.int32(prev_child[k2])
                new1 = np.int32(child[k1])
                new2 = np.int32(child[k2])
                raw_conv[2 * k1] += new1 * new1 - old1 * old1
                raw_conv[2 * k2] += new2 * new2 - old2 * old2
                raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)

            # Cross-terms between different changed position pairs
            for pa in range(carry, d_parent):
                a1 = 2 * pa
                a2 = a1 + 1
                new_a1 = np.int32(child[a1])
                new_a2 = np.int32(child[a2])
                old_a1 = np.int32(prev_child[a1])
                old_a2 = np.int32(prev_child[a2])
                for pb in range(pa + 1, d_parent):
                    b1 = 2 * pb
                    b2 = b1 + 1
                    new_b1 = np.int32(child[b1])
                    new_b2 = np.int32(child[b2])
                    old_b1 = np.int32(prev_child[b1])
                    old_b2 = np.int32(prev_child[b2])
                    raw_conv[a1 + b1] += np.int32(2) * (new_a1 * new_b1 - old_a1 * old_b1)
                    raw_conv[a1 + b2] += np.int32(2) * (new_a1 * new_b2 - old_a1 * old_b2)
                    raw_conv[a2 + b1] += np.int32(2) * (new_a2 * new_b1 - old_a2 * old_b1)
                    raw_conv[a2 + b2] += np.int32(2) * (new_a2 * new_b2 - old_a2 * old_b2)

            # Cross-terms between changed bins and unchanged bins
            for pos in range(carry, d_parent):
                k1 = 2 * pos
                k2 = k1 + 1
                delta1 = np.int32(child[k1]) - np.int32(prev_child[k1])
                delta2 = np.int32(child[k2]) - np.int32(prev_child[k2])
                for j in range(first_changed_bin):
                    cj = np.int32(child[j])
                    raw_conv[k1 + j] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + j] += np.int32(2) * delta2 * cj

            # Quick-check: recompute W_int (multiple bins changed)
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                qc_W_int = np.int64(0)
                for i in range(qc_lo, qc_hi + 1):
                    qc_W_int += np.int64(child[i])

        else:
            # === DEEP CARRY: attempt subtree prune before full recompute ===
            fixed_len = 2 * carry          # number of fixed child bins

            if fixed_len >= 4:  # need at least 4 bins for a meaningful check
                # Compute partial autoconvolution (fixed bins only)
                partial_conv_len = 2 * fixed_len - 1
                for k in range(partial_conv_len):
                    conv[k] = np.int32(0)
                for i in range(fixed_len):
                    ci = np.int32(child[i])
                    conv[2 * i] += ci * ci
                    for j in range(i + 1, fixed_len):
                        conv[i + j] += np.int32(2) * ci * np.int32(child[j])
                # Prefix sum
                for k in range(1, partial_conv_len):
                    conv[k] += conv[k - 1]

                # Compute fixed-region prefix_c for W_int
                prefix_c[0] = 0
                for i in range(fixed_len):
                    prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

                # Window scan with W_int_max thresholds
                subtree_pruned = False
                first_unfixed_parent = carry

                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    dyn_base_ell = dyn_base_ell_arr[ell_idx]
                    two_ell_inv_4n = two_ell_arr[ell_idx]

                    # Only check windows fully contained in partial conv
                    n_windows_partial = partial_conv_len - n_cv + 1
                    if n_windows_partial <= 0:
                        continue

                    for s_lo in range(n_windows_partial):
                        s_hi = s_lo + n_cv - 1
                        ws = np.int64(conv[s_hi])
                        if s_lo > 0:
                            ws -= np.int64(conv[s_lo - 1])

                        # W_int_max: fixed child bins + unfixed parent bins
                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

                        # Fixed part
                        fixed_hi = hi_bin
                        if fixed_hi > fixed_len - 1:
                            fixed_hi = fixed_len - 1
                        if fixed_hi >= lo_bin:
                            lo_clamp = lo_bin
                            if lo_clamp < 0:
                                lo_clamp = 0
                            W_int_fixed = prefix_c[fixed_hi + 1] - prefix_c[lo_clamp]
                        else:
                            W_int_fixed = np.int64(0)

                        # Unfixed part
                        unfixed_lo_bin = lo_bin
                        if unfixed_lo_bin < fixed_len:
                            unfixed_lo_bin = fixed_len
                        if unfixed_lo_bin <= hi_bin:
                            p_lo = unfixed_lo_bin // 2
                            p_hi = hi_bin // 2
                            if p_lo < first_unfixed_parent:
                                p_lo = first_unfixed_parent
                            if p_hi >= d_parent:
                                p_hi = d_parent - 1
                            if p_lo <= p_hi:
                                W_int_unfixed = parent_prefix[p_hi + 1] - parent_prefix[p_lo]
                            else:
                                W_int_unfixed = np.int64(0)
                        else:
                            W_int_unfixed = np.int64(0)

                        W_int_max = W_int_fixed + W_int_unfixed
                        dyn_x = dyn_base_ell + two_ell_inv_4n * np.float64(W_int_max)
                        dyn_it = np.int64(dyn_x * one_minus_4eps)
                        if ws > dyn_it:
                            subtree_pruned = True
                            break

                if subtree_pruned:
                    n_subtree_pruned += 1
                    # Skip entire subtree: fast-forward trailing cursors
                    for i in range(carry + 1, d_parent):
                        cursor[i] = hi_arr[i]
                    # Rebuild child for current cursor
                    for pos in range(carry, d_parent):
                        child[2 * pos] = cursor[pos]
                        child[2 * pos + 1] = parent_int[pos] - cursor[pos]
                    # Full recompute of raw_conv
                    for k in range(conv_len):
                        raw_conv[k] = np.int32(0)
                    for i in range(d_child):
                        ci = np.int32(child[i])
                        raw_conv[2 * i] += ci * ci
                        for j in range(i + 1, d_child):
                            raw_conv[i + j] += np.int32(2) * ci * np.int32(child[j])
                    # Quick-check: recompute W_int after subtree recompute
                    if qc_ell > 0:
                        qc_lo = qc_s - (d_child - 1)
                        if qc_lo < 0:
                            qc_lo = 0
                        qc_hi = qc_s + qc_ell - 2
                        if qc_hi > d_child - 1:
                            qc_hi = d_child - 1
                        qc_W_int = np.int64(0)
                        for i in range(qc_lo, qc_hi + 1):
                            qc_W_int += np.int64(child[i])
                    continue

            # === Not subtree-pruned: original full recompute path ===
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = parent_int[pos] - cursor[pos]

            for k in range(conv_len):
                raw_conv[k] = np.int32(0)
            for i in range(d_child):
                ci = np.int32(child[i])
                raw_conv[2 * i] += ci * ci
                for j in range(i + 1, d_child):
                    raw_conv[i + j] += np.int32(2) * ci * np.int32(child[j])

            # Quick-check: recompute W_int after full recompute
            if qc_ell > 0:
                qc_lo = qc_s - (d_child - 1)
                if qc_lo < 0:
                    qc_lo = 0
                qc_hi = qc_s + qc_ell - 2
                if qc_hi > d_child - 1:
                    qc_hi = d_child - 1
                qc_W_int = np.int64(0)
                for i in range(qc_lo, qc_hi + 1):
                    qc_W_int += np.int64(child[i])

    return n_surv, n_subtree_pruned


def _compute_bin_ranges(parent_int, m, c_target, d_child):
    """Compute per-bin lo/hi cursor ranges and total children count.

    Returns (lo_arr, hi_arr, total_children) or None if any bin has empty range.
    """
    d_parent = len(parent_int)
    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    # Cauchy-Schwarz bound on continuous ||f*f||_∞ ≥ d_child·c_i²/m²
    # doesn't go through test-value, so no correction needed
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
    x_cap = min(x_cap, x_cap_cs)
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)

    lo_arr = np.empty(d_parent, dtype=np.int32)
    hi_arr = np.empty(d_parent, dtype=np.int32)
    total_children = 1
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            return None
        lo_arr[i] = lo
        hi_arr[i] = hi
        total_children *= (hi - lo + 1)

    return lo_arr, hi_arr, total_children


def _default_buf_cap(d_child):
    """Default survivor buffer capacity, scaled by dimension."""
    if d_child <= 16:
        return 10_000_000
    elif d_child <= 32:
        return 5_000_000
    else:
        return 100_000      # 100K rows ~25 MB at d=64; survival rate ≈ 0 at L4+


def process_parent_fused(parent_int, m, c_target, n_half_child, buf_cap=None):
    """Wrapper: compute x_cap, allocate buffer, call fused kernel.

    Parameters
    ----------
    buf_cap : int or None
        Max rows for the output buffer.  *None* → ``_default_buf_cap(d_child)``.
        If the kernel reports more survivors than fit, the buffer is
        re-allocated at the exact size and the kernel re-run.

    Returns
    -------
    survivors : (K, d_child) int32 array
    total_children : int  (total Cartesian product size, for stats)
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    result = _compute_bin_ranges(parent_int, m, c_target, d_child)
    if result is None:
        return np.empty((0, d_child), dtype=np.int32), 0
    lo_arr, hi_arr, total_children = result

    if total_children == 0:
        return np.empty((0, d_child), dtype=np.int32), 0

    if buf_cap is None:
        buf_cap = _default_buf_cap(d_child)
    max_buf = min(total_children, buf_cap)
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)

    n_survivors, _ = _fused_generate_and_prune(
        parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf)

    if n_survivors > max_buf:
        # Overflow: re-allocate exact-size buffer and re-run.
        max_buf = n_survivors
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)
        n2, _ = _fused_generate_and_prune(
            parent_int, n_half_child, m, c_target, lo_arr, hi_arr, out_buf)
        assert n2 == n_survivors, (
            f"Non-deterministic kernel: first run {n_survivors}, "
            f"retry {n2}")
        n_survivors = n2

    return out_buf[:n_survivors].copy(), total_children


def process_parent_verbose(parent_int, m, c_target, n_half_child,
                            parent_idx, n_parents):
    """Like process_parent_fused but logs intra-parent progress.

    Splits the Cartesian product along cursor[0]'s range so we can
    log between slices.  Falls back to single-shot for small parents.

    Returns
    -------
    survivors : (K, d_child) int32 array
    total_children : int
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    label = f"parent {parent_idx+1}/{n_parents}"

    result = _compute_bin_ranges(parent_int, m, c_target, d_child)
    if result is None:
        _log(f"       {label}: empty range, skipped")
        return np.empty((0, d_child), dtype=np.int32), 0
    lo_arr, hi_arr, total_children = result

    if total_children == 0:
        _log(f"       {label}: 0 children")
        return np.empty((0, d_child), dtype=np.int32), 0

    n_slices = int(hi_arr[0]) - int(lo_arr[0]) + 1

    # Small parent or only 1 slice → single-shot
    if total_children < 500_000 or n_slices <= 1:
        _log(f"       {label}: {total_children:,} children (single pass)...")
        surv, tc = process_parent_fused(parent_int, m, c_target, n_half_child)
        _log(f"       {label}: done, {len(surv):,} survivors")
        return surv, tc

    # Large parent → split by cursor[0] value for progress
    _log(f"       {label}: {total_children:,} children, "
         f"{n_slices} slices on bin[0]")

    children_per_slice = total_children // n_slices
    all_survivors = []
    total_survived = 0
    t_start = time.time()

    slice_lo = lo_arr.copy()
    slice_hi = hi_arr.copy()

    for si, v0 in enumerate(range(int(lo_arr[0]), int(hi_arr[0]) + 1)):
        slice_lo[0] = np.int32(v0)
        slice_hi[0] = np.int32(v0)

        slice_buf_cap = _default_buf_cap(d_child)
        max_buf = min(children_per_slice, slice_buf_cap)
        out_buf = np.empty((max_buf, d_child), dtype=np.int32)

        n_surv, _ = _fused_generate_and_prune(
            parent_int, n_half_child, m, c_target,
            slice_lo, slice_hi, out_buf)

        if n_surv > max_buf:
            # Overflow: re-allocate and re-run
            max_buf = n_surv
            out_buf = np.empty((max_buf, d_child), dtype=np.int32)
            n2, _ = _fused_generate_and_prune(
                parent_int, n_half_child, m, c_target,
                slice_lo, slice_hi, out_buf)
            assert n2 == n_surv, (
                f"Non-deterministic kernel: first run {n_surv}, retry {n2}")
            n_surv = n2
        if n_surv > 0:
            all_survivors.append(out_buf[:n_surv].copy())
            total_survived += n_surv

        # Log every slice, or at least every 5 seconds
        done_slices = si + 1
        elapsed = time.time() - t_start
        if done_slices == n_slices or done_slices % max(1, n_slices // 20) == 0:
            rate = done_slices / elapsed if elapsed > 0 else 0
            eta = (n_slices - done_slices) / rate if rate > 0 else 0
            pct = done_slices / n_slices * 100
            _log(f"       {label}: slice {done_slices}/{n_slices} "
                 f"({pct:.0f}%) {total_survived:,} surv, "
                 f"ETA {_fmt_time(eta)}")

    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    elapsed = time.time() - t_start
    _log(f"       {label}: done in {_fmt_time(elapsed)}, "
         f"{len(survivors):,} survivors")
    return survivors, total_children


# =====================================================================
# JIT warmup
# =====================================================================

def _warmup_jit():
    """Warm up Numba JIT for common array dimensions at import time."""
    for d in (4, 8):
        dummy = np.zeros((1, d), dtype=np.int32)
        _prune_dynamic_int32(dummy, d // 2, 20, 1.3)
        _prune_dynamic_int64(dummy, d // 2, 300, 1.3)
        _canonical_mask(dummy)
        _canonicalize_inplace(dummy.copy())
    # Warm up sorted merge kernel
    _dm = np.zeros((1, 4), dtype=np.int32)
    _sorted_merge_dedup_kernel(_dm, _dm, np.zeros((2, 4), dtype=np.int32))

_warmup_jit()


# =====================================================================
# Level 0: generate all compositions, prune, collect survivors
# =====================================================================

def run_level0(n_half, m, c_target, verbose=True):
    """Run Level 0: enumerate compositions, prune, collect survivors.

    Matches GPU semantics exactly:
      - Canonical palindrome filter (only c <= rev(c))
      - Dynamic per-window threshold (integer-space, W_int-dependent)

    Returns
    -------
    dict with: survivors (N, d) int32, n_survivors, n_pruned_asym,
               n_pruned_test, elapsed, proven
    """
    d = 2 * n_half
    S = m
    n_total = count_compositions(d, S)
    corr = correction(m)

    if verbose:
        _log(f"\n[L0] d={d}, m={m}, compositions={n_total:,}")
        _log(f"     correction={corr:.6f}, threshold={c_target+corr:.6f}")

    t0 = time.time()
    all_survivors = []
    n_pruned_asym = 0
    n_pruned_test = 0
    n_processed = 0
    n_non_canonical = 0
    n_batches = 0
    last_report = t0

    for batch in generate_compositions_batched(d, S, batch_size=200_000):
        n_processed += len(batch)
        n_batches += 1

        # Canonical filter (match GPU: only c <= rev(c))
        canon = _canonical_mask(batch)
        n_non_canonical += int(np.sum(~canon))
        batch = batch[canon]
        if len(batch) == 0:
            continue

        # Asymmetry filter
        needs_check = asymmetry_prune_mask(batch, n_half, m, c_target)
        n_asym_batch = int(np.sum(~needs_check))
        n_pruned_asym += n_asym_batch

        candidates = batch[needs_check]
        if len(candidates) == 0:
            continue

        # Dynamic per-window threshold (matches GPU exactly)
        survived_mask = _prune_dynamic(candidates, n_half, m, c_target)
        n_pruned_test += int(np.sum(~survived_mask))

        survivors = candidates[survived_mask]
        if len(survivors) > 0:
            all_survivors.append(survivors)

        # Progress: report every 2 seconds or every batch if slow
        now = time.time()
        if verbose and (now - last_report >= 2.0):
            pct = n_processed / n_total * 100 if n_total > 0 else 0
            n_surv_so_far = sum(len(s) for s in all_survivors)
            elapsed_so_far = now - t0
            rate = n_processed / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (n_total - n_processed) / rate if rate > 0 else 0
            _log(f"     [L0] {n_processed:,}/{n_total:,} ({pct:.1f}%) "
                 f"| {n_surv_so_far:,} survivors | "
                 f"ETA {_fmt_time(eta)}")
            last_report = now

    elapsed = time.time() - t0

    if all_survivors:
        all_survivors = np.vstack(all_survivors)
    else:
        all_survivors = np.empty((0, d), dtype=np.int32)

    n_survivors = len(all_survivors)
    proven = n_survivors == 0

    if verbose:
        _log(f"     {elapsed:.2f}s: {n_processed:,} compositions processed")
        _log(f"     asym pruned: {n_pruned_asym:,}, "
             f"test pruned: {n_pruned_test:,}, "
             f"survivors: {n_survivors:,}")
        if proven:
            _log(f"     PROVEN at L0!")

    return {
        'survivors': all_survivors,
        'n_survivors': n_survivors,
        'n_pruned_asym': n_pruned_asym,
        'n_pruned_test': n_pruned_test,
        'n_processed': n_processed,
        'elapsed': elapsed,
        'proven': proven,
    }


# =====================================================================
# Refinement: uniform full-bin split (legacy — kept as fallback)
# =====================================================================

def generate_children_uniform(parent_int, m, c_target):
    """Generate all child compositions from a parent via uniform 2-split.

    Each parent bin c_i is split into (a, b) where a + b = c_i.
    Both sub-bins are capped at x_cap (energy bound) for efficiency.

    Parameters
    ----------
    parent_int : (d_parent,) int array
    m : int
    c_target : float

    Returns
    -------
    (N_children, d_child) int32 array where d_child = 2 * d_parent
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent

    # x_cap: single-bin energy cap
    corr = 2.0 / m + 1.0 / (m * m)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
    # Cauchy-Schwarz bound on continuous ||f*f||_∞ ≥ d_child·c_i²/m²
    x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
    x_cap = min(x_cap, x_cap_cs)
    x_cap = min(x_cap, m)
    x_cap = max(x_cap, 0)

    # Build per-bin split options
    per_bin_choices = []
    for i in range(d_parent):
        b_i = int(parent_int[i])
        lo = max(0, b_i - x_cap)
        hi = min(b_i, x_cap)
        if lo > hi:
            # This parent can't produce valid children
            return np.empty((0, d_child), dtype=np.int32)
        per_bin_choices.append(list(range(lo, hi + 1)))

    # Total children = product of choice counts
    total = 1
    for choices in per_bin_choices:
        total *= len(choices)

    if total == 0:
        return np.empty((0, d_child), dtype=np.int32)

    # For very large expansions, use chunked generation to avoid OOM
    if total > 50_000_000:
        return _generate_children_chunked(parent_int, per_bin_choices,
                                          d_parent, d_child, total)

    children = np.empty((total, d_child), dtype=np.int32)
    idx = 0
    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            children[idx, 2 * i] = combo[i]
            children[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1

    return children


def _generate_children_chunked(parent_int, per_bin_choices, d_parent,
                                d_child, total):
    """Generate children in chunks to avoid memory blowup."""
    chunk_size = 10_000_000
    chunks = []
    buf = np.empty((chunk_size, d_child), dtype=np.int32)
    idx = 0

    for combo in itertools.product(*per_bin_choices):
        for i in range(d_parent):
            buf[idx, 2 * i] = combo[i]
            buf[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1
        if idx == chunk_size:
            chunks.append(buf[:idx].copy())
            idx = 0

    if idx > 0:
        chunks.append(buf[:idx].copy())

    return np.vstack(chunks) if chunks else np.empty((0, d_child), dtype=np.int32)


def test_children(children_int, n_half_child, m, c_target):
    """Prune children via asymmetry + dynamic threshold.

    NO canonical filter at refinement levels — applying it here would
    silently drop canonical children whose parent is non-canonical
    (rev(P) for canonical P), since rev(P) is never in our parent list.
    Instead, survivors are canonicalized and deduped after testing.

    Returns
    -------
    (survivors, stats) where survivors is (K, d_child) int32
    """
    if len(children_int) == 0:
        d_child = children_int.shape[1] if children_int.ndim == 2 else 2
        return np.empty((0, d_child), dtype=np.int32), {
            'n_tested': 0, 'n_canonical': 0, 'n_asym': 0, 'n_test': 0,
            'n_survived': 0
        }

    N, d_child = children_int.shape

    # Asymmetry filter
    needs_check = asymmetry_prune_mask(children_int, n_half_child, m, c_target)
    n_asym = int(np.sum(~needs_check))
    candidates = np.ascontiguousarray(children_int[needs_check])

    if len(candidates) > 0:
        # Dynamic per-window threshold
        survived_mask = _prune_dynamic(candidates, n_half_child, m, c_target)
        survivors = np.ascontiguousarray(candidates[survived_mask])
        n_test = int(np.sum(~survived_mask))
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)
        n_test = 0

    # Canonicalize survivors using Numba parallel kernel
    if len(survivors) > 0:
        _canonicalize_inplace(survivors)

    return survivors, {
        'n_tested': N,
        'n_canonical': 0,
        'n_asym': n_asym,
        'n_test': n_test,
        'n_survived': len(survivors),
    }


# =====================================================================
# Multiprocessing helpers
# =====================================================================

def _init_worker_threads(n_threads):
    """Limit Numba parallelism in each worker to avoid oversubscription."""
    numba.set_num_threads(n_threads)


def _process_single_parent_fused(args):
    """Worker: generate + prune children for one parent using fused kernel.

    Avoids materializing the full children array — generates each child
    on-the-fly and prunes inline.

    Accepts 5-element tuple (parent, m, c_target, n_half_child, batch_size)
    or 6-element tuple with optional buf_cap as last element.
    """
    if len(args) >= 6:
        parent, m, c_target, n_half_child, batch_size, buf_cap = args[:6]
    else:
        parent, m, c_target, n_half_child, batch_size = args
        buf_cap = None

    survivors, total_children = process_parent_fused(
        parent, m, c_target, n_half_child, buf_cap=buf_cap)

    n_survived = len(survivors)

    if n_survived > 0:
        result = survivors
    else:
        result = None

    return result, {
        'children': total_children,
        'asym': 0,
        'test': 0,
        'survived': n_survived,
    }


def _process_single_parent_legacy(args):
    """Legacy worker: generate children for one parent, prune, return survivors.

    Kept as fallback — the fused version is the default.
    """
    parent, m, c_target, n_half_child, batch_size = args

    children = generate_children_uniform(parent, m, c_target)
    n_children = len(children)

    if n_children == 0:
        return None, {'children': 0, 'asym': 0, 'test': 0, 'survived': 0}

    parent_survivors = []
    total_asym = 0
    total_test = 0
    total_survived = 0

    for start in range(0, n_children, batch_size):
        end = min(start + batch_size, n_children)
        batch = children[start:end]

        survivors, stats = test_children(batch, n_half_child, m, c_target)
        total_asym += stats['n_asym']
        total_test += stats['n_test']
        total_survived += stats['n_survived']

        if len(survivors) > 0:
            parent_survivors.append(survivors)

    if parent_survivors:
        result = np.vstack(parent_survivors)
    else:
        result = None

    return result, {
        'children': n_children,
        'asym': total_asym,
        'test': total_test,
        'survived': total_survived,
    }


def _process_single_parent(args):
    """Worker: generate + prune children for one parent.

    Uses the fused kernel by default.
    """
    return _process_single_parent_fused(args)


# =====================================================================
# Shared-memory multiprocessing helpers
# =====================================================================

def _init_worker_shm(mmap_path, shape, dtype_str, m, c_target, n_half_child,
                     numba_threads):
    """Pool initializer: open mmap of parent array and store params in globals."""
    numba.set_num_threads(numba_threads)
    global _shared_parents, _shm_m, _shm_c_target, _shm_n_half_child
    _shared_parents = np.memmap(mmap_path, dtype=np.dtype(dtype_str),
                                mode='r', shape=shape)
    _shm_m = m
    _shm_c_target = c_target
    _shm_n_half_child = n_half_child


def _process_parent_shm(idx):
    """Worker: process parent at index idx from shared memory array."""
    parent = _shared_parents[idx].copy()  # local copy from shared mem
    survivors, total_children = process_parent_fused(
        parent, _shm_m, _shm_c_target, _shm_n_half_child)
    n_survived = len(survivors)
    result = survivors if n_survived > 0 else None
    return result, {
        'children': total_children,
        'asym': 0,
        'test': 0,
        'survived': n_survived,
    }


# =====================================================================
# Checkpoint helpers
# =====================================================================

def _save_checkpoint(output_dir, level, survivors, meta):
    """Save survivors array and metadata after a completed level."""
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f'checkpoint_L{level}_survivors.npy')
    meta_path = os.path.join(output_dir, 'checkpoint_meta.json')

    np.save(npy_path, survivors)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=_convert)

    _log(f"     Checkpoint saved: {npy_path} "
         f"({survivors.nbytes / 1e9:.2f} GB, {len(survivors):,} rows)")


def _load_checkpoint(resume_dir, n_half, m, c_target):
    """Load checkpoint if it exists and parameters match.

    Returns (survivors, level_num, info) or None.
    """
    meta_path = os.path.join(resume_dir, 'checkpoint_meta.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Validate parameters match
    if (meta['n_half'] != n_half or meta['m'] != m
            or meta['c_target'] != c_target):
        _log(f"     Checkpoint found but parameters don't match:")
        _log(f"       checkpoint: n_half={meta['n_half']}, m={meta['m']}, "
             f"c_target={meta['c_target']}")
        _log(f"       requested:  n_half={n_half}, m={m}, "
             f"c_target={c_target}")
        return None

    level = meta['level_completed']
    npy_path = os.path.join(resume_dir,
                            f'checkpoint_L{level}_survivors.npy')
    if not os.path.exists(npy_path):
        _log(f"     Checkpoint meta found but {npy_path} missing")
        return None

    survivors = np.load(npy_path, mmap_mode='r')
    _log(f"     Loaded checkpoint: L{level} complete, "
         f"{len(survivors):,} survivors (d={survivors.shape[1]})")

    info = meta.get('info', {})
    return survivors, level, info


# =====================================================================
# CPU detection (cgroup-aware for containers)
# =====================================================================

def _effective_cpu_count():
    """Detect actual usable CPUs, accounting for cgroup limits in containers.

    On bare metal, returns mp.cpu_count().  In Docker/RunPod containers,
    mp.cpu_count() returns the host CPU count (e.g. 192) even though the
    container may be cgroup-limited to 32 vCPUs.  This reads the cgroup
    quota to return the correct value.
    """
    logical = mp.cpu_count()
    # Try cgroup v1
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:
            quota = int(f.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read().strip())
        if quota > 0:
            cgroup_cpus = max(1, int(quota / period))
            return min(logical, cgroup_cpus)
    except (FileNotFoundError, ValueError, OSError):
        pass
    # Try cgroup v2
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            parts = f.read().strip().split()
            if parts[0] != 'max':
                cgroup_cpus = max(1, int(int(parts[0]) / int(parts[1])))
                return min(logical, cgroup_cpus)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return logical


# =====================================================================
# Cascade runner
# =====================================================================

def run_cascade(n_half, m, c_target, max_levels=10, n_workers=None,
                verbose=True, output_dir='data', resume_dir=None):
    """Run the full CPU cascade: L0 + refinement levels.

    Parameters
    ----------
    n_half : int
        Initial n_half (d0 = 2 * n_half).
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.
    max_levels : int
        Max refinement levels after L0.
    n_workers : int or None
        Number of parallel workers for refinement levels.
        None = auto-detect CPU count.
    verbose : bool
        Print progress.
    output_dir : str
        Directory for checkpoints and results.
    resume_dir : str or None
        Directory to look for checkpoint files.  If None, uses output_dir.

    Returns
    -------
    dict with cascade results.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = max(1, n_workers)

    # Ensure enough file descriptors for spawn-based multiprocessing
    # (each worker needs ~6 fds for pipes/semaphores)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        needed = n_workers * 8 + 256  # generous headroom
        if soft < needed:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (min(needed, hard), hard))
    except (ImportError, ValueError, OSError):
        pass  # Windows or insufficient permissions — proceed anyway
    d0 = 2 * n_half
    corr = correction(m)
    n_total = count_compositions(d0, m)

    if verbose:
        _log(f"\n{'='*70}")
        _log(f"CPU CASCADE PROVER")
        _log(f"  n_half={n_half}, m={m}, d0={d0}, c_target={c_target}")
        _log(f"  correction={corr:.6f}, effective threshold={c_target+corr:.6f}")
        _log(f"  L0 compositions: {n_total:,}")
        _log(f"  workers: {n_workers} (logical CPUs: {mp.cpu_count()}, "
             f"effective: {_effective_cpu_count()})")
        _log(f"  max refinement levels: {max_levels}")
        _log(f"{'='*70}")

    if resume_dir is None:
        resume_dir = output_dir

    t_total = time.time()

    # --- Try to resume from checkpoint ---
    resume_result = _load_checkpoint(resume_dir, n_half, m, c_target)
    start_level = 0  # 0 means run L0 fresh

    if resume_result is not None:
        current_configs, last_completed, saved_info = resume_result
        start_level = last_completed + 1
        d_parent = current_configs.shape[1]
        # n_half doubles each level: at level L, n_half_parent = n_half * 2^L
        n_half_parent = n_half * (2 ** last_completed)

        info = saved_info if isinstance(saved_info, dict) else {}
        # Ensure required keys exist
        info.setdefault('n_half', n_half)
        info.setdefault('m', m)
        info.setdefault('d0', d0)
        info.setdefault('c_target', c_target)
        info.setdefault('correction', corr)
        info.setdefault('levels', [])

        if verbose:
            _log(f"\n  RESUMING from L{last_completed} checkpoint")
            _log(f"  {len(current_configs):,} survivors at d={d_parent}")
            _log(f"  Skipping L0 through L{last_completed}")

    if start_level == 0:
        # --- Level 0 (fresh run) ---
        l0 = run_level0(n_half, m, c_target, verbose=verbose)

        info = {
            'n_half': n_half, 'm': m, 'd0': d0, 'c_target': c_target,
            'correction': corr,
            'l0_time': l0['elapsed'],
            'l0_survivors': l0['n_survivors'],
            'l0_pruned_asym': l0['n_pruned_asym'],
            'l0_pruned_test': l0['n_pruned_test'],
            'levels': [],
        }

        if l0['proven']:
            info['proven_at'] = 'L0'
            info['total_time'] = time.time() - t_total
            return info

        current_configs = l0['survivors']
        d_parent = d0
        n_half_parent = n_half

        # Checkpoint L0 survivors
        _save_checkpoint(output_dir, 0, current_configs, {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'level_completed': 0,
            'd_survivors': d_parent,
            'n_survived': len(current_configs),
            'info': info,
        })

    # --- Refinement levels ---
    for level_num in range(max(1, start_level), max_levels + 1):
        d_child = 2 * d_parent
        n_half_child = 2 * n_half_parent
        n_parents = len(current_configs)

        if n_parents == 0:
            break

        # --- Pre-filter: skip parents where any bin > 2*x_cap ---
        # Such parents produce zero children (empty cursor range in every
        # bin exceeding x_cap), but still incur IPC + dispatch overhead.
        corr_pf = 2.0 / m + 1.0 / (m * m)
        thresh_pf = c_target + corr_pf + 1e-9
        x_cap_pf = int(math.floor(m * math.sqrt(thresh_pf / d_child)))
        # Cauchy-Schwarz bound: no correction needed
        x_cap_cs_pf = int(math.floor(m * math.sqrt(c_target / d_child)))
        x_cap_pf = min(x_cap_pf, x_cap_cs_pf, m)
        max_bin_val = 2 * x_cap_pf
        feasible_mask = np.all(current_configs <= max_bin_val, axis=1)
        n_infeasible = n_parents - int(np.sum(feasible_mask))
        if n_infeasible > 0:
            current_configs = np.ascontiguousarray(current_configs[feasible_mask])
            n_parents = len(current_configs)
            if verbose:
                _log(f"     Pre-filtered {n_infeasible:,} infeasible parents "
                     f"(bin > {max_bin_val})")

        if n_parents == 0:
            break

        # Shuffle parents for unbiased ETA estimation and better load balance.
        # Lex order from dedup correlates with per-parent cost (bin values
        # determine child count, varying ~1000x).  Fixed seed for reproducibility.
        # Ensure writable (checkpoint loads as read-only mmap).
        if not current_configs.flags.writeable:
            current_configs = np.array(current_configs)
        rng = np.random.RandomState(42)
        rng.shuffle(current_configs)

        if verbose:
            _log(f"\n[L{level_num}] d_parent={d_parent} -> d_child={d_child}, "
                 f"{n_parents:,} parents")

        t_level = time.time()
        total_children = 0
        total_survived = 0
        report_interval = _progress_interval(n_parents)

        # Memory-safe survivor collection: accumulate in RAM up to a
        # budget, then spill to disk shards.  Final dedup merges shards.
        bytes_per_row = d_child * 4
        try:
            import psutil
            avail_bytes = psutil.virtual_memory().available
        except ImportError:
            avail_bytes = int(64e9 * 0.80)
        # Reserve memory for shared array + workers + OS.
        # _fast_dedup needs ~3x the array size (input + sort index + output),
        # so the safe in-RAM batch is 1/4 of available budget.
        shm_bytes = 0  # mmap: parent array lives in OS page cache, not process RSS
        survivor_mem_budget = max(int(1e9),
            (avail_bytes - shm_bytes - int(10e9)) // 4)
        shard_threshold = max(100_000, survivor_mem_budget // bytes_per_row)
        if verbose:
            _log(f"     Survivor spool: {survivor_mem_budget/1e9:.1f} GB "
                 f"in-RAM ({shard_threshold:,} rows), then disk shards")

        all_survivors = []
        all_survivors_rows = 0
        shard_dir = os.path.join(output_dir, f'_shards_L{level_num}')
        shard_paths = []
        n_shards = 0

        def _flush_to_shard():
            nonlocal all_survivors, all_survivors_rows, n_shards
            if not all_survivors:
                return
            batch = np.vstack(all_survivors)
            batch = _fast_dedup(batch)
            os.makedirs(shard_dir, exist_ok=True)
            path = os.path.join(shard_dir, f'shard_{n_shards:04d}.npy')
            np.save(path, batch)
            shard_paths.append(path)
            n_shards += 1
            if verbose:
                _log(f"     Flushed shard {n_shards}: {len(batch):,} unique rows "
                     f"({batch.nbytes/1e9:.2f} GB)")
            all_survivors = []
            all_survivors_rows = 0

        if n_workers > 1 and n_parents > n_workers:
            # --- Parallel path: shared memory + index dispatch ---

            # Memory-aware worker cap.
            # Workers and dedup don't peak concurrently: dedup runs in the
            # main process between imap_unordered batches, and the shard
            # flush frees the accumulator before the next fill cycle.
            # Reserve: 1x survivor spool (pre-flush peak) + 4 GB OS/main.
            _buf_cap = _default_buf_cap(d_child)
            per_worker_bytes = _buf_cap * d_child * 4 + 150 * 1024 * 1024
            reserved = shm_bytes + survivor_mem_budget + int(4e9)
            worker_mem_budget = max(int(1e9), avail_bytes - reserved)
            max_by_mem = max(1, int(worker_mem_budget / per_worker_bytes))
            if n_workers > max_by_mem:
                if verbose:
                    _log(f"     Memory cap: {n_workers} -> {max_by_mem} workers "
                         f"(avail={avail_bytes/1e9:.1f}GB, "
                         f"per_worker={per_worker_bytes/1e9:.2f}GB, "
                         f"shm={shm_bytes/1e9:.2f}GB)")
                n_workers_level = max_by_mem
            else:
                n_workers_level = n_workers

            numba_threads = min(max(1, _effective_cpu_count() // n_workers_level),
                                numba.config.NUMBA_NUM_THREADS)
            chunksize = max(1, min(n_parents // (n_workers_level * 20), 128))

            # Write parent array to a temp file; workers mmap it read-only
            parents_shape = current_configs.shape
            parents_dtype_str = current_configs.dtype.str
            parents_nbytes = current_configs.nbytes
            fd, mmap_path = tempfile.mkstemp(
                suffix=f'_L{level_num}_parents.dat', dir=output_dir)
            os.close(fd)
            current_configs.tofile(mmap_path)
            del current_configs  # free RAM; workers mmap from disk

            if verbose:
                mmap_gb = parents_nbytes / 1e9
                _log(f"     (parallel: {n_workers_level} workers, "
                     f"chunksize={chunksize}, "
                     f"numba_threads={numba_threads}, "
                     f"mmap={mmap_gb:.2f} GB)")

            completed = 0
            last_report = time.time()
            last_checkpoint_time = time.time()
            checkpoint_interval = 1800  # 30 minutes
            ctx = mp.get_context("spawn")
            try:
                with ctx.Pool(
                        n_workers_level,
                        initializer=_init_worker_shm,
                        initargs=(mmap_path, parents_shape,
                                  parents_dtype_str,
                                  m, c_target, n_half_child,
                                  numba_threads)) as pool:
                    for surv, stats in pool.imap_unordered(
                            _process_parent_shm, range(n_parents),
                            chunksize=chunksize):
                        total_children += stats['children']
                        total_survived += stats['survived']
                        completed += 1

                        if surv is not None:
                            all_survivors.append(surv)
                            all_survivors_rows += len(surv)

                        # Flush to disk when in-RAM budget exceeded
                        if all_survivors_rows >= shard_threshold:
                            _flush_to_shard()

                        now = time.time()
                        if verbose and (completed % report_interval == 0
                                        or now - last_report >= 5.0):
                            elapsed_so_far = now - t_level
                            rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                            eta = (n_parents - completed) / rate if rate > 0 else 0
                            pct = completed / n_parents * 100
                            _log(f"     [{completed}/{n_parents}] ({pct:.1f}%) "
                                 f"{total_survived:,} survivors so far, "
                                 f"shards={n_shards}, "
                                 f"ETA {_fmt_time(eta)}")
                            last_report = now

                        # Intra-level checkpoint: save progress every 30 min
                        if now - last_checkpoint_time >= checkpoint_interval:
                            progress_path = os.path.join(
                                output_dir,
                                f'_progress_L{level_num}.json')
                            progress = {
                                'level': level_num,
                                'completed': completed,
                                'total_parents': n_parents,
                                'total_children': total_children,
                                'total_survived': total_survived,
                                'elapsed_seconds': now - t_level,
                                'timestamp': time.strftime(
                                    '%Y-%m-%d %H:%M:%S'),
                            }
                            try:
                                with open(progress_path, 'w') as pf:
                                    json.dump(progress, pf, indent=2)
                                if verbose:
                                    _log(f"     Checkpoint: {completed:,}/"
                                         f"{n_parents:,} completed "
                                         f"({progress_path})")
                            except OSError:
                                pass
                            last_checkpoint_time = now
            finally:
                try:
                    os.remove(mmap_path)
                except OSError:
                    pass  # Windows: file may still be held if worker crashed

        else:
            # --- Sequential path: use verbose per-parent progress ---
            for p_idx in range(n_parents):
                parent = current_configs[p_idx]

                if verbose:
                    survivors, n_children = process_parent_verbose(
                        parent, m, c_target, n_half_child,
                        p_idx, n_parents)
                else:
                    survivors, n_children = process_parent_fused(
                        parent, m, c_target, n_half_child)
                total_children += n_children
                n_survived_this = len(survivors)
                total_survived += n_survived_this

                if n_survived_this > 0:
                    all_survivors.append(survivors)
                    all_survivors_rows += n_survived_this

                if all_survivors_rows >= shard_threshold:
                    _flush_to_shard()

        elapsed_level = time.time() - t_level

        # --- Merge shards + remaining in-RAM survivors ---
        if all_survivors:
            # Flush last batch
            _flush_to_shard()

        remaining_shards = []
        if shard_paths:
            # Multi-shard: load and merge-dedup incrementally
            if verbose:
                _log(f"     Merging {len(shard_paths)} shards...")
            merged, remaining_shards = _merge_dedup_shards(
                shard_paths, d_child, verbose)
            if merged is not None:
                all_survivors = merged
                try:
                    os.rmdir(shard_dir)
                except OSError:
                    pass
            else:
                # Survivors too large for RAM — save as sharded checkpoint
                all_survivors = np.empty((0, d_child), dtype=np.int32)
        elif all_survivors:
            all_survivors = np.vstack(all_survivors)
            all_survivors = _fast_dedup(all_survivors)
        else:
            all_survivors = np.empty((0, d_child), dtype=np.int32)

        if remaining_shards:
            # Count rows across shards (approximate — may have cross-shard dupes)
            n_survived = 0
            for sp in remaining_shards:
                sz = os.path.getsize(sp) - 128
                n_survived += sz // (d_child * 4)
            if verbose:
                total_gb = sum(os.path.getsize(p) for p in remaining_shards) / 1e9
                _log(f"     Survivors on disk: {n_survived:,} rows in "
                     f"{len(remaining_shards)} shards ({total_gb:.1f} GB)")
                _log(f"     TOO LARGE for RAM — cannot continue cascade.")
                _log(f"     Shards saved in: {shard_dir}")
        else:
            n_survived = len(all_survivors)

        if n_parents > 0:
            factor = n_survived / n_parents
        else:
            factor = 0

        lvl_info = {
            'level': level_num,
            'd_parent': d_parent,
            'd_child': d_child,
            'parents_in': n_parents,
            'total_children': total_children,
            'children_per_parent': total_children / max(1, n_parents),
            'survivors_out': n_survived,
            'expansion_factor': factor,
            'elapsed': elapsed_level,
        }
        info['levels'].append(lvl_info)

        if verbose:
            _log(f"     {elapsed_level:.2f}s: {total_children:,} children "
                 f"({total_children/max(1,n_parents):.1f}/parent)")
            _log(f"     survivors: {n_survived:,} (factor={factor:.4f}x)")
            if n_survived == 0:
                _log(f"     PROVEN at L{level_num}!")

        if n_survived == 0:
            info['proven_at'] = f'L{level_num}'
            break

        if remaining_shards:
            # Survivors too large to fit in RAM — stop cascade here.
            # Shards are already on disk for manual inspection/resume.
            info['stopped_at'] = f'L{level_num}'
            info['stopped_reason'] = 'survivors_exceed_ram'
            info['shard_paths'] = remaining_shards
            break

        # Prepare next level
        current_configs = all_survivors
        d_parent = d_child
        n_half_parent = n_half_child

        # Checkpoint survivors after each completed level
        _save_checkpoint(output_dir, level_num, current_configs, {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'level_completed': level_num,
            'd_survivors': d_child,
            'n_survived': n_survived,
            'info': info,
        })

    info['total_time'] = time.time() - t_total

    if verbose:
        _log(f"\n{'='*70}")
        if 'proven_at' in info:
            _log(f"PROVEN: c >= {c_target} (cascade converges at "
                 f"{info['proven_at']})")
        else:
            n_remain = len(current_configs) if len(current_configs) > 0 else 0
            _log(f"NOT PROVEN: {n_remain:,} survivors remain at "
                 f"d={d_parent}")
        _log(f"Total time: {_fmt_time(info['total_time'])}")
        _log(f"{'='*70}")

    return info


# =====================================================================
# Progress helpers
# =====================================================================

def _log(msg):
    """Print with immediate flush so remote/piped output is visible."""
    print(msg, flush=True)


def _progress_interval(n_total):
    """Choose a sensible progress reporting interval based on total count."""
    if n_total <= 10:
        return 1
    if n_total <= 100:
        return 10
    if n_total <= 1000:
        return 100
    if n_total <= 10_000:
        return 500
    return 1000


# =====================================================================
# Formatting helpers
# =====================================================================

def _fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.2f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.2f}h'


def print_summary(info):
    """Print a compact summary table."""
    print(f"\nCASCADE SUMMARY: n_half={info['n_half']}, m={info['m']}, "
          f"d0={info['d0']}, c_target={info['c_target']}")
    print(f"  L0: {_fmt_time(info['l0_time'])}, "
          f"{info['l0_survivors']:,} survivors")

    if info.get('levels'):
        print(f"\n  {'Level':>5} | {'Parents':>10} | {'Children':>12} | "
              f"{'Ch/Par':>8} | {'Survivors':>10} | {'Factor':>10} | "
              f"{'Time':>10}")
        print(f"  {'-'*75}")

        for lvl in info['levels']:
            factor = lvl['expansion_factor']
            if factor == 0:
                fstr = '0x'
            elif factor < 0.01:
                fstr = f'{factor:.6f}x'
            else:
                fstr = f'{factor:.4f}x'

            print(f"  L{lvl['level']:>4} | {lvl['parents_in']:>10,} | "
                  f"{lvl['total_children']:>12,} | "
                  f"{lvl['children_per_parent']:>8.1f} | "
                  f"{lvl['survivors_out']:>10,} | "
                  f"{fstr:>10} | "
                  f"{_fmt_time(lvl['elapsed']):>10}")

    proven_at = info.get('proven_at')
    if proven_at:
        print(f"\n  PROVEN at {proven_at} "
              f"(total: {_fmt_time(info['total_time'])})")
    else:
        last_lvl = info['levels'][-1] if info.get('levels') else None
        remain = last_lvl['survivors_out'] if last_lvl else info['l0_survivors']
        print(f"\n  NOT PROVEN — {remain:,} survivors remain "
              f"(total: {_fmt_time(info['total_time'])})")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CPU-only cascade prover (no GPU, no dimension limits)')
    parser.add_argument('--n_half', type=int, default=2,
                        help='Initial n_half (d0 = 2*n_half, default: 2)')
    parser.add_argument('--m', type=int, default=20,
                        help='Grid resolution (default: 20)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--max_levels', type=int, default=10,
                        help='Max refinement levels after L0 (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--resume', nargs='?', const='data', default=None,
                        help='Resume from checkpoint (optionally specify dir, '
                             'default: data)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()

    info = run_cascade(
        n_half=args.n_half,
        m=args.m,
        c_target=args.c_target,
        max_levels=args.max_levels,
        n_workers=args.workers,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        resume_dir=args.resume,
    )

    print_summary(info)

    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.output_dir,
                        f'cpu_cascade_{ts}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=convert)
    print(f"\nResult saved to {path}")


if __name__ == '__main__':
    main()
