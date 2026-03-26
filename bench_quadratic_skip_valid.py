"""
Benchmark: Surplus-Proportional Adaptive Skip in Gray Code Traversal
====================================================================

Tests whether the proposed quadratic-skip optimisation improves
performance on the L3→L4 cascade transition.

Methodology
-----------
1.  Two @njit kernels are compiled:
      • baseline – faithful reproduction of the production Gray-code
        fused-generate-and-prune loop (no subtree pruning, which is
        orthogonal and *generous* to the skip proposal because it
        leaves MORE inner sweeps for the skip to exploit).
      • skip_variant – identical code **plus** the proposed
        surplus-proportional quadratic skip after every quick-check
        hit.

2.  Both kernels are run on every L3-checkpoint parent (d_parent=32 →
    d_child=64, m=20, c_target=1.3).

3.  Correctness: survivor counts from both kernels must match the
    production kernel exactly on every parent.

4.  Timing: wall-clock comparison (JIT-warm first, then timed run).

5.  Skip statistics: the skip variant reports how many skip
    opportunities arose, how many actually skipped ≥1 extra step,
    average skip length, and total children bypassed.

Quadratic-skip derivation (conservative)
-----------------------------------------
When cursor[pos] = x₀ and the quick-check window (ℓ, s) prunes with
surplus Δ₀ = S_w(x₀) − T_w(x₀) > 0, the window sum is quadratic:

    S_w(x) = S_w(x₀) + a·(x−x₀)² + b_s·(x−x₀)

and the threshold is at most linear:

    T_w(x) = T_w(x₀) + α_T·(x−x₀)

D(x) = S_w(x) − T_w(x) = Δ₀ + a·δ² + (b_s − α_T)·δ    [δ = x−x₀]

We find max δ (in the sweep direction) such that D(x₀+kδ_dir) > 0 for
all k = 1…skip.  Uses floor of the smaller positive root of the
continuous quadratic → conservative (sound).

The threshold comparison uses floor(T*·(1−4ε)), so
    S_w > T_continuous  ⟹  S_w > floor(T_continuous·(1−4ε))  (since T≥0).
"""
import math
import os
import sys
import time

import numpy as np
import numba
from numba import njit

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CS   = os.path.join(_ROOT, "cloninger-steinerberger")
sys.path.insert(0, _CS)

from pruning import correction

# Also import the production wrapper so we can cross-check
sys.path.insert(0, os.path.join(_CS, "cpu"))
import run_cascade as _rc

# ---------------------------------------------------------------------------
# Parameters (must match checkpoint_meta.json)
# ---------------------------------------------------------------------------
N_HALF   = 2
M        = 20
C_TARGET = 1.3


# ===================================================================
# Baseline kernel – production Gray code without subtree pruning
# ===================================================================

@njit(cache=False)
def _kernel_baseline(parent_int, n_half_child, m, c_target,
                     lo_arr, hi_arr, out_buf):
    """Production Gray-code kernel, minus subtree pruning.

    Returns (n_surv, n_visited, n_qc_hit, n_full_scan).
    """
    d_parent = parent_int.shape[0]
    d_child  = 2 * d_parent
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    left_sum = np.int64(0)
    for i in range(d_parent // 2):
        left_sum += np.int64(parent_int[i])
    if np.float64(left_sum) / m_d >= threshold_asym or \
       np.float64(left_sum) / m_d <= 1.0 - threshold_asym:
        return 0, 0, 0, 0

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n   = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS  = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv  = 0
    conv_len = 2 * d_child - 1

    # Pre-computed per-ell constants
    ell_count = conv_len
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr      = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx]      = 2.0 * np.float64(ell) * inv_4n

    # ell scan order (same as production)
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used  = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell); ell_used[ell-2] = 1; oi += 1
    for ell in (d_child, d_child+1, d_child-1, d_child+2, d_child-2,
                d_child*2, d_child + d_child//2, d_child//2):
        if 2 <= ell <= 2*d_child and ell_used[ell-2] == 0:
            ell_order[oi] = np.int32(ell); ell_used[ell-2] = 1; oi += 1
    for ell in range(2, 2*d_child + 1):
        if ell_used[ell-2] == 0:
            ell_order[oi] = np.int32(ell); oi += 1

    # Build initial child + conv
    cursor = np.empty(d_parent, dtype=np.int32)
    child  = np.empty(d_child,  dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    for i in range(d_parent):
        cursor[i] = lo_arr[i]
        child[2*i]   = lo_arr[i]
        child[2*i+1] = parent_int[i] - lo_arr[i]

    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2*i] += ci * ci
            for j in range(i+1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i+j] += np.int32(2) * ci * cj

    # Sparse NZ tracking (d_child >= 32)
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos  = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0
    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i; nz_pos[i] = nz_count; nz_count += 1

    # Gray code setup
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix      = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i; radix[n_active] = r; n_active += 1

    gc_a     = np.zeros(n_active, dtype=np.int32)
    gc_dir   = np.ones(n_active,  dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    qc_ell   = np.int32(0)
    qc_s     = np.int32(0)
    qc_W_int = np.int64(0)

    n_visited   = 0
    n_qc_hit    = 0
    n_full_scan = 0

    # === MAIN LOOP ===
    while True:
        n_visited += 1

        # --- Quick-check ---
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
                n_qc_hit += 1

        # --- Full window scan ---
        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i+1] = prefix_c[i] + np.int64(child[i])
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
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
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
                        qc_s   = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                n_full_scan += 1
                # Store survivor (canonicalized)
                use_rev = False
                for i in range(d_child):
                    jj = d_child - 1 - i
                    if child[jj] < child[i]:
                        use_rev = True; break
                    elif child[jj] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # === GRAY CODE ADVANCE ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0
        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        if gc_a[j] == 0 or gc_a[j] == radix[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # === INCREMENTAL CONV UPDATE ===
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

        raw_conv[2*k1]  += new1*new1 - old1*old1
        raw_conv[2*k2]  += new2*new2 - old2*old2
        raw_conv[k1+k2] += np.int32(2) * (new1*new2 - old1*old2)

        if use_sparse:
            if old1 != 0 and new1 == 0:
                p = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k1] = -1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
            if old2 != 0 and new2 == 0:
                p = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
        else:
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
            for jj in range(k2 + 1, d_child):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj

        # qc_W_int update
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

    return n_surv, n_visited, n_qc_hit, n_full_scan


# ===================================================================
# Skip-variant kernel – baseline + quadratic surplus skip
# ===================================================================

@njit(cache=False)
def _kernel_skip(parent_int, n_half_child, m, c_target,
                 lo_arr, hi_arr, out_buf):
    """Gray-code kernel with surplus-proportional quadratic skip.

    After a quick-check prune, computes the quadratic D(δ) = S_w − T_w
    as a function of cursor displacement δ, and finds the largest
    contiguous skip within the current Gray-code sweep direction.

    Returns (n_surv, n_visited, n_qc_hit, n_full_scan,
             n_skip_attempts, n_skip_successes, total_children_skipped).
    """
    d_parent = parent_int.shape[0]
    d_child  = 2 * d_parent
    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    left_sum = np.int64(0)
    for i in range(d_parent // 2):
        left_sum += np.int64(parent_int[i])
    if np.float64(left_sum) / m_d >= threshold_asym or \
       np.float64(left_sum) / m_d <= 1.0 - threshold_asym:
        return 0, 0, 0, 0, 0, 0, 0

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n   = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS  = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    max_survivors = out_buf.shape[0]
    n_surv  = 0
    conv_len = 2 * d_child - 1

    # Per-ell constants
    ell_count = conv_len
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr      = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx]      = 2.0 * np.float64(ell) * inv_4n

    # ell scan order
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used  = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell); ell_used[ell-2] = 1; oi += 1
    for ell in (d_child, d_child+1, d_child-1, d_child+2, d_child-2,
                d_child*2, d_child + d_child//2, d_child//2):
        if 2 <= ell <= 2*d_child and ell_used[ell-2] == 0:
            ell_order[oi] = np.int32(ell); ell_used[ell-2] = 1; oi += 1
    for ell in range(2, 2*d_child + 1):
        if ell_used[ell-2] == 0:
            ell_order[oi] = np.int32(ell); oi += 1

    # Build initial child + conv
    cursor   = np.empty(d_parent, dtype=np.int32)
    child    = np.empty(d_child,  dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)

    for i in range(d_parent):
        cursor[i] = lo_arr[i]
        child[2*i]   = lo_arr[i]
        child[2*i+1] = parent_int[i] - lo_arr[i]

    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2*i] += ci * ci
            for j in range(i+1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i+j] += np.int32(2) * ci * cj

    # Sparse NZ tracking
    use_sparse = d_child >= 32
    nz_list = np.empty(d_child, dtype=np.int32)
    nz_pos  = np.full(d_child, -1, dtype=np.int32)
    nz_count = 0
    if use_sparse:
        for i in range(d_child):
            if child[i] != 0:
                nz_list[nz_count] = i; nz_pos[i] = nz_count; nz_count += 1

    # Gray code setup
    n_active = 0
    active_pos = np.empty(d_parent, dtype=np.int32)
    radix_arr  = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent - 1, -1, -1):
        r = hi_arr[i] - lo_arr[i] + 1
        if r > 1:
            active_pos[n_active] = i; radix_arr[n_active] = r; n_active += 1

    gc_a     = np.zeros(n_active, dtype=np.int32)
    gc_dir   = np.ones(n_active,  dtype=np.int32)
    gc_focus = np.empty(n_active + 1, dtype=np.int32)
    for i in range(n_active + 1):
        gc_focus[i] = i

    qc_ell   = np.int32(0)
    qc_s     = np.int32(0)
    qc_W_int = np.int64(0)

    n_visited   = 0
    n_qc_hit    = 0
    n_full_scan = 0
    n_skip_attempts   = 0    # times we tried quadratic skip
    n_skip_successes  = 0    # times skip > 0
    total_children_skipped = 0

    # === MAIN LOOP ===
    while True:
        n_visited += 1

        # --- Quick-check ---
        quick_killed = False
        qc_surplus = np.float64(0.0)
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
                n_qc_hit += 1
                # Surplus for quadratic skip (use conservative continuous T)
                qc_surplus = np.float64(ws_qc) - dyn_x_qc

        # ============================================================
        # QUADRATIC SKIP – only when quick-check fires
        # ============================================================
        if quick_killed and n_active > 0:
            # What position will the NEXT Gray-code step advance?
            next_j = gc_focus[0]
            # CRITICAL: only skip for innermost position (j=0).
            # For j>0, advancing by >1 skips entire inner subtrees
            # whose children have NOT been verified by the quadratic
            # model (the model only covers cursor[pos] variation with
            # inner bins fixed at their current values).
            if next_j == 0:
                pos_next = active_pos[next_j]
                dir_next = gc_dir[next_j]

                # Remaining steps in current sweep direction
                if dir_next > 0:
                    remaining = radix_arr[next_j] - 1 - gc_a[next_j]
                else:
                    remaining = gc_a[next_j]

                if remaining > 1:
                    n_skip_attempts += 1

                    # Child bins controlled by this cursor position
                    k1 = 2 * pos_next
                    k2 = k1 + 1
                    p_val = np.int32(parent_int[pos_next])
                    x0 = np.int32(child[k1])  # current value

                    # --- Compute quadratic coefficients of D(δ) ---
                    # D(δ) = Δ₀ + a·δ² + β·δ
                    # where δ = displacement in cursor (dir_next per step)
                    #
                    # Quadratic coeff 'a' from self/mutual terms:
                    #   a = I(2k1∈W) + I(2k2∈W) − 2·I(k1+k2∈W)
                    # where W = window [qc_s .. qc_s + qc_ell − 2]
                    w_lo = np.int32(qc_s)
                    w_hi = np.int32(qc_s + qc_ell - 2)

                    idx_2k1  = 2 * k1       # = 4·pos_next
                    idx_2k2  = 2 * k2       # = 4·pos_next + 2
                    idx_k1k2 = k1 + k2      # = 4·pos_next + 1

                    I1 = np.int32(1) if (w_lo <= idx_2k1 and idx_2k1 <= w_hi) else np.int32(0)
                    I2 = np.int32(1) if (w_lo <= idx_2k2 and idx_2k2 <= w_hi) else np.int32(0)
                    I3 = np.int32(1) if (w_lo <= idx_k1k2 and idx_k1k2 <= w_hi) else np.int32(0)

                    a_coeff = np.float64(I1 + I2 - 2 * I3)

                    # Linear coeff 'b_s' from self/mutual + cross terms:
                    #   b_s = 2·x0·I1 − 2·(p−x0)·I2 + 2·(p−2x0)·I3
                    #       + 2·(L1 − L2)
                    # where L1 = Σ child[j] for j with k1+j ∈ W and j≠k1,k2
                    #       L2 = Σ child[j] for j with k2+j ∈ W and j≠k1,k2
                    x0f = np.float64(x0)
                    pf  = np.float64(p_val)

                    b_self = 2.0*x0f*np.float64(I1) - 2.0*(pf - x0f)*np.float64(I2) \
                             + 2.0*(pf - 2.0*x0f)*np.float64(I3)

                    L1 = np.float64(0.0)
                    L2 = np.float64(0.0)
                    for jj in range(d_child):
                        if jj == k1 or jj == k2:
                            continue
                        cj = np.float64(child[jj])
                        if cj == 0.0:
                            continue
                        idx1 = k1 + jj
                        if w_lo <= idx1 and idx1 <= w_hi:
                            L1 += cj
                        idx2 = k2 + jj
                        if w_lo <= idx2 and idx2 <= w_hi:
                            L2 += cj

                    b_s = b_self + 2.0 * (L1 - L2)

                    # Threshold linear coeff α_T from W_int:
                    # W_int range for the quick-check window
                    qc_lo_bin = qc_s - (d_child - 1)
                    if qc_lo_bin < 0:
                        qc_lo_bin = 0
                    qc_hi_bin = qc_s + qc_ell - 2
                    if qc_hi_bin > d_child - 1:
                        qc_hi_bin = d_child - 1

                    # How does W_int change per unit δ?
                    # child[k1] changes by +1 per +δ, child[k2] changes by −1
                    k1_in_range = (qc_lo_bin <= k1 and k1 <= qc_hi_bin)
                    k2_in_range = (qc_lo_bin <= k2 and k2 <= qc_hi_bin)
                    dW_per_delta = np.float64(0.0)
                    if k1_in_range:
                        dW_per_delta += 1.0
                    if k2_in_range:
                        dW_per_delta -= 1.0

                    alpha_T = two_ell_arr[qc_ell - 2] * dW_per_delta

                    # D(δ) = Δ₀ + a·δ² + β·δ  where β = b_s − α_T
                    # Scale δ by dir_next: actual δ per step = dir_next
                    # So D(k) = Δ₀ + a·(k·dir)² + β·(k·dir)
                    #          = Δ₀ + a·k² + β·dir·k
                    # (since dir² = 1)
                    beta = (b_s - alpha_T) * np.float64(dir_next)
                    delta_0 = qc_surplus  # S_w(x0) - T_w(x0)

                    # Find max k ∈ [1, remaining-1] such that D(k) > 0
                    # D(k) = a·k² + β·k + Δ₀
                    # We need this > 0 for k = 1, 2, ..., skip
                    max_skip = remaining - 1  # max possible extra steps

                    if max_skip > 0:
                        skip = 0
                        if a_coeff == 0.0:
                            # Linear: D(k) = β·k + Δ₀
                            if beta >= 0.0:
                                # D is non-decreasing, always > 0
                                skip = max_skip
                            elif delta_0 > 0.0:
                                # D crosses zero at k = -Δ₀/β
                                k_cross = -delta_0 / beta
                                skip = min(max_skip, int(math.floor(k_cross - 1e-12)))
                                if skip < 0:
                                    skip = 0
                        else:
                            # Quadratic: find roots of a·k² + β·k + Δ₀ = 0
                            disc = beta * beta - 4.0 * a_coeff * delta_0
                            if disc <= 0.0:
                                # No real roots → D never crosses zero
                                # (or touches once). Since D(0) = Δ₀ > 0,
                                # D > 0 everywhere.
                                if a_coeff > 0.0:
                                    skip = max_skip
                                else:
                                    # a < 0, disc ≤ 0: means D(0)=Δ₀ > 0 and
                                    # D always positive (concave, no roots)
                                    skip = max_skip
                            else:
                                sqrt_disc = math.sqrt(disc)
                                # Roots: k = (−β ± √disc) / (2a)
                                r1 = (-beta - sqrt_disc) / (2.0 * a_coeff)
                                r2 = (-beta + sqrt_disc) / (2.0 * a_coeff)
                                # D(0) = Δ₀ > 0, so D > 0 between/outside
                                # roots depending on sign of a.
                                if a_coeff > 0.0:
                                    # Convex: D > 0 outside [r1, r2].
                                    # Since D(0)=Δ₀>0, 0 is outside [r1,r2].
                                    # D > 0 for k < r1 or k > r2.
                                    # If r1 > 0: skip up to floor(r1) − 1
                                    # If r1 ≤ 0: D>0 for all k>r2,
                                    #   but we need contiguous from k=1
                                    if r1 > 1.0:
                                        skip = min(max_skip, int(math.floor(r1 - 1e-12)) - 1)
                                        if skip < 0:
                                            skip = 0
                                    elif r2 < 1.0:
                                        # Both roots < 1, D(1) > 0, D > 0
                                        # for all k ≥ 1 (convex)
                                        skip = max_skip
                                    else:
                                        # r1 ≤ 1 ≤ r2: D(1) might be ≤ 0
                                        skip = 0
                                else:
                                    # Concave (a < 0): D > 0 inside (r1, r2).
                                    # Since D(0) > 0, 0 ∈ (r1, r2).
                                    # D > 0 for k < r2
                                    if r2 > 1.0:
                                        skip = min(max_skip, int(math.floor(r2 - 1e-12)) - 1)
                                        if skip < 0:
                                            skip = 0
                                    else:
                                        skip = 0

                        # === EXECUTE SKIP ===
                        if skip > 0:
                            n_skip_successes += 1
                            total_children_skipped += skip

                            # Multi-step Gray code advance:
                            # Advance position next_j by (skip+1) steps
                            # (1 normal + skip extra)
                            total_advance = skip + 1

                            # Reset focus BEFORE boundary check (matches
                            # normal advance order: lines 1679-1692)
                            gc_focus[0] = 0

                            gc_a[next_j] += total_advance * dir_next
                            cursor[pos_next] = lo_arr[pos_next] + gc_a[next_j]

                            # Boundary check (may overwrite gc_focus[next_j])
                            if gc_a[next_j] == 0 or gc_a[next_j] == radix_arr[next_j] - 1:
                                gc_dir[next_j] = -gc_dir[next_j]
                                gc_focus[next_j] = gc_focus[next_j + 1]
                                gc_focus[next_j + 1] = next_j + 1

                            # Update child and raw_conv for multi-step jump
                            old1_s = np.int32(child[k1])
                            old2_s = np.int32(child[k2])
                            child[k1] = cursor[pos_next]
                            child[k2] = parent_int[pos_next] - cursor[pos_next]
                            new1_s = np.int32(child[k1])
                            new2_s = np.int32(child[k2])
                            d1 = new1_s - old1_s
                            d2 = new2_s - old2_s

                            raw_conv[2*k1]  += new1_s*new1_s - old1_s*old1_s
                            raw_conv[2*k2]  += new2_s*new2_s - old2_s*old2_s
                            raw_conv[k1+k2] += np.int32(2) * (new1_s*new2_s - old1_s*old2_s)

                            if use_sparse:
                                if old1_s != 0 and new1_s == 0:
                                    pp = nz_pos[k1]; nz_count -= 1
                                    last = nz_list[nz_count]; nz_list[pp] = last
                                    nz_pos[last] = pp; nz_pos[k1] = -1
                                elif old1_s == 0 and new1_s != 0:
                                    nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
                                if old2_s != 0 and new2_s == 0:
                                    pp = nz_pos[k2]; nz_count -= 1
                                    last = nz_list[nz_count]; nz_list[pp] = last
                                    nz_pos[last] = pp; nz_pos[k2] = -1
                                elif old2_s == 0 and new2_s != 0:
                                    nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
                                for idx in range(nz_count):
                                    jj = nz_list[idx]
                                    if jj != k1 and jj != k2:
                                        cj = np.int32(child[jj])
                                        raw_conv[k1+jj] += np.int32(2) * d1 * cj
                                        raw_conv[k2+jj] += np.int32(2) * d2 * cj
                            else:
                                for jj in range(k1):
                                    cj = np.int32(child[jj])
                                    if cj != 0:
                                        raw_conv[k1+jj] += np.int32(2) * d1 * cj
                                        raw_conv[k2+jj] += np.int32(2) * d2 * cj
                                for jj in range(k2 + 1, d_child):
                                    cj = np.int32(child[jj])
                                    if cj != 0:
                                        raw_conv[k1+jj] += np.int32(2) * d1 * cj
                                        raw_conv[k2+jj] += np.int32(2) * d2 * cj

                            # Update qc_W_int for multi-step
                            if qc_ell > 0:
                                ql = qc_s - (d_child - 1)
                                if ql < 0:
                                    ql = 0
                                qh = qc_s + qc_ell - 2
                                if qh > d_child - 1:
                                    qh = d_child - 1
                                if ql <= k1 and k1 <= qh:
                                    qc_W_int += np.int64(d1)
                                if ql <= k2 and k2 <= qh:
                                    qc_W_int += np.int64(d2)

                            # Already advanced → skip normal advance below
                            continue

        # --- Full window scan (only if not quick-killed) ---
        if not quick_killed:
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i+1] = prefix_c[i] + np.int64(child[i])
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
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += np.int64(raw_conv[k])
                for s_lo in range(n_windows):
                    if s_lo > 0:
                        ws += np.int64(raw_conv[s_lo + n_cv - 1]) - np.int64(raw_conv[s_lo - 1])
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
                        qc_s   = np.int32(s_lo)
                        qc_W_int = W_int
                        break

            if not pruned:
                n_full_scan += 1
                use_rev = False
                for i in range(d_child):
                    jj = d_child - 1 - i
                    if child[jj] < child[i]:
                        use_rev = True; break
                    elif child[jj] > child[i]:
                        break
                if n_surv < max_survivors:
                    if use_rev:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[d_child - 1 - i]
                    else:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                n_surv += 1

        # === GRAY CODE ADVANCE (normal, single-step) ===
        j = gc_focus[0]
        if j == n_active:
            break
        gc_focus[0] = 0
        pos = active_pos[j]
        gc_a[j] += gc_dir[j]
        cursor[pos] = lo_arr[pos] + gc_a[j]

        if gc_a[j] == 0 or gc_a[j] == radix_arr[j] - 1:
            gc_dir[j] = -gc_dir[j]
            gc_focus[j] = gc_focus[j + 1]
            gc_focus[j + 1] = j + 1

        # Incremental conv update
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

        raw_conv[2*k1]  += new1*new1 - old1*old1
        raw_conv[2*k2]  += new2*new2 - old2*old2
        raw_conv[k1+k2] += np.int32(2) * (new1*new2 - old1*old2)

        if use_sparse:
            if old1 != 0 and new1 == 0:
                p = nz_pos[k1]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k1] = -1
            elif old1 == 0 and new1 != 0:
                nz_list[nz_count] = k1; nz_pos[k1] = nz_count; nz_count += 1
            if old2 != 0 and new2 == 0:
                p = nz_pos[k2]; nz_count -= 1
                last = nz_list[nz_count]; nz_list[p] = last
                nz_pos[last] = p; nz_pos[k2] = -1
            elif old2 == 0 and new2 != 0:
                nz_list[nz_count] = k2; nz_pos[k2] = nz_count; nz_count += 1
            for idx in range(nz_count):
                jj = nz_list[idx]
                if jj != k1 and jj != k2:
                    cj = np.int32(child[jj])
                    raw_conv[k1+jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2+jj] += np.int32(2) * delta2 * cj
        else:
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1+jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2+jj] += np.int32(2) * delta2 * cj
            for jj in range(k2 + 1, d_child):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1+jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2+jj] += np.int32(2) * delta2 * cj

        # qc_W_int update
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

    return (n_surv, n_visited, n_qc_hit, n_full_scan,
            n_skip_attempts, n_skip_successes, total_children_skipped)


# ===================================================================
# Benchmark harness
# ===================================================================

def run_benchmark(max_parents=None):
    """Run the full benchmark on L3→L4 transition."""
    # Load L3 survivors
    ckpt_path = os.path.join(_ROOT, "data", "checkpoint_L3_survivors.npy")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: L3 checkpoint not found at {ckpt_path}")
        sys.exit(1)

    parents = np.load(ckpt_path)
    if max_parents is not None and max_parents < len(parents):
        parents = parents[:max_parents]
    n_parents = len(parents)
    d_parent = parents.shape[1]
    d_child  = 2 * d_parent
    n_half_child = d_child // 2  # n_half_child = d_child / 2

    print(f"Loaded {n_parents} L3 parents (d_parent={d_parent}, d_child={d_child})")
    print(f"Parameters: n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"n_half_child={n_half_child}")

    # Compute x_cap to report sweep lengths
    corr = correction(M, n_half_child)
    thresh = C_TARGET + corr + 1e-9
    x_cap = int(math.floor(M * math.sqrt(thresh / d_child)))
    x_cap_cs = int(math.floor(M * math.sqrt(C_TARGET / d_child)))
    x_cap = min(x_cap, x_cap_cs, M)
    x_cap = max(x_cap, 0)
    print(f"x_cap={x_cap}, max_radix={x_cap+1}, max_sweep_len={x_cap}")
    print()

    # ---------------------------------------------------------------
    # Phase 1: JIT warmup (compile both kernels on a small parent)
    # ---------------------------------------------------------------
    print("Phase 1: JIT compilation warmup...")
    warmup_parent = parents[0].copy()
    result = _rc._compute_bin_ranges(warmup_parent, M, C_TARGET, d_child, n_half_child)
    if result is None:
        print("  WARNING: first parent has empty range, trying next")
        for idx in range(1, n_parents):
            warmup_parent = parents[idx].copy()
            result = _rc._compute_bin_ranges(warmup_parent, M, C_TARGET, d_child, n_half_child)
            if result is not None:
                break
    lo_arr, hi_arr, total_children = result

    buf_cap = min(total_children, 100_000)
    buf_base = np.empty((buf_cap, d_child), dtype=np.int32)
    buf_skip = np.empty((buf_cap, d_child), dtype=np.int32)

    t0 = time.perf_counter()
    _kernel_baseline(warmup_parent, n_half_child, M, C_TARGET,
                     lo_arr, hi_arr, buf_base)
    t1 = time.perf_counter()
    print(f"  Baseline kernel compiled in {t1-t0:.2f}s")

    t0 = time.perf_counter()
    _kernel_skip(warmup_parent, n_half_child, M, C_TARGET,
                 lo_arr, hi_arr, buf_skip)
    t1 = time.perf_counter()
    print(f"  Skip kernel compiled in {t1-t0:.2f}s")
    print()

    # ---------------------------------------------------------------
    # Phase 2: Cross-check baseline against production kernel
    # ---------------------------------------------------------------
    print("Phase 2: Cross-checking baseline vs production on first 20 parents...")
    mismatches = 0
    check_count = min(20, n_parents)
    for i in range(check_count):
        p = parents[i].copy()
        result = _rc._compute_bin_ranges(p, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo, hi, tc = result
        bc = min(tc, 100_000)
        buf_prod = np.empty((bc, d_child), dtype=np.int32)
        buf_b    = np.empty((bc, d_child), dtype=np.int32)

        n_prod, _ = _rc._fused_generate_and_prune_gray(
            p, n_half_child, M, C_TARGET, lo, hi, buf_prod)
        n_base = _kernel_baseline(p, n_half_child, M, C_TARGET, lo, hi, buf_b)[0]

        if n_prod != n_base:
            print(f"  MISMATCH parent {i}: production={n_prod}, baseline={n_base}")
            mismatches += 1

    if mismatches == 0:
        print(f"  OK: all {check_count} parents match production kernel")
    else:
        print(f"  WARNING: {mismatches} mismatches! Baseline kernel may differ.")
        print(f"  (Production includes subtree pruning which baseline omits;")
        print(f"   survivor count should match since subtree pruning only skips")
        print(f"   already-pruned subtrees.)")
    print()

    # ---------------------------------------------------------------
    # Phase 3: Full benchmark – Baseline
    # ---------------------------------------------------------------
    print(f"Phase 3: Timing baseline on all {n_parents} parents...")
    total_surv_base = 0
    total_visited_base = 0
    total_qc_base = 0
    total_full_base = 0

    t_start_base = time.perf_counter()
    for i in range(n_parents):
        p = parents[i].copy()
        result = _rc._compute_bin_ranges(p, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo, hi, tc = result
        bc = min(tc, 100_000)
        buf = np.empty((bc, d_child), dtype=np.int32)
        ns, nv, nqc, nfs = _kernel_baseline(
            p, n_half_child, M, C_TARGET, lo, hi, buf)
        total_surv_base += ns
        total_visited_base += nv
        total_qc_base += nqc
        total_full_base += nfs
    t_end_base = time.perf_counter()
    dt_base = t_end_base - t_start_base

    print(f"  Time:       {dt_base:.4f}s")
    print(f"  Survivors:  {total_surv_base:,}")
    print(f"  Visited:    {total_visited_base:,}")
    print(f"  QC hits:    {total_qc_base:,} ({100*total_qc_base/max(1,total_visited_base):.1f}%)")
    print(f"  Full scans: {total_full_base:,}")
    print()

    # ---------------------------------------------------------------
    # Phase 4: Full benchmark – Skip variant
    # ---------------------------------------------------------------
    print(f"Phase 4: Timing skip variant on all {n_parents} parents...")
    total_surv_skip = 0
    total_visited_skip = 0
    total_qc_skip = 0
    total_full_skip = 0
    total_skip_attempts = 0
    total_skip_successes = 0
    total_skipped_children = 0

    t_start_skip = time.perf_counter()
    for i in range(n_parents):
        p = parents[i].copy()
        result = _rc._compute_bin_ranges(p, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo, hi, tc = result
        bc = min(tc, 100_000)
        buf = np.empty((bc, d_child), dtype=np.int32)
        ns, nv, nqc, nfs, sa, ss, sc = _kernel_skip(
            p, n_half_child, M, C_TARGET, lo, hi, buf)
        total_surv_skip += ns
        total_visited_skip += nv
        total_qc_skip += nqc
        total_full_skip += nfs
        total_skip_attempts += sa
        total_skip_successes += ss
        total_skipped_children += sc
    t_end_skip = time.perf_counter()
    dt_skip = t_end_skip - t_start_skip

    print(f"  Time:       {dt_skip:.4f}s")
    print(f"  Survivors:  {total_surv_skip:,}")
    print(f"  Visited:    {total_visited_skip:,}")
    print(f"  QC hits:    {total_qc_skip:,} ({100*total_qc_skip/max(1,total_visited_skip):.1f}%)")
    print(f"  Full scans: {total_full_skip:,}")
    print()

    # ---------------------------------------------------------------
    # Phase 5: Correctness verification (every parent must match)
    # ---------------------------------------------------------------
    print("Phase 5: Correctness verification (all parents)...")
    mismatches_skip = 0
    for i in range(n_parents):
        p = parents[i].copy()
        result = _rc._compute_bin_ranges(p, M, C_TARGET, d_child, n_half_child)
        if result is None:
            continue
        lo, hi, tc = result
        bc = min(tc, 100_000)
        buf_b = np.empty((bc, d_child), dtype=np.int32)
        buf_s = np.empty((bc, d_child), dtype=np.int32)
        ns_b = _kernel_baseline(p, n_half_child, M, C_TARGET, lo, hi, buf_b)[0]
        ns_s = _kernel_skip(p, n_half_child, M, C_TARGET, lo, hi, buf_s)[0]
        if ns_b != ns_s:
            mismatches_skip += 1
            print(f"  MISMATCH parent {i}: baseline={ns_b}, skip={ns_s}")
            if mismatches_skip >= 10:
                print(f"  ... stopping after 10 mismatches")
                break

    if mismatches_skip == 0:
        print(f"  OK: all {n_parents} parents produce identical survivor counts")
    else:
        print(f"  FAIL: {mismatches_skip} parents with different survivor counts!")
    print()

    # ---------------------------------------------------------------
    # Phase 6: Summary report
    # ---------------------------------------------------------------
    print("=" * 65)
    print("BENCHMARK REPORT: Surplus-Proportional Adaptive Skip")
    print("=" * 65)
    print(f"Parents tested:       {n_parents}")
    print(f"d_parent={d_parent}, d_child={d_child}, m={M}, c_target={C_TARGET}")
    print(f"x_cap={x_cap}, max_radix={x_cap+1}")
    print()
    print(f"{'Metric':<35} {'Baseline':>12} {'Skip':>12} {'Delta':>10}")
    print("-" * 65)
    print(f"{'Wall-clock time (s)':<35} {dt_base:>12.4f} {dt_skip:>12.4f} "
          f"{'%.1f%%' % (100*(dt_skip - dt_base)/max(1e-9, dt_base)):>10}")
    print(f"{'Children visited':<35} {total_visited_base:>12,} {total_visited_skip:>12,} "
          f"{total_visited_skip - total_visited_base:>10,}")
    print(f"{'Quick-check hits':<35} {total_qc_base:>12,} {total_qc_skip:>12,}")
    print(f"{'Full window scans':<35} {total_full_base:>12,} {total_full_skip:>12,}")
    print(f"{'Survivors':<35} {total_surv_base:>12,} {total_surv_skip:>12,} "
          f"{'MATCH' if total_surv_base == total_surv_skip else 'MISMATCH':>10}")
    print()
    print("Skip-specific statistics:")
    print(f"  Skip attempts (remaining > 1):    {total_skip_attempts:,}")
    print(f"  Successful skips (skip > 0):       {total_skip_successes:,}")
    if total_skip_successes > 0:
        print(f"  Average skip length:               {total_skipped_children / total_skip_successes:.2f}")
    print(f"  Total children skipped:            {total_skipped_children:,}")
    if total_visited_base > 0:
        print(f"  Fraction of children skipped:      "
              f"{100*total_skipped_children/total_visited_base:.2f}%")
    print()

    # Overhead analysis
    coeff_ops_per_attempt = d_child  # O(d) for L1, L2 sums
    total_coeff_ops = total_skip_attempts * coeff_ops_per_attempt
    saved_ops_per_skip = d_child  # each skipped child saves ~O(d) conv update
    total_saved_ops = total_skipped_children * saved_ops_per_skip
    print("Estimated operation count:")
    print(f"  Coefficient computation overhead:   {total_coeff_ops:,} ops "
          f"({total_skip_attempts} attempts x {coeff_ops_per_attempt} ops)")
    print(f"  Saved by skipping:                  {total_saved_ops:,} ops "
          f"({total_skipped_children} skipped x {saved_ops_per_skip} ops)")
    net = total_saved_ops - total_coeff_ops
    print(f"  Net:                                {net:>+,} ops "
          f"({'WIN' if net > 0 else 'LOSS'})")
    print()

    if dt_skip <= dt_base:
        print(f"VERDICT: Skip variant is {100*(dt_base - dt_skip)/dt_base:.1f}% "
              f"FASTER (wall-clock)")
    else:
        print(f"VERDICT: Skip variant is {100*(dt_skip - dt_base)/dt_base:.1f}% "
              f"SLOWER (wall-clock)")

    if mismatches_skip > 0:
        print("WARNING: CORRECTNESS FAILURE — skip variant produces different results!")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--max-parents", type=int, default=None,
                    help="Limit to first N parents (for quick iteration)")
    _args = _p.parse_args()
    run_benchmark(max_parents=_args.max_parents)
