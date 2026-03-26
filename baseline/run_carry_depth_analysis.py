"""Analyze subtree pruning effectiveness by carry depth.

Instruments the odometer kernel at a finer granularity: for each deep carry,
records the carry position (= number of fixed parent positions) and whether
the subtree prune succeeded.
"""
import os
import struct
import sys
import time
import math

import numpy as np
from numba import njit

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cs_dir = os.path.join(_root, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import _compute_bin_ranges

N_HALF = 2
M = 20
C_TARGET = 1.4
D_PARENT = 32
D_CHILD = 64
N_HALF_CHILD = N_HALF * (2 ** 4)

CHECKPOINT_PATH = os.path.join(_root, 'data', 'checkpoint_L3_survivors.npy')


def load_partial_npy(path, n_cols=32, dtype=np.int32):
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        magic = f.read(6)
        major, minor = struct.unpack('BB', f.read(2))
        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]
        _ = f.read(header_len)
        header_offset = f.tell()
    data_bytes = fsize - header_offset
    row_bytes = n_cols * np.dtype(dtype).itemsize
    n_rows = data_bytes // row_bytes
    data = np.memmap(path, dtype=dtype, mode='r',
                     offset=header_offset, shape=(n_rows, n_cols))
    return np.array(data)


@njit(cache=True)
def _analyze_carry_depths(parent_int, n_half_child, m, c_target,
                          lo_arr, hi_arr,
                          carry_counts, carry_successes, carry_children_skipped):
    """Like the instrumented odometer, but records per-carry-depth statistics.

    carry_counts[c]:           number of deep carries at position c
    carry_successes[c]:        number of successful subtree prunes at position c
    carry_children_skipped[c]: total children skipped at position c
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent

    m_d = np.float64(m)
    threshold_asym = math.sqrt(c_target / 2.0)

    left_sum_parent = np.int64(0)
    for i in range(d_parent // 2):
        left_sum_parent += np.int64(parent_int[i])
    left_frac = np.float64(left_sum_parent) / m_d
    if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
        return 0, 0, 0  # asymmetry-skipped

    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * np.float64(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS

    conv_len = 2 * d_child - 1
    carry_threshold = d_parent // 4

    parent_prefix = np.empty(d_parent + 1, dtype=np.int64)
    parent_prefix[0] = 0
    for i in range(d_parent):
        parent_prefix[i + 1] = parent_prefix[i] + np.int64(parent_int[i])

    cursor = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        cursor[i] = lo_arr[i]

    child = np.empty(d_child, dtype=np.int32)
    raw_conv = np.empty(conv_len, dtype=np.int32)
    prefix_c = np.empty(d_child + 1, dtype=np.int64)
    conv = np.empty(conv_len, dtype=np.int32)

    # Build initial child
    for i in range(d_parent):
        child[2 * i] = cursor[i]
        child[2 * i + 1] = parent_int[i] - cursor[i]

    # Full raw_conv
    for k in range(conv_len):
        raw_conv[k] = np.int32(0)
    for i in range(d_child):
        ci = np.int32(child[i])
        if ci != 0:
            raw_conv[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int32(child[j])
                if cj != 0:
                    raw_conv[i + j] += np.int32(2) * ci * cj

    # Per-ell constants
    ell_count = 2 * d_child - 1
    dyn_base_ell_arr = np.empty(ell_count, dtype=np.float64)
    two_ell_arr = np.empty(ell_count, dtype=np.float64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        dyn_base_ell_arr[idx] = dyn_base * np.float64(ell) * inv_4n
        two_ell_arr[idx] = 2.0 * np.float64(ell) * inv_4n

    # ell scan order
    ell_order = np.empty(ell_count, dtype=np.int32)
    ell_used = np.zeros(ell_count, dtype=np.int32)
    oi = 0
    phase1_end = min(16, 2 * d_child)
    for ell in range(2, phase1_end + 1):
        ell_order[oi] = np.int32(ell)
        ell_used[ell - 2] = np.int32(1)
        oi += 1
    for ell in (d_child, d_child + 1, d_child - 1, d_child + 2, d_child - 2,
                d_child * 2, d_child + d_child // 2, d_child // 2):
        if 2 <= ell <= 2 * d_child and ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            ell_used[ell - 2] = np.int32(1)
            oi += 1
    for ell in range(2, 2 * d_child + 1):
        if ell_used[ell - 2] == 0:
            ell_order[oi] = np.int32(ell)
            oi += 1

    qc_ell = np.int32(0)
    qc_s = np.int32(0)
    qc_W_int = np.int64(0)

    n_visited = np.int64(0)
    n_deep_total = np.int64(0)
    n_subtree_total = np.int64(0)

    while True:
        n_visited += 1

        # --- Test current child (quick-check + full scan) ---
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
            prefix_c[0] = 0
            for i in range(d_child):
                prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

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
                        qc_s = np.int32(s_lo)
                        qc_W_int = W_int
                        break

        # --- Advance cursor (odometer) ---
        carry = d_parent - 1
        while carry >= 0:
            cursor[carry] += 1
            if cursor[carry] <= hi_arr[carry]:
                break
            cursor[carry] = lo_arr[carry]
            carry -= 1

        if carry < 0:
            break

        n_changed = d_parent - carry

        if n_changed == 1:
            # Fast path
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
            raw_conv[2 * k1] += new1 * new1 - old1 * old1
            raw_conv[2 * k2] += new2 * new2 - old2 * old2
            raw_conv[k1 + k2] += np.int32(2) * (new1 * new2 - old1 * old2)
            for jj in range(k1):
                cj = np.int32(child[jj])
                if cj != 0:
                    raw_conv[k1 + jj] += np.int32(2) * delta1 * cj
                    raw_conv[k2 + jj] += np.int32(2) * delta2 * cj
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
            # Short carry: update changed positions
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = parent_int[pos] - cursor[pos]
            # Full recompute
            for k in range(conv_len):
                raw_conv[k] = np.int32(0)
            for i in range(d_child):
                ci = np.int32(child[i])
                if ci != 0:
                    raw_conv[2 * i] += ci * ci
                    for j in range(i + 1, d_child):
                        cj = np.int32(child[j])
                        if cj != 0:
                            raw_conv[i + j] += np.int32(2) * ci * cj
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
            # Deep carry: attempt subtree prune
            n_deep_total += 1
            carry_counts[carry] += 1
            fixed_len = 2 * carry

            subtree_pruned = False
            if fixed_len >= 4:
                partial_conv_len = 2 * fixed_len - 1
                for k in range(partial_conv_len):
                    conv[k] = np.int32(0)
                for i in range(fixed_len):
                    ci = np.int32(child[i])
                    if ci != 0:
                        conv[2 * i] += ci * ci
                        for j in range(i + 1, fixed_len):
                            cj = np.int32(child[j])
                            if cj != 0:
                                conv[i + j] += np.int32(2) * ci * cj
                for k in range(1, partial_conv_len):
                    conv[k] += conv[k - 1]

                prefix_c[0] = 0
                for i in range(fixed_len):
                    prefix_c[i + 1] = prefix_c[i] + np.int64(child[i])

                first_unfixed_parent = carry
                for ell_oi in range(ell_count):
                    if subtree_pruned:
                        break
                    ell = ell_order[ell_oi]
                    n_cv = ell - 1
                    ell_idx = ell - 2
                    dyn_base_ell = dyn_base_ell_arr[ell_idx]
                    two_ell_inv_4n = two_ell_arr[ell_idx]

                    n_windows_partial = partial_conv_len - n_cv + 1
                    if n_windows_partial <= 0:
                        continue

                    for s_lo in range(n_windows_partial):
                        s_hi = s_lo + n_cv - 1
                        ws = np.int64(conv[s_hi])
                        if s_lo > 0:
                            ws -= np.int64(conv[s_lo - 1])

                        lo_bin = s_lo - (d_child - 1)
                        if lo_bin < 0:
                            lo_bin = 0
                        hi_bin = s_lo + ell - 2
                        if hi_bin > d_child - 1:
                            hi_bin = d_child - 1

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
                n_subtree_total += 1
                carry_successes[carry] += 1
                # Count children in subtree
                subtree_size = np.int64(1)
                for i in range(carry + 1, d_parent):
                    subtree_size *= np.int64(hi_arr[i] - lo_arr[i] + 1)
                carry_children_skipped[carry] += subtree_size

                # Fast-forward
                for i in range(carry + 1, d_parent):
                    cursor[i] = hi_arr[i]
                for pos in range(carry, d_parent):
                    child[2 * pos] = cursor[pos]
                    child[2 * pos + 1] = parent_int[pos] - cursor[pos]
                for k in range(conv_len):
                    raw_conv[k] = np.int32(0)
                for i in range(d_child):
                    ci = np.int32(child[i])
                    if ci != 0:
                        raw_conv[2 * i] += ci * ci
                        for j in range(i + 1, d_child):
                            cj = np.int32(child[j])
                            if cj != 0:
                                raw_conv[i + j] += np.int32(2) * ci * cj
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

            # Not pruned: full recompute
            for pos in range(carry, d_parent):
                child[2 * pos] = cursor[pos]
                child[2 * pos + 1] = parent_int[pos] - cursor[pos]
            for k in range(conv_len):
                raw_conv[k] = np.int32(0)
            for i in range(d_child):
                ci = np.int32(child[i])
                if ci != 0:
                    raw_conv[2 * i] += ci * ci
                    for j in range(i + 1, d_child):
                        cj = np.int32(child[j])
                        if cj != 0:
                            raw_conv[i + j] += np.int32(2) * ci * cj
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

    return n_visited, n_deep_total, n_subtree_total


def main():
    print(f"Loading partial L3 survivors...")
    parents = load_partial_npy(CHECKPOINT_PATH, n_cols=D_PARENT, dtype=np.int32)
    n_total = len(parents)

    rng = np.random.default_rng(42)
    n_sample = 200
    indices = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    indices.sort()
    sample = parents[indices]
    del parents
    print(f"  {len(sample)} parents sampled from {n_total:,} readable rows")

    # Warmup JIT
    print("Warming up JIT...", flush=True)
    lo = np.zeros(D_PARENT, dtype=np.int32)
    hi = np.zeros(D_PARENT, dtype=np.int32)
    cc = np.zeros(D_PARENT, dtype=np.int64)
    cs = np.zeros(D_PARENT, dtype=np.int64)
    csk = np.zeros(D_PARENT, dtype=np.int64)
    parent = np.zeros(D_PARENT, dtype=np.int32)
    parent[0] = M
    _analyze_carry_depths(parent, N_HALF_CHILD, M, C_TARGET, lo, hi, cc, cs, csk)
    print("  Done.", flush=True)

    # Accumulate per-carry-depth stats
    total_carry_counts = np.zeros(D_PARENT, dtype=np.int64)
    total_carry_successes = np.zeros(D_PARENT, dtype=np.int64)
    total_carry_children_skipped = np.zeros(D_PARENT, dtype=np.int64)
    total_visited = 0
    total_cartesian = 0

    # Also collect per-parent radix info
    radix_info = []

    print(f"\nProcessing {len(sample)} parents...", flush=True)
    t0 = time.time()

    for i in range(len(sample)):
        parent = sample[i]
        result = _compute_bin_ranges(parent, M, C_TARGET, D_CHILD,
                                     n_half_child=N_HALF_CHILD)
        if result is None:
            continue

        lo_arr, hi_arr, total_children = result
        if total_children == 0:
            continue

        total_cartesian += total_children

        # Compute radices
        radices = []
        for j in range(D_PARENT):
            r = hi_arr[j] - lo_arr[j] + 1
            if r > 1:
                radices.append(r)
        radix_info.append(radices)

        cc = np.zeros(D_PARENT, dtype=np.int64)
        cs = np.zeros(D_PARENT, dtype=np.int64)
        csk = np.zeros(D_PARENT, dtype=np.int64)

        n_vis, n_deep, n_sub = _analyze_carry_depths(
            parent, N_HALF_CHILD, M, C_TARGET, lo_arr, hi_arr, cc, cs, csk)

        total_visited += n_vis
        total_carry_counts += cc
        total_carry_successes += cs
        total_carry_children_skipped += csk

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(sample)}] elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    # --- Report ---
    carry_threshold = D_PARENT // 4  # = 8

    print(f"{'='*80}")
    print(f"  CARRY DEPTH ANALYSIS ({len(sample)} parents)")
    print(f"{'='*80}")
    print(f"  Total Cartesian: {total_cartesian:,}")
    print(f"  Total visited:   {total_visited:,}")
    print(f"  Carry threshold: {carry_threshold} (n_changed > {carry_threshold} = deep)")
    print()

    # Per-carry-depth table
    print(f"  {'Carry':>5} {'Fixed':>6} {'Unfixed':>7} {'Checks':>10} "
          f"{'Success':>10} {'Rate':>7} {'Skipped':>14} {'% of Cart':>9}")
    print(f"  {'-'*74}")

    total_checks = 0
    total_success = 0
    total_skipped = 0

    for c in range(D_PARENT):
        if total_carry_counts[c] == 0:
            continue
        n_changed = D_PARENT - c
        if n_changed <= carry_threshold:
            continue

        fixed_bins = 2 * c
        unfixed_bins = D_CHILD - fixed_bins
        checks = total_carry_counts[c]
        successes = total_carry_successes[c]
        skipped = total_carry_children_skipped[c]
        rate = 100 * successes / max(1, checks)
        pct_cart = 100 * skipped / max(1, total_cartesian)

        total_checks += checks
        total_success += successes
        total_skipped += skipped

        print(f"  {c:>5} {fixed_bins:>6} {unfixed_bins:>7} {checks:>10,} "
              f"{successes:>10,} {rate:>6.1f}% {skipped:>14,} {pct_cart:>8.2f}%")

    print(f"  {'-'*74}")
    print(f"  {'TOTAL':>5} {'':>6} {'':>7} {total_checks:>10,} "
          f"{total_success:>10,} "
          f"{100*total_success/max(1,total_checks):>6.1f}% "
          f"{total_skipped:>14,} "
          f"{100*total_skipped/max(1,total_cartesian):>8.2f}%")

    # Radix distribution
    print(f"\n  Active position radix distribution:")
    all_radices = []
    n_active_list = []
    for radices in radix_info:
        all_radices.extend(radices)
        n_active_list.append(len(radices))

    if all_radices:
        arr = np.array(all_radices)
        print(f"    n_active per parent: mean={np.mean(n_active_list):.1f}, "
              f"min={np.min(n_active_list)}, max={np.max(n_active_list)}")
        print(f"    Radix values: mean={np.mean(arr):.1f}, median={np.median(arr):.0f}, "
              f"min={np.min(arr)}, max={np.max(arr)}")

        # Radix histogram
        for r in range(2, min(np.max(arr) + 2, 22)):
            count = np.sum(arr == r)
            if count > 0:
                pct = 100 * count / len(arr)
                print(f"      radix={r:>2}: {count:>5} positions ({pct:.1f}%)")

    # Estimate Gray code subtree pruning potential
    print(f"\n  Gray code subtree pruning potential (with reversed ordering):")
    print(f"  (Assumes similar success rates to odometer at comparable fixed-region sizes)")
    print()

    # For each j_min, estimate
    for j_min in [1, 2, 3, 4, 5]:
        # Estimate number of checks per parent and subtree sizes
        est_checks_total = 0
        est_skipped_total = 0
        est_subtree_sizes = []

        for radices in radix_info:
            n_active = len(radices)
            if j_min >= n_active:
                continue
            # Number of subtrees at this level
            subtree_size = 1
            for k in range(j_min):
                subtree_size *= radices[k]  # inner positions (reversed = rightmost)
            total_children_this = 1
            for r in radices:
                total_children_this *= r
            n_checks = total_children_this // max(1, subtree_size)
            est_checks_total += n_checks
            est_subtree_sizes.append(subtree_size)

            # Fixed child bins = 2 * (n_active - j_min) active + all inactive
            n_fixed_active = n_active - j_min
            n_fixed_total = n_fixed_active + (D_PARENT - n_active)  # + inactive
            fixed_child_bins = 2 * n_fixed_total

            # Estimate success rate based on fixed fraction
            fixed_frac = fixed_child_bins / D_CHILD
            # Interpolate from empirical data
            if fixed_frac >= 0.9:
                est_rate = 0.60
            elif fixed_frac >= 0.8:
                est_rate = 0.45
            elif fixed_frac >= 0.6:
                est_rate = 0.30
            elif fixed_frac >= 0.4:
                est_rate = 0.15
            else:
                est_rate = 0.05

            est_skipped_total += int(n_checks * est_rate * subtree_size)

        if est_checks_total > 0 and est_subtree_sizes:
            avg_subtree = np.mean(est_subtree_sizes)
            check_cost_us = est_checks_total * 1.5  # ~1.5μs per check
            skip_pct = 100 * est_skipped_total / max(1, total_cartesian)

            print(f"    j_min={j_min}: "
                  f"~{est_checks_total:,} checks, "
                  f"avg subtree={avg_subtree:.0f}, "
                  f"est skip={skip_pct:.1f}% of Cartesian, "
                  f"check overhead={check_cost_us/1e6:.3f}s")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
