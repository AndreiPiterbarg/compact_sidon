"""Exhaustive correctness test for CPU pruner (_prune_dynamic).

For d=4, m=20, c_target=1.3, this script:
  1. Generates ALL canonical d=4 compositions (c <= rev(c) lex) summing to 20
  2. Computes exact brute-force test values per window
  3. Simulates _prune_dynamic in pure Python (integer-space, no Numba)
  4. Checks for over-pruning and missed prunes
  5. Reports full agreement/disagreement statistics

KEY CORRECTNESS INVARIANT:
  The pruner uses per-window DYNAMIC thresholds:
    threshold(ell, s_lo) = c_target + 1/m^2 + 2*W_int/m^2
  where W_int = sum of c_i for bins contributing to window (ell, s_lo).

  A composition is correctly prunable if there EXISTS a window where
  its test value exceeds that window's dynamic threshold.

  The conservative uniform threshold (c_target + 2/m + 1/m^2) is an
  upper bound on the dynamic threshold (attained when W_int = m, i.e.,
  all bins contribute). Compositions pruned with max_tv below the
  conservative threshold are NOT bugs -- they are correctly pruned by
  a tighter per-window threshold.

  The REAL correctness check is:
    No composition is pruned unless tv > dynamic_threshold for some window.
"""
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
D = 4
N_HALF = D // 2   # = 2
M = 20
C_TARGET = 1.3
DBL_EPS = 2.220446049250313e-16

# ---------------------------------------------------------------------------
# Step 1: Generate ALL canonical d=4, m=20 compositions
# ---------------------------------------------------------------------------

def generate_all_canonical_d4(S):
    """Generate all (c0, c1, c2, c3) with sum=S, c <= rev(c) lex."""
    comps = []
    for c0 in range(S // 2 + 1):
        r0 = S - c0
        c1_max = r0 - c0  # = S - 2*c0
        for c1 in range(c1_max + 1):
            r1 = r0 - c1
            c2_max = r1 - c0  # = S - 2*c0 - c1
            for c2 in range(c2_max + 1):
                c3 = r1 - c2
                # canonical: c0 < c3, or (c0 == c3 and c1 <= c2)
                if c0 < c3:
                    comps.append((c0, c1, c2, c3))
                elif c0 == c3 and c1 <= c2:
                    comps.append((c0, c1, c2, c3))
                # else: skip non-canonical
    return comps


# ---------------------------------------------------------------------------
# Step 2: Brute-force exact test values (floating point, per window)
# ---------------------------------------------------------------------------

def compute_exact_conv(c_vec, n_half, m):
    """Compute autoconvolution in a-coordinates.
    a_i = 4 * n_half * c_i / m
    conv[k] = sum_{i+j=k} a_i * a_j
    """
    d = len(c_vec)
    conv_len = 2 * d - 1
    scale = 4.0 * n_half / m
    a = [c_vec[i] * scale for i in range(d)]
    conv = [0.0] * conv_len
    for i in range(d):
        for j in range(d):
            conv[i + j] += a[i] * a[j]
    return conv, a


def compute_all_window_test_values(c_vec, n_half, m):
    """Compute test value for every valid window (ell, s_lo).
    Returns dict: (ell, s_lo) -> tv
    """
    d = len(c_vec)
    conv_len = 2 * d - 1
    conv, a = compute_exact_conv(c_vec, n_half, m)

    cumconv = [0.0] * conv_len
    cumconv[0] = conv[0]
    for k in range(1, conv_len):
        cumconv[k] = cumconv[k-1] + conv[k]

    tvs = {}
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = cumconv[s_hi]
            if s_lo > 0:
                ws -= cumconv[s_lo - 1]
            tv = ws / (4.0 * n_half * ell)
            tvs[(ell, s_lo)] = tv
    return tvs


def compute_max_test_value(c_vec, n_half, m):
    """Compute the maximum test value over all windows."""
    tvs = compute_all_window_test_values(c_vec, n_half, m)
    return max(tvs.values())


# ---------------------------------------------------------------------------
# Step 3: Pure-Python simulation of _prune_dynamic (integer space)
# ---------------------------------------------------------------------------

def simulate_prune_dynamic(c_vec, n_half, m, c_target):
    """Simulate _prune_dynamic exactly as the Numba code does.

    Returns (pruned: bool, details: dict with per-window info)
    """
    d = len(c_vec)
    conv_len = 2 * d - 1

    m_d = float(m)
    dyn_base = c_target * m_d * m_d + 1.0 + 1e-9 * m_d * m_d
    inv_4n = 1.0 / (4.0 * float(n_half))

    # Integer autoconvolution (exactly as in _prune_dynamic)
    conv = [0] * conv_len
    for i in range(d):
        ci = int(c_vec[i])
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            conv[i + j] += 2 * ci * int(c_vec[j])

    # Prefix sum
    for k in range(1, conv_len):
        conv[k] += conv[k - 1]

    # Prefix sum of c (for W_int)
    prefix_c = [0] * (d + 1)
    for i in range(d):
        prefix_c[i + 1] = prefix_c[i] + int(c_vec[i])

    # Window scan
    pruned = False
    pruning_window = None
    window_details = {}

    for ell in range(2, 2 * d + 1):
        if pruned:
            break
        n_cv = ell - 1
        dyn_base_ell = dyn_base * float(ell) * inv_4n
        two_ell_inv_4n = 2.0 * float(ell) * inv_4n
        n_windows = conv_len - n_cv + 1

        for s_lo in range(n_windows):
            s_hi = s_lo + n_cv - 1
            ws = conv[s_hi]
            if s_lo > 0:
                ws -= conv[s_lo - 1]

            # W_int computation
            lo_bin = s_lo - (d - 1)
            if lo_bin < 0:
                lo_bin = 0
            hi_bin = s_lo + ell - 2
            if hi_bin > d - 1:
                hi_bin = d - 1
            W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]

            dyn_x = dyn_base_ell + two_ell_inv_4n * float(W_int)
            dyn_it = int(dyn_x * (1.0 - 4.0 * DBL_EPS))

            window_details[(ell, s_lo)] = {
                'ws_int': ws,
                'W_int': W_int,
                'dyn_it': dyn_it,
                'dyn_x': dyn_x,
                'pruned': ws > dyn_it,
            }

            if ws > dyn_it:
                pruned = True
                pruning_window = (ell, s_lo)
                break

    return pruned, pruning_window, window_details


# ---------------------------------------------------------------------------
# Step 4: Compute dynamic thresholds in continuous space for reference
# ---------------------------------------------------------------------------

def compute_dynamic_threshold_continuous(c_vec, n_half, m, c_target, ell, s_lo):
    """Compute the dynamic threshold for a specific window in continuous space.

    threshold = c_target + 1/m^2 + 2*W_int/m^2

    where W_int = sum of c_i for bins i in [lo_bin, hi_bin].
    """
    d = len(c_vec)
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    W_int = sum(int(c_vec[i]) for i in range(lo_bin, hi_bin + 1))

    threshold = c_target + 1.0 / (m * m) + 2.0 * W_int / (m * m)
    return threshold, W_int


# ---------------------------------------------------------------------------
# Step 5: Also run the actual _prune_dynamic from the codebase
# ---------------------------------------------------------------------------

def run_actual_pruner(comps_array, n_half, m, c_target):
    """Run the actual Numba _prune_dynamic on all compositions."""
    # Import from the codebase
    cs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'cloninger-steinerberger')
    cpu_dir = os.path.join(cs_dir, 'cpu')
    if cs_dir not in sys.path:
        sys.path.insert(0, cs_dir)
    if cpu_dir not in sys.path:
        sys.path.insert(0, cpu_dir)

    from run_cascade import _prune_dynamic

    batch = np.array(comps_array, dtype=np.int32)
    survived = _prune_dynamic(batch, n_half, m, c_target)
    return survived


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXHAUSTIVE CORRECTNESS TEST: CPU pruner vs brute-force reference")
    print(f"  d={D}, n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print("=" * 70)

    # Step 1: Generate all canonical compositions
    print("\n[1] Generating all canonical d=4, m=20 compositions...")
    comps = generate_all_canonical_d4(M)
    print(f"    Found {len(comps)} canonical compositions")

    # Verify a few known properties
    for c in comps:
        assert sum(c) == M, f"Sum mismatch: {c} sums to {sum(c)}"
        assert all(x >= 0 for x in c), f"Negative entry: {c}"
        # Check canonical: c <= rev(c) lex
        rev_c = c[::-1]
        assert c <= rev_c, f"Not canonical: {c} > {rev_c}"
    print(f"    All {len(comps)} pass basic validation (sum, non-negative, canonical)")

    # Step 2: Compute brute-force test values
    print("\n[2] Computing brute-force test values for all compositions...")
    max_tvs = {}
    all_window_tvs = {}
    for c in comps:
        max_tvs[c] = compute_max_test_value(c, N_HALF, M)
        all_window_tvs[c] = compute_all_window_test_values(c, N_HALF, M)

    conservative_threshold = C_TARGET + 2.0 / M + 1.0 / (M * M)
    print(f"    Conservative threshold (c_target + 2/m + 1/m^2) = {conservative_threshold:.10f}")

    n_below_cons = sum(1 for tv in max_tvs.values() if tv <= conservative_threshold)
    n_above_cons = sum(1 for tv in max_tvs.values() if tv > conservative_threshold)
    print(f"    max_tv <= conservative_threshold: {n_below_cons}")
    print(f"    max_tv >  conservative_threshold: {n_above_cons}")

    # Step 3: Simulate _prune_dynamic in pure Python
    print("\n[3] Simulating _prune_dynamic in pure Python (integer space)...")
    sim_results = {}
    for c in comps:
        pruned, pruning_window, details = simulate_prune_dynamic(c, N_HALF, M, C_TARGET)
        sim_results[c] = {
            'pruned': pruned,
            'pruning_window': pruning_window,
            'details': details,
        }

    n_sim_pruned = sum(1 for r in sim_results.values() if r['pruned'])
    n_sim_survived = sum(1 for r in sim_results.values() if not r['pruned'])
    print(f"    Simulation pruned: {n_sim_pruned}")
    print(f"    Simulation survived: {n_sim_survived}")

    # Step 4: Run the actual Numba _prune_dynamic
    print("\n[4] Running actual Numba _prune_dynamic from codebase...")
    comps_list = list(comps)
    survived_mask = run_actual_pruner(comps_list, N_HALF, M, C_TARGET)

    n_actual_pruned = int(np.sum(~survived_mask))
    n_actual_survived = int(np.sum(survived_mask))
    print(f"    Actual pruned: {n_actual_pruned}")
    print(f"    Actual survived: {n_actual_survived}")

    # Step 5: Compare all three: brute-force, simulation, actual
    print("\n[5] Cross-validation...")
    print("-" * 70)

    sim_vs_actual_mismatch = []  # Simulation disagrees with actual
    over_prune_dynamic = []   # CRITICAL: Pruned but tv < dynamic threshold for ALL windows
    below_conservative = []   # INFO: Pruned with max_tv < conservative (but above dynamic)
    missed_prune = []         # Not pruned but tv > dynamic threshold for some window

    for idx, c in enumerate(comps_list):
        actual_survived = bool(survived_mask[idx])
        sim_pruned = sim_results[c]['pruned']
        actual_pruned = not actual_survived
        max_tv = max_tvs[c]

        # Check simulation vs actual agreement
        if sim_pruned != actual_pruned:
            sim_vs_actual_mismatch.append({
                'comp': c,
                'sim_pruned': sim_pruned,
                'actual_pruned': actual_pruned,
                'max_tv': max_tv,
            })

        # Check dynamic threshold consistency per-window
        # A composition is correctly pruned if there EXISTS a window where
        # tv > dynamic_threshold for that window
        has_violating_window = False
        violating_details = None
        window_tvs = all_window_tvs[c]
        for (ell, s_lo), tv in window_tvs.items():
            dyn_thresh, W_int = compute_dynamic_threshold_continuous(
                c, N_HALF, M, C_TARGET, ell, s_lo)
            if tv > dyn_thresh:
                has_violating_window = True
                violating_details = (ell, s_lo, tv, dyn_thresh, W_int)
                break

        # CRITICAL: Pruned but NO window exceeds its dynamic threshold
        # This would mean the integer-space rounding is too aggressive
        if actual_pruned and not has_violating_window:
            over_prune_dynamic.append({
                'comp': c,
                'max_tv': max_tv,
                'pruning_window': sim_results[c]['pruning_window'],
            })

        # INFO: Pruned with max_tv below conservative threshold
        # (correctly pruned by tighter per-window dynamic threshold)
        if actual_pruned and max_tv < conservative_threshold:
            below_conservative.append({
                'comp': c,
                'max_tv': max_tv,
                'violating_window': violating_details,
            })

        if not actual_pruned and has_violating_window:
            missed_prune.append({
                'comp': c,
                'max_tv': max_tv,
            })

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # 5a: Simulation vs Actual agreement
    if sim_vs_actual_mismatch:
        print(f"\n*** FAIL: SIMULATION vs ACTUAL MISMATCH: "
              f"{len(sim_vs_actual_mismatch)} ***")
        for item in sim_vs_actual_mismatch[:10]:
            print(f"    comp={item['comp']}, sim_pruned={item['sim_pruned']}, "
                  f"actual_pruned={item['actual_pruned']}, "
                  f"max_tv={item['max_tv']:.10f}")
    else:
        print(f"\n[PASS] Simulation and actual Numba code agree on all "
              f"{len(comps)} compositions.")

    # 5b: CRITICAL — dynamic over-pruning (integer rounding too aggressive?)
    if over_prune_dynamic:
        print(f"\n*** CRITICAL BUG: {len(over_prune_dynamic)} compositions pruned "
              f"by integer code but NOT by continuous dynamic threshold ***")
        print(f"    The integer-space rounding is too aggressive!")
        print(f"    These compositions have tv <= dynamic_threshold for ALL "
              f"windows, yet were pruned.")
        for item in over_prune_dynamic[:10]:
            print(f"    comp={item['comp']}, max_tv={item['max_tv']:.10f}, "
                  f"pruning_window={item['pruning_window']}")
    else:
        print(f"\n[PASS] No over-pruning: every pruned composition has "
              f"tv > dynamic_threshold for at least one window.")

    # 5c: INFO — pruned below conservative (uniform) threshold
    #     This is expected and correct when the dynamic threshold is tighter.
    if below_conservative:
        print(f"\n[INFO] {len(below_conservative)} compositions pruned with "
              f"max_tv < conservative threshold ({conservative_threshold:.10f}).")
        print(f"    This is CORRECT: they exceed their per-window dynamic "
              f"threshold (which is tighter than the uniform bound).")
        for item in below_conservative[:10]:
            vw = item['violating_window']
            if vw:
                ell, s_lo, tv, dyn_thresh, W_int = vw
                print(f"    comp={item['comp']}, max_tv={item['max_tv']:.10f}, "
                      f"pruned at (ell={ell},s_lo={s_lo}): "
                      f"tv={tv:.10f} > dyn_thresh={dyn_thresh:.10f} "
                      f"(W_int={W_int})")
            else:
                print(f"    comp={item['comp']}, max_tv={item['max_tv']:.10f}")
    else:
        print(f"\n[INFO] All pruned compositions also exceed the conservative "
              f"threshold ({conservative_threshold:.10f}).")

    # 5d: Missed prunes (not a correctness bug, just waste)
    if missed_prune:
        print(f"\n[INFO] {len(missed_prune)} compositions NOT pruned by CPU code "
              f"but have tv > dynamic_threshold for some window in continuous "
              f"space.")
        print(f"    (Not a bug -- integer rounding is conservative, which is "
              f"correct.)")
        for item in missed_prune[:5]:
            print(f"    comp={item['comp']}, max_tv={item['max_tv']:.10f}")
    else:
        print(f"\n[PASS] No missed prunes (integer code prunes everything that "
              f"continuous thresholds would).")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total canonical compositions: {len(comps)}")
    print(f"  Pruned (actual Numba):        {n_actual_pruned}")
    print(f"  Survived (actual Numba):      {n_actual_survived}")
    print(f"  Pruned (simulation):          {n_sim_pruned}")
    print(f"  Survived (simulation):        {n_sim_survived}")
    print(f"  max_tv > conservative thr:    {n_above_cons}")
    print(f"  max_tv <= conservative thr:   {n_below_cons}")
    print(f"")
    print(f"  Sim vs Actual mismatches:     {len(sim_vs_actual_mismatch)}")
    print(f"  CRITICAL dynamic over-prune:  {len(over_prune_dynamic)}")
    print(f"  Below-conservative (correct): {len(below_conservative)}")
    print(f"  Missed prunes (waste):        {len(missed_prune)}")

    # Print survivors for inspection
    if n_actual_survived > 0 and n_actual_survived <= 50:
        print(f"\n  Surviving compositions ({n_actual_survived}):")
        for idx, c in enumerate(comps_list):
            if survived_mask[idx]:
                tv = max_tvs[c]
                print(f"    {c}  max_tv={tv:.10f}")

    # Final verdict
    print("\n" + "=" * 70)
    if over_prune_dynamic:
        print("VERDICT: *** FAIL *** -- Over-pruning detected!")
        print("  Compositions pruned whose continuous test value is below the")
        print("  continuous dynamic threshold for ALL windows.")
        return 1
    elif sim_vs_actual_mismatch:
        print("VERDICT: *** FAIL *** -- Simulation/actual mismatch!")
        return 1
    else:
        print("VERDICT: PASS -- CPU pruner is correct.")
        print("  Every pruned composition has tv > dynamic_threshold for at")
        print("  least one window. No over-pruning detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
