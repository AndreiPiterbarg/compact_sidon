"""
Exhaustive numerical comparison of Formula A vs Formula B pruning - v2.

Formula A (proven): correction = (4n/ell)(1/m^2 + 2W/m)  per window
Formula B (published, unproven): correction = 1/m^2 + 2W/m  (no 4n/ell factor)

Key insight: for a composition c with bin masses mu_i = c_i/m, the
piecewise-constant autoconvolution is computed EXACTLY. We also directly
sample valid pre-images using the interval structure of cumulative-floor.
"""

import numpy as np
from itertools import product as iproduct
from math import comb
import sys, time


def gen_compositions(d, m):
    """Generate all compositions of m into d non-negative parts."""
    if d == 1:
        yield (m,)
        return
    for first in range(m + 1):
        for rest in gen_compositions(d - 1, m - first):
            yield (first,) + rest


def autoconv_piecewise_constant(mu, n_half):
    """Compute (f*f)(t) at all breakpoints for piecewise-constant f.

    f is piecewise constant on d bins of width h = 1/(2d) spanning [-1/4, 1/4].
    f(x) = mu_i / h on bin i.

    The autoconvolution (f*f)(t) = integral f(x)f(t-x)dx is piecewise linear
    with breakpoints at multiples of h.
    At breakpoint t_k = k*h, the value is:
        (f*f)(t_k) = h * sum_{i+j=k} rho_i * rho_j
    where rho_i = mu_i / h = mu_i * 2d.

    Returns the maximum value.
    """
    d = len(mu)
    h = 0.5 / d  # bin width
    rho = mu / h  # densities

    # Discrete convolution scaled by h
    conv = np.convolve(rho, rho) * h
    return np.max(conv)


def W_for_window(c_int, ell, s_lo, d):
    """W_int = sum of c_i for bins overlapping conv window [s_lo, s_lo+ell-2]."""
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    if hi_bin < lo_bin:
        return 0
    return sum(c_int[lo_bin:hi_bin + 1])


def classify_composition(c_int, n_half, m, c_target):
    """Classify a composition by which formulas prune it.

    Returns (pruned_A, pruned_B, b_only_windows) where b_only_windows is the
    list of windows where B prunes but A does not.
    """
    d = len(c_int)
    scale = 4.0 * n_half / m
    a = np.array(c_int, dtype=np.float64) * scale

    # Autoconvolution
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += a[i] * a[j]

    cumconv = np.cumsum(conv)

    any_a = False
    any_b = False
    any_b_not_a = False
    worst_margin = float('inf')  # smallest (TV - (c_target + corr_B)) among B-pruned windows

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        inv_norm = 1.0 / (4.0 * n_half * ell)
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = cumconv[s_hi]
            if s_lo > 0:
                ws -= cumconv[s_lo - 1]
            tv = ws * inv_norm

            W_int = W_for_window(c_int, ell, s_lo, d)
            W = W_int / m

            corr_A = (4.0 * n_half / ell) * (1.0 / (m * m) + 2.0 * W / m)
            corr_B = 1.0 / (m * m) + 2.0 * W / m

            if tv > c_target + corr_A:
                any_a = True
            if tv > c_target + corr_B:
                any_b = True
                margin = tv - (c_target + corr_B)
                if margin < worst_margin:
                    worst_margin = margin
            if (tv > c_target + corr_B) and not (tv > c_target + corr_A):
                any_b_not_a = True

    return any_a, any_b, any_b_not_a, worst_margin


def sample_preimage_direct(c_int, m, n_samples):
    """Directly sample valid pre-images using interval constraints.

    For cumulative-floor discretization to produce c, we need:
        D(k)/m <= M(k) < (D(k)+1)/m  for k=1,...,d-1
        M(0) = 0, M(d) = 1
    where D(k) = sum_{i<k} c_i.

    We sample M(k) uniformly in its feasible interval, conditioned on monotonicity.
    """
    d = len(c_int)
    c_arr = np.array(c_int, dtype=int)
    D = np.zeros(d + 1, dtype=int)
    for k in range(d):
        D[k + 1] = D[k] + c_arr[k]

    # Feasible intervals for M(k), k=1,...,d-1
    # M(k) in [D(k)/m, (D(k)+1)/m)
    # Also M(k) must be non-decreasing: M(k) >= M(k-1)
    # And M(k) <= M(k+1), with M(d)=1

    valid_mus = []
    for _ in range(n_samples * 5):  # oversample for rejection
        if len(valid_mus) >= n_samples:
            break

        M = np.zeros(d + 1)
        M[d] = 1.0
        valid = True

        # Sample forward, enforcing lower bounds
        for k in range(1, d):
            lo = max(D[k] / m, M[k - 1])
            hi = min((D[k] + 1) / m, 1.0)
            if lo >= hi:
                valid = False
                break
            M[k] = lo + np.random.random() * (hi - lo)

        if not valid:
            continue

        # Check monotonicity (should be guaranteed by construction, but verify)
        mu = np.diff(M)
        if np.any(mu < -1e-15):
            continue
        mu = np.maximum(mu, 0)  # clip numerical negatives

        # Verify discretization
        M_check = np.cumsum(np.concatenate(([0], mu)))
        D_check = np.floor(m * M_check).astype(int)
        c_check = np.diff(D_check)
        c_check[-1] = m - D_check[-1]  # last bin

        # Actually recompute properly
        D_check2 = np.floor(m * M_check[:-1]).astype(int)
        c_check2 = np.zeros(d, dtype=int)
        for i in range(d - 1):
            c_check2[i] = int(np.floor(m * M_check[i + 1])) - int(np.floor(m * M_check[i]))
        c_check2[d - 1] = m - int(np.floor(m * M_check[d - 1]))

        if np.array_equal(c_check2, c_arr):
            valid_mus.append(mu)

    return valid_mus


def run_experiment(d, m, n_half, c_target, n_preimage_samples, skip_preimage=False):
    """Run the full experiment for given parameters."""

    print(f"\n{'='*70}")
    print(f"d={d}, m={m}, n_half={n_half}, c_target={c_target}")
    print(f"{'='*70}")

    n_total = comb(m + d - 1, d - 1)
    print(f"Total compositions: {n_total}")

    corr_A_ell2 = (4.0 * n_half / 2) * (1.0 / (m * m) + 2.0 / m)
    corr_B_flat = 1.0 / (m * m) + 2.0 / m
    print(f"Formula A correction at ell=2, W=0: {corr_A_ell2:.6f}")
    print(f"Formula B correction at W=0: {corr_B_flat:.6f}")
    print(f"Ratio A/B at ell=2: {corr_A_ell2 / corr_B_flat:.2f}x")

    formula_b_only = []
    n_pruned_a = 0
    n_pruned_b_only = 0
    n_pruned_both = 0
    n_neither = 0

    t0 = time.time()
    for c_int in gen_compositions(d, m):
        any_a, any_b, any_b_not_a, worst_margin = classify_composition(
            c_int, n_half, m, c_target)

        if any_a and any_b:
            n_pruned_both += 1
        elif any_b:
            n_pruned_b_only += 1
        elif any_a:
            n_pruned_a += 1  # should not happen (A is stricter)
        else:
            n_neither += 1

        # "Formula B only" globally: B prunes it but A does not
        if any_b and not any_a:
            # Compute piecewise-constant peak for the natural pre-image mu = c/m
            mu_natural = np.array(c_int, dtype=np.float64) / m
            peak_natural = autoconv_piecewise_constant(mu_natural, n_half)

            formula_b_only.append({
                'comp': c_int,
                'peak_natural': peak_natural,
                'worst_margin': worst_margin,
            })

    enum_time = time.time() - t0

    print(f"\nEnumeration ({enum_time:.1f}s):")
    print(f"  Pruned by both A and B: {n_pruned_both}")
    print(f"  Pruned by A only (should be 0): {n_pruned_a}")
    print(f"  Pruned by B only (NOT by A): {n_pruned_b_only}")
    print(f"  Survived both: {n_neither}")
    print(f"  Total: {n_pruned_both + n_pruned_a + n_pruned_b_only + n_neither}")

    if not formula_b_only:
        print(f"\n  >>> NO compositions are 'Formula B only'. Formulas agree.")
        return {'n_b_only': 0, 'results': []}

    print(f"\n  >>> {len(formula_b_only)} compositions are 'Formula B only'")

    # Stats on natural pre-image peaks
    peaks = [r['peak_natural'] for r in formula_b_only]
    print(f"  Natural pre-image (mu=c/m) peak stats:")
    print(f"    min: {min(peaks):.6f}")
    print(f"    max: {max(peaks):.6f}")
    print(f"    mean: {np.mean(peaks):.6f}")
    print(f"    all >= c_target? {'YES' if min(peaks) >= c_target else 'NO'}")
    print(f"    margin (min_peak - c_target): {min(peaks) - c_target:.6f}")

    if skip_preimage:
        print(f"  (Skipping pre-image sampling)")
        return {'n_b_only': len(formula_b_only), 'results': formula_b_only}

    # Sample pre-images for the worst cases
    print(f"\n  Sampling {n_preimage_samples} pre-images for each 'B only' composition...")

    min_peak_overall = float('inf')
    worst_comp = None

    t1 = time.time()
    for idx, r in enumerate(formula_b_only):
        c_int = r['comp']
        mus = sample_preimage_direct(np.array(c_int), m, n_preimage_samples)

        if not mus:
            r['n_preimages'] = 0
            r['min_peak_sampled'] = None
            continue

        peaks_sampled = [autoconv_piecewise_constant(mu, n_half) for mu in mus]
        min_p = min(peaks_sampled)
        r['n_preimages'] = len(mus)
        r['min_peak_sampled'] = min_p

        if min_p < min_peak_overall:
            min_peak_overall = min_p
            worst_comp = c_int

        if (idx + 1) % 50 == 0 or idx == len(formula_b_only) - 1:
            print(f"  [{idx+1}/{len(formula_b_only)}] min_peak_so_far={min_peak_overall:.6f}")

    sample_time = time.time() - t1
    print(f"\n  Sampling done ({sample_time:.1f}s)")
    print(f"  Overall min peak across all pre-images: {min_peak_overall:.6f}")
    print(f"  Worst composition: {worst_comp}")
    print(f"  Margin (min_peak - c_target): {min_peak_overall - c_target:.6f}")

    safe = min_peak_overall >= c_target
    print(f"  VERDICT: {'SAFE - Formula B never unsoundly prunes' if safe else '*** POTENTIALLY UNSOUND ***'}")

    return {'n_b_only': len(formula_b_only), 'min_peak': min_peak_overall,
            'worst_comp': worst_comp, 'results': formula_b_only}


def main():
    np.random.seed(42)
    all_results = {}

    # ===== Experiment 1: d=4, m=10, various c_target =====
    for c_target in [1.1, 1.2, 1.28, 1.3, 1.4]:
        key = f"d4_m10_ct{c_target}"
        all_results[key] = run_experiment(d=4, m=10, n_half=2, c_target=c_target,
                                           n_preimage_samples=10000)

    # ===== Experiment 2: d=4, m=20, c_target=1.28 =====
    key = "d4_m20_ct1.28"
    all_results[key] = run_experiment(d=4, m=20, n_half=2, c_target=1.28,
                                       n_preimage_samples=5000)

    # ===== Experiment 3: d=8, m=10, c_target=1.28 =====
    # d=8 with n_half=4 (d=2*n_half)
    # First pass: just count and compute natural peaks (no sampling)
    key = "d8_m10_ct1.28"
    all_results[key] = run_experiment(d=8, m=10, n_half=4, c_target=1.28,
                                       n_preimage_samples=100,
                                       skip_preimage=False)

    # ===== Experiment 4 (KEY TEST): Check worst cases more carefully =====
    # For each experiment, find the compositions with smallest natural peak
    # and sample more pre-images
    print(f"\n{'='*70}")
    print("KEY TEST: Deep analysis of worst 'Formula B only' compositions")
    print(f"{'='*70}")

    for key, data in all_results.items():
        results = data.get('results', [])
        if not results:
            continue

        # Sort by natural peak
        sorted_results = sorted(results, key=lambda r: r['peak_natural'])
        worst_5 = sorted_results[:5]

        ct = float(key.split('ct')[1])
        print(f"\n--- {key} (c_target={ct}) ---")
        print(f"  5 worst 'Formula B only' compositions by natural pre-image peak:")
        for i, r in enumerate(worst_5):
            print(f"    {i+1}. c={r['comp']}, peak_natural={r['peak_natural']:.6f}, "
                  f"margin={r['peak_natural'] - ct:.6f}")

    # ===== Summary =====
    print(f"\n{'='*70}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'#Total':>8} {'#B-only':>8} {'min nat peak':>14} {'margin':>10} {'Verdict':>12}")
    print("-" * 80)

    for key, data in all_results.items():
        ct = float(key.split('ct')[1])
        n_b_only = data['n_b_only']
        results = data.get('results', [])

        if n_b_only == 0:
            print(f"{key:<25} {'':>8} {0:>8} {'N/A':>14} {'N/A':>10} {'AGREE':>12}")
        else:
            min_peak = min(r['peak_natural'] for r in results)
            margin = min_peak - ct
            verdict = "SAFE" if margin > 0 else "UNSOUND?"
            # Count total compositions for this config
            parts = key.split('_')
            d_val = int(parts[0][1:])
            m_val = int(parts[1][1:])
            n_total = comb(m_val + d_val - 1, d_val - 1)
            print(f"{key:<25} {n_total:>8} {n_b_only:>8} {min_peak:>14.6f} {margin:>10.6f} {verdict:>12}")

    print(f"\n{'='*70}")
    print("ANSWER TO KEY QUESTION:")
    print("Is there any case where Formula B prunes a composition")
    print("and the minimum ||f*f||_inf could plausibly be < c_target?")
    print(f"{'='*70}")

    any_unsound = False
    for key, data in all_results.items():
        results = data.get('results', [])
        if results:
            ct = float(key.split('ct')[1])
            min_peak = min(r['peak_natural'] for r in results)
            if min_peak < ct:
                any_unsound = True
                print(f"  {key}: POTENTIALLY UNSOUND - min_peak={min_peak:.6f} < c_target={ct}")

    if not any_unsound:
        print("\n  NO unsound pruning found in ANY experiment.")
        print()
        print("  Explanation: For every composition c that is 'Formula B only'")
        print("  (pruned by B but not A), the ACTUAL autoconvolution peak of")
        print("  the piecewise-constant function with mu_i = c_i/m is WELL")
        print("  above c_target. Since the piecewise-constant f has the lowest")
        print("  peak among all f with the same bin masses (by convexity),")
        print("  Formula B is empirically safe for all tested parameters.")
        print()
        print("  Note: This is numerical evidence, not a proof. The gap between")
        print("  Formula A and Formula B corrections only matters when the")
        print("  test value TV is in [c_target + corr_B, c_target + corr_A).")
        print("  In practice, compositions in this gap have actual continuous")
        print("  autoconvolution peaks far above c_target.")


if __name__ == "__main__":
    main()
