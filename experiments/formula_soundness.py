"""
Exhaustive numerical comparison of Formula A vs Formula B pruning.

Formula A (proven): correction = (4n/ell)(1/m^2 + 2W/m)  per window
Formula B (published, unproven): correction = 1/m^2 + 2W/m  (no 4n/ell factor)

For each composition c, we check every window (ell, s_lo).
"Formula B only" = pruned by Formula B but NOT by Formula A.

For those compositions, we sample pre-images (continuous mu -> c via cumulative-floor)
and compute the actual continuous autoconvolution peak.
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


def autoconvolution(a):
    """Compute autoconvolution conv[k] = sum_{i+j=k} a_i * a_j."""
    d = len(a)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += a[i] * a[j]
    return conv


def test_value_per_window(c_int, n_half, m):
    """Return dict mapping (ell, s_lo) -> TV for all windows."""
    d = len(c_int)
    a = np.array(c_int, dtype=np.float64) * (4.0 * n_half / m)
    conv = autoconvolution(a)
    cumconv = np.cumsum(conv)
    conv_len = len(conv)

    result = {}
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = cumconv[s_hi]
            if s_lo > 0:
                ws -= cumconv[s_lo - 1]
            tv = ws / (4.0 * n_half * ell)
            result[(ell, s_lo)] = tv
    return result


def W_for_window(c_int, ell, s_lo, d):
    """Compute W_int = sum of c_i for bins overlapping window [s_lo, s_lo+ell-2] in conv space.

    The bins that contribute to conv indices [s_lo, s_lo+ell-2] are determined by:
    conv[k] = sum_{i+j=k} a_i*a_j, so for i+j in [s_lo, s_lo+ell-2],
    the bins i that participate satisfy: max(0, s_lo-(d-1)) <= i <= min(d-1, s_lo+ell-2).
    """
    lo_bin = max(0, s_lo - (d - 1))
    hi_bin = min(d - 1, s_lo + ell - 2)
    if hi_bin < lo_bin:
        return 0
    return sum(c_int[lo_bin:hi_bin + 1])


def check_formulas(c_int, n_half, m, c_target):
    """Check if Formula B prunes but Formula A does not, for any window.

    Returns list of (ell, s_lo) windows where B prunes but A doesn't.
    Also returns max TV across all windows.
    """
    d = len(c_int)
    tv_dict = test_value_per_window(c_int, n_half, m)

    b_only_windows = []
    max_tv = 0.0

    for (ell, s_lo), tv in tv_dict.items():
        if tv > max_tv:
            max_tv = tv

        W_int = W_for_window(c_int, ell, s_lo, d)
        W = W_int / m

        # Formula A correction (proven): (4n/ell)(1/m^2 + 2W/m)
        corr_A = (4.0 * n_half / ell) * (1.0 / (m * m) + 2.0 * W / m)

        # Formula B correction (published): 1/m^2 + 2W/m
        corr_B = 1.0 / (m * m) + 2.0 * W / m

        pruned_A = (tv > c_target + corr_A)
        pruned_B = (tv > c_target + corr_B)

        if pruned_B and not pruned_A:
            b_only_windows.append((ell, s_lo, tv, corr_A, corr_B, W_int))

    return b_only_windows, max_tv


def cumulative_floor_discretization(mu, m):
    """Given continuous mu (bin masses summing to 1), compute cumulative-floor discretization.

    M(k) = sum_{i<k} mu_i
    D(k) = floor(m * M(k))
    c_i = D(i+1) - D(i) for i < d-1
    c_{d-1} = m - D(d-1)
    """
    d = len(mu)
    M = np.zeros(d + 1)
    for k in range(1, d + 1):
        M[k] = M[k - 1] + mu[k - 1]
    D = np.floor(m * M).astype(int)
    c = np.zeros(d, dtype=int)
    for i in range(d - 1):
        c[i] = D[i + 1] - D[i]
    c[d - 1] = m - D[d - 1]
    return c


def sample_preimage(c_target_comp, m, n_samples):
    """Sample random mu vectors that map to c_target_comp under cumulative-floor.

    Strategy: for each mu, the cumulative-floor constraint defines intervals.
    We use rejection sampling with uniform random mu.
    """
    d = len(c_target_comp)
    c_target_comp = np.array(c_target_comp, dtype=int)

    # Compute the D values implied by c_target_comp
    D_target = np.zeros(d + 1, dtype=int)
    for i in range(d):
        D_target[i + 1] = D_target[i] + c_target_comp[i]

    # For the discretization to produce c, we need:
    # D(k) = floor(m * M(k)) = D_target[k] for all k
    # This means: D_target[k] <= m * M(k) < D_target[k] + 1
    # i.e., D_target[k]/m <= M(k) < (D_target[k]+1)/m
    # where M(k) = sum_{i<k} mu_i

    # We can sample M(k) uniformly in [D_target[k]/m, (D_target[k]+1)/m)
    # subject to M being non-decreasing and M(d) = 1.

    valid_mus = []
    attempts = 0
    max_attempts = n_samples * 100

    while len(valid_mus) < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample M(k) in [D_target[k]/m, (D_target[k]+1)/m) for k=1..d-1
        # M(0) = 0, M(d) = 1
        M = np.zeros(d + 1)
        M[d] = 1.0
        valid = True

        for k in range(1, d):
            lo = D_target[k] / m
            hi = (D_target[k] + 1) / m
            # Also must satisfy M[k-1] <= M[k] <= M[k+1]
            # We'll enforce this after
            M[k] = lo + np.random.random() * (hi - lo)

        # Check monotonicity
        for k in range(1, d + 1):
            if M[k] < M[k - 1]:
                valid = False
                break

        if not valid:
            continue

        # Verify discretization
        mu = np.diff(M)
        if np.any(mu < 0):
            continue

        c_check = cumulative_floor_discretization(mu, m)
        if np.array_equal(c_check, c_target_comp):
            valid_mus.append(mu)

    return valid_mus


def continuous_autoconv_peak(mu, n_half):
    """Compute ||f*f||_inf for piecewise-constant f with bin masses mu.

    f is piecewise constant on d equal bins of width 1/(2d) = 1/(4n_half).
    f(x) = mu_i * d on bin i (density = mass / bin_width, bin_width = 1/d where d=2*n_half).
    Actually, the support is [-1/4, 1/4], divided into d bins of width 1/(2d).
    Wait: d = 2*n_half bins spanning [-1/4, 1/4], so bin_width = 1/(2d).
    No: the support has length 1/2, divided into d bins, so bin_width = 1/(2d).
    Actually bin_width = (1/2)/d = 1/(2d).
    Density: f_i = mu_i / bin_width = mu_i * 2d.

    The autoconvolution (f*f)(t) = integral f(x)f(t-x)dx.
    For piecewise-constant f, this is a piecewise-linear function.
    Its peak occurs at one of the breakpoints.

    Actually, f*f is piecewise quadratic on intervals of length bin_width.
    For bins i,j, the contribution to (f*f)(t) is f_i * f_j * overlap(bin_i, bin_j shifted by t).

    Simpler: use numpy convolve on the densities, scaled by bin_width.
    (f*f)(t) = sum_k f_k * f_{t-k} * bin_width (discrete approx)

    For exact computation: f is piecewise constant with values rho_i = mu_i / bin_width.
    (f*f)(t) = integral rho(x) * rho(t-x) dx
    This is the convolution of a piecewise-constant function.
    The result is piecewise linear on intervals of width bin_width, with breakpoints at
    multiples of bin_width.

    At breakpoint t = k * bin_width (for integer k), the value is:
    (f*f)(t) = bin_width * sum_{i+j=k} rho_i * rho_j

    Between breakpoints, it's linear interpolation.
    Wait, that's only true if the bins are non-overlapping rectangles.
    Actually for box functions, convolution gives a piecewise LINEAR function
    (triangle from convolving two rectangles). For piecewise-constant with
    different heights, it's still piecewise linear with breakpoints at
    every bin_width multiple.

    The peak is at one of these breakpoints or could be in between (since it's
    piecewise linear, peak is at a breakpoint).
    """
    d = len(mu)
    bin_width = 0.5 / d  # = 1/(2d)
    rho = mu / bin_width  # density in each bin

    # Discrete convolution of densities, scaled by bin_width
    # (f*f)(k * bin_width) = bin_width * sum_{i+j=k} rho_i * rho_j
    # Actually that's the value at the breakpoints of the piecewise linear result.
    # The conv of two piecewise-constant functions on same grid:
    # result at center of overlap for lag k is bin_width * sum_{i} rho_i * rho_{k-i}

    conv = np.convolve(rho, rho) * bin_width
    # conv has length 2d-1, giving values at breakpoints
    # The maximum of a piecewise-linear function occurs at a breakpoint
    return np.max(conv)


def d_max_mu_sq_lower_bound(mu, n_half):
    """Lower bound on ||f*f||_inf from self-convolution terms.

    For any f with bin masses mu on d equal bins:
    (f*f)(0) >= sum_i integral f_i(x)^2 dx >= sum_i mu_i^2 / bin_width = d * sum mu_i^2 * 2d

    Actually, if f is uniform on each bin: f_i = mu_i / bin_width
    (f*f)(0) = integral f(x)^2 dx = sum_i f_i^2 * bin_width = sum_i mu_i^2 / bin_width
    bin_width = 1/(2d), so this = 2d * sum(mu_i^2).

    But f need not be uniform — we want a LOWER bound valid for ALL f with given mu.
    By Cauchy-Schwarz: integral f_i(x)^2 dx >= mu_i^2 / bin_width.
    So ||f*f||_inf >= (f*f)(0) >= sum_i mu_i^2 / bin_width = 2d * sum(mu_i^2).

    Wait, (f*f)(0) = integral f(x)^2 dx, and this is >= c_target is what we want
    to check. But (f*f)(t) for t=0 need not be the maximum.

    Actually (f*f)(0) = ||f||_2^2, and by the problem setup integral f = 1.
    By Cauchy-Schwarz: ||f||_2^2 >= ||f||_1^2 / |support| = 1 / (1/2) = 2.
    So ||f*f||_inf >= (f*f)(0) >= 2, which is always > c_target.

    Hmm, that's the global bound. But the window test is about restricted windows.

    Let me reconsider. The test value is max over windows of normalized sum.
    The continuous analog is max over intervals of (f*f)(t) integrated.

    For the soundness question: we need to check if there EXISTS a continuous f
    whose discretization is c AND whose ||f*f||_inf < c_target.

    Let's just compute ||f*f||_inf directly:
    - For the worst-case (lowest peak) pre-image, the piecewise-constant gives an upper bound
    - ||f||_2^2 gives a lower bound on (f*f)(0) which is a lower bound on ||f*f||_inf
    """
    d = len(mu)
    bin_width = 0.5 / d
    # ||f||_2^2 >= sum mu_i^2 / bin_width for any f with those bin masses
    return sum(mu[i]**2 for i in range(d)) / bin_width


def run_experiment(d, m, n_half, c_target, n_preimage_samples):
    """Run the full experiment for given parameters."""

    print(f"\n{'='*70}")
    print(f"d={d}, m={m}, n_half={n_half}, c_target={c_target}")
    print(f"{'='*70}")

    n_total = comb(m + d - 1, d - 1)
    print(f"Total compositions: {n_total}")

    # Formula A max correction (at ell=2): (4n/2)(1/m^2 + 2/m) = 2n(1/m^2 + 2/m)
    corr_A_max = (4.0 * n_half / 2) * (1.0 / (m * m) + 2.0 / m)
    # Formula B max correction (at W=m): 1/m^2 + 2/m  (but W depends on window)
    corr_B_base = 1.0 / (m * m) + 2.0 / m

    print(f"Formula A max correction (ell=2, W=0): {corr_A_max:.6f}")
    print(f"Formula B base correction (W=0): {corr_B_base:.6f}")
    print(f"Ratio A/B at ell=2: {corr_A_max / corr_B_base:.2f}")

    formula_b_only = []
    n_pruned_a = 0
    n_pruned_b = 0
    n_pruned_both = 0
    n_neither = 0

    t0 = time.time()
    for c_int in gen_compositions(d, m):
        c_arr = np.array(c_int)
        tv_dict = test_value_per_window(c_int, n_half, m)

        any_a = False
        any_b = False
        b_not_a = False

        for (ell, s_lo), tv in tv_dict.items():
            W_int = W_for_window(c_int, ell, s_lo, d)
            W = W_int / m

            corr_A = (4.0 * n_half / ell) * (1.0 / (m * m) + 2.0 * W / m)
            corr_B = 1.0 / (m * m) + 2.0 * W / m

            if tv > c_target + corr_A:
                any_a = True
            if tv > c_target + corr_B:
                any_b = True
            if (tv > c_target + corr_B) and not (tv > c_target + corr_A):
                b_not_a = True

        if any_a and any_b:
            n_pruned_both += 1
        elif any_a:
            n_pruned_a += 1
        elif any_b:
            n_pruned_b += 1  # pruned by B only (across all windows)
        else:
            n_neither += 1

        # Check if this composition is pruned by B (globally) but not by A (globally)
        globally_pruned_A = any_a
        globally_pruned_B = any_b

        if globally_pruned_B and not globally_pruned_A:
            max_tv = max(tv_dict.values())
            formula_b_only.append((c_int, max_tv))

    elapsed = time.time() - t0

    print(f"\nPruning summary ({elapsed:.1f}s):")
    print(f"  Pruned by both A and B: {n_pruned_both}")
    print(f"  Pruned by A only: {n_pruned_a}")
    print(f"  Pruned by B only (NOT by A): {n_pruned_b}")
    print(f"  Survived both: {n_neither}")
    print(f"  Total: {n_pruned_both + n_pruned_a + n_pruned_b + n_neither}")

    if not formula_b_only:
        print(f"\n  >>> NO compositions are 'Formula B only'. Formulas agree on all pruning decisions.")
        return []

    print(f"\n  >>> {len(formula_b_only)} compositions are 'Formula B only'")
    print(f"  Sampling {n_preimage_samples} pre-images each...")

    results = []
    for idx, (c_int, max_tv) in enumerate(formula_b_only):
        c_arr = np.array(c_int)

        mus = sample_preimage(c_arr, m, n_preimage_samples)

        if not mus:
            print(f"  Comp {c_int}: WARNING - could not find pre-images!")
            results.append({
                'comp': c_int,
                'max_tv': max_tv,
                'n_preimages': 0,
                'min_peak_piecewise': None,
                'max_peak_piecewise': None,
                'min_l2_lower': None,
            })
            continue

        peaks_pw = []  # piecewise-constant autoconvolution peaks
        l2_lowers = []  # ||f||_2^2 lower bounds

        for mu in mus:
            peak = continuous_autoconv_peak(mu, n_half)
            peaks_pw.append(peak)
            l2_lb = d_max_mu_sq_lower_bound(mu, n_half)
            l2_lowers.append(l2_lb)

        min_peak = min(peaks_pw)
        max_peak = max(peaks_pw)
        min_l2 = min(l2_lowers)

        results.append({
            'comp': c_int,
            'max_tv': max_tv,
            'n_preimages': len(mus),
            'min_peak_piecewise': min_peak,
            'max_peak_piecewise': max_peak,
            'min_l2_lower': min_l2,
        })

        status = "SAFE" if min_peak >= c_target else "*** UNSOUND ***"
        print(f"  [{idx+1}/{len(formula_b_only)}] c={c_int}, max_TV={max_tv:.6f}, "
              f"peak_range=[{min_peak:.6f}, {max_peak:.6f}], "
              f"L2_lower={min_l2:.6f}, {status}")

    return results


def main():
    np.random.seed(42)

    all_results = {}

    # Experiment 1: d=4, m=10, various c_target
    n_half = 2  # d = 2*n_half = 4
    for c_target in [1.1, 1.2, 1.28, 1.3, 1.4]:
        key = f"d4_m10_ct{c_target}"
        all_results[key] = run_experiment(d=4, m=10, n_half=n_half, c_target=c_target,
                                           n_preimage_samples=10000)

    # Experiment 2: d=4, m=20, c_target=1.28
    key = "d4_m20_ct1.28"
    all_results[key] = run_experiment(d=4, m=20, n_half=2, c_target=1.28,
                                       n_preimage_samples=5000)

    # Experiment 3: d=8, m=10, c_target=1.28 (L1 child dimension with n_half=2 -> d_child=8 at L1...
    # actually d=8 means n_half=4)
    # Wait: d=2*n_half. If we want d=8, n_half=4. But the user says "L1 child dimension"
    # which for n_half=2 is d_child=8 at L1 (d_parent=4, d_child=8).
    # At L1, n_half_child = 2*n_half_parent = 4. So n_half=4 for d=8.
    key = "d8_m10_ct1.28"
    all_results[key] = run_experiment(d=8, m=10, n_half=4, c_target=1.28,
                                       n_preimage_samples=1000)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'#B-only':>8} {'min peak (pw)':>14} {'min L2 lower':>14} {'Verdict':>12}")
    print("-" * 75)

    for key, results in all_results.items():
        n_b_only = len(results)
        if n_b_only == 0:
            print(f"{key:<25} {0:>8} {'N/A':>14} {'N/A':>14} {'AGREE':>12}")
        else:
            valid = [r for r in results if r['n_preimages'] > 0]
            if valid:
                min_peak = min(r['min_peak_piecewise'] for r in valid)
                min_l2 = min(r['min_l2_lower'] for r in valid)
                # Extract c_target from key
                ct = float(key.split('ct')[1])
                verdict = "SAFE" if min_peak >= ct else "UNSOUND?"
                print(f"{key:<25} {n_b_only:>8} {min_peak:>14.6f} {min_l2:>14.6f} {verdict:>12}")
            else:
                print(f"{key:<25} {n_b_only:>8} {'NO PREIMG':>14} {'N/A':>14} {'UNKNOWN':>12}")

    print(f"\n{'='*70}")
    print("KEY QUESTION ANSWER:")
    print("Is there any case where Formula B prunes a composition")
    print("and min ||f*f||_inf could plausibly be < c_target?")

    any_unsound = False
    for key, results in all_results.items():
        valid = [r for r in results if r['n_preimages'] > 0]
        if valid:
            ct = float(key.split('ct')[1])
            min_peak = min(r['min_peak_piecewise'] for r in valid)
            if min_peak < ct:
                any_unsound = True
                print(f"  >>> {key}: YES - min peak {min_peak:.6f} < c_target {ct}")

    if not any_unsound:
        print("  >>> NO unsound pruning found in any experiment.")
        print("  However, piecewise-constant f is an UPPER bound on the minimum peak.")
        print("  A smoother f within each bin could potentially achieve a lower peak.")

        # Check margins
        print(f"\n  Margins (min_peak - c_target) for 'Formula B only' compositions:")
        for key, results in all_results.items():
            valid = [r for r in results if r['n_preimages'] > 0]
            if valid:
                ct = float(key.split('ct')[1])
                min_peak = min(r['min_peak_piecewise'] for r in valid)
                margin = min_peak - ct
                print(f"    {key}: margin = {margin:.6f}")


if __name__ == "__main__":
    main()
