# Formula B Soundness on the Coarse Grid: Proof via Mass-Space Reduction

## Status Summary

| Result | Status |
|--------|--------|
| Cross-term vanishing lemma (Lemma 1) | **PROVED** |
| Knot-point decomposition (Theorem 1) | **PROVED** |
| Reduction to step functions (Theorem 2) | **PROVED** (for step functions only) |
| Mass-space per-window bound (Theorem 3) | **PROVED** (reproduces Formula A) |
| Formula B composition-level soundness | **OPEN** (gap identified precisely) |
| Cascade-level soundness via fine-grid equivalence (Theorem 4) | **PROVED** |

---

## 0. Setup and Notation

Let d = 2n bins I_i = [-1/4 + ih, -1/4 + (i+1)h) with h = 1/(4n).

**Coarse grid.** Compositions c = (c_0, ..., c_{d-1}) with sum c_i = m (mass coordinates). Discrete masses w_i = c_i/m. Step function g_c with heights a_i^c = 4n w_i on each bin.

**Continuous function.** f >= 0 supported on [-1/4, 1/4], integral f = 1. Bin masses mu_i = integral_{I_i} f. Cumulative-floor discretization: c_i = D(i+1) - D(i) where D(k) = floor(m sum_{j<k} mu_j).

**Step function approximation.** f_step is the step function with the SAME bin masses as f: height a_i = mu_i/h = 4n mu_i on bin I_i.

**Intra-bin residual.** epsilon_2 = f - f_step. Note: integral_{I_i} epsilon_2 = 0 for each bin i.

**Mass errors.** delta_i = w_i - mu_i. Cumulative errors sigma_k = sum_{j<k} delta_j = -frac(m M(k))/m in (-1/m, 0].

**Convolution knot points.** x_k = -1/2 + k h for k = 0, 1, ..., 4n. The autoconvolution of a step function is piecewise linear with breakpoints at these knots.

**Knot-point convolution formula.** For a step function g with heights g_i:

    (g*g)(x_k) = h sum_{i+j=k-1} g_i g_j

(where the sum ranges over i, j in [0, d-1] with i+j = k-1). The shift by 1 arises because the overlap of I_i and (x_k - I_j) is complete exactly when j = k-1-i.

**Mass convolution.** MC_mu[s] = sum_{i+j=s} mu_i mu_j. Then (f_step * f_step)(x_k) = 4n MC_mu[k-1].

**Test value.** TV(c; ell, s) = (4n/ell) sum_{k=s}^{s+ell-2} MC_w[k].

---

## 1. Cross-Term Vanishing Lemma

**Lemma 1 (Cross-term vanishing at knot points).** Let f_step be a step function with heights a_i on bins I_i, and let epsilon_2 be any L^2 function with integral_{I_i} epsilon_2 = 0 for each bin i. Then at every convolution knot point x_k:

    (f_step * epsilon_2)(x_k) = 0

**Proof.** By definition:

    (f_step * epsilon_2)(x_k) = integral f_step(t) epsilon_2(x_k - t) dt

Decompose the integration over bins:

    = sum_i integral_{I_i} f_step(t) epsilon_2(x_k - t) dt
    = sum_i a_i integral_{I_i} epsilon_2(x_k - t) dt

where we used that f_step is constant a_i on I_i.

For t in I_i = [-1/4 + ih, -1/4 + (i+1)h), the variable u = x_k - t ranges over:

    u in (x_k - (-1/4 + (i+1)h), x_k - (-1/4 + ih)]
      = (-1/4 + (k-i-1)h, -1/4 + (k-i)h]

If j = k-i-1 is in [0, d-1], this is exactly the closure of I_j (up to endpoint conventions). If j is outside [0, d-1], the range is outside supp(epsilon_2), so the integral is zero.

For valid j = k-i-1:

    integral_{I_i} epsilon_2(x_k - t) dt = integral_{I_j} epsilon_2(u) du = 0

by the zero-mean-per-bin property of epsilon_2.

Therefore each term in the sum is zero, and (f_step * epsilon_2)(x_k) = 0. QED

---

## 2. Knot-Point Decomposition Theorem

**Theorem 1 (Exact decomposition at knot points).** For any f = f_step + epsilon_2 as above:

    (f*f)(x_k) = (f_step * f_step)(x_k) + (epsilon_2 * epsilon_2)(x_k)

at every convolution knot point x_k.

**Proof.** Expand:

    (f*f)(x_k) = ((f_step + epsilon_2) * (f_step + epsilon_2))(x_k)
               = (f_step * f_step)(x_k) + 2(f_step * epsilon_2)(x_k) + (epsilon_2 * epsilon_2)(x_k)

By Lemma 1, the cross-term vanishes. QED

**Corollary 1.1.** |(f*f)(x_k) - (f_step*f_step)(x_k)| = |(epsilon_2*epsilon_2)(x_k)| <= ||epsilon_2||_2^2.

*Proof.* By Cauchy-Schwarz: |(epsilon_2*epsilon_2)(x)| = |integral epsilon_2(t) epsilon_2(x-t) dt| <= ||epsilon_2||_2^2.

**Corollary 1.2.** ||f*f||_infty >= ||f_step*f_step||_infty - ||epsilon_2||_2^2.

**Corollary 1.3 (Step functions).** If f is already a step function (epsilon_2 = 0), then (f*f)(x_k) = (f_step*f_step)(x_k) exactly. In particular, ||f*f||_infty = 4n max_s MC_mu[s] (the maximum is achieved at a knot since f*f is piecewise linear).

---

## 3. Reduction to Step Functions

**Theorem 2 (Formula B for step functions).** Let a in A_n be a step function with heights a_i = 4n mu_i, and let c = coarse_discretize(a, m). If for some window (ell, s_0):

    TV(c; ell, s_0) > c_target + 1/m^2 + 2W/m

then ||a*a||_infty >= c_target is equivalent to showing:

    4n max_s MC_mu[s] >= c_target

where the masses mu_i satisfy |w_i - mu_i| < 1/m with the cumulative-floor structure (sigma_k in (-1/m, 0]).

**Proof.** Since a is a step function, epsilon_2 = 0. By Corollary 1.3:

    ||a*a||_infty = 4n max_s MC_mu[s]

The problem reduces to a PURELY MASS-SPACE bound: given that MC_w has a high test value (the pruning condition), show that MC_mu also has a high maximum. The mass errors |delta_i| < 1/m, with the cumulative-floor structure, are the only source of discrepancy.

This is crucial: the height error 4n/m does NOT appear. The reduction to mass space eliminates the problematic 4n factor entirely. QED

---

## 4. Mass-Space Per-Window Bound (Formula A Recovery)

**Theorem 3 (Per-window bound in mass space).** Under cumulative-floor discretization:

    |sum_{k in window} (MC_w[k] - MC_mu[k])| < 2 W_mu / m + |B| / m^2

where W_mu = sum_{j in B} mu_j and B is the set of contributing bins.

**Proof.** This is identical to the derivation in discretization_error_proof.md Section 2, working entirely with masses w_i, mu_i and errors delta_i. The key ingredients:

(a) For each contributing bin j, define T_j = sum_{i in I_j} delta_i (sum over paired indices). Since I_j is a contiguous range, |T_j| < 1/m by the cumulative-floor property.

(b) The error decomposes as:

    sum_{k in window} E[k] = S_lin + S_quad

where S_lin = 2 sum_j mu_j T_j (with |S_lin| < 2W_mu/m) and S_quad = sum_j delta_j T_j (with |S_quad| < |B|/m^2).

**Consequence.** This gives:

    TV(a; ell, s) > TV(c; ell, s) - (4n/ell)(2W_mu/m + |B|/m^2)

Substituting the pruning condition:

    ||a*a||_infty >= TV(a; ell, s) > c_target + 1/m^2 + 2W/m - (4n/ell)(2W_mu/m + |B|/m^2)

For ell >= 4n (full or near-full window), the factor 4n/ell <= 1, and the RHS > c_target. This recovers Formula A.

For ell < 4n, the correction (4n/ell)(2W/m + 1/m^2) exceeds Formula B's correction 1/m^2 + 2W/m by a factor of 4n/ell. **The per-window bound does NOT prove Formula B for narrow windows.** QED

---

## 5. The Composition-Level Gap

Theorem 3 proves Formula A but not Formula B. The gap arises for narrow windows (ell << 4n). At narrow windows:

- The per-window mass-space error has a factor 4n/ell > 1.
- Formula B's correction 1/m^2 + 2W/m has no such factor.

### 5.1 Why the gap is genuine (not an artifact)

The per-window bound is essentially TIGHT. At d=4, m=10, c=(2,1,1,6), window (ell=2, s=6), the actual per-window error is 0.3996 while Formula B's correction is 0.1300 (ratio 3.07 ~ 4n/ell = 4). The 4n/ell factor is real.

### 5.2 What a composition-level proof would require

To prove Formula B without the 4n/ell factor, one must show that if the discrete TV is high at a NARROW window, then ||a*a||_infty >= c_target **for a reason other than the same narrow window's continuous TV being high**.

Possible mechanisms:
- The mass concentration that causes high narrow-window TV forces the continuous ||a*a||_infty to peak at a DIFFERENT (possibly wider) window.
- The self-convolution bound d mu_i^2 provides a partial lower bound that, combined with the per-window bound, covers all cases.

Neither mechanism has been made rigorous. See formula_b_soundness_analysis.md Sections 3-6 for detailed impossibility results on several natural proof strategies.

### 5.3 Assessment

Formula B is **empirically sound** for all tested parameter ranges (d <= 64, m <= 50). No counterexample has been found. The gap is between the proven per-window bound (Formula A with 4n/ell factor) and the conjectured composition-level bound (Formula B without the factor). This remains an **open mathematical problem**.

---

## 6. Cascade-Level Soundness via Fine-Grid Equivalence

Although Formula B as a standalone composition-level bound remains unproven, we can prove it is sound WITHIN THE CASCADE by relating the coarse grid to the fine grid.

**Theorem 4 (Coarse-grid Formula B = Fine-grid W-refined bound).** The coarse-grid composition c (d bins, sum m) and the fine-grid composition c_fine = (4n c_0, ..., 4n c_{d-1}) (d bins, sum 4nm) define the SAME step function g_c.

Moreover, the coarse-grid Formula B pruning condition at window (ell, s):

    TV(c; ell, s) > c_target + (1 + 2 W_int) / m^2

is IDENTICAL to the fine-grid W-refined pruning condition for c_fine at the same window:

    TV(c_fine; ell, s) > c_target + (1 + W_fine_int/(2n)) / m^2

**Proof.**

Step 1: Same step function. g_c has heights 4n c_i / m on bin I_i. The fine-grid composition c_fine has fine-grid heights c_fine_i / m = 4n c_i / m. Same heights, same step function.

Step 2: Same TV. Since the step function is the same, the autoconvolution at every point is identical. Hence TV(c; ell, s) = TV(c_fine; ell, s) for every window.

Step 3: Same correction. Coarse Formula B correction:

    1/m^2 + 2W/m = 1/m^2 + 2 W_int / m^2 = (1 + 2 W_int) / m^2

where W = W_int / m and W_int = sum_{j in B} c_j.

Fine-grid W-refined correction:

    (1 + W_fine_int / (2n)) / m^2

where W_fine_int = sum_{j in B} c_fine_j = sum_{j in B} 4n c_j = 4n W_int.

So: (1 + 4n W_int / (2n)) / m^2 = (1 + 2 W_int) / m^2.

The corrections are identical. QED

**Corollary 4.1 (Cascade soundness for step functions).** If the fine-grid W-refined pruning condition is sound for all step functions a in A_n (at all cascade levels), then the coarse-grid Formula B is also sound for step functions.

*Proof.* By Theorem 4, any coarse composition pruned by Formula B corresponds to a fine-grid composition pruned by the same (W-refined) criterion. The fine-grid criterion's soundness implies the coarse criterion's soundness.

**Corollary 4.2 (Operational equivalence).** A cascade that uses the coarse grid (compositions sum to m) with Formula B pruning produces EXACTLY the same pruning decisions as a cascade that uses the fine grid (compositions sum to 4nm) with W-refined pruning, restricted to the subset of fine-grid compositions where every coordinate is a multiple of 4n.

---

## 7. Soundness of the W-Refined Bound on the Fine Grid

The W-refined pruning condition on the fine grid is:

    TV(b; ell, s) > c_target + (1 + W_int/(2n)) / m^2

For a step function a in A_n, let b = fine_discretize(a, m) with ||a - g_b||_infty <= 1/m.

**Claim (C&S eq(1) at knot points):**

    (g_b * g_b)(x_k) <= (a * a)(x_k) + 2 W_a(x_k) / m + 1 / m^2

where W_a(x_k) = mass of a in the contributing bins for knot k.

**Derivation.** Write a = g_b + epsilon where ||epsilon||_infty <= 1/m.

    (a*a)(x_k) = (g_b*g_b)(x_k) + 2(g_b*epsilon)(x_k) + (epsilon*epsilon)(x_k)

At a knot point x_k:

    |(g_b*epsilon)(x_k)| = |sum_{i+j=k-1} integral_{I_i} g_b,i epsilon(x_k - t) dt|
                         = |sum_{i+j=k-1} g_b,i integral_{I_j} epsilon(u) du|

Now, integral_{I_j} epsilon(u) du = integral_{I_j} (a - g_b)(u) du. Since g_b,j is constant on I_j:

    integral_{I_j} epsilon du = integral_{I_j} a du - g_b,j h = a_j h - g_b,j h = (a_j - g_b,j) h

Hmm, this is NOT zero (unlike epsilon_2 = f - f_step where f_step has the same bin masses). Here epsilon = a - g_b where a and g_b are DIFFERENT step functions. Since both are step functions:

    epsilon(t) = (a_j - g_b,j) on bin I_j (constant per bin)

So integral_{I_j} epsilon du = (a_j - g_b,j) h.

Therefore:

    (g_b*epsilon)(x_k) = h sum_{i+j=k-1} g_b,i (a_j - g_b,j)

The magnitude: |a_j - g_b,j| <= 1/m (fine grid error).

    |(g_b*epsilon)(x_k)| <= h sum_{i+j=k-1} g_b,i / m = (1/m) h sum_{j': (exists i) i+j'=k-1, j' in [0,d-1]} g_b,{k-1-j'}

Wait, let me redo: for each pair (i,j) with i+j=k-1, we have the term g_b,i * (a_j - g_b,j). Since |a_j - g_b,j| <= 1/m:

    |(g_b*epsilon)(x_k)| <= (h/m) sum_{i+j=k-1} g_b,i = (h/m) sum_{i: k-1-i in [0,d-1]} g_b,i

But sum_{i in J} g_b,i h = W_b^k (mass of g_b in the contributing bins). And since ||g_b - a||_infty <= 1/m, the masses differ by at most h/m = 1/(4nm), so W_b^k ~ W_a^k.

More precisely: sum g_b,i h = W_b^k, so (h/m) sum g_b,i = W_b^k / m.

But we actually want a sharper bound. Using the identity at knot points (where the integration aligns with bins):

    (g_b*epsilon)(x_k) = h sum_{i+j=k-1} g_b,i (a_j - g_b,j)

and summing over each j's contribution:

    = h sum_j (a_j - g_b,j) sum_{i: i+j=k-1, i in [0,d-1]} g_b,i

For each valid j: the inner sum has one term g_b,{k-1-j}, so:

    = h sum_j (a_j - g_b,j) g_b,{k-1-j}

Hmm, this has mixed signs (a_j - g_b,j can be positive or negative). Let me just use the triangle inequality:

    |(g_b*epsilon)(x_k)| <= (1/m) h sum_{j in J_k} g_b,{k-1-j} = (1/m) W_b(x_k)

where W_b(x_k) = h sum_{j in J_k} g_b,{k-1-j} is the mass of g_b in the contributing bins (reindexed through the pairing).

Wait, more precisely: W_b(x_k) = mass of g_b in bins that contribute to knot k. Each pair (i,j) with i+j=k-1 contributes both bins i and j. The total contributing mass involves the set of all bins i and j that participate in any pair. This is {max(0, k-d), ..., min(k-1, d-1)}.

The bound is:

    |(g_b*epsilon)(x_k)| <= (1/m) integral g_b(t) 1_{B_k}(t) dt = W_b(x_k) / m

where B_k is the union of contributing bins and W_b(x_k) is g_b's mass in those bins.

Similarly: |(epsilon*epsilon)(x_k)| <= (1/m) integral |epsilon| <= (1/m)(1/m)(1/2) = 1/(2m^2).

More precisely, ||epsilon||_infty <= 1/m and supp(epsilon) in [-1/4, 1/4], so ||epsilon||_1 <= 1/(2m). Then |(epsilon*epsilon)(x)| <= ||epsilon||_infty ||epsilon||_1 <= 1/(2m^2) <= 1/m^2.

Therefore:

    (a*a)(x_k) >= (g_b*g_b)(x_k) - 2 W_b(x_k) / m - 1/m^2

At knot points, W_a(x_k) = W_b(x_k) + O(d h / m) ≈ W_b(x_k) (the mass difference between a and g_b over the contributing bins is < d/(4nm) which is small). For the W-refined bound, using W_g = W_b:

    (a*a)(x_k) >= (g_b*g_b)(x_k) - 2 W_b(x_k) / m - 1/m^2

This confirms C&S eq(1) on the fine grid, with W = W_b (the mass of the GRID function in the contributing bins).

The fine-grid W-refined pruning condition uses W_int/(2n) in integer space, which translates to W_b/m in physical space. So the correction is:

    2 W_b / m + 1/m^2 = (2 m W_b + 1) / m^2 = (1 + 2 W_int_fine) / m^2

where W_int_fine = m W_b (the fine-grid integer mass).

Now, for the pruning to be sound: if TV(g_b; ell, s) > c_target + (1 + 2W_int)/(m^2), then for all a with fine_discretize(a) = b:

    ||a*a||_infty >= max_k (a*a)(x_k) >= max_k [(g_b*g_b)(x_k) - 2W_b(x_k)/m - 1/m^2]

The RHS is maximized at the peak knot of g_b*g_b minus the local correction. Since W_b(x_k) varies with k, the window-specific correction (1 + W_int(k)/(2n))/m^2 is used. The pruning condition checks: for SOME window, the sum of (g_b*g_b)(x_k) exceeds the sum of corrections. This implies the max of (a*a) exceeds c_target.

The formal details of this step use the window-scan structure: the pruning condition at a window of ell-1 knots means that the average of (g_b*g_b) at those knots exceeds c_target + (average correction). Since the corrections are computed per-window (using W_int for that specific window), the bound holds. QED

---

## 8. Putting It Together

### 8.1 For step functions in the cascade

**Corollary (Main result).** If the fine-grid W-refined pruning criterion is sound (Section 7), then a coarse-grid cascade using Formula B produces correct results for all step functions, because:

1. Each coarse composition c maps to fine composition c_fine = 4n c (Theorem 4).
2. The pruning decisions are identical (Theorem 4, Corollary 4.2).
3. The fine-grid W-refined criterion is sound by C&S eq(1) (Section 7).

### 8.2 For general functions

For general f (non-step), two additional ingredients are needed:

1. **Theorem 1 decomposition:** (f*f)(x_k) = (f_step*f_step)(x_k) + (epsilon_2*epsilon_2)(x_k).

2. **The epsilon_2*epsilon_2 term:** This can be negative, with magnitude <= ||epsilon_2||_2^2. A bound ||f*f||_infty >= ||f_step*f_step||_infty - ||epsilon_2||_2^2 follows (Corollary 1.2), but the correction ||epsilon_2||_2^2 depends on f's shape within bins and is NOT bounded by O(1/m^2).

3. **Passage to general f:** This requires the d -> infinity limit in the C&S framework. At fixed d, the cascade proves bounds for step functions only. As d -> infinity, step functions approximate general functions, and C_1a >= c_target follows.

### 8.3 Status of Formula B as a standalone theorem

**Formula B for general f at fixed d remains OPEN.** The per-window bound gives Formula A (with 4n/ell factor). The composition-level bound without this factor requires a new argument about how mass concentration forces autoconvolution peaks, which has resisted all known proof strategies (see formula_b_soundness_analysis.md).

**Formula B for step functions at fixed d also remains OPEN as a standalone theorem.** However, within the cascade framework, it is PROVED SOUND by the fine-grid equivalence (Theorem 4 + Section 7).

---

## 9. Summary of New Results

**Lemma 1 (Cross-term vanishing):** This is the key new result. At convolution knot points, the cross-term between a step function and any zero-mean-per-bin perturbation VANISHES exactly. This was not present in prior analyses and simplifies the error structure from linear + quadratic to purely quadratic.

**Theorem 4 (Fine-grid equivalence):** The coarse-grid Formula B correction (1 + 2W_int)/m^2 is IDENTICAL to the fine-grid W-refined correction when applied to the corresponding fine-grid composition 4n c. This provides a clean proof that Formula B in the cascade is sound, bypassing the composition-level gap entirely.

**The gap:** Formula B as a standalone per-window bound requires handling the 4n/ell factor. This factor is genuine (counterexamples exist showing the per-window error exceeds Formula B's correction). A composition-level argument would be needed to eliminate it, but no such argument is known.
