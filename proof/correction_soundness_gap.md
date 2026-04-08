> **Note (2026-04-07): RESOLVED.** The bug described in this document has been fixed. The code now uses the C&S fine grid: compositions sum to $S = 4nm$ (not $m$), heights are multiples of $1/m$, and the C&S Lemma 3 correction $2/m + 1/m^2$ applies directly. The analysis below correctly identified the root cause (coarse-grid height granularity of $4n/m$ instead of $1/m$) and led to the fix. The document is preserved as a record of the diagnosis.

# Correction Term Soundness Gap: Complete Analysis

## 1. What C&S Lemma 3 Actually Says

From Cloninger & Steinerberger (arXiv:1403.7988, Section 3):

**Setup.** Define height vectors:
- A_n = {a in R_>=0^{2n} : sum a_i = 4n} (exact step functions)
- B_{n,m} = {b in (1/m N)^{2n} : sum b_i = 4n} (quantized step functions)

Each vector a or b defines a step function on [-1/4, 1/4] with 2n bins of width 1/(4n).
Heights in B_{n,m} are multiples of **1/m** (the height granularity).

**Lemma 2 (C&S).** For all a in A_n, there exists b in B_{n,m} with ||a - b||_inf <= 1/m.

This is straightforward: each height a_i is rounded to the nearest multiple of 1/m, then the sum constraint is restored by adjusting at most one coordinate.

**Lemma 3 (C&S).** Write a = b + epsilon, where ||epsilon||_inf <= 1/m. Then:

    b*b = a*a - 2(a*epsilon) + epsilon*epsilon

Bounding each term pointwise:
- |(a*epsilon)(x)| <= ||epsilon||_inf * integral(|a|) = (1/m) * 1 = 1/m
- |(epsilon*epsilon)(x)| <= ||epsilon||_inf^2 * |support| <= 1/m^2

Therefore:

    (a*a)(x) <= (b*b)(x) + 2/m + 1/m^2     for all x

Taking sup over x and inf over a:

    **c >= b_{n,m} - 2/m - 1/m^2**

where b_{n,m} = min_{b in B_{n,m}} max_windows TV(b).

**Key point:** The bound 2/m + 1/m^2 relies on ||epsilon||_inf <= 1/m, which holds because B_{n,m} has height granularity 1/m.

---

## 2. What Our Code Does

Our code parameterizes step functions differently:
- Integer masses c_i in N, with sum c_i = m
- Heights a_i = c_i * 4n/m

The height granularity is **4n/m**, not 1/m.

### Height granularity comparison

| | C&S paper (B_{n,m}) | Our code |
|---|---|---|
| Parameters | k_i integers, sum k_i = 4nm | c_i integers, sum c_i = m |
| Heights | b_i = k_i/m (multiples of 1/m) | a_i = c_i * 4n/m (multiples of 4n/m) |
| Granularity | 1/m | 4n/m |
| Grid points per bin | ~4nm | m |
| Rounding error | <= 1/(2m) (nearest) or <= 1/m (Lemma 2) | < 4n/m (cumulative floor) |

The paper's grid B_{n,m} is **4n times finer** than our code's grid.

### Proof that our height error is 4n/m

Under cumulative-floor discretization with parameter m:
- Exact mass in bin i: mu_i (with sum mu_i = 1)
- Integer mass: c_i, where c_i = D(i+1) - D(i), D(k) = floor(m * M(k))
- Mass error: |c_i/m - mu_i| < 1/m (proven in discretization_error_proof.md, Section 1)

Converting to heights:
- Exact height: h_i = 4n * mu_i
- Code height: a_i = 4n * c_i / m
- Height error: |h_i - a_i| = 4n * |mu_i - c_i/m| < 4n/m

Therefore: **||epsilon||_inf < 4n/m** for our discretization.

---

## 3. The Correct Correction for Our Grid

Applying the C&S Lemma 3 argument with ||epsilon||_inf < 4n/m:

    |(a*epsilon)(x)| <= ||epsilon||_inf * integral(|a|) < (4n/m) * 1 = 4n/m
    |(epsilon*epsilon)(x)| <= ||epsilon||_inf^2 * |support| < (4n/m)^2 * (1/2) = 8n^2/m^2

Therefore:

    (a*a)(x) <= (b*b)(x) + 2*(4n/m) + (4n/m)^2 * C

More precisely:

    **correction_true = 8n/m + 16n^2/m^2**

(using the same argument structure as Lemma 3, with the tighter ||epsilon||^2 bound).

### Comparison with what the code uses

| n (= d/2) | m | Code's correction (2/m + 1/m^2) | True correction (8n/m + 16n^2/m^2) | Ratio |
|---|---|---|---|---|
| 2 (L0) | 3 | 0.778 | 12.44 | 16x |
| 2 (L0) | 20 | 0.103 | 0.96 | 9.4x |
| 4 (L1) | 3 | 0.778 | 39.1 | 50x |
| 4 (L1) | 20 | 0.103 | 2.24 | 21.8x |
| 8 (L2) | 3 | 0.778 | 135.1 | 174x |
| 8 (L2) | 20 | 0.103 | 6.64 | 64.5x |
| 16 (L3) | 20 | 0.103 | 22.4 | 218x |
| 24 (C&S) | 50 | 0.041 | 5.68 | 139x |

**The code's correction is always too small by a factor of approximately 4n.**

**The true correction grows with n** (cascade level). Deeper levels need larger m to compensate.

---

## 4. Why This Produces False Proofs

At m=3, c_target=1.51:

The code's pruning condition: TV_disc > c_target + (3 + 2*W_int)/m^2

This uses a correction of at most (3 + 2*3)/9 = 1.0.

But the near-optimal function (||f*f|| = 1.5029) maps to some composition at m=3.
Its step function has concentrated mass (3 integers in d bins), giving high TV_disc.
The height error is up to 4n/m, meaning the step function can overestimate the
true autoconvolution by up to 8n/m in TV space.

At L2 (n=8, m=3): the overestimate can be up to 64/3 = 21.3 in TV space.
The code only accounts for 0.78.

So: TV_disc = ||f*f|| + 21 (massive overestimate)
But: threshold = c_target + 0.78 = 2.29

Since TV_disc >> threshold, the composition is pruned.
The code concludes ||f*f|| > c_target = 1.51.
But ||f*f|| = 1.5029 < 1.51.

**The proof is wrong because the correction is 27x too small.**

---

## 5. Numerical Verification

### Example: m=3, d=4 (n=2)

f with mass mu = (0.4, 0.3, 0.2, 0.1).

Cumulative floor: D = (0, floor(0.4*3), floor(0.7*3), floor(0.9*3), 3) = (0, 1, 2, 2, 3).
So c = (1, 1, 0, 1).

**Mass redistribution:**

| Bin | Exact mu | Integer c/m | Error delta | Height exact (4n*mu) | Height code (4n*c/m) |
|-----|----------|-------------|-------------|---------------------|---------------------|
| 0 | 0.400 | 0.333 | -0.067 | 3.20 | 2.67 |
| 1 | 0.300 | 0.333 | +0.033 | 2.40 | 2.67 |
| 2 | 0.200 | 0.000 | -0.200 | 1.60 | 0.00 |
| 3 | 0.100 | 0.333 | +0.233 | 0.80 | 2.67 |

Bin 2 lost 100% of its mass. Bin 3 gained 233%. Height error ||epsilon||_inf = 4*2*0.233 = 1.87.
Compare with 1/m = 0.333. **The actual height error is 5.6x the code's assumed bound.**

**Autoconvolution at k=4:**

conv_exact[4] = 2*h1_exact*h3_exact + h2_exact^2 = 2*2.4*0.8 + 1.6^2 = 3.84 + 2.56 = 6.40
conv_code[4]  = 2*h1_code*h3_code + h2_code^2  = 2*2.67*2.67 + 0   = 14.22

Difference: 14.22 - 6.40 = **7.82**
Code's correction (in conv space): 2/m + 1/m^2 = 0.778 (times 4n*ell for integer space)

In TV space (ell=2, s=4):
TV_code - TV_exact = (1/(4*2*2)) * 7.82 = 0.489
Code's correction in TV space: 0.778
True correction (8n/m + 16n^2/m^2): 5.33 + 7.11 = 12.44

For this specific window: 0.489 < 0.778 < 12.44. The code's correction happens to cover
this case. But this is one specific window of one specific composition. At deeper levels
(larger n), the overestimate grows and the code's correction cannot keep up.

### Why m=3 proves c_target=1.51 (falsely)

At L2 (d=16, n=8, m=3): step functions have 3 non-zero bins out of 16.
Heights in non-zero bins: 4*8*1/3 = 10.67 each.

A near-uniform continuous function (mu_i ~ 1/16) maps to a composition with
3 concentrated bins. The step function's autoconvolution peaks at ~10.67^2/(4*8*2) ~ 1.78
in TV space for a narrow window.

The exact step function's TV for the same window: ~1/(4*8*2) * 16 * (4*8/16)^2 ~ 0.25

Overestimate: ~1.53. Code's correction: 0.78. **Not enough.**

---

## 6. Why the Paper's Result (c >= 1.28) Is Not Affected

The C&S paper uses m=50 with the **fine grid** (heights are multiples of 1/50 = 0.02).
Their k_i are arbitrary integers summing to 4nm = 4*24*50 = 4800.
The rounding error is <= 1/50 = 0.02.
Lemma 3 gives correction 2/50 + 1/2500 = 0.04.

This is correct because they use the fine grid matching Lemma 3's assumptions.

Our code uses m mass quanta (not 4nm height quanta), giving a coarser grid.
The code applies the paper's formula as if the height precision were 1/m,
but it's actually 4n/m.

---

## 7. Implications

### 7.1 False proofs at small m

Any cascade run with c_target close to C_upper and m small enough will produce
a false proof. The user demonstrated this with m=3, c_target=1.51 > C_upper=1.5029.

### 7.2 The correction grows with cascade depth

The true correction 8n/m + 16n^2/m^2 grows as n doubles at each cascade level:

| Level | n | True correction (m=20) |
|-------|---|----------------------|
| L0 | 2 | 0.96 |
| L1 | 4 | 2.24 |
| L2 | 8 | 6.64 |
| L3 | 16 | 22.4 |

The threshold = c_target + correction grows rapidly, making pruning increasingly
difficult at deeper levels. Near-optimal compositions (TV ~ 1.5) cannot exceed
a threshold of c_target + 22.4 = 23.4.

### 7.3 What needs to happen

To use the paper's correction (2/m + 1/m^2) correctly, the code must use the
paper's fine grid:
- Replace integer masses c_i (summing to m) with height quanta k_i (summing to 4nm)
- This gives height granularity 1/m and rounding error <= 1/m
- Lemma 3 then applies as stated

The cost: the search space increases from C(m + d - 1, d - 1) to C(4nm + d - 1, d - 1),
which is exponentially larger. The cascade must handle this increased branching factor.

Alternatively: derive a tighter bound specifically for the coarse grid that doesn't
require the 4n factor. The cumulative-floor discretization has special structure
(sigma_k in (-1/m, 0]) that might enable this. The existing
discretization_error_proof.md derives (4n/ell)(2W/m + |B|/m^2), which has the
4n/ell factor but is tighter for wide windows. A bound that avoids the 4n factor
entirely would require a new argument.

---

## 8. The Lean Axiom Is False

The file lean/Sidon/DiscretizationError.lean contains:

```lean
axiom cs_lemma3_discrete_conv_gap ...
    (sum over window of (DA(a_g, k) - DA(a_f, k))) <=
      (2 / m + 1 / m ^ 2) * (4 * n * ell)
```

This axiom claims the sum of autoconvolution differences (integer step function minus
exact step function) is bounded by (2/m + 1/m^2) * 4n*ell.

Dividing by 4n*ell gives: TV_g - TV_f <= 2/m + 1/m^2 (per-window).

**This is false for the cumulative-floor discretization when n is large relative to m.**

At n=8, m=3: the TV overestimate can reach ~1.53, far exceeding 2/3 + 1/9 = 0.778.

The axiom was stated without proof. It should be removed or replaced with the
correct bound.

---

## 9. Summary

| Claim | Status |
|-------|--------|
| C&S Lemma 3 (2/m + 1/m^2) for the fine grid | **Correct** (proven in the paper) |
| Same correction for our coarse grid (heights = multiples of 4n/m) | **Incorrect** (height error is 4n/m, not 1/m) |
| True correction for our grid | **8n/m + 16n^2/m^2** (derived above) |
| Code uses 2/m + 1/m^2 | **Unsound** (too small by factor ~4n) |
| Lean axiom cs_lemma3_discrete_conv_gap | **False** (unproven, and contradicted by m=3 experiment) |
| m=3 proof of c_target=1.51 | **Invalid** (false proof due to insufficient correction) |
| Vacuity table (comparing threshold to C_upper) | **Captures a real soundness boundary**, not just a heuristic |
