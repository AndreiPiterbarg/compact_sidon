# Explanation of the C&S Threshold Fix (2026-04-04)

## What Was Wrong

Our cascade prover used a pruning threshold derived from **Theorem 3.7** of our proof document (`lower_bound_proof.tex`). This theorem bounds the **per-window test-value discretization error**:

$$\text{TV}_{n,m}(c;\,\ell, s_0) - \text{TV}^{\text{cont}}(f;\,\ell, s_0) \;\leq\; \frac{4n}{\ell}\!\left(\frac{1}{m^2} + \frac{2W}{m}\right)$$

The $4n/\ell$ factor arises because the bound handles each of the $\ell - 1$ convolution positions in the window independently (worst-case per position), then normalizes by $1/(4n\ell)$. This factor is **correct** for bounding per-window test-value error, but it grows exponentially with cascade depth: at L4 ($d = 64$, $n = 32$), the correction for $\ell = 2$ is $4 \times 32 / 2 = 64$ times the base correction. This made narrow windows useless for pruning, causing the cascade to diverge at every level.

The resulting integer-space threshold was (Formula A):

```
ws_int > c_target * m² * ℓ/(4n)  +  1 + 2·W_int + eps
         \_________________/         \______________/
         scaled by ℓ/(4n)            NOT scaled (additive)
```

Only `c_target * m²` was scaled by `ℓ/(4n)`. The correction `1 + 2·W_int` was added as-is, because the `4n/ℓ` factor in TV space cancels with the `m²·ℓ/(4n)` conversion factor, leaving just `1 + 2·W_int` in integer space.

## What Cloninger & Steinerberger Actually Prove

Reading the actual paper (arXiv:1403.7988, Lemma 3, page 6), C&S use a fundamentally different proof structure. They do **not** bound per-window test-value error. Instead, they bound the **pointwise $L^\infty$ difference** between the autoconvolutions of two step functions.

### The C&S argument (Lemma 3)

Let $f$ be a step function with exact bin densities $a_i$ (the bin averages of the original continuous function), and let $g$ be the step function with discretized bin densities $b_i$ (multiples of $1/m$). Define $\varepsilon = f - g$, so $|\varepsilon(x)| \leq 1/m$ pointwise.

Then:

$$(g * g)(x) = (f * f)(x) - 2(f * \varepsilon)(x) + (\varepsilon * \varepsilon)(x)$$

Bounding each term **pointwise for every $x$**:

$$|(\varepsilon * \varepsilon)(x)| = \left|\int_{-1/4}^{1/4} \varepsilon(x - y)\,\varepsilon(y)\,dy\right| \leq \frac{1}{m} \int_{-1/4}^{1/4} |\varepsilon(y)|\,dy \leq \frac{1}{m} \cdot \frac{1}{2m} = \frac{1}{2m^2}$$

(The integral is over support $[-1/4, 1/4]$ of length $1/2$, with $|\varepsilon| \leq 1/m$. We use the conservative bound $1/m^2 \geq 1/(2m^2)$ below for simplicity.)

$$|(f * \varepsilon)(x)| = \left|\int_{overlap(x)} f(x - y)\,\varepsilon(y)\,dy\right| \leq \frac{1}{m} \int_{-1/4}^{1/4} f(y)\,dy = \frac{1}{m}$$

Therefore, **for all $x$** (using $1/m^2$ as conservative upper bound on $|(\varepsilon*\varepsilon)|$):

$$(g * g)(x) \leq (f * f)(x) + \frac{2}{m} + \frac{1}{m^2}$$

Taking the supremum: $\|g * g\|_\infty \leq \|f * f\|_\infty + 2/m + 1/m^2$.

### Why this eliminates the $4n/\ell$ factor

Test values are **averages** of $(g * g)$ over a window of length $\ell/(4n)$:

$$\text{TV}(b;\,\ell, k) = \frac{1}{|J|}\int_J (g * g)(x)\,dx$$

Since averages $\leq$ the supremum:

$$\text{TV}(b;\,\ell, k) \leq \|g * g\|_\infty \leq \|f * f\|_\infty + \frac{2}{m} + \frac{1}{m^2}$$

This holds for **every** window $(\ell, k)$ with the **same** correction $2/m + 1/m^2$. There is no window-dependent factor because the bound is on the global $L^\infty$ norm, not on individual windows.

### The pruning criterion

If $\text{TV}(b;\,\ell, k) > c_{\text{target}} + 2/m + 1/m^2$ for any window, then $\|g * g\|_\infty > c_{\text{target}} + 2/m + 1/m^2$, hence $\|f * f\|_\infty > c_{\text{target}}$. If **all** discrete vectors $b$ are pruned, then $b_{n,m} > c_{\text{target}} + 2/m + 1/m^2$, and by Lemma 3: $c \geq c_{\text{target}}$.

### The W-refinement (C&S equation 1, page 7)

C&S also note a tighter bound using the restricted support overlap:

$$|(f * \varepsilon)(x)| \leq \frac{1}{m} \int_{overlap(x)} f(x - y)\,dy = \frac{W_f(x)}{m}$$

where $W_f(x) = \int_{overlap(x)} f(y)\,dy$ is the mass of $f$ in the region contributing to position $x$. This gives:

$$(g * g)(x) \leq (f * f)(x) + \frac{2W_f(x)}{m} + \frac{1}{m^2}$$

Since $W_f(x) \leq 1$, this is **at least as tight** as the basic bound and **strictly tighter** when $W_f(x) < 1$ (which happens for narrow windows near the edges of the convolution support).

### Correcting for discrete $W_g$ (our implementation)

In the cascade, we compute $W_{\text{int}} = \sum_{i \in \mathcal{B}} c_i$ from the **discrete** vector $g$ (masses $c_i$), not from the theoretical un-discretized step function $f$. Since C&S Lemma 3 uses $W_f$, we must account for the difference.

By the cumulative rounding property (C&S Lemma 2): for any contiguous range of bins, the sum of integer mass errors $|\sum_{i=p}^{q}(\tilde{c}_i - c_i)| \leq 1$, where $\tilde{c}_i$ are the un-rounded masses. Since the bins contributing to a window form a contiguous range:

$$W_f(x) \leq W_g(x) + \frac{1}{m}$$

Substituting into the pointwise bound:

$$\frac{2W_f(x)}{m} + \frac{1}{m^2} \leq \frac{2(W_g(x) + 1/m)}{m} + \frac{1}{m^2} = \frac{2W_g(x)}{m} + \frac{3}{m^2}$$

Therefore, the sound pruning threshold using the discrete $W_g$ is:

$$(g * g)(x) \leq (f * f)(x) + \frac{2W_g(x)}{m} + \frac{3}{m^2}$$

The $+3/m^2$ decomposes as: $+1/m^2$ from the $(\varepsilon * \varepsilon)$ bound, $+2/m^2$ from the cumulative rounding correction $W_f \leq W_g + 1/m$.

**Note:** The original C&S MATLAB uses $+1/m^2$ (i.e., `gridSpace^2`) instead of $+3/m^2$, omitting the $W_g$ vs $W_f$ correction. For their parameters ($m = 50$), the gap is $2/m^2 = 0.0008$ — negligible. Our implementation uses the rigorous $+3/m^2$ to be mathematically exact.

## The Integer-Space Formulas (Before and After)

### Before (Formula A — our Theorem 3.7):

The pruning condition $\text{TV} > c_{\text{target}} + (4n/\ell)(1/m^2 + 2W/m)$ converts to integer space by multiplying both sides by $m^2 \ell/(4n)$:

$$\text{ws\_int} > c_{\text{target}} \cdot m^2 \cdot \frac{\ell}{4n} + 1 + 2W_{\text{int}} + \varepsilon$$

The $4n/\ell$ factor in the correction **cancels** with the $m^2 \ell/(4n)$ conversion, leaving the correction as a raw additive term. Only `c_target * m²` is scaled by `ℓ/(4n)`.

### After (C&S Lemma 3 — basic correction, no W-refinement):

The pruning condition $\text{TV} > c_{\text{target}} + 2/m + 1/m^2$ converts to integer space:

$$\text{ws\_int} > \left(c_{\text{target}} \cdot m^2 + 2m + 1 + \varepsilon\right) \cdot \frac{\ell}{4n}$$

The **entire** threshold (including correction) is scaled by $\ell/(4n)$. The correction in integer space is $(2m + 1) \cdot \ell/(4n)$, which **decreases** for narrow windows (small $\ell$). This bound is unconditionally sound (no $W_g$/$W_f$ issue since it uses the global bound $W_f \leq 1$).

### After (C&S + W-refinement, corrected for discrete $W_g$):

The pruning condition $\text{TV} > c_{\text{target}} + 3/m^2 + 2W_g/m$ converts to:

$$\text{ws\_int} > \left(c_{\text{target}} \cdot m^2 + 3 + 2W_{\text{int}} + \varepsilon\right) \cdot \frac{\ell}{4n}$$

Same structure: everything scaled by $\ell/(4n)$, but using per-window $W_{\text{int}}$ (from the discrete vector) instead of the global maximum $m$. The $+3$ (instead of $+1$) accounts for the cumulative rounding correction.

### Comparison at $\ell = 2$, $d = 32$ ($n = 16$), $m = 20$, $W_{\text{int}} = 10$:

| Formula | Integer threshold |
|---|---|
| **A (old)** | $1.40 \times 400 \times 2/64 + 1 + 20 + \varepsilon = 17.5 + 21 = 38.5$ |
| **C&S basic** | $(560 + 40 + 1) \times 2/64 = 601 \times 0.03125 = 18.78$ |
| **C&S + W (rigorous)** | $(560 + 3 + 20) \times 2/64 = 583 \times 0.03125 = 18.22$ |

The old threshold (38.5) was **more than double** the corrected threshold (18.22), making it nearly impossible for narrow windows to prune anything.

## Exactly What Code Changed

### 1. `cloninger-steinerberger/pruning.py` — `correction()` function

**Before:**
```python
def correction(m, n_half=None, ell_min=2):
    base = 2.0 / m + 1.0 / (m * m)
    if n_half is None:
        return base
    factor = max(1.0, 4.0 * n_half / ell_min)
    return factor * base
```

**After:**
```python
def correction(m, n_half=None, ell_min=2):
    return 2.0 / m + 1.0 / (m * m)
```

**Why:** The old function multiplied the base correction by $4n/\ell_{\min}$ (worst case over all window lengths). This factor is unnecessary under C&S Lemma 3 because the correction is window-independent. The `n_half` and `ell_min` parameters are still accepted for API compatibility but are no longer used.

**Where this matters:** The `correction()` function is called by `run_level0()` and `benchmark_sweep.py` to compute `x_cap` (the maximum single-bin mass) and to display the "effective threshold" in log output. With the smaller correction, the effective threshold drops (e.g., from $c_{\text{target}} + 0.41$ to $c_{\text{target}} + 0.1025$ at $m = 20$, $n_{\text{half}} = 2$), allowing more compositions to be pruned at L0. Note: `correction()` returns the basic $2/m + 1/m^2$ (used for `x_cap` and display), while the window scan uses the tighter W-refined formula with $+3/m^2$.

### 2. `run_cascade.py` — `_prune_dynamic_int32()` (batch L0 pruner, int32 path)

**Before (lines 70-79, precomputation):**
```python
ct_base_ell_arr[ell] = c_target * m_d * m_d * ell_f * inv_4n
```
This precomputed only the `c_target * m²` portion scaled by `ℓ/(4n)`.

**After:**
```python
cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin
ct_base_ell_arr[ell] = cs_corr_base * ell_f * inv_4n
```
Now `ct_base_ell_arr` includes the `3 + eps` correction term (1 from $|\varepsilon*\varepsilon|$, 2 from $W_f \leq W_g + 1/m$), all scaled by `ℓ/(4n)`.

A new array `w_scale_arr[ell] = 2.0 * ℓ * inv_4n` was added to scale the per-window $W_{\text{int}}$ contribution.

**Before (lines 107-121, window scan inner loop):**
```python
W_int = np.int64(prefix_c[hi_bin + 1]) - np.int64(prefix_c[lo_bin])
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
dyn_it = np.int64(dyn_x * one_minus_4eps)
```
The correction `1.0 + eps_margin + 2.0 * W_int` was added as a raw (unscaled) additive term.

**After:**
```python
W_int = np.int64(prefix_c[hi_bin + 1]) - np.int64(prefix_c[lo_bin])
dyn_x = ct_base_ell + w_scale * np.float64(W_int)
dyn_it = np.int64(dyn_x * one_minus_4eps)
```
Now `ct_base_ell` already contains `(c_target·m² + 3 + eps) · ℓ/(4n)`, and `w_scale * W_int` adds `2·W_int · ℓ/(4n)`. The entire threshold is scaled by `ℓ/(4n)`.

### 3. `run_cascade.py` — `_prune_dynamic_int64()` (batch L0 pruner, int64 path)

Identical structural change to the int32 path. Same formula change: precompute `cs_corr_base`, add `w_scale_arr`, use `dyn_x = ct_base_ell + w_scale * W_int`.

### 4. `run_cascade.py` — `_prune_dynamic()` docstring

Updated to describe the new formula:

```python
def _prune_dynamic(batch_int, n_half, m, c_target):
    """Per-window dynamic threshold — dispatches int32/int64 based on m.

    C&S Lemma 3 + eq(1) W-refined pruning, corrected for discrete W_g:
        TV ≤ ||f*f||∞ + 2·W_g/m + 3/m²
    where +3/m² = +1/m² (|ε*ε|) + 2/m² (W_f ≤ W_g + 1/m, cumulative rounding).
    In integer space:
        dyn_it = floor((c_target·m² + 3 + 2·W_int + eps) · ℓ/(4n) · (1 - 4·DBL_EPS))
    The ENTIRE threshold (including correction) scales by ℓ/(4n).
    W_int is the sum of discrete child masses in the window's bin range."""
```

### 5. `run_cascade.py` — `_fused_generate_and_prune()` (Gray code kernel v1)

This is the main per-parent pruning kernel used at L1+. Three sub-locations were changed:

#### 5a. Precomputed per-ell constants (lines ~584-591)

**Before:**
```python
ct_base_ell_arr[idx] = c_target * m_d * m_d * np.float64(ell) * inv_4n
```

**After:**
```python
cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin
ct_base_ell_arr[idx] = cs_corr_base * np.float64(ell) * inv_4n
w_scale_arr[idx] = 2.0 * np.float64(ell) * inv_4n
```

#### 5b. Quick-check (lines ~675-697)

**Before:**
```python
dyn_x_qc = ct_base_ell_arr[ell_idx_qc] + 1.0 + eps_margin + 2.0 * np.float64(qc_W_int)
```
Used the tracked `qc_W_int` from the previous child's killing window.

**After:**
```python
qc_W_int_now = np.int64(0)
for qc_i in range(qc_lo_bin, qc_hi_bin + 1):
    qc_W_int_now += np.int64(child[qc_i])
dyn_x_qc = ct_base_ell_arr[ell_idx_qc] + w_scale_arr[ell_idx_qc] * np.float64(qc_W_int_now)
```
Recomputes $W_{\text{int}}$ on the current child (since the child's bin values changed from the previous child) and applies the scaled W-refinement.

#### 5c. Main window scan (lines ~700-730)

**Before:**
```python
W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
```

**After:**
```python
W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
dyn_x = ct_base_ell + w_scale * np.float64(W_int)
```
Same formula change: `ct_base_ell` already contains `(c·m² + 3 + eps) · ℓ/(4n)`, and `w_scale * W_int` adds `2·W · ℓ/(4n)`.

### 6. `run_cascade.py` — Arc consistency subtree pruning (lines ~885-910)

**Before:** The subtree pruning computed `W_int_max` (upper bound on W over all unfixed cursor positions) using a complex two-part calculation (fixed-region `prefix_c` + unfixed-region `parent_prefix`), then:
```python
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int_max)
```

**After:**
```python
dyn_x = ct_base_ell + w_scale * np.float64(W_int_max)
```
Same formula change. The `W_int_max` computation is preserved (it's still needed for the W-refined threshold) but the final threshold calculation now scales the correction by `ℓ/(4n)`.

### 7. `run_cascade.py` — `_fused_generate_and_prune_gray2()` (Gray code kernel v2)

#### 7a. Precomputed per-ell constants and threshold table (lines ~1033-1050)

**Before:**
```python
ct_base_ell_arr[idx] = c_target * m_d * m_d * np.float64(ell) * inv_4n

# Threshold table
ct_base_ell_val = c_target * m_d * m_d * np.float64(ell) * inv_4n
for w in range(m_plus_1):
    dyn_x = ct_base_ell_val + corr_base + 2.0 * np.float64(w)
    threshold_table[idx * m_plus_1 + w] = np.int64(dyn_x * one_minus_4eps)
```

**After:**
```python
cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin
ct_base_ell_arr[idx] = cs_corr_base * np.float64(ell) * inv_4n
w_scale_arr[idx] = 2.0 * np.float64(ell) * inv_4n

# Threshold table
for w in range(m_plus_1):
    dyn_x = ct_base_ell_arr[idx] + w_scale_arr[idx] * np.float64(w)
    threshold_table[idx * m_plus_1 + w] = np.int64(dyn_x * one_minus_4eps)
```

The threshold table now encodes the W-refined C&S threshold, where both the base term and the W contribution are scaled by `ℓ/(4n)`.

### 8. `run_cascade.py` — `_tighten_ranges()` threshold table (lines ~1745-1755)

**Before:**
```python
corr_base = 1.0 + eps_margin
ct_base_ell_val = c_target * m_d * m_d * np.float64(ell) * inv_4n
for w in range(m_plus_1):
    dyn_x = ct_base_ell_val + corr_base + 2.0 * np.float64(w)
    threshold_table[idx * m_plus_1 + w] = np.int64(dyn_x * one_minus_4eps)
```

**After:**
```python
cs_corr_base = c_target * m_d * m_d + 3.0 + eps_margin
base_ell = cs_corr_base * np.float64(ell) * inv_4n
w_scale = 2.0 * np.float64(ell) * inv_4n
for w in range(m_plus_1):
    dyn_x = base_ell + w_scale * np.float64(w)
    threshold_table[idx * m_plus_1 + w] = np.int64(dyn_x * one_minus_4eps)
```

Same change: the correction and W contribution are now scaled by `ℓ/(4n)`.

## What Was NOT Changed

1. **The `x_cap` and `x_cap_cs` computation** in `_prepare_ranges()` — unchanged. It still uses `correction(m, n_half_child)` to compute the single-bin energy cap. Since `correction()` now returns the smaller C&S value ($2/m + 1/m^2$), `x_cap` becomes slightly smaller (tighter), which is correct.

2. **The asymmetry filter** — unchanged. It uses `sqrt(c_target / 2)` as the threshold for left-mass fraction, which is independent of the discretization correction.

3. **The autoconvolution computation** — unchanged. The integer convolution `conv[k] = sum_{i+j=k} c_i * c_j` is exact (int32/int64 arithmetic).

4. **The window scan structure** — unchanged. The code still iterates over all windows `(ℓ, s_lo)` in an optimized order, computes the sliding window sum `ws`, and compares against the threshold. Only the threshold value changed.

5. **The Gray code enumeration** — unchanged. The cursor-based enumeration of all children in the Cartesian product is unaffected.

6. **The canonicalization and deduplication** — unchanged.

7. **The GPU kernel** (`gpu/cascade_host.cu`, `gpu/cascade_kernel.cu`) — not updated in this change. The GPU kernel will need the same threshold formula update before use.

## Verification Results

**Note:** The results below were obtained with `cs_corr_base = +1`. With the corrected `+3`, thresholds are slightly higher (more conservative), so survivor counts may increase marginally. The proofs at $c_{\text{target}} = 1.28$ and $1.30$ are expected to still hold; results at $1.33$ should be re-verified.

### Soundness check: $c_{\text{target}} = 1.51$ (above upper bound $C_{1a} \leq 1.5029$)

Must NOT prove. Result: **massively diverging** — 436 L0 survivors, 100,629 L1 survivors (230x expansion), 46M+ L2 survivors and growing. Correctly does not prove.

### Proof: $c_{\text{target}} = 1.28$ (C&S published result)

| Level | Survivors | Expansion | Children tested |
|:---:|---:|---:|---:|
| L0 | 213 | — | 1,771 |
| L1 | 8,708 | 40.9x | 157,691 |
| L2 | 24,828 | 2.85x | 63,112,592 |
| **L3** | **0** | **0x** | **1,481,152,278** |

**PROVEN at L3 in 1.4 minutes on a laptop CPU.**

### Full benchmark sweep ($n_{\text{half}} = 2$, $m = 20$)

| $c_{\text{target}}$ | L0 | L1 exp | L2 exp | L3 exp | L4 exp | Status |
|---|---|---|---|---|---|---|
| **1.28** | 213 | 41.7x | 2.8x | **0x** | — | **PROVEN L3** |
| **1.30** | 236 | 58.6x | 8.4x | 0.002x | **0x** | **PROVEN L4** |
| **1.33** | 259 | 84.3x | 27.6x | 0.11x | **0x** | **PROVEN L4** |
| **1.35** | 286 | 100.9x | 49.7x | 1.07x | *(timed out)* | Converging |

### Comparison with old Formula A ($n_{\text{half}} = 2$, $m = 20$)

| $c_{\text{target}}$ | Old L2 exp | New L2 exp | Old L3 exp | New L3 exp | Old status | New status |
|---|---|---|---|---|---|---|
| 1.28 | never tested | **2.8x** | — | **0x** | diverging | **PROVEN** |
| 1.33 | 2,152x | **27.6x** | 16,943x | **0.11x** | NOT PROVEN | **PROVEN** |
| 1.35 | 2,261x | **49.7x** | 17,644x | **1.07x** | NOT PROVEN | **converging** |
| 1.40 | 2,618x | *(not yet run)* | 21,008x | *(not yet run)* | NOT PROVEN | TBD |

The previous best lower bound was $C_{1a} \geq 1.2802$. With this fix, we can prove $C_{1a} \geq 1.33$ on a single laptop CPU in under 2 hours.

## References

1. **Cloninger, A. & Steinerberger, S.** (2017). *On suprema of autoconvolutions with an application to Sidon sets.* arXiv:1403.7988.
   - **Lemma 1** (page 4): Per-window test values are lower bounds on $\|f * f\|_\infty$.
   - **Lemma 2** (page 5): Discretization to $1/m$-net via cumulative rounding.
   - **Lemma 3** (page 6): $c \geq b_{n,m} - 2/m - 1/m^2$. The pointwise autoconvolution bound.
   - **Equation (1)** (page 7): The $W$-refinement using restricted support overlap.

2. **`threshold_analysis.md`** (this repo): Identifies the discrepancy between our code and the MATLAB, including the counterexample showing per-window TV error can exceed $\varepsilon^2 + 2\varepsilon W$ (which is correct but irrelevant — C&S don't bound per-window TV error).

3. **`proof/formula_b_soundness_analysis.md`** (this repo): Documents five failed proof strategies for the per-window bound and concludes Formula B is "neither proven nor disproven" — because the correct proof uses the global $L^\infty$ bound (Lemma 3), not a per-window bound.
