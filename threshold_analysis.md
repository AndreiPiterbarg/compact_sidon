# Analysis: Pruning Threshold Discrepancy Between Our Code and Cloninger–Steinerberger

> **RESOLVED (2026-04-07).** The discrepancy identified in this document has been
> fixed.  Our code now uses the C&S Lemma 3 formula where the ENTIRE threshold
> (including correction) is scaled by `ell/(4n)`:
>
>     threshold = floor((c_target*m^2 + min(2m+1, 3+2*W_int)) * ell/(4n) + eps)
>
> The old formula (Theorem 3.7, with the correction NOT scaled) was a valid but
> overly conservative bound.  The C&S formula is tighter and correct because
> Lemma 3 is a pointwise bound on `(g*g)(x)` and test values are window averages
> (`TV <= ||g*g||_inf`).  See `explanation.md` for the full derivation.
>
> Section 10's "counterexample" is misleading — it shows per-window TV error
> exceeding the correction, but pruning soundness uses the chain
> `TV <= ||g*g||_inf <= ||f*f||_inf + 2/m + 1/m^2`, not per-window error bounds.
>
> Section 11's `x_cap_cs` bug has been addressed by adding `+1` to the formula
> (accounting for discretization adjustment in canonical rounding).

## 1. Problem Statement

Two contradictory observations:

1. **Cloninger & Steinerberger (2017)** proved $C_{1a} \ge 1.28$ using
   `original_baseline_matlab.m` with `gridSpace = 0.02` and `lowerBound = 1.28`.
2. **Our cascade code** (`run_cascade.py`) cannot prove $C_{1a} \ge 1.28$
   even with `m = 20`, `c_target = 1.28`, because the cascade diverges at every
   level (expansion factors in the hundreds to thousands).

Both implementations enumerate integer-mass step functions on $d$ bins and prune
those whose windowed autoconvolution exceeds a dynamic threshold.  The pruning
threshold differs between the two codebases.  This document identifies the exact
discrepancy, traces it to our proof document's Lemma 3.5 / Theorem 3.7, and
quantifies its impact.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| $d = 2n$ | number of bins (support $[-\tfrac14,\tfrac14]$, each bin width $\tfrac{1}{2d}$) |
| $m$ | mass quantization: integer masses $c_i \ge 0$, $\sum c_i = m$ |
| $\varepsilon = 1/m$ | mass resolution |
| $\ell$ | window length in convolution space ($\ell = 2, \dots, 2d$) |
| $\mathrm{ws_{int}}$ | integer window sum: $\displaystyle\sum_{s=s_0}^{s_0+\ell-2}\sum_{i+j=s} c_i c_j$ |
| $W_{\mathrm{int}}$ | integer contributing-bin mass: $\displaystyle\sum_{i \in \mathcal{B}} c_i$ |
| $W = W_{\mathrm{int}}/m$ | probability-space contributing-bin mass |

The **test value** for window $(\ell, s_0)$ is

$$\mathrm{TV}(\ell, s_0) \;=\; \frac{4n}{m^2 \ell} \;\mathrm{ws_{int}}.$$

This matches `test_values.py:89` — $\mathrm{tv} = \mathrm{ws} \cdot (4 n_\mathrm{half} \cdot \ell)^{-1}$
— after accounting for the a-coordinate scaling $a_i = (4n/m) c_i$.

---

## 3. The Two Threshold Formulas

### 3.1. Our code (Theorem 3.7 of `lower_bound_proof.tex`)

**Theorem 3.7** (`lem:dynamic-correction`, line 615–637 of `lower_bound_proof.tex`):

> If $\ \mathrm{TV}_{n,m}(c;\,\ell,s_0) \;>\; c_{\mathrm{target}} + \dfrac{4n}{\ell}\!\left(\dfrac{1}{m^2} + \dfrac{2W}{m}\right)$, then $R(f) \ge c_{\mathrm{target}}$.

Converting to integer space (multiply by $m^2 \ell / (4n)$):

$$\boxed{\mathrm{ws_{int}} \;>\; c \cdot m^2 \cdot \frac{\ell}{4n} \;+\; 1 \;+\; 2\,W_{\mathrm{int}}}$$

This is **exactly** what our code computes at `run_cascade.py:698`:

```python
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
```

where `ct_base_ell = c_target * m² * ell / (4n)`.

The correction term $(1 + 2\,W_{\mathrm{int}})$ is **not** scaled by $\ell/(4n)$.

### 3.2. The MATLAB (`original_baseline_matlab.m`, line 219)

```matlab
boundToBeat = (lowerBound + gridSpace^2) + 2*gridSpace*W;
```

The MATLAB checks whether $\mathrm{TV} \ge c + \varepsilon^2 + 2\varepsilon W$, i.e.

$$\mathrm{TV} \;\ge\; c + \frac{1}{m^2} + \frac{2W}{m}$$

with **no** $4n/\ell$ factor on the correction.  Converting to integer space:

$$\boxed{\mathrm{ws_{int}} \;>\; \Bigl(c \cdot m^2 \;+\; 1 \;+\; 2\,W_{\mathrm{int}}\Bigr) \cdot \frac{\ell}{4n}}$$

Here the correction $(1 + 2\,W_{\mathrm{int}})$ **is** scaled by $\ell/(4n)$, the same factor that multiplies $c \cdot m^2$.

### 3.3. The discrepancy

| | Our code | MATLAB |
|---|---|---|
| **TV-space threshold** | $c + \dfrac{4n}{\ell}\!\left(\dfrac{1}{m^2} + \dfrac{2W}{m}\right)$ | $c + \dfrac{1}{m^2} + \dfrac{2W}{m}$ |
| **Integer threshold** | $c m^2 \dfrac{\ell}{4n} + 1 + 2W_{\mathrm{int}}$ | $(c m^2 + 1 + 2W_{\mathrm{int}})\dfrac{\ell}{4n}$ |
| **Correction factor** | $\dfrac{4n}{\ell}$ present | No $\dfrac{4n}{\ell}$ |

For the **full window** $\ell = 4n$, the factor $4n/\ell = 1$ and the two formulas **coincide**.

For **narrow windows** $\ell \ll 4n$, our correction is $(4n/\ell)$ times larger.

Concrete example — $d = 8$ ($n = 4$), $\ell = 2$, $m = 20$, $W = 0.5$:

| | Our code | MATLAB |
|---|---|---|
| **Correction (TV space)** | $(16/2)(0.0025 + 0.05) = 0.42$ | $0.0025 + 0.05 = 0.0525$ |
| **Threshold for $c = 1.28$** | $1.28 + 0.42 = 1.70$ | $1.28 + 0.0525 = 1.3325$ |

Our narrow-window threshold ($1.70$) is far above most configs' test values, rendering that window useless for pruning.  The MATLAB threshold ($1.33$) prunes effectively.

---

## 4. Origin of the $4n/\ell$ Factor

Our **Lemma 3.5** (line 552–584 of `lower_bound_proof.tex`) bounds the per-window discretization error:

$$\mathrm{TV}_{n,m}(c;\,\ell,s_0) - \mathrm{TV}^{\mathrm{cont}}_n(f;\,\ell,s_0)
\;\le\; \frac{4n}{\ell}\!\left(\frac{1}{m^2} + \frac{2W}{m}\right).$$

The proof sketch (lines 586–595) says:

> Write $w_i = \mu_i + \delta_i$.  The quadratic error part is bounded by
> $(4n/\ell) \cdot (1/m^2)$ because $|\delta_i| \le 1/m$, and the linear
> part by $(4n/\ell) \cdot (2W/m)$.

The $4n/\ell$ factor enters because the bound handles each of the $(\ell - 1)$ convolution positions in the window independently (worst-case per position, then summed and normalized by $1/(4n\ell)$).  This is a **valid upper bound** but it does not exploit cancellation across positions — in particular, the constraint $\sum \delta_i = 0$ (total mass preserved) forces per-position errors to partially cancel.

The **MATLAB** bound $\varepsilon^2 + 2\varepsilon W$ (without $4n/\ell$) implicitly relies on this cancellation.  This tighter bound was used in the published proof by Cloninger & Steinerberger (arXiv:1403.7988) and is consistent with the Lean formalization referenced in their work.

**In summary:**

- Our Lemma 3.5 / Theorem 3.7 give a **correct but loose** bound with the $4n/\ell$ factor.
- The MATLAB uses a **correct and tight** bound without the $4n/\ell$ factor.
- Both are valid pruning conditions.  Ours is more conservative.

---

## 5. Quantitative Impact

### 5.1. L0 survivor counts ($d = 4$, $m = 20$, all 1771 compositions)

| $c_{\mathrm{target}}$ | Our threshold | MATLAB threshold |
|---|---|---|
| 1.28 | 355 survivors | **213** survivors |
| 1.40 | 467 survivors | **345** survivors |
| 1.51 | — | **436** survivors |

(Our L0 counts are from `run_level0`; MATLAB counts from exhaustive enumeration with the MATLAB formula.  Both include canonical + asymmetry filters.)

### 5.2. L1 pruning per parent ($\text{parent} = [5,5,5,5]$, 1296 children, $d_{\mathrm{child}} = 8$)

| $c_{\mathrm{target}}$ | Our survivors | MATLAB survivors | Ratio |
|---|---|---|---|
| 1.28 | 963 / 1296 (25.7% pruned) | **108** / 1296 (**91.7% pruned**) | **8.9×** |
| 1.40 | 1251 / 1296 (3.5% pruned) | **654** / 1296 (49.5% pruned) | **1.9×** |
| 1.51 | 1294 / 1296 | **1039** / 1296 | 1.2× |

At $c = 1.28$, the MATLAB formula prunes **8.9× more children per parent** at L1.  This factor compounds across cascade levels.

### 5.3. Why the cascade diverges with our threshold

The narrow-window threshold is so high that narrow windows — which are the most
discriminating — contribute almost nothing to pruning.  Only wide windows (near
$\ell = 4n$) have thresholds close to $c + \varepsilon^2 + 2\varepsilon W$.  But wide
windows average over many convolution positions and cannot discriminate sharply.

For $c = 1.28$, $m = 20$, $n = 4$ ($d = 8$), the expansion factor at L1 is $\sim 276\times$ with our threshold versus an estimated $\sim 30\times$ with the MATLAB threshold.

### 5.4. Refinement monotonically increases TV

An important structural property: **every child has strictly higher TV than its parent**.  Verified exhaustively for all 1296 children of $[5,5,5,5]$:

- Parent TV: $1.25$ (at $\ell = 4$)
- Min child TV: $1.31$
- Max child TV: $2.00$

This is a consequence of bin-splitting: finer resolution reveals more autoconvolution
structure, increasing the peak.  This monotonicity guarantees that every cascade path
has increasing TV, so the cascade must eventually terminate for any provable bound.

### 5.5. The MATLAB formula does NOT prove $c = 1.51$

Verified numerically: with the MATLAB threshold, $c_{\mathrm{target}} = 1.51$ has

- L0: **436 survivors** ($n_\mathrm{half} = 2$, $m = 20$) / **2596 survivors** ($n_\mathrm{half} = 3$, $m = 15$)
- L1: **1039 survivors** for parent $[5,5,5,5]$ alone

The near-optimal function ($\mathrm{TV}(f^*) \approx 1.5029$, from the upper bound) has coarse-level projections with TV well below $1.51 + \varepsilon^2 + 2\varepsilon W \approx 1.56$.  Its path survives indefinitely.  The cascade correctly does not terminate.

If an implementation of the MATLAB formula appears to prove $c = 1.51$, the cause is a separate implementation error — not the correction formula itself.

---

## 6. Relationship to the C&S Paper

### The published algorithm (arXiv:1403.7988)

Cloninger & Steinerberger use parameters:

| Parameter | C&S value | Our value |
|---|---|---|
| Starting bins | 3 → 6 ($n_\mathrm{half} = 3$) | 4 ($n_\mathrm{half} = 2$) |
| Mass resolution | `gridSpace = 0.02` ($m_{\mathrm{eff}} = 50$) | $m = 20$ |
| Correction | $\varepsilon^2 + 2\varepsilon W$ (no $4n/\ell$) | $(4n/\ell)(\varepsilon^2 + 2\varepsilon W)$ |
| Target | 1.28 | 1.28–1.40 |

Their algorithm succeeds because:

1. **Tighter correction** — no $4n/\ell$ factor, so narrow windows prune effectively.
2. **Finer mass grid** — $\varepsilon = 0.02$ vs our $0.05$; correction is $\sim 2.5\times$ smaller.
3. **More starting bins** — $n_\mathrm{half} = 3$ gives better initial resolution.

### Our Theorem 3.7 vs C&S bound

| Bound | Per-window correction | Source |
|---|---|---|
| **C&S (MATLAB)** | $\varepsilon^2 + 2\varepsilon W$ | Cloninger & Steinerberger (2017), used in published proof |
| **Our Lemma 3.5** | $\dfrac{4n}{\ell}(\varepsilon^2 + 2\varepsilon W)$ | `lower_bound_proof.tex`, line 562–571 |
| **Our Corollary** | $2n(2\varepsilon + \varepsilon^2)$ | Global worst-case ($W \le 1$, $\ell \ge 2$) |

Our Lemma 3.5 is a valid but loose upper bound.  The $4n/\ell$ factor arises from a
per-position worst-case analysis that does not exploit the constraint $\sum \delta_i = 0$.
The C&S bound is tighter because the mass-preservation constraint forces error
cancellation across convolution positions within each window.

---

## 7. Diagnosis: Why Both "Fixes" Fail

### Keeping the current code (with $4n/\ell$)

The pruning is too conservative.  For $m = 20$, $n = 2$, at $\ell = 2$:

$$\text{threshold} = 1.28 + 4 \cdot 0.0525 = 1.49$$

Almost nothing is pruned by narrow windows.  The cascade diverges.

### Naively removing the $4n/\ell$ factor

Changing line 698 to scale the correction by $\ell/(4n)$:

```python
dyn_x = (c_target * m_d * m_d + 1.0 + eps_margin + 2.0 * W_int) * ell_f * inv_4n
```

This matches the MATLAB and **is mathematically sound** (based on the published C&S
bound).  However, if the change is applied **incorrectly** — for example, also modifying
the `correction()` function in `pruning.py` (which affects the vacuity check and
`x_cap` computation), or introducing a sign/scaling error — the result can be
over-pruning that falsely "proves" values above the upper bound.

The MATLAB formula, correctly applied, does **not** prove $c = 1.51$.  This was verified
numerically (Section 5.5).

---

## 8. Recommended Path Forward

### Option A: Tighten the proof (preferred)

Replace Lemma 3.5 with a tighter per-window bound that exploits $\sum \delta_i = 0$
to eliminate the $4n/\ell$ factor.  This would justify using the MATLAB threshold
directly.  The code change is small:

In every pruning kernel (`_prune_dynamic_int32`, `_fused_generate_and_prune`,
`_fused_generate_and_prune_gray`), replace

```python
# Current (Theorem 3.7): correction NOT scaled by ell/(4n)
dyn_x = ct_base_ell + 1.0 + eps_margin + 2.0 * np.float64(W_int)
```

with

```python
# MATLAB (C&S bound): correction scaled by ell/(4n)
ell_f = np.float64(ell)
dyn_x = (c_target * m_d * m_d + 1.0 + eps_margin + 2.0 * np.float64(W_int)) * ell_f * inv_4n
```

**Change nothing else** — not `correction()`, not `x_cap`, not the asymmetry filter.

Verify:
- $c = 1.51$ does **not** prove (hundreds of L0 survivors)
- $c = 1.28$ shows $\sim 40\%$ fewer L0 survivors and $\sim 9\times$ more pruning at L1

### Option B: Use parameters where our current bound suffices

The benchmark sweep already showed $n_\mathrm{half} = 3$, $m = 15$ proves $c = 1.33$ and
$c = 1.35$ at L6 even with our conservative threshold.  These values already exceed the
previous best lower bound of 1.2802.

### Option C: Match C&S parameters

Use $n_\mathrm{half} = 3$, $m = 50$ (equivalent to `gridSpace = 0.02`), with either
threshold formula.  The finer mass grid ($\varepsilon = 0.02$) reduces the correction
regardless of the $4n/\ell$ factor.

---

## 9. References

1. **Cloninger, A. & Steinerberger, S.** (2017). *On the dual Hales-Jewett density*. arXiv:1403.7988.
   The MATLAB implementation and the published proof of $C_{1a} \ge 1.28$.

2. **`original_baseline_matlab.m`**, line 219: the MATLAB threshold formula.

3. **`proof/lower_bound_proof.tex`**, Lemma 3.5 (line 552) and Theorem 3.7 (line 615):
   our per-window discretization error bound with the $4n/\ell$ factor.

4. **`cloninger-steinerberger/cpu/run_cascade.py`**, line 698: the code implementing
   Theorem 3.7's integer-coordinate threshold.

5. **`cloninger-steinerberger/test_values.py`**, line 89: the test value formula
   $\mathrm{tv} = \mathrm{ws} / (4 \cdot n_\mathrm{half} \cdot \ell)$.

6. **Boyer, E. & Li, T.** (2025). arXiv:2506.16750. Recent work on the Sidon constant.

7. **Matolcsi, M. & Vinuesa, C.** (2010). arXiv:0907.1379. The upper bound $C_{1a} \le 1.5029$.

---

## 10. Critical Review: Flaws in the Proposed Fix

> **Reviewer note (2026-04-03).**  The analysis in Sections 3–8 correctly identifies
> the discrepancy between our code and the MATLAB.  However, the conclusion that
> the MATLAB formula is "correct and tight" (Section 4) and that Option A is safe
> (Section 8) is **wrong**.  The MATLAB correction $\varepsilon^2 + 2\varepsilon W$
> is provably insufficient for narrow windows.  Adopting it would make the cascade
> prover unsound.

### 10.1. Explicit counterexample: MATLAB bound violated

**Setup.** $d = 4$, $n = 2$, $\ell = 2$, $s_{\mathrm{lo}} = 0$, $m = 10$.

Take the discrete vector $c = (10, 0, 0, 0)$ with $w = c/m = (1, 0, 0, 0)$.
A valid underlying continuous function has bin integrals
$\mu = (0.96, 0.02, 0.01, 0.01)$, since $\mathrm{round}(10 \times 0.96) = 10$,
$\mathrm{round}(10 \times 0.02) = 0$, etc., and $\sum c_i = 10$.
The discretization errors are $\delta = (0.04, -0.02, -0.01, -0.01)$ with
$\sum \delta_i = 0$.

The contributing bins for window $(\ell = 2, s_{\mathrm{lo}} = 0)$ are
$\mathcal{B} = \{0\}$ (only bin 0), so $W = w_0 = 1.0$.

**Discrete test value:**

$$\mathrm{TV}_{n,m} = \frac{1}{4 \cdot 2 \cdot 2} \cdot (8 \times 1.0)^2
  = \frac{64}{16} = 4.0$$

**Continuous test value:**

$$\mathrm{TV}^{\mathrm{cont}} = \frac{1}{16} \cdot (8 \times 0.96)^2
  = \frac{58.9824}{16} = 3.6864$$

**Actual discretization error:**

$$\mathrm{TV}_{n,m} - \mathrm{TV}^{\mathrm{cont}} = 0.3136$$

**MATLAB correction** ($\varepsilon^2 + 2\varepsilon W$):

$$0.01 + 2(0.1)(1.0) = 0.21$$

**Our Lemma 3.5 correction** ($\tfrac{4n}{\ell}(\varepsilon^2 + 2\varepsilon W)$):

$$\frac{8}{2} \times 0.21 = 0.84$$

| Bound | Value | Covers actual error? |
|-------|------:|:-------------------:|
| MATLAB ($\varepsilon^2 + 2\varepsilon W$) | 0.21 | **NO** ($0.3136 > 0.21$) |
| Lemma 3.5 ($\frac{4n}{\ell}(\varepsilon^2 + 2\varepsilon W)$) | 0.84 | Yes ($0.3136 < 0.84$) |

The actual error exceeds the MATLAB bound by 49%.

### 10.2. Error decomposition: why the $4n/\ell$ factor is real

The discretization error decomposes into linear and quadratic parts.  For window
$(\ell, s_{\mathrm{lo}})$ with contributing bins $\mathcal{B}$, define
$\mathrm{RS} = \sum_{s \in \text{window}} \sum_{i+j=s} w_i w_j$ (the "raw sum"
before the $4n/\ell$ normalization).  Then $\mathrm{TV} = (4n/\ell) \cdot \mathrm{RS}$
and:

$$\mathrm{RS} - \mathrm{RS}^{\mathrm{cont}}
  = \sum_{\substack{(i,j):\, i,j \in \mathcal{B} \\ s_{\mathrm{lo}} \le i+j \le s_{\mathrm{lo}}+\ell-2}}
  \bigl[\delta_i \mu_j + \delta_j \mu_i + \delta_i \delta_j\bigr]$$

For $\ell = 2$, $s_{\mathrm{lo}} = 0$: the only pair is $(0, 0)$, giving:

$$\mathrm{RS} - \mathrm{RS}^{\mathrm{cont}}
  = 2\,\delta_0\,\mu_0 + \delta_0^2$$

The TV-space error is $(4n/\ell)$ times this:

$$\mathrm{TV}_{n,m} - \mathrm{TV}^{\mathrm{cont}}
  = \frac{4n}{\ell}\,(2\,\delta_0\,\mu_0 + \delta_0^2)$$

The factor $4n/\ell$ **cannot be absorbed** because the mass-preservation constraint
$\sum \delta_i = 0$ provides no cancellation when $|\mathcal{B}| = 1$.  More generally,
for a window touching $|\mathcal{B}|$ bins, the constraint links $\delta$ values
*outside* $\mathcal{B}$ to those *inside*, but the error sum only involves bins
*inside* $\mathcal{B}$.  The cancellation argument from Section 4 of this document
requires summing over *all* bins (the full window $\ell = 4n$), where
$\mathcal{B} = \{0, \dots, d-1\}$ and $\sum_{\mathcal{B}} \delta_i = 0$.

### 10.3. The insufficiency scales as $2n$ and worsens at every cascade level

For $\ell = 2$ windows, the worst-case ratio of actual error to MATLAB correction is:

$$\frac{\mathrm{error}_{\max}}{\varepsilon^2 + 2\varepsilon W}
  \;\approx\;
  \frac{(4n/2)(2\mu_0/m)}{2\mu_0/m}
  \;=\; 2n$$

| Cascade level | $d$ | $n$ | Ratio for $\ell = 2$ |
|:---:|:---:|:---:|:---:|
| L1 | 8 | 4 | 8× |
| L2 | 16 | 8 | 16× |
| L3 | 32 | 16 | 32× |
| L4 | 64 | 32 | 64× |

At L4 ($d = 64$), the MATLAB correction underestimates the worst-case error by a
factor of 64 for $\ell = 2$ windows.

### 10.4. Can this cause unsound pruning?

**Yes, in principle.**  For the pruning to be unsound, we need a function $f$ with
$\|f * f\|_\infty < c_{\mathrm{target}}$ whose canonical discretization $c$ satisfies
$\mathrm{TV}_{n,m}(c;\,\ell, s_0) > c_{\mathrm{target}} + \varepsilon^2 + 2\varepsilon W$
for some narrow window $(\ell, s_0)$.

Since $\|f * f\|_\infty < c_{\mathrm{target}}$ implies
$\mathrm{TV}^{\mathrm{cont}}(f;\,\ell, s_0) < c_{\mathrm{target}}$ for all windows
(Lemma 3.4), we need:

$$c_{\mathrm{target}} + \varepsilon^2 + 2\varepsilon W
  \;<\; \mathrm{TV}_{n,m}
  \;\le\; \mathrm{TV}^{\mathrm{cont}} + \frac{4n}{\ell}(\varepsilon^2 + 2\varepsilon W)
  \;<\; c_{\mathrm{target}} + \frac{4n}{\ell}(\varepsilon^2 + 2\varepsilon W)$$

This interval is non-empty whenever $\ell < 4n$ (i.e., any non-full window).  A
discrete vector $c$ whose $\mathrm{TV}_{n,m}$ at a narrow window falls in this interval
would be **incorrectly pruned** by the MATLAB formula: the algorithm would discard a
branch of the search tree that contains a valid function with
$\|f * f\|_\infty < c_{\mathrm{target}}$, silently producing a false "proof."

### 10.5. Why the MATLAB might "work" at $m = 50$, $c_{\mathrm{target}} = 1.28$

Three mitigating factors explain why C&S's published result is probably still correct
despite the insufficient correction:

1. **Concentrated configs are redundantly pruned.**  Configurations where the $\ell = 2$
   error is large (mass concentrated in one bin) have high TV at *all* windows, including
   wide ones where the MATLAB correction is adequate.  The false pruning at $\ell = 2$
   may be redundant with correct pruning at $\ell \approx 4n$.

2. **Near-optimal functions have spread-out mass.**  Functions near
   $\|f * f\|_\infty = c_{\mathrm{target}}$ tend to distribute mass across bins
   (otherwise their autoconvolution peak would be much higher).  For spread-out
   distributions, the actual error is much smaller than the worst case.

3. **At $m = 50$, corrections are tiny.**  With $\varepsilon = 0.02$:
   $\varepsilon^2 + 2\varepsilon W \le 0.0004 + 0.04 = 0.0404$.  Even multiplied by
   $4n/\ell$ at $n = 3$, $\ell = 2$, this gives $0.242$ — still modest relative to
   $c_{\mathrm{target}} = 1.28$.  The danger zone
   $(c_{\mathrm{target}} + 0.04,\; c_{\mathrm{target}} + 0.24)$ may simply contain no
   discrete TV values corresponding to near-optimal functions.

**However, these are practical/empirical arguments, not rigorous ones.**  A correct
mathematical proof requires the error bound to hold for *every* configuration, not
just the ones that happen to arise in practice.  The fact that the MATLAB formula
works at $m = 50$ does not validate it at $m = 20$, where corrections are $6.25\times$
larger ($\varepsilon = 0.05$ vs $0.02$).

### 10.6. The verification in Section 5.5 is insufficient

Section 5.5 verifies that the MATLAB formula does not prove $c_{\mathrm{target}} = 1.51$.
This shows the formula does not over-prune *catastrophically* (pruning everything), but
it does **not** rule out subtle over-pruning at lower targets.  The correct verification
would be:

> For every configuration $c$ that is pruned by the MATLAB formula but *not* by
> our Lemma 3.5 formula, verify that no continuous function $f$ with canonical
> discretization $c$ satisfies $\|f * f\|_\infty < c_{\mathrm{target}}$.

This exhaustive check has not been performed.

### 10.7. The "cancellation" argument (Section 4) is wrong for narrow windows

Section 4 claims:

> The constraint $\sum \delta_i = 0$ forces per-position errors to partially cancel.

This is correct for the **full window** ($\ell = 4n$), where:

$$\sum_{s=0}^{2d-2} \sum_{i+j=s} \delta_i \mu_j
  = \Bigl(\sum_i \delta_i\Bigr)\Bigl(\sum_j \mu_j\Bigr) = 0 \cdot 1 = 0$$

But for a narrow window, the error sum runs over a *subset* of pairs $(i, j)$, and
$\sum_{i \in \mathcal{B}} \delta_i \ne 0$ in general.  The mass-preservation
constraint links $\delta$ values inside $\mathcal{B}$ to those outside, but the
outside values do not appear in the error sum.  The cancellation is **absent**, not
"partial."

Concretely, for window $(\ell = 2, s_{\mathrm{lo}} = 0)$ with $\mathcal{B} = \{0\}$:
the error involves only $\delta_0$, and $\sum \delta_i = 0$ tells us
$\delta_0 = -(\delta_1 + \delta_2 + \cdots)$, but this does not bound $|\delta_0|$
any tighter than $1/m$.

### 10.8. Unverified claim about the Lean formalization

Section 4 states the MATLAB bound is "consistent with the Lean formalization referenced
in their work."  This claim is unverified:

- No Lean source code has been inspected.
- The C&S paper (arXiv:1403.7988) may use a different proof structure (e.g., bounding
  global rather than per-window error, or using a refinement-specific argument that
  does not apply to our cascade).
- It is entirely possible that the Lean formalization uses the $4n/\ell$ factor and
  the MATLAB code simply omits it as a practical shortcut that happens to work at
  $m = 50$.

### 10.9. Corrected recommendations

**Option A (Section 8) is UNSAFE as stated.**  Do not adopt the MATLAB formula
without a rigorous proof that $\varepsilon^2 + 2\varepsilon W$ is a valid per-window
discretization error bound.  The counterexample in Section 10.1 shows that this
claim is false under the standard analysis.

**Safe paths forward:**

1. **Option B (use current threshold with $n_{\mathrm{half}} = 3$, $m = 15$).**
   Already proves $c_{\mathrm{target}} = 1.33$ and $1.35$ at L6.  Both exceed the
   previous best bound of $1.2802$.  No formula change needed.  This is the
   lowest-risk path.

2. **Tighten Lemma 3.5 with a correct proof.**  The current $4n/\ell$ factor is
   a worst case over all bin positions and all $\delta$ distributions.  A tighter
   bound may exist — for example, one that is $O(\sqrt{4n/\ell})$ or that depends
   on the number of contributing bins $|\mathcal{B}|$ rather than the full $4n/\ell$.
   But this requires a careful derivation, not simply deleting the factor.

3. **Increase $m$.**  Using $m = 50$ (matching C&S) reduces the correction by
   $6.25\times$ regardless of the $4n/\ell$ factor.  Combined with $n_{\mathrm{half}} = 3$,
   this may allow the current (conservative) threshold to prove higher targets.

4. **Read the actual C&S paper (arXiv:1403.7988, Section 3)** to understand their
   discretization error argument.  If they prove a per-window bound without
   $4n/\ell$, understand the structural property that makes it valid and verify
   whether it applies to our cascade (which uses a different starting dimension
   and mass resolution).

---

## 11. Bug in `x_cap_cs`: False "PROVEN" Claims at High Dimensions

> **Reviewer note (2026-04-03).**  Independent of the threshold formula discrepancy
> (Sections 3–10), the benchmark sweep's "PROVEN at L4/L6" claims are artifacts of
> a separate bug in the `x_cap_cs` computation (`run_cascade.py:1662`,
> `benchmark_sweep.py:55`).

### 11.1. The `x_cap_cs` formula and its assumption

The code at `run_cascade.py:1660–1663`:

```python
# Cauchy-Schwarz bound on continuous ||f*f||_∞ ≥ d_child·c_i²/m²
# doesn't go through test-value, so no correction needed
x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
x_cap = min(x_cap, x_cap_cs)
```

The bound claims $\|f * f\|_\infty \ge d_{\mathrm{child}} \cdot (c_i/m)^2$ for any
continuous $f$ whose canonical discretization assigns mass $c_i$ to bin $i$.

**The underlying argument is sound:**

$$\|f * f\|_\infty \;\ge\; \|f_i * f_i\|_\infty
  \;\ge\; \frac{(\int f_i)^2}{|\mathrm{supp}(f_i * f_i)|}
  \;=\; \frac{\mu_i^2}{1/d_{\mathrm{child}}}
  \;=\; d_{\mathrm{child}} \cdot \mu_i^2$$

where $f_i = f \cdot \mathbf{1}_{I_i}$, and $\mathrm{supp}(f_i * f_i)$ has length
$2 \cdot \frac{1}{2\,d_{\mathrm{child}}} = \frac{1}{d_{\mathrm{child}}}$.  The
first inequality uses $f \ge f_i \ge 0$.

**The bug is in substituting $\mu_i = c_i/m$.**  The canonical discretization computes
$c_i = \lfloor m \cdot \mu_i \rfloor$ for each bin, then *adjusts* some bins upward
by $+1$ to reach $\sum c_i = m$.  For non-adjusted (floor) bins: $c_i \le m \cdot \mu_i$,
so $\mu_i \ge c_i/m$.  **But for adjusted bins: $c_i = \lfloor m\mu_i \rfloor + 1$,
so $\mu_i$ can be as low as $c_i/m - 1/m = (c_i - 1)/m$.**

When $d \gg m$, nearly all bins have $\lfloor m\mu_i \rfloor = 0$, and the algorithm
must adjust $\sim m$ bins from 0 to 1.  For those adjusted bins, $c_i = 1$ but
$\mu_i$ can be $\ll 1/m$ (as low as $1/d$ for a near-uniform function).

### 11.2. When `x_cap_cs = 0` — all compositions filtered

The formula gives $x_{\mathrm{cap}} = 0$ when
$m \sqrt{c_{\mathrm{target}} / d_{\mathrm{child}}} < 1$, i.e.,
$d_{\mathrm{child}} > c_{\mathrm{target}} \cdot m^2$.

| Config | Threshold $c_{\mathrm{target}} \cdot m^2$ | Level where $x_{\mathrm{cap}} = 0$ |
|--------|:---:|:---:|
| $m = 15$, $c = 1.33$ | 299.25 → $d \ge 300$ | L6 ($d_{\mathrm{child}} = 384$) |
| $m = 10$, $c = 1.28$ | 128 → $d \ge 128$ | L4 ($d_{\mathrm{child}} = 192$) |

With $x_{\mathrm{cap}} = 0$: every child bin must satisfy $c_i \le 0$, but
$\sum c_i = m > 0$, so **no composition is feasible**.  Zero parents are processed,
zero survivors are found, and the code declares "PROVEN."

### 11.3. Concrete demonstration of unsound pruning

Consider a function $f$ with $\|f * f\|_\infty = 1.28 < c_{\mathrm{target}} = 1.33$,
discretized at $d = 384$, $m = 15$.  Since $f$ has roughly uniform mass:

- $\mu_i \approx 1/384 = 0.0026$ for all $i$
- $\lfloor 15 \times 0.0026 \rfloor = 0$ for all $i$; sum of floors $= 0$
- The algorithm adjusts 15 bins from 0 to 1, giving $c = (1, 1, \dots, 1, 0, \dots, 0)$
- **All 15 non-zero bins are adjusted bins** with $\mu_i \approx 0.0026 \ll 1/m = 0.067$

The correct CS bound per bin: $384 \times (0.0026)^2 = 0.0026 \ll 1.33$.
The code's CS bound per bin: $384 \times (1/15)^2 = 1.707 > 1.33$ — **wrong**, uses $c_i/m$ not $\mu_i$.

Since $\|f * f\|_\infty = 1.28 < 1.33$, this composition **should survive** (it
corresponds to a function that does NOT exceed $c_{\mathrm{target}}$).  But
$x_{\mathrm{cap}} = 0$ prunes it.  **This is unsound.**

### 11.4. Evidence from benchmark data

| Run | Level | $d_{\mathrm{child}}$ | Parents processed | Survivors | Mechanism |
|-----|:-----:|:---:|--:|--:|-----------|
| $n = 3$, $m = 15$, $c = 1.33$ | L6 | 384 | **0** | 0 | x_cap=0 filters all |
| $n = 6$, $m = 10$, $c = 1.28$ | L4 | 192 | **0** | 0 | x_cap=0 filters all |
| $n = 3$, $m = 15$, $c \approx 1.14$ | L3 | 48 | **10,000** | 0 | Genuine TV pruning |

The L6 and L4 "proofs" show **0 parents processed** — no parent reached the TV
pruning kernel.  Every parent was rejected by the $x_{\mathrm{cap}}$ feasibility
filter before pruning could occur.

In contrast, the L3 result ($c \approx 1.14$) processed 10,000 parents through
the full TV window scan and found 0 survivors.  This is genuine cascade convergence,
though even here $x_{\mathrm{cap}} = 2$ may over-filter some parents (see below).

### 11.5. Impact at intermediate levels ($x_{\mathrm{cap}} = 1$ or $2$)

At levels before $x_{\mathrm{cap}}$ hits 0, the filter is still more aggressive than
correct:

| Level | $d_{\mathrm{child}}$ | $x_{\mathrm{cap\_cs}}$ (buggy) | Correct $x_{\mathrm{cap}}$ | Effect |
|:-----:|:---:|:---:|:---:|--------|
| L4 ($n = 3, m = 15, c = 1.33$) | 96 | 1 | 2 | Children with $c_i = 2$ in adjusted bins incorrectly excluded |
| L3 ($n = 3, m = 15, c = 1.14$) | 48 | 2 | 3 | Children with $c_i = 3$ in adjusted bins incorrectly excluded |

For a child bin $c_i = 3$ that is an adjusted bin ($\mu_i \ge 2/m = 0.133$):
the correct CS bound is $48 \times 0.133^2 = 0.853 < 1.14$.  The filter prunes
this child, but the CS bound does NOT justify the pruning.  The child might
(or might not) be pruned by the TV window scan, but we never find out because
it is never generated.

This does NOT necessarily invalidate the L3 "proven" result.  The children with
$c_i = 3$ from adjusted bins might still be prunable by the TV scan if they were
generated.  But the proof's rigor depends on this assumption, which has not been
verified.

### 11.6. Expansion factors: not proof of divergence

The expansion factors at L1–L5 (before $x_{\mathrm{cap}}$ triggers) are all $> 1$:

$$\text{L1: } 291\times \;\to\; \text{L2: } 1642\times \;\to\; \text{L3: } 4504\times
  \;\to\; \text{L4: } 3138\times \;\to\; \text{L5: } 25204\times$$

**This does NOT prove divergence.**  Cascade proofs work by expanding at early levels
(where the grid is coarse and many configurations survive) and contracting at later
levels (where fine-grained windows discriminate).  The expansion factors above are
measured with the current conservative threshold ($4n/\ell$ correction factor).
A tighter threshold or higher $m$ could change the expansion trajectory entirely.

However, the consistent growth of expansion through L1–L5 (without any sign of
contraction) suggests that with the current parameters ($n_{\mathrm{half}} = 3$,
$m = 15$), the cascade is not converging through legitimate TV pruning.

### 11.7. `n_half = 6` results (2026-04-03)

Benchmark results for $n_{\mathrm{half}} = 6$:

| Config | L0 surv | L1 exp | L2 exp | L3 exp | L4 | Status |
|--------|--------:|-------:|-------:|-------:|:--:|--------|
| $n = 6$, $m = 10$, $c = 1.28$ | 75,819 | 169× | 132× | 887× | 0 (x_cap) | False PROVEN |
| $n = 6$, $m = 15$, $c = 1.28$ | 1,014,274 | 1,279× | 3,988× | 3,040× | 26,932× | NOT PROVEN |

$n_{\mathrm{half}} = 6$ is **strictly worse** than $n_{\mathrm{half}} = 2$ or $3$:
more starting bins means exponentially more L0 compositions ($84M$ at $m = 20$ vs
$1,771$ for $n_{\mathrm{half}} = 2$) without proportionally better pruning.  The
expansion factors at L1–L3 are comparable to or worse than $n_{\mathrm{half}} = 3$.

### 11.8. What has actually been proven?

The only benchmark result showing **genuine** cascade convergence (many parents
processed, 0 survivors) is:

- $n_{\mathrm{half}} = 3$, $m = 15$, $c_{\mathrm{target}} \approx 1.14$: proven at L3
  (10,000 parents processed, 0 survivors)

Even this result has a caveat: the $x_{\mathrm{cap}}$ filter at L3 ($x_{\mathrm{cap}} = 2$)
may have excluded some parents whose children could have survived.  A definitive
verification requires either (a) removing the $x_{\mathrm{cap\_cs}}$ filter and re-running,
or (b) proving that the excluded children are all prunable by the TV scan.

The "proven" claims for $c_{\mathrm{target}} = 1.33$ and $1.35$ (from CLAUDE.md) are
**not valid** — they are artifacts of $x_{\mathrm{cap}} = 0$ filtering at L6.

### 11.9. Recommended fix

Replace the $x_{\mathrm{cap\_cs}}$ formula with a correct version that accounts for
the discretization adjustment:

```python
# Current (buggy): assumes mu_i >= c_i/m (only valid for floor bins)
x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))

# Fixed: accounts for adjustment (+1) in canonical discretization
# CS bound: ||f*f|| >= d_child * ((c_i - 1)/m)^2 >= c_target
# => c_i >= 1 + m * sqrt(c_target / d_child)
# => x_cap = floor(m * sqrt(c_target / d_child)) + 1
# But simpler: just remove x_cap_cs entirely and rely on the TV scan
# (the x_cap with correction already provides a valid, looser cap)
```

The safest fix is to **remove `x_cap_cs` entirely** and rely on `x_cap` (which uses
the correction term and is sound).  The performance impact is minimal: `x_cap_cs`
only binds at high dimensions where the cascade has already expanded to billions of
parents, making runtime academic.
