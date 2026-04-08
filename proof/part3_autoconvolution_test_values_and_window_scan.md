> **Note (2026-04-07):** The code now uses the C&S fine grid (compositions summing to $S = 4nm$, heights $= c_i/m$). The height conversion formula $a_i = c_i \cdot 4n/m$ referenced below applies to the old coarse-grid parameterization. Under the fine grid, $a_i = c_i/m$ directly, and the test value formulas simplify accordingly.

# Part 3: Autoconvolution, Test Values & Window Scan — Verification Report

**Scope:** Verify the core mathematical computation: given a mass vector, compute the test value correctly.

**Result: ALL 71 CHECKS PASSED**

**Verification script:** `tests/verify_part3.py`

**Files audited:**

- `test_values.py` (150 lines): `_test_values_jit`, `compute_test_values_batch`, `compute_test_value_single`
- `solvers.py:136–142` (d=4 unrolled convolution), `solvers.py:268–278` (d=6 unrolled convolution)
- `solvers.py:149–164` (d=4 window scan), `solvers.py:286–301` (d=6 window scan), `solvers.py:461–491` (generic window scan)
- `run_cascade.py:89–113` (lo\_bin/hi\_bin dynamic threshold, int32 path), `run_cascade.py:163–186` (int64 path), `run_cascade.py:662–691` (cascade kernel)
- `cpu/benchmark.py:249–261` (`dynamic_threshold`)

**MATLAB mapping:**

- Lines 195–197: `functionMult = matrix_tmp(:,pairsGpu(:,1)) .* matrix_tmp(:,pairsGpu(:,2))` → Python `conv[i+j] += a_i * a_j`
- Lines 210–212: `convFunctionVals = functionMult(indices,:) * sumIndicesStore{j}` → Python prefix-sum + window extraction
- Line 215: `convFunctionVals = convFunctionVals * (2*numBins)/j` → Python `inv_norm = 1/(4*n*ell)`
- Lines 109–113: `sumIndicesStore{j}`, `binsContribute{j}` → Python uses prefix sums on `lo_bin`/`hi_bin`

---

## Item 1: Autoconvolution Formula

**Claim:** `conv[k] = Σ_{i+j=k} a_i·a_j` for `k = 0, ..., 2d-2`. The symmetry optimization `conv[2i] += a_i²; conv[i+j] += 2·a_i·a_j` for `j > i` gives the same result.

### Proof

The full double sum can be partitioned by the relationship between $i$ and $j$:

$$\text{conv}[k] = \sum_{\substack{i+j=k \\ 0 \le i,j < d}} a_i a_j = \underbrace{\sum_{\substack{i+j=k \\ i=j}} a_i^2}_{\text{diagonal}} + \underbrace{2 \sum_{\substack{i+j=k \\ i < j}} a_i a_j}_{\text{off-diagonal}}$$

**Diagonal terms:** Occur only when $k = 2i$ with $0 \le i < d$. Each contributes $a_i^2$. The code writes `conv[2*ii] += ai * ai` for each `ii`, matching exactly.

**Off-diagonal terms:** For each pair $(i, j)$ with $i < j$ and $i + j = k$, the full sum includes both $(i, j)$ and $(j, i)$, each contributing $a_i a_j$. Total contribution: $2 a_i a_j$. The code writes `conv[ii + jj] += 2.0 * ai * (c[jj] * scale)` for `jj > ii`, matching exactly.

Since both approaches compute the same partitioned sum, they are algebraically identical. $\square$

**Code locations:**
- Symmetry-optimized: `solvers.py:460–467` (generic kernel)
- Full double loop: `test_values.py:72–77` (generic fallback in `_test_values_jit`)

### Computational Verification

| d | Trials | Max diff (naive vs symmetry) |
|---|--------|----------------------------|
| 2 | 20 | 0.0 |
| 3 | 20 | 9.09e-13 |
| 4 | 20 | 4.55e-13 |
| 5 | 20 | 4.55e-13 |
| 6 | 20 | 4.55e-13 |
| 7 | 20 | 4.55e-13 |
| 8 | 20 | 2.27e-13 |

All differences are at machine epsilon level (floating-point reordering).

### Verdict: PROVED + VERIFIED

---

## Item 2: d=4 Unrolled Convolution Coefficients

**Claim:** The coefficients at `test_values.py:46–52` match $(a_0 + a_1 x + a_2 x^2 + a_3 x^3)^2$.

**Code:**
```python
conv[0] = a0 * a0
conv[1] = 2.0 * a0 * a1
conv[2] = a1 * a1 + 2.0 * a0 * a2
conv[3] = 2.0 * (a0 * a3 + a1 * a2)
conv[4] = a2 * a2 + 2.0 * a1 * a3
conv[5] = 2.0 * a2 * a3
conv[6] = a3 * a3
```

### Proof by Direct Expansion

$$P(x)^2 = \left(\sum_{i=0}^{3} a_i x^i\right)^2 = \sum_{k=0}^{6} \left(\sum_{\substack{i+j=k \\ 0 \le i,j \le 3}} a_i a_j\right) x^k$$

Enumerating all contributing pairs $(i, j)$ for each power of $x$:

| $k$ | Pairs $(i, j)$ | Coefficient |
|-----|----------------|-------------|
| 0 | $(0,0)$ | $a_0^2$ |
| 1 | $(0,1), (1,0)$ | $2 a_0 a_1$ |
| 2 | $(0,2), (1,1), (2,0)$ | $a_1^2 + 2 a_0 a_2$ |
| 3 | $(0,3), (1,2), (2,1), (3,0)$ | $2(a_0 a_3 + a_1 a_2)$ |
| 4 | $(1,3), (2,2), (3,1)$ | $a_2^2 + 2 a_1 a_3$ |
| 5 | $(2,3), (3,2)$ | $2 a_2 a_3$ |
| 6 | $(3,3)$ | $a_3^2$ |

Each line matches the code term-for-term. $\square$

The same coefficients appear in `solvers.py:136–142` (d=4 fused kernel).

### Computational Verification

- 50 random trials against naive double-loop AND `numpy.poly1d` polynomial squaring: max diff < 1e-10.
- Specific known value: $a = (1,2,3,4) \Rightarrow \text{conv} = [1, 4, 10, 20, 25, 24, 16]$. Matches.

### Verdict: PROVED + VERIFIED

---

## Item 3: d=6 Unrolled Convolution Coefficients

**Claim:** The coefficients at `test_values.py:60–70` match $(a_0 + a_1 x + \cdots + a_5 x^5)^2$.

**Code:**
```python
conv[0]  = a0 * a0
conv[1]  = 2.0 * a0 * a1
conv[2]  = 2.0 * a0 * a2 + a1 * a1
conv[3]  = 2.0 * (a0 * a3 + a1 * a2)
conv[4]  = 2.0 * (a0 * a4 + a1 * a3) + a2 * a2
conv[5]  = 2.0 * (a0 * a5 + a1 * a4 + a2 * a3)
conv[6]  = 2.0 * (a1 * a5 + a2 * a4) + a3 * a3
conv[7]  = 2.0 * (a2 * a5 + a3 * a4)
conv[8]  = 2.0 * a3 * a5 + a4 * a4
conv[9]  = 2.0 * a4 * a5
conv[10] = a5 * a5
```

### Proof by Direct Expansion

For each $k \in \{0, \ldots, 10\}$, enumerate all pairs $(i, j)$ with $i + j = k$ and $0 \le i, j \le 5$:

| $k$ | Pairs | Coefficient |
|-----|-------|-------------|
| 0 | $(0,0)$ | $a_0^2$ |
| 1 | $(0,1),(1,0)$ | $2a_0 a_1$ |
| 2 | $(0,2),(1,1),(2,0)$ | $2a_0 a_2 + a_1^2$ |
| 3 | $(0,3),(1,2),(2,1),(3,0)$ | $2(a_0 a_3 + a_1 a_2)$ |
| 4 | $(0,4),(1,3),(2,2),(3,1),(4,0)$ | $2(a_0 a_4 + a_1 a_3) + a_2^2$ |
| 5 | $(0,5),(1,4),(2,3),(3,2),(4,1),(5,0)$ | $2(a_0 a_5 + a_1 a_4 + a_2 a_3)$ |
| 6 | $(1,5),(2,4),(3,3),(4,2),(5,1)$ | $2(a_1 a_5 + a_2 a_4) + a_3^2$ |
| 7 | $(2,5),(3,4),(4,3),(5,2)$ | $2(a_2 a_5 + a_3 a_4)$ |
| 8 | $(3,5),(4,4),(5,3)$ | $2a_3 a_5 + a_4^2$ |
| 9 | $(4,5),(5,4)$ | $2a_4 a_5$ |
| 10 | $(5,5)$ | $a_5^2$ |

Each matches the code. $\square$

The same coefficients appear in `solvers.py:268–278` (d=6 fused kernel).

### Computational Verification

- 50 random trials against naive double-loop AND polynomial squaring: max diff < 1e-10.
- `solvers.py` d=6 duplicate verified identical to `test_values.py` d=6.

### Verdict: PROVED + VERIFIED

---

## Item 4: Generic Loop Equivalence

**Claim:** The generic double loop `conv[i+j] += a_i·a_j` (no symmetry, `test_values.py:72–77`) produces the same result as the symmetry-optimized version (`solvers.py:460–467`).

### Proof

This is an immediate corollary of Item 1. The full double loop computes $\text{conv}[k] = \sum_{i+j=k} a_i a_j$ by definition. The symmetry-optimized version computes the same sum via the diagonal/off-diagonal partition. Both yield identical results. $\square$

### Computational Verification

| d | Trials | Method | Max diff |
|---|--------|--------|----------|
| 2–6 | 100 each | naive vs symmetry | ≤ 9.09e-13 |
| 7–8 | 30 each | naive vs symmetry | ≤ 4.55e-13 |
| 10, 12 | 30 each | naive vs symmetry | ≤ 4.55e-13 |
| 4 | 100 | unrolled vs generic | 2.27e-13 |
| 6 | 100 | unrolled vs generic | 5.68e-14 |

**Gap filled:** The unrolled d=4/d=6 coefficients are now verified against the generic loop, which was an identified gap in the existing test suite.

### Verdict: PROVED + VERIFIED

---

## Item 5: Normalization Equivalence MATLAB ↔ Python

**Claim:** The MATLAB and Python test-value computations are algebraically identical despite different coordinate conventions.

### Definitions

| | MATLAB | Python |
|---|--------|--------|
| Mass coordinate | $f_i$ (raw mass in bin $i$) | $a_i$ (density in bin $i$) |
| Bin width | $w = 1/(2 \cdot \text{numBins}) = 1/(4n)$ | $w = 1/(4n)$ |
| Relation | $f_i = a_i \cdot w = a_i / (4n)$ | $a_i = c_i \cdot 4n/m$ |
| Window sum | $W_\text{matlab} = \sum f_i f_j$ | $W_\text{python} = \sum a_i a_j$ |
| Normalization | Multiply by $(2 \cdot \text{numBins})/j$ | Divide by $4n \cdot \ell$ |

### Proof

**Step 1 — Window sum conversion.**

$$W_\text{matlab} = \sum_{(i,j) \in \text{window}} f_i f_j = \sum_{(i,j) \in \text{window}} \frac{a_i}{4n} \cdot \frac{a_j}{4n} = \frac{W_\text{python}}{(4n)^2}$$

**Step 2 — MATLAB normalization.**

MATLAB line 215: `convFunctionVals = convFunctionVals * (2*numBins)/j`.

With $\text{numBins} = d = 2n$ and $j = \ell$:

$$\text{tv}_\text{matlab} = W_\text{matlab} \cdot \frac{2d}{\ell} = \frac{W_\text{python}}{(4n)^2} \cdot \frac{4n}{\ell} = \frac{W_\text{python}}{4n \cdot \ell}$$

**Step 3 — Python normalization.**

`test_values.py:89`: `inv_norm = 1.0 / (4.0 * n_half * ell)`, so:

$$\text{tv}_\text{python} = W_\text{python} \cdot \frac{1}{4n \cdot \ell} = \frac{W_\text{python}}{4n \cdot \ell}$$

**Conclusion:** $\text{tv}_\text{matlab} = \text{tv}_\text{python}$. $\square$

### Window Correspondence

MATLAB window of "size $j$" covers $j$ half-bin positions. A pair with 1-indexed sum $S$ is assigned to half-bin positions $S-1$ and $S$. The pair is counted in a window if both positions are within the window: $k \le S-1$ and $S \le k+j-1$, giving $k+1 \le S \le k+j-1$, which is $j-1$ possible sum values.

Python window of "size $\ell$" contains $n_\text{cv} = \ell - 1$ convolution entries.

With $j = \ell$: both cover the same $\ell - 1$ convolution positions. $\square$

### Dynamic Threshold Equivalence

MATLAB line 219:
```
boundToBeat = (lowerBound + gridSpace^2) + 2*gridSpace * (matrix_tmp * binsContribute{j})
```

With `gridSpace = 1/m` and $f_i = c_i / m$:

$$\text{boundToBeat} = c_\text{target} + \frac{1}{m^2} + \frac{2}{m} \sum_{\text{contributing } i} \frac{c_i}{m} = c_\text{target} + \frac{1 + 2 W_\text{int}}{m^2}$$

Python (`benchmark.py:261`):
```python
return c_target + (1.0 + 2.0 * W_int) * inv_m_sq + fp_margin
```

These are identical (modulo the $10^{-9}$ floating-point safety margin). $\square$

### Computational Verification

Simulated the MATLAB approach (sum raw mass products, multiply by $2d/\ell$) against Python's approach for $n_\text{half} \in \{2, 3, 4\}$, 20 random trials each. Max difference: < 1e-10. All match.

### Verdict: PROVED + VERIFIED

---

## Item 6: Window-Max Computation and Off-by-One Verification

**Claim:** The prefix-sum window-max computation correctly evaluates $\max_{\ell, s} \frac{1}{4n\ell} \sum_{k=s}^{s+\ell-2} \text{conv}[k]$ with no off-by-one errors.

### Proof of Off-by-One Correctness

**Window endpoints:**
$$s_\text{hi} = s_\text{lo} + (n_\text{cv} - 1) = s_\text{lo} + (\ell - 1) - 1 = s_\text{lo} + \ell - 2$$

**Number of entries:** $s_\text{hi} - s_\text{lo} + 1 = (\ell - 2) + 1 = \ell - 1 = n_\text{cv}$. Correct.

**Iteration bounds:**
- $s_\text{lo}$ ranges from 0 to $\text{conv\_len} - n_\text{cv} = (2d - 1) - (\ell - 1) = 2d - \ell$
- $s_\text{hi}$ ranges from $\ell - 2$ to $(2d - \ell) + (\ell - 2) = 2d - 2 = \text{conv\_len} - 1$

So $s_\text{hi}$ never exceeds the last valid index. Correct.

### Proof of Prefix-Sum Correctness

After in-place prefix sum: $\text{conv}[k] = \sum_{t=0}^{k} \text{conv}_\text{orig}[t]$.

The window sum:
$$ws = \text{conv}[s_\text{hi}] - (s_\text{lo} > 0\ ?\ \text{conv}[s_\text{lo} - 1] : 0)$$
$$= \sum_{t=0}^{s_\text{hi}} \text{conv}_\text{orig}[t] - \sum_{t=0}^{s_\text{lo}-1} \text{conv}_\text{orig}[t] = \sum_{t=s_\text{lo}}^{s_\text{hi}} \text{conv}_\text{orig}[t]$$

This is the standard prefix-sum sliding window technique. Correct. $\square$

### Computational Verification

| d | Trials | Max diff (prefix-sum vs direct) |
|---|--------|-------------------------------|
| 2, 3, 4, 5, 6, 8 | 30 each | 0 (exact) |

Also verified that `compute_test_value_single` matches the naive reference implementation for d=2,4,6,8 (20 trials each, max diff = 0).

### Verdict: PROVED + VERIFIED

---

## Item 7: Early-Stop Correctness

### Claim 1: The $\ell=2$ shortcut (`test_values.py:29–36`) returns a lower bound.

**Code:**
```python
max_a = max(a_i for i in range(d))
if max_a * max_a * inv_ell2 > early_stop:
    result[b] = max_a * max_a * inv_ell2
    continue
```

**Proof:** The shortcut computes $\frac{\max_i(a_i)^2}{4n \cdot 2}$, which is the test value for $\ell = 2$ at the window containing only $\text{conv}[2i_{\max}] = a_{i_{\max}}^2$. Since the true test value maximizes over ALL $(\ell, s_\text{lo})$ combinations including this one:

$$\text{tv}_\text{true} \ge \frac{a_{i_{\max}}^2}{8n} = \text{tv}_\text{shortcut}$$

If the lower bound exceeds `early_stop`, the true value also exceeds it. $\square$

### Claim 2: Early exit within the window loop (`test_values.py:97–99`) returns a lower bound.

**Proof:** The variable `best` tracks the maximum test value found so far across all checked windows. When `best > early_stop`, the loop exits. Since unchecked windows can only increase `best`:

$$\text{tv}_\text{true} \ge \text{best} > \text{early\_stop}$$

For pruning (config is pruned if $\text{tv} > c_\text{target}$), the lower bound suffices. $\square$

### Claim 3: `compute_test_value_single` returns the true maximum.

**Proof:** By inspection of `test_values.py:127–149`: no `do_early` check, no `done` flag, no `break` on threshold. The loop exhaustively evaluates all $(\ell, s_\text{lo})$ combinations and returns the global maximum. $\square$

### Computational Verification

For $d \in \{4, 6\}$, 200 random integer configurations, 5 threshold levels each:
- Early-stop value never exceeds no-early-stop value (max diff: 0).
- When early-stop truncates, the returned value always exceeds the threshold.

### Verdict: PROVED + VERIFIED

---

## Item 8: `compute_test_values_batch` vs `compute_test_value_single`

**Claim:** When `prune_target=0` (no early-stop), the batch and single functions return identical results:
$$\texttt{compute\_test\_values\_batch}(c, n, m, 0)[i] = \texttt{compute\_test\_value\_single}(c_i \cdot 4n/m,\ n)$$

### Proof

**Input conversion:** Batch computes `a_i = batch_int[b, i] * scale` where `scale = 4.0 * n_half / m = 4n/m`. Single takes `a_i` directly. If single receives $c_i \cdot 4n/m$, both functions operate on the same $a_i$ values.

**Convolution computation:** Batch uses unrolled coefficients for $d = 4, 6$ and the generic double loop for other $d$. Single uses the generic double loop. Items 1–4 proved these are algebraically equivalent.

**Window-max computation:** With `prune_target=0` (`early_stop=0.0`), the batch function's `do_early` flag is `False`, so no early exit occurs. Both functions execute the same exhaustive loop over all $(\ell, s_\text{lo})$ with the same normalization $1/(4n\ell)$.

Therefore the results agree (up to floating-point reordering). $\square$

### Computational Verification

| d | m values | Configs tested | Max diff |
|---|----------|----------------|----------|
| 4 | 5, 10, 20, 50, 100 | 56–500 each | ≤ 1.33e-15 |
| 6 | 5, 10, 16 | 252–500 each | ≤ 1.78e-15 |
| 8 | 5, 8 | 200–300 each | ≤ 1.78e-15 |

All differences are at double-precision machine epsilon.

### Verdict: PROVED + VERIFIED

---

## Item 9: MATLAB `binsContribute{j}` vs Python `lo_bin`/`hi_bin`

**Claim:** Python's formulas `lo_bin = max(0, s_lo - (d-1))` and `hi_bin = min(d-1, s_lo + ell - 2)` correctly identify the set of bins contributing mass to convolution window $(s_\text{lo}, \ell)$, and this set is identical to MATLAB's `binsContribute{j}`.

### Proof — Python Formulas

**Definition:** Bin $i$ contributes to window $[s_\text{lo}, s_\text{hi}]$ iff there exists $j \in \{0, \ldots, d-1\}$ such that $s_\text{lo} \le i + j \le s_\text{hi}$.

Rearranging: $s_\text{lo} - i \le j \le s_\text{hi} - i$. Intersecting with $0 \le j \le d-1$:

$$\max(0, s_\text{lo} - i) \le j \le \min(d-1, s_\text{hi} - i)$$

This has a solution iff:
- $s_\text{lo} - i \le d - 1$ (i.e., $i \ge s_\text{lo} - (d-1)$), AND
- $s_\text{hi} - i \ge 0$ (i.e., $i \le s_\text{hi} = s_\text{lo} + \ell - 2$)

Combined with $0 \le i \le d-1$:

$$\text{lo\_bin} = \max(0,\ s_\text{lo} - (d-1)), \qquad \text{hi\_bin} = \min(d-1,\ s_\text{lo} + \ell - 2)$$

The contributing bins form the contiguous range $[\text{lo\_bin},\ \text{hi\_bin}]$. $\square$

### Proof — MATLAB Correspondence

MATLAB's `binsContribute{j}` (lines 109–114) marks bin $i_1$ (1-indexed) as contributing to window $k$ of size $j$ iff there exists pair $(i_1, j_1)$ with $1 \le j_1 \le d$ such that the pair's both half-bin positions fall within the window.

MATLAB condition: $k+1 \le i_1 + j_1 \le k + j - 1$ (pair sum $S = i_1 + j_1$).

Converting to 0-indexed ($i = i_1 - 1$, $j' = j_1 - 1$, $s_\text{lo} = k - 1$, $\ell = j$):

$$s_\text{lo} + 1 \le (i+1) + (j'+1) \le s_\text{lo} + 1 + \ell - 1$$
$$s_\text{lo} - 1 \le i + j' \le s_\text{lo} + \ell - 2$$

Wait — this appears off by one, but simplifying correctly:

$$k + 1 \le S \le k + j - 1$$
$$S = (i+1) + (j'+1) = i + j' + 2$$
$$k + 1 \le i + j' + 2 \le k + j - 1$$
$$k - 1 \le i + j' \le k + j - 3$$

With $s_\text{lo} = k - 1$ and $\ell = j$:
$$s_\text{lo} \le i + j' \le s_\text{lo} + \ell - 2$$

This is **exactly** the Python condition $s_\text{lo} \le i + j \le s_\text{hi}$ where $s_\text{hi} = s_\text{lo} + \ell - 2$.

Therefore MATLAB's `binsContribute{j}` and Python's `lo_bin`/`hi_bin` identify the **same** set of contributing bins. $\square$

### Proof — `W_int` via Prefix Sum

`W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]` where `prefix_c[i] = c_0 + c_1 + ... + c_{i-1}`.

This evaluates $\sum_{i=\text{lo\_bin}}^{\text{hi\_bin}} c_i$, which is the sum of integer mass coordinates in the contributing bins. By the formulas above, this matches MATLAB's `matrix_tmp * binsContribute{j}` after coordinate conversion ($f_i = c_i / m$). $\square$

### Computational Verification

**Formula verification (exhaustive over all valid (ell, s_lo)):**

| d | Windows checked | Mismatches |
|---|----------------|------------|
| 2 | 4 | 0 |
| 3 | 15 | 0 |
| 4 | 34 | 0 |
| 5 | 63 | 0 |
| 6 | 104 | 0 |
| 8 | 224 | 0 |

**W_int prefix-sum verification:** For $d \in \{4, 6, 8\}$, 30 random integer configurations each, all $(\ell, s_\text{lo})$ combinations checked against brute-force bin summation. Zero mismatches.

### Verdict: PROVED + VERIFIED

---

## Additional: Unrolled d=4/d=6 vs Generic Path in `_test_values_jit`

This fills the gap identified in the checklist: "No test verifies the unrolled d=4/d=6 coefficients against the generic loop."

The batch function (`compute_test_values_batch`) dispatches to unrolled code for $d = 4, 6$ and the generic double-loop for all other $d$. The single function (`compute_test_value_single`) always uses the generic double-loop. Comparing batch (unrolled) vs single (generic) at `prune_target=0` tests the unrolled coefficients:

| d | m | Configs | Max diff |
|---|---|---------|----------|
| 4 | 10 | 286 | 8.88e-16 |
| 4 | 20 | 1,000 | 1.33e-15 |
| 4 | 50 | 1,000 | 1.33e-15 |
| 6 | 8 | 500 | 2.22e-16 |
| 6 | 12 | 500 | 8.88e-16 |
| 6 | 16 | 500 | 2.22e-16 |
| 8 | 5 | 200 | 1.78e-15 |
| 8 | 8 | 200 | 4.44e-16 |

All at machine epsilon.

---

## Additional: Analytic Verification of Known Test Values

| Configuration | $n_\text{half}$ | Expected conv | Expected tv | Got tv | Match |
|--------------|----------------|--------------|-------------|--------|-------|
| $a = [2,2,2,2]$ (uniform) | 2 | $[4,8,12,16,12,8,4]$ | 1.25 (at $\ell=4$) | 1.25 | Yes |
| $a = [8,0,0,0]$ (concentrated) | 2 | $[64,0,0,0,0,0,0]$ | 4.0 (at $\ell=2$) | 4.0 | Yes |
| $a = [2,2,2,2,2,2]$ (uniform d=6) | 3 | — | $4/3$ (at $\ell=6$) | 1.3333... | Yes |

**Derivation for uniform d=4:** $a_i = 2$ for all $i$. $\text{conv}[k] = 4 \cdot |\{(i,j): i+j=k, 0 \le i,j \le 3\}|$. The number of pairs peaks at $k=3$ with 4 pairs, giving $\text{conv}[3] = 16$. The best window is $\ell=4$, $n_\text{cv}=3$, window $\{2,3,4\}$: $W = 12 + 16 + 12 = 40$, $\text{tv} = 40 / (4 \cdot 2 \cdot 4) = 40/32 = 1.25$.

---

## Additional: Test Value is a Lower Bound on $\|f*f\|_\infty$

**Theorem (Cloninger-Steinerberger):** For a piecewise-constant function $f$ with density $a_i$ on bin $i$ (width $1/(4n)$) and $\int f = 1$, the discrete test value satisfies $\text{tv} \le \|f*f\|_\infty$.

**Verification:** For $d \in \{4, 6\}$, 200 random normalized configurations each ($\sum a_i = 4n$ so $\int f = 1$). Computed $\|f*f\|_\infty$ by numerical integration on a 2000-point grid. The test value never exceeded the continuous supremum. Zero violations out of 400 tests.

---

## Summary Table

| Item | Description | Mathematical Proof | Computational Verification | Status |
|------|------------|-------------------|---------------------------|--------|
| 1 | Autoconvolution formula: full vs symmetry-optimized | Diagonal/off-diagonal partition | d=2..8, 20 trials each | **PROVED + VERIFIED** |
| 2 | d=4 unrolled coefficients match $(P(x))^2$ | Direct pair enumeration | 50 random + polynomial check + known values | **PROVED + VERIFIED** |
| 3 | d=6 unrolled coefficients match $(P(x))^2$ | Direct pair enumeration for all 11 terms | 50 random + polynomial check | **PROVED + VERIFIED** |
| 4 | Generic double-loop $\equiv$ symmetry-optimized | Corollary of Item 1 | d=2..12 + unrolled vs generic | **PROVED + VERIFIED** |
| 5 | MATLAB $\leftrightarrow$ Python normalization equivalence | Coordinate transform: $f_i = a_i/(4n)$, $\times(4n/\ell)$ = $\div(4n\ell)$ | Simulated MATLAB on 60 trials | **PROVED + VERIFIED** |
| 6 | Window-max: prefix-sum correctness and off-by-one | Index arithmetic + standard technique | d=2..8, 30 trials each | **PROVED + VERIFIED** |
| 7 | Early-stop returns lower bounds; single returns true max | Monotonicity of max; exhaustive loop in single | 200 trials × 5 thresholds per d | **PROVED + VERIFIED** |
| 8 | Batch $\equiv$ single when `prune_target=0` | Same $a_i$, equivalent conv, identical window loop | d=4,6,8 at m=5..100 | **PROVED + VERIFIED** |
| 9 | `lo_bin`/`hi_bin` $\equiv$ MATLAB `binsContribute` | Feasibility condition on pairs; 0/1-index conversion | Exhaustive d=2..8 + W\_int prefix-sum | **PROVED + VERIFIED** |

**Total: 71 checks passed, 0 failed.**

**Conclusion: The autoconvolution, test-value computation, and window scan are mathematically correct, match the MATLAB baseline, and are free of off-by-one errors. The unrolled specializations for d=4 and d=6 are verified against both the generic loop and polynomial squaring. The normalization between MATLAB and Python coordinate systems is algebraically proven equivalent.**
