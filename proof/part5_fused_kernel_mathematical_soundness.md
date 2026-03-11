# Part 5: Fused Generate+Prune Kernel — Mathematical Soundness

**Scope:** Prove from first principles that every optimization in `_fused_generate_and_prune` (run_cascade.py:499-988) is mathematically sound. Where Part 4 verified computational equivalence with reference implementations, Part 5 proves each optimization cannot introduce false negatives (missed survivors) or false positives (unsound pruning).

**Result: ALL 9 ITEMS PROVED SOUND**

**Files audited:**

- `run_cascade.py:499-988` — `_fused_generate_and_prune` (490 lines)
- `run_cascade.py:991-1020` — `_compute_bin_ranges`
- `run_cascade.py:1033-1078` — `process_parent_fused`
- `run_cascade.py:1081-1173` — `process_parent_verbose`
- `initial_baseline.m:132-251` — MATLAB reference

**MATLAB mapping:**

- Lines 132-153: Child bin creation $\to$ Python odometer cursor iteration
- Lines 155-189: Cartesian product indexing $\to$ Python cursor $[\text{lo}[i], \text{hi}[i]]$ ranges
- Lines 195-233: Autoconvolution + threshold check $\to$ Python `raw_conv` + window scan
- Lines 238-251: Survivor collection $\to$ Python `out_buf` + canonicalization

---

## Item 1: Odometer Completeness

**Claim:** The odometer (lines 720-727) visits every point in the Cartesian product $\prod_{i=0}^{d_p - 1} [\text{lo}[i],\, \text{hi}[i]]$ exactly once.

### Proof

The odometer is a mixed-radix counter with digit $i$ ranging over $[\text{lo}[i], \text{hi}[i]]$:

```python
carry = d_parent - 1
while carry >= 0:
    cursor[carry] += 1
    if cursor[carry] <= hi_arr[carry]: break
    cursor[carry] = lo_arr[carry]
    carry -= 1
```

**Initialization:** $\text{cursor}[i] = \text{lo}[i]$ for all $i$ (line 572-573). This is the lexicographic minimum of the product space.

**Successor generation:** Starting at the rightmost digit ($\text{carry} = d_p - 1$):
1. Increment $\text{cursor}[\text{carry}]$.
2. If within range, stop — we have the lexicographic successor.
3. If overflow, reset $\text{cursor}[\text{carry}] = \text{lo}[\text{carry}]$ and carry left.
4. If carry propagates past position 0 ($\text{carry} < 0$), the product is exhausted.

This is the standard right-to-left lexicographic enumeration. It produces every element of the Cartesian product in order, exactly once, with total iterations $= \prod_i (\text{hi}[i] - \text{lo}[i] + 1)$.

**Edge cases:**
- **Empty range** ($\text{lo}[i] > \text{hi}[i]$): Prevented — `_compute_bin_ranges` returns `None` (line 1014), so the kernel is never called.
- **Single-value range** ($\text{lo}[i] = \text{hi}[i]$): Cursor starts at $\text{lo}[i]$, increment makes it $\text{hi}[i] + 1 > \text{hi}[i]$, so it resets and carries. This is a degenerate digit of radix 1. Correct.
- **All cursors at max:** Every position overflows and resets, carry reaches $-1$, breaking the outer loop. Correct.

**Flow.** The first child is built and its `raw_conv` computed before entering the main loop. Inside: process current child $\to$ odometer advance $\to$ build new child $\to$ loop. Every product element is visited exactly once. $\square$

---

## Item 2: Hoisted Asymmetry Check

**Claim:** $\sum_{i=0}^{n_{\text{half,child}} - 1} \text{child}[i] = \sum_{k=0}^{d_p / 2 - 1} \text{parent}[k]$, constant across all children. If this value triggers the asymmetry test, ALL children of this parent are pruned.

### Proof

The child is constructed (line 590-592) as:

$$\text{child}[2k] = \text{cursor}[k], \qquad \text{child}[2k + 1] = \text{parent}[k] - \text{cursor}[k]$$

Therefore $\text{child}[2k] + \text{child}[2k + 1] = \text{parent}[k]$ for all $k$.

With $n_{\text{half,child}} = d_p$ (the cascade doubles $n_\text{half}$ at each level, so $n_{\text{half,child}} = 2 n_{\text{half,parent}} = d_p$):

$$\sum_{i=0}^{d_p - 1} \text{child}[i] = \sum_{k=0}^{d_p / 2 - 1} \bigl(\text{child}[2k] + \text{child}[2k + 1]\bigr) = \sum_{k=0}^{d_p / 2 - 1} \text{parent}[k]$$

This holds because $d_p = 2 n_{\text{half,parent}}$ is always even, so the sum telescopes into exactly $d_p / 2$ complete parent-bin pairs. The cursor values cancel completely.

**Soundness of hoisting:** If $\text{left\_frac} = \text{left\_sum} / m \ge \sqrt{c_\text{target} / 2}$ (or $\le 1 - \sqrt{c_\text{target} / 2}$), the asymmetry argument gives $\|f * f\|_\infty \ge 2 \cdot \text{left\_frac}^2 \ge c_\text{target}$ for every child sharing this left\_frac. Since left\_frac is constant across all children, the hoisted check correctly prunes all of them.

The code (lines 546-551):

```python
left_sum_parent = sum(parent_int[0 : d_parent // 2])
left_frac = left_sum_parent / m
if left_frac >= threshold_asym or left_frac <= 1.0 - threshold_asym:
    return 0, 0
```

matches the identity exactly. $\square$

---

## Item 3: Initial Autoconvolution

**Claim:** Lines 632-639 correctly compute $\text{raw\_conv}[k] = \sum_{i+j=k,\, 0 \le i,j < d} c_i \cdot c_j$.

### Proof

The discrete autoconvolution decomposes as:

$$\text{raw\_conv}[k] = \underbrace{\sum_{\substack{2i = k \\ 0 \le i < d}} c_i^2}_{\text{diagonal}} + \underbrace{2 \sum_{\substack{i + j = k \\ 0 \le i < j < d}} c_i c_j}_{\text{off-diagonal}}$$

The code:

```python
for i in range(d_child):
    ci = child[i]
    raw_conv[2 * i] += ci * ci                         # diagonal
    for j in range(i + 1, d_child):
        raw_conv[i + j] += 2 * ci * child[j]           # off-diagonal
```

- First term: adds $c_i^2$ to $\text{raw\_conv}[2i]$ for each $i$, covering all diagonal entries.
- Second term: for each ordered pair $i < j$, adds $2 c_i c_j$ to $\text{raw\_conv}[i + j]$. The factor 2 accounts for both orderings $(i, j)$ and $(j, i)$.

Together this equals the full autoconvolution. This matches the MATLAB computation `functionMult = matrix_tmp(:,pairs(:,1)) .* matrix_tmp(:,pairs(:,2))` (line 195-196), which computes all ordered products then sums by convolution index. $\square$

---

## Item 4: Fast Path — 1 Position Changed

**Claim:** When `n_changed == 1` (lines 735-771), the incremental `raw_conv` update accounts for all affected terms.

### Proof

When `carry == d_parent - 1`, only the last parent position changed. The affected child bins are:

$$k_1 = 2(d_p - 1) = d_\text{child} - 2, \qquad k_2 = k_1 + 1 = d_\text{child} - 1$$

Let $\delta_1 = \text{new}_1 - \text{old}_1$, $\delta_2 = \text{new}_2 - \text{old}_2$. The change to $\text{raw\_conv}[s]$ involves only terms where at least one index is $k_1$ or $k_2$:

| Entry | Formula | Code (line) |
|-------|---------|-------------|
| $\text{raw\_conv}[2k_1]$ | $\text{new}_1^2 - \text{old}_1^2$ | 750 |
| $\text{raw\_conv}[2k_2]$ | $\text{new}_2^2 - \text{old}_2^2$ | 751 |
| $\text{raw\_conv}[k_1 + k_2]$ | $2(\text{new}_1 \text{new}_2 - \text{old}_1 \text{old}_2)$ | 753 |
| $\text{raw\_conv}[k_1 + j]$ for $j < k_1$ | $2 \delta_1 c_j$ | 757 |
| $\text{raw\_conv}[k_2 + j]$ for $j < k_1$ | $2 \delta_2 c_j$ | 758 |

**Completeness check:** Are there unchanged bins $j > k_2$? No — $k_2 = d_\text{child} - 1$ is the last bin. Are there bins between $k_1$ and $k_2$ (exclusive)? No — $k_2 = k_1 + 1$. The loop `for j in range(k1)` covers all unchanged bins $\{0, \ldots, k_1 - 1\}$. All affected terms are accounted for. $\square$

---

## Item 5: Short Carry Path — 2..threshold Positions Changed

**Claim:** When $2 \le n_\text{changed} \le \lfloor d_p / 4 \rfloor$ (lines 773-839), the incremental update covers all terms involving at least one changed bin.

### Proof

**Changed region.** When carry propagates from $d_p - 1$ to position $\text{carry}$, positions $\text{carry}, \text{carry} + 1, \ldots, d_p - 1$ all changed (the odometer resets $\text{carry} + 1$ through $d_p - 1$ to $\text{lo}[\cdot]$, and increments position $\text{carry}$). The corresponding child bins span indices $[2 \cdot \text{carry},\, d_\text{child} - 1]$.

**Unchanged region.** Child bins $[0,\, 2 \cdot \text{carry} - 1]$. There are no unchanged bins after the changed region, since the changed region extends to $d_\text{child} - 1$.

The update decomposes into four disjoint groups covering all terms $\Delta\text{raw\_conv}[s] = \sum_{i+j=s} (\text{new}[i]\,\text{new}[j] - \text{old}[i]\,\text{old}[j])$ where at least one of $i, j$ is changed:

**(a) Self + mutual within each changed parent position** (lines 787-796). For each changed position $p$, updates $\text{raw\_conv}[2k_1]$, $\text{raw\_conv}[2k_2]$, and $\text{raw\_conv}[k_1 + k_2]$ where $k_1 = 2p$, $k_2 = 2p + 1$.

**(b) Cross-terms between different changed positions** (lines 799-816). For each pair of changed positions $p_a < p_b$, updates all four inter-position pairs: $(a_1, b_1)$, $(a_1, b_2)$, $(a_2, b_1)$, $(a_2, b_2)$. Since $a_1 < a_2 < b_1 < b_2$, all resulting convolution indices are distinct from (a).

**(c) Cross-terms between changed and unchanged bins** (lines 818-827). For each changed bin $k$ and unchanged bin $j < 2 \cdot \text{carry}$: $\text{raw\_conv}[k + j] \mathrel{+}= 2 \delta_k c_j$.

**(d) Terms involving only unchanged bins.** Delta is zero. No update needed.

Groups (a)-(d) are disjoint and exhaustive. In (c), `child[j]` for $j < 2 \cdot \text{carry}$ is unchanged from `prev_child[j]`, so the unchanged values are used correctly (child was updated only for positions $\ge \text{carry}$ at lines 782-784). $\square$

---

## Item 6: Subtree Pruning Soundness

**Claim:** When the deep-carry subtree prune fires (lines 841-961), every child in the skipped subtree would also be pruned by the full window scan.

### Proof

Let $F = 2 \cdot \text{carry}$ denote the number of fixed child bins (bins $0, \ldots, F - 1$). The subtree consists of all children sharing these fixed bins but varying in the unfixed bins ($F, \ldots, d_\text{child} - 1$).

The subtree prune computes a partial autoconvolution from fixed bins only (lines 847-854) and checks windows fully contained in the partial convolution range ($s_\text{hi} < 2F - 1$). The pruning condition is $\text{ws\_partial} > \text{dyn\_it}(W_\text{int,max})$.

Three inequalities establish soundness:

**Inequality 1: $\text{ws\_full} \ge \text{ws\_partial}$ for any child in the subtree.**

For any convolution index $k < 2F - 1$:

$$\text{full\_conv}[k] = \underbrace{\sum_{\substack{i+j=k \\ i,j < F}} c_i c_j}_{\text{partial\_conv}[k]} + \underbrace{\sum_{\substack{i+j=k \\ i \ge F \text{ or } j \ge F}} c_i c_j}_{\ge\, 0}$$

Since $c_i \ge 0$ for all $i$, every cross-term involving an unfixed bin is non-negative. Therefore $\text{full\_conv}[k] \ge \text{partial\_conv}[k]$ for all $k$ in the partial range. Summing over any window: $\text{ws\_full} \ge \text{ws\_partial}$. $\square$

**Inequality 2: $W_\text{int,actual} \le W_\text{int,max}$ for any child in the subtree.**

$W_\text{int,max} = W_\text{int,fixed} + W_\text{int,unfixed}$ where:

- $W_\text{int,fixed} = \sum_{i \in [\text{lo\_bin},\, \min(\text{hi\_bin},\, F-1)]} c_i$ — **exact** for the fixed bins.
- $W_\text{int,unfixed} = \sum_{k=p_\text{lo}}^{p_\text{hi}} \text{parent}[k]$ — an **upper bound** on the unfixed bins' contribution.

The upper bound holds because for each unfixed parent position $k$: $\text{child}[2k] + \text{child}[2k + 1] = \text{parent}[k]$ and both $\ge 0$, so any subset of unfixed child bins in the window contributes at most $\text{parent}[k]$ per parent position.

The fixed and unfixed ranges partition $[\text{lo\_bin}, \text{hi\_bin}]$ without gap or overlap:
- Fixed covers $[\text{lo\_bin},\, \min(\text{hi\_bin}, F - 1)]$
- Unfixed covers $[\max(F, \text{lo\_bin}),\, \text{hi\_bin}]$ (only when $\text{hi\_bin} \ge F$)

Together they cover $[\text{lo\_bin}, \text{hi\_bin}]$ completely. $\square$

**Inequality 3: $\text{dyn\_it}(W)$ is non-decreasing in $W$.**

$$\text{dyn\_it}(W) = \bigl\lfloor (\text{dyn\_base\_ell} + \text{two\_ell} \cdot W) \cdot (1 - 4\varepsilon) \bigr\rfloor$$

Since $\text{two\_ell} = 2\ell / (4n) > 0$ and $(1 - 4\varepsilon) > 0$, the argument of $\lfloor \cdot \rfloor$ is strictly increasing in $W$. Hence $\text{dyn\_it}$ is non-decreasing. $\square$

**Chain conclusion.** If $\text{ws\_partial} > \text{dyn\_it}(W_\text{int,max})$, then for any child in the subtree:

$$\text{ws\_full} \ge \text{ws\_partial} > \text{dyn\_it}(W_\text{int,max}) \ge \text{dyn\_it}(W_\text{int,actual})$$

so the full window scan would also prune the child. The entire subtree is safely skippable. $\square$

**Post-prune cursor fast-forward (lines 936-961).** Setting $\text{cursor}[\text{carry}+1 : d_p] = \text{hi}[\text{carry}+1 : d_p]$ ensures the next odometer increment carries through the entire trailing range, advancing $\text{cursor}[\text{carry}]$ (or further left). The child is rebuilt and `raw_conv` fully recomputed from scratch. The `continue` statement processes this fast-forwarded child through the normal scan — it will be pruned by the same killing window that fired the subtree prune (since that window's ws for the full child is $\ge$ the partial ws that already exceeded the relaxed threshold). $\square$

---

## Item 7: Quick-Check Mechanism — No False Pruning

**Claim:** The quick-check (lines 642-653) never prunes a child that the full window scan would not also prune. The tracked $W_\text{int}$ is maintained exactly.

### Proof

**Equivalence to full scan.** The quick-check computes the window sum and threshold for a single specific $(\ell, s_\text{lo})$ pair:

$$\text{ws\_qc} = \sum_{k = s_\text{lo}}^{s_\text{lo} + \ell - 2} \text{raw\_conv}[k]$$
$$\text{dyn\_it\_qc} = \bigl\lfloor (\text{dyn\_base\_ell}[\ell - 2] + \text{two\_ell}[\ell - 2] \cdot W_\text{int,qc}) \cdot (1 - 4\varepsilon) \bigr\rfloor$$

This is identical to what the full scan computes at the same window (lines 678-694). The full scan uses prefix-summed `conv[]` to get the same window sum; the quick-check sums `raw_conv[]` directly — both yield $\sum_{k=s}^{s+\ell-2} \text{raw\_conv}[k]$. If $\text{ws\_qc} > \text{dyn\_it\_qc}$, the full scan would find the same result at the same window. $\square$

**Safety on miss.** If the quick-check does not kill, the full scan runs (line 655: `if not quick_killed`). The full scan checks ALL $(\ell, s_\text{lo})$ windows exhaustively. No windows are skipped. $\square$

**$W_\text{int}$ tracking is exact.**

*Fast path (lines 760-771):* When only bins $k_1, k_2$ change by $\delta_1, \delta_2$ (int32 values), the tracked window's $W_\text{int}$ changes by $\delta_1$ iff $k_1 \in [\text{qc\_lo}, \text{qc\_hi}]$ and by $\delta_2$ iff $k_2 \in [\text{qc\_lo}, \text{qc\_hi}]$. The code:

```python
qc_lo = max(qc_s - (d_child - 1), 0)
qc_hi = min(qc_s + qc_ell - 2, d_child - 1)
if qc_lo <= k1 <= qc_hi: qc_W_int += int64(delta1)
if qc_lo <= k2 <= qc_hi: qc_W_int += int64(delta2)
```

Here `qc_lo` and `qc_hi` are precisely `lo_bin` and `hi_bin` for the tracked window (matching the formulas proved in Part 3, Item 9). The deltas are exact int32 differences added to an int64 accumulator — no floating-point, no rounding, no accumulation error regardless of the number of consecutive 1-changed steps.

*Short carry (lines 830-839):* Full recompute from `child[qc_lo..qc_hi]`. Exact. $\square$

*Deep carry / full recompute (lines 950-961, 976-986):* Full recompute from `child[qc_lo..qc_hi]`. Exact. $\square$

**Direction of hypothetical error.** Even if $W_\text{int,qc}$ were too small (hypothetically), $\text{dyn\_it\_qc}$ would be too small, making $\text{ws} > \text{dyn\_it\_qc}$ easier to satisfy — a false quick-kill. But $W_\text{int,qc}$ being too large would make $\text{dyn\_it\_qc}$ too large, causing a missed quick-kill (performance loss only). Since $W_\text{int,qc}$ is exact, neither case arises. $\square$

---

## Item 8: Survivor Canonicalization

**Claim:** Lines 700-718 store $\min(\text{child}, \text{rev}(\text{child}))$ lexicographically for every survivor.

### Proof

The code compares $\text{rev}(\text{child})$ with $\text{child}$ lexicographically:

```python
use_rev = False
for i in range(d_child):
    j = d_child - 1 - i
    if child[j] < child[i]:      # rev(child)[i] < child[i]
        use_rev = True
        break
    elif child[j] > child[i]:    # rev(child)[i] > child[i]
        break
```

At position $i$: $\text{rev}(\text{child})[i] = \text{child}[d - 1 - i]$.

- If $\text{rev}(\text{child})[i] < \text{child}[i]$: reversal is lexicographically smaller $\Rightarrow$ `use_rev = True`.
- If $\text{rev}(\text{child})[i] > \text{child}[i]$: child is lexicographically smaller $\Rightarrow$ break (keep `use_rev = False`).
- If equal: continue to next position.

For palindromes ($\text{child} = \text{rev}(\text{child})$), all positions tie, the loop exhausts, and `use_rev` remains `False`. The stored value is `child`, which equals `rev(child)`. For non-palindromes, the first differing position determines which is smaller.

The conditional copy (lines 711-718) stores `rev(child)` if `use_rev`, otherwise `child`. This is $\min(\text{child}, \text{rev}(\text{child}))$ lexicographically. $\square$

**Consistency with `_canonicalize_inplace`** (lines 219-243): That function uses an identical comparison but loops only to $d / 2$ — an optimization that is valid because non-palindromes must differ at some position $i < d/2$, and palindromes have all ties. Both produce the same canonical form. $\square$

---

## Item 9: Buffer Overflow Detection and Determinism

**Claim:** The wrapper `process_parent_fused` (lines 1033-1078) correctly detects when the kernel produces more survivors than the initial buffer can hold, re-allocates, and re-runs to capture all survivors.

### Proof

The kernel's survivor counter `n_surv` increments for every surviving child regardless of buffer capacity (line 718), while the buffer write is conditional on `n_surv < max_survivors` (line 711). The returned `n_surv` is therefore the true survivor count, potentially exceeding `max_survivors`.

The wrapper (lines 1064-1076):

```python
n_survivors, _ = _fused_generate_and_prune(...)
if n_survivors > max_buf:
    max_buf = n_survivors
    out_buf = np.empty((max_buf, d_child), dtype=np.int32)
    n2, _ = _fused_generate_and_prune(...)
    assert n2 == n_survivors
```

If overflow occurs, the buffer is re-allocated at the exact required size and the kernel is re-run.

**Determinism guarantee.** The assertion `n2 == n_survivors` verifies that the re-run produces the same count. The kernel is deterministic because:
1. No random number generation.
2. No threading (`@njit` without `parallel=True`; no `prange`).
3. All floating-point operations use the same inputs and execute in the same order on re-run, producing identical results under IEEE 754 semantics. $\square$

---

## Summary

| # | Item | Lines | Verdict |
|---|------|-------|---------|
| 1 | Odometer completeness | 720-727 | **PROVED SOUND** |
| 2 | Hoisted asymmetry check | 546-551 | **PROVED SOUND** |
| 3 | Initial autoconvolution | 632-639 | **PROVED SOUND** |
| 4 | Fast path (1-changed) | 735-771 | **PROVED SOUND** |
| 5 | Short carry path (2..threshold changed) | 773-839 | **PROVED SOUND** |
| 6 | Subtree pruning (deep carry) | 841-961 | **PROVED SOUND** |
| 7 | Quick-check mechanism | 583-587, 643-653, 760-771 | **PROVED SOUND** |
| 8 | Survivor canonicalization | 700-718 | **PROVED SOUND** |
| 9 | Buffer overflow detection + determinism | 1033-1078 | **PROVED SOUND** |

**Conclusion.** Every optimization in the fused generate+prune kernel is mathematically sound. The odometer visits all children (Item 1). The hoisted asymmetry check is justified by the algebraic identity $\sum \text{child}_\text{left} = \sum \text{parent}_\text{left}$ (Item 2). The initial autoconvolution matches the standard formula (Item 3). Incremental updates — fast path (Item 4), short carry (Item 5) — account for all terms involving changed bins, with no bins missed. Subtree pruning (Item 6) rests on three provable inequalities: $\text{ws\_full} \ge \text{ws\_partial}$, $W_\text{int,actual} \le W_\text{int,max}$, and monotonicity of $\text{dyn\_it}$. The quick-check (Item 7) computes an identical threshold test to the full scan with exact $W_\text{int}$ tracking. Canonicalization (Item 8) computes the lexicographic minimum of child and its reversal. Buffer overflow is detected and handled deterministically (Item 9). No optimization can cause a false negative (missed survivor) or unsound pruning (false positive).
