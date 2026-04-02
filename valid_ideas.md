# Algorithmic Improvements for Cascade Pruning

> **Problem:** Expansion factors of 10,000x--100,000x per cascade level make proving
> $C_{1a} > 1.3$ infeasible. L4 alone has ~7.4 trillion children to test (~3 days CPU).
> The cascade DOES converge (76K survivors at L4), but we can't afford to generate and
> test the 99.999999% that get pruned.
>
> **Goal:** Reduce the number of children generated/tested by 100x--1000x through
> provably correct algorithmic changes.

---

## 1. Multi-Level Hierarchical Subtree Pruning

### Current State

Subtree pruning fires ONLY when Gray code digit `J_MIN=7` advances
(`run_cascade.py:1370`). It checks whether the fixed left-prefix's partial
autoconvolution already exceeds the threshold for all possible inner
configurations. When it fires, it skips `product(range[0..6])` children.
**But it only fires at ONE level.**

### Proposal

Check at EVERY Gray code level `j >= J_MIN_LOWEST` (e.g., j >= 2), not just
j=7. When digit `j` advances, the inner digits `0..j-1` are about to sweep
through `product(range[0..j-1])` children. If the partial-conv check passes,
ALL of them are skipped.

### Correctness Proof

The check is identical to the existing subtree prune. It computes the partial
autoconvolution of the fixed prefix (digits `j..n_active-1`), uses `W_int_max`
for the threshold to account for the worst-case unfixed contribution, and
prunes only if the fixed-prefix window sum EXCEEDS the adjusted threshold.

Since `W_int_max >= W_int` for any child in the subtree, the prune is sound.
Adding more check points doesn't change the correctness of any individual
check -- it just checks more often.

### Impact Analysis

At L4 (d_child=64, ~32 active positions), current subtree pruning at j=7
catches 20--50% of children. Many subtrees at j=3, 4, 5, 6, 8, 9, ... are
ALSO prunable. Each level-j check costs O(fixed_len^2) to compute the partial
conv, but skips O(product(range[0..j-1])) children.

For high j (say j=20), the fixed prefix is small (cheap check) but the subtree
is enormous. For low j (say j=3), the check is expensive but the subtree is
small. The optimal strategy is to check at ALL levels and let the cost/benefit
sort itself out.

**Expected impact: 5x--50x reduction in children tested.**

### Implementation Notes

Replace the single `if j == J_MIN` block at line 1370 with:

```python
if j >= J_MIN_LOWEST and n_active > j:
    # Same subtree prune logic, using active_pos[j-1] as fixed boundary
    ...
```

Use an adaptive J_MIN_LOWEST: set it to the smallest j where the subtree size
`product(range[0..j-1])` exceeds a cost threshold (e.g., > 100 children to
justify the O(fixed_len^2) check).

---

## 2. Guaranteed Minimum Contribution from Unfixed Region

### Current State

Subtree pruning compares `partial_ws` (fixed-prefix-only window sum) against
`threshold(ell, W_int_max)`. It completely ignores that the unfixed bins have
GUARANTEED MINIMUM contributions to the window sum. The unfixed region is
treated as if it contributes 0 to the autoconvolution.

### Proposal

For each unfixed position `p` with cursor range `[lo_p, hi_p]` and parent
value `B_p`, the child bins have guaranteed minimums:

```
min_val[2p]   = lo_p
min_val[2p+1] = B_p - hi_p
```

These contribute guaranteed terms to the autoconvolution:

**Self-terms (unfixed x unfixed, same position):**
```
conv[4p]   += lo_p^2                          (at minimum)
conv[4p+2] += (B_p - hi_p)^2                  (at minimum)
conv[4p+1] += 2 * lo_p * (B_p - hi_p)         (at minimum)
```

**Cross-terms (fixed x unfixed):**
For each fixed bin q with known value child[q]:
```
conv[q + 2p]   += 2 * child[q] * lo_p         (at minimum)
conv[q + 2p+1] += 2 * child[q] * (B_p - hi_p) (at minimum)
```

Add these guaranteed contributions to `partial_ws` before comparing against
the threshold.

### Correctness Proof

All child masses are non-negative integers. Self-terms `child[i]^2` are
monotonically increasing in `child[i]`. Since `child[2p] >= lo_p` by
construction, `child[2p]^2 >= lo_p^2`. Cross-terms `2 * child[q] * child[i]`
are monotone in both arguments (both non-negative), so replacing `child[i]`
with its minimum gives a valid lower bound. The window sum is a sum of
non-negative terms, so adding guaranteed minimums to the partial window sum
preserves soundness:

```
actual_window_sum >= partial_ws_fixed + unfixed_self_min + cross_min
                  >= partial_ws_fixed   (current check)
```

The enhanced check is strictly tighter than the current one.

### Impact Analysis

**Concrete example (L4, m=20, d_child=64):** A parent bin with value B=4,
cursor range [1,3] gives min values child[2p] >= 1, child[2p+1] >= 1.
Self-terms contribute at least 1^2 + 1^2 = 2. Cross-term with a single fixed
bin of value 3: at least 2*3*1 + 2*3*1 = 12. For 10 such unfixed positions,
the guaranteed extra contribution is ~140. With thresholds around 150--300,
this can flip a non-prunable subtree into a prunable one.

**Expected impact: 2x--5x increase in subtree prune rate.** Multiplies with
Proposal 1 for a combined 10x--250x.

### Implementation Notes

After computing `partial_conv` for the fixed prefix, add a second pass:

```python
# Add guaranteed minimum contributions from unfixed bins
for p in unfixed_positions:
    min_lo = lo_arr[p]
    min_hi = parent_int[p] - hi_arr[p]
    # Self-terms
    partial_conv[4*p] += min_lo * min_lo
    partial_conv[4*p + 2] += min_hi * min_hi
    partial_conv[4*p + 1] += 2 * min_lo * min_hi
    # Cross-terms with fixed bins
    for q in range(fixed_len):
        if child[q] != 0:
            partial_conv[q + 2*p] += 2 * child[q] * min_lo
            partial_conv[q + 2*p + 1] += 2 * child[q] * min_hi
```

Cost: O(fixed_len * n_unfixed) per check, negligible vs. the O(fixed_len^2)
partial conv computation.

---

## 3. Arc Consistency / Constraint Propagation on Cursor Ranges

### Current State

Cursor ranges are computed with a single global Cauchy-Schwarz bound:
`x_cap = floor(m * sqrt(c_target / d_child))` (`run_cascade.py:1549`).
For m=20, d_child=32: x_cap=4. This same cap applies to ALL positions
regardless of the parent's specific mass distribution or interactions between
positions.

### Proposal

Before Cartesian-product enumeration, run constraint propagation to tighten
per-position cursor ranges:

```
For each position p, for each value v in [lo_p, hi_p]:
    Set child[2p] = v, child[2p+1] = B_p - v

    For each window (ell, s_lo) that bins 2p or 2p+1 contribute to:
        self_contribution = v^2, (B_p-v)^2, 2*v*(B_p-v) at appropriate indices

        min_other = sum over ALL other positions q != p of:
            minimum possible contribution of position q to this window
            (using guaranteed minimums: lo_q for child[2q], B_q - hi_q for child[2q+1])

        total_min = self_contribution_in_window + min_other_in_window

        If total_min > threshold(ell, W_int_min):
            Value v is INFEASIBLE -> remove from range

Repeat until no ranges change (typically 1-2 rounds of convergence)
```

### Correctness Proof

If for a given value `v` of cursor `p`, the window sum exceeds the threshold
even when ALL OTHER positions are at their MINIMUM contribution, then `v` is
infeasible regardless of what other positions do. All contributions are
non-negative (products of non-negative integers), so replacing unknowns with
their minimums gives a valid lower bound on the window sum.

**~~For the threshold, we use `W_int_min`~~ BUG: MUST USE `W_int_max`.**

The original reasoning was: "W_int_min makes the threshold as LOW as possible,
which is the hardest case for the prune. If pruning succeeds even with
W_int_min, it succeeds for all possible W_int values >= W_int_min."

**This reasoning is BACKWARDS and the check is UNSOUND.** Here's why:

The prune condition is `ws > threshold(ell, W_int)`. Both `ws` and `W_int`
depend on the assignment of other positions. Using W_int_min gives the LOWEST
threshold (easiest to exceed). But the ACTUAL threshold for a non-minimum
assignment is HIGHER, and ws may not have increased enough to exceed it.

**Concrete failure mode:** When only ONE bin of position q falls in the window
(bin split), increasing child[2q] by 1 increases W_int by 1 (threshold +2)
but ws increases by only child[2q]^2 → (child[2q]+1)^2 = 2*child[2q]+1.
When child[2q]=0 (at minimum), ws increases by 1 while threshold increases
by 2. The gap erodes by 1 per unit. A margin of N can be closed after N
positions each contribute 1 unit of gap erosion.

**The fix:** Use `W_int_MAX` (maximum possible mass in window). Then:
- `threshold(W_int_max) >= threshold(W_int)` for all actual W_int
- `ws >= min_ws > threshold(W_int_max) >= threshold(W_int)` → SOUND

This is more conservative (fewer values eliminated) but CORRECT. The existing
subtree pruning code in `run_cascade.py:1454` correctly uses W_int_max for
exactly this reason.

Removing infeasible values from a cursor range doesn't eliminate any child
that could have survived -- it only removes children guaranteed to be pruned.

### Impact Analysis

The Cartesian product size is `product(hi_p - lo_p + 1)`. Reducing even a few
positions' ranges from 5 values to 3 values shrinks the product exponentially:
`(3/5)^k` for k tightened positions.

**Example (d_parent=16, m=20, d_child=32):**
- Global x_cap = 4, so a parent bin with value 5 gives range [1, 4] = 4 choices
- After arc consistency, if 2 values are eliminated: range becomes [2, 3] = 2 choices
- For 8 positions tightened from 4 to 2 choices: reduction = (2/4)^8 = 1/256
- For 16 positions tightened from 5 to 3: reduction = (3/5)^16 ~ 1/1500

**Pre-computation cost:** O(d_parent * max_range * d_child * ell_count) per
parent. For d_parent=32, max_range=5, d_child=64, ell_count~120: about 1.2M
operations. This is negligible compared to testing millions of children.

**Expected impact: 10x--1000x reduction in Cartesian product size** for
parents with moderate-to-large bin values.

### Implementation Notes

Add a function `_tighten_ranges()` called from `process_parent_fused()` after
`_compute_bin_ranges()` and before `_fused_generate_and_prune_gray()`:

```python
@njit(cache=True)
def _tighten_ranges(parent_int, lo_arr, hi_arr, m, c_target, d_child, n_half_child):
    """Arc consistency: remove cursor values that can't produce survivors."""
    d_parent = parent_int.shape[0]
    inv_4n = 1.0 / (4.0 * n_half_child)
    c_target_m2 = c_target * m * m
    eps_margin = 1e-9 * m * m
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    conv_len = 2 * d_child - 1
    changed = True

    while changed:
        changed = False
        for p in range(d_parent):
            new_lo = lo_arr[p]
            new_hi = hi_arr[p]
            B_p = parent_int[p]

            for v in range(lo_arr[p], hi_arr[p] + 1):
                v1 = v
                v2 = B_p - v
                infeasible = False

                # Check windows that bins 2p, 2p+1 contribute to
                for ell in range(2, min(2*d_child+1, conv_len+2)):
                    if infeasible:
                        break
                    n_cv = ell - 1

                    for s_lo in range(max(0, 2*p - n_cv + 1),
                                      min(conv_len - n_cv + 1, 2*p + 2 + 1)):
                        # Compute self-contribution of position p to this window
                        self_ws = 0
                        for k in range(s_lo, s_lo + n_cv):
                            # conv[k] contributions from position p
                            if k == 4*p:
                                self_ws += v1 * v1
                            elif k == 4*p + 2:
                                self_ws += v2 * v2
                            elif k == 4*p + 1:
                                self_ws += 2 * v1 * v2

                        # Compute min contribution from other positions
                        min_other = 0
                        for q in range(d_parent):
                            if q == p:
                                continue
                            mq1 = lo_arr[q]            # min child[2q]
                            mq2 = parent_int[q] - hi_arr[q]  # min child[2q+1]
                            for k in range(s_lo, s_lo + n_cv):
                                if k == 4*q:
                                    min_other += mq1 * mq1
                                elif k == 4*q + 2:
                                    min_other += mq2 * mq2
                                elif k == 4*q + 1:
                                    min_other += 2 * mq1 * mq2

                        total_min = self_ws + min_other

                        # Threshold with minimum W_int
                        lo_bin = max(0, s_lo - (d_child - 1))
                        hi_bin = min(d_child - 1, s_lo + ell - 2)
                        # FIX: use W_int_MAX (not W_int_min!)
                        W_int_max = 0
                        for i in range(lo_bin, hi_bin + 1):
                            pp = i // 2
                            if i % 2 == 0:
                                W_int_max += hi_arr[pp]   # max child[2pp]
                            else:
                                W_int_max += parent_int[pp] - lo_arr[pp]  # max child[2pp+1]

                        c_ell = c_target_m2 * ell * inv_4n
                        dyn_x = c_ell + 1.0 + eps_margin + 2.0 * W_int_max
                        dyn_it = int(dyn_x * one_minus_4eps)

                        if total_min > dyn_it:
                            infeasible = True
                            break
                    if infeasible:
                        break

                if infeasible:
                    if v == new_lo:
                        new_lo = v + 1
                    elif v == new_hi:
                        new_hi = v - 1

            if new_lo != lo_arr[p] or new_hi != hi_arr[p]:
                lo_arr[p] = new_lo
                hi_arr[p] = new_hi
                changed = True

    # Recompute total_children
    total = 1
    for i in range(d_parent):
        r = hi_arr[i] - lo_arr[i] + 1
        if r <= 0:
            return 0
        total *= r
    return total
```

**Important:** This only tightens from the edges (lo and hi). Values in the
middle of the range that are infeasible create "holes" that the Cartesian
product structure can't represent. Tightening from edges is safe and easy.
Interior infeasible values would require a different enumeration strategy.

---

## 4. Partial-Overlap Window Checks in Subtree Pruning

### Current State

Subtree pruning only considers windows FULLY CONTAINED within the partial
convolution range `[0, 2*fixed_len - 2]` (line 1406: `n_windows_partial =
partial_conv_len - n_cv + 1`). Windows that extend into the unfixed region
are skipped entirely, even though the fixed prefix contributes to them.

### Proposal

For windows that partially overlap the fixed and unfixed regions, compute:
- **Exact** contribution from fixed x fixed terms (conv indices in
  `[0, 2*fixed_len - 2]` are known)
- **Lower bound** from fixed x unfixed and unfixed x unfixed terms (using
  guaranteed minimums from Proposal 2)
- Compare total lower bound against threshold

This expands the set of usable windows from O(fixed_len^2) positions to
O(d_child^2) positions.

### Correctness Proof

Same argument as Proposal 2. All contributions are non-negative products of
non-negative integers. Replacing unknown child masses with their guaranteed
minimums gives a valid lower bound on the window sum. The threshold uses
W_int_max (maximum possible mass in window) to be conservative.

### Impact Analysis

This is especially important when the fixed prefix is SMALL (high-j subtree
checks from Proposal 1). With a fixed prefix of only 4 bins, the current code
can only check windows of length <= 7 (within partial_conv of length 7). With
partial-overlap, ALL 120+ window lengths become available.

**Expected impact: 2x--3x increase in subtree prune rate**, especially for
high-j checks that enable the largest subtree skips.

---

## Combined Impact Projection

| Improvement | Standalone Factor | Combined Factor |
|---|---|---|
| 1. Multi-level subtree | 5x--50x | -- |
| 2. Min unfixed contribution | 2x--5x | 10x--250x with (1) |
| 3. Arc consistency | 10x--1000x | 100x--250,000x with (1+2) |
| 4. Partial-overlap windows | 2x--3x | Multiplies into total |

The effects are multiplicative because they attack different parts:
- **(3)** shrinks the Cartesian product before enumeration starts
- **(1)** skips large subtrees during enumeration
- **(2)** and **(4)** make each subtree check more powerful

**For L4 (currently ~7.4T children, ~3 days CPU):** A combined 1000x reduction
brings this to ~7.4B children, feasible in minutes on a single GPU.

## Implementation Priority

1. **Start with (3) Arc Consistency** -- highest standalone impact, clean
   pre-processing step, doesn't change core enumeration logic
2. **Then (1) Multi-level subtree** -- straightforward extension of existing
   code at line 1370
3. **Then (2) Min unfixed contribution** -- enhances (1), moderate code change
4. **Finally (4) Partial-overlap** -- most complex, enhances (1+2)
