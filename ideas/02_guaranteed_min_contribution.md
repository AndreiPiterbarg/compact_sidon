# 2. Guaranteed Minimum Contribution from Unfixed Region

## Current State

Subtree pruning compares `partial_ws` (fixed-prefix-only window sum) against
`threshold(ell, W_int_max)`. It completely ignores that the unfixed bins have
GUARANTEED MINIMUM contributions to the window sum. The unfixed region is
treated as if it contributes 0 to the autoconvolution.

## Proposal

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

## Correctness Proof

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

## Impact Analysis

**Concrete example (L4, m=20, d_child=64):** A parent bin with value B=4,
cursor range [1,3] gives min values child[2p] >= 1, child[2p+1] >= 1.
Self-terms contribute at least 1^2 + 1^2 = 2. Cross-term with a single fixed
bin of value 3: at least 2*3*1 + 2*3*1 = 12. For 10 such unfixed positions,
the guaranteed extra contribution is ~140. With thresholds around 150--300,
this can flip a non-prunable subtree into a prunable one.

**Expected impact: 2x--5x increase in subtree prune rate.** Multiplies with
Proposal 1 for a combined 10x--250x.

## Implementation Notes

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
