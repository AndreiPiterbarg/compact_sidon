# 3. Arc Consistency / Constraint Propagation on Cursor Ranges

## Current State

Cursor ranges are computed with a single global Cauchy-Schwarz bound:
`x_cap = floor(m * sqrt(c_target / d_child))` (`run_cascade.py:1549`).
For m=20, d_child=32: x_cap=4. This same cap applies to ALL positions
regardless of the parent's specific mass distribution or interactions between
positions.

## Proposal

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

## Correctness Proof

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

## Impact Analysis

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

## Implementation Notes

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
