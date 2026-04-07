# 4. Partial-Overlap Window Checks in Subtree Pruning

## Current State

Subtree pruning only considers windows FULLY CONTAINED within the partial
convolution range `[0, 2*fixed_len - 2]` (line 1406: `n_windows_partial =
partial_conv_len - n_cv + 1`). Windows that extend into the unfixed region
are skipped entirely, even though the fixed prefix contributes to them.

## Proposal

For windows that partially overlap the fixed and unfixed regions, compute:
- **Exact** contribution from fixed x fixed terms (conv indices in
  `[0, 2*fixed_len - 2]` are known)
- **Lower bound** from fixed x unfixed and unfixed x unfixed terms (using
  guaranteed minimums from Proposal 2)
- Compare total lower bound against threshold

This expands the set of usable windows from O(fixed_len^2) positions to
O(d_child^2) positions.

## Correctness Proof

Same argument as Proposal 2. All contributions are non-negative products of
non-negative integers. Replacing unknown child masses with their guaranteed
minimums gives a valid lower bound on the window sum. The threshold uses
W_int_max (maximum possible mass in window) to be conservative.

## Impact Analysis

This is especially important when the fixed prefix is SMALL (high-j subtree
checks from Proposal 1). With a fixed prefix of only 4 bins, the current code
can only check windows of length <= 7 (within partial_conv of length 7). With
partial-overlap, ALL 120+ window lengths become available.

**Expected impact: 2x--3x increase in subtree prune rate**, especially for
high-j checks that enable the largest subtree skips.
