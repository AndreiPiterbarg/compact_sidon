# Algorithmic Improvements for Cascade Pruning

> **Problem:** Expansion factors of 10,000x--100,000x per cascade level make proving
> $C_{1a} > 1.3$ infeasible. L4 alone has ~7.4 trillion children to test (~3 days CPU).
> The cascade DOES converge (76K survivors at L4), but we can't afford to generate and
> test the 99.999999% that get pruned.
>
> **Goal:** Reduce the number of children generated/tested by 100x--1000x through
> provably correct algorithmic changes.

## Ideas

1. [Multi-Level Hierarchical Subtree Pruning](01_multi_level_subtree_pruning.md)
2. [Guaranteed Minimum Contribution from Unfixed Region](02_guaranteed_min_contribution.md)
3. [Arc Consistency / Constraint Propagation on Cursor Ranges](03_arc_consistency.md)
4. [Partial-Overlap Window Checks in Subtree Pruning](04_partial_overlap_windows.md)

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
