# Interval Branch-and-Bound for val(d)

A rigorous lower-bound method for

    val(d) := min_{mu in Delta_d} max_{W in W_d}  mu^T M_W mu

with no discretisation correction: unlike the Cloninger-Steinerberger
cascade (which works on an integer grid and pays a `-2/m - 1/m^2`
penalty), this method partitions the continuous simplex into dyadic
boxes and bounds each with interval arithmetic.

## Choosing dimension `d` for a given target `c`

The method certifies `val(d) >= c`, which together with `val(d) <= C_{1a}`
gives `C_{1a} >= c`. For a target `c` the choice of `d` is constrained by:

1. **Feasibility:** must have `val(d) > c`. Literature estimates (roughly
   `val(d) ~ C_{1a} - const/d`):
   - `val(10) ≈ 1.24137`
   - `val(12) ≈ 1.27072`
   - `val(14) ≈ 1.28396`
   - `val(16) ≈ 1.29-1.30`   (uncertain)
   - `val(18) ≈ 1.305`        (tight)
   - `val(20) ≈ 1.315`
   So for `c = 1.30` the smallest `d` that might work is `d=16` or `d=18`.

2. **Slack matters exponentially:** our bounds need `val(d) - c` to be
   >= a few thousandths. Observed behaviour:
   - Slack `0.02` (e.g. d=10 target=1.22): easy, finishes in minutes.
   - Slack `0.01` (e.g. d=14 target=1.27): hard, tens of minutes on 30 cores.
   - Slack `0.001` (e.g. d=10 target=1.24): INFEASIBLE at default
     `min_box_width`, even at ~10^8 nodes.

3. **Tree size grows ~10x per +2 in `d`** empirically.

**Practical recipe for target `c`:** pick the smallest `d` with `val(d) > c + 0.01`.
If the budget allows, use `d + 2` for a safer margin. For `c = 1.30`, probably
`d = 18` is the sweet spot (slack ~0.005 and tree size ~100x d=14).

## Approach

1. **Symmetry reduction (single-pair orbit cover).**
   We search only `H_d := { mu in Delta_d : mu_0 <= mu_{d-1} }`. Every
   orbit of the time-reversal involution `sigma(i) = d-1-i` has a
   representative in `H_d`: if `mu_0 <= mu_{d-1}`, then `mu` itself is
   in `H_d`; otherwise `sigma(mu)` is in `H_d` (and achieves the same
   objective by sigma-invariance). Hence any lower bound that holds on
   `H_d` is a lower bound on `val(d)`.

   In the initial box we enforce the weaker but sufficient constraint
   `mu_0 <= 1/2`, which follows from `mu_0 <= mu_{d-1}` combined with
   the simplex inequality `mu_0 + mu_{d-1} <= 1`. The algorithm
   therefore certifies a superset of `H_d`, so any lower bound it
   produces is still valid for `val(d)`.

   We deliberately do NOT use the stronger all-pairs cut
   `{ mu_i <= mu_{d-1-i} for all i < d/2 }`: that would require a
   symmetric-minimiser theorem for the discrete `val(d)` problem that
   is not self-contained. The single-pair cover is sufficient for
   orbit coverage and is rigorous without convexity assumptions.

2. **Best-first branch and bound.**
   The queue holds boxes prioritised by their current lower bound (the
   least-certified box is processed first). A box is certified if some
   window W has `min_{mu in B cap Delta_d} mu^T M_W mu >= target_c`.
   Otherwise it is split at the widest axis.

3. **Three bounds, combined (max).**
   * *Natural interval*: `sum_{(i,j) in supp(W)} lo_i lo_j * scale_W`.
   * *Autoconvolution complement*: uses `sum_s c_s(mu) = 1` on the
     simplex; yields `scale_W * (1 - sum_{not supp} hi_i hi_j)`.
   * *McCormick greedy LP*: SW linear envelope plus a sort-based LP
     over `simplex cap box` (closed-form, O(d log d), no scipy).

4. **Exact-arithmetic rigor replay.**
   Every leaf accepted by the float64 fast path is re-verified with
   the same bound re-computed in `fractions.Fraction`. The McCormick
   Fraction LP is a greedy sort, matching the float path exactly.
   No rigor retries in normal operation.

## Status

| d  | target  | success | nodes    | depth | time   |
| -- | ------- | ------- | -------- | ----- | ------ |
| 4  | 1.10    | yes     | 599      | 30    | 0.2 s  |
| 4  | 1.10233 | yes     | 1,841    | ~40   | 0.6 s  |
| 4  | 1.11    | no      | 1,762    | 66    | 0.8 s  |
| 10 | 1.24    | *in progress* |

## Files

* `bnb.py` — best-first driver.
* `bound_eval.py` — natural / autoconv / McCormick bounds (float +
  Fraction).
* `box.py` — dyadic-rational Box with exact simplex intersection.
* `symmetry.py` — half-simplex cuts (Option C).
* `windows.py` — wrapper over `lasserre.core.build_window_matrices`.
* `rigorous_check.py` — leaf-level Fraction verification.
* `run_d4.py` / `run_d10.py` / `run_d14.py` — top-level drivers.
* `proof.md` — emitted only if `run_d14.py` certifies.

## Running

```
python -m interval_bnb.run_d10 --target 1.24 --time_budget_s 3600
python -m interval_bnb.run_d14 --target 1.2802
```

## Rigor notes

* Box endpoints are stored as `fractions.Fraction`; midpoint splits
  are exact.
* Every float64 bound that certifies a box is recomputed in Fraction
  arithmetic with the same formula. A box is only closed if the
  Fraction value is `>= target_q`.
* The McCormick LP used in rigor is the *no-symmetry-cuts* LP, which
  is weaker than the fast-path LP would be WITH sym cuts -- but we
  deliberately omit sym cuts from both paths so they match exactly.
  Symmetry reduction is applied at the outer level (Option C initial
  box), not inside the LP.
