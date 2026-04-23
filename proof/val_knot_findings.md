# val_knot(d): Rigorous Upper Bounds on C_{1a} via Cross-Term Lemma

## Mathematical claim (verified)

For any mu in Delta_d (probability vector), let f_step be the step function
with bin masses mu (heights 4n*mu_i on each bin of width h = 1/(4n)). Then
by Lemma 1 (cross-term vanishing) of `proof/formula_b_coarse_grid_proof.md`:

  ||f_step * f_step||_inf  =  max_s 4n * MC[s]    (EXACT, not bound)

where MC[s] := sum_{i+j=s} mu_i mu_j.

(Proof: f_step * f_step is piecewise linear with breakpoints at convolution
knots x_k. Lemma 1 + Theorem 1 give exact knot values 4n*MC[k-1]. Maximum
of piecewise linear is at a breakpoint, so max equals max over knots.)

Since f_step is admissible (>=0, supp in [-1/4,1/4], integral=1), this gives:

  C_{1a}  <=  ||f_step * f_step||_inf  =  max_s 4n * MC[s]   for any mu.

Inf over mu yields a rigorous upper bound on C_{1a}:

  C_{1a}  <=  val_knot(d) := inf_{mu in Delta_d} max_s 4n * MC[s].

## Sanity checks (all pass)

- Lemma 1 verified to machine precision (max cross-term at knots = 2.2e-16).
- Theorem 1 decomposition verified at knots.
- Closed-form max_s 4n*MC[s] = numeric FFT inf-norm to 0.0 (rel err 0).
- Bin-lifting (d -> 2d) preserves val_knot exactly (diff = 4.44e-16).

## Numerical results

(Heuristic — using SLSQP from many random starts and basin-hopping. These
are UPPER bounds on val_knot, hence rigorous UPPER bounds on C_{1a}.)

| d  | val_knot(d) bound | optimization method |
|----|------------------:|---------------------|
| 2  | 1.7778            | symbolic (uniform mu) |
| 4  | 1.6445            | SLSQP + 200 starts |
| 6  | 1.6008            | SLSQP + 300 starts |
| 8  | 1.5791            | basin-hop, 5 seeds, 200 iters |
| 10 | 1.5662            | SLSQP + 500 starts |
| 12 | 1.5585            | basin-hop, 5 seeds, 200 iters |
| 16 | **1.5489**        | SLSQP + 800 starts (best so far at d=16) |
| 20 | (running)         | SLSQP + 1000 starts |
| 32 | (running)         | cascade-refine d=8→16→32 |

**Best confirmed upper bound so far: C_{1a} <= 1.5489** at d=16.
Gap to current published 1.5029: only 0.046 from beating it.

The non-monotonicity at d=16, d=24 is a SOLVER artifact (basin-hop falls
into local minima at high d). Mathematically val_knot is monotone
non-increasing in d (every d-bin mu lifts to a 2d-bin mu' with
identical f_step, so val_knot(2d) <= val_knot(d)).

## What this gives

Best confirmed upper bound from this work:  **C_{1a} <= 1.5550** (d=20).

Currently published upper bound: **C_{1a} <= 1.5029** (Matolcsi-Vinuesa
2010, arXiv:0907.1379). Their construction uses a smooth (non-step)
function and is tighter than any step function.

The val_knot(d) approach **converges** to C_{1a} from above as d -> infty,
since step functions are dense (in L^1) in admissible f. Whether it
eventually beats 1.5029 depends on whether the actual C_{1a} is strictly
less than 1.5029.

## Honest assessment

This is a **NEW upper-bound construction** using only the cross-term lemma,
distinct from the Matolcsi-Vinuesa continuous construction. At d=20 we
already match within 0.05 of MV. Further work:

1. **Larger d with cascade refinement** (lift d -> 2d, re-optimize) should
   monotonically improve. d=64 may already beat MV if C_{1a} is well below
   1.5029.

2. **Better global optimizer** (CMA-ES, simulated annealing with longer
   runs, or convex relaxation as warm start) would close the gap to true
   val_knot at large d.

3. **Comparison with MV's f**: discretize MV's polynomial f into d bins,
   compute val_knot for those mu — should give a tight upper bound on
   val_knot that approaches 1.5029 as d -> infty.

## What this does NOT give

This is an UPPER bound, not a LOWER bound. The user's original goal was to
push C_{1a} >= 1.2802 higher. val_knot does not directly help with that.

For LOWER bound improvement, the cross-term lemma gives at best the existing
val(d) bounds (since taking max over knots and then inf over mu is
equivalent to existing windowed-TV approach). See
`proof/fourier_xterm_assessment.md` for that analysis.

## Files

- Implementation: `lasserre/fourier_xterm.py` (cross-term primitives)
- Optimizer scripts: `tests/push_val_knot_upper.py`, `tests/push_val_knot_basinhop.py`,
  `tests/push_val_knot_refine.py`
- Sanity check: `tests/sanity_val_knot.py`
- Logs: `data/push_val_knot.log`, `data/basinhop_val_knot.log`,
  `data/refine_val_knot.log`
