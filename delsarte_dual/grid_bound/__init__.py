"""Grid-evaluation multi-moment dual bound for the Sidon autocorrelation
constant C_{1a}.

Phase 1 target
--------------
Reproduce the Matolcsi-Vinuesa (2010) lower bound 1.2748 through a fully
rigorous pipeline:
  - All transcendentals (Bessel J_0, pi) via python-flint arb midpoint-radius
    interval arithmetic.
  - All algebraic inputs (delta, u, MV's 119 G-coefficients) as exact fmpq.
  - Cell-interval B&B on the admissible moment variable z_1 in [0, sqrt(mu(M))]
    with the Lemma 3.4 bathtub box; rejection only when the arb upper bound of
    Phi over the cell is strictly < 0.
  - Independent fmpq/arb-only verifier that reconsumes the emitted certificate
    and re-checks every quantitative claim.

Phase 1 reuses MV's verified formulas verbatim; no filter-panel generation
(F4/F5/F9) or multi-moment extension yet -- those are Phase 2.
"""
