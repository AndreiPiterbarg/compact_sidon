# Honest Assessment: Cross-Term Lemma + Fourier Truncation

## What is mathematically true

The cross-term vanishing lemma (Lemma 1, `formula_b_coarse_grid_proof.md`) is
**numerically verified** in `lasserre/fourier_xterm.py` to machine precision:

    (f_step * eps_2)(x_k) = 0  exactly at every convolution knot x_k,

whenever int_{I_i} eps_2 = 0 on every bin.

Theorem 1 then gives the EXACT decomposition

    (f*f)(x_k) = (f_step * f_step)(x_k) + (eps_2 * eps_2)(x_k)
              = 4n * MC[k-1] + (eps_2 * eps_2)(x_k)

at every knot.

## What this gives us as a lower bound

For ANY admissible f:

    ||f*f||_inf  >=  (f*f)(x_k)
                 =   4n * MC[k-1] + (eps_2 * eps_2)(x_k)
                 >=  4n * MC[k-1] - ||eps_2||_2^2     (Cauchy-Schwarz)

Take max over k:

    ||f*f||_inf  >=  4n * max_k MC[k-1]  -  ||eps_2||_2^2.   (1)

Since this holds for any admissible f:

    C_{1a}  >=  inf_f [ 4n * max_k MC[k-1] - ||eps_2||_2^2 ].

## Why (1) does NOT give a new lower bound on C_{1a}

The infimum on the RHS of (1) is taken over admissible f. We split f into
the bin-step part f_step (parameterized by mu) and the residual eps_2.
Both are free variables. The objective:

    g(mu, eps) = 4n * max_k MC[k-1](mu)  -  ||eps||_2^2

is MINIMIZED by maximising ||eps||_2^2 (subject to f >= 0 on each bin).
Since ||eps||_2^2 can be made arbitrarily large for spiky f, the bound
becomes vacuous: the infimum on the RHS of (1) is **-infinity** without
extra constraints on eps.

Restricting to f = f_step (eps = 0) recovers the standard knot bound:

    C_{1a}  >=  inf_mu  4n * max_k MC[k-1]  =  val(d),

which is the existing Cloninger-Steinerberger / Lasserre lower bound. No
improvement.

For ANY restriction on ||eps||_2^2 (say <= R), the bound becomes

    C_{1a}  >=  val(d) - R,

which is WORSE than val(d). So adding the residual hurts, not helps.

## The genuine direction (NOT implemented here)

The cross-term lemma IS useful for two distinct purposes, neither of
which is "Fourier-truncated SDP for new lower bounds":

(A) **Soundness of the cascade Formula B**: Theorem 4 of
    `formula_b_coarse_grid_proof.md` uses the lemma to prove that the
    coarse cascade's pruning condition is sound for STEP FUNCTIONS,
    bridging the cell certification to the discrete Lasserre framework.

(B) **Faster exact evaluation**: at any knot x_k, the lemma replaces
    a numeric convolution with the closed-form 4n * MC[k-1] (no
    integration error). Useful as a primitive in cell certification
    code (see `cloninger-steinerberger/cpu/qp_bound.py` for the
    Cauchy-Schwarz-based variant we already use).

## What `lasserre/fourier_xterm.py` actually delivers

It is a **rigorous primitives library**:

1. `step_autoconv_at_knot(mu, k)` — exact closed-form value.
2. `step_autoconv_continuous(mu, t)` — exact piecewise-linear interpolation.
3. `verify_cross_term_at_knot(mu, eps, n_quad)` — numerical certification
   of Lemma 1 to quadrature precision.
4. `lemma1_lower_bound_at_knots(mu, eps)` — sound but provably loose
   Cauchy-Schwarz bound (Section 1 of this doc).

These are correct and useful as primitives. They do not, by themselves,
beat val(d) as a lower bound on C_{1a}. The Lasserre hierarchy
(`lasserre/solvers.py`) and the cascade prover
(`cloninger-steinerberger/cpu/run_cascade_coarse_v2.py`) remain the
operative routes for tightening the lower bound.

## The path forward (per agent audit, `proof/tightest_valid_pruning_bound.md`)

The audit identified only TWO lines of attack with realistic potential:

1. **Lasserre L3 at d=16** with the existing infrastructure
   (`tests/lasserre_highd.py --d 16 --order 3`). Expected lb in
   [1.29, 1.31] — would clear 1.2802 if the discretization gap
   delta_16 = |C_{1a} - val(16)| is small enough.

2. **Bound the discretization gap analytically**. This is the
   load-bearing step. Without delta_d in hand, even val(16) = 1.319
   only certifies C_{1a} >= 1.319 - delta_16, and delta_16 is not
   yet bounded in the repo.

The Fourier-truncated cross-term approach does NOT contribute to either
of these. Documented here for the record so future explorers do not
revisit it.
