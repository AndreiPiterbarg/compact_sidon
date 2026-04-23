# delsarte_dual/

A Delsarte-type infinite-dimensional dual for the Sidon autocorrelation
constant $C_{1a}$, with a rigorous mpmath interval verification pipeline.

## Quick start

```bash
python -m delsarte_dual.run_all --restarts-f1 20 --restarts-f2 20 --subdiv 2048
```

## Layout

| File | Role |
|---|---|
| `theory.md` | Derivation of the dual bound; admissibility conditions; positive-definiteness certificates per family. |
| `family_f1_selberg.py` | Fejér-modulated test function $g$, closed-form $\widehat g$. |
| `family_f2_gauss_poly.py` | Gaussian × even polynomial. Degree ≤ 2 in $t^2$. |
| `family_f3_vaaler.py` | **Documented stub.** Full Vaaler implementation deferred. |
| `rigorous_max.py` | Interval B&B for $\max g$ on $[-\tfrac12,\tfrac12]$; interval quadrature. |
| `optimise.py` | scipy/Nelder-Mead search over each family (float64). |
| `verify.py` | End-to-end mpmath verification yielding a ball $[L_\text{low},L_\text{high}]$. |
| `run_all.py` | Orchestrator: optimise → verify → report. |
| `postmortem.md` | Honest write-up of what was tried and where it fell short. |
| `run.log` | Log of the last end-to-end run. |

## What the rigorous pipeline certifies

For each family $F$ and best parameters $\theta^\*$ it produces a ball
$$
[\, L_\text{low}(\theta^\*),\ L_\text{high}(\theta^\*)\,]
$$
such that $L_\text{low}(\theta^\*) \le C_{1a}$ is a **proof value**, where
$$
L(\theta) \;=\; \frac{\int_{\mathbb R}\widehat g_\theta(\xi)\,w(\xi)\,d\xi}{\max_{t\in[-1/2,1/2]} g_\theta(t)},\qquad w(\xi)=\max(0,1-\tfrac{\pi}{2}|\xi|)^2.
$$
The weight $w$ comes from the rigorous Paley-Wiener inequality
$|\widehat f(\xi)|\ge 1-\tfrac{\pi}{2}|\xi|$ for $f$ supported on
$[-\tfrac14,\tfrac14]$ with $\int f=1$. See `theory.md` Section 2.

## Correctness gates

- **G1** Reproduce 1.2802 from a MV-style $g$: see `optimise.matolcsi_vinuesa_reference_f1`.
- **G2** Unit tests in `tests/test_delsarte_dual.py`.
- **G3** Each family prints its ball and the parameters.
- **G4** Max-$g$ verified with relative width ≤ `--rel-tol` (default $10^{-8}$).
- **G5** Positive-definiteness certificate documented in `theory.md` per family.

If all five pass **and** the claimed new bound exceeds 1.2802, only then do
we update `CLAUDE.md`.

## Dependencies

Already available in the project environment:
`mpmath>=1.3, sympy>=1.12, scipy>=1.10, numpy>=2.0`.
Optionally: `python-flint` for Arb backend (not required here).

## Honest status

See `postmortem.md`.
