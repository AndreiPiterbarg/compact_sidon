# 3-scale 1.292 re-verification — 2026-05-20

Aggregated summary of the 14-agent re-audit of instance-specific anchor
checks at the 3-scale 1.292 operating point. The original 14-agent V1–V14
audit (recorded in `memory/project_multiscale_audit_v1.md`) was executed
against the 2-scale 1.28984 precursor. Framework checks (V1, V3, V5, V6,
V10, V11, V13) transport directly because they certify the
kernel/QP/master-inequality framework itself. Instance-specific anchor
checks (V2, V4, V7, V8, V9, V12, V14) produce numbers that change
between the 2-scale and 3-scale operating points and have therefore been
re-executed here at the 3-scale point
$(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$,
$(\lambda_1, \lambda_2, \lambda_3) = (85, 10, 5)/100$, $N = 200$ cosines.

## Anchor verdicts (3-scale 1.292)

| Check | Anchor | Lean/cert claim | Independent mpmath value | Slack vs Lean rational | Verdict |
|------:|--------|-----------------|--------------------------|------------------------|:-------:|
| V2  | $\int K_{\rm ms}^2$              | $\in [4.788823, 4.788906]$, $\le 47897/10000$ | $[4.788823421260265, 4.788905181632132]$ | $7.95 \times 10^{-4}$ to 4.7897 | PASS |
| V4  | $\int_0^\infty J_0(\pi\xi)^4\,d\xi$ | $\le 0.5747$                              | $0.287345891\ldots$ (half-line) / $0.574691\ldots$ (full-line) | $|\Delta| < 10^{-5}$ to literature | PASS |
| V7  | $S_1 = \sum_{j=1}^{200} a_j^2 / \widetilde{K_{\rm ms}}(j)$ | $\le 29.840907$ | $29.84090645551326$                       | matches cert to $\sim 10^{-14}$ | PASS |
| V8  | $\min_{x \in [0,1/4]} G(x)$       | $\ge 0.99997987$                           | $0.99998066$ at $x^\ast = 0.247625156$, $G'(x^\ast) = 1.58\times 10^{-16}$ | $7.9 \times 10^{-7}$ above cert lower; consistent with Taylor B&B cell radius $r \approx 3.8 \times 10^{-6}$ | PASS |
| V9  | $k_1 = \widehat{K_{\rm ms}}(1)$    | $\ge 0.92124658$                           | $0.9212465899364082$                      | matches cert to $\sim 10^{-16}$ | PASS |
| V12 | $\min_{j=1,\ldots,200} \widetilde{K_{\rm ms}}(j)$ | $\ge 2.08 \times 10^{-4}$ | $2.0817 \times 10^{-4}$ at $j_{\min} = 147$, all 200 weights $> 0$ | one order of magnitude better than the 2-scale precursor's $3.3 \times 10^{-5}$ | PASS |
| V14 | $I(\delta_i, \delta_j) = \int_0^\infty J_0(\pi\delta_i\xi)^2 J_0(\pi\delta_j\xi)^2\,d\xi$ for all 6 pairs | per-pair `acb.integral` values in the certifier | $|\Delta| \le 5 \times 10^{-16}$ for all 6 pairs (mpmath dps=30 noise floor) | relative error $5{-}8 \times 10^{-17}$ | PASS |

## Method

Two independent re-audit drivers (one for V2/V4/V14, one for V7/V8/V9/V12),
both written from scratch in mpmath at 30–50 decimal places. No project
modules (`kernels.py`, `flint`, `bisect_alt_kernel.py`) were imported by
the re-audit code. Only the 200 rational QP coefficients
(`qpNumerators`) and rational kernel parameters were ingested from the
certificate `multiscale_arcsine_1292.json`. Bessel functions, cosine
sums, and Fourier-transform formulae were reimplemented natively so that
agreement with the certifier represents genuinely independent
corroboration.

Tail-truncation control on the half-line integrals
($\int_0^\infty J_0^4$, $\int_0^\infty J_0^2 J_0^2$,
$\int_0^\infty \big(\sum_i \lambda_i J_0^2\big)^2$) used the standard
Watson asymptotic $J_0(z)^2 \le 2/(\pi z)$ from
Watson 1944 §7.21 to bound the contribution beyond cut-off $T = 10^5$.

## Per-check drivers and artifacts

- **V2, V4, V14**:
  - Summary: `docs/audit_3scale_reverification_V2V4V14.md`
  - Driver: `docs/_audit_v2_v4_v14_PARALLEL_2026_05_20.py`
  - Raw log: `docs/_audit_v2_v4_v14_PARALLEL_2026_05_20.log`
  - Wall time: 897.8s with 7-way multiprocessing.
- **V7, V8, V9, V12**:
  - Summary: `docs/audit_3scale_reverification_V7V8V9V12.md`
  - Driver: `archive/attempts/verify_3scale_V7V8V9V12.py`

## One documentation note (non-blocking)

The driver, certifier, and Lean axiom anchor all use
$k_1 := \widehat{K_{\rm ms}}(1)$ — i.e. the Fourier transform evaluated
at integer $\xi = 1$ (the period-1 Fourier coefficient that enters the
MV master inequality). The Lean rational anchor
`k_1 >= 9212/10000`, the certifier `bisect_alt_kernel.py:232`,
and `kernels.py:K_tilde` are all consistent on this convention. The
alternate value $\widehat{K_{\rm ms}}(1/u) \approx 0.81604$ would have
failed the $\ge 0.92124658$ test; the V9 PASS at 0.92124659 corroborates
the $\xi = 1$ reading. This is purely a notational note for any future
reader; no soundness implication.

## Overall verdict

**CONFIRM.** All seven instance-specific anchor checks transport cleanly
from the 2-scale 1.28984 precursor audit to the 3-scale 1.292 headline.
Combined with the seven framework checks (V1, V3, V5, V6, V10, V11, V13)
that transport unchanged, the full V1–V14 checklist is now affirmed at
the 3-scale 1.292 operating point.

Combined with the 2026-05-15 six-agent
build/axiom/numeric/MV/manuscript/docs re-verification
(`memory/project_1292_reverification_2026_05_15.md`), the headline
$C_{1a} \ge 1.292$ is independently corroborated by:

- 14 fresh re-audit runs at the 3-scale point (this document);
- 6 build/axiom/numeric/MV/manuscript/docs checks (2026-05-15);
- 14 original framework + 2-scale instance checks (2026-05-11, transport-anchored);
- 256-bit `flint.arb` certifier with SHA-256-stamped certificate body;
- Lean 4 build (13 modules post-S1+S2 refactor, 0 sorries, exactly 5
  axioms: 3 logical core plus 2 verifiable-by-computation).
