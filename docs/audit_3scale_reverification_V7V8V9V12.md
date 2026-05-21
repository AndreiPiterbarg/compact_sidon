# 3-scale 1.292 re-verification: V7 / V8 / V9 / V12

**Date:** 2026-05-20
**Auditor:** Fresh independent verifier (instance-specific anchor lane)
**Scope:** Re-verify the four instance-specific numerical anchors of the
3-scale 1.292 certificate (`multiscale_arcsine_1292.json`,
SHA-256 body hash `5fa9ae37...`) at the
3-scale point
$(\delta_1,\delta_2,\delta_3) = (138, 55, 25)/1000$,
$(\lambda_1,\lambda_2,\lambda_3) = (85, 10, 5)/100$,
$u = 319/500 = 0.638$,
$N = 200$ cosines.
**Method:** mpmath at 50-decimal-digit precision, **no** import from
the project's `kernels.py` or `optimize_G.py`.
**Driver script:** [`../archive/attempts/verify_3scale_V7V8V9V12.py`](../archive/attempts/verify_3scale_V7V8V9V12.py).

## Independent formula re-derivation

Following `lower_bound_proof.tex` and `kernels.py:MultiScaleArcsineKernel`:

- Each arcsine component is the *autoconvolution* $\eta_\delta \ast \eta_\delta$
  of the arcsine density $\eta_\delta(x) = (1/\pi\sqrt{\delta^2 - x^2}) \mathbf{1}_{|x|<\delta}$.
  Its Fourier transform is $\widehat{K_{\rm arc}(\delta;\cdot)}(\xi) = J_0(\pi \delta \xi)^2$.
- The multi-scale kernel:
  $\widehat{K_{\rm ms}}(\xi) = \sum_{i=1}^{3} \lambda_i J_0(\pi \delta_i \xi)^2 \ge 0$.
- $G$ is the 200-cosine multiplier $G(x) = \sum_{j=1}^{200} a_j \cos(2\pi j x / u)$
  (no constant term; `optimize_G.py` line 5 and lines 100–102).
- $S_1 = \sum_{j=1}^{200} a_j^2 / w_j$ with $w_j = \widehat{K_{\rm ms}}(j/u)$.
- $k_1 = \widehat{K_{\rm ms}}(1)$ (period-1 Fourier coefficient at $\xi = 1$
  — see `bisect_alt_kernel.py` line 232:
  `k1_float = float(kernel.K_tilde(1, prec_bits=prec_bits).mid())`).
- The 200 QP coefficients $a_j$ are taken **verbatim** as exact rationals
  with common denominator $10^8$ from `multiscale_arcsine_1292.json`,
  field `body.G.coeffs_q`.

## Per-check results

| Anchor | Certificate (`flint.arb`) | This re-verification (mpmath, 50 dps) | $\lvert\Delta\rvert$ | Verdict |
|---|---|---|---|---|
| $S_1$ | $29.840906455513267$ (upper) | $29.84090645551326375$ | $\le 4 \times 10^{-15}$ | **CONFIRM** |
| $\min G$ | $\ge 0.9999798743824747$ (cert lower) | $0.99998066176478175$ (true min at $x^* = 0.247625155\ldots$) | $7.9 \times 10^{-7}$ above the cert lower; consistent with the Taylor B&B cell half-width $r = 1/(8 \cdot 32768) \approx 3.8\times 10^{-6}$ | **CONFIRM** |
| $k_1$ | $0.9212465899364083$ (mid) | $0.92124658993640824$ | $\le 1 \times 10^{-16}$ | **CONFIRM** |
| $\min_j \widetilde K_{\rm ms}(j/u)$ | $\ge 2.08 \times 10^{-4}$ (paper claim) | $2.0817 \times 10^{-4}$ at $j = 147$; all 200 weights strictly positive | sub-claim slack $1.7 \times 10^{-7}$ above the floor | **CONFIRM** |

All four anchors lie strictly inside (or on the correct side of) their
arb enclosures, and the Lean rational bounds remain on the correct
strict side of the arb endpoint:

- Lean anchor $S_1 \le 29841/1000 = 29.841$: cert upper $29.840907$ < $29.841$ (slack $9.3 \times 10^{-5}$). $\checkmark$
- Lean anchor $\min G \ge 998/1000 = 0.998$: cert lower $0.99998$ > $0.998$ (slack $1.9 \times 10^{-3}$). $\checkmark$
- Lean anchor $k_1 \ge 9212/10000 = 0.9212$: cert mid $0.92124659$ > $0.9212$ (slack $4.6 \times 10^{-5}$). $\checkmark$

## Notes and corroborations

### V7 ($S_1$)
mpmath sum agrees with the arb upper $29.840906455513267$ to
$\approx 4\times 10^{-15}$ — well below the
$3.6\times 10^{-14}$ radius reported in
`body.anchors.S_1.repr = "[29.8409064555133 +/- 3.63e-14]"`.

### V8 ($\min G$)
- mpmath `findroot` on $G'(x) = 0$ in
  $[0.247617, 0.247633]$ converged to $x^* = 0.2476251559934671151$
  with $G'(x^*) = 1.58 \times 10^{-16}$ (zero at 50-dps).
- $G(x^*) = 0.99998066176478175$. This is the **true** minimum,
  $7.9 \times 10^{-7}$ above the cert's lower bound — entirely consistent
  with the Taylor B&B cell half-width $r \approx 3.8 \times 10^{-6}$
  (the conservatism comes from the second-derivative interval enclosure
  on each cell, not numerical roundoff). The certified lower bound is
  conservative *by construction*; it is meant to be a rigorous lower
  bound, not a sharp evaluation.
- Endpoint check: $G(0) = 1.00003$, $G(1/4) = 0.99999996$, both
  above the interior minimum.

### V9 ($k_1$)
- **Convention note.** The task statement defines $k_1 = \widetilde K(1/u)$,
  but the codebase (`bisect_alt_kernel.py` and `kernels.py:K_tilde`)
  computes $k_1 = \widehat K(\xi = 1)$ — i.e., the period-1 Fourier
  coefficient, which is the canonical MV-master-inequality input.
  We computed *both* values for transparency:
  - $\widehat{K_{\rm ms}}(1)     = 0.9212465899364082$ (cert match).
  - $\widehat{K_{\rm ms}}(1/u) = 0.8160395822666129$ (alt convention).
  The cert and the Lean anchor both use the $\xi = 1$ convention; the
  CLAUDE.md gloss "$k_1 = \widetilde{K_{\rm ms}}(1/u)$" appears to be a
  documentation slip and should be corrected to "$k_1 = \widehat{K_{\rm ms}}(1)$".
- This is **not** a soundness issue: the certifier, the Lean rational
  anchor, and the cert all use the same $\xi = 1$ convention.

### V12 (Bochner positivity at QP frequencies)
- All 200 weights $w_j = \widehat{K_{\rm ms}}(j/u)$ are strictly positive
  to mpmath 50-dps precision.
- $\min_j w_j = 2.08173795 \times 10^{-4}$ at $j_{\min} = 147$
  ($\xi = 147/0.638 \approx 230.4$).
- This is **substantially better** than the single-scale baseline minimum
  ($\sim 10^{-6}$ near zeros of $J_0$), confirming the rescue mechanism
  documented in CLAUDE.md and `docs/proof_outline.md` that drives $S_1$
  down from 87.4 (single-scale) to 29.84 (three-scale).
- Soft FLAG noted in the prior 2-scale audit ($\min_j w_j = 3.3 \times 10^{-5}$
  at the 2-scale 1.28984 instance) is resolved here: the 3-scale instance
  has $\min_j w_j \ge 2.08 \times 10^{-4}$, an order of magnitude improvement.

## Coordinated summary (for downstream merger)

| Check | Anchor | Claim | Independent | Verdict |
|---|---|---|---|---|
| V7  | $S_1$         | $\le 29.840907$              | $29.84090645551326$            | **CONFIRM** (agreement $\le 4\times 10^{-15}$) |
| V8  | $\min G$      | $\ge 0.99997987$            | $\min G = 0.99998066$, $x^* = 0.24762516$ | **CONFIRM** ($7.9 \times 10^{-7}$ above cert lower; consistent with Taylor B&B radius) |
| V9  | $k_1$         | $\ge 0.92124658$             | $0.9212465899364082$           | **CONFIRM** (matches cert mid to $\sim 10^{-16}$; documentation slip in CLAUDE.md regarding evaluation point noted but **not** a soundness issue) |
| V12 | $\min_j w_j$  | $\ge 2.08\times 10^{-4}$ | $2.0817 \times 10^{-4}$ at $j=147$ | **CONFIRM** (all 200 weights $>0$; rescue mechanism confirmed) |

**Overall verdict:** all four instance-specific anchors at the 3-scale
1.292 point are independently confirmed at mpmath 50-digit precision.
No discrepancy $> 10^{-6}$ in any anchor. No FLAGs.

## Files

- Driver: `archive/attempts/verify_3scale_V7V8V9V12.py`
- Certificate (read-only): `delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
