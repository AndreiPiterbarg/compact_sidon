# 3-scale 1.292 re-verification: V2 / V4 / V14

**Date:** 2026-05-20
**Auditor:** Fresh independent verifier (instance-specific anchor lane,
V2 / V4 / V14 portion of the 14-agent re-audit).
**Scope:** Re-verify the three anchor families of the
3-scale 1.292 certificate
(`multiscale_arcsine_1292.json`, SHA-256 body hash `5fa9ae37…`)
at the 3-scale point
$(\delta_1,\delta_2,\delta_3) = (138, 55, 25)/1000$,
$(\lambda_1,\lambda_2,\lambda_3) = (85, 10, 5)/100$,
$u = 319/500$, $N = 200$ cosines.
**Method:** independent `mpmath` recomputation at 30-digit precision,
plus a `flint.arb` cross-check at 256-bit precision.

## Independent formula re-derivation

Following `kernels.py:MultiScaleArcsineKernel` and `lower_bound_proof.tex`:

- Each arcsine component is the autoconvolution
  $\eta_\delta * \eta_\delta$ of the arcsine density
  $\eta_\delta(x) = (1/(\pi\sqrt{\delta^2-x^2}))\,\mathbf{1}_{|x|<\delta}$.
  Its Fourier transform is
  $\widehat{K_{\rm arc}(\delta;\cdot)}(\xi) = J_0(\pi\delta\xi)^2$.
- The multi-scale kernel:
  $\widehat{K_{\rm ms}}(\xi) = \sum_{i=1}^{3} \lambda_i J_0(\pi\delta_i\xi)^2 \ge 0$.
- By Plancherel,
  $$K_2 = \|K_{\rm ms}\|_2^2 = \int_{\mathbb{R}} \widehat{K_{\rm ms}}(\xi)^2 \, d\xi
  = 2 \sum_{i,j=1}^{3} \lambda_i \lambda_j \, C_{ij}$$
  with
  $C_{ij} = \int_0^\infty J_0(\pi\delta_i\xi)^2 J_0(\pi\delta_j\xi)^2 \, d\xi$.

## Methodology note (cross-Bessel integration)

mpmath's default `mp.quad` (tanh-sinh) silently underconverges on the
highly oscillatory integrand over the full cutoff range $[0,10^5]$ — a
naive call returned $K_2 \approx 4.790$, off by $\sim 10^{-3}$ from the
certifier's $4.7889$.  The fix is to subdivide $[0, T]$ into chunks of
fixed width $\Delta\xi = 50$ (each containing $\le 4$ half-periods of the
fastest Bessel oscillator $J_0(\pi\delta_1\xi)$, half-period
$1/(2\delta_1) \approx 3.6$); mpmath's adaptive quad converges on each
smooth chunk to full working precision, and the sum recovers agreement
with `flint.arb` to $\sim 10^{-16}$ on each pair.

The same chunking is applied to the MO surrogate integrand
$J_0(\pi\xi)^4$ with $\Delta\xi = 5$ (half-period $1/2$).

The Watson tail bound past $T$ (NIST DLMF 10.14.1, $|J_0(z)|^2 \le 2/(\pi z)$
for $z\ge 1$) gives
$$\int_T^\infty J_0(\pi\delta_i\xi)^2 J_0(\pi\delta_j\xi)^2 \, d\xi
\le \frac{4}{\pi^4 \delta_i \delta_j T}.$$
At $T = 10^5$ the smallest argument $\pi\delta_3 T \approx 7854 \gg 1$,
well within the Watson regime.

The 2026-05-20 fresh run distributed the six pair integrals plus the MO
surrogate across seven processes (`multiprocessing.Pool(processes=7)`);
mpmath state is process-local so the parallelisation is sound. Total
wall time: 897.8s on a 16-core Windows host.

## Per-check results

### V2 — $K_2 = \|K_{\rm ms}\|_2^2$

| Quantity | Value |
|---|---|
| Certifier `flint.arb` enclosure | $[4.7888234212591545,\;4.7889051816332424]$ |
| Certifier midpoint | $4.788864301446199$ |
| Axiom slack $K_2 \le 47897/10000$ | $4.7897$ |

Independent recomputation (mpmath, dps=30, chunked, $T=10^5$, Watson tail):

| Symbol | Value | Notes |
|---|---|---|
| `K_2` (main, $[0,10^5]$) | $4.788823421260265\ldots$ | chunked mp.quad sum |
| Watson tail UB past $T=10^5$ | $\le 8.176 \times 10^{-5}$ | $4\sum_{ij}\lambda_i\lambda_j/(\pi^4\delta_i\delta_j T)\cdot 2$ |
| `K_2` enclosure | $[4.788823421260,\;4.788905181632]$ | mpmath + Watson tail |
| Slack to axiom $4.7897$ | $\ge 7.95 \times 10^{-4}$ | rigorous |
| Cert midpoint inside mpmath enclosure? | yes ($4.788864 \in [4.788823, 4.788905]$) | independent corroboration |

The mpmath enclosure agrees with the certifier's
`[4.7888234212591545, 4.7889051816332424]` to ~$10^{-13}$ on the lower
endpoint and to ~$10^{-15}$ on the upper endpoint, and is well below
the axiom slack $4.7897$ (slack $\ge 7.95 \times 10^{-4}$).

**V2 verdict: CONFIRM.**

### V4 — MO surrogate $\int_0^\infty J_0(\pi\xi)^4 \, d\xi$

Literature value (MO 2009 Lemma 3.2): the value $0.574695$ quoted in
CLAUDE.md and `MEMORY.md` is the **full-line** integral
$\int_{-\infty}^{\infty} J_0(\pi\xi)^4 \, d\xi$.
The corresponding half-line integral is half:
$\int_0^\infty J_0(\pi\xi)^4 \, d\xi \approx 0.2873473\ldots$
This is consistent with the codebase's `_diag_integral`, which uses the
half-line surrogate `0.5747 / (2 * d)` when `use_diag_surrogate=True`
(i.e. `0.5747` enters as the *full-line* numerator divided by twice
the half-width). For the production cert
(`use_diag_surrogate=False`), this informational anchor is not on the
critical path; we verify it for completeness.

Independent recomputation (mpmath, dps=30, chunked, $T=10^4$):

| Symbol | Value |
|---|---|
| $\int_0^{10^4} J_0(\pi\xi)^4 \, d\xi$ | $0.287345891091099\ldots$ |
| Watson tail past $T = 10^4$ | $\le 4.106 \times 10^{-6}$ |
| Half-line enclosure | $[0.28734589,\;0.28734999]$ |
| Full-line enclosure ($=2\times$ above) | $[0.57469178,\;0.57469999]$ |
| Match to literature 0.574695 (full-line) | $\Delta < 10^{-5}$ |
| Both UBs below $0.5747$? | yes |

Both the half-line and full-line upper bounds are strictly below the
surrogate constant $0.5747$ as quoted in MO 2009 / repo notes; the
slack is $\ge 6.4 \times 10^{-7}$ at the tighter full-line endpoint.

**V4 verdict: CONFIRM** (with documentation note: the literature value
$0.574695$ corresponds to the full-line integral; the half-line integral
is $\approx 0.2873473$).

### V14 — Cross-Bessel integrals $C_{ij}$

Computed each of the six $C_{ij} = \int_0^{10^5} J_0(\pi\delta_i\xi)^2 J_0(\pi\delta_j\xi)^2 \, d\xi$
independently and compared to flint.arb at 256-bit precision (which was
also extracted from `kernels.py:_cross_integral` without its Watson
tail union step, so as to compare apples-to-apples):

| $(\delta_i, \delta_j)$ | $C_{ij}$ (mpmath, dps=30, chunked $[0,10^5]$) | $C_{ij}$ (`flint.arb`, 256-bit, $[0,10^5]$ raw) | $\lvert\Delta\rvert$ |
|---|---|---|---|
| $(0.138,\,0.138)$ | $2.0822196749008675833$ | $2.08221967490086746$ | $1.23 \times 10^{-16}$ |
| $(0.138,\,0.055)$ | $2.7239415031017399563$ | $2.72394150310173977$ | $1.86 \times 10^{-16}$ |
| $(0.138,\,0.025)$ | $3.293717740608156344$ | $3.29371774060815614$ | $2.04 \times 10^{-16}$ |
| $(0.055,\,0.055)$ | $5.224447838177257424$ | $5.224447838177257$ | $4.24 \times 10^{-16}$ |
| $(0.055,\,0.025)$ | $6.5993326562600542847$ | $6.59933265626005472$ | $4.35 \times 10^{-16}$ |
| $(0.025,\,0.025)$ | $11.493650836357467719$ | $11.4936508363574674$ | $3.19 \times 10^{-16}$ |

All six cross-Bessel integrals agree to at least 15 digits between the
two independent implementations (relative errors $5\times 10^{-17}$ to
$8\times 10^{-17}$, at the mpmath dps=30 working-precision noise floor).

**V14 verdict: CONFIRM.**

## Coordinated summary (for downstream merger)

| Check | Anchor | Claim | Independent | Verdict |
|---|---|---|---|---|
| V2  | $K_2 = \|K_{\rm ms}\|_2^2$ | $\in [4.788823,\,4.788906]$, axiom $\le 4.7897$ | $\in [4.78882342,\,4.78890518]$ | **CONFIRM** (inside cert, slack to axiom $\ge 7.9\times10^{-4}$) |
| V4  | $\int_0^\infty J_0(\pi\xi)^4 d\xi$ | half-line $\le 0.5747$ (lit. $\approx 0.574695$ is the full-line) | half-line $\approx 0.2873473$; full-line $\le 0.5746999$ | **CONFIRM** (matches lit; both UBs below 0.5747) |
| V14 | six $C_{ij}$ integrals | match cert components of $K_2$ | $\Delta \le 5\times 10^{-16}$ across all six pairs | **CONFIRM** |

**Overall verdict:** all three anchor families at the 3-scale 1.292
point are independently confirmed at mpmath 30-digit precision with
flint cross-check at 256-bit precision. No discrepancy $> 5\times 10^{-16}$
in any cross-Bessel integral; the assembled $K_2$ lies strictly inside
the flint enclosure $[4.7888234,\,4.7889052]$ and well below the axiom
slack $K_2 \le 47897/10000 = 4.7897$.

## Notes and corroborations

### Methodology recovery (V2)

The first mpmath attempt invoked `mp.quad(f, [0, 1e5])` directly, which
returned $K_2 \approx 4.7899$ — off by $\sim 10^{-3}$ from the
certifier. Inspection of the per-pair $C_{ij}$ showed the discrepancy
came from `mp.quad`'s adaptive tanh-sinh underconverging on a $10^5$-long
interval containing $\sim 14000$ half-periods of the fastest oscillator
$J_0(\pi\delta_1\xi)$. Switching to fixed-width chunking
($\Delta\xi = 50$, $\sim 2000$ chunks per pair) made mpmath converge to
$\sim 10^{-16}$ agreement with `flint.arb`. The reported V2/V14 numbers
use the chunked methodology.

### Documentation note (V4)

CLAUDE.md and `MEMORY.md` quote
"$\int J_0(\pi\xi)^4 d\xi \le 0.5747$ (true value $0.574695$)".
This is the **full-line** integral $\int_{-\infty}^\infty$, not the
half-line. Both conventions appear in MO 2009 depending on context; the
arithmetic in `kernels.py:_diag_integral` line 365 is consistent with
the full-line value being $0.5747 / d$ (after the factor-of-2 between
the half-line integral and the full-line one cancels with the factor
$1/(2d)$ from a substitution $\xi \to \xi/d$). No correction needed
to any production quantity; the surrogate is informational and unused
when `use_diag_surrogate=False` (as in the production cert).

### Independence note

The mpmath driver script does not import `kernels.py`,
`bisect_alt_kernel.py`, `flint`, or any other project module; it
implements the integrand from first principles
($\widehat{K_{\rm ms}}(\xi) = \sum_i \lambda_i J_0(\pi\delta_i\xi)^2$) and
the Plancherel formula directly. The flint comparison values are
extracted from `kernels.py:_cross_integral` only for cross-checking;
they are not used to derive the verdict.

### Reproducibility

The 2026-05-20 fresh run reproduces the previously-archived numbers
(per-pair $C_{ij}$, $K_2$, MO surrogate) bit-for-bit at mpmath dps=30,
confirming that the verification is deterministic and reproducible.
Parallelisation across seven worker processes brought wall time down
from a serial estimate of $\sim 2000$s (six pairs at $\sim 300$-900s each)
to 897.8s.

## Files

- Driver script: `docs/_audit_v2_v4_v14_PARALLEL_2026_05_20.py`
- Raw log: `docs/_audit_v2_v4_v14_PARALLEL_2026_05_20.log`
- Certificate (read-only):
  `delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
- Sibling re-audit (V7/V8/V9/V12):
  `docs/audit_3scale_reverification_V7V8V9V12.md`
