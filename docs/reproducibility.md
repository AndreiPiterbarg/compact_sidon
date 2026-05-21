# Reproducibility

Exact commands to reproduce the `flint.arb` certificate for the
**Piterbarg--Bajaj--Vincent Bound** $C_{1a} \ge 1292/1000 = 1.292$ and
to build the Lean 4 formalization.

## Prerequisites

- **Python** 3.11 or newer.
- **Python packages.** `python-flint >= 0.6` (arb / acb / fmpq backend),
  `numpy`, `mpmath`, `cvxpy`. The QP step prefers `mosek` (academic licence
  available); a `clarabel`, `scs`, or `ecos` fallback is used automatically.
- **Lean 4 toolchain.** The repository pins `leanprover/lean4:v4.29.1`
  via `lean/lean-toolchain`, with `mathlib` pinned to commit
  [`5e932f97dd25535344f80f9dd8da3aab83df0fe6`](https://github.com/leanprover-community/mathlib4/commit/5e932f97dd25535344f80f9dd8da3aab83df0fe6)
  (post-Nov 2025). The `v4.29.1` bump is required because the
  formalisation relies on the $L^2$-Plancherel API
  (`MeasureTheory.Lp.fourierTransformₗᵢ`) and convolution--Fourier
  duality (`Real.fourier_mul_convolution_eq`), both first available
  at that mathlib commit. Install
  [`elan`](https://github.com/leanprover/elan) and let `lake` pick up
  the pinned versions on first build.

## One-line install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install python-flint numpy mpmath cvxpy mosek
```

`mosek` may be replaced by `clarabel`; the driver selects the first solver
it finds. The loose-pin dependency floor used by the broader repository is
recorded in [`../requirements.txt`](../requirements.txt) (`numpy>=2.0`,
`mpmath>=1.3`, `python-flint>=0.6`, etc.); the certifier driver runs on any
combination satisfying those floors.

## Reproducing the certificate

The certifier driver lives at
[`delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py).
Run it as a module:

```bash
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

Defaults reproduce the published bound: the three-scale arcsine kernel at
$(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$ with weights $(85, 10,
5)/100$, a 200-coefficient cosine multiplier $G$ re-optimized against this
kernel, all anchors in arb at 256-bit precision, cell-search bisection on
$M$ targeting `1292/1000`.

### Expected output

The driver prints the five anchors and writes a self-contained certificate
to `delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`.
Reference values are recorded in
[`reference_anchors.json`](../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json):

| Anchor      | Bound                                          |
|-------------|------------------------------------------------|
| $k_1$       | $\ge 0.92124658$ (radius $< 7 \times 10^{-77}$) |
| $K_2$       | $\in [4.788823,\; 4.788906]$                   |
| $S_1$       | $\le 29.840907$                                |
| $\min G$    | $\ge 0.99997987$                               |
| gain $a$    | $\ge 0.21009214$                               |
| $M_{\rm cert}$ (production) | $= 66167/51200 \approx 1.29232422$ |
| $M_{\rm cert}$ (slack-anchor) | $\ge 1.29215650$ (`reference_anchors.json`) |
| Headline rational target | $1292/1000$ |

Wall time on a modern laptop is roughly 11 s at 256-bit precision.

### Certificate hash

The emitted JSON has the form `{"sha256_of_body": <digest>, "body": {...}}`.
To re-derive the digest:

```bash
python -c "import json, hashlib; d = json.load(open('delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json')); print(hashlib.sha256(json.dumps(d['body'], indent=2, sort_keys=True).encode()).hexdigest())"
```

It must match the certificate's `sha256_of_body` field.

### Independent verifier

`delsarte_dual/grid_bound/certify.py` re-checks every quantitative claim
using only `python-flint` primitives:

```bash
python -m delsarte_dual.grid_bound.certify \
  delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json
```

Exit code `0` on success.

## Building the Lean formalization

```bash
cd lean
lake build                     # full proof chain
lake build Sidon.MultiScale    # headline module only
```

Expected result: exit code `0`, no `sorry` warnings.

### Axiom inventory

```bash
cd lean
lake env lean AxiomCheckMV.lean        # MV-lemmas axiom closure
lake env lean AxiomCheckBundleDefs.lean # bundle definitions axiom closure
lake env lean AxiomCheckFourier.lean   # Fourier-aux axiom closure
lake env lean AxiomCheckTorus.lean     # torus-Parseval axiom closure
```

The four per-module check files print the axiom dependency closure of
their respective imports. After `lake build`, `#print axioms
Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` reports

```
'Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ]
```

Exactly **two** user axioms appear in the dependency closure, both
*verifiable-by-computation* (i.e. rigorously certified numerical
assertions): each is a logically decidable inequality about a specific
real number, backed by `flint.arb` at 256-bit precision via the driver
[`../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py).
They appear as `axiom` rather than `theorem` only because mathlib does
not yet ship a Bessel interval-arithmetic library; the FlySpeck
formalisation of Kepler's conjecture used the same convention.

- `K2_analytic_le_K2UpperQ`: $K_2(K_{\rm ms}) := \int K_{\rm ms}^2 \le
  47897/10000$. Analogue of "Mathematica computed $K_2 \approx 4.788$"
  in MV 2010, but backed by `flint.arb` at 256-bit precision (paper
  Lemma 4.2). The integrand $K_{\rm ms}^2$ is the explicit three-scale
  arcsine autoconvolution.
- `gain_analytic_ge_gainLowerQ`: $\texttt{gain\_analytic} =
  (4/u_{\rm real}) \cdot \texttt{min\_G\_analytic}^2 /
  \texttt{S\_1\_analytic} \ge 20925/100000$. The RHS is a *concrete
  defined* `noncomputable def` (post-Option-B, 2026-05-20) built from
  (i) the 200 QP coefficient numerators
  `Sidon.MultiScale.qpNumerators : List ℤ` embedded with common
  denominator $10^8$ (length verified by `native_decide`), and (ii) the
  Bessel-form period-$u$ Fourier coefficient
  `Ktilde_ms j := Σᵢ λᵢ · besselJ0(πjδᵢ/u)²` built on
  `Sidon.Bessel.besselJ0`. Analogue of MV's Mathematica citation of $a$,
  certifier-coupled in arb (paper Lemmas 4.3--4.5).

#### Cross-binding the 200 QP coefficients

Because the 200 QP coefficients now appear in two locations -- the
JSON certificate and the Lean `qpNumerators` list -- the sibling
script `audit_qp_coeffs.py` verifies that the integer numerators
agree exactly between the certificate body (under the
`qp_coefficients` field) and the Lean embedding at
[`lean/Sidon/MultiScale.lean:523`](../lean/Sidon/MultiScale.lean).
Run

```bash
python audit_qp_coeffs.py
```

after either source is regenerated; exit code `0` iff the 200
integer numerators are bit-identical in both files.

In addition to these two axioms, the headline theorem carries an
**analytic admissibility-bundle record** `ExtremiserPrimitives f`
whose fields are Lean restatements of MO~2009 Lemmas~3.1--3.4 /
MV~2010 Lemma 3.1 outputs (Eqs.(1)--(4)) at the specific pair
$(f, K_{\rm ms})$. Those lemmas apply to $K_{\rm ms}$ directly
(a pdf supported in $[-\delta_1, \delta_1]$ with
$\widetilde{K_{\rm ms}}(j) \ge 0$ and $K_{\rm ms} \in L^2$), and the
paper discharges the bundle fields by direct citation. The Lean
theorem retains them as named hypothesis fields only because the
$L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge that
`Sidon.TorusParseval` and `Sidon.FourierAux` are built around has not
yet been packaged into a one-line mathlib call.

The previous macro axiom `MV_master_inequality_for_extremiser` is
**now a Lean theorem** (post-Wave-12 restructuring); its
content factors through the bundle hypothesis plus the two
verifiable-by-computation axioms above. The quadratic inversion
`master_inequality_M_lower`, the slack-monotonicity lift
`MV_master_via_slack_monotonicity`, the full MV-Lemmas chain
`MV_master_inequality_from_MV_lemmas`, and the five slack-soundness
statements (`K_two_upper_bound`, `k_one_lower_bound`,
`S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound`) are also
Lean *theorems* -- none of them contributes an axiom to the dependency
closure. See [`formalization.md`](formalization.md) for the axiom
statements, the theorem statements, and the module layout (thirteen
modules totalling roughly 7.5 kLoC). The Schwartz-class headline
variants previously listed here
(`autoconvolution_ratio_ge_1292_1000_schwartz` and
`autoconvolution_ratio_ge_1292_1000_schwartz_residual`) were retired
during the S1+S2 refactor (2026-05) as vacuously true: by
Paley--Wiener combined with Carlson's theorem, no nontrivial Schwartz
function $f$ compactly supported in $(-1/4, 1/4)$ can satisfy the
periodic Parseval-split predicate they relied on.
