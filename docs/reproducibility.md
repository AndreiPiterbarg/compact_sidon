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
Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` (the *conditional*
headline, which assumes an `ExtremiserPrimitives f` bundle) reports

```
'Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ]
```

Exactly **two** user axioms appear, both *verifiable-by-computation*.
The *unconditional* headline `#print axioms
Sidon.MultiScale.C1a_ge_1292_unconditional` (which constructs the bundle
from raw admissibility via `ExtremiserPrimitives.of_admissible`) reports
**four** user axioms:

```
'Sidon.MultiScale.C1a_ge_1292_unconditional' depends on axioms:
  [propext, Classical.choice, Quot.sound,
   Sidon.MultiScale.K2_analytic_le_K2UpperQ,
   Sidon.MultiScale.gain_analytic_ge_gainLowerQ,
   Sidon.MultiScale.min_G_analytic_ge_minGLowerQ,
   Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active]
```

All four are *verifiable-by-computation* (i.e. rigorously certified
numerical assertions): each is a logically decidable inequality about a
specific real number, backed by `flint.arb` at 256-bit precision via the
driver
[`../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py)
and mpmath-corroborated.
They appear as `axiom` rather than `theorem` only because mathlib does
not yet ship a Bessel interval-arithmetic library; the FlySpeck
formalisation of Kepler's conjecture used the same convention.

- `K2_analytic_le_K2UpperQ` (both headlines): $K_2(K_{\rm ms}) := \int K_{\rm ms}^2 \le
  47897/10000$. Analogue of "Mathematica computed $K_2 \approx 4.788$"
  in MV 2010, but backed by `flint.arb` at 256-bit precision (paper
  Lemma 4.2). The integrand $K_{\rm ms}^2$ is the explicit three-scale
  arcsine autoconvolution.
- `gain_analytic_ge_gainLowerQ` (both headlines): $\texttt{gain\_analytic} =
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
- `min_G_analytic_ge_minGLowerQ` (unconditional headline only):
  $\texttt{min\_G\_analytic} \ge \texttt{minGLowerQ} = 998/1000$, i.e.
  $\min_{[0,1/4]} G \ge 0.998$ (32768-cell Taylor branch-and-bound;
  paper Lemma 4.3). It enters only the unconditional closure, where the
  constructor `of_admissible` (rather than a consumer-supplied bundle)
  must establish the multiplier floor.
- `Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`
  (unconditional headline only): $\widetilde{K_{\rm ms}}(j) > 0$ for
  every $j \in \{1, \dots, 200\}$ (so the QP denominators in $S_1$ are
  finite; certifier minimum $\ge 2.08 \times 10^{-4}$ at $j = 147$,
  paper Lemma 4.6).

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

The *conditional* headline carries, in addition to its two axioms, an
**analytic admissibility-bundle record** `ExtremiserPrimitives f`
whose fields are Lean restatements of MO~2009 Lemmas~3.1--3.4 /
MV~2010 Lemma 3.1 outputs (Eqs.(1)--(4)) at the specific pair
$(f, K_{\rm ms})$. Those lemmas apply to $K_{\rm ms}$ directly
(a pdf supported in $[-\delta_1, \delta_1]$ with
$\widetilde{K_{\rm ms}}(j) \ge 0$ and $K_{\rm ms} \in L^2$), and the
paper discharges the bundle fields by direct citation; the conditional
Lean theorem retains them as named hypothesis fields. The *unconditional*
headline removes this hypothesis: the bundle is *constructed* axiom-free
from raw admissibility by `ExtremiserPrimitives.of_admissible` (in
`Sidon.Constructor.Assembly`), which mechanises the $L^1 \cap L^2$
Plancherel + period-$u$ Parseval bridge built around
`Sidon.TorusParseval`, `Sidon.FourierAux`, and the 17-module
`Sidon.Constructor.*` chain. This establishes $C_{1a} \ge 1.292$ to at
least the rigour of the accepted MV 2010 proof of $C_{1a} \ge 1.2748$,
strengthened by the axiom-free mechanisation of the analytic content and
arb-backed, mpmath-corroborated numerics; the honest caveat is that the
numerical axioms remain computer-assisted and the verification to date is
rigorous AI-agent self-audit rather than third-party referee review.

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
statements, the theorem statements, and the module layout (30 modules
totalling ~15.6 kLoC: 13 core `Sidon/*.lean` plus 17
`Sidon/Constructor/*.lean`). The Schwartz-class headline
variants previously listed here
(`autoconvolution_ratio_ge_1292_1000_schwartz` and
`autoconvolution_ratio_ge_1292_1000_schwartz_residual`) were retired
during the S1+S2 refactor (2026-05) as vacuously true: by
Paley--Wiener combined with Carlson's theorem, no nontrivial Schwartz
function $f$ compactly supported in $(-1/4, 1/4)$ can satisfy the
periodic Parseval-split predicate they relied on.
