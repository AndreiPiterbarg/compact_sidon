# A New Lower Bound for the Supremum of Autoconvolutions

This repository accompanies the preprint *A New Lower Bound for the Supremum of Autoconvolutions* (Piterbarg, Bajaj, Vincent), in which we propose a new lower bound on the Sidon autocorrelation constant:

$$C_{1a} \ge \frac{1292}{1000} = 1.292.$$

This improves on the published lower bound of $1.28$ due to Cloninger and Steinerberger (2017) and on the rigorous analytic bound of $1.27481$ established by Matolcsi and Vinuesa (2010). (A separate *unpublished* $1.2802$ is attributed in Tao's `optimizationproblems` repository to Xie (2026, AI-assisted, unaudited) — **not** to Cloninger–Steinerberger.) The argument extends the Matolcsi-Vinuesa dual framework: the single arcsine kernel in their master inequality is replaced by a convex combination of three arcsine kernels, and the cosine multiplier is re-optimized as a $200$-mode expansion.

The paper proof is unconditional at MV~2010's standard of rigour: the four atomic identities MV~2010 Eq.(1)--Eq.(4) are shown to hold for $K_{\mathrm{ms}}$ in Lemma 2.5 (a convex-combination extension of MO~2009 Lemmas~3.1--3.4 / MV~2010 Lemma~3.1; the proof uses only that the admissibility hypotheses (K1)--(K4) are preserved under convex combination — $K_{\mathrm{ms}}$ is a pdf supported in $[-0.138, 0.138]$ with nonnegative periodic Fourier coefficients and $\|K_{\mathrm{ms}}\|_2^2 < \infty$, the only hypotheses those lemmas require). The Lean theorem retains the bundle as a named hypothesis record `ExtremiserPrimitives f` for mathlib-API reasons; packaging a one-call constructor is engineering, not a logical gap. Two **new** numerical inequalities specific to the three-scale kernel and the 200-mode multiplier — analogues *in role* of the Mathematica citations in Matolcsi–Vinuesa — are certified externally in `flint.arb` interval arithmetic at 256-bit precision; Lean 4 machine-checks the algebraic assembly, master inequality, quadratic inversion, slack-soundness, and rational closing.

## Problem statement

Let $\mathcal{F} = \{ f \in L^1(\mathbb{R}) : f \ge 0,\ \mathrm{supp}(f) \subseteq (-1/4,\, 1/4),\ \int f > 0 \}$. The autoconvolution constant is

$$C_{1a} = \inf_{f \in \mathcal{F}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}.$$

By homogeneity one may normalize $\int f = 1$, in which case $C_{1a} = \inf_f \|f * f\|_\infty \ge 1$.

## Repository layout

| Path | Contents |
|------|----------|
| [`lower_bound_proof.pdf`](lower_bound_proof.pdf) / [`lower_bound_proof.tex`](lower_bound_proof.tex) | The paper. |
| [`audit_consistency.py`](audit_consistency.py) | End-to-end audit script: re-runs the production pipeline and verifies every numerical claim across the paper, the Lean module, the JSON anchors, the READMEs, and the docs. |
| [`delsarte_dual/grid_bound/`](delsarte_dual/grid_bound/) | The Matolcsi-Vinuesa machinery in `flint.arb`: cell-search certifier, master inequality, Taylor branch-and-bound for $\min G$, independent verifier. |
| [`delsarte_dual/grid_bound_alt_kernel/`](delsarte_dual/grid_bound_alt_kernel/) | The multi-scale arcsine kernel, the QP for $G$, and the certifier driver `bisect_alt_kernel.py`. |
| [`delsarte_dual/grid_bound_alt_kernel/certificates/`](delsarte_dual/grid_bound_alt_kernel/certificates/) | `reference_anchors.json` (canonical anchors); a fresh run emits `multiscale_arcsine_1292.json` here. |
| [`lean/Sidon/`](lean/Sidon/) | The Lean 4 formalization (~7.5 kLoC, thirteen modules: `Defs`, `Bessel`, `FourierAux`, `TorusParseval`, `MVLemmas`, `MasterFromLemmas`, `BundleDefs`, `BundleEq1`, `BundleEq2Schwartz`, `BundleEq3Schwartz`, `BundleEq4`, `BilinearParseval`, `MultiScale`). All modules are axiom-free except `MultiScale`, which declares the headline theorem and the two verifiable-by-computation user axioms. |
| [`tests/`](tests/) | `pytest` suite covering kernel admissibility, Bochner positivity, the QP solver, and the single-scale baseline. |
| [`docs/`](docs/) | Secondary documentation: proof outline, reproducibility, formalization notes, audit specification, attempts archive. |

## Running the full audit

The single command

```bash
python audit_consistency.py
```

re-runs the production pipeline at 256-bit `flint.arb` precision and
verifies every quantitative claim across all five publication surfaces
(LaTeX paper, Lean module, JSON anchors, READMEs, docs).  Pass
`--verbose` to print every individual check.  Exit code `0` iff every
claim is sound; `1` otherwise.

The eight check sections cover kernel-parameter consistency,
slack-rational soundness, Lean axiom RHS soundness, tight-decimal claims
across surfaces, the LaTeX Proposition 5.1 strict-failure arithmetic
(exact rational verification of `tau - Phi(1292/1000) = 307/3190000`),
per-lemma slack values, the `K_2 = bulk + tail` Watson-tail
decomposition, and the published bound.  Run this first whenever any
numerical value is changed.

For the focused unit-test suite (kernel admissibility, Bochner
positivity, QP convergence, single-scale baseline):

```bash
pytest tests/grid_bound_alt_kernel/
```

To rebuild the Lean formalisation:

```bash
cd lean && lake build
lake env lean AxiomCheckMV.lean  # per-module axiom inventory (also: BundleDefs, Fourier, Torus)
```

The build covers all thirteen Lean modules
(`Sidon.{Defs, Bessel, FourierAux, TorusParseval, MVLemmas,
MasterFromLemmas, BundleDefs, BundleEq1, BundleEq2Schwartz,
BundleEq3Schwartz, BundleEq4, BilinearParseval, MultiScale}`,
~7.5 kLoC on top of mathlib `v4.29.1` at commit
`5e932f97dd25535344f80f9dd8da3aab83df0fe6`). All modules are
axiom-free except `MultiScale`, which declares exactly two
*verifiable-by-computation* user axioms (rigorously certified numerical
assertions) in the headline's dependency closure (see
[Axiom budget](#axiom-budget) below). The headline theorem
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` is parameterised
by an `ExtremiserPrimitives` bundle of MO 2009 / MV 2010 inequalities
discharged by direct citation, with aliases
`autoconvolution_ratio_ge_1_292` (decimal restatement) and
`C1a_ge_1292` (flipped direction) exported from the same namespace.

### Recent corrections

The post-refactor build incorporates: (i) `f ∘ f` is the convolutional
form matching MV 2010 (not a pointwise product); (ii) `K_arc` is the
autoconvolution $\eta_\delta * \eta_\delta$; (iii) the two numerical
axioms bound concrete defined real expressions over the 200 embedded
QP coefficients (`qpNumerators`) and the Bessel-form `Ktilde_ms`, not
opaque symbols; (iv) the bundle's third equation field `hEq3_ge` is
the inequality `2/u + 2·u²·S_cos ≤ LHS1 + LHS2`, derivable from
finite-`J` Parseval + Bochner positivity alone; (v) Schwartz variants
were retired as vacuous via Paley–Wiener + Carlson. The build is clean
(`lake build`, 0 sorries) under the 5-axiom budget (3 Lean core + 2
verifiable-by-computation). See [REPORT.md §4](REPORT.md) for the full
refactor history.

## Key documents in `docs/`

- [`docs/proof_outline.md`](docs/proof_outline.md) — mathematical summary of the proof (master inequality, three-scale kernel, certified anchors, strict-failure witness, lift mechanism, axiom-budget comparison with MV 2010).
- [`docs/reproducibility.md`](docs/reproducibility.md) — exact instructions to reproduce the certificate and build the Lean formalization.
- [`docs/formalization.md`](docs/formalization.md) — the thirteen Lean modules, the headline theorem, the two verifiable-by-computation axioms, and the `ExtremiserPrimitives` record.
- [`docs/verification.md`](docs/verification.md) — public audit specification: fourteen independent checks.

## Axiom budget

The Lean headline theorem
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` reaches exactly
two *verifiable-by-computation* user axioms in its dependency closure,
plus an analytic admissibility-bundle record whose fields are Lean
restatements of MO~2009 / MV~2010 outputs at $(f, K_{\rm ms})$
discharged in the paper by direct citation. The architecture matches
the published Matolcsi--Vinuesa proof but strengthens each layer.

The dependency closure splits into three categorically distinct kinds:

- **Logical axioms** (`propext`, `Classical.choice`, `Quot.sound`) --
  Lean 4 core; trusted without proof; cannot be derived by any finite
  computation.
- **Verifiable-by-computation axioms** -- the two numerical ones.
  Logically *decidable* inequalities about specific real numbers,
  backed by `flint.arb` at 256-bit precision
  (`delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`); they appear as
  `axiom` rather than `theorem` only because mathlib lacks a Bessel
  interval-arithmetic library to discharge them mechanically. They
  are not conjectural -- both are provable; the FlySpeck
  formalisation of Kepler's conjecture used the same convention.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`. Not
  an axiom but a *hypothesis* of the headline.

The three architectural layers:

1. **Analytic content formally proved in Lean** -- ~6.1 kLoC axiom-free
   across `Sidon.{Bessel, FourierAux, TorusParseval, MVLemmas,
   MasterFromLemmas, BundleDefs, BundleEq1, BundleEq2Schwartz,
   BundleEq3Schwartz, BundleEq4, BilinearParseval}` covering the Bessel
   arcsine Fourier-transform identity, the $L^2$-Plancherel / Schwartz
   machinery (on top of mathlib's
   `MeasureTheory.Lp.fourierTransformₗᵢ`), the period-$u$ torus
   Parseval, the four MV Lemma 3.1 atomic primitives (Eqs.(1)--(4))
   with their dedicated discharge modules, the bilinear Parseval
   identity, and the algebraic assembly of the master inequality.
2. **Two verifiable-by-computation axioms** in `Sidon.MultiScale`,
   both analogues of "Mathematica computed this value" in MV 2010
   but backed by `flint.arb` at 256-bit precision. After Option B
   (2026-05-20) both axioms bound *concrete defined* real analytic
   functionals (not opaque symbols):
   - `K2_analytic_le_K2UpperQ`: $\int K_{\rm ms}(x)^2\,\mathrm{d}x
     \le 47897/10000$, where $K_{\rm ms}$ is the explicit three-scale
     arcsine kernel. Paper Lemma 4.2.
   - `gain_analytic_ge_gainLowerQ`: the literal Lean defined functional
     $\texttt{gain\_analytic} := (4 / u_{\rm real}) \cdot
     \min_G^{\,2} / S_1 \ge 20925/100000$, where $\min_G = \mathrm{sInf}
     (G_{\rm concrete}(\![0,1/4]\!))$ and $S_1 = \sum_{j=1}^{200}
     a_j^2 / \widetilde{K_{\rm ms}}(j)$ are evaluated against the 200
     QP coefficients `qpNumerators` (denominator $10^8$, embedded in
     `Sidon.MultiScale`) and the Bessel-form Fourier coefficient
     $\widetilde{K_{\rm ms}}(j)$ built from `Sidon.Bessel.besselJ0`.
     Paper Lemmas 4.3--4.5.
3. **One admissibility-bundle record**
   `ExtremiserPrimitives f`, whose fields are Lean restatements of the
   four MO~2009 / MV~2010 Lemma 3.1 outputs (Eqs.(1)--(4)) at the pair
   $(f, K_{\rm ms})$. MO~2009 Lemmas~3.1--3.4 / MV~2010 Lemma 3.1
   apply to $K_{\rm ms}$ directly (a pdf supported in
   $[-\delta_1, \delta_1]$ with $\widetilde{K_{\rm ms}}(j) \ge 0$ and
   $K_{\rm ms} \in L^2$), so the paper discharges the bundle fields by
   direct citation. The Lean theorem retains them as named hypothesis
   fields only because mathlib `v4.29.1` does not yet expose a
   reusable $L^1 \cap L^2$ Plancherel + period-$u$ Parseval API in the
   form the bundle consumes; the building blocks live in
   `Sidon.TorusParseval` and `Sidon.FourierAux`.

This is a standard axiom architecture for computer-assisted
real-number proofs (Flyspeck cited Kepler's interval arithmetic; the
polynomial-method cap-set proof cited Lagrange polynomial bounds; the
PFR formalisation cited numerical Plünnecke--Ruzsa constants). The mathematical content is in the
Lean theorems; the verifiable-by-computation axioms encode only
"evaluate this specific integral and compare it to this specific
rational."

The distinction between *conjectural* and
*verifiable-by-computation* axioms matters because the natural
critic's question -- "are you assuming something unprovable?" -- has
a clean answer here: **no**. Both numerical axioms are provable;
they are simply not yet formalised in Lean for engineering reasons.

**Caveats.** (i) The Lean theorem carries `ExtremiserPrimitives f`
as a named hypothesis record. Its fields are MO~2009 / MV~2010 Lemma~3.1
outputs at $(f, K_{\rm ms})$, discharged in the paper by direct citation,
exactly as MV~2010 discharged its single-arcsine instance via
MO~2009. The `hEq3_ge` field (replacing the earlier `hEq3` equality)
is genuinely Lean-derivable from finite-`J` Parseval plus Bochner
positivity, no period-`u` Poisson summation needed; the other three
fields (`hEq1`, `hEq2`, `hEq4`) remain paper-discharged inequalities. The Lean retains them as hypothesis fields because the
$L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge has not yet
been packaged into a one-call mathlib constructor — a separate
mathlib-engineering note that does not bear on the validity of the
citation-discharge itself, and is independent of the Bessel
interval-arithmetic engineering gap behind the two numerical axioms.
(ii) The two verifiable-by-computation axioms depend on trusting
`flint.arb` (peer-reviewed, Johansson 2017 IEEE TC, but not
Lean-verified). (iii) Replacing them with verified Lean numerics
would require a separate multi-year subproject (rigorous interval
arithmetic + verified quadrature + verified Bessel + Taylor
branch-and-bound, ~6000-10000 lines), with no upside for the
mathematical claim.

## References

- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), 439-447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), 3191-3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988).
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), 219-235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121).
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), 1281-1292. [arblib.org](https://arblib.org/).
- Watson, G. N. *A Treatise on the Theory of Bessel Functions*, 2nd ed., Cambridge University Press, 1944.
- AlphaEvolve project. *AI-driven optimization of analytic constants arising in extremal combinatorics and additive number theory*, preprint, 2025. [arXiv:2511.02864](https://arxiv.org/abs/2511.02864).
