# A New Lower Bound for the Supremum of Autoconvolutions

This repository accompanies the preprint *A New Lower Bound for the Supremum of Autoconvolutions* (Piterbarg, Bajaj, Vincent), in which we propose a new lower bound on the Sidon autocorrelation constant:

$$C_{1a} \ge \frac{1292}{1000} = 1.292.$$

This improves on the published lower bound of $1.28$ due to Cloninger and Steinerberger (2017) and on the rigorous analytic bound of $1.27481$ established by Matolcsi and Vinuesa (2010). (A separate *unpublished* $1.2802$ is attributed in Tao's `optimizationproblems` repository to Xie (2026, AI-assisted, unaudited) — **not** to Cloninger–Steinerberger.) The argument extends the Matolcsi-Vinuesa dual framework: the single arcsine kernel in their master inequality is replaced by a convex combination of three arcsine kernels, and the cosine multiplier is re-optimized as a $200$-mode expansion.

**We have established $C_{1a} \ge 1.292$ to at least the rigor of Matolcsi–Vinuesa 2010** — the accepted, peer-reviewed proof of $C_{1a} \ge 1.2748$. MV's proof has three layers: (i) formal analytic lemmas (their 3.1–3.4, via Martin–O'Bryant 2009) proved on paper; (ii) specific numerical values cited from Mathematica ($\int J_0^4 \le 0.5747$, $m_G$, $S_1$, gain $a = 0.0713$); (iii) algebraic assembly into the master inequality and quadratic inversion. Our proof matches this architecture and strengthens each layer. **(1) Analytic content:** the four atomic identities MV~2010 Eq.(1)--Eq.(4) are shown to hold for $K_{\mathrm{ms}}$ in Lemma 2.5 (a convex-combination extension of MO~2009 Lemmas~3.1--3.4 / MV~2010 Lemma~3.1; the proof uses only that the admissibility hypotheses (K1)--(K4) are preserved under convex combination — $K_{\mathrm{ms}}$ is a pdf supported in $[-0.138, 0.138]$ with nonnegative periodic Fourier coefficients and $\|K_{\mathrm{ms}}\|_2^2 < \infty$, the only hypotheses those lemmas require). This is *also mechanized axiom-free in Lean*: the unconditional headline `C1a_ge_1292_unconditional` **constructs** the entire MV-Lemma-3.1 bundle for any admissible $f$ via `ExtremiserPrimitives.of_admissible`, so the bundle is no longer a named hypothesis on that path. **(2) Numerical content:** the four numerical inequalities specific to the three-scale kernel and the 200-mode multiplier are the *same kind* of computer-checked fact as MV's Mathematica citations, but certified in `flint.arb` interval arithmetic at 256-bit precision (rigorous, not heuristic CAS), independently corroborated by mpmath, and hash-anchored. **(3) Assembly:** Lean 4 machine-checks the algebraic assembly, master inequality, quadratic inversion, slack-soundness, and rational closing — with a positive closing margin $307/3190000$ (true threshold $M^\ast = 1.29203$, so $1.292$ is a safe round-down). **Honest caveats:** this is computer-assisted rigor at MV's standard — the four numerical axioms are decidable and arb-backed but *not eliminated* (a pure-Lean-kernel proof would need Flyspeck-scale verified interval arithmetic in mathlib); and verification is by rigorous AI-agent self-audit across multiple independent passes, not third-party referee review.

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
| [`lean/Sidon/`](lean/Sidon/) | The Lean 4 formalization. Thirteen *core* modules (7655 LoC): `Defs`, `Bessel`, `FourierAux`, `TorusParseval`, `MVLemmas`, `MasterFromLemmas`, `BundleDefs`, `BundleEq1`, `BundleEq2Schwartz`, `BundleEq3Schwartz`, `BundleEq4`, `BilinearParseval`, `MultiScale`; all axiom-free except `MultiScale` (which declares the conditional headline and three verifiable-by-computation axioms: `K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`, `min_G_analytic_ge_minGLowerQ`). |
| [`lean/Sidon/Constructor/`](lean/Sidon/Constructor/) | The bundle constructor `ExtremiserPrimitives.of_admissible` and the unconditional headline `C1a_ge_1292_unconditional` (17 modules, 7922 LoC: `ConvFourier`, `KernelFacts`, `KernelL2`, `YoungConvolution`, `FieldsEasy`, `FieldsParseval`, `FieldEq4`, `Eq2Split`, `Eq2Period1`, `MOLemma21`, `PeriodUParseval`, `PoissonSampling`, `PoissonSummable`, `CauchySchwarzFloor`, `Glue`, `LatticePositivity`, `Assembly`). All axiom-free except `LatticePositivity`, which declares the fourth axiom `K_ms_fourier_lattice_pos_active`. |
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

The build covers all **30 Lean modules** on top of mathlib `v4.29.1`
at commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`: the 13 *core*
modules `Sidon.{Defs, Bessel, FourierAux, TorusParseval, MVLemmas,
MasterFromLemmas, BundleDefs, BundleEq1, BundleEq2Schwartz,
BundleEq3Schwartz, BundleEq4, BilinearParseval, MultiScale}`
(7655 LoC) plus the 17 `Sidon.Constructor.*` modules (7922 LoC);
~15.6 kLoC total. All modules are axiom-free except `MultiScale`
(which declares three numerical axioms — `K2_analytic_le_K2UpperQ`,
`gain_analytic_ge_gainLowerQ`, `min_G_analytic_ge_minGLowerQ`) and
`Constructor.LatticePositivity` (which declares the fourth,
`K_ms_fourier_lattice_pos_active`). There are **two headlines** (see
[Two headlines](#two-headlines-conditional-and-unconditional) below):
the **conditional**
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` (aliases
`autoconvolution_ratio_ge_1_292`, `C1a_ge_1292`), which carries an
`ExtremiserPrimitives` bundle hypothesis and reaches **two**
verifiable-by-computation axioms; and the **unconditional**
`Sidon.MultiScale.C1a_ge_1292_unconditional`, which takes only
admissibility, *constructs* that bundle, and reaches **four**.

### Recent corrections

The post-refactor build incorporates: (i) `f ∘ f` is the convolutional
form matching MV 2010 (not a pointwise product); (ii) `K_arc` is the
autoconvolution $\eta_\delta * \eta_\delta$; (iii) the two numerical
axioms bound concrete defined real expressions over the 200 embedded
QP coefficients (`qpNumerators`) and the Bessel-form `Ktilde_ms`, not
opaque symbols; (iv) the bundle's third equation field `hEq3_ge` is
the inequality `2/u + 2·u²·S_cos ≤ LHS1 + LHS2`, derivable from
finite-`J` Parseval + Bochner positivity alone; (v) Schwartz variants
were retired as vacuous via Paley–Wiener + Carlson; (vi) the
`ExtremiserPrimitives` real parameters `m_G, S_G, S_cos, LHS1, LHS2` are
now **bound** to the concrete analytic functionals via five `*_eq`
hypothesis fields, so the bundle forces the canonical `(f, K_ms)` values
and is no longer satisfiable by arbitrary reals. The build is clean
(`lake build`, 0 sorries): the conditional headline reaches a 5-axiom
budget (3 Lean core + 2 verifiable-by-computation), the unconditional
headline 7 (3 Lean core + 4). See [REPORT.md §4](REPORT.md) for the full
refactor history.

## Key documents in `docs/`

- [`docs/proof_outline.md`](docs/proof_outline.md) — mathematical summary of the proof (master inequality, three-scale kernel, certified anchors, strict-failure witness, lift mechanism, axiom-budget comparison with MV 2010).
- [`docs/reproducibility.md`](docs/reproducibility.md) — exact instructions to reproduce the certificate and build the Lean formalization.
- [`docs/formalization.md`](docs/formalization.md) — the 30 Lean modules (13 core + 17 Constructor), the two headline theorems (conditional and unconditional), the four verifiable-by-computation axioms, and the `ExtremiserPrimitives` record.
- [`docs/verification.md`](docs/verification.md) — public audit specification: fourteen independent checks.

## Axiom budget

The **conditional** Lean headline theorem
`Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000` reaches exactly
two *verifiable-by-computation* user axioms in its dependency closure,
plus an analytic admissibility-bundle record (`ExtremiserPrimitives f`)
whose fields are Lean restatements of MO~2009 / MV~2010 outputs at
$(f, K_{\rm ms})$ discharged in the paper by direct citation. The
**unconditional** headline `Sidon.MultiScale.C1a_ge_1292_unconditional`
*constructs* that bundle from raw admissibility and so reaches **four**
verifiable-by-computation axioms (the two below, plus
`min_G_analytic_ge_minGLowerQ` and `K_ms_fourier_lattice_pos_active`);
see [Two headlines](#two-headlines-conditional-and-unconditional) below.
The architecture matches the published Matolcsi--Vinuesa proof but
strengthens each layer.

The dependency closure splits into three categorically distinct kinds:

- **Logical axioms** (`propext`, `Classical.choice`, `Quot.sound`) --
  Lean 4 core; trusted without proof; cannot be derived by any finite
  computation.
- **Verifiable-by-computation axioms** -- two reached by the
  conditional headline (`K2_analytic_le_K2UpperQ`,
  `gain_analytic_ge_gainLowerQ`), four by the unconditional headline
  (those two plus `min_G_analytic_ge_minGLowerQ` and
  `K_ms_fourier_lattice_pos_active`). All are logically *decidable*
  inequalities about specific real numbers, backed by `flint.arb` at
  256-bit precision
  (`delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`) and
  independently corroborated by mpmath; they appear as `axiom` rather
  than `theorem` only because mathlib lacks a Bessel
  interval-arithmetic library to discharge them mechanically. They
  are not conjectural -- all are provable; the FlySpeck
  formalisation of Kepler's conjecture used the same convention.
- **Analytic admissibility bundle** -- `ExtremiserPrimitives f`. Not
  an axiom: a *hypothesis* of the conditional headline, and
  *constructed* from raw admissibility by `of_admissible` for the
  unconditional headline.

The three architectural layers:

1. **Analytic content formally proved in Lean** -- axiom-free across
   `Sidon.{Bessel, FourierAux, TorusParseval, MVLemmas,
   MasterFromLemmas, BundleDefs, BundleEq1, BundleEq2Schwartz,
   BundleEq3Schwartz, BundleEq4, BilinearParseval}` covering the Bessel
   arcsine Fourier-transform identity, the $L^2$-Plancherel / Schwartz
   machinery (on top of mathlib's
   `MeasureTheory.Lp.fourierTransformₗᵢ`), the period-$u$ torus
   Parseval, the four MV Lemma 3.1 atomic primitives (Eqs.(1)--(4))
   with their dedicated discharge modules, the bilinear Parseval
   identity, and the algebraic assembly of the master inequality;
   PLUS the entire 17-module `Sidon.Constructor.*` layer (7922 LoC,
   axiom-free except `LatticePositivity`) that *constructs* the
   MV-Lemma-3.1 bundle for any admissible $f$. The MV-paper analytic
   layer is thus mechanized in Lean, not merely assumed.
2. **Verifiable-by-computation axioms** (two for the conditional
   headline, four for the unconditional) in `Sidon.MultiScale` and
   `Sidon.Constructor.LatticePositivity`, each an analogue of
   "Mathematica computed this value" in MV 2010
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
   `Sidon.TorusParseval` and `Sidon.FourierAux`. The record's real
   parameters `m_G, S_G, S_cos, LHS1, LHS2` are bound to the concrete
   analytic functionals by five `*_eq` hypothesis fields (`m_G_eq`,
   `S_G_eq`, `S_cos_eq`, `LHS1_eq`, `LHS2_eq`), so any instance forces
   the canonical $(f, K_{\rm ms})$ values rather than arbitrary reals.

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
a clean answer here: **no**. All four numerical axioms are provable;
they are simply not yet formalised in Lean for engineering reasons.

**Caveats.** (i) The **conditional** headline carries
`ExtremiserPrimitives f` as a named hypothesis record (the
**unconditional** headline *constructs* it via `of_admissible`). Its
fields are MO~2009 / MV~2010 Lemma~3.1
outputs at $(f, K_{\rm ms})$, discharged in the paper by direct citation,
exactly as MV~2010 discharged its single-arcsine instance via
MO~2009. The `hEq3_ge` field (replacing the earlier `hEq3` equality)
is genuinely Lean-derivable from finite-`J` Parseval plus Bochner
positivity, no period-`u` Poisson summation needed; the other three
fields (`hEq1`, `hEq2`, `hEq4`) are paper-discharged inequalities for
the conditional headline and Lean-constructed (in the `Constructor/`
layer) for the unconditional headline. On the conditional path the
Lean retains them as hypothesis fields because the
$L^1 \cap L^2$ Plancherel + period-$u$ Parseval bridge is not yet
packaged into a one-call mathlib constructor — a separate
mathlib-engineering note that does not bear on the validity of the
citation-discharge itself, and is independent of the Bessel
interval-arithmetic engineering gap behind the numerical axioms.
(ii) The verifiable-by-computation axioms (two for the conditional
headline, four for the unconditional) depend on trusting
`flint.arb` (peer-reviewed, Johansson 2017 IEEE TC, but not
Lean-verified); they are mpmath-corroborated but not eliminated.
(iii) Replacing them with verified Lean numerics
would require a separate multi-year subproject (rigorous interval
arithmetic + verified quadrature + verified Bessel + Taylor
branch-and-bound, ~6000-10000 lines), comparable to the Flyspeck
effort for Kepler, with no upside for the mathematical claim.
(iv) Verification is by rigorous AI-agent self-audit across multiple
independent passes (numerical re-derivation, proof tracing, certifier
re-runs, `#print axioms`), **not** third-party referee review.

### Two headlines: conditional and unconditional

The formalization exports **two** headline theorems, both `lake build`
green with 0 `sorry`:

- **Conditional** — `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
  (with aliases `autoconvolution_ratio_ge_1_292`, `C1a_ge_1292`). Takes
  the analytic admissibility-bundle record `ExtremiserPrimitives f` as a
  named hypothesis. Its dependency closure reaches the Lean-core trio
  plus **two** verifiable-by-computation numerical axioms
  (`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`).
- **Unconditional** (over the admissible class) —
  `Sidon.MultiScale.C1a_ge_1292_unconditional` (the symbol resolves
  through `Sidon.Constructor.Assembly`). It takes **only** raw
  admissibility hypotheses on $f$ —
  `Integrable f volume`, `MemLp f 2 volume`,
  `support f ⊆ Ioo(-1/4, 1/4)`, `∀ x, 0 ≤ f x`, `∫ f = 1` — and
  concludes `autoconvolution_ratio f ≥ 1292/1000`. The entire
  MV Lemma 3.1 bundle is now **constructed**, not assumed: the bundle
  record is produced by `ExtremiserPrimitives.of_admissible`, whose
  proof discharges all four Eqs.(1)--(4) fields from admissibility plus
  the sanctioned numerical axioms (no `ExtremiserPrimitives` hypothesis).
  Its dependency closure reaches the Lean-core trio plus **four**
  verifiable-by-computation numerical axioms: the two above, plus
  `Sidon.MultiScale.min_G_analytic_ge_minGLowerQ`
  ($\min_{[0,1/4]} G \ge 998/1000$, used to discharge the bundle's
  positivity field) and the active-set lattice positivity axiom
  `Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`
  (below). This is MV-parity computer-assisted rigor: the four numerical
  axioms remain external `flint.arb` facts; it is **not** a fully
  kernel-checked numeric proof.

The constructor and its supporting analytic infrastructure live in a new
**`lean/Sidon/Constructor/`** module layer (17 files, 7922 LoC, all
axiom-free except `LatticePositivity` which declares the one new axiom):
`ConvFourier`, `KernelFacts`, `KernelL2`, `YoungConvolution`,
`FieldsEasy`, `FieldsParseval`, `FieldEq4`, `Eq2Split`, `Eq2Period1`,
`MOLemma21`, `PeriodUParseval`, `PoissonSampling`, `PoissonSummable`,
`CauchySchwarzFloor`, `Glue`, `LatticePositivity`, `Assembly`. Building
the bundle from raw admissibility required (all proved axiom-free in
Lean): the $L^1 \cap L^2$ convolution Fourier identity; `MemLp K_ms 2`
via a three-function Hölder/Young estimate
$L^{4/3} \star L^{4/3} \to L^2$; a period-$u$ Poisson sampling lemma
(generalizing mathlib's period-1 / continuous-only version); the
MO 2009 Lemma 2.1 period-$u$ Parseval pairing; period-1 Parseval for the
$f \circ f$ bound; and the Cauchy--Schwarz floor. The effort also fixed
two real bugs: a convention/scaling error in the bundle's `S_cos` term
(the master used a $2u^2$ coefficient where the correct one is $2/u$),
corrected in both the Lean source and the manuscript, and a false
`h_parseval_split` form.

### Active-set lattice positivity (unconditional-headline path)

Driving the unconditional headline all the way to raw admissibility
requires the QP active-set period-$u$ denominators to be *strictly
positive*:

$$\texttt{K\_ms\_fourier\_lattice}(j)
  = \sum_i \lambda_i\, J_0(\pi \delta_i\, j/u)^2 > 0,
  \qquad 1 \le j \le 200,$$

i.e. that the three transcendental zero-sets of
$J_0(\pi\delta_1\cdot), J_0(\pi\delta_2\cdot), J_0(\pi\delta_3\cdot)$ do
not coincide at any active lattice point $j/u$ (so each denominator of
$S_1 = \sum_{j} a_j^2/\widetilde{K_{\rm ms}}(j)$ is well-defined and
nonzero). This residual is resolved in
`Sidon.Constructor.LatticePositivity`:

- The **structural prefix $1 \le j \le 16$** is a Lean *theorem*
  (`pos_of_abs_le_16`, axiom-free): the smallest scale $\delta_3$ keeps
  $|\pi\delta_3 j/u| < 2$ for $j \le \lfloor 2u/(\pi\delta_3)\rfloor =
  \lfloor 1276/(25\pi)\rfloor = 16$, where the alternating-series bound
  $J_0(z) \ge 1-(z/2)^2$ certifies $J_0 > 0$. (Sharp for this clean
  bound: at $j=17$, $|\pi\delta_3\cdot 17/u| \approx 2.093 > 2$.)
- The **tail $17 \le j \le 200$** is a **third verifiable-by-computation
  axiom** `K_ms_fourier_lattice_pos_active` (stated over the full range
  $1\le j\le 200$, with the prefix folded back in as a theorem so the
  axiom's effective content is only the transcendental tail). It is
  *logically decidable* — a finite conjunction of 200 strict
  $J_0$-sum inequalities about specific real numbers — and backed by the
  same `flint.arb` certificate at 256-bit precision as the other two
  (`delsarte_dual/grid_bound_alt_kernel/audit_lattice_positivity.py`,
  which reports $\min_{1\le j\le 200}\widetilde{K_{\rm ms}}(j) \approx
  2.0817\times 10^{-4}$, certified arb lower endpoint, attained at
  $j=147$). An analogue *in role* of MV 2010's Mathematica $J_0$
  citations, but strictly more rigorous. It is **not** derivable from
  `gain_analytic_ge_gainLowerQ`: that axiom records only a rational
  *upper* bound on $S_1$, and Lean's $x/0=0$ convention lets a zero
  denominator *decrease* $S_1$ while still satisfying the bound — so the
  gain axiom cannot force positivity, and the fact is recorded as a
  distinct axiom rather than folded in.

## References

- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), 439-447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), 3191-3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988).
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), 219-235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121).
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), 1281-1292. [arblib.org](https://arblib.org/).
- Watson, G. N. *A Treatise on the Theory of Bessel Functions*, 2nd ed., Cambridge University Press, 1944.
- AlphaEvolve project. *AI-driven optimization of analytic constants arising in extremal combinatorics and additive number theory*, preprint, 2025. [arXiv:2511.02864](https://arxiv.org/abs/2511.02864).
