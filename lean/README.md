# Sidon Autocorrelation: Lean 4 Formalisation

A Lean 4 / Mathlib formalisation of the rigorous lower bound

```
    C_{1a} = inf { ‖f * f‖_∞ / (∫ f)² :
                   f >= 0, supp(f) ⊆ [-1/4, 1/4], 0 < ∫ f < ∞ }
           ≥ 1292/1000  =  1.292,
```

established in *A New Lower Bound for the Supremum of
Autoconvolutions*.  The proof builds a 3-scale arcsine kernel and
applies the Matolcsi–Vinuesa (2010) master inequality.  All numerical
anchors are discharged by a `flint.arb` certifier at 256-bit precision;
the Python implementation lives in
`../delsarte_dual/grid_bound_alt_kernel/`.

The formalisation has **no `sorry`**.  Beyond Lean's three core
logical axioms (`Classical.choice`, `propext`, `Quot.sound`), the
headline theorem's dependency closure reaches **exactly two
verifiable-by-computation user axioms**, both declared in
`Sidon/MultiScale.lean` (see "Axiom inventory" below).  The previously
exported macro axiom `MV_master_inequality_for_extremiser` is now a
Lean *theorem*.

## Build

```bash
lake build                            # all thirteen modules
lake env lean AxiomCheckMV.lean       # per-module axiom inventories
lake env lean AxiomCheckBundleDefs.lean
lake env lean AxiomCheckFourier.lean
lake env lean AxiomCheckTorus.lean
```

`lake build` should report `Build completed successfully` with no
`sorry` warnings.  The four `AxiomCheck*.lean` files print the axiom
closure of their respective module imports; after `lake build`,
`#print axioms Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
reports the headline's axiom dependency closure (three core +
two user).

## Layout

| Path | Lines | Role |
|------|------:|------|
| `Sidon.lean` | 52 | Root re-exporter for the proof chain. |
| `Sidon/Defs.lean` | 55 | Core definitions (`autoconvolution_ratio`, convolutional `f ∘ f`). |
| `Sidon/Bessel.lean` | 958 | Bessel $J_0$ power series, autoconvolution arcsine FT identity, Watson tail bound. |
| `Sidon/FourierAux.lean` | 606 | Schwartz Plancherel, $L^p$ bridge, $L^1$-pairing. |
| `Sidon/TorusParseval.lean` | 785 | Period-$u$ Parseval, lattice Fourier, bilinear pairing. |
| `Sidon/MVLemmas.lean` | 767 | MV Lemma 3.1 Eqs.(1)–(4) + inner-product floor + `mv_eq3_ge` / `mv_eq3_ge_of_eq`. |
| `Sidon/MasterFromLemmas.lean` | 130 | Algebraic assembly Eqs.(1)–(4) ⇒ Eq.(6). |
| `Sidon/BundleDefs.lean` | 597 | `ExtremiserPrimitives` record. |
| `Sidon/BundleEq1.lean` | 347 | Discharge of bundle field `hEq1` (MV Eq.(1)). |
| `Sidon/BundleEq2Schwartz.lean` | 624 | Discharge of bundle field `hEq2` (MV Eq.(2)). |
| `Sidon/BundleEq3Schwartz.lean` | 371 | Discharge of bundle field `hEq3_ge` (MV Eq.(3), inequality form). |
| `Sidon/BundleEq4.lean` | 445 | Discharge of bundle field `hEq4` (MV Eq.(4)). |
| `Sidon/BilinearParseval.lean` | 434 | Bilinear Parseval pairings used by the bundle discharges. |
| `Sidon/MultiScale.lean` | 1648 | Conditional headline + 3 verifiable-by-computation axioms (K2, gain, min_G) + admissibility bundle. |
| `AxiomCheck{BundleDefs,Fourier,MV,Torus}.lean` | — | Per-module axiom inventories. |
| `lakefile.lean`, `lake-manifest.json`, `lean-toolchain` | — | Lake build configuration, Mathlib lock, pinned toolchain. |

The twelve auxiliary core modules are axiom-free; the 1648-line
`Sidon.MultiScale` houses three of the four verifiable-by-computation
axioms (K2, gain, min_G).  Beyond these 13 core modules, the
`Sidon/Constructor/` layer (17 modules, 7915 LoC, all axiom-free except
`LatticePositivity`, which declares the fourth axiom
`K_ms_fourier_lattice_pos_active`) *constructs* the admissibility bundle
and proves the **unconditional** headline (below): 30 modules /
~15.6 kLoC in total.
Earlier exploratory modules (legacy monolithic proof, Algorithm /
CoarseCascade drafts, single-scale and 2-scale Cascade variants,
alternative-kernel directions) have been moved to `../archive/lean/`.

## Headline theorem

```lean
theorem Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
    (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤)
    (P : ExtremiserPrimitives f) :
    autoconvolution_ratio f ≥ (1292 : ℝ) / 1000
```

Equivalent restatements `autoconvolution_ratio_ge_1_292` (decimal form)
and `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`) are exported
from the same namespace.

The headline above is **conditional**: it takes an `ExtremiserPrimitives f`
hypothesis record (below).  The **unconditional** headline
`Sidon.MultiScale.C1a_ge_1292_unconditional` (in
`Sidon/Constructor/Assembly.lean`) takes ONLY admissibility —
`Integrable f`, `MemLp f 2`, `Function.support f ⊆ Set.Ioo (-(1/4)) (1/4)`,
`∀ x, 0 ≤ f x`, `∫ f = 1` — and *constructs* the bundle via
`ExtremiserPrimitives.of_admissible`, carrying no analytic hypothesis.
It reaches **four** verifiable-by-computation axioms (the two below plus
`min_G_analytic_ge_minGLowerQ` and
`Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`).

## Axiom inventory

Beyond Lean's three core axioms (`Classical.choice`, `propext`,
`Quot.sound`), the **conditional** headline reaches exactly **two user
axioms** (declared in `Sidon/MultiScale.lean`); the **unconditional**
headline additionally reaches `min_G_analytic_ge_minGLowerQ`
(`min_{[0,1/4]} G ≥ 998/1000`, in `Sidon/MultiScale.lean`) and
`K_ms_fourier_lattice_pos_active` (`∀ j ∈ [1,200], K̃_ms(j) > 0`, in
`Sidon/Constructor/LatticePositivity.lean`) — four in total.  All four
are logically decidable, `flint.arb`-backed at 256-bit, and
independently mpmath-corroborated.  The two common to both headlines:

| Axiom | Statement |
|---|---|
| `K2_analytic_le_K2UpperQ`        | $\int K_{\rm ms}^2\,dx \le \texttt{K2UpperQ} = 47897/10000$.  Certifier interval $[4.788823, 4.788906]$; slack margin $\approx 7.9 \times 10^{-4}$. |
| `gain_analytic_ge_gainLowerQ`    | $(4/u_{\rm real}) \cdot \texttt{min\_G\_analytic}^2 / \texttt{S\_1\_analytic} \ge \texttt{gainLowerQ} = 20925/100000$.  Certifier value $\ge 0.21009214$; slack margin $\approx 8.4 \times 10^{-4}$. |

Both axioms bound *concrete defined* analytic functionals (not opaque
symbols): `K2_analytic`, `min_G_analytic`, `S_1_analytic`, and
`gain_analytic` are noncomputable definitions over the 200 embedded
QP coefficients `qpNumerators` (denominator $10^8$) and the Bessel-form
period-$u$ Fourier coefficient `Ktilde_ms` built from
`Sidon.Bessel.besselJ0`.  Each axiom is logically *decidable* — a
statement about specific integers that any sufficient implementation
of interval arithmetic + the Bessel power series can decide; both are
discharged externally by `flint.arb` at 256-bit precision in
`../delsarte_dual/grid_bound_alt_kernel/`.  The arrangement is the
same convention used by the FlySpeck formalisation of Kepler's
conjecture.

The previously exported single-axiom packaging
`MV_master_inequality_for_extremiser` is now a Lean **theorem**,
assembled from the bundle hypothesis (below) plus the two
verifiable-by-computation axioms.

## `ExtremiserPrimitives` bundle

The headline takes a final hypothesis: a record
`ExtremiserPrimitives f` packaging the four MV Lemma 3.1 outputs at
$(f, K_{\rm ms})$:

```lean
structure ExtremiserPrimitives (f : ℝ → ℝ) where
  m_G S_G S_cos LHS1 LHS2 : ℝ
  K2_ge_1  : 1 ≤ K2_analytic
  gain_eq  : gain_analytic = 2 * m_G^2 / S_G
  R_ge_1   : 1 ≤ autoconvolution_ratio f
  S_G_pos  : 0 < S_G
  hEq1     : LHS1 ≤ autoconvolution_ratio f
  hEq2     : LHS2 ≤ 1 + Real.sqrt (autoconvolution_ratio f - 1)
                    * Real.sqrt (K2_analytic - 1)
  hEq3_ge  : 2/uQ + 2 * uQ^2 * S_cos ≤ LHS1 + LHS2
  hEq4     : uQ^2 * S_cos ≥ m_G^2 / S_G
```

The four fields `hEq1`, `hEq2`, `hEq3_ge`, `hEq4` are Lean restatements
of MV Lemma 3.1 Eqs.(1)–(4) for the specific $(f, K_{\rm ms})$ pair.
MO~2009 Lemmas~3.1–3.4 / MV~2010 Lemma~3.1 apply to $K_{\rm ms}$
directly — the only hypotheses they require are (K1)–(K4) (pdf,
support $\subseteq [-\delta_1, \delta_1]$, $\widetilde{K_{\rm ms}}(j)
\ge 0$, $K_{\rm ms} \in L^2$), all preserved under convex combination.
The paper discharges `hEq1`, `hEq2`, `hEq4` by direct citation; the
inequality form `hEq3_ge` (which replaced the earlier equality `hEq3`)
is genuinely Lean-derivable from finite-`J` Parseval + Bochner
positivity alone (`Sidon.MVLemmas.mv_eq3_ge` and `mv_eq3_ge_of_eq`).
In the **conditional** headline the bundle is a hypothesis (discharged
in the paper by direct citation, exactly as MV~2010 did at single
arcsine).  It is now also **constructed unconditionally** in
`Sidon/Constructor/`: `ExtremiserPrimitives.of_admissible` builds the
record from raw admissibility, yielding `C1a_ge_1292_unconditional`.
The period-$u$ Parseval pairing (MO~2009 Lemma~2.1) and the
$L^1 \cap L^2$ machinery — formerly only-a-hypothesis because mathlib
lacked them — are now proven axiom-free
(`Constructor.PoissonSampling`, `MOLemma21`, `YoungConvolution`,
`Eq2Period1`, `CauchySchwarzFloor`).

**Rigor relative to Matolcsi–Vinuesa.**  This establishes
$C_{1a} \ge 1.292$ to at least the rigor of MV~2010 (the accepted proof
of $C_{1a} \ge 1.2748$): the analytic content is proven — by citation
to MO~2009 / MV~2010 applied to $K_{\rm ms}$ as MV did, and now also
mechanized axiom-free in Lean — and the numerical content is the same
*kind* of computer-checked fact as MV's Mathematica citations, but
backed by `flint.arb` interval arithmetic at 256-bit (rigorous, not
heuristic) and mpmath-corroborated.  The four numerical axioms remain
(computer-assisted rigor, not a fully kernel-checked numeric proof);
verification is by rigorous self-audit, not third-party referee review.

## Slack-soundness theorems

Five Lean **theorems** record that the rational slacks are on the
correct side of the certifier-reported decimals (paper Lemmas 4.1–4.5):

| Theorem | Content |
|---|---|
| `K_two_upper_bound`  | `K2UpperQ ≥ 4788906/1000000` |
| `k_one_lower_bound`  | `K1LowerQ ≤ 92124658/100000000` |
| `S_one_upper_bound`  | `S1UpperQ ≥ 2984091/100000` |
| `min_G_lower_bound`  | `minGLowerQ ≤ 9999798/10000000` |
| `gain_lower_bound`   | `gainLowerQ ≤ 21009214/100000000` |

## Construction

3-scale arcsine kernel:

```
    K(x)  =  λ₁ K_arc(δ₁)(x) + λ₂ K_arc(δ₂)(x) + λ₃ K_arc(δ₃)(x),
    K̂(ξ)  =  λ₁ J₀(πδ₁ξ)² + λ₂ J₀(πδ₂ξ)² + λ₃ J₀(πδ₃ξ)²,
```

at the rational anchors

```
    δ₁ = 138/1000,    λ₁ = 85/100,
    δ₂ =  55/1000,    λ₂ = 10/100,
    δ₃ =  25/1000,    λ₃ =  5/100,
    u  = 638/1000  (= 1/2 + δ₁),
```

with a 200-coefficient cosine `G` re-optimised at this kernel.  Bochner
admissibility of `K̂` is automatic: each `J₀(πδᵢξ)²` is the square of a
real Bessel function, and a convex combination preserves
positive-semi-definiteness.

## References

* Matolcsi, M., Vinuesa, C.  *Improved bounds on the supremum of
  autoconvolutions.*  J. Math. Anal. Appl. **372** (2010), 439-447,
  arXiv:0907.1379.
* Cloninger, A., Steinerberger, S.  *On suprema of autoconvolutions
  with an application to Sidon sets.*  Proc. Amer. Math. Soc.
  **145** (2017), 3191-3200, arXiv:1403.7988.
* Martin, G., O'Bryant, K.  *The symmetric subset problem in
  continuous Ramsey theory.*  Experiment. Math. 16 (2007), 145-165,
  arXiv:0807.5121.
* Cohn, H., Elkies, N.  *New upper bounds on sphere packings I.*
  Annals of Mathematics (2) 157 (2003), 689–714.
