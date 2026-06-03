# Verification Protocol

Public audit specification for the **Piterbarg--Bajaj--Vincent Bound**,
the rigorous lower bound
$C_{1a} \ge 1292/1000 = 1.292$ (three-scale arcsine convex combination
with a $200$-cosine reoptimised multiplier $G$ in the Matolcsi--Vinuesa
master inequality). The fourteen checks below are independent and
local: each can be carried out without reading the others, and each
returns a binary verdict (CONFIRM / FLAG) on a single claim.

The bound is established to at least the rigour of the accepted
Matolcsi--Vinuesa 2010 proof of $C_{1a} \ge 1.2748$: MV combine formal
analytic lemmas, Mathematica numerics, and algebra; this work matches or
strengthens each layer (analytic content cited to MO 2009 / MV 2010 at
the admissible kernel $K_{\rm ms}$ and additionally mechanised
axiom-free in Lean for the unconditional headline; numerics of the same
kind as MV's Mathematica citations but arb-backed at 256-bit and
mpmath-corroborated; assembly exact-rational with positive margin
$307/3190000$). Honest caveat: the numerical axioms (two for the
conditional headline, four for the unconditional) remain
computer-assisted, and these checks are a rigorous AI-agent self-audit,
not third-party referee review.

Authoritative artefacts:

| Artefact | Path |
|----------|------|
| Paper | [`../lower_bound_proof.pdf`](../lower_bound_proof.pdf) / [`../lower_bound_proof.tex`](../lower_bound_proof.tex) |
| Lean formalisation | [`../lean/Sidon/MultiScale.lean`](../lean/Sidon/MultiScale.lean) |
| Certifier driver | [`../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py`](../delsarte_dual/grid_bound_alt_kernel/bisect_alt_kernel.py) |
| Reference anchors | [`../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`](../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json) |
| Independent verifier | [`../delsarte_dual/grid_bound/certify.py`](../delsarte_dual/grid_bound/certify.py) |

## Authoritative numerical anchors

These are the values quoted by the certifier and consumed by Lean. All
quantities are computed in `flint.arb` interval arithmetic at $256$-bit
precision; the Lean verifiable-by-computation axioms quote rational
slack relaxations of the endpoints. (Recall: "verifiable-by-computation
axiom" denotes a logically decidable inequality about specific real
numbers, backed by a reproducible algorithm, currently un-formalised in
Lean only because mathlib lacks a Bessel interval-arithmetic library.)

### Kernel parameters (exact rationals)

| Symbol | Value | Role |
|--------|-------|------|
| $\delta_1$ | $138/1000$ | first arcsine half-width |
| $\delta_2$ | $55/1000$  | second arcsine half-width |
| $\delta_3$ | $25/1000$  | third arcsine half-width |
| $\lambda_1$ | $85/100$  | weight on $\delta_1$ component |
| $\lambda_2$ | $10/100$  | weight on $\delta_2$ component |
| $\lambda_3$ | $5/100$   | weight on $\delta_3$ component |
| $u$ | $638/1000 = 1/2 + \delta_1$ | period of the Fourier reduction |
| $N$ | $200$ | number of cosine modes in $G$ |
| $M_{\text{target}}$ | $1292/1000$ | strict-failure witness $M$ |

### Rigorous anchors (from the arb certifier)

| Symbol | Reported (arb) | Lean rational bound | Slack |
|--------|----------------|---------------------|-------|
| $k_1$ (mid)        | $0.92124658993\ldots$ (rad $\le 7 \times 10^{-77}$) | $\ge 9212/10000$  | $4.6 \times 10^{-5}$ |
| $K_2$ (upper)      | $4.7889051816$ | $\le 47897/10000$ | $7.9 \times 10^{-4}$ |
| $S_1$ (upper)      | $29.8409064555132666$ | $\le 29841/1000$  | $9.3 \times 10^{-5}$ |
| $\min G$ (lower)   | $0.9999798743824747$ | $\ge 998/1000$   | $1.9 \times 10^{-3}$ |
| gain $a$ (lower)   | $0.2100921474866837$ | $\ge 20925/100000$ | $8.4 \times 10^{-4}$ |
| $M_{\text{cert}}$ (production) | $\ge 1.2923242187500000$ ($66167/51200$) | $\ge 1292/1000$ | $3.2 \times 10^{-4}$ |
| $M_{\text{cert}}$ (slack-anchor) | $\ge 1.2921564960222152$ | $\ge 1292/1000$ | $2.2 \times 10^{-4}$ |

Direction sanity (must all be strict): the Lean rational bound is on the
correct side of the arb endpoint for each anchor.

## Verification checklist

Each task below states a question, the files involved, the test that
discharges it, and the failure mode that should be reported. The
verifier is expected to report a verdict (CONFIRM or FLAG) per item; a
single FLAG that is mathematically substantive blocks acceptance.

**Audit transport scope.** The original V1--V14 audit was run on the
two-scale 1.28984 precursor. Framework-level checks (V1, V3, V5, V6,
V10, V11, V13) transport unchanged to the three-scale 1.292 headline
because they concern the pipeline structure, the universality of MV
Lemma 3.1, and the build/hash reproducibility, none of which depend on
the specific weights or scales. Instance-specific anchors (V2, V4, V7,
V8, V9, V12, V14) were re-verified at the 3-scale 1.292 point on
2026-05-20; the parallel re-audit results are documented in
[`audit_3scale_reverification.md`](audit_3scale_reverification.md).

### 1. Pipeline sanity vs. single-scale baseline

When the certifier is run at the single-arcsine anchors used by
Matolcsi--Vinuesa ($\delta = 138/1000$, $\lambda = 1$, $N = 119$), it
must reproduce the published baseline $M_{\text{cert}} \approx 1.27484$.
This isolates a regression in the multi-scale code path. CONFIRM
requires agreement to four decimal digits; FLAG requires disagreement
$> 10^{-3}$ or infeasibility.

### 2. Independent recomputation of $K_2$

At `mpmath.mp.dps = 50`, compute the six Bessel cross-integrals
$C_{ij} = \int_{\mathbb{R}} J_0(\pi \delta_i \xi)^2 J_0(\pi \delta_j
\xi)^2 d\xi$ for $i, j \in \{1,2,3\}$, sum
$K_2 = \sum_{i,j} \lambda_i \lambda_j C_{ij}$, and add the explicit tail
bound past $T = 10^5$. CONFIRM requires the mpmath value to lie inside
the arb enclosure $[4.788823, 4.788906]$.

### 3. Watson tail bound

The certifier uses $|J_0(z)|^2 \le 2/(\pi z)$ for $z \ge 1$ to bound the
$K_2$ tail past $\xi = T = 10^5$, giving $\text{tail} \le (8/\pi^4)
\cdot C^2 / T$ with $C = \sum_i \lambda_i / \delta_i$ (each of the two
Bessel factors contributes $|J_0(\pi\delta_i\xi)|^2 \le 2/(\pi^2\delta_i\xi)$,
so the product carries $\pi^4$, not $\pi^2$). Confirm Watson
1944, §7.21 (cf. NIST DLMF 10.14.1); verify that the smallest argument
$\pi \delta_3 T \approx 7854$ exceeds the validity threshold; verify the
arithmetic.

### 4. Rescaling-invariance of the gain at $\min G < 1$

The reoptimised multiplier has $\min G \approx 0.99997987 < 1$ (slack
rational anchor $\ge 998/1000$). The MV gain
$a = (4/u) \cdot \min G^2 / S_1$ is invariant under $G \mapsto cG$, and
the master inequality only enters through $a$. Verify the invariance
algebraically and check that the un-renormalised
$(0.99997987^2 / 29.841)$ equals the renormalised $(1 / S_1')$ value to
six digits.

### 5. Bochner positivity at the QP frequencies

For every $j \in \{1, \ldots, 200\}$ the periodic Fourier coefficient
$\widetilde K_{\rm ms}(j) = \sum_i \lambda_i J_0(\pi \delta_i j / u)^2$
must be strictly positive (so that $1 / \widetilde K(j)$ in $S_1$ is
finite). Compute $\min_j \widetilde K_{\rm ms}(j)$ at $\text{dps} = 30$
and verify $\ge 2 \times 10^{-4}$ (the actual minimum lies near
$2.08 \times 10^{-4}$; see `docs/proof_outline.md`); compare with the
single-scale minimum $\sim 10^{-6}$ to confirm the rescue mechanism
that underlies the $S_1$ drop.

### 6. Master-inequality universality

Matolcsi--Vinuesa derived the master inequality for the single-arcsine
kernel. Re-read MV (2010), Lemma 3.1, line by line and confirm that
every step (Cauchy--Schwarz, Plancherel, Bochner positivity of $\widehat
K$, Beurling--Selberg majorant of $\mathbf{1}_{[0, 1/(2M)]}$) uses only
the abstract admissibility hypotheses (nonnegativity, support, unit
mass, periodic Bochner positivity). Flag any step that secretly uses
unimodality, smoothness, or log-concavity of $K$.

### 7. Sufficiency of the $z_1$-free form at $M = 1.292$

The Lean theorem `MV_master_inequality_for_extremiser` uses the
$z_1$-free form
$M + 1 + \sqrt{(M - 1)(K_2 - 1)} \ge 2/u + a$. Plug in the rational
anchors and verify that at $M = 1.292$ the left-hand side equals
$66879/20000 = 3.34395$, the right-hand side equals
$\ge 4267003/1276000 \approx 3.344046$, and the difference is at least
$9.6 \times 10^{-5}$. Confirm the implication direction: LHS $<$ RHS at
$M$ forces $M \ge 1.292$ for the extremiser.

### 8. Taylor branch-and-bound rigor for $\min G$

The certifier partitions $[0, 1/4]$ into $n = 32768$ closed cells of
half-width $r = 1/(8n) \approx 3.8 \times 10^{-6}$ and, on each cell,
forms the second-order Taylor enclosure
$G(c) + G'(c)[-r, r] + G''(\text{cell})[-r^2/2, r^2/2]$ around the
centre $c$, with $G''$ evaluated as an arb interval over the entire
cell.  The certified lower bound is the minimum of the per-cell
enclosures' lower endpoints; this returns $\min G \ge 0.99997987$.
Re-derive any one cell's enclosure in arb, confirm the implementation
is in `grid_bound/G_min.py:min_G_lower_bound`, and verify the cell
evaluation runs in arb (not float).

### 9. QP for the reoptimised $G$

Independently solve the QP for the $200$ cosine coefficients of $G$ at
$w_j = \widetilde K_{\rm ms}(j)$ (CLARABEL or MOSEK), round coefficients
to $\mathbb{Q}$ with denominator $10^{8}$, recompute
$S_1 = \sum_j a_j^2 / w_j$, and verify agreement to five digits with
the certificate's $S_1 \le 29.841$.

### 10. Lean module sanity

`lake build Sidon` returns exit code zero with no `sorry` warnings,
across all **30 modules (~15.6 kLoC)**: the 13 core modules under
`lean/Sidon/` (`Defs`, `Bessel`,
`FourierAux`, `TorusParseval`, `MVLemmas`, `MasterFromLemmas`,
`BundleDefs`, `BundleEq1`, `BundleEq2Schwartz`, `BundleEq3Schwartz`,
`BundleEq4`, `BilinearParseval`, `MultiScale`; 7655 LoC) and the 17
modules under `lean/Sidon/Constructor/` (7922 LoC), on top
of `mathlib` pinned to `v4.29.1` / commit
`5e932f97dd25535344f80f9dd8da3aab83df0fe6`. The formalisation exports
**two headlines** with distinct axiom budgets:

- *Conditional* `autoconvolution_ratio_ge_1292_1000` (assumes an
  `ExtremiserPrimitives f` bundle): exactly **two verifiable-by-computation
  axioms**, both declared in `Sidon.MultiScale`.
- *Unconditional* `C1a_ge_1292_unconditional` (in
  `Sidon.Constructor.Assembly`; constructs the bundle from raw
  admissibility via `ExtremiserPrimitives.of_admissible`): **four
  verifiable-by-computation axioms** — the two above plus
  `min_G_analytic_ge_minGLowerQ` (`Sidon.MultiScale`) and
  `K_ms_fourier_lattice_pos_active`
  (`Sidon.Constructor.LatticePositivity`).

All modules are axiom-free except `Sidon.MultiScale` and
`Sidon.Constructor.LatticePositivity`, which together declare the four
axioms in the headlines' dependency closures:

- `K2_analytic_le_K2UpperQ` (both headlines) -- the analytic functional
  $K_2(K_{\rm ms}) := \int K_{\rm ms}(x)^2\,\mathrm{d}\mathrm{volume}$
  is at most $\texttt{K2UpperQ} = 47897/10000$; verified by `flint.arb`
  at 256-bit precision (analogue of MV 2010's Mathematica citation of
  $K_2$).
- `gain_analytic_ge_gainLowerQ` (both headlines) -- the *concrete defined* analytic
  gain $\texttt{gain\_analytic} := (4 / u_{\rm real}) \cdot
  \texttt{min\_G\_analytic}^2 / \texttt{S\_1\_analytic}$ is at least
  $\texttt{gainLowerQ} = 20925/100000$. After **Option B** (2026-05-20)
  the RHS is a non-opaque `noncomputable def` built from the 200
  embedded QP coefficients (`Sidon.MultiScale.qpNumerators : List ℤ`,
  common denominator $10^8$, length verified by `native_decide`) and
  the Bessel-form period-$u$ Fourier coefficient
  `Ktilde_ms j := Σᵢ λᵢ · besselJ0(πjδᵢ/u)²` (using
  `Sidon.Bessel.besselJ0`). Certifier-coupled in arb (analogue of MV
  2010's Mathematica citation of $a$). Cross-binding of the 200
  coefficients between the JSON certificate and the Lean embedding
  is checked by the sibling script `audit_qp_coeffs.py`.
- `min_G_analytic_ge_minGLowerQ` (unconditional headline only) --
  $\texttt{min\_G\_analytic} \ge \texttt{minGLowerQ} = 998/1000$
  ($\min_{[0,1/4]} G \ge 0.998$; 32768-cell Taylor B&B).
- `K_ms_fourier_lattice_pos_active` (unconditional headline only) --
  $\widetilde{K_{\rm ms}}(j) > 0$ for every $j \in \{1,\dots,200\}$
  (certifier minimum $\ge 2.08 \times 10^{-4}$).

The previous macro axiom `MV_master_inequality_for_extremiser` is
now a Lean *theorem*, derived from the two verifiable-by-computation
axioms plus the analytic admissibility-bundle record
`ExtremiserPrimitives f` that the conditional headline takes as its fifth
argument. The bundle fields are Lean restatements of MO~2009 /
MV~2010 Lemma 3.1 outputs (Eqs.(1)--(4)) at the pair $(f, K_{\rm ms})$,
discharged in the paper by direct citation (those lemmas apply to
$K_{\rm ms}$ directly). For the conditional headline they are retained
as Lean hypothesis fields only because the corresponding mathlib bridge
had not yet been packaged into a one-call constructor; the unconditional
headline now supplies exactly that constructor (`of_admissible`).

The quadratic inversion `master_inequality_M_lower`, the
slack-monotonicity lift `MV_master_via_slack_monotonicity`, the
full chain `MV_master_inequality_from_MV_lemmas`, the master
inequality theorem `MV_master_inequality_for_extremiser`, and the
five slack-soundness statements (`K_two_upper_bound`,
`k_one_lower_bound`, `S_one_upper_bound`, `min_G_lower_bound`,
`gain_lower_bound`) are Lean *theorems*. The `#print axioms`
listing for `autoconvolution_ratio_ge_1292_1000` reports the two
verifiable-by-computation user axioms plus Lean's three core logical
axioms (`propext`, `Classical.choice`, `Quot.sound`); the listing for
`C1a_ge_1292_unconditional` reports the four verifiable-by-computation
axioms plus the same three core logical axioms.

### 11. Reproducibility and certificate hash

Re-run
`python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel` and
verify every certificate field (anchors, bisection history, cell
counts, wall time) matches the reference. The emitted JSON has
`{"sha256_of_body": …, "body": {…}}`; the digest must equal the stored
hash.

### 12. Cross-comparison with the prior baseline

The chain $1.27481 < 1.28 < 1.292$ states that this proof strictly
improves on the rigorous Matolcsi--Vinuesa~(2010) bound and on the
published Cloninger--Steinerberger~(2017) bound $1.28$.  (The
unpublished $1.2802$ listed in Tao's `optimizationproblems` `1a.md` is
attributed there to Xie~(2026, Grok), **not** to
Cloninger--Steinerberger, and is not part of this chain.)  Each
rational inequality is
`norm_num`-decidable; the hypothesis sets of the three results agree
(nonnegativity, support in $(-1/4, 1/4)$, positive integral, finite
$\|f * f\|_\infty$).

### 13. Cross-Bessel cross-checks

Compute each of $I(\delta_1, \delta_2)$, $I(\delta_1, \delta_3)$,
$I(\delta_2, \delta_3)$ independently at $\text{dps} = 50$ via
`mpmath.quadosc` or Bessel-zero splits with explicit tail past
$T = 10^5$, and recombine to obtain $K_2$. Compare with the arb cert
$K_2 \in [4.788823, 4.788906]$; agreement to five digits is CONFIRM.
Each cross-integral must be positive and decay consistent with the
Watson asymptotic; the (incorrect) conjecture $I(a,b) =
0.5747 / \max(a, b)$ should appear nowhere.

### 14. Form discrepancy: refined vs. $z_1$-free

The Python certifier uses the refined MV form
$\Phi(M, y) = M + 1 + 2 y k_1 + \sqrt{(M-1-2y^2)(K_2-1-2k_1^2)}
- (2/u + a)$ (cell-search over $y \in [0, \mu(M)]$).  The Lean
theorem `MV_master_inequality_for_extremiser` uses only the (weaker)
$z_1$-free form obtained from the refined form by Cauchy--Schwarz on
the two square-root pairs.  The $z_1$-free form alone must suffice
at the rational target $M = 1292/1000$ (see check 7); the refined
form is used only inside the Python bisection to certify the
(sharper) anchor $M_{\text{cert}} = 66167/51200 \approx 1.29232422$.

## Acceptance

The bound is verified iff every check returns CONFIRM. A FLAG that is
mathematically substantive (not numerical noise) blocks acceptance and
must be triaged before publication.
