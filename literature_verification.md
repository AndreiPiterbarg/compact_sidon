# An autocorrelation constant related to Sidon sets — verification of the lower bound $C_{1a} \ge 1.292$

## Description of constant

$C_{1a}$ is the largest constant for which one has

$$
\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\,dx \;\geq\; C_{1a} \left(\int_{-1/4}^{1/4} f(x)\,dx\right)^{2}
$$

for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$.
<a href="#1a-def">[1a-def]</a>

Equivalently, with $\mathcal{F}$ the family of non-negative $f \in L^{1}(\mathbb{R})$ supported in $(-1/4,1/4)$ with positive integral, $C_{1a} = \inf_{f \in \mathcal{F}} R(f)$, where $R(f) = \lVert f\ast f \rVert_{\infty} / (\int f)^{2}$. Throughout, $M$ denotes $R(f)$; a lower bound asserts $M \ge c$ for every $f \in \mathcal{F}$.
<a href="#PBV-def">[PBV-def]</a>

## Known upper bounds

Not affected by this work; the canonical page tracks them. Current best: $\approx 1.502862$ (AI-assisted, 2026). The bound proved here therefore leaves the gap $1.292 \le C_{1a} \le 1.502862$.
<a href="#1a-page">[1a-page]</a>

## Known lower bounds

Rows one to six are reproduced from the canonical page
<a href="#1a-lb">[1a-lb]</a>; the last row is established here.

| Bound | Reference | Comments |
| ----- | --------- | -------- |
| $1$ | Trivial | $f\ast f$ has integral $(\int f)^2$ over an interval of length $1$, so its maximum is at least $(\int f)^2$. |
| $1.182778$ | [[MO2004](#MO2004)] | |
| $1.262$ | [[MO2009](#MO2009)] | |
| $1.2748$ | [[MV2009](#MV2009)] | Best prior rigorous analytic bound; stated as $1.27481$ in the preprint. |
| $1.28$ | [[CS2017](#CS2017)] | Best prior *published* lower bound (discrete branch-and-prune search). |
| $1.2802$ | [[XX2026](#XX2026)] | Unpublished, AI-assisted (Grok), unaudited; figure not independently reproduced. The canonical page attributes this to Xie (2026), *not* to Cloninger–Steinerberger. |
| $1.292$ | [[PBV2026](#PBV2026)] | This work; $+0.012$ over the previous best *published* lower bound $1.28$ of [[CS2017](#CS2017)], and $+0.0118$ over the unaudited Xie 2026 figure $1.2802$ ([[XX2026](#XX2026)]) if the latter is sound. Rigorous *at the same standard as MV 2010*: the Fourier-analytic identities invoked are MV 2010 Lemma 3.3 and MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3, applied to a three-scale arcsine kernel; the reduction from $\mathcal{F}$ to square-integrable $f$ is by Schinzel–Schmidt 2002 Theorem 1 <a href="#SS2002">[SS2002]</a>, exactly as in MV 2010 §2. Machine-checked in Lean for the algebraic assembly, master inequality, quadratic inversion, and rational closing; the cited MO 2009 / MV 2010 / SS2002 identities are encoded as Lean hypothesis fields and discharged in the paper by direct citation, applied to the admissible kernel $K_{\mathrm{ms}}$. Two numerical inputs ($K_2$ and the gain $a$) are certified in `flint.arb` at 256-bit precision. |

## How the bound is established

**Reduction to square-integrable $f$.** The infimum of $R$ over
$\mathcal{F}$ is unchanged if one restricts to nonnegative step
functions in $\mathcal{F}$, by Schinzel–Schmidt 2002 Theorem 1
<a href="#SS2002">[SS2002]</a>. Nonnegative step functions in
$\mathcal{F}$ are bounded with compact support, hence in
$L^{1}\cap L^{2}$, so $R$ is well-defined for them and the reduction
gives $\inf_{\mathcal{F}} R = \inf_{\mathcal{F}\cap L^{2}} R$. This
is the reduction used verbatim in MV 2010 §2 to justify their
square-integrability assumption:
*"the value of $S$ does not change if one considers nonnegative step
functions in $\mathcal{F}$ only. This is proved in Theorem 1 in [4].
Therefore the reader may assume that $f$ is square integrable
whenever this is needed."* <a href="#MV-reduction">[MV-reduction]</a>
The ratio $R(f) = \lVert f\ast f\rVert_{\infty}/(\int f)^{2}$ is invariant under positive scaling $f \mapsto c f$ ($c>0$), so we may further assume $\int f = 1$; with this normalisation $f$ is a (square-integrable) pdf on $(-1/4, 1/4)$, which is the exact hypothesis form of MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3 and MV 2010 Lemma 3.1. The Lean headline takes only the unnormalised hypothesis $\int f > 0$ since the conclusion `autoconvolution_ratio f ≥ 1292/1000` is itself scale-invariant.

The Lean theorem `autoconvolution_ratio_ge_1292_1000` is stated for
any admissible $f \in L^{1}$ with the standard support/positivity
hypotheses and the convolution-boundedness hypothesis
`h_conv_fin`, taking an analytic-primitives record
`ExtremiserPrimitives f` whose fields are Lean restatements of the
outputs of MO 2009 Lemmas 3.1–3.4 / MV 2010 Lemma 3.1 evaluated at
the specific pair $(f, K_{\mathrm{ms}})$. The hypotheses of those
MO/MV lemmas are satisfied by $K_{\mathrm{ms}}$ directly: it is a
probability density (a convex combination of arcsine-autoconvolution
pdfs), supported in $[-\delta_1,\delta_1]=[-0.138,0.138]$, with
nonnegative periodic Fourier coefficients
$\widetilde{K_{\mathrm{ms}}}(j)=\sum_i\lambda_i J_0(\pi j\delta_i/u)^2\ge 0$,
and square-integrable ($K_2\le 4.7897$); the period
$u=1/2+\delta_1$ meets MO's $1/2+\delta\le u$ condition with equality.
The bundle fields `hEq1`–`hEq4` are Lean restatements of MO 2009
Lemmas 3.1–3.4 / MV 2010 Lemma 3.1 applied at $(f, K_{\mathrm{ms}})$.
The paper discharges them by direct citation to MO/MV, exactly as
MV 2010 discharged its single-arcsine applications by citation to
MO 2009. The Lean theorem retains these outputs as named hypothesis
fields rather than deriving them inside Lean — a paper-side
citation-discharge, not a missing proof. This is the standard
convention for cited classical analysis in computer-assisted
real-number proofs (Flyspeck, PFR, Cohn–Elkies sphere-packing). Two
distinct things must be kept separate here: (a) the
citation-discharge of MO/MV at $K_{\mathrm{ms}}$ is *mathematical
content*, dispatched in exactly the same way MV 2010 dispatched its
own applications of MO 2009; (b) the absence of a packaged mathlib
API that would let the bundle fields be derived in Lean without a
named hypothesis record is *engineering*, but irrelevant to the
validity of (a). The building blocks for (b) live in
`Sidon.TorusParseval` and `Sidon.FourierAux`; their packaging into a
one-call $L^{1}\cap L^{2}$ Plancherel + period-$u$ Parseval
constructor in the form `Sidon.MVLemmas` consumes is a mathlib-side
QoL item, not a logical gap in the proof. The SS2002 reduction step
itself is *not* formalised in Lean — it is invoked exactly as in MV
2010 §2, by citation to <a href="#SS2002">[SS2002]</a>.

**The dual framework.** A lower bound must hold for every admissible
$f$ simultaneously, so it cannot come from a single example. Following
Matolcsi–Vinuesa, one fixes an auxiliary pair: a *Bochner-admissible*
kernel $K$ — non-negative, even, with
$\mathrm{supp}(K) \subseteq [-\delta,\delta]$, $\int K = 1$, and
non-negative periodic Fourier coefficients
$\widetilde{K}(j) := \widehat{K}(j/u) \ge 0$ — and a non-negative
cosine multiplier $G$. Because $f$ is supported in $(-1/4,1/4)$, $f\ast f$
is supported in $(-1/2,1/2)$; taking period $u = 1/2+\delta$ converts
the constraint into a trigonometric inequality, and the pair $(K,G)$
produces, by duality, a single quadratic inequality satisfied by every
admissible $f$. Any valid pair yields some bound; a well-chosen pair
yields a strong one <a href="#PBV-master">[PBV-master]</a>.

**The master inequality.** Apply MO 2009 Lemmas 2.1–3.3 / MV 2010 Lemma 3.1 to the pair $(f, K_{\mathrm{ms}})$, where $f$ is a square-integrable pdf supported in $(-1/4, 1/4)$ (cf. "Applicability to $K_{\mathrm{ms}}$" below) and $G$ is the cosine multiplier described in the next paragraph. Assembling the period-$u$ torus split (Eq.(3); MO Lemma 3.3), the constant-plus-tail Parseval split for $\int(f\circ f)K$ (proof of MO Lemma 3.2), the lattice $F$-bound $\sum_j \lvert\widehat f(j)\rvert^{4}\le \lVert f\ast f\rVert_{\infty}$ (also from the proof of MO Lemma 3.2, using $\lVert f\ast f\rVert_1 = 1$), the multiplier floor (the "Eq.(4) floor" — see below), and one Cauchy–Schwarz step on the QP weights yields MV 2010 Eq.(10) in its general form: for every admissible $f$,

$$\tfrac{2}{u} + a \;\le\; M + 1 + 2 z_1^{2} k_1 + \sqrt{(M - 1 - 2 z_1^{4})(K_2 - 1 - 2 k_1^{2})},$$

with $M = R(f)$, $z_1 = \lvert\widehat f(1)\rvert$, $K_2 = \lVert K_{\mathrm{ms}}\rVert_2^{2}$, $k_1 = \widehat{K_{\mathrm{ms}}}(1)$, $u = 1/2 + \delta_1$, and $a$ defined in the table below <a href="#PBV-master">[PBV-master]</a>.

**The 'Eq.(4) floor'.** The bundle field `hEq4` of `ExtremiserPrimitives`
(Lean literal: `(uQ : ℝ) ^ 2 * S_cos ≥ m_G ^ 2 / S_G`) encodes the
lower bound

$$u^{2} \sum_{j\ne 0} (\Re \widetilde f(j))^{2}\, \widetilde K(j) \;\ge\; m_G^{2} \big/ S_G, \qquad S_G \;:=\; \sum_{j:\,\widetilde G(j)\ne 0} \widetilde G(j)^{2} \big/ \widetilde K(j),$$

which is MV 2010 Lemma 3.1 part (4) applied at $(f, K_{\mathrm{ms}}, G)$, where $G$ is an even, real-valued, $u$-periodic function that takes positive values on $[-1/4, 1/4]$ with $\widetilde G(0)=0$, and $m_G := \min_{0\le x\le 1/4} G(x)$ (the half-interval suffices by evenness of $G$). For the cosine multiplier $G(x) = \sum_{k=1}^{N} a_k\cos(2\pi kx/u)$ described in the next paragraph, $\widetilde G$ denotes the Parseval-normalised period-$u$ Fourier coefficient

$$\widetilde G(j) \;:=\; u^{-1/2}\int_{-u/2}^{u/2} G(x)\, e^{-2\pi i j x/u}\,dx,$$

so $\widetilde G(0)=0$, $\widetilde G(\pm j) = a_j\sqrt{u}/2$ for $1\le j\le N$, and $\widetilde G(j) = 0$ for $|j|>N$. Writing $S_1 := \sum_{j=1}^{N} a_j^{2}/\widetilde K(j)$ (which matches Lean's `S_1_analytic` and the anchor-table entry below), the displayed $S_G$ specialises by even symmetry to

$$S_G \;=\; 2\sum_{j=1}^{N}\bigl(a_j\sqrt{u}/2\bigr)^{2}\big/\widetilde K(j) \;=\; \tfrac{u}{2}\,S_1.$$

The bundle is instantiated with $S_G := u_{\mathrm{real}}\cdot S_1/2$, in agreement with this derivation, and the Lemma 3.1(3) gain identity $a = 2 m_G^{2}/S_G$ then reduces to $a = (4/u) m_G^{2}/S_1$ — exactly the formula recorded in the anchor table below. The Lean encoding of the underlying inner-product floor is `Sidon.MV.mv_inner_product_floor` in `Sidon.MVLemmas`, whose hypothesis is `∀ x ∈ Icc (-1/4) (1/4), m_G ≤ G x` — exactly MV's `[-1/4,1/4]` positivity hypothesis, not the larger interval $(-1/2,1/2)$.

**The $z_1$-free reduction (this work).** Equation (10) above carries $z_1$, an uncontrolled Fourier coefficient of the unknown extremiser. We absorb it by Cauchy–Schwarz:

*Lemma ($z_1$-absorption).* For any real $z_1, k_1$ with $2 z_1^{4}\le M-1$ and $2 k_1^{2}\le K_2-1$,

$$2 z_1^{2} k_1 + \sqrt{(M-1-2 z_1^{4})(K_2-1-2 k_1^{2})} \;\le\; \sqrt{(M-1)(K_2-1)}.$$

*Proof.* Apply the elementary inequality $\sqrt{ac}+\sqrt{bd}\le\sqrt{(a+b)(c+d)}$ (Cauchy–Schwarz on the vectors $(\sqrt a,\sqrt b),(\sqrt c,\sqrt d)$) to $(a,b,c,d) = (2 z_1^{4}, M-1-2 z_1^{4}, 2 k_1^{2}, K_2-1-2 k_1^{2})$. Then $\sqrt{ac} = \sqrt{4 z_1^{4} k_1^{2}} = 2 z_1^{2} k_1$ and $(a+b)(c+d) = (M-1)(K_2-1)$. $\square$

Combining the lemma with Eq.(10) yields the $z_1$-free master inequality

$$M + 1 + \sqrt{(M-1)(K_2-1)} \;\ge\; \frac{2}{u} + a, \tag{*}$$

in which only the kernel/multiplier quantities $K_2$ and $a$ appear. This $z_1$-absorption is the one analytic step in the present work beyond MV 2010; the displayed proof is the entire derivation. The corresponding Lean theorem is in `Sidon.MasterFromLemmas`.

**The three-scale kernel.** Let $\eta(x) := (2/\pi) (1-4x^2)^{-1/2} \mathbf{1}_{|x|<1/2}$ be the arcsine density of half-width $1/2$, and write the $\delta$-rescaled autoconvolution

$$K_{\mathrm{arc}}(\delta; x) \;:=\; \delta^{-1}\,(\eta * \eta)(x/\delta).$$

(Equivalently, with the paper's and Lean's $\delta$-scaled density $\eta_\delta(x) := \delta^{-1}(2/\pi)(1-(2x/\delta)^2)^{-1/2}\mathbf{1}_{|x|<\delta/2}$ supported on $(-\delta/2,\delta/2)$, one has $K_{\mathrm{arc}}(\delta;\cdot) = \eta_\delta\ast \eta_\delta$; the two presentations yield the identical kernel on $[-\delta,\delta]$.)
Then $K_{\mathrm{arc}}(\delta; \cdot)$ is a probability density supported on $[-\delta,\delta]$ (since $\mathrm{supp}(\eta \ast \eta) = [-1, 1]$ and the rescaling contracts by $\delta$), and its continuous Fourier transform, under the convention $\widehat g(\xi) := \int_{\mathbb{R}} g(x) e^{-2\pi i x \xi} dx$, is $\widehat{K_{\mathrm{arc}}}(\delta;\xi) = J_0(\pi\delta\xi)^{2}\ge 0$. Any convex combination of such kernels is therefore Bochner-admissible. Matolcsi–Vinuesa used a single
arcsine kernel. Here $K = K_{\mathrm{ms}}$ is a convex combination of
**three**, with half-widths $(\delta_1,\delta_2,\delta_3) = (138,55,25)/1000$, weights $(\lambda_1,\lambda_2,\lambda_3) = (85,10,5)/100$, and period $u = 1/2+\delta_1 = 638/1000$; its periodic
coefficients are $\widetilde{K_{\mathrm{ms}}}(j) = \sum_i \lambda_i J_0(\pi j\delta_i/u)^{2} \ge 0$. The multiplier $G$ is re-optimised
for this kernel as a $200$-term cosine sum
<a href="#PBV-kernel">[PBV-kernel]</a>. The $200$ rational coefficients $a_j$ are produced by a convex QP (`cvxpy` with solver fallback MOSEK $\to$ CLARABEL $\to$ SCS $\to$ ECOS), then rounded to a common denominator $\le 10^{8}$ and stored as the `coeffs_q` array in the SHA-256-anchored certificate `multiscale_arcsine_1292.json` (representative entries $a_1 = 95403771/50000000$, $a_2 = -161500769/100000000$). The solver's correctness does not enter the proof: only the recorded rationals are used downstream, and the certifier independently verifies $m_G\ge 998/1000$ and $S_1\le 29841/1000$ from them by `flint.arb` interval arithmetic. These parameters were chosen
by numerical search; the bound below is valid regardless of whether
they are globally optimal.

**Applicability to $K_{\mathrm{ms}}$.** MO 2009 Lemmas 3.1–3.4 / MV
2010 Lemma 3.1 require only that $K$ is a probability density
supported in $[-\delta,\delta]$ with nonnegative periodic Fourier
coefficients $\widetilde{K}(j)\ge 0$ and $K\in L^{2}$; no specific
kernel form (single arcsine, autoconvolution, etc.) is assumed.
$K_{\mathrm{ms}}$ satisfies all of these as a single admissible
kernel: it is a pdf (convex combination of pdfs) supported in
$[-\delta_1,\delta_1]$ (since each $K_{\mathrm{arc}}(\delta_i;\cdot)$
is supported in $[-\delta_i,\delta_i]\subseteq[-\delta_1,\delta_1]$),
its periodic Fourier coefficients
$\widetilde{K_{\mathrm{ms}}}(j) = \sum_i\lambda_i J_0(\pi j\delta_i/u)^2$
are nonnegative (each term $\ge 0$ by Fourier-transform linearity
applied to convex combinations), and $K_2\le 4.7897<\infty$. MO Lemma 2.1's hypothesis $\alpha_1 + \alpha_2 \le u$ (verbatim:
*"If $\alpha_1+\alpha_2 \le u$, ..."* — see
<a href="#MO-primitives">[MO-primitives]</a> loc 1) accepts equality.
We invoke it at $(\alpha_1, \alpha_2) = (1/2, \delta_1)$ with
$1/2 + \delta_1 = u$, which is precisely the boundary case that
MO 2009 themselves use: their proof of Lemma 3.3 reads verbatim
*"The function $f\ast f+f\circ f$ is square-integrable and supported
on $(-\tfrac12,\tfrac12)$. Since the inequality $\tfrac12+\delta\le u$
is satisfied, we may apply Lemma 2.1"* (MO 2009, p.~11), and MV 2010
adopts the same parameter $u=\tfrac12+\delta$ (MV 2010, p.~2)
together with kernels $K$ stated as *supported in $[-\delta,\delta]$*
(closed; MV 2010, p.~2). The boundary equality is therefore not a
gap in our application but a case already exercised in both source
papers. Two remarks make the rigour explicit. (i) The proof of MO
Lemma 2.1 (MO 2009, p.~5) needs only that the translates
$g_1(\cdot+ku)$, $k\neq0$, are disjoint from the integration
interval $(-\alpha_2,\alpha_2)$; at $\alpha_1+\alpha_2=u$ those
translates are supported in $(u-\alpha_1,\infty)=(\alpha_2,\infty)$
and $(-\infty,-\alpha_2)$, which are *open* sets disjoint from the
*open* interval $(-\alpha_2,\alpha_2)$ — the endpoints $\pm\alpha_2$
are excluded on both sides, so equality is admissible in the
hypothesis. (ii) MO Lemma 2.1 is a statement about $L^{2}$ classes
(both sides are unchanged under modification of $g_i$ on a
Lebesgue-null set), so even though $K_{\mathrm{arc}}(\delta_i;\cdot) =\eta_{\delta_i}\ast \eta_{\delta_i}$ has closed support
$[-\delta_i,\delta_i]$, it is continuous near and vanishes at the
endpoints $\pm\delta_i$ (there the supports of the two arcsine factors
$\eta_{\delta_i}\subseteq(-\delta_i/2,\delta_i/2)$ meet only at a
single point, so the convolution integral is zero), and since
$\lbrace \pm\delta_i\rbrace$ is Lebesgue-null it may in any case be taken to
represent its $L^{2}$ class as a function supported in the open
interval $(-\delta_i,\delta_i)$ (note $K_{\mathrm{arc}}$ is *not*
globally continuous: it carries a logarithmic singularity at the
origin, where $\int\eta_{\delta_i}^{2}=\infty$);
the same applies to $K_{\mathrm{ms}}=\sum_i\lambda_i K_{\mathrm{arc}}(\delta_i;\cdot)$ on $(-\delta_1,\delta_1)$.
With this $L^{2}$-class representative the hypothesis of MO Lemma
2.1 holds with strict inequality on the support sets, and the
identity is in any case insensitive to measure-zero modifications
of $K_{\mathrm{ms}}$. The cited MO/MV primitives therefore apply
to $K_{\mathrm{ms}}$ directly, with no analytic content introduced
beyond what MO 2009 / MV 2010 explicitly state.

**The five certified functionals.** Each quantity below is enclosed
in `flint.arb` interval arithmetic at 256-bit precision and rounded
*outward* to an exact rational
<a href="#PBV-anchors">[PBV-anchors]</a>:

| Functional | Certified bound | Decimal |
| ---------- | --------------- | ------- |
| $k_1 = \widehat{K_{\mathrm{ms}}}(1)$ (kernel mass moment) | $\ge 9212/10000$ | $0.9212$ |
| $K_2 = \lVert K_{\mathrm{ms}}\rVert_2^{2}$ (kernel energy) | $\le 47897/10000$ | $4.7897$ |
| $w_{\min} := \min_{1\le j\le 200}\widetilde{K_{\mathrm{ms}}}(j)$ (positivity of QP denominators) | $\ge 1/5000$ | $2.0\times 10^{-4}$ |
| $S_1 = \sum_{j=1}^{200} a_j^{2}/\widetilde{K_{\mathrm{ms}}}(j)$ (multiplier denominator) | $\le 29841/1000$ | $29.841$ |
| $m_G = \min_{[0,1/4]} G$ (multiplier minimum) | $\ge 998/1000$ | $0.998$ |
| $a = (4/u) m_G^{2}/S_1$ (gain) | $\ge 20925/100000$ | $0.20925$ |

The $z_1$-free inequality consumes only $K_2$ and $a$; through
$a = (4/u) m_G^{2}/S_1$ it depends on $m_G$ and $S_1$. The remaining
quantity $k_1$ does not enter the rational headline — it is used only
by the sharper *refined* inequality that the Arb cell-search employs
to certify the tighter $M_{\mathrm{cert}}\approx 1.29232$
<a href="#PBV-fail">[PBV-fail]</a>.

**How each anchor is certified.** $K_2$ is split into a bulk integral
on $[0,T]$ with $T = 10^{5}$, computed by `flint.arb` adaptive
Gauss–Legendre quadrature (certified bulk
$\in [4.78882342, 4.78890519]$), and a tail past $T$ bounded
analytically by the classical Bessel envelope
$\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$ for $z>0$ (Watson 1944,
§7.21), which gives tail $\le 8.19\times 10^{-5}$
<a href="#Watson">[Watson]</a>; bulk plus tail rounds outward to
$47897/10000$. The minimum $m_G$ is certified
by partitioning $[0,1/4]$ into $32768$ closed cells and forming the
second-order Taylor enclosure of $G$ on each cell in arb interval
arithmetic; the minimum of the per-cell lower endpoints is
$\ge 0.99997987 > 998/1000$. $S_1$ and $k_1$ are evaluated as exact
rational sums in arb at radii below $10^{-70}$
<a href="#PBV-anchors">[PBV-anchors]</a>. The positivity certificate
$w_{\min} \ge 1/5000$ is obtained by evaluating
$\widetilde{K_{\mathrm{ms}}}(j) = \sum_i \lambda_i J_0(\pi j\delta_i/u)^{2}$
in `flint.arb` at $256$-bit precision for each $j\in\lbrace 1,\dots,200\rbrace$
and taking the minimum lower endpoint; the rigorous lower endpoint
of the minimum is $2.0817\times 10^{-4}$ at $j=147$, which majorises
$1/5000=2.0\times 10^{-4}$. This guarantees $S_1$ is finite and
furnishes the positivity hypothesis under which the $S_1$ upper bound
is certified.

**Closing the inequality.** Set $\Phi(M) = M + 1 + \sqrt{(M-1)(K_2-1)}$ and $\tau = 2/u + a$. The master inequality says
$\Phi(R(f)) \ge \tau$ for every admissible $f$. Since $\Phi$ is
continuous and strictly increasing in $M$, if $\Phi(M_0) < \tau$ at a
fixed rational $M_0$, then $R(f) > M_0$ for all $f$, so $C_{1a} \ge M_0$ <a href="#PBV-inversion">[PBV-inversion]</a>. Take
$M_0 = 1292/1000$. From $u = 638/1000$ and $a \ge 20925/100000$,

$$
\tau \;=\; \frac{2}{u} + a \;\ge\; \frac{2000}{638} + \frac{20925}{100000} \;=\; \frac{4267003}{1276000} \;=\; 3.344046\ldots .
$$

From $K_2 \le 47897/10000$ and
$(M_0-1)(K_2-1) \le (292/1000)(37897/10000) = 11065924/10^{7}$, the
rational $105195/10^{5}$ majorises the square root because
$(105195/10^{5})^{2} = 11065988025/10^{10} \ge 11065924/10^{7}$, so

$$
\Phi(1292/1000) \;\le\; \frac{1292}{1000} + 1 + \frac{105195}{10^{5}} \;=\; \frac{66879}{20000} \;=\; 3.34395 .
$$

Hence the inequality strictly fails at $M_0$, with an exactly
rational margin

$$
\tau - \Phi(1292/1000) \;\ge\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\ge\; 9.6\times10^{-5} \;>\; 0 ,
$$

yielding $C_{1a} \ge 1292/1000 = 1.292$ <a href="#PBV-fail">[PBV-fail]</a>.
The closing step is exact rational arithmetic, independent of any
floating-point computation.

**Mechanisation.** The analytic reduction — admissibility of
$K_{\mathrm{ms}}$, the master inequality, the quadratic inversion,
and the rational closing arithmetic — is mechanised in Lean 4
(**30 modules, $\approx 15{,}577$ lines**: 13 core `Sidon/*.lean` at
7655 lines plus 17 `Sidon/Constructor/*.lean` at 7922 lines, `mathlib
v4.29.1`, no `sorry`). The formalisation exports two headlines, both
concluding `autoconvolution_ratio f ≥ 1292/1000`, where
`autoconvolution_ratio` is the Lean definition of $R(f) = \lVert f\ast f\rVert_{\infty}/(\int f)^{2}$ in `Sidon.Defs`:
- The *conditional* `autoconvolution_ratio_ge_1292_1000` takes an
  analytic-primitives record `ExtremiserPrimitives f` whose fields are
  Lean restatements of the cited MV/MO results
  <a href="#MV-primitives">[MV-primitives]</a>
  <a href="#MO-primitives">[MO-primitives]</a>; its dependency closure
  is Lean's three logical axioms (`propext`, `Classical.choice`,
  `Quot.sound`) plus **two** numerical user axioms recording the
  certifier's outputs for $K_2$ and $a$.
- The *unconditional* `C1a_ge_1292_unconditional` takes only raw
  admissibility hypotheses and *constructs* the bundle via
  `ExtremiserPrimitives.of_admissible`; its dependency closure adds
  **two further** numerical user axioms (`min_G_analytic_ge_minGLowerQ`
  for $\min_{[0,1/4]} G \ge 998/1000$, and `K_ms_fourier_lattice_pos_active`
  for $\widetilde{K_{\mathrm{ms}}}(j) > 0$ on $j\in\lbrace 1,\dots,200\rbrace$),
  for **four** numerical user axioms in total.
The bundle has 13
fields: five real parameters (`m_G`, `S_G`, `S_cos`, `LHS1`, `LHS2`),
four numerical-sanity hypotheses (`K2_ge_1`, `R_ge_1`, `S_G_pos`,
`gain_eq`), and the four MV Lemma 3.1 outputs at $(f, K_{\mathrm{ms}})$
(`hEq1`, `hEq2`, `hEq3_ge`, `hEq4`). The previously exported
Schwartz-class variants `_schwartz` / `_schwartz_residual` (and their
backing modules `Sidon.MultiScaleSchwartz` / `Sidon.SchwartzAtomicDischarge`)
were retired by the S1+S2 refactor when the Schwartz
`ParsevalSplitSchwartz` predicate was shown to be vacuously satisfiable
by Paley–Wiener + Carlson; only the general bundle survives. The
slack-soundness statements, the quadratic inversion, the
`of_admissible` constructor, and the
assembly are ordinary Lean theorems <a href="#PBV-lean">[PBV-lean]</a>.
The numerical anchors are reproducible from a SHA-256-anchored
certificate and are cross-checked at 50 decimal digits by an `mpmath`
script that does not call `flint.arb` (independent library and
code-path; same mathematical formulas)
<a href="#PBV-cert">[PBV-cert]</a>.

**Concrete Lean anchoring (Option B).** The numerical user axiom
`gain_analytic_ge_gainLowerQ` now bounds a *concrete defined real
expression* in Lean rather than an opaque symbol, so the axiom
genuinely asserts an inequality on the analytic functional
$(4/u)\cdot m_G^{2}/S_1$. The supporting definitions in
`lean/Sidon/MultiScale.lean` are:

* `qpNumerators : List ℤ` (line 569) — the 200 QP coefficient
  numerators with common denominator $10^{8}$; the cosine sum
  `G_concrete` reads them via `qpNumerators.getD i 0` (total, returning
  the default $0$ out of range), so no list-length lemma is needed.
* `noncomputable def G_concrete (x : ℝ) : ℝ` (line 775) — the
  explicit $200$-term cosine multiplier
  $G(x) = \sum_{i=0}^{199} (a_i/10^{8}) \cos(2\pi (i+1) x / u)$,
  with $a_i =$ `qpNumerators.getD i 0`.
* `noncomputable def Ktilde_ms (j : ℕ) : ℝ` (line 784) — the
  Bessel-form period-$u$ Fourier coefficient
  $\widetilde{K_{\mathrm{ms}}}(j) = \sum_i \lambda_i J_0(\pi j \delta_i/u)^{2}$,
  written in terms of `Sidon.Bessel.besselJ0`.
* `noncomputable def S_1_analytic : ℝ` (line 792) — the concrete
  sum $\sum_{i=0}^{199} (a_i/10^{8})^{2} / \widetilde{K_{\mathrm{ms}}}(i+1)$.
* `noncomputable def min_G_analytic : ℝ` (line 797) — the analytic
  infimum $\inf_{x\in[0,1/4]} G_{\mathrm{concrete}}(x)$, encoded as
  `sInf (G_concrete '' Set.Icc (0 : ℝ) (1/4))`.
* `noncomputable def gain_analytic : ℝ` (line 830) — the analytic
  functional $(4/u)\cdot m_G^{2}/S_1$, encoded as
  `(4 / uQ_real) * min_G_analytic^2 / S_1_analytic`. This is a
  concrete `noncomputable def`, not an opaque symbol with a trivial
  body.

Both numerical axioms therefore bind genuine real analytic
functionals over the explicit kernel and multiplier:
`K2_analytic_le_K2UpperQ` bounds
`K2_analytic := ∫ K_ms² ∂volume` (the $L^{2}$ norm squared of
$K_{\mathrm{ms}}$ on $\mathbb{R}$), and `gain_analytic_ge_gainLowerQ`
bounds the displayed `gain_analytic`. The `ExtremiserPrimitives`
bundle takes `m_G : ℝ` and `S_G : ℝ` as named real parameters
together with the gain identity `gain_eq : gain_analytic = 2·m_G²/S_G`
as a field; the consumer instantiates with `m_G := min_G_analytic`
and `S_G := uQ_real · S_1_analytic / 2`, and the gain identity is
then a real algebraic identity proved by `field_simp; ring`.

The 200 numerators in `qpNumerators` are emitted from the same
SHA-256-anchored JSON certificate that the `flint.arb` certifier
consumes (the `coeffs_q` array in `multiscale_arcsine_1292.json`;
see <a href="#PBV-cert">[PBV-cert]</a>). The Python certifier's
emit step writes both the JSON `coeffs_q` numerators and the Lean
`qpNumerators` list from a single source, and the bind-check script
`audit_qp_coeffs.py` (sibling to `audit3_mpmath.py`) reparses the
Lean `qpNumerators` list, the JSON `coeffs_q` array, and the
certifier's in-memory rationals, asserting bit-identical equality
across all three surfaces. The conditional headline reaches a
five-axiom inventory (three Lean core + two numerical); the
unconditional headline `C1a_ge_1292_unconditional` reaches seven
(three Lean core + four numerical). Both build clean
(`cd lean && lake build`) with zero `sorry`. An additional
6-agent independent re-verification of the three-scale 1.292
instance was performed on 2026-05-15, confirming the then-current
5-axiom budget (the unconditional headline and its two further
certifier axioms postdate that audit), build-green status,
bit-for-bit reproducibility of the
certificate, and `mpmath`-corroborated numerics (the earlier
14-agent audit was on the 2-scale 1.28984 predecessor; its findings
on the kernel / QP / master-inequality framework transport directly
to the present three-scale 1.292 headline).

## Scope of the claim

This work is at strict parity with the published MV 2010 proof of
$C_{1a}\ge 1.27481$ on every component of the proof tree, and
strictly above it on three:

| Component | MV 2010 (published) | This work |
| --------- | ------------------- | --------- |
| $\mathcal{F}\to$ square-integrable reduction | Cited by reference to Schinzel–Schmidt 2002 Theorem 1 (MV §2; see <a href="#MV-reduction">[MV-reduction]</a>) | **Cited identically** to <a href="#SS2002">[SS2002]</a> Theorem 1 (see "Reduction to square-integrable $f$" above) |
| MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3 (period-$u$ Parseval, constant-plus-tail split, lattice $F$-bound, torus split for $f\ast f + f\circ f$) | Cited by reference, and *applied to admissible $f$* in MV's English-prose proof: *"Lemma 3.1. [Lemmas 3.1, 3.2, 3.3, 3.4 in [6]]"* (MV p. 3; see <a href="#MV-primitives">[MV-primitives]</a>) | **Cited identically** and applied directly to the admissible kernel $K_{\mathrm{ms}}$ (a pdf supported in $[-\delta_1,\delta_1]$ with $\widetilde{K_{\mathrm{ms}}}(j)\ge 0$ and $K_{\mathrm{ms}}\in L^{2}$ — the only hypotheses MO 2009 Lemmas 3.1–3.4 / MV 2010 Lemma 3.1 require). The Lean theorem encodes the resulting outputs at $(f, K_{\mathrm{ms}})$ as named hypothesis fields (`hEq1`, `hEq2`, `hEq3_ge`, `hEq4`) of the record `ExtremiserPrimitives f`, and the paper discharges those fields by direct citation to MO 2009 Lemma 3.3 / Lemma 3.2 applied to $K_{\mathrm{ms}}$, exactly as MV 2010 discharged its single-arcsine applications by citation to MO 2009. The verbatim MO/MV statements are recorded in <a href="#MO-primitives">[MO-primitives]</a> / <a href="#MV-primitives">[MV-primitives]</a>. (The citation-discharge of MO/MV at $K_{\mathrm{ms}}$ is mathematical content; the absence of a packaged mathlib API in the form the bundle consumes is mere engineering — see "How the bound is established" above for this distinction.) No new analytic content is introduced beyond what MV cites. |
| Master inequality + $z_1$-absorption + quadratic inversion + slack soundness + kernel admissibility | Proved analytically in MV §3 (English-prose proofs) | **Strictly above MV**: formally proved in Lean (axiom-free outside the numerical user axioms — two for the conditional headline, four for the unconditional — across 30 modules / ~15,577 lines: the 13 core `Defs`, `Bessel`, `FourierAux`, `TorusParseval`, `MVLemmas`, `MasterFromLemmas`, `BundleDefs`, `BundleEq1`, `BundleEq2Schwartz`, `BundleEq3Schwartz`, `BundleEq4`, `BilinearParseval`, `MultiScale`, plus the 17 axiom-free `Sidon/Constructor/*` modules mechanising `of_admissible`); 0 `sorry` |
| Numerical anchors ($K_2$, $a$, $S_1$, $m_G$, $k_1$) | Mathematica 6 / LOQO; values reported as decimals (MV p. 7); no public certificate; no independent re-verification documented | **Strictly above MV**: `flint.arb` interval arithmetic at 256-bit precision, outward-rounded to exact rationals; SHA-256-anchored certificate `multiscale_arcsine_1292.json`; cross-check at 50 decimal digits via `audit3_mpmath.py` (independent arbitrary-precision library implementing the same mathematical formulas, so it catches arithmetic and library bugs but not formula bugs) |
| Closing arithmetic | Numerical substitution (MV p. 7) | **Strictly above MV**: exact rational arithmetic, $\tau-\Phi(1292/1000)\ge 307/3190000\ge 9.6\times 10^{-5}$, machine-checked in Lean by `norm_num` |

**In summary.** At the level of *provability*, every classical-analysis
ingredient we cite (Schinzel–Schmidt 2002 Theorem 1; MO 2009 Lemmas 2.1,
2.2, 3.2, 3.3) is *exactly the same* ingredient MV 2010 cites, in
*exactly* the same role: those lemmas accept any admissible $K$
(pdf, supported in $[-\delta,\delta]$, $\widetilde{K}(j)\ge 0$,
$K\in L^{2}$), and $K_{\mathrm{ms}}$ qualifies directly. The Lean
theorem encodes the resulting outputs at $(f, K_{\mathrm{ms}})$ as
named hypothesis fields and the paper discharges them by direct
citation to MO 2009 / MV 2010, exactly as MV 2010 discharged its
single-kernel applications by citation to MO 2009. Beyond that, every
step *between* those classical ingredients — admissibility of
$K_{\rm ms}$, the assembled master inequality, the quadratic
inversion, the slack-soundness theorems, and the final rational
arithmetic closing $\Phi(1292/1000) < \tau$ — is machine-checked in
Lean 4 (axiom-free, 0 `sorry`), whereas MV's counterparts are
English-prose proofs.

**Trust set.** Beyond Lean's logical axioms, the bound depends on
two *kinds* of input, of the same shape every published
computer-assisted real-number proof uses (Flyspeck, PFR,
Cohn–Elkies sphere-packing):

* *Analytic primitives* — MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3 / MV 2010
  Lemma 3.1, applied to the admissible kernel $K_{\mathrm{ms}}$, and
  Schinzel–Schmidt 2002 Theorem 1 for the $\mathcal{F}\to L^{2}$
  reduction. Cited identically to MV 2010; any reader who accepts
  MV 2010 on these citations accepts ours. (For the conditional
  headline these enter as the `ExtremiserPrimitives f` hypothesis; for
  the unconditional headline they are mechanised axiom-free inside
  `ExtremiserPrimitives.of_admissible`.)
* *Numerical anchors* — $K_2$, $a$, and the supporting $S_1$, $m_G$,
  $k_1$, $w_{\min}$ for the new three-scale kernel and 200-mode $G$.
  These are **new computations** specific to the kernel/multiplier
  pair, *not* facts contained in MV 2010. The Lean numerical axioms
  (two for the conditional headline — $K_2$ and $a$; four for the
  unconditional headline — additionally $\min G \ge 998/1000$ and the
  lattice positivity $\widetilde{K_{\mathrm{ms}}}(j)>0$ on
  $j\in\lbrace 1,\dots,200\rbrace$) bind
  defined analytic functionals over the explicit kernel:
  `K2_analytic_le_K2UpperQ` asserts
  $\int_{\mathbb{R}} K_{\mathrm{ms}}^{2} d\mathrm{vol} \le 47897/10000$,
  and `gain_analytic_ge_gainLowerQ` asserts
  $(4/u_{\mathrm{real}})\cdot\bigl(\inf_{x\in[0,1/4]} G_{\mathrm{concrete}}(x)\bigr)^{2} / \sum_{i=0}^{199} (a_i/10^{8})^{2}/\widetilde{K_{\mathrm{ms}}}(i+1) \ge 20925/100000$,
  where $G_{\mathrm{concrete}}$ is the explicit 200-term cosine sum
  with coefficients `qpNumerators[i]/10⁸` and $\widetilde{K_{\mathrm{ms}}}$
  the Bessel-form period-$u$ Fourier coefficient (all of these are
  defined Lean expressions; see "Concrete Lean anchoring (Option B)"
  above). The Lean `S_1_analytic` uses Lean's convention $a/0 = 0$;
  for the specific $(\delta_i, u)$ values, the certifier validates
  $\widetilde{K_{\mathrm{ms}}}(j) \ge 2\times 10^{-4}$ for
  $j\in\lbrace 1,\dots,200\rbrace$ externally (the `w_min` check), so no zero
  terms occur and the Lean expression equals the analytic functional.
  The certifier's positivity check is part of the trust set; a
  Lean-side formalisation would require a Bessel-zero-localisation
  lemma that mathlib does not currently expose. We trust `flint.arb`
  (Johansson 2017, peer-reviewed), the Python certifier driver
  `bisect_alt_kernel.py`, and the independent `mpmath` cross-check
  `audit3_mpmath.py`; the SHA-256 anchor pins the certificate. This
  layer is implementationally stronger than MV's numerical layer
  (Mathematica 6 / LOQO, no public certificate, no independent
  re-verification documented): our anchors are outward-rounded
  interval enclosures rather than heuristic floats, and the codepath
  is reproducible.

The net effect is therefore: same analytic trust as MV 2010, plus a
disjoint computer-assisted numerical layer that is implementationally
more rigorous but contains new content. This is the same overall
shape as Flyspeck (Hales 2017) and the PFR formalisation: cited
classical analysis + new computer-assisted numerics.

## Additional comments and links

- Canonical page: [`constants/1a.md`](https://teorth.github.io/optimizationproblems/constants/1a.html).
  This submission adds the $1.292$ row and changes no other row.
- Reproduce: `python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel`
  (certificate); `python audit_consistency.py` (cross-surface audit);
  `cd lean && lake build`, then `lake env lean AxiomCheckBundleDefs.lean`
  / `AxiomCheckFourier.lean` / `AxiomCheckMV.lean` /
  `AxiomCheckTorus.lean` (build and per-module axiom inventory)
  <a href="#PBV-cert">[PBV-cert]</a>.
- Toolchain (bit-identical reproduction): `python-flint` $0.8.0$,
  `mpmath` $1.3.0$, `numpy` $\ge 2.0$, Python $\ge 3.10$; Lean
  `leanprover/lean4:v4.29.1` with `mathlib` commit
  `5e932f97dd25535344f80f9dd8da3aab83df0fe6` (pinned in
  `lean/lean-toolchain` and `lean/lakefile.lean`). The certifier
  records `prec_bits=256` in
  `certificates/reference_anchors.json` and `multiscale_arcsine_1292.json`.

## References

- <a id="1a-page"></a>**[1a-page]** Tao, Terence (ed.). *An autocorrelation constant related to Sidon sets.* `teorth/optimizationproblems`, `constants/1a.md`. [Page](https://teorth.github.io/optimizationproblems/constants/1a.html)
	- <a id="1a-def"></a>**[1a-def]**
	  **loc:** `constants/1a.md`, "Description of constant".
	  **quote:** "$C_{1a}$ is the largest constant for which one has $\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x) dx \geq C_{1a} (\int_{-1/4}^{1/4} f(x) dx)^2$ for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$."
	- <a id="1a-lb"></a>**[1a-lb]**
	  **loc:** `constants/1a.md`, "Known lower bounds" table.
	  **quote:** Rows, in order: $1$ (Trivial); $1.182778$ ([MO2004]); $1.262$ ([MO2009]); $1.2748$ ([MV2009]); $1.28$ ([CS2017]); $1.2802$ ([XX2026], "Unpublished improvement, Grok"). The page attributes $1.2802$ to [XX2026] and $1.28$ to [CS2017]; it does not attribute $1.2802$ to Cloninger–Steinerberger.

- <a id="PBV2026"></a>**[PBV2026]** Piterbarg, Andrei; Bajaj, Jai; Vincent, Derrick. *A New Lower Bound for the Supremum of Autoconvolutions.* Preprint, 2026. This repository: `lower_bound_proof.tex` / `.pdf`; Lean under `lean/Sidon/`.
	- <a id="PBV-def"></a>**[PBV-def]**
	  **loc:** `lower_bound_proof.tex`, Subsection "The Constant $C_{1a}$" (`subsec:intro-constant`, lines 332–357).
	  **quote:** "$C_{1a} = \inf_{f\in\mathcal{F}} R(f)$, $R(f) := \lVert f\ast f\rVert_{\infty}/(\int f)^2$", with $\mathcal{F}$ the non-negative $f\in L^1(\mathbb{R})$ such that $\mathrm{supp}(f) \subseteq (-1/4,1/4)$ and $\int f > 0$.
	- <a id="PBV-main"></a>**[PBV-main]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Main result" (`thm:main`, lines 413–424).
	  **quote:** "Let $f:\mathbb{R}\to\mathbb{R}$ be nonnegative with $\mathrm{supp}(f)\subseteq(-1/4,1/4)$, $\int f>0$, and $\lVert f\ast f\rVert_{\infty}<\infty$. Then $\lVert f\ast f\rVert_{\infty}/(\int f)^2 \ge 1292/1000$. In particular $C_{1a}\ge 1292/1000=1.292$."
	- <a id="PBV-master"></a>**[PBV-master]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Master inequality, $z_1$-free form" (`thm:mv-master`, lines 477–488); the $z_1$-absorption is in the proof (lines 490–522).
	  **quote:** "For $K$ Bochner-admissible at scale $\delta$, $u=1/2+\delta$, and $G$ admissible with constant $m_G$ (defining $S_1$, $a$): for every $f\in\mathcal{F}$ with $\int f>0$, $M + 1 + \sqrt{(M-1)(K_2-1)} \ge 2/u + a$." The sharp form additionally carries $k_1=\widehat{K}(1)$ and $z_1=\lvert\widehat{f}(1)\rvert$ and reduces to this form by Cauchy–Schwarz.
	- <a id="PBV-inversion"></a>**[PBV-inversion]**
	  **loc:** `lower_bound_proof.tex`, Lemma "Quadratic inversion" (`lem:inversion`, lines 558–569).
	  **quote:** "$\Phi(M) := M+1+\sqrt{(M-1)(K_2-1)}$ is continuous and strictly increasing on $[1,\infty)$ with $\Phi(1)=2$; for $\tau\ge 2$, every $f\in\mathcal{F}$ with $\Phi(R(f))\ge\tau$ satisfies $R(f)\ge M_\ast$, the unique solution of $\Phi(M_\ast )=\tau$. With $\tau=2/u+a$, $C_{1a}\ge M_\ast$."
	- <a id="PBV-kernel"></a>**[PBV-kernel]**
	  **loc:** `lower_bound_proof.tex`, Definition "Three-scale kernel" (`def:Kms`, lines 854–874) and Theorem "Admissibility" (`thm:admissibility`, lines 895–907).
	  **quote:** $(\delta_1,\delta_2,\delta_3)=(138,55,25)/1000$, $(\lambda_1,\lambda_2,\lambda_3)=(85,10,5)/100$, $u=1/2+\delta_1=638/1000$; $K_{\mathrm{ms}}=\sum_i \lambda_i K_{\mathrm{arc}}(\delta_i;\cdot)$ is Bochner-admissible with $\widetilde{K_{\mathrm{ms}}}(j)=\sum_i \lambda_i J_0(\pi j\delta_i/u)^2 \ge 0$; multiplier degree $N=200$.
	- <a id="PBV-anchors"></a>**[PBV-anchors]**
	  **loc:** `lower_bound_proof.tex`, Lemmas `lem:k1`, `lem:K2`, `lem:S1`, `lem:mG`, `lem:a` (lines 1041–1135) and table `tab:anchors` (lines 1158–1175).
	  **quote:** $k_1 := \widehat{K_{\mathrm{ms}}}(1) \ge 9212/10000$; $K_2 := \lVert K_{\mathrm{ms}}\rVert_{2}^{2} \le 47897/10000$ (proof: bulk $[4.78882342, 4.78890519]$ by arb adaptive Gauss–Legendre on $[0,10^{5}]$, tail $\le 8.19\times 10^{-5}$ via the Watson envelope $\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$ for $z>0$); $S_1 = \sum_{j=1}^{200} a_j^{2}/\widetilde{K_{\mathrm{ms}}}(j) \le 29841/1000$; $m_G := \min_{[0,1/4]} G \ge 998/1000$ (32768-cell second-order Taylor B&B in arb); $a = (4/u) m_G^{2}/S_1 \ge 20925/100000$.
	- <a id="PBV-fail"></a>**[PBV-fail]**
	  **loc:** `lower_bound_proof.tex`, Proposition "Strict failure at the rational witness" (`prop:fail`, lines 1219–1227; proof through line 1258).
	  **quote:** "$\Phi(1292/1000) \le 66879/20000 = 3.34395 < 4267003/1276000 \le \tau$, with margin $\tau-\Phi(1292/1000) \ge 307/3190000 \ge 9.6\times10^{-5} > 0$." The Arb cell-search using the sharper refined inequality (which involves $k_1$) independently re-certifies $M_{\mathrm{cert}}\ge 1.29232$.
	- <a id="PBV-lean"></a>**[PBV-lean]**
	  **loc (core module).** `lean/Sidon/MultiScale.lean`: numerical axioms `K2_analytic_le_K2UpperQ` at line 998, `gain_analytic_ge_gainLowerQ` at line 1026, and `min_G_analytic_ge_minGLowerQ` at line 1065; `ExtremiserPrimitives` structure at line 1465; conditional headline `autoconvolution_ratio_ge_1292_1000` at line 1586; slack-soundness theorems `K_two_upper_bound`, `k_one_lower_bound`, `S_one_upper_bound`, `min_G_lower_bound`, `gain_lower_bound` at lines 1090–1112; `autoconvolution_ratio` definition in `Sidon.Defs`. (Line anchors verified against source at `mathlib v4.29.1`, commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`.)
	  **loc (constructor chain).** The unconditional headline and the admissibility-to-bundle constructor live under `lean/Sidon/Constructor/` (17 axiom-free modules, 7922 LoC): `ExtremiserPrimitives.of_admissible` at `Constructor/Assembly.lean:107`; unconditional headline `C1a_ge_1292_unconditional` at `Constructor/Assembly.lean:246`; the fourth numerical axiom `K_ms_fourier_lattice_pos_active` at `Constructor/LatticePositivity.lean:187`. The whole formalisation totals 30 modules (~15.6 kLoC: 13 core `Sidon/*.lean` at 7655 LoC plus the 17 `Sidon/Constructor/*.lean` at 7922 LoC). The previously exported Schwartz variants (`autoconvolution_ratio_ge_1292_1000_schwartz` and `_schwartz_residual`) and their backing modules (`Sidon.MultiScaleSchwartz` and `Sidon.SchwartzAtomicDischarge`) were retired by the S1+S2 refactor and no longer exist in the repository. (The file names `BundleEq2Schwartz.lean` and `BundleEq3Schwartz.lean` retain the `Schwartz` suffix from the pre-S1+S2 module layout, but their contents now serve the general bundle discharge for the headline; renaming is a deferred cosmetic.)
	  **axiom inventory (two headlines).** The formalisation exports two headlines with distinct dependency closures:
	  • *Conditional* `autoconvolution_ratio_ge_1292_1000` takes `(P : ExtremiserPrimitives f)` as a hypothesis and concludes `autoconvolution_ratio f ≥ 1292/1000`; its user-axiom closure is exactly two kernel-specific facts, `K2_analytic_le_K2UpperQ` ($K_2 \le 47897/10000$) and `gain_analytic_ge_gainLowerQ` ($a \ge 20925/100000$).
	  • *Unconditional* `C1a_ge_1292_unconditional` takes only raw admissibility hypotheses (`Integrable f`, `MemLp f 2`, $\mathrm{supp} f \subseteq (-1/4,1/4)$, $f\ge 0$, $\int f = 1$) and *constructs* the bundle via `ExtremiserPrimitives.of_admissible`; its user-axiom closure is four kernel-specific facts — the two above plus `min_G_analytic_ge_minGLowerQ` ($\min_{[0,1/4]} G \ge 998/1000$) and `K_ms_fourier_lattice_pos_active` ($\widetilde{K_{\mathrm{ms}}}(j) > 0$ for every $j\in\lbrace 1,\dots,200\rbrace$). All four are logically decidable, `flint.arb`-backed at 256-bit precision, mpmath-corroborated, and SHA-256-anchored. Both closures additionally reach Lean's three core logical axioms (`propext`, `Classical.choice`, `Quot.sound`).
	  The four MV Lemma 3.1 output fields `hEq1`, `hEq2`, `hEq3_ge`, `hEq4` of `ExtremiserPrimitives` are Lean restatements of MV 2010 Lemma 3.3 / MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3 — see <a href="#MV-primitives">[MV-primitives]</a> and <a href="#MO-primitives">[MO-primitives]</a> for the verbatim source statements. For the conditional headline these fields are assumed; for the unconditional headline `of_admissible` discharges them axiom-free (modulo the two additional certifier axioms). The remaining bundle fields are five real parameters (`m_G`, `S_G`, `S_cos`, `LHS1`, `LHS2`) — bound to the concrete analytic functionals via five `*_eq` hypothesis fields (`m_G_eq`, `S_G_eq`, `S_cos_eq`, `LHS1_eq`, `LHS2_eq`), so the bundle forces the canonical $(f, K_{\mathrm{ms}})$ values rather than arbitrary reals — together with the numerical-sanity hypotheses (`K2_ge_1`, `R_ge_1`, `S_G_pos`, `gain_eq`). The slack-soundness theorems are one-line `norm_num` checks.
	- <a id="PBV-cert"></a>**[PBV-cert]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`, `multiscale_arcsine_1292.json`; `audit3_mpmath.py`; `README.md`.
	  **quote:** `multiscale_arcsine_1292.json` has `sha256_of_body = 5fa9ae372b23d07f73f41d73c1740926115eb494b6ba3840551458ba8143a7c2` and `M_cert = 66167/51200` ($1.29232421875$). `reference_anchors.json` records the anchors ($k_1=0.9212465899364083$, $K_2\in[4.7888234212591545, 4.7889051816332424]$, $S_1=29.8409064555132666$, $m_G=0.9999798743824747$, $a=0.2100921474866837$) and the kernel parameters (`deltas` $138/55/25 /1000$, equivalently $69/500$, $11/200$, $1/40$; `lambdas` $85/10/5 /100$, equivalently $17/20$, $1/10$, $1/20$; `u` $638/1000 = 319/500$; `n_coeffs` $200$; `prec_bits` $256$). `audit3_mpmath.py` recomputes $K_2$ and $a$ at 50 digits independently of `flint.arb`.

- <a id="MO2004"></a>**[MO2004]** Martin, Greg; O'Bryant, Kevin. *The symmetric subset problem in continuous Ramsey theory.* Exp. Math. **16** (2007), no. 2, 145–165. [arXiv:math/0410004](https://arxiv.org/abs/math/0410004)

- <a id="MO2009"></a>**[MO2009]** Martin, Greg; O'Bryant, Kevin. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), no. 1, 219–235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121)
	- <a id="MO-primitives"></a>**[MO-primitives]** Verbatim source for the four Fourier-analytic primitives invoked by the present work and packaged in Lean as the `hEq1`/`hEq2`/`hEq3_ge`/`hEq4` fields of `ExtremiserPrimitives`.
	  **loc 1:** MO 2009 (arXiv:0807.5121v2), p. 5, Lemma 2.1 (period-$u$ Parseval — the *foundational* form of the torus split).
	  **quote 1:** "For $i\in\lbrace 1,2\rbrace$, suppose that $g_i$ is a square-integrable function supported on $(-\alpha_i,\alpha_i)$. If $\alpha_1+\alpha_2\le u$, then $\int_{\mathbb{R}} g_1(x)\overline{g_2(x)} dx = u\sum_{r\in\mathbb{Z}} \widetilde{g_1}(r)\overline{\widetilde{g_2}(r)}.$"
	  **loc 2:** MO 2009, p. 6, Lemma 2.2 (1-periodic Parseval — used in the constant-plus-tail split).
	  **quote 2:** "If $g_1$ and $g_2$ are square-integrable functions supported on $(-\tfrac12,\tfrac12)$, then $\int_{\mathbb{R}} g_1(x)\overline{g_2(x)} dx = \sum_{r\in\mathbb{Z}} \widehat{g_1}(r)\overline{\widehat{g_2}(r)}$; in particular $\lVert g_1\rVert_2^2 = \sum_{r\in\mathbb{Z}}\lvert \widehat{g_1}(r)\rvert^2$."
	  **loc 3:** MO 2009, p. 10, proof of Lemma 3.2 (contains the constant-plus-tail split and the lattice $F$-bound *verbatim*).
	  **quote 3:** "$\int_{\mathbb{R}}(f\circ f(x))K(x) dx = \sum_{r\in\mathbb{Z}} \widehat{f\circ f}(r)\overline{\widehat{K}(r)} = \sum_{r\in\mathbb{Z}}\lvert\widehat{f}(r)\rvert^2\overline{\widehat{K}(r)} = 1 + \sum_{r\ne 0}\lvert\widehat{f}(r)\rvert^2\overline{\widehat{K}(r)}$ … $\sum_{r\in\mathbb{Z}}\lvert\widehat{f}(r)\rvert^4 = \sum_{r\in\mathbb{Z}}\lvert\widehat{f\ast f}(r)\rvert^2 = \lVert f\ast f\rVert_2^2 \le \lVert f\ast f\rVert_\infty$" (the final inequality uses $\lVert f\ast f\rVert_1 = 1$).
	  **loc 4:** MO 2009, p. 11, Lemma 3.3 (the period-$u$ torus split applied to $f\ast f + f\circ f$; this is the statement MV 2010 cites as their Eq.(3)).
	  **quote 4:** "Let $f$ be a square-integrable pdf supported on $(-\tfrac14,\tfrac14)$, and let $K$ be a pdf supported on $(-\delta,\delta)$. Then $\int_{\mathbb{R}}(f\ast f(x) + f\circ f(x))K(x) dx = \frac{2}{u} + 2u^2\sum_{j\ne 0}(\Re\widetilde{f}(j))^2\Re\widetilde{K}(j).$"

- <a id="MV2009"></a>**[MV2009]** Matolcsi, Máté; Vinuesa, Carlos. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), no. 2, 439–447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379) — source of the $\ge 1.27481$ bound and of the dual framework underlying <a href="#PBV-master">[PBV-master]</a>.
	- <a id="MV-primitives"></a>**[MV-primitives]** Verbatim source for the same three Fourier-analytic primitives, as MV 2010 consolidates and sharpens them. MV 2010 explicitly attributes them to MO 2009 (p. 3): *"Lemma 3.1. [Lemmas 3.1, 3.2, 3.3, 3.4 in [6]]"*, where [6] is MO 2009.
	  **loc 1:** MV 2010 (arXiv:0907.1379v2), p. 3, Lemma 3.1, Eq.(3) (period-$u$ torus split for $f\ast f + f\circ f$, cited from MO 2009 Lemma 3.3).
	  **quote 1:** "$\int (f\ast f(x) + f\circ f(x))K(x) dx = \frac{2}{u} + 2u^2 \sum_{j\ne 0}(\Re\widetilde{f}(j))^2 \widetilde{K}(j).$"
	  **loc 2:** MV 2010, pp. 5–6, Lemma 3.3 (sharpened master inequality; the proof contains the constant-plus-tail split and the lattice $F$-bound used in the $z_1$-free derivation).
	  **quote 2 (statement, Eq.(9)):** "Using the notation $z_1 = \lvert\widehat{f}(1)\rvert$ and $k_1 = \widehat{K}(1)$, $\int (f\circ f(x))K(x) dx \le 1 + 2 z_1^2 k_1 + \sqrt{\lVert f\ast f\rVert_\infty - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2}.$"
	  **quote 2 (proof excerpt, p. 6):** "$\int (f\circ f(x))K(x) dx = \sum_{j\in\mathbb{Z}} \widehat{f\circ f}(j)\widehat{K}(j) = 1 + 2 z_1^2 k_1 + \sum_{j\ne 0,\pm 1}\lvert\widehat{f}(j)\rvert^2 \widehat{K}(j) \le 1 + 2 z_1^2 k_1 + \sqrt{\sum_{j\ne 0,\pm 1}\lvert\widehat{f}(j)\rvert^4}\sqrt{\sum_{j\ne 0,\pm 1}\widehat{K}(j)^2} = 1 + 2 z_1^2 k_1 + \sqrt{\lVert f\ast f\rVert_2^2 - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2} \le 1 + 2 z_1^2 k_1 + \sqrt{\lVert f\ast f\rVert_\infty - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2}.$"
	  **loc 3:** MV 2010, p. 7, Eq.(10) (master inequality in the assembled form used by the present work).
	  **quote 3:** "$\frac{2}{u} + a \le \lVert f\ast f\rVert_\infty + 1 + 2 z_1^2 k_1 + \sqrt{\lVert f\ast f\rVert_\infty - 1 - 2 z_1^4}\sqrt{0.5747/\delta - 1 - 2 k_1^2}.$"
	  **loc 4:** MV 2010 (arXiv:0907.1379v2), p. 3, Lemma 3.1, Eq.(4) (multiplier floor — lower bound on the period-$u$ Fourier energy of $f$ weighted by $\widetilde K$, in terms of the multiplier minimum $m_G$ and the QP denominator).
	  **quote 4 (verbatim):** "Let $G$ be an even, real-valued, $u$-periodic function that takes positive values on $[-1/4, 1/4]$, and satisfies $\widetilde G(0) = 0$. Then $u^{2} \sum_{j\ne 0} (\Re \widetilde f(j))^{2} \widetilde K(j) \ge \big(\min_{0\le x\le 1/4} G(x)\big)^{2} \cdot \big(\sum_{j: \widetilde G(j)\ne 0} \widetilde G(j)^{2} / \widetilde K(j)\big)^{-1}.$" This is precisely the `hEq4` field of `ExtremiserPrimitives` (Lean literal: `(uQ : ℝ) ^ 2 * S_cos ≥ m_G ^ 2 / S_G`) displayed under "The 'Eq.(4) floor'" above — the inequality is stated by MV in this squared form directly; no Sedrakyan/Engel step is interposed in MV's statement (the underlying lattice Cauchy–Schwarz step is, however, used in the manuscript's proof of Lemma 2.5(4) when *deriving* this floor from the atomic MO 2009 primitives, cf. `lower_bound_proof.tex` lines 675–682). The positivity hypothesis is on $[-1/4, 1/4]$ (matching the Lean `mv_inner_product_floor` hypothesis `∀ x ∈ Icc (-1/4) (1/4), m_G ≤ G x`), and via evenness of $G$ reduces to $[0, 1/4]$, which is exactly the interval over which the certifier's 32768-cell Taylor B&B in `delsarte_dual/grid_bound/G_min.py` proves $m_G \ge 998/1000$.

- <a id="MV-reduction"></a>**[MV-reduction]** Verbatim source for the $\mathcal{F}\to$ square-integrable reduction invoked by the present work, exactly as in MV 2010.
	  **loc:** MV 2010 (arXiv:0907.1379v2), p. 2, §2 ("Notation"), opening paragraph.
	  **quote:** "Let $\mathcal{F}$ denote the set of nonnegative real functions $f$ supported in $[-1/4,1/4]$ such that $\int f(x) dx = 1$. We define the autoconvolution of $f$, $f\ast f(x) = \int f(t)f(x-t) dt$ and its autocorrelation, $f\circ f(x) = \int f(t)f(x+t) dt$. We are interested in $S = \inf_{f\in\mathcal{F}}\lVert f\ast f\rVert_\infty$. **We remark here that the value of $S$ does not change if one considers nonnegative step functions in $\mathcal{F}$ only. This is proved in Theorem 1 in [4]. Therefore the reader may assume that $f$ is square integrable whenever this is needed.**" The "[4]" here is Schinzel–Schmidt 2002 <a href="#SS2002">[SS2002]</a>.

- <a id="SS2002"></a>**[SS2002]** Schinzel, Andrzej; Schmidt, Wolfgang M. *Comparison of $L^1$- and $L^\infty$-norms of squares of polynomials.* Acta Arithmetica **104** (2002), no. 3, 283–296. [DOI](https://doi.org/10.4064/aa104-3-4) — reference [4] of MV 2010. Theorem 1, jointly with the step-function construction in its §2, supplies the $L^{1}\to$step$\subset L^{2}$ reduction in MV §2 and inherited by the present work.
	  **loc:** SS 2002, Introduction (p. 283), Theorem 1; with the bridging identity stated at the top of p. 284 and the step-function construction proving assertion (ii) on p. 285 (also cited as "Theorem 1 in [4]" of MV 2010).
	  **quote 1 (Introduction, p. 283, verbatim).** "Let $\mathcal{P}(n)$ be the set of polynomials $P(X) = Q(X)^{2}$ where $Q$ is a nonzero polynomial of degree $< n$ with nonnegative real coefficients. We are interested in $A(n) = n^{-1} \sup_{P \in \mathcal{P}(n)} |P|_{1}/|P|_{\infty},$ where $|P|_{1}$ is the sum, and $|P|_{\infty}$ the maximum of the coefficients of $P$. Let $\mathcal{F}$ be the set of functions $f = g\ast g$ where $\ast$ denotes convolution and $g$ runs through nonnegative, not identically zero, integrable functions with support in $[0,1]$. Functions in $\mathcal{F}$ have support in $[0,2]$. We set $B = \sup_{f \in \mathcal{F}} |f|_{1}/|f|_{\infty}$ where $|f|_{1}$ is the $L^{1}$-norm and $|f|_{\infty}$ the sup norm of $f$."
	  **quote 2 (Theorem 1, p. 283, verbatim).** "**Theorem 1.** *For natural $n, l$,* (i) $A(n) \le A(nl),$ (ii) $A(n) \le B,$ (iii) $A(n) > B(1 - 6 n^{-1/3}).$"
	  **quote 3 (immediate consequence, p. 284, verbatim).** "It follows that $B = \lim_{n \to \infty} A(n) = \sup_{n} A(n).$"
	  **quote 4 (proof of assertion (ii), p. 285, verbatim).** "We now turn to (ii). Let $P \in \mathcal{P}(n)$ be given, say $P = Q^{2}$ with $Q = a_{0} + a_{1}X + \ldots + a_{n-1}X^{n-1}.$ Let $g$ be the function with support in $[0,1)$ having $g(x) = a_{i}$ for $i/n \le x < (i+1)/n$ $(i = 0, 1, \ldots, n-1),$ i.e., for $\lfloor nx\rfloor = i$. Then $|g|_{1} = n^{-1}|Q|_{1},$ so that $f = g\ast g$ has $|f|_{1} = n^{-2}|Q|_{1}^{2} = n^{-2}|P|_{1}.$"
	  **how this is the step-function reduction MV invokes.** The polynomial ratio $n^{-1}|P|_{1}/|P|_{\infty}$ is, by quote 4, exactly $|f|_{1}/|f|_{\infty}$ for $f = g\ast g$ the autoconvolution of the step function $g$ with heights $(a_{0}, \ldots, a_{n-1})$ on bins of width $1/n$. Hence $A(n)$ is the supremum of $|f|_{1}/|f|_{\infty}$ over autoconvolutions of $n$-bin nonnegative step functions, $B$ is the supremum over all autoconvolutions of nonnegative $L^{1}$ functions on $[0,1]$, and Theorem 1(ii)+(iii) together with the corollary $B = \lim A(n) = \sup_{n} A(n)$ assert that the sup over the $L^{1}$ cone is attained as a limit along nonnegative step functions. Translating from "sup of $|f|_{1}/|f|_{\infty}$ over $f = g\ast g$" to "inf of $\lVert f\ast f\rVert_{\infty}/(\int f)^{2}$ over $f \in \mathcal{F}$ in MV's normalisation" is a positive scaling: under MV's normalisation $\int g = 1$ on a translate of $[-1/4, 1/4]$, $|f|_{\infty} = \lVert g\ast g\rVert_{\infty}$ and $|f|_{1} = (\int g)^{2} = 1,$ so $|f|_{1}/|f|_{\infty} = 1/\lVert g\ast g\rVert_{\infty}$ and the SS sup over $|f|_{1}/|f|_{\infty}$ is the reciprocal of MV's inf over $\lVert g\ast g\rVert_{\infty}$. This is exactly the reduction MV §2 invokes: "*the value of $S$ does not change if one considers nonnegative step functions in $\mathcal{F}$ only. This is proved in Theorem 1 in [4].*" (see <a href="#MV-reduction">[MV-reduction]</a>). In the present work it is invoked in that same role — to justify reducing $\mathcal{F}$ to its $L^{2}$ subfamily (nonnegative step functions are bounded and compactly supported, hence in $L^{1}\cap L^{2}$) before applying the period-$u$ Parseval machinery.
	  **provenance:** verbatim text obtained from the IMPAN Open-Access PDF, *Acta Arithmetica* **104** (2002), no. 3, 283–296, DOI [10.4064/aa104-3-4](https://doi.org/10.4064/aa104-3-4); quotes 1–4 are from pp. 283–285 of that PDF. MSC 26C05 (polynomials) and 26D15 (inequalities for sums/integrals).

- <a id="CS2017"></a>**[CS2017]** Cloninger, Alexander; Steinerberger, Stefan. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), no. 8, 3191–3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — previous best *published* rigorous lower bound $1.28$; established by a finite branch-and-prune search over rational step-function configurations $B_{n,m}$ (CS 2017, Lemma 1 reduces the problem to step functions; the final disproof of counterexamples is computational).

- <a id="XX2026"></a>**[XX2026]** Xie, Xinyuan. *Unpublished improvement to the lower bound for $C_{1a}$ (claiming $C_{1a} \ge 1.2802$).* 2026. Listed on the canonical page as "Unpublished improvement, Grok".

- <a id="Watson"></a>**[Watson]** Watson, G. N. *A Treatise on the Theory of Bessel Functions.* 2nd ed., Cambridge University Press, 1944. §7.21 contains the classical envelope $\lvert J_0(z)\rvert \le \sqrt{2/(\pi z)}$ for $z>0$, equivalently $\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$. Used to control the $K_2$ tail past $\xi=10^{5}$ (`lower_bound_proof.tex` line 1072; the same attribution is recorded in `delsarte_dual/grid_bound_alt_kernel/kernels.py`).

- <a id="Flint"></a>**[Flint]** Johansson, Fredrik. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), no. 8, 1281–1292. [arblib.org](http://arblib.org/) — the interval-arithmetic library underlying every certified anchor.
