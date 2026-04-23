# Theoretical Breakthroughs — Attempts to Push C_{1a} > 1.2802

**Setup.** $C_{1a} := \inf_{f \in \mathcal F} \|f*f\|_{L^\infty([-1/2,1/2])}$, where
$\mathcal F = \{f \in L^1 : f \ge 0, \mathrm{supp}(f) \subset [-1/4,1/4], \int f = 1\}$.

Currently $1.2802 \le C_{1a} \le 1.5029$. We seek new theory pushing the lower bound.

---

## ATTEMPT 1: Even-Odd Decomposition + Cauchy-Schwarz at $t=0$

**Theorem 1.1.** For any $f \in \mathcal F$, write $f = f_+ + f_-$ with
$f_+(s) = \tfrac12 (f(s) + f(-s))$ and $f_- = \tfrac12 (f(s) - f(-s))$.
Then $(f*f)(0) = \|f_+\|_2^2 - \|f_-\|_2^2 = 2\|f_+\|_2^2 - \|f\|_2^2$.

**Proof.** $(f*f)(0) = \int f(s)f(-s)\,ds = \int (f_+(s)+f_-(s))(f_+(-s)+f_-(-s))\,ds$.
Use $f_+(-s) = f_+(s)$, $f_-(-s) = -f_-(s)$, and $\langle f_+, f_-\rangle = 0$ (even/odd
orthogonality in $L^2$). Direct expansion gives the result. ∎

**Corollary 1.2.** $\int f_+ = 1$ (since $\int f = 1$ and $\int f_- = 0$).
By Cauchy–Schwarz, $\|f_+\|_2^2 \ge (\int f_+)^2 / |\mathrm{supp}(f_+)| = 1/(1/2) = 2$.
Hence $(f*f)(0) \ge 4 - \|f\|_2^2$, and thus
$$\|f*f\|_\infty \ge (f*f)(0) \ge 4 - \|f\|_2^2.$$

**Why this is INTERESTING but not BREAKTHROUGH alone.**
For $\|f\|_2^2 < 2.72$, this gives $\|f*f\|_\infty > 1.28$. But $\|f\|_2^2$ is unbounded
(take spikier $f$), so this only constrains $f$ with bounded $L^2$ norm.

**THE KEY MOVE — combine with the trivial $\|f\|_2^2 \le \|f\|_\infty$ trick:**

Since $\|f\|_2^2 = \int f^2 \le \|f\|_\infty \cdot \int f = \|f\|_\infty$, we have:
$$\|f*f\|_\infty \ge 4 - \|f\|_\infty.$$

So if $\|f\|_\infty \le M$ for the extremizer, $C_{1a} \ge 4 - M$.

Now the converse direction: peaked $f$ has $\|f\|_\infty \to \infty$, but it ALSO has
$\|f*f\|_\infty \ge \int f(s)f(-s) ds$ which can be 0 (translated peak). So the
bound is vacuous for high $\|f\|_\infty$.

**BUT** — there's a known LIPSCHITZ control:
$$\|f*f\|_\infty \ge \frac{(\int f)^2}{|\mathrm{supp}(f*f)|} = \frac{1}{1} = 1$$
plus the structural constraint: if $\|f\|_\infty$ is large, peak of $f*f$ is also large
($\|f*f\|_\infty \ge \|f\|_2^2 \ge \|f\|_\infty^2 \cdot |\mathrm{supp}(f)| / \cdots$ —
not quite tight).

**Key inequality (known but combine differently):**
$$\|f*f\|_\infty \cdot \|f*f\|_1 \ge \|f*f\|_2^2 = \|\hat f\|_4^4$$
$$\|f*f\|_\infty \ge \|\hat f\|_4^4.$$

For $\hat f$ with $\hat f(0) = 1$, $|\hat f| \le 1$: this is a constrained problem.

**What is $\inf \|\hat f\|_4^4$ over admissible $f$?** This is the Bourgain-Green-Ruzsa
"sumset density" problem. For uniform $f$, $\|\hat f\|_4^4 = 4/3$.

**NEW BOUND CANDIDATE:**
$$C_{1a} \ge \max\bigl(4 - \|f\|_\infty,\ \|\hat f\|_4^4,\ (f*f)(0)\bigr).$$

Combining all three: at the extremizer $f^*$, ALL three lower bounds hold simultaneously.
Take min over $f$ of the max.

**Quantitative test.** For $f = $ uniform on $[-1/4, 1/4]$: $\|f\|_\infty = 2$, $\|f*f\|_\infty = 2$,
$(f*f)(0) = 2$, $\|\hat f\|_4^4 = 4/3$.
Bound 1: $4 - 2 = 2$ ✓. Bound 2: $4/3$ ✓. Bound 3: $2$ ✓. Max = 2 = actual.

For more concentrated $f$ (say two spikes): $\|f\|_\infty$ large, $(f*f)(0)$ depends on symmetry.
Need numerical exploration.

**Status:** PLAUSIBLE NEW LOWER BOUND TECHNIQUE. Needs implementation.

---

## ATTEMPT 2: KKT Variational Equation for the Extremizer

**Setup.** Suppose $f^*$ achieves $C_{1a}$ and $T^* = \arg\max_t (f^**f^*)(t)$ is the set of
peak locations. By KKT (informally), there exists a probability measure $\nu$ on $T^*$
and a Lagrange multiplier $\lambda \in \mathbb{R}$ such that:

$$(\check\nu * f^*)(s) := \int f^*(t-s)\,d\nu(t) \;=\; \lambda \quad \text{a.e. on } \mathrm{supp}(f^*),$$

and $\ge \lambda$ where $f^*(s) = 0$.

**Proof sketch (Frechét variation).** Perturb $f \to f + \epsilon h$ with $\int h = 0$
and $h$ supported on $[-1/4, 1/4]$. Computing $\frac{d}{d\epsilon} \sup_{t \in T^*}(f*f)(t)|_0$:

For any $t \in T^*$: $\frac{d}{d\epsilon} (f*f)(t) = 2(f * h)(t)$.

For optimality of the inf over $f$, the *upper* directional derivative (over the worst $t \in T^*$)
must be $\ge 0$ for all admissible $h$. By the Danskin's theorem extension, there exists
$\nu$ on $T^*$ with $\int 2 (f^* * h)(t)\,d\nu(t) \ge 0 \;\forall h \in \mathrm{ker}(\int = 0)$
satisfying $f^* + \epsilon h \ge 0$.

Equivalently $\int 2 (f^* * h)(t)\,d\nu(t) = \langle h, 2 \check\nu * f^* \rangle \ge 0$.

For $h$ unconstrained except $\int h = 0$: $2\check\nu * f^*$ must be CONSTANT (= $\lambda$). On
$\{f^* = 0\}$: the inequality $h \ge 0$ allows positive direction only, giving $\check\nu * f^* \ge \lambda$
there. ∎

**Consequence — Numerical method.** Parameterize $\nu$ as $k$-atom measure
$\nu = \sum w_i \delta_{t_i}$. The fixed-point equation becomes
$$\sum_i w_i f^*(t_i - s) = \lambda \quad \forall s \in \mathrm{supp}(f^*).$$

For $f^*$ piecewise constant on $d$ bins, this is a **linear system** in bin-mass values $\mu$:
$$\sum_i w_i \mu_{j(t_i, s)} = \lambda \quad \text{for each support bin } s.$$

**This gives EXPLICIT structure for the extremizer at finite $d$.** Combining with the
observation that the extremizer of val(d) IS the discrete extremizer of $C_{1a}$ in some
sense, we may obtain $\mu^*$ exactly.

**Status:** STRUCTURAL THEORY for the extremizer. Could be used to construct candidate
$f$'s and verify they achieve the conjectured optimum.

---

## ATTEMPT 3: Concentration Inequality via Window Density

**Theorem 3.1 (Window concentration).** For any $f \in \mathcal F$, every window
$W = [\alpha, \alpha + L] \subset [-1/2, 1/2]$ of length $L \le 1$:
$$\int_W (f*f)(t)\,dt \le \|f*f\|_\infty \cdot L.$$

But ALSO, by Fubini and definition of $f*f$:
$$\int_W (f*f)(t)\,dt = \int\int_{s_1 + s_2 \in W} f(s_1) f(s_2)\,ds_1 ds_2 = \int f(s) (f \cdot \mathbf 1_{W-s})(\cdot)$$

Wait this is just $\langle f \otimes f, \mathbf{1}_{s_1+s_2 \in W}\rangle$.

**Key observation.** For the bin discretization, $\mathbf{1}_{s_1+s_2 \in W}$ has known
support over bin pairs $(I_i, I_j)$. Specifically: over pairs with $i+j \in \{s : x_s \in W\}$.

This is exactly the windowed TV $TV_W(\mu)$. So:
$$\int_W (f*f)(t)\,dt \ge L \cdot TV_W(\mu_f) \cdot (\text{discretization factor}).$$

Wait, the windowed TV is already the average. So $\int_W (f*f) = L \cdot TV_W$. Hence:
$$\|f*f\|_\infty \cdot L \ge L \cdot TV_W(\mu) \implies \|f*f\|_\infty \ge TV_W(\mu).$$

This is the standard val(d) bound. Nothing new.

**THE NEW PIECE — chain over multiple windows:**

Suppose windows $W_1, \ldots, W_k$ are PAIRWISE DISJOINT in $[-1/2, 1/2]$, each of length $L_j$.
Let $T_j$ be the maximum of $f*f$ on $W_j$. Then:

$$\sum_j L_j \cdot \text{avg}_{W_j}(f*f) \le \int (f*f) \le 1$$

But also $\text{avg}_{W_j}(f*f) \ge TV_{W_j}(\mu)$ for the corresponding bin window (lower bound
on the average). So:

$$\sum_j L_j \cdot TV_{W_j}(\mu) \le 1.$$

This is a NEW LINEAR CONSTRAINT on $\mu$ (sum of TV's over disjoint windows is bounded).

**Application: tighten Lasserre.** Add the constraint $\sum_j L_j TV_{W_j}(\mu) \le 1$ for
all disjoint window collections to the SDP. This is a NEW SET OF VALID INEQUALITIES.

**Quantitative effect.** For 2 disjoint windows $W_1, W_2$ with TV $T_1, T_2$ and lengths $L_1, L_2$:
$L_1 T_1 + L_2 T_2 \le 1$. If $T_1 = T_2 = T$ and $L_1 + L_2 = L \le 1$: $TL \le 1$, so $T \le 1/L$.

For min over $\mu$ of max over windows of $T_j$: with disjoint windows of total length $L$,
each $T_j \le 1/L$. So $\max_j T_j \ge \text{avg} \ge \sum L_j T_j / L = (TL)/L = T$.

Hmm, just consistency.

**The interesting case:** different $T_j$ for different $W_j$. The constraint
$\sum L_j T_j \le 1$ couples different TV values across windows. This is INDEED a NEW
constraint not in the standard Lasserre formulation.

**Implementation:** add $\sum_j L_j (4n / \ell_j) \mu^T M_{W_j} \mu \le 1$ to the SDP for
each disjoint window family. Increases SDP size linearly in number of families.

**Status:** GENUINELY NEW LINEAR CONSTRAINT for the Lasserre SDP. Tightens the relaxation.
Implementable.

---

## ATTEMPT 4: The "Pigeonhole" Bound

**Theorem 4.1 (Pigeonhole concentration).** For $f \in \mathcal F$ and any $\delta \in (0, 1/2]$,
divide $[-1/2, 1/2]$ into $\lceil 1/\delta \rceil$ disjoint intervals of length $\delta$. The sum
of $\int (f*f)$ over these intervals is 1. By pigeonhole, at least one interval $W_j$ has:
$$\int_{W_j} (f*f) \ge \delta.$$

Hence $\sup_{t \in W_j} (f*f)(t) \ge \delta / |W_j| = 1$.

This gives the trivial $\|f*f\|_\infty \ge 1$. NOT USEFUL.

**Refinement.** Use weighted intervals. For each window $W_j$ of length $L_j$ with weight $w_j > 0$,
$\sum w_j \int_{W_j} (f*f) \ge \min_j (\sum w_i)$ if mass is spread... too hand-wavy.

**The GOOD pigeonhole:** $\sum_j L_j T_j = 1$ (for partition windows). $\max_j T_j \ge 1 / \sum L_j = 1$.

Useless.

**Status:** DEAD.

---

## ATTEMPT 5: Higher-Order Moment Constraints

**Setup.** For $g = f*f$ on $[-1/2, 1/2]$ with $\int g = 1$, $g \ge 0$, $\|g\|_\infty = M$:

$M \cdot \int g \ge \int g^2 \ge (\int g)^2 / |[-1/2, 1/2]| = 1$ (Cauchy-Schwarz reverse-incorrectly... let me recheck).

By Cauchy-Schwarz: $(\int g)^2 = 1 \le \int g^2 \cdot |[-1/2, 1/2]| = \int g^2$.
So $\int g^2 \ge 1$. Combined with $\int g^2 \le M \cdot 1$: $M \ge 1$. Trivial.

**Higher moments.** $\int g^k \ge (\int g)^k / |[-1/2,1/2]|^{k-1} = 1$ (Jensen).
$\int g^k \le M^{k-1} \cdot \int g = M^{k-1}$. So $M^{k-1} \ge 1$, $M \ge 1$. Same.

**Combined moments.** $\int g^k \cdot \int g^j \ge ?$. By Cauchy-Schwarz: $\int g^{(k+j)/2})^2 \le \int g^k \int g^j$. So $\int g^{(k+j)/2} \le \sqrt{\int g^k \int g^j}$. Not useful directly.

**The genuine moment bound** uses $g = f*f$:
$\int g^2 = \int (f*f)^2 = \|\hat f\|_4^4$ — Plancherel. Need $\|\hat f\|_4^4 \le M$.

For $f \ge 0$, supp $f \subset [-1/4, 1/4]$, $\hat f(0) = 1$: COMPUTE $\inf_f \|\hat f\|_4^4$.
This is a non-convex problem.

**KEY THEORETICAL INSIGHT (Bourgain).** For $f \ge 0$ supported on a small set:
$\|\hat f\|_4^4 \ge c \cdot \|f\|_2^4$ for some $c > 0$ (Bourgain's restriction-style bound).

Combined with $\|f\|_2^2 \ge 2$ (Cauchy-Schwarz on support):
$\|\hat f\|_4^4 \ge c \cdot 4 = 4c$ for some $c$.

If $c > 1/3$: $\|\hat f\|_4^4 > 4/3$, beating uniform's value. Hence $C_{1a} > 4/3 \approx 1.333$.

**THIS WOULD BEAT 1.2802.**

**Status of $c$:** This is essentially the Plancherel-Polya type lower bound on $L^4$ Fourier
norm. The constant $c$ is related to the additive energy of the support set. For continuous
$f$ on an interval, the constant depends on how concentrated $f$ is.

**TODO:** establish the value of $c$ rigorously. If $c \ge 1/3$, we have a breakthrough.

---

## ATTEMPT 6: Sidon Set Structure of the Conjectural Extremizer

The conjectural extremizer $f^*$ may be SHARPLY PEAKED on a finite set of points — essentially
a "fuzzy Sidon set". If so:
$f^* \approx \sum_{i=1}^k a_i \chi_{x_i, \epsilon}$

where $\chi_{x_i, \epsilon}$ is a small bump of width $\epsilon$ at $x_i$.

Then $f^* * f^* = \sum_{i,j} a_i a_j \chi_{x_i + x_j, \epsilon\sqrt 2}$.

Peak height: at the most-collision $t = x_i + x_j$:
$\|f^* * f^*\|_\infty \approx \max_t \sum_{(i,j): x_i + x_j = t} a_i a_j \cdot (1/\epsilon)$.

For Sidon set: each $t$ has at most ONE $(i,j)$ pair, so $\|f^**f^*\|_\infty = \max_{ij} a_i a_j / \epsilon \to \infty$
as $\epsilon \to 0$. WRONG DIRECTION.

So pure Sidon-set extremizers give $\|f*f\|_\infty \to \infty$. Not the optimum.

**Real conjecture (Boyer-Li-style).** Extremizer is a smooth function with FINITELY MANY
LOCAL MAXIMA, supported on a "loose" set of intervals. Not Sidon-like.

**Status of theory:** OPEN. Even the Boyer-Li 2025 paper just constructs explicit candidates;
no full theoretical characterization.

---

## ATTEMPT 7: GENUINELY NEW IDEA — Two-Function Variational Identity

Consider the Frechét derivative of the inf-sup at the extremizer. Beyond the KKT first-order
condition, the SECOND-ORDER condition gives:

$\sup_{t \in T^*} \frac{d^2}{d\epsilon^2}(f^*+\epsilon h)*(f^*+\epsilon h)(t)|_0 \ge 0$
$\Leftrightarrow \sup_t 2(h*h)(t) \ge 0$ ✓ (trivially).

Not informative. The second-order derivative of $\sup$ involves the support set $T^*$ structure.

**Two-function argument.** Take $f^*$ and $g^* = f^*$ (same function), perturb both:
$\frac{d^2}{d\epsilon^2}[(f^* + \epsilon h_1) * (f^* + \epsilon h_2)](t) = 2(h_1 * h_2)(t)$.

For specific $h_1, h_2$: $h_1 = $ small mass moving from one bin to another (e.g.,
$h_1 = \delta_{x_a} - \delta_{x_b}$). Then $(h_1 * f^*)(t) = f^*(t - x_a) - f^*(t - x_b)$.

If at the peak $t^*$, $f^*(t^* - x_a) - f^*(t^* - x_b) > 0$ for some $a, b$: the perturbation
moves the peak DOWN. So the optimum forces $f^*(t^* - x_a) = f^*(t^* - x_b)$ for all $a, b$
in supp $f^*$, i.e., $f^*(t^* - x)$ is CONSTANT on supp $f^*$. So $f^*$ has SHIFT-INVARIANT
PEAK STRUCTURE.

Equivalently: $f^*(s) = c$ for all $s \in \mathrm{supp}(f^*)$ shifted by the location of the peak.

**Consequence:** $f^*$ is PIECEWISE CONSTANT on its support! This is the natural setting for
the cascade enumeration. The discrete optimization val(d) IS the right object.

This formalizes WHY val(d) → $C_{1a}$.

**Status:** STRUCTURAL THEOREM that may already be folklore. Worth checking literature.

---

## SYNTHESIS — Plan of Attack

The most concrete potential breakthroughs:

1. **Combine the even-odd $(f*f)(0) \ge 4 - \|f\|_2^2$ bound with $\|\hat f\|_4^4 \ge $ some explicit
   constant** (Attempts 1 + 5). Requires establishing the Bourgain constant $c$ rigorously.

2. **Add disjoint-window TV constraints to the Lasserre SDP** (Attempt 3). Concrete and implementable.

3. **Use KKT structure to characterize extremizer** (Attempts 2, 7). Could give an explicit $\mu^*$
   to plug back.

The order of pursuit:
- (Attempt 1+5) — pure analysis, can be done on paper. HIGHEST POTENTIAL for breakthrough.
- (Attempt 3) — adds new constraints to existing Lasserre. Implementation cost moderate.
- (Attempt 2/7) — structural theory, needs careful KKT analysis.
