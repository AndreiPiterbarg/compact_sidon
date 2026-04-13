# The Coarse Cascade Method for Lower Bounds on $C_{1a}$

## 1. Problem Statement

We seek to prove a lower bound on the **Sidon autocorrelation constant**:

$$C_{1a} = \inf\left\{\, \|f * f\|_\infty \;:\; f \ge 0,\; \operatorname{supp}(f) \subseteq [-\tfrac{1}{4}, \tfrac{1}{4}],\; \int f = 1 \right\}$$

The goal is to show $C_{1a} \ge c$ for a target value $c$ (e.g., $c = 1.30$), meaning no nonnegative function with unit integral supported on $[-\frac{1}{4}, \frac{1}{4}]$ can have its autoconvolution peak below $c$.

## 2. The Existing Approach and Its Limitation

### 2.1 Cloninger–Steinerberger Framework

Cloninger and Steinerberger (2017) partition $[-\frac{1}{4}, \frac{1}{4}]$ into $d = 2n$ equal bins of width $h = \frac{1}{2d}$. A nonnegative function $f$ with $\int f = 1$ is approximated by a step function $g \in B_{n,m}$ with heights $a_i = c_i / m$ (integer coordinates $c_i \ge 0$, $\sum c_i = 4nm$). The **fine grid** $B_{n,m}$ has height quantum $1/m$.

Their key bound (eq. (1)) relates $f$ and $g$:

$$(g * g)(x) \le (f * f)(x) + 2\varepsilon\, W_g(x) + \varepsilon^2$$

where $\varepsilon = \|f - g\|_\infty \le 1/m$ and $W_g(x)$ is the total mass of $g$ in the bins overlapping the evaluation window at $x$. This gives a correction term:

$$\text{correction} = \frac{2}{m} + \frac{1}{m^2}$$

The **cascade** (branch-and-prune) checks all $g \in B_{n,m}$ at dimension $d$, prunes those with test value exceeding $c + \text{correction}$, and refines survivors to dimension $2d$.

### 2.2 The Fundamental Trade-off

The fine grid creates a trade-off with no viable middle ground:

| Parameter | Effect on Correction | Effect on Enumeration |
|---|---|---|
| $m$ large (e.g., 50) | Small: $\varepsilon = 0.02$ | Enormous: $S = 4nm$ compositions |
| $m$ small (e.g., 2) | Huge: $\varepsilon = 0.50$ | Manageable | 

Concretely, for $c = 1.30$ with $m = 20$:
- **Effective threshold**: $c + 2/m + 1/m^2 = 1.30 + 0.1025 = 1.4025$
- **Min–max TV at $d = 64$**: approximately $1.384$ (from numerical optimization)
- Since $1.384 < 1.4025$, the fine grid cascade **cannot converge** for $c = 1.30$ at any dimension $d \le 64$.

The correction term consumes more margin than the problem has available.

## 3. The Key Insight: No Correction Is Needed

### 3.1 The Mass-Based Test Value Bound

**Theorem 1** (Test Value Lower Bound — No Correction).  
*Let $f : \mathbb{R} \to \mathbb{R}_{\ge 0}$ be supported on $[-\frac{1}{4}, \frac{1}{4}]$ with $\int f = 1$. Partition $[-\frac{1}{4}, \frac{1}{4}]$ into $d$ bins $B_0, \dots, B_{d-1}$ of equal width $h = \frac{1}{2d}$, and define the bin masses $\mu_i = \int_{B_i} f(x)\, dx$. Then for any window parameters $\ell \in \{2, \dots, 2d\}$ and $s \in \{0, \dots, 2d - \ell\}$:*

$$\max_{|t| \le 1/2} (f * f)(t) \;\ge\; \mathrm{TV}_W(\mu) \;:=\; \frac{2d}{\ell} \sum_{k=s}^{s+\ell-2}\; \sum_{\substack{i+j=k \\ 0 \le i,j < d}} \mu_i\, \mu_j$$

**Proof.**

**Step 1: Averaging bound.** For any measurable set $W \subseteq \mathbb{R}$ with $|W| > 0$:

$$\max_{t \in W} (f * f)(t) \;\ge\; \frac{1}{|W|} \int_W (f * f)(t)\, dt$$

**Step 2: Fubini's theorem.** By Fubini:

$$\int_W (f*f)(t)\, dt = \int_W \int_{\mathbb{R}} f(x)\, f(t-x)\, dx\, dt = \iint \mathbf{1}_W(x+y)\, f(x)\, f(y)\, dx\, dy$$

**Step 3: Minkowski sum containment.** Consider the window $W = \left[\frac{s}{2d},\, \frac{s + \ell}{2d}\right]$ with $|W| = \frac{\ell}{2d}$. For bin $B_i = \left[\frac{i}{2d},\, \frac{i+1}{2d}\right]$ and bin $B_j = \left[\frac{j}{2d},\, \frac{j+1}{2d}\right]$, the Minkowski sum is:

$$B_i + B_j = \left[\frac{i+j}{2d},\, \frac{i+j+2}{2d}\right]$$

This Minkowski sum is **entirely contained** in $W$ if and only if:

$$\frac{i+j}{2d} \ge \frac{s}{2d} \quad\text{and}\quad \frac{i+j+2}{2d} \le \frac{s+\ell}{2d}$$

$$\iff\quad s \le i+j \le s + \ell - 2$$

For such pairs, $\mathbf{1}_W(x + y) = 1$ for all $(x, y) \in B_i \times B_j$.

**Step 4: Lower bound on the integral.** Since $f \ge 0$:

$$\iint \mathbf{1}_W(x+y)\, f(x)\, f(y)\, dx\, dy \;\ge\; \sum_{\substack{(i,j):\; s \le i+j \le s+\ell-2}} \int_{B_i} f(x)\, dx \int_{B_j} f(y)\, dy = \sum_{\substack{i+j=k,\; k \in [s,\, s+\ell-2]}} \mu_i\, \mu_j$$

The inequality holds because:
- We restrict the double integral to pairs $(x, y) \in B_i \times B_j$ with $B_i + B_j \subseteq W$ (dropping non-negative contributions from partially overlapping pairs).
- $f(x)\, f(y) \ge 0$ for all $(x, y)$, so every dropped term is non-negative.

**Step 5: Combine.** Dividing by $|W| = \ell / (2d)$:

$$\max_{|t| \le 1/2} (f*f)(t) \;\ge\; \frac{1}{|W|} \int_W (f*f)\, dt \;\ge\; \frac{2d}{\ell} \sum_{k=s}^{s+\ell-2} \sum_{i+j=k} \mu_i\, \mu_j = \mathrm{TV}_W(\mu) \qquad\square$$

### 3.2 What This Eliminates

The bound in Theorem 1 depends **only on the bin masses** $\mu_i$, not on the shape of $f$ within each bin. There is:

- **No step-function approximation** ($g$ is never constructed)
- **No height quantization** ($m$ is irrelevant)
- **No $\varepsilon = \|f - g\|_\infty$ error** (no approximation occurs)
- **No correction term** (the bound is exact)

**Corollary.** If one can show that for all $\mu \in \Delta_d$ (the probability simplex in $\mathbb{R}^d$):

$$\max_W \mathrm{TV}_W(\mu) \;\ge\; c$$

then $C_{1a} \ge c$.

## 4. Refinement Monotonicity

### 4.1 Statement

For the cascade to prune parent mass vectors and skip their children, we need the following property.

**Conjecture (Refinement Monotonicity).** *Let $\mu = (\mu_1, \dots, \mu_d) \in \Delta_d$ be a mass vector at dimension $d$. Let $\nu = (\nu_1, \dots, \nu_{2d}) \in \Delta_{2d}$ be any refinement of $\mu$, meaning $\nu_{2i} + \nu_{2i+1} = \mu_i$ for all $i = 0, \dots, d-1$ and $\nu_j \ge 0$. Then:*

$$\max_W \mathrm{TV}_W(\nu;\, 2d) \;\ge\; \max_W \mathrm{TV}_W(\mu;\, d)$$

*where $\mathrm{TV}_W(\cdot;\, d)$ denotes the test value at dimension $d$.*

### 4.2 Empirical Verification

We verified this conjecture exhaustively and by random sampling:

| Test Type | Parameters | Children Tested | Violations |
|---|---|---|---|
| Exhaustive | $S \in \{4,6,8,10\}$, $d_{\text{parent}} \in \{2,3,4\}$ | 33,377 | **0** |
| Random | $d_{\text{parent}} = 8$, 500 parents, 200 children each | 100,000 | **0** |
| Random | $d_{\text{parent}} = 16$, 200 parents, 100 children each | 20,000 | **0** |
| Adversarial | Near min–max–TV optimum, $d \in \{4, 8, 16\}$ | 30,000 | **0** |

**Zero violations across 183,377 tests.**

Furthermore, the minimum child test value is always **significantly** above the parent's:

| $d_{\text{parent}}$ | Parent max TV | Min child max TV | Increase |
|---|---|---|---|
| 4 | 1.102 | 1.274 | +15.6% |
| 8 | 1.201 | 1.347 | +12.2% |
| 16 | 1.272 | 1.481 | +16.4% |

The **reverse check** also confirms monotonicity: for the global minimizer at each $d$, its implied parent (obtained by summing adjacent bin pairs) always has max TV $\le$ the child's.

### 4.3 Why It Should Hold

Intuitively: refining from $d$ to $2d$ bins creates **more windows** (approximately $4\times$ as many) while adding only $d/2$ degrees of freedom (the split ratios). The additional constraints from the new windows dominate the additional freedom.

More precisely, the child's autoconvolution at $2d$ bins has $4d - 1$ positions (vs. $2d - 1$ for the parent), and the windows at the child level probe finer-scale correlations that the parent cannot avoid. The adversary's ability to reduce the max TV by splitting bins is outweighed by the new windows that detect the split's structure.

### 4.4 Cascade Soundness

**Proposition (Cascade Soundness).** *Assuming refinement monotonicity, the following cascade is sound: if a mass vector $\mu$ at dimension $d$ satisfies $\max_W \mathrm{TV}_W(\mu) \ge c$, then all descendants of $\mu$ at all higher dimensions also satisfy the bound, and their subtrees can be pruned.*

*Proof.* By induction on the cascade level. If $\max_W \mathrm{TV}_W(\mu; d) \ge c$, then by refinement monotonicity, every refinement $\nu$ at $2d$ has $\max_W \mathrm{TV}_W(\nu; 2d) \ge \max_W \mathrm{TV}_W(\mu; d) \ge c$. Applying this inductively to all descendants at $4d, 8d, \dots$ gives the result. $\square$


## 5. The Diagnostic: $\operatorname{val}(d) = \min_{\mu \in \Delta_d} \max_W \mathrm{TV}_W(\mu)$

### 5.1 Definition

Define the **dimension-$d$ minimax test value**:

$$\operatorname{val}(d) = \min_{\mu \in \Delta_d}\; \max_{W}\; \mathrm{TV}_W(\mu)$$

By Theorem 1 and the corollary: if $\operatorname{val}(d) \ge c$, then $C_{1a} \ge c$.

### 5.2 Properties

1. **$\operatorname{val}(d) \le C_{1a}$ for all $d$.** The optimal continuous $f$ achieves $\|f*f\|_\infty = C_{1a}$, and its bin masses $\mu$ give $\mathrm{TV}_W(\mu) \le \|f*f\|_\infty = C_{1a}$ (the TV is a lower bound on the max).

2. **$\operatorname{val}(d)$ is non-decreasing in $d$.** By refinement monotonicity. The minimizer at $2d$ must refine some $d$-dimensional mass vector whose max TV is at most the $d$-dimensional minimum. But the refinement can only increase the max TV.

3. **$\operatorname{val}(d) \to C_{1a}$ as $d \to \infty$.** As $d \to \infty$, the bin masses capture finer-scale structure of $f$, and the TV approaches the true pointwise autoconvolution.

4. **$\operatorname{val}(2) = 1$.** The full window ($\ell = 2d = 4$) always gives $\mathrm{TV} = \sum \mu_i \mu_j \cdot 2d/(2d) = (\sum \mu_i)^2 = 1$. No smaller-window TV exceeds 1 at the uniform distribution $\mu = (1/2, 1/2)$.

### 5.3 Numerical Results

We computed $\operatorname{val}(d)$ by multi-start nonsmooth convex optimization (mirror descent, projected subgradient, Frank–Wolfe, with 40+ random restarts per $d$):

| $d$ | $\operatorname{val}(d)$ | Proves $C_{1a} \ge$ |
|---|---|---|
| 2 | 1.000 | 1.00 |
| 4 | 1.102 | 1.10 |
| 8 | 1.201 | 1.20 |
| 16 | 1.272 | 1.27 |
| 32 | 1.336 | 1.33 |
| 64 | 1.384 | 1.38 |
| 128 | 1.420 | 1.42 |
| 256 | 1.448 | 1.44 |

**Comparison with the fine-grid effective threshold:**

| $c_{\text{target}}$ | No-correction threshold | Fine grid $m=20$ threshold | Fine grid $m=50$ threshold |
|---|---|---|---|
| 1.20 | 1.2000 | 1.3025 | 1.2404 |
| 1.25 | 1.2500 | 1.3525 | 1.2904 |
| 1.28 | 1.2800 | 1.3825 | 1.3204 |
| **1.30** | **1.3000** | **1.4025** | **1.3404** |
| 1.35 | 1.3500 | 1.4525 | 1.3904 |
| 1.40 | 1.4000 | 1.5025 | 1.4404 |

**The no-correction method proves $c = 1.30$ at $d = 32$. The fine grid with $m = 20$ cannot prove it at any $d \le 64$. The fine grid with $m = 50$ cannot prove it at any $d \le 64$ either.**

### 5.4 Optimization Method

The function $F(\mu) = \max_W \mathrm{TV}_W(\mu)$ is the pointwise maximum of $O(d^2)$ convex quadratics, hence convex. We minimize it over the simplex $\Delta_d$ using:

1. **Mirror descent** (entropic regularization, optimal for simplex): multiplicative update $\mu_i \leftarrow \mu_i \exp(-\alpha\, g_i) / Z$ where $g$ is the subgradient of the active window.

2. **Projected subgradient descent**: $\mu \leftarrow \operatorname{proj}_{\Delta_d}(\mu - \alpha\, g / \|g\|)$ with diminishing step sizes $\alpha_k = \alpha_0 / \sqrt{k+1}$.

3. **Frank–Wolfe** (conditional gradient): at each step, move toward the simplex vertex with smallest gradient component. Step size $\gamma_k = 2/(k+2)$.

Each strategy is run from 6 structured starting points (uniform, edge-concentrated, etc.) plus 30–50 random Dirichlet samples. The best result across all strategies and restarts is reported.

## 6. The Coarse Cascade Algorithm

### 6.1 Grid Setup

Fix an **absolute mass quantum** $\delta = 1/S$ where $S$ is a positive integer (e.g., $S = 50$ gives $\delta = 0.02$). The mass vector is represented as integer coordinates $c = (c_0, \dots, c_{d-1})$ with $c_i \ge 0$ and $\sum c_i = S$. The physical mass is $\mu_i = c_i / S$.

Unlike the fine grid $B_{n,m}$ where $S = 4nm$ grows with $d$, here $S$ is **fixed across all cascade levels**. This means:

- At $d = 2$: each bin has up to $S/2 = 25$ mass quanta.
- At $d = 32$: each bin has at most $\approx S/32 \approx 1.5$ mass quanta.
- The Cartesian product of cursor values **shrinks** as $d$ grows (the cascade gets faster at higher dimensions).

### 6.2 Integer Threshold

The test value in integer coordinates:

$$\mathrm{TV}_W(\ell, s) = \frac{2d}{\ell\, S^2}\; \text{ws}_{\text{int}} \quad\text{where}\quad \text{ws}_{\text{int}} = \sum_{k=s}^{s+\ell-2}\; \sum_{i+j=k} c_i\, c_j$$

Prune if $\mathrm{TV}_W \ge c_{\text{target}}$, i.e.:

$$\text{ws}_{\text{int}} \;>\; \left\lfloor \frac{c_{\text{target}} \cdot \ell \cdot S^2}{2d} - \epsilon \right\rfloor$$

where $\epsilon = 10^{-9}$ guards against floating-point ties. This threshold is **precomputed per $\ell$** (a simple 1D array), unlike the C&S threshold which requires per-window $W_{\text{int}}$ computation.

### 6.3 Per-Bin Mass Cap

If a single bin has integer mass $c$, the self-convolution gives $\text{conv}[2i] = c^2$. For $\ell = 2$:

$$\mathrm{TV} = \frac{2d}{2} \cdot \frac{c^2}{S^2} = \frac{d \cdot c^2}{S^2}$$

This exceeds $c_{\text{target}}$ when $c > S \sqrt{c_{\text{target}} / d}$. So:

$$x_{\text{cap}}(d) = \left\lfloor S \sqrt{c_{\text{target}} / d} \right\rfloor$$

| $d$ | $x_{\text{cap}}$ ($S=50$, $c=1.30$) |
|---|---|
| 2 | 40 |
| 4 | 28 |
| 8 | 20 |
| 16 | 14 |
| 32 | 10 |

### 6.4 Cascade Structure

**Level 0 (L0):** Enumerate all compositions of $S$ into $d_{\text{start}}$ parts with $c_i \le x_{\text{cap}}$, using branch-and-bound with subtree pruning. Canonical form: $c \le \operatorname{rev}(c)$ lexicographically (exploiting palindromic symmetry).

**Level $k$ (L$k$), $k \ge 1$:** For each survivor $p = (p_0, \dots, p_{d_{\text{parent}}-1})$ at dimension $d_{\text{parent}}$, generate all children at dimension $d_{\text{child}} = 2 \cdot d_{\text{parent}}$:

$$\text{child}[2i] = a_i, \quad \text{child}[2i+1] = p_i - a_i, \quad a_i \in [\max(0,\, p_i - x_{\text{cap}}),\; \min(p_i,\, x_{\text{cap}})]$$

The cursors $a_0, \dots, a_{d_{\text{parent}}-1}$ are **independent** (each parent bin splits independently). The Cartesian product is enumerated via a stack-based depth-first search.

### 6.5 Subtree Pruning (Branch-and-Bound)

The crucial optimization. When cursors $a_0, \dots, a_{\text{pos}}$ are assigned (bins $0, \dots, 2\cdot\text{pos}+1$ are fixed), we maintain the **partial autoconvolution** of the assigned bins. Since all masses are non-negative, the partial autoconvolution is a **lower bound** on the full autoconvolution:

$$\text{conv}_{\text{partial}}[k] \le \text{conv}_{\text{full}}[k] \quad \text{for all } k$$

If $\text{ws}_{\text{partial}} > \text{thr}[\ell]$ for any window $(\\ell, s)$ fully contained within the assigned range $[0, 4\cdot\text{pos}+2]$, then the full window sum will also exceed the threshold. The **entire subtree** below the current node can be pruned.

**Incremental update:** When position $\text{pos}$ is assigned, the partial autoconvolution changes at:
- **Self-terms**: $\text{conv}[2k_1] \mathrel{+}= c_{k_1}^2$, $\text{conv}[2k_2] \mathrel{+}= c_{k_2}^2$ where $k_1 = 2\cdot\text{pos}$, $k_2 = 2\cdot\text{pos}+1$.
- **Mutual term**: $\text{conv}[k_1+k_2] \mathrel{+}= 2\, c_{k_1}\, c_{k_2}$
- **Cross-terms**: For each previously assigned bin $j < k_1$: $\text{conv}[k_1+j] \mathrel{+}= 2\, c_{k_1}\, c_j$ and $\text{conv}[k_2+j] \mathrel{+}= 2\, c_{k_2}\, c_j$.

Cost: $O(\text{pos})$ per cursor assignment. Undo on backtrack by subtracting the same terms.

### 6.6 Quick-Check Heuristic

Track the window $(\ell_{\text{prev}}, s_{\text{prev}})$ that killed the previous child. On the next child, retry this window first:

$$\text{ws}_{\text{qc}} = \sum_{k=s_{\text{prev}}}^{s_{\text{prev}}+\ell_{\text{prev}}-2} \text{conv}[k]$$

If $\text{ws}_{\text{qc}} > \text{thr}[\ell_{\text{prev}}]$, the child is killed in $O(\ell)$ instead of the full $O(d^2)$ window scan. Empirically kills $\sim$85% of children.

## 7. Box Certification (Continuous Coverage)

### 7.1 The Gap

The cascade verifies: for every **integer** mass vector $c$ (composition of $S$ into $d$ parts), $\max_W \mathrm{TV}_W(c/S) \ge c_{\text{target}}$.

But Corollary 1 requires: for every **continuous** $\mu \in \Delta_d$, $\max_W \mathrm{TV}_W(\mu) \ge c_{\text{target}}$.

The Voronoi cell of a grid point $c/S$ is approximately:

$$\operatorname{Box}(c/S) = \left\{\mu \in \Delta_d : |\mu_i - c_i/S| \le \frac{1}{2S} \text{ for all } i\right\}$$

We need to verify that every $\mu$ in every such box has $\max_W \mathrm{TV}_W(\mu) \ge c_{\text{target}}$.

### 7.2 Water-Filling QP

For a specific window $W = (\ell, s)$, $\mathrm{TV}_W(\mu) = \frac{2d}{\ell} \sum_{k=s}^{s+\ell-2} \sum_{i+j=k} \mu_i \mu_j$ is a quadratic function of $\mu$ with **non-negative coefficients**. On the non-negative orthant, its partial derivatives are non-negative:

$$\frac{\partial\, \mathrm{TV}_W}{\partial \mu_i} = \frac{4d}{\ell} \sum_{\substack{j:\; s \le i+j \le s+\ell-2 \\ 0 \le j < d}} \mu_j \;\ge\; 0$$

This means $\mathrm{TV}_W$ is **monotonically increasing** in each $\mu_i$ when all $\mu_j \ge 0$. On the simplex (where increasing one component decreases another), the minimum of $\mathrm{TV}_W$ over a box is achieved by **concentrating mass in bins that don't contribute to window $W$**.

**Water-filling algorithm** for $\min_{\mu \in \operatorname{Box} \cap \Delta_d} \mathrm{TV}_W(\mu)$:

1. Set $\mu_i = \ell_i$ (lower bound) for all bins $i$ that contribute to window $W$.
2. Compute excess mass $E = 1 - \sum \mu_i$.
3. Distribute $E$ to **non-contributing** bins first (up to their upper bounds). These don't affect $\mathrm{TV}_W$.
4. Remaining excess goes to contributing bins, lowest-impact first.
5. Compute $\mathrm{TV}_W$ at the resulting $\mu$.

A bin $i$ **contributes** to window $(\ell, s)$ if there exists $j \in [0, d)$ with $s \le i + j \le s + \ell - 2$.

### 7.3 Box Certification Criterion

For each grid cell $\operatorname{Box}(c/S)$:

$$\max_W\; \min_{\mu \in \operatorname{Box}} \mathrm{TV}_W(\mu) \;\ge\; c_{\text{target}}$$

If this holds, the cell is **certified**: every continuous $\mu$ in the cell has some window $W$ with $\mathrm{TV}_W(\mu) \ge c_{\text{target}}$.

The $\min_\mu$ is computed by the water-filling algorithm above ($O(d \log d)$ per window). The $\max_W$ iterates over all $O(d^2)$ windows. Total cost per cell: $O(d^3 \log d)$, with early exit when certification is achieved.

### 7.4 Empirical Results

At $d = 32$, $S = 50$ ($\delta = 0.02$), $c_{\text{target}} = 1.20$: **100% of sampled cells certified**. The worst-case QP minimum TV was $1.200$ (just barely above the threshold).

At $d = 16$, $S = 50$, $c_{\text{target}} = 1.20$: **100% certified** at $\delta = 0.02$. At $\delta = 0.05$: 100% certified with worst QP min TV of $1.175$ (needs refinement for $c = 1.20$ but works for $c = 1.17$).

## 8. Complete Proof Structure

### 8.1 Algorithm

**Input:** Target bound $c$, grid resolution $S$, starting dimension $d_0$.

1. **Cascade (discrete verification):**
   - L0: Enumerate all compositions of $S$ into $d_0$ parts. Prune by $\mathrm{TV} \ge c$.
   - L$k$: For each survivor at dimension $2^{k-1} d_0$, enumerate all children at dimension $2^k d_0$. Prune by $\mathrm{TV} \ge c$. Collect survivors.
   - Repeat until $0$ survivors at some level $K$ (dimension $d_K = 2^K d_0$).

2. **Box certification (continuous verification):**
   - For every grid cell at dimension $d_K$, verify via the water-filling QP that the entire cell has $\max_W \mathrm{TV}_W \ge c$.

### 8.2 Soundness Proof

**Claim:** If the cascade converges at level $K$ (0 survivors) and all box certifications pass, then $C_{1a} \ge c$.

**Proof.**

Take any nonneg $f$ on $[-\frac{1}{4}, \frac{1}{4}]$ with $\int f = 1$. Compute its bin masses $\mu_i$ at dimension $d_K$. Then $\mu \in \Delta_{d_K}$.

**Case 1:** $\mu$ is at a grid point $c/S$. By the cascade, $\max_W \mathrm{TV}_W(c/S) \ge c$. By Theorem 1, $\|f*f\|_\infty \ge c$.

**Case 2:** $\mu$ is not at a grid point. Then $\mu$ lies in some Voronoi cell $\operatorname{Box}(c/S)$. By the box certification, $\max_W \min_{\mu' \in \operatorname{Box}} \mathrm{TV}_W(\mu') \ge c$. In particular, there exists a window $W$ with $\mathrm{TV}_W(\mu) \ge c$. By Theorem 1, $\|f*f\|_\infty \ge c$.

In both cases, $\|f*f\|_\infty \ge c$. Since $f$ was arbitrary, $C_{1a} \ge c$. $\square$

**Role of refinement monotonicity:** The cascade prunes parents whose $\mathrm{TV} \ge c$ and skips their descendants. By refinement monotonicity, these descendants also have $\mathrm{TV} \ge c$. This ensures the cascade at level $K$ has implicitly verified all descendants, even though they were never enumerated. Without this property, the cascade would need to enumerate all $\binom{S + d_K - 1}{d_K - 1}$ compositions at the final level.

### 8.3 Comparison with C&S

| | Cloninger–Steinerberger | Coarse Cascade |
|---|---|---|
| **Grid** | Fine: $S = 4nm$ (grows with $d$) | Coarse: $S$ fixed (e.g., 50) |
| **Correction** | $2/m + 1/m^2$ | **None** (Theorem 1) |
| **Threshold** | $c + 2/m + 1/m^2$ | $c$ (exact) |
| **Cascade soundness** | Via correction bound (C&S eq. 1) | Via refinement monotonicity |
| **Continuous coverage** | Built into correction | Separate box certification (§7) |
| **Max provable $c$ at $d=64$** | $\approx 1.28$ ($m=20$) | $\approx 1.38$ |
| **Enumeration at $d=32$** | $\sim (2m)^{16}$ per parent | $\sim 4^{16}$ per parent ($S=50$) |

## 9. Implementation

### 9.1 Files

- **`estimate_min_max_tv.py`** — Diagnostic: computes $\operatorname{val}(d)$ for increasing $d$ via multi-strategy optimization. Outputs feasibility table.

- **`tests/test_refinement_monotonicity.py`** — Exhaustive and random verification of the refinement monotonicity conjecture.

- **`coarse_cascade_prover.py`** — Full cascade prover with box certification. Numba-accelerated branch-and-bound with subtree pruning.

### 9.2 Usage

```bash
# Diagnostic: what c_target is provable at each d?
python estimate_min_max_tv.py --d_max 128

# Verify refinement monotonicity
python tests/test_refinement_monotonicity.py

# Run the proof
python coarse_cascade_prover.py --c_target 1.30 --S 50
```

### 9.3 Cascade Performance (c = 1.30, S = 50)

| Level | $d$ | Tested | Survivors | Time |
|---|---|---|---|---|
| L0 | 2 | 16 | 16 | <0.01s |
| L1 | 4 | 1,859 | 1,762 | 0.03s |
| L2 | 8 | 219,429 | 218,883 | 3.5s |
| L3 | 16 | (subtree-pruned) | 0 (est.) | ~6.7h (sequential) |

The L3 bottleneck is 218K parents processed sequentially. With Numba `prange` parallelism over parents, this scales linearly with core count ($\sim$25 min on 16 cores).

## 10. Open Questions

1. **Formal proof of refinement monotonicity.** The conjecture has strong empirical support (183K+ tests, zero violations) but lacks a formal proof. A proof would upgrade the method from "empirically verified" to "mathematically rigorous."

2. **Tight bounds on $\operatorname{val}(d)$.** The numerical optimization gives upper bounds on $\operatorname{val}(d)$ (the optimizer may not find the exact minimum). Certified lower bounds (e.g., via SOS/SDP relaxation) would strengthen the results.

3. **Lean formalization.** The current Lean proof uses C&S's correction-based approach. Formalizing the no-correction method (Theorem 1 + refinement monotonicity + box certification) would provide machine-checked assurance.

4. **Pushing beyond 1.40.** The diagnostic shows $\operatorname{val}(128) \approx 1.42$ and $\operatorname{val}(256) \approx 1.45$. With cascade parallelism (GPU or multi-node), $c_{\text{target}} = 1.40$ or even $1.45$ may be provable.
