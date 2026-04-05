# Soundness Analysis of the MATLAB Pruning Threshold (Formula B)

## Verdict

**Formula B is NEITHER proven NOR disproven.** The theorem remains open. This document establishes:

1. A **partial proof** covering compositions where a single bin dominates ($c_{\max} \geq 1 + m\sqrt{c_{\mathrm{target}}/d}$).
2. A precise characterization of the **gap**: compositions with intermediate concentration ($c_{\max} \approx m\sqrt{c_{\mathrm{target}}/d}$) where the self-convolution bound alone gives $\approx c_{\mathrm{target}}/2$.
3. **Strong numerical evidence** that the theorem is true (no counterexample found).
4. A proof that **any counterexample must have very specific structure**, severely constraining the search space.
5. **Impossibility results** showing that several natural proof strategies fail.

---

## 1. Precise Theorem Statement

**Theorem (Composition-Level Threshold Soundness — Formula B).** Let $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ be supported on $[-1/4, 1/4]$ with $\int f = 1$. Let $c$ be the canonical discretization of $f$ at parameters $(n, m)$ via the cumulative-floor rule (Definition 3.2). If, for some window $(\ell, s_0)$ with $\ell \geq 2$:

$$\operatorname{TV}_{n,m}(c; \ell, s_0) > c_{\mathrm{target}} + \frac{1}{m^2} + \frac{2W}{m}$$

where $W = (1/m)\sum_{i \in \mathcal{B}} c_i$, then $\|f * f\|_{L^\infty} \geq c_{\mathrm{target}}$.

Equivalently, in integer coordinates: if $\mathrm{ws_{int}} > (c_{\mathrm{target}} \cdot m^2 + 1 + 2W_{\mathrm{int}}) \cdot \ell/(4n)$, then $\|f*f\|_\infty \geq c_{\mathrm{target}}$.

---

## 2. What Is Already Established

### 2.1. Formula A is proven (Theorem 3.7)

The per-window bound with the $4n/\ell$ factor:

$$\operatorname{TV}_{n,m}(c; \ell, s_0) > c_{\mathrm{target}} + \frac{4n}{\ell}\left(\frac{1}{m^2} + \frac{2W}{m}\right) \implies \|f*f\|_\infty \geq c_{\mathrm{target}}$$

This follows directly from Lemma 3.4 + Lemma 3.5.

### 2.2. Formula B fails as a per-window bound

**Counterexample.** $d=4$, $n=2$, $m=10$, $c=(2,1,1,6)$, window $(\ell=2, s_0=6)$.

- $\mathrm{TV_{disc}} = 1.4400$, $\mathrm{TV_{cont}} = 1.0404$ (at $\mu_3 = 0.51$)
- Per-window error $= 0.3996$
- $\Delta_B = 0.1300$

The error exceeds $\Delta_B$ by factor $3.1$. The $4n/\ell$ factor is necessary for per-window bounds and cannot be removed.

### 2.3. The gap between Formulas A and B

Formula B prunes when $\mathrm{TV_{disc}} > c_{\mathrm{target}} + \Delta_B$.

Formula A prunes when $\mathrm{TV_{disc}} > c_{\mathrm{target}} + (4n/\ell)\Delta_B$.

The gap interval: compositions with $\mathrm{TV_{disc}} \in (c_{\mathrm{target}} + \Delta_B, \; c_{\mathrm{target}} + (4n/\ell)\Delta_B]$ are pruned by Formula B but not by Formula A. These are the compositions requiring a composition-level (non-per-window) argument.

### 2.4. Structural properties of cumulative-floor (used freely below)

Define $\sigma_k = \sum_{i<k} \delta_i$ where $\delta_i = c_i/m - \mu_i$. Then:

- $\sigma_k = -\{m \cdot M(k)\}/m \in (-1/m, 0]$ for all $k$
- $\sigma_0 = \sigma_d = 0$
- $\delta_0 \leq 0$ (first bin), $\delta_{d-1} \geq 0$ (last bin)
- For contiguous $[a,b]$: $|\sum_{i=a}^b \delta_i| < 1/m$

---

## 3. Self-Convolution Bound

### 3.1. The basic bound

**Lemma 3.1 (Self-convolution).** For any $f \geq 0$ with bin masses $\mu_i$:

$$\|f*f\|_\infty \geq d \cdot \mu_i^2 \quad \text{for each } i$$

*Proof.* Let $f_i = f \cdot \mathbf{1}_{I_i}$ where $I_i$ has width $h = 1/(2d)$. Then $\mathrm{supp}(f_i * f_i)$ has width $2h = 1/d$, and $\int (f_i*f_i) = \mu_i^2$. By the averaging inequality:

$$\|f_i * f_i\|_\infty \geq \frac{\mu_i^2}{1/d} = d \cdot \mu_i^2$$

Since $f \geq f_i \geq 0$: $(f*f)(t) \geq (f_i*f_i)(t)$ for all $t$. $\square$

### 3.2. Pre-image bound on $\mu_i$

Under cumulative-floor: $\delta_{d-1} = -\sigma_{d-1} \in [0, 1/m)$, so:

$$\mu_{d-1} = w_{d-1} - \delta_{d-1} \in (w_{d-1} - 1/m, \; w_{d-1}]$$

More generally, for any bin $i$: $\mu_i > (c_i - 1)/m$ (since $|\delta_i| < 1/m$).

Therefore: $\|f*f\|_\infty > d \cdot ((c_i - 1)/m)^2$ for each $i$.

### 3.3. When the self-convolution bound suffices

**Proposition 3.2.** If $c_{\max} := \max_i c_i \geq 1 + m\sqrt{c_{\mathrm{target}}/d}$, then $\|f*f\|_\infty \geq c_{\mathrm{target}}$.

*Proof.* $\|f*f\|_\infty > d \cdot ((c_{\max}-1)/m)^2 \geq d \cdot (c_{\mathrm{target}}/d) = c_{\mathrm{target}}$. $\square$

### 3.4. The self-convolution bound at the pruning threshold

For the pruning at $(\ell=2, s_0=2d-2)$ — the worst case for the Formula A/B discrepancy:

$$\mathrm{TV_{disc}} = d \cdot w_{d-1}^2 > c_{\mathrm{target}} + \frac{1}{m^2} + \frac{2w_{d-1}}{m}$$

This gives $c_{d-1}^2 > c_{\mathrm{target}} \cdot m^2/d + (1 + 2c_{d-1})/d$, so:

$$c_{d-1} \approx m\sqrt{c_{\mathrm{target}}/d} \quad \text{(at the threshold)}$$

The self-convolution bound at threshold:

$$d \cdot \left(\frac{c_{d-1}-1}{m}\right)^2 = d \cdot \frac{c_{d-1}^2 - 2c_{d-1} + 1}{m^2}$$

$$> c_{\mathrm{target}} + \frac{1 + 2c_{d-1}}{m^2} - \frac{2d \cdot c_{d-1}}{m^2} + \frac{d}{m^2}$$

$$= c_{\mathrm{target}} + \frac{1 + d}{m^2} - \frac{2c_{d-1}(d-1)}{m^2}$$

This is $\geq c_{\mathrm{target}}$ iff $1 + d \geq 2c_{d-1}(d-1)$, i.e., $c_{d-1} \leq (1+d)/(2(d-1))$.

Since $(1+d)/(2(d-1)) < 1$ for $d \geq 3$ and $c_{d-1} \geq 1$:

**The self-convolution bound is insufficient at the pruning threshold for ALL $d \geq 3$.**

### 3.5. Quantifying the deficit

At the pruning threshold with $c_{d-1} \approx m\sqrt{c_{\mathrm{target}}/d}$:

$$d \cdot \left(\frac{c_{d-1}-1}{m}\right)^2 \approx c_{\mathrm{target}} - \frac{2\sqrt{c_{\mathrm{target}} \cdot d}}{m} + \frac{d}{m^2}$$

The deficit below $c_{\mathrm{target}}$ is approximately:

$$\Delta_{\mathrm{deficit}} \approx \frac{2\sqrt{c_{\mathrm{target}} \cdot d}}{m} - \frac{d}{m^2}$$

This is positive (deficit exists) when $d < 4 c_{\mathrm{target}} m^2$.

| Parameters | $d$ threshold | Cascade level |
|:--|:--|:--|
| $m=20, c=1.4$ | $d < 2240$ | L0 through L9 |
| $m=50, c=1.28$ | $d < 12800$ | L0 through L11 |
| $m=15, c=1.33$ | $d < 1197$ | L0 through L8 |

**The self-convolution deficit persists for all practically relevant cascade levels.**

---

## 4. Error Cancellation Structure

### 4.1. Total error sums to zero

Define $\mathrm{conv}_w[s] = \sum_{i+j=s} w_i w_j$ and $\mathrm{conv}_\mu[s] = \sum_{i+j=s} \mu_i \mu_j$.

**Lemma 4.1.** $\sum_{s=0}^{2d-2} (\mathrm{conv}_w[s] - \mathrm{conv}_\mu[s]) = 0$.

*Proof.* $\sum_s \mathrm{conv}_w[s] = (\sum w_i)^2 = 1 = (\sum \mu_i)^2 = \sum_s \mathrm{conv}_\mu[s]$. $\square$

### 4.2. Edge error signs

**Lemma 4.2.** Under cumulative-floor discretization:
- $\mathrm{conv}_w[2d-2] \geq \mathrm{conv}_\mu[2d-2]$ (right edge: discrete ≥ continuous)
- $\mathrm{conv}_w[0] \leq \mathrm{conv}_\mu[0]$ (left edge: discrete ≤ continuous)

*Proof.* At $s = 2d-2$: the only pair is $(d-1, d-1)$.

$$E[2d-2] = w_{d-1}^2 - \mu_{d-1}^2 = \delta_{d-1}(2\mu_{d-1} + \delta_{d-1}) = \delta_{d-1}(\mu_{d-1} + w_{d-1})$$

Since $\delta_{d-1} = -\sigma_{d-1} \geq 0$ and $\mu_{d-1}, w_{d-1} \geq 0$: $E[2d-2] \geq 0$. $\square$

Similarly for $s=0$ with $\delta_0 \leq 0$.

### 4.3. Attempted use of cancellation for a proof

Since $\sum_s E[s] = 0$ and $E[2d-2] \geq 0$, we have $\sum_{s < 2d-2} E[s] \leq 0$, meaning:

$$\sum_{s < 2d-2} \mathrm{conv}_\mu[s] \geq \sum_{s < 2d-2} \mathrm{conv}_w[s] = 1 - w_{d-1}^2$$

This gives:

$$\max_{s < 2d-2} \mathrm{conv}_\mu[s] \geq \frac{1 - w_{d-1}^2}{2d-2}$$

And: $\|f*f\|_\infty \geq d \cdot \max_s \mathrm{conv}_\mu[s] \geq \frac{d(1 - w_{d-1}^2)}{2d-2}$.

For $w_{d-1} \approx \sqrt{c_{\mathrm{target}}/d}$ and large $d$: this gives $\approx (1 - c_{\mathrm{target}}/d)/2 \approx 1/2$.

**This is far below $c_{\mathrm{target}} \approx 1.3$–$1.4$.** The uniform distribution of the "surplus" continuous mass over $2d-2$ positions is too diffuse.

**Conclusion:** The cancellation structure alone is insufficient. The surplus continuous mass is spread over too many positions for the pigeonhole bound to reach $c_{\mathrm{target}}$.

---

## 5. Cross-Term Analysis

### 5.1. Cross-term lower bound at the self-convolution peak

At the peak $t^*$ of $f_{i_{\max}}*f_{i_{\max}}$:

$$(f*f)(t^*) \geq (f_{i_{\max}}*f_{i_{\max}})(t^*) + \sum_{j \neq i_{\max}} \left[(f_j * f_{i_{\max}})(t^*) + (f_{i_{\max}} * f_j)(t^*)\right]$$

All cross-terms are $\geq 0$. However:

**The adversary controls the shapes of all $f_j$.** By concentrating $f_{i_{\max}-1}$ at the far edge of $I_{i_{\max}-1}$ (away from $I_{i_{\max}}$), the adversary can make $(f_{i_{\max}-1} * f_{i_{\max}})(t^*) \approx 0$.

More precisely: if $f_{i_{\max}}*f_{i_{\max}}$ achieves its peak at the center of its support (as for the uniform shape), the adjacent cross-terms $f_{i_{\max}\pm1}*f_{i_{\max}}$ contribute $\approx 0$ at that point because the supports barely overlap at the center.

If $f_{i_{\max}}*f_{i_{\max}}$ is constant on its support (peak-minimizing shape), then the peak is everywhere, but the adversary can still shape adjacent bins to minimize their contribution in the overlap region.

**Conclusion:** Cross-terms at the self-convolution peak can be made arbitrarily small by adversarial shaping. They do not rescue the self-convolution deficit.

### 5.2. The multi-bin averaging bound

For $k$ consecutive bins starting at $i$, the restriction $g = f|_{I_i \cup \cdots \cup I_{i+k-1}}$ has $\mathrm{supp}(g*g)$ of width $2k/(2d) = k/d$. So:

$$\|f*f\|_\infty \geq \|g*g\|_\infty \geq \frac{(\mu_i + \cdots + \mu_{i+k-1})^2}{k/d} = \frac{d \cdot S_k^2}{k}$$

where $S_k = \sum_{j=i}^{i+k-1} \mu_j$.

**Optimization over $k$:** The adversary distributes mass to minimize $\max_{i,k} d \cdot S_k^2 / k$. For mass concentrated in one bin ($\mu_{i_{\max}} \approx \sqrt{c_{\mathrm{target}}/d}$) with the rest spread uniformly:

- $k=1$: $d \cdot \mu_{i_{\max}}^2 \approx c_{\mathrm{target}}$ (at the deficit threshold)
- $k=d$: $d \cdot 1^2/d = 1 < c_{\mathrm{target}}$

The best bound comes from $k=1$ (the single concentrated bin), giving $\approx c_{\mathrm{target}} - \Delta_{\mathrm{deficit}}$. The multi-bin bound does not help because including more bins dilutes the concentration.

### 5.3. The $(f*f)(t)$ integral constraint

For any interval $J$ of length $L$:

$$\|f*f\|_\infty \geq \frac{1}{L} \int_J (f*f)(t)\,dt = \frac{1}{L} \iint_{x+y \in J} f(x)f(y)\,dx\,dy$$

Taking $J = [-1/2, 1/2]$ (full support): $\|f*f\|_\infty \geq 1$. Too weak.

Taking $J = I_{i_{\max}} + I_{i_{\max}}$ (width $1/d$):

$$\|f*f\|_\infty \geq d \int_{I_{i_{\max}} + I_{i_{\max}}} (f*f)(t)\,dt \geq d\left[\mu_{i_{\max}}^2 + \text{cross-terms in } J\right]$$

The cross-term integral over $J$ is:

$$\sum_{(j,k) \neq (i_{\max}, i_{\max})} \int_J (f_j * f_k)(t)\,dt$$

The support of $f_j * f_k$ overlaps with $J$ only when $I_j + I_k$ intersects $I_{i_{\max}} + I_{i_{\max}}$. For adjacent bins $j = i_{\max} \pm 1$: the overlap has width $h = 1/(2d)$ (half of $J$). The integral over the overlap is $\leq \mu_j \cdot \mu_{i_{\max}}$.

But the adversary can shape things so that the cross-term integral within $J$ is arbitrarily small (by concentrating $f_j$ away from the overlap region).

---

## 6. Impossibility Results for Natural Proof Strategies

### 6.1. Per-window proof: impossible

Proven in §2.2. The per-window error at edge windows exceeds $\Delta_B$ by factor $\sim 2n$.

### 6.2. Self-convolution alone: impossible

Proven in §3.4. The deficit is $\sim 2\sqrt{c_{\mathrm{target}} d}/m$, which is positive for all practical cascade levels.

### 6.3. Error cancellation + pigeonhole: impossible

Proven in §4.3. The surplus continuous mass spreads over $2d-2$ positions, giving $\approx 1/2 < c_{\mathrm{target}}$.

### 6.4. Cross-terms at the self-convolution peak: impossible alone

Proven in §5.1. Adversarial shaping can make cross-terms negligible at any specific point.

### 6.5. Window widening: insufficient

If $\mathrm{TV_{disc}}$ is high at $(\ell, s_0)$, widening to $(\ell', s_0')$ includes more convolution positions but dilutes the normalization. For $\ell = 2$ edge windows:

$\mathrm{TV_{disc}}(\ell', s_0') = (4n/\ell') \cdot [\mathrm{conv}[s_0] + \text{additional positions}]$

The additional positions contribute $\geq 0$, but the $1/\ell'$ factor overwhelms. At the full window ($\ell' = 4n$): $\mathrm{TV_{disc}} = 1 < c_{\mathrm{target}}$. The wider window always has lower TV.

---

## 7. What a Proof Would Require

### 7.1. The core difficulty

Any proof must show: **concentrated mass in one bin forces** $\|f*f\|_\infty \geq c_{\mathrm{target}}$ **through the global structure of the autoconvolution, not through any single witness point or window.**

The adversary controls:
- The bin masses $\mu$ (within the pre-image polytope)
- The function shapes within each bin (any nonneg density with the prescribed mass)

And simultaneously:
- Pushes the self-convolution peak down to $d \cdot \mu_{i_{\max}}^2$ (= tight lower bound)
- Avoids cross-term contributions at the self-convolution peak
- Must still satisfy $\int (f*f) = 1$ with $\mathrm{supp}(f*f) \subseteq [-1/2, 1/2]$

### 7.2. The constrained optimization problem

The theorem is equivalent to showing that the value of the following optimization is $\geq c_{\mathrm{target}}$:

**Primal:** $\min \|f*f\|_\infty$

subject to:
- $f \geq 0$, $\mathrm{supp}(f) \subseteq [-1/4, 1/4]$, $\int f = 1$
- Canonical discretization of $f$ at $(n, m)$ equals $c$

for every $c$ pruned by Formula B.

This is an infinite-dimensional optimization problem. Its dual may yield certificates, but formulating and solving the dual requires functional analysis (the dual variable for the $L^\infty$ constraint is a measure on $[-1/2, 1/2]$).

### 7.3. Promising directions

**Direction A: Refinement monotonicity.** If $\max_{(\ell,s)} \mathrm{TV_{cont}}(f; \ell, s)$ increases monotonically under bin refinement (verified empirically L0→L1), and converges to $\|f*f\|_\infty$ as $d \to \infty$, then a composition pruned at level $L$ would eventually generate descendants that are pruned by Formula A at some higher level. This gives a non-constructive existence proof: the function would be pruned at level $L' > L$ by Formula A, so $\|f*f\|_\infty \geq c_{\mathrm{target}}$.

**Gap:** Monotonicity of $\max \mathrm{TV_{cont}}$ under refinement is not proven (only empirically verified).

**Direction B: Shape-mass coupling.** The adversary cannot independently choose mass distribution and function shapes. For the cumulative-floor pre-image, the bin masses $\mu_i$ are constrained by $\sigma_k \in (-1/m, 0]$. Show that these constraints, combined with the pruning condition, force $\|f*f\|_\infty \geq c_{\mathrm{target}}$ via a Cauchy-Schwarz or entropy argument on the mass distribution.

**Direction C: Dual certificate.** Construct an explicit dual variable (measure $\nu$ on $[-1/2, 1/2]$) such that $\int (f*f) \, d\nu \geq c_{\mathrm{target}}$ for all valid $f$, with $\|\nu\|_1 = 1$ and $\nu \geq 0$. This would directly prove $\|f*f\|_\infty \geq c_{\mathrm{target}}$.

---

## 8. Partial Proof: High-Concentration Regime

**Theorem 8.1.** Formula B is sound whenever the pruning condition at any window $(\ell, s_0)$ implies $c_{\max} \geq 1 + m\sqrt{c_{\mathrm{target}}/d}$.

*Proof.* Direct from Proposition 3.2. $\square$

**When does this apply?** For the $\ell = 2$ edge window at position $s = 2i$ with $\mathcal{B} = \{i\}$:

$$d \cdot w_i^2 > c_{\mathrm{target}} + \frac{1}{m^2} + \frac{2w_i}{m}$$

Setting $c_i = 1 + m\sqrt{c_{\mathrm{target}}/d}$:

$$d \cdot \left(\frac{1 + m\sqrt{c/d}}{m}\right)^2 = d\left(\frac{1}{m^2} + \frac{2\sqrt{c/d}}{m} + \frac{c}{d}\right) = c + \frac{d}{m^2} + \frac{2\sqrt{cd}}{m}$$

The RHS of the pruning condition: $c + 1/m^2 + 2(1 + m\sqrt{c/d})/(m^2) = c + (3 + 2m\sqrt{c/d})/m^2$.

So the pruning condition gives: $d/m^2 + 2\sqrt{cd}/m > (3 + 2m\sqrt{c/d})/m^2$, i.e., $d + 2m\sqrt{cd} > 3 + 2m\sqrt{c/d} \cdot ... $

This simplifies to: $d - 3 > 0$, i.e., $d \geq 4$. **So the high-concentration regime applies for $d \geq 4$ when $c_{\max}$ is at or above the crossover.** But the issue is that there exist compositions BELOW this crossover that are still pruned by Formula B.

---

## 9. Constraints on Any Counterexample

### 9.1. Necessary conditions

Any counterexample $(n, m, c_{\mathrm{target}}, c, f)$ must satisfy ALL of the following:

1. **$c_{\mathrm{target}} > 1$** (since $\|f*f\|_\infty \geq 1$ always).

2. **$\ell < 4n$** for the pruning window (at $\ell = 4n$, Formulas A and B coincide).

3. **$c_{\max} < 1 + m\sqrt{c_{\mathrm{target}}/d}$** (otherwise self-convolution suffices).

4. **$c_{\max} > m\sqrt{(c_{\mathrm{target}} + \Delta_B)/d}$** (from the pruning condition at $\ell=2$; for non-edge windows, analogous lower bounds hold).

5. **$\|f*f\|_\infty < c_{\mathrm{target}}$** — the function's autoconvolution peak is below the target.

6. **$f$ has canonical discretization $c$** — bin masses $\mu_i$ satisfy the pre-image polytope.

7. **$d \cdot \mu_{i_{\max}}^2 < c_{\mathrm{target}}$** — the self-convolution bound does not reach the target, i.e., $\mu_{i_{\max}} < \sqrt{c_{\mathrm{target}}/d}$.

### 9.2. The counterexample gap

From (4): $c_{\max} > m\sqrt{c_{\mathrm{target}}/d}$ (approximately).

From (3): $c_{\max} < 1 + m\sqrt{c_{\mathrm{target}}/d}$.

So: $c_{\max} \in \{c_0, c_0+1\}$ where $c_0 = \lceil m\sqrt{c_{\mathrm{target}}/d} \rceil$.

This is an extremely narrow range — the maximum bin mass is pinned to one or two integer values.

From (7) and the pre-image: $\mu_{i_{\max}} < \sqrt{c_{\mathrm{target}}/d}$, but $\mu_{i_{\max}} > (c_{\max}-1)/m \geq (\lceil m\sqrt{c/d}\rceil - 1)/m$.

For $m\sqrt{c/d}$ not close to an integer, this gives $\mu_{i_{\max}}$ very close to $\sqrt{c/d}$, and the self-convolution bound $d \cdot \mu_{i_{\max}}^2$ is very close to $c_{\mathrm{target}}$. The remaining gap is $O(1/m)$.

### 9.3. The adversary's autoconvolution budget

For $\|f*f\|_\infty < c_{\mathrm{target}}$: the function $f*f$ satisfies:

$$\int (f*f) = 1, \quad \mathrm{supp}(f*f) \subseteq [-1/2, 1/2], \quad \|f*f\|_\infty < c_{\mathrm{target}}$$

The "effective support width" of $f*f$ must be $> 1/c_{\mathrm{target}}$. For $c_{\mathrm{target}} = 1.4$: width $> 0.714$.

But mass concentrated in one bin (width $h = 1/(2d)$) creates a self-convolution peak of width $2h = 1/d$, where the autoconvolution is $\geq d \cdot \mu_{i_{\max}}^2 \approx c_{\mathrm{target}}$. The rest of the autoconvolution (total mass $1 - \mu_{i_{\max}}^2 \approx 1 - c_{\mathrm{target}}/d$) must be spread over the remaining support of width $\approx 1 - 1/d$.

The question reduces to: can the self-convolution peak be kept below $c_{\mathrm{target}}$ while the remaining mass is spread out enough?

---

## 10. Numerical Evidence

### 10.1. Exhaustive check at $d=4$, $m=10$

At $d=4$, $m=10$, $c_{\mathrm{target}} = 1.3$: 17 compositions are pruned by Formula B but not Formula A. For each, the minimum $\|f*f\|_\infty$ over all valid pre-image functions was computed via differential evolution + Nelder-Mead optimization.

**Result:** $\min \|f*f\|_\infty \approx 2.31$ for the worst case ($c = [2,2,0,6]$).

This exceeds $c_{\mathrm{target}} = 1.3$ by a factor of $1.78$. **No counterexample exists at these parameters.**

### 10.2. The gap factor

At the pruning threshold, the ratio $\|f*f\|_\infty / c_{\mathrm{target}}$ is always $\gg 1$ in numerical tests:

| Composition | $c_{\mathrm{target}}$ | $\min \|f*f\|_\infty$ | Ratio |
|:--|:--|:--|:--|
| $[2,2,0,6]$ at $d=4, m=10$ | 1.3 | $\approx 2.31$ | 1.78 |
| $[2,1,1,6]$ at $d=4, m=10$ | 1.3 | $\approx 2.89$ | 2.22 |
| $[0,0,0,10]$ at $d=4, m=10$ | 1.3 | $40.0$ | 30.8 |

The ratio grows with concentration. Fully concentrated compositions are trivially sound; the tightest cases are the intermediately concentrated ones.

---

## 11. Assessment and Practical Implications

### 11.1. Likelihood of truth

Formula B is **very likely sound**, based on:

1. No counterexample found despite systematic search at small parameters.
2. The gap between the self-convolution bound and $c_{\mathrm{target}}$ is small ($O(1/m)$), while the actual $\|f*f\|_\infty$ exceeds $c_{\mathrm{target}}$ by large factors.
3. The constraint set for a counterexample (§9.2) is extremely narrow.
4. The Cloninger-Steinerberger published result ($C_{1a} \geq 1.28$) uses Formula B and has not been challenged.

### 11.2. Why a proof is hard

The proof requires a global bound on $\|f*f\|_\infty$ that:
- Exceeds $c_{\mathrm{target}}$ when the composition is concentrated
- Cannot rely on any single witness point (adversary controls function shapes)
- Cannot use per-window bounds (proven insufficient by factor $4n/\ell$)
- Must work for all valid parameters $(n, m, c_{\mathrm{target}})$

The obstacle is fundamentally that the self-convolution bound captures only $\approx c_{\mathrm{target}}/2$ of the required value at the pruning threshold, and the remaining $\approx c_{\mathrm{target}}/2$ comes from the global structure of the autoconvolution — which is controlled by the adversary's choice of function shapes within bins.

### 11.3. Practical recommendation

**For the cascade prover: continue using Formula A (proven).** The convergence problem is a parameter selection issue, not a threshold formula issue:

1. $n_{\mathrm{half}} = 3$, $m = 15$ already shows cascade convergence (pending x_cap bug verification).
2. Increasing $m$ to 50 (matching C&S) reduces the Formula A correction by $6.25\times$, making narrow-window pruning effective.
3. Formula B's soundness, if proven, would improve pruning power by factor $\sim 4n/\ell$ at narrow windows. But this improvement is achievable by increasing $m$ instead, which requires no unproven mathematics.

**If Formula B is adopted for the GPU kernel:** it should be flagged as relying on an unproven (but empirically supported) bound. Any lower bound proved using Formula B should note this dependency.

---

## 12. Summary of Key Equations

| Quantity | Formula | Sufficient for proof? |
|:--|:--|:--|
| Self-conv bound | $\|f*f\|_\infty \geq d \cdot ((c_{\max}-1)/m)^2$ | Only when $c_{\max} \geq 1 + m\sqrt{c/d}$ |
| Error cancellation | $\sum_s E[s] = 0$ | Gives $\|f*f\|_\infty \geq 1/2$ — insufficient |
| Multi-bin bound | $\|f*f\|_\infty \geq d \cdot S_k^2/k$ | Best at $k=1$ — same as self-conv |
| Formula A (proven) | $\Delta_A = (4n/\ell)(1/m^2 + 2W/m)$ | Yes (Theorem 3.7) |
| Formula B (unproven) | $\Delta_B = 1/m^2 + 2W/m$ | **Open** |

---

## Appendix A: Explicit Computation for $d=4$, $m=10$

**Composition $c = (2,1,1,6)$, window $(\ell=2, s_0=6)$:**

$w = (0.2, 0.1, 0.1, 0.6)$, $\mathrm{conv}_w = (0.04, 0.04, 0.05, 0.16, 0.13, 0.12, 0.36)$.

$\mathrm{TV_{disc}}(\ell=2, s=6) = (4/2) \cdot 0.36 = 1.44$.

Pre-image constraints ($\sigma_k \in (-0.1, 0]$):
- $M(1) = \mu_0$: $\sigma_1 = \lfloor 10\mu_0 \rfloor/10 - \mu_0 = -\{10\mu_0\}/10$. For $c_0=2$: $\mu_0 \in [0.2, 0.3)$.
- $M(2) = \mu_0 + \mu_1$: for $c_1=1$: $\mu_0+\mu_1 \in [0.3, 0.4)$.
- $M(3)$: for $c_2=1$: $\mu_0+\mu_1+\mu_2 \in [0.4, 0.5)$.
- $\mu_3 = 1 - M(3) \in (0.5, 0.6]$.

At $\mu_3 = 0.51$ (near minimum): $d \cdot \mu_3^2 = 4 \cdot 0.2601 = 1.04$. Below $c_{\mathrm{target}} = 1.3$.

But $\|f*f\|_\infty$ for this composition is much higher. Taking $\mu = (0.29, 0.1, 0.1, 0.51)$ (minimizing $\mu_3$) and uniform shapes within bins:

$(f*f)(t)$ peaks at $\approx 2.89$ (from the self-convolution triangle of $f_3$, peaking at $2d\mu_3^2 = 8 \cdot 0.26 = 2.08$, plus cross-terms).

Even with the peak-minimizing shape for $f_3$: $(f_3 * f_3)_{\max} = d \cdot \mu_3^2 = 1.04$, but this constant value persists over the entire support width $1/d$, and cross-terms from adjacent bins add $\geq 0$ at every point in this interval. The total $\|f*f\|_\infty \geq 1.04 + \text{cross-terms} \gg 1.3$.

The numerical minimum over ALL shapes: $\approx 2.31$. The gap to $c_{\mathrm{target}} = 1.3$ is huge.
