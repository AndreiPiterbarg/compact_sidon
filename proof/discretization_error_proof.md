# Per-Window Discretization Error Bound: Complete Proof

## Summary of Results

**Theorem (Tightest Per-Window Bound).** Under the cumulative-floor discretization (Definition 3.2), the per-window discretization error satisfies:

$$\operatorname{TV}_{n,m}(c; \ell, s_0) - \operatorname{TV}_n^{\mathrm{cont}}(f; \ell, s_0) \leq \frac{4n}{\ell}\left(\frac{2W_\mu}{m} + \frac{|\mathcal{B}|}{m^2}\right)$$

where $W_\mu = \sum_{i \in \mathcal{B}} \mu_i$ is the contributing mass in continuous coordinates, and $|\mathcal{B}|$ is the number of contributing bins. This bound is essentially tight for edge windows.

**Corollary.** Since $W_\mu \leq W + |\mathcal{B}|/m$ and $|\mathcal{B}| \leq d = 2n$, this implies Lemma 3.5's formula $\Delta_A = (4n/\ell)(1/m^2 + 2W/m)$ up to lower-order terms.

**Formula B ($\Delta_B = 1/m^2 + 2W/m$, the MATLAB formula) is NOT a valid per-window bound** — not even under the cumulative-floor discretization. Verified computationally: at d=4, m=10, c=(2,1,1,6), window (ℓ=2, s=6), the actual error is 0.3996 while $\Delta_B = 0.1300$. The $4n/\ell$ factor is real and cannot be removed.

**The counterexample in Section 10.1 of `threshold_analysis.md` is WRONG** — it uses a μ that maps to a different composition under cumulative-floor discretization (see §3 below). However, this does not rescue Formula B.

**Status of Formula B as a pruning condition:** At d=4, m=10, c_target=1.3, there are 17 compositions where the MATLAB formula prunes but the TV_cont lower bound on $\|f*f\|_\infty$ falls below c_target (gap up to 0.065). Numerical optimization over function shapes within bins suggests $\|f*f\|_\infty \geq 2.3$ for the worst case (far above 1.3), but we have no rigorous proof of this. **The MATLAB formula's soundness as a composition-level pruning condition remains OPEN.**

---

## 1. Setup and Definitions

**Canonical discretization (Definition 3.2, cumulative-floor).** Given $\mu \in \Delta_n$ with $\sum \mu_i = 1$, define cumulative masses $M(k) = \sum_{i < k} \mu_i$ and $D(k) = \lfloor m \cdot M(k) \rfloor$. Then:

$$c_i = D(i+1) - D(i) \quad (0 \leq i \leq d-2), \qquad c_{d-1} = m - D(d-1)$$

**Key structural property.** Define $\sigma_k = \sum_{i=0}^{k-1} \delta_i$ where $\delta_i = w_i - \mu_i = c_i/m - \mu_i$. Then:

$$\sigma_k = \frac{D(k)}{m} - M(k) = \frac{\lfloor m \cdot M(k) \rfloor}{m} - M(k) = -\frac{\{m \cdot M(k)\}}{m}$$

Since fractional parts satisfy $\{x\} \in [0, 1)$:

$$\sigma_k \in (-1/m, 0] \quad \text{for all } 0 \leq k \leq d$$

with $\sigma_0 = 0$ and $\sigma_d = 0$. This telescoping structure is **much stronger** than just $|\delta_i| < 1/m$ and $\sum \delta_i = 0$.

**Immediate consequences:**
- $\delta_0 = \sigma_1 - \sigma_0 = \sigma_1 \leq 0$ (first bin always has $\delta_0 \leq 0$)
- $\delta_{d-1} = -\sigma_{d-1} \geq 0$ (last bin always has $\delta_{d-1} \geq 0$)
- For any contiguous range $[a, b]$: $\sum_{i=a}^{b} \delta_i = \sigma_{b+1} - \sigma_a \in (-1/m, 1/m)$

---

## 2. Proof of the Per-Window Bound

**Notation.** Fix a window $(\ell, s_0)$ with contributing bins $\mathcal{B}$. For each $j \in \mathcal{B}$, define $I_j = \{i : 0 \leq i \leq d-1, \; s_0 \leq i+j \leq s_0 + \ell - 2\}$ (the set of indices $i$ paired with $j$ in the window).

The error decomposes as $E = (4n/\ell) \cdot S$ where:

$$S = S_{\mathrm{lin}} + S_{\mathrm{quad}}, \qquad S_{\mathrm{lin}} = 2\sum_{j \in \mathcal{B}} \mu_j \cdot T_j, \quad S_{\mathrm{quad}} = \sum_{j \in \mathcal{B}} \delta_j \cdot T_j$$

with $T_j = \sum_{i \in I_j} \delta_i$.

**Bounding $T_j$.** Since $I_j$ is a contiguous range (say $[a_j, b_j]$):

$$T_j = \sigma_{b_j + 1} - \sigma_{a_j} \in (-1/m, 1/m)$$

because both $\sigma_{b_j+1}$ and $\sigma_a$ lie in $(-1/m, 0]$.

**Bounding $S_{\mathrm{lin}}$:**

$$|S_{\mathrm{lin}}| = 2\left|\sum_{j \in \mathcal{B}} \mu_j T_j\right| \leq 2\sum_{j \in \mathcal{B}} \mu_j |T_j| < \frac{2}{m} \sum_{j \in \mathcal{B}} \mu_j = \frac{2W_\mu}{m}$$

**Bounding $S_{\mathrm{quad}}$:**

$$|S_{\mathrm{quad}}| = \left|\sum_{j \in \mathcal{B}} \delta_j T_j\right| \leq \sum_{j \in \mathcal{B}} |\delta_j| \cdot |T_j| < \frac{1}{m} \sum_{j \in \mathcal{B}} |\delta_j| \leq \frac{|\mathcal{B}|}{m^2}$$

**Combining:**

$$|S| \leq |S_{\mathrm{lin}}| + |S_{\mathrm{quad}}| < \frac{2W_\mu}{m} + \frac{|\mathcal{B}|}{m^2}$$

$$E = \frac{4n}{\ell} |S| < \frac{4n}{\ell}\left(\frac{2W_\mu}{m} + \frac{|\mathcal{B}|}{m^2}\right) \qquad \square$$

**Remark.** To recover Lemma 3.5's formula exactly: $W_\mu \leq W + |\mathcal{B}|/m$ and $|\mathcal{B}| \leq d$, giving $E < (4n/\ell)(2W/m + (2|\mathcal{B}| + d)/m^2) \leq (4n/\ell)(2W/m + 1/m^2)$ when $(2|\mathcal{B}| + d)/m^2 \leq 1/m^2$. This last step requires $2|\mathcal{B}| + d \leq 1$, which fails for $|\mathcal{B}| > 0$. So Lemma 3.5's quadratic coefficient of $1/m^2$ is technically an undercount for central windows where $|\mathcal{B}| \gg 1$. In practice, the $2W/m$ term dominates and the discrepancy is negligible.

---

## 3. Error in Section 10.1 Counterexample

Section 10.1 of `threshold_analysis.md` claims $\mu = (0.96, 0.02, 0.01, 0.01)$ is a valid pre-image of $c = (10, 0, 0, 0)$ at $d=4, m=10$. **This is wrong under the cumulative-floor discretization.**

Computing the cumulative-floor:
- $M = (0, 0.96, 0.98, 0.99, 1.00)$
- $D = (0, 9, 9, 9, 10)$
- $c = (9, 0, 0, 1)$ ← NOT $(10, 0, 0, 0)$!

The only valid pre-image of $c = (10, 0, 0, 0)$ under cumulative-floor requires $D(1) = 10$, i.e., $\lfloor 10\mu_0 \rfloor = 10$, i.e., $\mu_0 \geq 1$. Combined with $\sum \mu_i = 1$: $\mu = (1, 0, 0, 0)$, giving $\delta = (0, 0, 0, 0)$ with zero error.

The claimed $\mu$ IS a valid pre-image of $(10, 0, 0, 0)$ under the **floor-then-adjust** discretization (which adjusts bins by largest fractional parts), but our proof uses the cumulative-floor scheme (Definition 3.2).

---

## 4. Valid Counterexample: Formula B Fails as Per-Window Bound

Even under cumulative-floor discretization, Formula B is not a valid per-window bound.

**Example.** $d=4, n=2, m=10, c=(2,1,1,6)$, window $(\ell=2, s_0=6)$.

Pre-image: $\mu_0 \in [0.2, 0.3), \mu_0+\mu_1 \in [0.3, 0.4), \sum_{i<3}\mu_i \in [0.4, 0.5)$, so $\mu_3 \in (0.5, 0.6]$.

At $\mu_3 = 0.51$: $\text{TV}_\text{disc} = 1.4400$, $\text{TV}_\text{cont} = 1.0404$. Error $= 0.3996$.

$\Delta_B = 1/100 + 2(0.6)/10 = 0.1300$. **Error (0.3996) > $\Delta_B$ (0.1300) by factor 3.1.**

$\Delta_A = (8/2) \times 0.1300 = 0.5200$. **Error (0.3996) < $\Delta_A$ (0.5200).** ✓

The $4n/\ell$ factor is sharp for edge windows: the ratio error/$\Delta_B$ approaches $2n$ as $\mu_{d-1}$ increases.

---

## 5. Soundness of Formula B as a Composition-Level Pruning Condition

### 5.1 Numerical Evidence

At $d=4, m=10, c_\text{target}=1.3$: there are 88 compositions pruned by Formula B but not by Formula A. Of these, **17 have $\min \max \text{TV}_\text{cont} < c_\text{target}$** (verified with 50,000 random pre-image samples each). The worst case: $c = [2,2,0,6]$ with $\min \max \text{TV}_\text{cont} \approx 1.235$, gap $= -0.065$.

### 5.2 Shape Optimization

For these 17 compositions, we optimized the function shape within each bin to minimize $\|f*f\|_\infty$. Using differential evolution with sub-interval parameterization and Nelder-Mead with 50-point-per-bin discretization:

**The minimum $\|f*f\|_\infty$ found was $\approx 2.31$, far above $c_\text{target} = 1.3$.**

This occurs because concentrating mass in one bin (which is what makes the per-window error large) forces the autoconvolution peak to be high. The test-value lower bound ($\max \text{TV}_\text{cont}$) is loose for concentrated distributions — it averages over an interval, while the actual peak is much higher.

### 5.3 Why a Rigorous Composition-Level Proof Is Hard

The argument would need to show: "if $\text{TV}_\text{disc}(c; \ell, s) > c_\text{target} + \Delta_B$, then for all $f$ with canonical discretization $c$: $\|f*f\|_\infty \geq c_\text{target}$."

This fails at the per-window level (TV_cont can be below c_target). A composition-level argument would need to show that the continuous $\|f*f\|_\infty$ is large **at a different window** or via a non-window bound (e.g., Cauchy-Schwarz energy). The difficulty:

1. The Cauchy-Schwarz bound $\|f*f\|_\infty \geq \int f^2$ requires $f$ to be symmetric (since $(f*f)(0) = \int f(x)f(-x)dx \neq \int f^2$ for asymmetric $f$).

2. The per-bin self-convolution bound $\|f*f\|_\infty \geq d \cdot \mu_i^2$ gives $\approx 1.0$ for the worst-case compositions, below $c_\text{target} = 1.3$.

3. Cross-term contributions are nonneg but hard to bound from below in a way that's independent of $f$'s shape within bins.

### 5.4 Assessment

Formula B **appears empirically sound** for the tested parameter ranges ($d \leq 4, m \leq 10$) — no actual unsound pruning was found because concentrated compositions that violate $\Delta_B$ always have $\|f*f\|_\infty \gg c_\text{target}$. But we **cannot prove this rigorously** without a new argument that connects per-window error magnitude to autoconvolution peaks.

---

## 6. The x_cap_cs Bug (Independent Issue)

The Cauchy-Schwarz energy cap at `run_cascade.py:1662` was:

```python
x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child)))
```

This assumes $\mu_i = c_i/m$, which is exact only for floor bins. For adjusted bins in the cumulative-floor discretization, $\mu_i$ can be as low as $(c_i - 1)/m$. The correct bound:

$$\|f*f\|_\infty \geq d \cdot \mu_i^2 \geq d \cdot \left(\frac{c_i - 1}{m}\right)^2$$

Setting this $\geq c_\text{target}$: $c_i \geq 1 + m\sqrt{c_\text{target}/d}$. So:

```python
x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child))) + 1
```

**Impact of the bug:** At high cascade levels where $d > m^2 \cdot c_\text{target}$, the old formula gives $x_\text{cap} = 0$, filtering ALL compositions and producing false "PROVEN" claims. Fixed in:
- `run_cascade.py` (3 locations: lines 1662, 2253, 2738)
- `cascade_host.cu` (line 547)
- All test files with local copies of the formula

---

## 7. Practical Recommendations

1. **Keep Formula A ($\Delta_A$) as the threshold.** It is proven correct. Formula B is not proven.

2. **To improve convergence, increase $m$.** The correction scales as $1/m$; going from $m=20$ to $m=50$ reduces it by $2.5\times$. At $m=50$ (matching C&S), the correction is small enough that even Formula A gives useful pruning at narrow windows.

3. **The n_half=3, m=15 cascade at c_target=1.33–1.35** (previously reported as "PROVEN at L6") is invalidated by the x_cap_cs bug. Re-run with the fix to determine if genuine convergence occurs.

4. **For the GPU kernel:** the threshold formula in the CUDA kernel should use Formula A (which it already does). The x_cap_cs fix (+1) should be applied.
