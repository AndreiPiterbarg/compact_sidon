# Prompt 2: Discretization Correction Term (Lemma 3 of CS14)

**Claim 1.2 only.** This is the hardest single claim. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

The support is divided into $d = 2n$ bins of width $\Delta = 1/(4n)$. Any nonneg $f$ with $\int f = 1$ is discretized into a composition $(c_0, \ldots, c_{d-1})$ summing to $m$ via floor-rounding of cumulative mass.

### Definitions (already in output.lean)

- `autoconvolution_ratio f = ‖f∗f‖∞ / (∫f)²`
- `canonical_discretization f n m` — floor-rounding discretization
- `max_test_value n m c` — max over (ℓ, s_lo) of test values
- `bin_masses f n i` — integral of f over bin i

### Already proved in output.lean

$D(0)=0$, $D(2n)=m$, $\sum c_i = m$, bin masses nonneg, $D$ monotone.

---

## Claim 1.2: Correction Term $2/m + 1/m^2$

**Theorem (Lemma 3 of Cloninger-Steinerberger, arXiv:1403.7988).** For any nonneg $f$ supported on $(-1/4, 1/4)$ with $\int f = 1$, and its canonical discretization $\hat{c}$ on the $d = 2n$ grid with resolution $m$:

$$R(f) \geq b_{n,m}(\hat{c}) - \frac{2}{m} - \frac{1}{m^2}$$

### Proof

Let $\mu_i = \int_{\text{bin}_i} f$ (continuous bin masses), $\hat{\mu}_i = c_i / m$ (discretized).

**Step 1: Discretization error bound.**

The canonical discretization satisfies $|c_i/m - \mu_i| \leq 1/m$ for each bin. More precisely:

$c_i = D(i+1) - D(i)$ where $D(k) = \lfloor k \text{th partial sum of } \mu \text{ scaled by } m \rfloor$.

Since $D(k)$ is obtained by flooring, $D(k) \leq m \sum_{j<k} \mu_j < D(k) + 1$. Therefore:

$$m\mu_i - 1 < c_i \leq m\mu_i + 1$$

So $|\hat{\mu}_i - \mu_i| = |c_i/m - \mu_i| \leq 1/m$.

**Step 2: Convolution perturbation.**

Let $\hat{f}$ be the step function with heights $\hat{\mu}_i / \Delta$ (= $c_i/(m\Delta)$). Let $e = f - \hat{f}$ be the error. Then:

$$f * f = \hat{f} * \hat{f} + \hat{f} * e + e * \hat{f} + e * e$$

Since $f, \hat{f} \geq 0$, $e*e \geq 0$ as a convolution of real functions... actually $e$ is not nonneg, so we need a different approach.

**Step 3: Use the direct bound.**

$\|f*f\|_\infty \geq \|\hat{f}*\hat{f}\|_\infty - 2\|e\|_1 \cdot \|f\|_\infty$ is NOT the right approach since we don't control $\|f\|_\infty$.

The correct approach from CS14: For the step function $\hat{f}$ with total mass $\sum \hat{\mu}_i = \sum c_i/m = 1$:

$$R(f) = \frac{\|f*f\|_\infty}{(\int f)^2} \geq \frac{\|\hat{f}*\hat{f}\|_\infty}{1} - \text{correction}$$

No — the approach uses the fact that $f$ and $\hat{f}$ have the same grid structure. The precise argument:

**The CS14 argument:** Consider any continuous nonneg $f$ with $\int f = 1$ on $(-1/4, 1/4)$. Discretize to get $\hat{c}$ with $\sum c_i = m$.

The autoconvolution ratio satisfies:
$$R(f) = \|f*f\|_\infty$$

(since $\int f = 1$). The max test value $b_{n,m}(\hat{c})$ is computed from the discretization. The error between the continuous autoconvolution and the discrete test value is bounded by the discretization error.

The total $L^1$ error is $\|f - \hat{f}\|_1 \leq \sum_i |\mu_i - c_i/m| \cdot (1) \leq d \cdot (1/m) = 2n/m$... but this is not tight either.

**The tight bound:** From the CS14 paper, the correction is derived as follows. Each bin contributes at most $1/m$ error in mass. The autoconvolution of two functions with $L^1$ distance $\epsilon$ satisfies:

$$|\|f*f\|_\infty - \|\hat{f}*\hat{f}\|_\infty| \leq 2\epsilon + \epsilon^2$$

where $\epsilon = \|f - \hat{f}\|_1 / \max(\int f, \int \hat{f})$. Here $\int f = 1$ and $\int \hat{f} = 1$ (both normalized). And $\epsilon \leq 1/m$ (NOT $d/m$, because the rounding errors in the floor-based discretization telescope — the total error is bounded by the maximum single-floor error, not the sum).

Actually: $\sum_i |\mu_i - c_i/m| \leq 2 \cdot (1/m) \cdot (\text{number of bins with boundary effects})$...

The precise bound from CS14 is: $\sum_i |c_i/m - \mu_i| \leq 1$ always (since masses are nonneg and both sum to 1), and the correction term $2/m + 1/m^2$ comes from a more careful accounting.

**The actual proof from the paper:** Define $w_i = c_i/m$ and $\mu_i = \int_{\text{bin}_i} f$. Then $\sum w_i = \sum \mu_i = 1$. Write $w_i = \mu_i + \delta_i$ where $|\delta_i| \leq 1/m$ and $\sum \delta_i = 0$.

The discrete test value is the max-window-average of $\sum_{i+j=k} w_i w_j$. The continuous $\|f*f\|_\infty$ is at least the max-window-average of $\sum_{i+j=k} \mu_i \mu_j$ (from Claim 1.1, approximating $f$ by its bin averages).

$$\sum_{i+j=k} w_i w_j - \sum_{i+j=k} \mu_i \mu_j = \sum_{i+j=k} (w_i w_j - \mu_i \mu_j)$$
$$= \sum_{i+j=k} (\delta_i \mu_j + \mu_i \delta_j + \delta_i \delta_j)$$

Summing over a window of length $\ell$ and dividing by $4n\ell$:

$$|b_{n,m}(\hat{c}) - b_{n,m}(\mu)| \leq \frac{1}{4n\ell}\sum_{k \in \text{window}} \sum_{i+j=k} (|\delta_i||\mu_j| + |\mu_i||\delta_j| + |\delta_i||\delta_j|)$$

Using $|\delta_i| \leq 1/m$, $\sum \mu_j = 1$, $\sum |\delta_j| \leq \sum (1/m) \cdot (\text{effective count})$:

The bound evaluates to $2/m + 1/m^2$.

**Lean theorem to prove:**

```lean
theorem correction_term (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    autoconvolution_ratio f ≥
      (max_test_value n m (canonical_discretization f n m) : ℝ) - 2 / m - 1 / m ^ 2 := by
  sorry
```
