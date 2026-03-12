# Prompt 1: Core Framework — Test Values and Thresholds

Prove Claims 1.1–1.4 of the autoconvolution lower bound proof. Attach `output.lean` as context — it contains all definitions and foundational lemmas already proved.

---

## Problem Context

We are proving $c \geq 1.4$ where:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

The proof works by discretization: approximate $f$ by step functions on a grid of $d = 2n$ bins of width $\Delta = 1/(4n)$, with integer masses $(c_0, \ldots, c_{d-1})$ summing to $m$.

### Key Definitions (already in output.lean)

- **Step function:** $f(x) = c_i/m$ on bin $i$, where bin $i = [i\Delta, (i+1)\Delta)$.
- **Rescaled coordinates:** $a_i = \frac{4n}{m} c_i$.
- **Discrete autoconvolution:** $\text{conv}[k] = \sum_{i+j=k} a_i a_j$ for $k = 0, \ldots, 2d-2$.
- **Test value:** $\text{TV}(\ell, s_{\text{lo}}) = \frac{1}{4n\ell} \sum_{k=s_{\text{lo}}}^{s_{\text{lo}} + \ell - 2} \text{conv}[k]$.
- **Max test value:** $b_{n,m}(c) = \max_{\ell \in \{2,\ldots,2d\}} \max_{s_{\text{lo}}} \text{TV}(\ell, s_{\text{lo}})$.
- **Contributing bins:** $\mathcal{B}(\ell, s_{\text{lo}}) = \{i \in [0, d-1] : \exists\, j \in [0, d-1],\; s_{\text{lo}} \leq i+j \leq s_{\text{lo}} + \ell - 2\}$.
- **Canonical discretization:** Floor-rounding of cumulative mass, already proved to sum to $m$ in output.lean.

### Already Proved in output.lean (available as context)

- All definitions formalized (autoconvolution_ratio, autoconvolution_constant, discrete_autoconvolution, test_value, max_test_value, is_composition, bin_masses, canonical_discretization, contributing_bins, canonical_cumulative_distribution)
- `canonical_cumulative_distribution_zero`: $D(0) = 0$
- `canonical_cumulative_distribution_2n`: $D(2n) = m$ when total mass $\neq 0$
- `bin_masses_nonneg`: bin masses of nonneg $f$ are nonneg
- `canonical_discretization_sum_zero_mass`: $\sum c_i = m$ when total mass $= 0$
- `canonical_cumulative_distribution_mono`: $D$ is monotone for nonneg $f$
- `sum_fin_telescope`: telescoping sum identity

---

## Claim 1.1: Test Value is a Lower Bound on $\|f*f\|_{L^\infty}$

**Theorem to prove:**

For any nonneg step function $f$ on the $d = 2n$ grid with integer masses $(c_0, \ldots, c_{d-1})$ summing to $m$ and $\int f = 1$:

$$b_{n,m}(c) \leq \|f * f\|_{L^\infty}$$

**Proof sketch:**

1. $(f * f)(x)$ is piecewise-linear on intervals of length $\Delta = 1/(4n)$, so for any interval $I$ of length $\ell\Delta$:
$$\|f*f\|_\infty \geq \frac{1}{|I|} \int_I (f*f)(x)\,dx$$

2. The integral of $(f*f)$ over $[s_{\text{lo}}\Delta,\, (s_{\text{lo}} + \ell - 1)\Delta]$ equals:
$$\int_I (f*f) = \frac{1}{(4n)^2 m^2} \cdot \Delta \sum_{k=s_{\text{lo}}}^{s_{\text{lo}}+\ell-2} \text{conv}[k] + \text{(boundary terms)}$$

3. The test value $\text{TV}(\ell, s_{\text{lo}})$ equals this average (or is a lower bound when boundary terms are dropped).

4. Maximizing over $(\ell, s_{\text{lo}})$ gives the tightest lower bound.

**Lean theorem statement suggestion:**

```lean
theorem test_value_le_Linfty_norm (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (max_test_value n m c : ℝ) ≤ autoconvolution_ratio (step_function n m c) := by
  sorry
```

where `step_function n m c` is the piecewise-constant function with height $c_i/m$ on bin $i$.

---

## Claim 1.2: Discretization Correction Term (Lemma 3 of CS14)

**Theorem to prove:**

For any nonneg $f$ supported on $(-1/4, 1/4)$ and its canonical discretization $\hat{c}$ on the $d = 2n$ grid with resolution $m$:

$$R(f) \geq b_{n,m}(\hat{c}) - \frac{2}{m} - \frac{1}{m^2}$$

**Proof sketch (from Cloninger-Steinerberger, Lemma 3):**

1. Let $f$ have total mass $\int f = 1$. Discretize: $c_i = \lfloor m \cdot \text{CDF}(i) \rfloor - \lfloor m \cdot \text{CDF}(i-1) \rfloor$.

2. The step function $\hat{f}$ with masses $c_i/m$ satisfies $\|\hat{f} - f\|_1 \leq d \cdot \frac{1}{2m}$ (each bin has rounding error $\leq 1/(2m)$ in mass... actually the error comes from the floor operation).

3. More precisely: $|c_i/m - \mu_i| \leq 1/m$ where $\mu_i = \int_{\text{bin}_i} f$. The total mass satisfies $\sum c_i/m = 1$ (proved in output.lean).

4. The correction: $\|f*f\|_\infty \geq \|\hat{f}*\hat{f}\|_\infty - 2\|f - \hat{f}\|_1 \cdot \|f\|_\infty \cdot \ldots$

   The key bound from CS14 is: $R(f) \geq b_{n,m}(\hat{c}) - 2/m - 1/m^2$.

**Lean theorem statement suggestion:**

```lean
theorem correction_term_bound (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    autoconvolution_ratio f ≥
      max_test_value n m (canonical_discretization f n m) - 2 / m - 1 / m ^ 2 := by
  sorry
```

---

## Claim 1.3: Dynamic Threshold is a Sound Refinement

**Theorem to prove:**

The per-window threshold $\text{thresh}(\ell, s_{\text{lo}}, c) = c_{\text{target}} + \frac{1}{m^2} + \frac{2W}{m}$ where $W = \frac{1}{m}\sum_{i \in \mathcal{B}} c_i$ is a **valid** (sound) pruning threshold. That is: if $\text{TV}(\ell, s_{\text{lo}}) > \text{thresh}(\ell, s_{\text{lo}}, c)$ for some window, then $R(f) \geq c_{\text{target}}$ for all $f$ whose discretization is $c$.

**Proof sketch:**

1. From Claim 1.2: $R(f) \geq b_{n,m}(c) - 2/m - 1/m^2$.

2. The correction $2/m + 1/m^2$ comes from $\|f - \hat{f}\|_1 \leq 1$ (total mass normalization). But for a specific window $(\ell, s_{\text{lo}})$, only the bins in $\mathcal{B}(\ell, s_{\text{lo}})$ contribute error. The contribution is bounded by $W = \frac{1}{m}\sum_{i \in \mathcal{B}} c_i \leq 1$.

3. The per-window correction is $\frac{2W}{m} + \frac{1}{m^2} \leq \frac{2}{m} + \frac{1}{m^2}$ (tighter when $W < 1$).

4. So if $\text{TV}(\ell, s_{\text{lo}}) > c_{\text{target}} + \frac{2W}{m} + \frac{1}{m^2}$ then $R(f) \geq c_{\text{target}}$.

**Lean theorem statement suggestion:**

```lean
theorem dynamic_threshold_sound (n m : ℕ) (c_target : ℝ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (h_tv : test_value n m c ℓ s_lo > c_target + 1/m^2 + 2 * (∑ i ∈ contributing_bins n ℓ s_lo, c i) / (m * m)) :
    ∀ f, (∀ x, 0 ≤ f x) → Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      canonical_discretization f n m = c → autoconvolution_ratio f ≥ c_target := by
  sorry
```

---

## Claim 1.4: Contributing Bins Formula

**Theorem to prove:**

Bin $i$ contributes to window $(\ell, s_{\text{lo}})$ iff $i \in [\max(0, s_{\text{lo}} - (d-1)),\; \min(d-1, s_{\text{lo}} + \ell - 2)]$.

Formally: for $i \in \{0, \ldots, d-1\}$ where $d = 2n$:

$$i \in \mathcal{B}(\ell, s_{\text{lo}}) \iff \max(0, s_{\text{lo}} - d + 1) \leq i \leq \min(d-1, s_{\text{lo}} + \ell - 2)$$

**Proof sketch:**

$i \in \mathcal{B}$ iff $\exists j \in [0, d-1]: s_{\text{lo}} \leq i + j \leq s_{\text{lo}} + \ell - 2$.

($\Rightarrow$) Given such $j$: $i \geq s_{\text{lo}} - j \geq s_{\text{lo}} - (d-1)$ and $i \leq s_{\text{lo}} + \ell - 2 - j \leq s_{\text{lo}} + \ell - 2$.

($\Leftarrow$) Given $\max(0, s_{\text{lo}} - d + 1) \leq i \leq \min(d-1, s_{\text{lo}} + \ell - 2)$:
Choose $j = \text{clamp}(s_{\text{lo}} - i, 0, d-1)$. Then $s_{\text{lo}} \leq i + j$ (from $j \geq s_{\text{lo}} - i$ when $s_{\text{lo}} - i \leq d - 1$) and $i + j \leq s_{\text{lo}} + \ell - 2$ (from $j \leq s_{\text{lo}} + \ell - 2 - i$ when $i \geq 0$).

**Lean theorem statement suggestion:**

```lean
theorem contributing_bins_iff (n : ℕ) (hn : n > 0) (ℓ s_lo : ℕ) (i : Fin (2 * n)) :
    i ∈ contributing_bins n ℓ s_lo ↔
      max 0 (s_lo - (2 * n - 1)) ≤ i.1 ∧ i.1 ≤ min (2 * n - 1) (s_lo + ℓ - 2) := by
  sorry
```
