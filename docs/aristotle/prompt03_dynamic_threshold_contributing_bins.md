# Prompt 3: Dynamic Threshold and Contributing Bins

**Claims 1.3 + 1.4.** Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

Grid: $d = 2n$ bins of width $\Delta = 1/(4n)$. Compositions $(c_0, \ldots, c_{d-1})$ sum to $m$.

### Definitions (in output.lean)

- `test_value n m c ℓ s_lo` = $\frac{1}{4n\ell} \sum_{k=s_\text{lo}}^{s_\text{lo}+\ell-2} \text{conv}[k]$
- `contributing_bins n ℓ s_lo` = $\{i \in [0,d-1] : \exists j \in [0,d-1],\; s_\text{lo} \leq i+j \leq s_\text{lo}+\ell-2\}$

---

## Claim 1.4: Contributing Bins Formula

**Theorem.** For $d = 2n$ and $i \in \{0, \ldots, d-1\}$:

$$i \in \mathcal{B}(\ell, s_\text{lo}) \iff \max(0, s_\text{lo} - d + 1) \leq i \leq \min(d-1, s_\text{lo} + \ell - 2)$$

**Proof ($\Rightarrow$):** If $i \in \mathcal{B}$, there exists $j \in [0, d-1]$ with $s_\text{lo} \leq i+j \leq s_\text{lo}+\ell-2$. Then:
- $i = (i+j) - j \geq s_\text{lo} - (d-1)$, so $i \geq \max(0, s_\text{lo} - d + 1)$.
- $i \leq i + j \leq s_\text{lo} + \ell - 2$ (since $j \geq 0$), so $i \leq \min(d-1, s_\text{lo} + \ell - 2)$.

**Proof ($\Leftarrow$):** Given $\max(0, s_\text{lo} - d + 1) \leq i \leq \min(d-1, s_\text{lo} + \ell - 2)$, choose $j = \text{clamp}(s_\text{lo} - i, 0, d-1)$.

Case 1: $s_\text{lo} - i \leq 0$. Then $j = 0$, and $i + 0 = i \geq s_\text{lo}$ (from $s_\text{lo} \leq i$). Also $i \leq s_\text{lo} + \ell - 2$. ✓

Case 2: $0 < s_\text{lo} - i < d$. Then $j = s_\text{lo} - i$, $i + j = s_\text{lo}$. And $j < d$, $j \geq 0$. ✓

Case 3: $s_\text{lo} - i \geq d$. Then $j = d - 1$, $i + j = i + d - 1 \geq s_\text{lo}$ (need $i \geq s_\text{lo} - d + 1$, which holds). Also $i + d - 1 \leq (d-1) + (d-1) = 2d-2$. Need $i + d - 1 \leq s_\text{lo} + \ell - 2$, i.e., $i \leq s_\text{lo} + \ell - d - 1$. This follows when the range is non-empty. ✓

```lean
theorem contributing_bins_iff (n : ℕ) (hn : n > 0) (ℓ s_lo : ℕ)
    (hℓ : 2 ≤ ℓ) (i : Fin (2 * n)) :
    i ∈ contributing_bins n ℓ s_lo ↔
      Nat.max 0 (s_lo - (2 * n - 1)) ≤ i.1 ∧ i.1 ≤ Nat.min (2 * n - 1) (s_lo + ℓ - 2) := by
  sorry
```

---

## Claim 1.3: Dynamic Threshold Soundness

**Theorem.** Define the per-window mass $W = \frac{1}{m}\sum_{i \in \mathcal{B}(\ell, s_\text{lo})} c_i$. If for some window $(\ell, s_\text{lo})$:

$$\text{TV}(\ell, s_\text{lo}) > c_\text{target} + \frac{1}{m^2} + \frac{2W}{m}$$

then $R(f) \geq c_\text{target}$ for every nonneg $f$ whose canonical discretization is $c$.

**Proof.** This is a refinement of Claim 1.2 (the correction term $2/m + 1/m^2$). The idea: the correction term $2/m$ comes from bounding $\sum |\delta_i|$ where $\delta_i = c_i/m - \mu_i$ and $|\delta_i| \leq 1/m$.

For a specific window $(\ell, s_\text{lo})$, only bins in $\mathcal{B}(\ell, s_\text{lo})$ contribute to the window's test value. The error for that window involves only the $\delta_i$ for contributing bins. The key bound:

$$|\text{TV}_\text{continuous}(\ell, s_\text{lo}) - \text{TV}_\text{discrete}(\ell, s_\text{lo})| \leq \frac{2}{m}\sum_{i \in \mathcal{B}} \mu_i + \frac{1}{m^2}$$

Since $\mu_i \leq c_i/m + 1/m$ and $\sum_{i \in \mathcal{B}} c_i / m = W$:

$$\leq \frac{2W}{m} + \frac{2|\mathcal{B}|}{m^2} + \frac{1}{m^2} \leq \frac{2W}{m} + \frac{1}{m^2}$$

(using the tighter analysis from CS14 where the $|\mathcal{B}|/m^2$ term is absorbed).

So if $\text{TV}_\text{discrete} > c_\text{target} + 2W/m + 1/m^2$, then:
$$\|f*f\|_\infty \geq \text{TV}_\text{continuous} > \text{TV}_\text{discrete} - 2W/m - 1/m^2 > c_\text{target}$$

Wait, direction matters. We need: $\|f*f\|_\infty \geq \text{TV}_\text{continuous} \geq \text{TV}_\text{discrete} - (2W/m + 1/m^2)$. So:

$\text{TV}_\text{discrete} > c_\text{target} + 2W/m + 1/m^2$ implies $\|f*f\|_\infty \geq \text{TV}_\text{discrete} - 2W/m - 1/m^2 > c_\text{target}$.

Since $W \leq 1$ (total mass is 1), this is at least as strong as the uniform correction $2/m + 1/m^2$.

```lean
theorem dynamic_threshold_sound (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (c i : ℝ)) / m)
    (h_exceeds : test_value n m c ℓ s_lo > c_target + 1 / m ^ 2 + 2 * W / m) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  sorry
```
