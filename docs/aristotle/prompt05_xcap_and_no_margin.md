# Prompt 5: Single-Bin x_cap and Asymmetry No-Margin

**Claims 2.2 + 2.3.** Both use the "restrict to a sub-region" argument. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

Grid: $d = 2n$ bins of width $\Delta = 1/(4n)$ on $(-1/4, 1/4)$. Bin $i$ covers $[-1/4 + i\Delta,\; -1/4 + (i+1)\Delta)$.

### Definitions (in output.lean)

- `bin_masses f n i` — integral of $f$ over bin $i$
- `canonical_discretization f n m` — floor-rounding discretization
- `autoconvolution_ratio f` = $\|f*f\|_{L^\infty} / (\int f)^2$

---

## Claim 2.3: Single-Bin Energy Cap (x_cap)

**Theorem.** For any nonneg $f$ supported on $(-1/4, 1/4)$ with $\int f = 1$, and any bin $i$ with mass $M_i = \int_{\text{bin}_i} f$:

$$\|f * f\|_{L^\infty} \geq d \cdot M_i^2$$

where $d = 2n$.

**Proof.**

1. Define $f_i(x) = f(x) \cdot \mathbf{1}_{\text{bin}_i}(x)$. Since $f \geq 0$: $f \geq f_i \geq 0$ pointwise.

2. Convolution monotonicity (same as Claim 2.1): $\|f*f\|_\infty \geq \|f_i * f_i\|_\infty$.

3. $\operatorname{supp}(f_i) \subseteq \text{bin}_i$ which has width $\Delta = 1/(4n)$.
   $\operatorname{supp}(f_i * f_i) \subseteq \{x+y : x, y \in \text{bin}_i\}$ which has length $2\Delta = 1/(2n)$.

4. By Fubini: $\int (f_i * f_i) = (\int f_i)^2 = M_i^2$.

5. By averaging principle: $\|f_i * f_i\|_\infty \geq M_i^2 / (2\Delta) = M_i^2 \cdot 2n = d \cdot M_i^2$.

**Consequence.** Set $M_i = c_i / m$ (for the discretized step function). Then $\|f*f\|_\infty \geq d \cdot c_i^2/m^2$. This exceeds $c_\text{target}$ when $c_i \geq m\sqrt{c_\text{target}/d}$, giving $x_\text{cap} = \lfloor m\sqrt{c_\text{target}/d} \rfloor$.

**Key point:** This is a DIRECT $L^\infty$ bound on the continuous function. No discretization correction $2/m + 1/m^2$ is needed.

```lean
theorem single_bin_energy_cap (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (i : Fin (2 * n)) :
    let M_i := bin_masses f n i
    autoconvolution_ratio f ≥ (2 * n : ℝ) * M_i ^ 2 := by
  sorry
```

---

## Claim 2.2: Asymmetry Margin is Unnecessary

The asymmetry pruning (Claim 2.1, proved in a separate prompt) uses threshold $L \geq \sqrt{c_\text{target}/2}$. An older version of the code added a safety margin of $1/(4m)$ to this comparison. This claim proves the margin is unnecessary.

### Fact 1: Discrete left_frac equals continuous left-half mass exactly

**Theorem.** For a step function on the $d = 2n$ grid:

$$\text{left\_frac} := \frac{1}{m}\sum_{i=0}^{n-1} c_i = \int_{-1/4}^{0} f =: L$$

**Proof.** Bin $n-1$ covers $[-1/4 + (n-1)\Delta, -1/4 + n\Delta) = [-\Delta, 0)$ (since $(n-1)\Delta = (n-1)/(4n)$ and $-1/4 + n/(4n) = -1/4 + 1/4 = 0$). Bin $n$ covers $[0, \Delta)$.

So the boundary $x = 0$ falls exactly between bin $n-1$ and bin $n$. No bin straddles it.

Therefore $L = \sum_{i=0}^{n-1} \int_{\text{bin}_i} f = \sum_{i=0}^{n-1} (c_i/m) = \text{left\_frac}$.

```lean
theorem left_frac_exact (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    let δ := (1 : ℝ) / (4 * n)
    -- Boundary of left half is at x = 0 = -1/4 + n * δ
    (-1/4 : ℝ) + n * δ = 0 := by
  field_simp; ring
```

### Fact 2: Refinement preserves left-half mass

**Theorem.** When parent $(c_0, \ldots, c_{d-1})$ at resolution $d = 2n$ is refined to child at resolution $2d = 4n$ by splitting each bin $c_i \to (a_i, c_i - a_i)$, the child's left-half sum equals the parent's left-half sum.

**Proof.** Parent's left half: bins $0, \ldots, n-1$, sum $= \sum_{i=0}^{n-1} c_i$.
Child's left half: bins $0, \ldots, 2n-1$, sum $= \sum_{i=0}^{n-1}(a_i + (c_i - a_i)) = \sum_{i=0}^{n-1} c_i$.

```lean
theorem refinement_preserves_left_sum (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ) (a : Fin (2 * n) → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_even : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin (2 * n), child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) = ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry
```

### Fact 3: No correction term needed

**Statement.** The asymmetry bound $\|f*f\|_\infty \geq 2L^2$ (Claim 2.1) is a direct $L^\infty$ bound. It does NOT go through the test-value framework, so the correction term $2/m + 1/m^2$ does NOT apply. The threshold $\sqrt{c_\text{target}/2}$ can be compared directly against `left_frac` with zero margin.

This is a meta-statement: the proof of Claim 2.1 (in a separate prompt) works directly with $\|f*f\|_\infty$, not via test values. Therefore the comparison $\text{left\_frac} \geq \sqrt{c_\text{target}/2}$ (which equals $L \geq \sqrt{c_\text{target}/2}$ by Fact 1) immediately gives $R(f) \geq c_\text{target}$.

```lean
-- This follows directly from Claim 2.1 + Fact 1
-- If left_frac >= sqrt(c_target/2), then L = left_frac >= sqrt(c_target/2),
-- so ||f*f||∞ >= 2L² >= c_target. No margin needed.
theorem asymmetry_no_margin (c_target : ℝ) (hct : 0 < c_target)
    (L : ℝ) (hL : L ≥ Real.sqrt (c_target / 2))
    (h_bound : ∀ L', 2 * L' ^ 2 ≤ c_target → L' < L) :  -- from Claim 2.1
    2 * L ^ 2 ≥ c_target := by
  sorry
```
