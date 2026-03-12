# Prompt 2: Pruning Rule Soundness

Prove Claims 2.1–2.4: every pruning rule used in the cascade is sound (never incorrectly eliminates a composition that could violate the bound). Attach `output.lean` as context — it contains all definitions and foundational lemmas already proved.

---

## Problem Context

We are proving $c \geq 1.4$ where:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

The proof uses branch-and-prune: enumerate step functions on a grid of $d = 2n$ bins of width $\Delta = 1/(4n)$, and prune those provably satisfying $R(f) \geq c_{\text{target}}$. A pruning rule is **sound** if whenever it eliminates a composition $c$, the corresponding $R(f) \geq c_{\text{target}}$ is guaranteed.

### Key Definitions

- **Step function:** $f(x) = c_i/m$ on bin $i \in [i\Delta, (i+1)\Delta)$, with $\sum c_i = m$.
- **$R(f) = \|f*f\|_{L^\infty} / (\int f)^2$** (autoconvolution ratio).
- **$(f*f)(x) = \int f(t)f(x-t)\,dt$** (autoconvolution / self-convolution).
- **Left-half mass:** $L = \int_{-1/4}^{0} f = \frac{1}{m}\sum_{i=0}^{n-1} c_i$ for $\int f = 1$.

### Already Proved in output.lean (available as context)

All formal definitions are established. Key lemmas: $D(0)=0$, $D(2n)=m$, $\sum c_i = m$, monotonicity of cumulative distribution, bin masses nonneg for nonneg $f$.

---

## Claim 2.1: Asymmetry Pruning — $\|f*f\|_\infty \geq 2L^2$

**Theorem to prove:**

Let $f \geq 0$ with $\operatorname{supp}(f) \subseteq [-1/4, 1/4]$ and $\int f = 1$. Let $L = \int_{-1/4}^{0} f$ (left-half mass). Then:

$$\|f * f\|_{L^\infty} \geq 2L^2$$

**Consequence:** If $L \geq \sqrt{c_{\text{target}}/2}$ (or $1 - L \geq \sqrt{c_{\text{target}}/2}$ by symmetry), then $R(f) = \|f*f\|_\infty \geq 2L^2 \geq c_{\text{target}}$.

**Proof:**

1. **Restriction.** Define $f_L = f \cdot \mathbf{1}_{(-1/4, 0)}$. Since $f \geq 0$, we have $f \geq f_L \geq 0$ pointwise. For nonneg functions, $(f*f)(x) \geq (f_L * f_L)(x)$ for all $x$, hence $\|f*f\|_\infty \geq \|f_L * f_L\|_\infty$.

2. **Support bound.** $\operatorname{supp}(f_L) \subseteq (-1/4, 0)$, so $\operatorname{supp}(f_L * f_L) \subseteq (-1/2, 0)$, an interval of length $1/2$.

3. **Fubini.** $\int (f_L * f_L) = (\int f_L)^2 = L^2$.

4. **Averaging principle.** For any nonneg $g$ supported on an interval of length $\lambda$: $\|g\|_\infty \geq \frac{1}{\lambda}\int g$. Applying with $g = f_L * f_L$ and $\lambda = 1/2$:

$$\|f_L * f_L\|_\infty \geq \frac{L^2}{1/2} = 2L^2$$

**Lean theorem statement:**

```lean
theorem asymmetry_bound (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Icc (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (L : ℝ) (hL : L = MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f)) :
    autoconvolution_ratio f ≥ 2 * L ^ 2 := by
  sorry
```

---

## Claim 2.2: Asymmetry Margin is Unnecessary

**Theorem to prove (three sub-facts):**

**Fact 1: Exactness.** For a step function on the $d = 2n$ grid, the discrete left-half mass fraction $\text{left\_frac} = \frac{1}{m}\sum_{i=0}^{n-1} c_i$ equals the continuous left-half mass $L = \int_{-1/4}^{0} f$ exactly.

*Proof:* The midpoint $x = 0$ falls exactly on the boundary between bin $n-1$ and bin $n$ (since bin $k$ covers $[-1/4 + k\Delta, -1/4 + (k+1)\Delta)$ and the boundary is at $-1/4 + n\Delta = -1/4 + n/(4n) = 0$). No bin straddles the boundary, so $L = \sum_{i=0}^{n-1} (c_i/m) \cdot \Delta \cdot (1/\Delta) = \sum_{i=0}^{n-1} c_i/m$.

**Fact 2: Refinement invariance.** When a parent $(c_0, \ldots, c_{d-1})$ is refined to child $(c_0^{(a)}, c_0^{(b)}, c_1^{(a)}, c_1^{(b)}, \ldots)$ where $c_i^{(a)} + c_i^{(b)} = c_i$, the child's left-half mass equals the parent's.

*Proof:* Parent left half = bins $0, \ldots, n-1$. Child left half = bins $0, \ldots, 2n-1$. Child bin $2i$ and $2i+1$ correspond to parent bin $i$, so $\sum_{j=0}^{2n-1} c_j^{\text{child}} = \sum_{i=0}^{n-1}(c_i^{(a)} + c_i^{(b)}) = \sum_{i=0}^{n-1} c_i$.

**Fact 3: No correction needed.** The asymmetry bound $\|f*f\|_\infty \geq 2L^2$ (Claim 2.1) is a **direct $L^\infty$ bound** that does not go through the test-value framework. Therefore the correction term $2/m + 1/m^2$ is not needed.

**Consequence:** The asymmetry threshold $\sqrt{c_{\text{target}}/2}$ can be compared directly against $\text{left\_frac}$ with no safety margin.

**Lean theorem statements:**

```lean
-- Fact 1: Discrete left_frac = continuous L
theorem left_frac_exact (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (f : ℝ → ℝ) (hf : ∀ i, c i = canonical_discretization f n m i) :
    (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) / m =
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico (-1/4 : ℝ) 0) f) := by
  sorry

-- Fact 2: Refinement preserves left-half mass
theorem refinement_preserves_left_mass (n m : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ) (child : Fin (4 * n) → ℕ)
    (h_split : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i) :
    ∑ i : Fin (2 * n), (child ⟨i.1, by omega⟩ : ℕ) = ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry
```

---

## Claim 2.3: Single-Bin Energy Cap (x_cap)

**Theorem to prove:**

For any nonneg function $g$ supported on $(-1/4, 1/4)$ with $\int g = 1$, discretized on $d = 2n$ bins with resolution $m$: if any bin has $c_i > \lfloor m\sqrt{c_{\text{target}}/d}\rfloor$, then $R(g) \geq c_{\text{target}}$.

**Proof:**

Let bin $i$ have mass $M_i = c_i/m$ (the step function height times bin width). Define $g_i = g \cdot \mathbf{1}_{\text{bin}_i}$.

1. **Restriction:** $g \geq g_i \geq 0$, so $\|g*g\|_\infty \geq \|g_i * g_i\|_\infty$.

2. **Support:** $\operatorname{supp}(g_i) \subseteq \text{bin}_i$ (width $\Delta = 1/(4n)$), so $\operatorname{supp}(g_i * g_i)$ has length $2\Delta = 1/(2n)$.

3. **Fubini:** $\int(g_i * g_i) = (\int g_i)^2 = M_i^2 = (c_i/m)^2$.

4. **Averaging:** $\|g_i * g_i\|_\infty \geq (c_i/m)^2 / (1/(2n)) = 2n \cdot c_i^2/m^2 = d \cdot c_i^2/m^2$.

5. **Threshold:** $R(g) \geq d \cdot c_i^2/m^2 \geq c_{\text{target}}$ when $c_i \geq m\sqrt{c_{\text{target}}/d}$.

**Key point:** This is a **direct $L^\infty$ bound** — no correction term needed.

**Lean theorem statement:**

```lean
theorem xcap_bound (n m : ℕ) (hn : n > 0) (hm : m > 0) (c_target : ℝ) (hct : 0 < c_target)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (i : Fin (2 * n)) (ci : ℕ) (hci : ci = canonical_discretization f n m i)
    (h_large : (ci : ℝ) ≥ m * Real.sqrt (c_target / (2 * n))) :
    autoconvolution_ratio f ≥ c_target := by
  sorry
```

---

## Claim 2.4: Integer Dynamic Threshold Equivalence

**Theorem to prove:**

The integer-space dynamic threshold computation:

$$\text{dyn\_it} = \lfloor (c_{\text{target}} \cdot m^2 + 1 + 10^{-9} m^2 + 2 W_{\text{int}}) \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \rfloor$$

where $W_{\text{int}} = \sum_{i \in \mathcal{B}} c_i$ and $\varepsilon = 2.22 \times 10^{-16}$, satisfies:

$$\text{dyn\_it} \geq \lfloor (c_{\text{target}} \cdot m^2 + 1 + 2W_{\text{int}}) \cdot \frac{\ell}{4n} \rfloor$$

That is, the computed threshold is **at least as large** as the exact mathematical threshold (conservative direction — harder to prune, safe).

**Proof sketch:**

1. The term $+10^{-9} m^2$ adds $\approx 4 \times 10^{-7}$ (for $m=20$) to the pre-floor argument.
2. The factor $(1 - 4\varepsilon)$ subtracts $\approx 8.9 \times 10^{-16}$ times the value.
3. Net effect: the addition ($\sim 10^{-7}$) dominates the subtraction ($\sim 10^{-13}$) by a factor of $\sim 10^6$.
4. Therefore the pre-floor argument is strictly larger than the exact value, and the floor is $\geq$ the exact floor.

**Also prove:** A composition is soundly pruned when `ws > dyn_it` (strict integer comparison), where `ws` is the exact integer window sum. Since `dyn_it` is the floor of a value $\geq$ the exact threshold, and `ws` is exact integer, `ws > dyn_it` implies the continuous test value exceeds the continuous threshold.

**Lean theorem statement:**

```lean
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ : ℕ) (W_int : ℕ)
    (hm : m > 0) (hn : n > 0) (hℓ : ℓ > 0)
    (eps : ℝ := 2.22e-16) :
    ⌊(c_target * m^2 + 1 + 1e-9 * m^2 + 2 * W_int) * (ℓ / (4 * n)) * (1 - 4 * eps)⌋ ≥
    ⌊(c_target * m^2 + 1 + 2 * W_int) * (ℓ / (4 * n))⌋ := by
  sorry
```
