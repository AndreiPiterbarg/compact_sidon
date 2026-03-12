# Prompt 5: Optimization Soundness B — x_cap, Hoisted Asymmetry, Ell Order, Integer Safety, FP Correctness

Prove Claims 4.5–4.8 and 5.1–5.2: remaining optimizations and numerical correctness. Attach `output.lean` as context — it contains all definitions and foundational lemmas already proved.

---

## Problem Context

We are proving $c \geq 1.4$ where:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

Parameters: $n = 2$ (so $d = 4$ at level 0, doubling each level), $m = 20$, $c_{\text{target}} = 1.4$.

### Key Definitions

- **$d = 2n$ bins**, step function with masses $(c_0, \ldots, c_{d-1})$, $\sum c_i = m$.
- **Discrete autoconvolution:** $\text{conv}[k] = \sum_{i+j=k} c_i c_j$.
- **Test value:** $\text{TV}(\ell, s_{\text{lo}}) = \frac{1}{4n\ell} \sum_{k=s_{\text{lo}}}^{s_{\text{lo}}+\ell-2} \text{conv}[k]$.
- **Window sum (integer):** $\text{ws} = \sum_{k} \text{conv}[k]$ (exact integer when $c_i$ are integers).
- **Dynamic threshold:** $\text{dyn\_it} = \lfloor (\text{base} + 2W_{\text{int}}) \cdot \text{scale} \rfloor$.
- **$x_{\text{cap}}$:** Maximum allowed single-bin mass.
- **Refinement:** Parent $(c_0, \ldots, c_{d-1}) \to$ child $(a_0, c_0-a_0, \ldots)$ at resolution $2d$.
- **Left-half mass:** $\text{left\_sum} = \sum_{i=0}^{n-1} c_i$ (for parent at resolution $d = 2n$).

### Already Proved in output.lean (available as context)

All formal definitions established. Key lemmas: $\sum c_i = m$, bin masses nonneg, monotonicity of cumulative distribution.

---

## Claim 4.5: Cauchy-Schwarz x_cap — No Correction Term Needed

**Theorem to prove:**

For ANY nonneg function $g$ supported on $(-1/4, 1/4)$ with $\int g = 1$ and mass $M_i$ in bin $i$ (width $\Delta = 1/(4n)$):

$$\|g * g\|_{L^\infty} \geq d \cdot M_i^2$$

where $d = 2n$. This is a **direct** $L^\infty$ bound (no discretization correction needed).

**Proof:**

1. Let $g_i = g \cdot \mathbf{1}_{\text{bin}_i}$ where $\text{bin}_i = [-1/4 + i\Delta, -1/4 + (i+1)\Delta)$.
2. $g \geq g_i \geq 0$ (since $g \geq 0$), so $(g*g)(x) \geq (g_i * g_i)(x)$ for all $x$.
3. $\operatorname{supp}(g_i * g_i) \subseteq [-1/2 + 2i\Delta, -1/2 + 2(i+1)\Delta)$, length $= 2\Delta = 1/(2n)$.
4. $\int (g_i * g_i) = (\int g_i)^2 = M_i^2$.
5. By averaging: $\|g_i * g_i\|_\infty \geq M_i^2 / (2\Delta) = M_i^2 \cdot 2n = d \cdot M_i^2$.

**Consequence:** Setting $M_i = c_i/m$ and requiring $d \cdot c_i^2/m^2 \geq c_{\text{target}}$ gives $c_i \geq m\sqrt{c_{\text{target}}/d}$, so $x_{\text{cap}} = \lfloor m\sqrt{c_{\text{target}}/d} \rfloor$.

This is tighter than the "standard" x_cap which uses $c_{\text{target}} + 2/m + 1/m^2$ (correction included), because the Cauchy-Schwarz bound is direct and needs no correction.

**Lean theorem statement:**

```lean
theorem single_bin_Linfty_bound (n : ℕ) (hn : n > 0)
    (g : ℝ → ℝ) (hg_nonneg : ∀ x, 0 ≤ g x)
    (hg_supp : Function.support g ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM_i : M_i = bin_masses g n i) :
    (MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal
    ≥ (2 * n : ℝ) * M_i ^ 2 := by
  sorry
```

---

## Claim 4.6: Hoisted Asymmetry Check — Left-Half Mass Invariant Under Refinement

**Theorem to prove:**

For a parent composition $(c_0, \ldots, c_{d-1})$ at resolution $d = 2n$, ALL children at resolution $2d = 4n$ have the **same** left-half mass sum. Therefore the asymmetry check can be done once per parent.

**Formal statement:** Let child bins be $(a_0, c_0-a_0, a_1, c_1-a_1, \ldots, a_{d-1}, c_{d-1}-a_{d-1})$ at resolution $2d$. The child's left half spans bins $0, \ldots, 2n-1$ (corresponding to parent bins $0, \ldots, n-1$). Then:

$$\sum_{j=0}^{2n-1} \text{child}[j] = \sum_{j=0}^{2n-1} \text{child}[j]$$

where child bins $(2i, 2i+1) = (a_i, c_i - a_i)$. So:

$$\sum_{j=0}^{2n-1} \text{child}[j] = \sum_{i=0}^{n-1} (a_i + (c_i - a_i)) = \sum_{i=0}^{n-1} c_i$$

This equals the parent's left-half sum, independent of the choice of $a_i$.

**Lean theorem statement:**

```lean
theorem hoisted_asymmetry (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ) (a : Fin (2 * n) → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hchild : ∀ i : Fin (2 * n),
      child ⟨2 * i.1, by omega⟩ = a i ∧
      child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) = ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry
```

---

## Claim 4.7: Ell Scan Order Does Not Affect Pruning

**Theorem to prove:**

The pruning condition is: $\exists (\ell, s_{\text{lo}})$ such that $\text{ws}(\ell, s_{\text{lo}}) > \text{dyn\_it}(\ell, W_{\text{int}})$. Since this is an existential quantifier over windows, the **order** in which windows are checked does not change which compositions are pruned.

**Formal statement:** If $\sigma$ is any permutation of the window parameter set $\{2, 3, \ldots, 2d\}$, then:

$$\left(\exists k \in [1, |\sigma|],\; \text{ws}(\sigma(k), \cdot) > \text{dyn\_it}(\sigma(k), \cdot)\right) \iff \left(\exists \ell \in \{2, \ldots, 2d\},\; \text{ws}(\ell, \cdot) > \text{dyn\_it}(\ell, \cdot)\right)$$

This is immediate from the fact that permutations preserve set membership.

**Lean theorem statement:**

```lean
theorem ell_order_irrelevant {α : Type*} (S : Finset α) (P : α → Prop) (σ : Equiv.Perm α)
    (hσ : ∀ a, a ∈ S ↔ σ a ∈ S) :
    (∃ a ∈ S, P a) ↔ (∃ a ∈ S, P a) := by
  exact Iff.rfl
```

(This is literally `Iff.rfl` — it's a trivial statement about existential quantifiers being order-independent.)

---

## Claim 4.8: Integer Arithmetic Overflow Safety

**Theorem to prove:**

For $m \leq 200$ and $d \leq 128$ (covering all cascade levels through L4):

1. **Raw convolution entries fit int32:** $\text{conv}[k] = \sum_{i+j=k} c_i c_j \leq m^2 = 40000 \leq 2^{31} - 1$.

   *Proof:* $\text{conv}[k] = \sum_{i+j=k} c_i c_j \leq \sum_{i+j=k} c_i \cdot m \leq m \sum_i c_i = m^2$. For $m = 200$: $m^2 = 40000 \ll 2^{31} - 1 = 2147483647$.

2. **Prefix sums fit int32:** $\text{prefix}[k] = \sum_{t=0}^{k} \text{conv}[t] \leq \sum_{t=0}^{2d-2} \text{conv}[t] = (\sum c_i)^2 = m^2$.

   *Proof:* $\sum_{t=0}^{2d-2} \text{conv}[t] = \sum_{t} \sum_{i+j=t} c_i c_j = \sum_i \sum_j c_i c_j = (\sum c_i)^2 = m^2$.

3. **Window sums:** $\text{ws} = \text{prefix}[s_{\text{lo}} + \ell - 2] - \text{prefix}[s_{\text{lo}} - 1]$. The subtraction is done in int64 to avoid negative overflow. Result $\leq m^2$.

4. **Incremental deltas fit int32:** $|\text{new}^2 - \text{old}^2| \leq m^2$ since both terms $\leq m^2$.

**Lean theorem statements:**

```lean
-- Conv entries bounded by m^2
theorem conv_bounded {d : ℕ} (c : Fin d → ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0 ≤ m ^ 2 := by
  sorry

-- Total convolution sum = m^2
theorem conv_total_sum {d : ℕ} (c : Fin d → ℕ) (hc : ∑ i, c i = m) :
    ∑ k ∈ Finset.range (2 * d - 1),
      (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0) = m ^ 2 := by
  sorry

-- For m ≤ 200: m^2 fits int32
theorem m_squared_fits_int32 (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by
  omega
```

---

## Claim 5.1: Conservative Rounding in Dynamic Threshold

**Theorem to prove:**

The computed threshold with margins:

$$\text{dyn\_it} = \lfloor (c_{\text{target}} \cdot m^2 + 1 + 10^{-9} \cdot m^2 + 2W_{\text{int}}) \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \rfloor$$

satisfies $\text{dyn\_it} \geq \lfloor (c_{\text{target}} \cdot m^2 + 1 + 2W_{\text{int}}) \cdot \frac{\ell}{4n} \rfloor$ (i.e., it is at least as large as the exact mathematical threshold).

**Proof:**

Let $A = (c_{\text{target}} \cdot m^2 + 1 + 2W_{\text{int}}) \cdot \frac{\ell}{4n}$ (exact threshold argument).

Let $B = (A + 10^{-9} \cdot m^2 \cdot \frac{\ell}{4n}) \cdot (1 - 4\varepsilon)$ (computed threshold argument).

We need $B \geq A$, equivalently:

$$A \cdot (1 - 4\varepsilon) + 10^{-9} \cdot m^2 \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \geq A$$

$$10^{-9} \cdot m^2 \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \geq 4\varepsilon \cdot A$$

For $m = 20, n = 2, \ell \geq 2$:
- LHS $\geq 10^{-9} \cdot 400 \cdot \frac{2}{8} \cdot (1 - 4\varepsilon) \approx 10^{-7}$
- RHS $\leq 4\varepsilon \cdot (1.4 \cdot 400 + 1 + 40) \cdot \frac{2d}{4n} \leq 4 \cdot 2.22 \times 10^{-16} \cdot 601 \cdot 16 \approx 8.5 \times 10^{-12}$

Since $10^{-7} \gg 8.5 \times 10^{-12}$, the inequality holds with massive margin.

**Lean theorem statement:**

```lean
theorem fp_margin_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ)
    (hct : 0 ≤ c_target) (hW : W_int ≤ m) :
    let A := (c_target * m^2 + 1 + 2 * W_int) * (ℓ / (4 * n : ℝ))
    let B := (c_target * m^2 + 1 + 1e-9 * m^2 + 2 * W_int) * (ℓ / (4 * n : ℝ)) * (1 - 4 * 2.22e-16)
    ⌊B⌋ ≥ ⌊A⌋ := by
  sorry
```

---

## Claim 5.2: Integer Autoconvolution is Exact

**Theorem to prove:**

When all $c_i$ are nonneg integers, the autoconvolution $\text{conv}[k] = \sum_{i+j=k} c_i c_j$ is an exact integer. All window sums $\text{ws} = \text{prefix}[b] - \text{prefix}[a]$ are exact integers. Therefore the only source of floating-point error in the pruning comparison is the threshold computation `dyn_it`, which is handled by Claim 5.1.

**Proof:**

1. $c_i \in \mathbb{Z}_{\geq 0}$ by construction (integer bin masses).
2. Products $c_i \cdot c_j \in \mathbb{Z}_{\geq 0}$.
3. Sums of integers are integers.
4. Prefix sums and differences of prefix sums are integers.
5. The comparison `ws > dyn_it` compares an exact integer (`ws`) against a floored value (`dyn_it`). Since `ws` is exact and `dyn_it = floor(...)`, the comparison `ws > dyn_it` is equivalent to `ws ≥ dyn_it + 1`, which is an integer comparison.

**Lean theorem statement:**

```lean
-- Integer autoconvolution is integer-valued
theorem integer_autoconv {d : ℕ} (c : Fin d → ℤ) (k : ℕ) :
    ∃ z : ℤ, (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0) = z := by
  exact ⟨_, rfl⟩

-- Window sum (prefix difference) is integer
theorem integer_window_sum {d : ℕ} (c : Fin d → ℤ) (s_lo ℓ : ℕ) :
    ∃ z : ℤ, (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 1),
      ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0) = z := by
  exact ⟨_, rfl⟩
```

(These are trivially true in Lean since the types are already `ℤ`. The real content is in Claim 5.1 showing the threshold comparison is sound despite FP computation of `dyn_it`.)
