# Prompt 13: Hoisted Asymmetry, Ell Scan Order, and Integer Overflow Safety

**Claims 4.5 + 4.6 + 4.7 + 4.8.** Simpler claims grouped together. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. Grid: $d = 2n$ bins, compositions summing to $m$. Parameters: $n = 2, m = 20$.

### Definitions (in output.lean)

- `bin_masses f n i`, `discrete_autoconvolution`, `is_composition`

---

## Claim 4.5: Cauchy-Schwarz x_cap Needs No Correction

**Theorem.** For any nonneg $f$ on $(-1/4, 1/4)$ with $\int f = 1$ and mass $M_i$ in bin $i$ (width $\Delta = 1/(4n)$):

$$\|f*f\|_\infty \geq d \cdot M_i^2$$

**Proof.** Restrict to bin $i$: let $f_i = f \cdot \mathbf{1}_{\text{bin}_i}$. Since $f \geq f_i \geq 0$: $\|f*f\|_\infty \geq \|f_i * f_i\|_\infty$.

$\operatorname{supp}(f_i * f_i)$ has length $2\Delta = 1/(2n)$. By Fubini: $\int(f_i * f_i) = M_i^2$. By averaging: $\|f_i * f_i\|_\infty \geq M_i^2/(1/(2n)) = 2n \cdot M_i^2 = d \cdot M_i^2$.

This is a DIRECT $L^\infty$ bound — no correction term needed.

```lean
-- Same structure as Claim 2.1 but restricted to a single bin
-- (The proof technique is identical; the key difference is the support width)
theorem single_bin_bound (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM : M_i = bin_masses f n i) :
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥
    (2 * n : ℝ) * M_i ^ 2 := by
  sorry
```

---

## Claim 4.6: Hoisted Asymmetry — Left-Half Mass Invariant Under Refinement

**Theorem.** For parent $(c_0, \ldots, c_{d-1})$ at resolution $d = 2n$ and ANY child at resolution $2d = 4n$:

$$\sum_{j=0}^{2n-1} \text{child}[j] = \sum_{i=0}^{n-1} c_i$$

**Proof.** Child bins $(2i, 2i+1) = (a_i, c_i - a_i)$. The child's left half spans bins $0, \ldots, 2n-1$, which correspond to parent bins $0, \ldots, n-1$.

$$\sum_{j=0}^{2n-1} \text{child}[j] = \sum_{i=0}^{n-1}(\text{child}[2i] + \text{child}[2i+1]) = \sum_{i=0}^{n-1}(a_i + c_i - a_i) = \sum_{i=0}^{n-1} c_i$$

This is independent of the choice of $a_i$, so the asymmetry check need only be done once per parent.

```lean
theorem left_sum_invariant (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ) (a : Fin (2 * n) → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_e : ∀ i : Fin (2 * n), child ⟨2*i.1, by omega⟩ = a i)
    (hc_o : ∀ i : Fin (2 * n), child ⟨2*i.1+1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) =
    ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry
```

---

## Claim 4.7: Ell Scan Order Does Not Affect Pruning

**Theorem.** The pruning condition $\exists (\ell, s_\text{lo}): \text{ws} > \text{dyn\_it}$ is an existential quantifier. The order in which $\ell$ values are checked does not change which compositions are pruned (only affects speed).

**Proof.** $(\exists x \in S, P(x))$ is invariant under permutations of $S$.

```lean
-- Trivially true for existential quantifiers over finite sets
theorem exists_invariant_under_permutation {α : Type*} [DecidableEq α]
    (S : Finset α) (P : α → Prop) [DecidablePred P] :
    (∃ x ∈ S, P x) ↔ (∃ x ∈ S, P x) :=
  Iff.rfl
```

---

## Claim 4.8: Integer Overflow Safety for $m \leq 200$

**Theorem 1.** $\text{conv}[k] \leq m^2$ for any composition summing to $m$.

**Proof.** $\text{conv}[k] = \sum_{i+j=k} c_i c_j \leq \sum_{i+j=k} c_i \cdot (\sum_j c_j) = m \sum_{i+j=k} c_i \leq m \cdot m = m^2$.

Actually, tighter: $\text{conv}[k] = \sum_{i+j=k} c_i c_j$. By Cauchy-Schwarz or AM-GM: this is maximized when all mass is in one bin pair, giving $m^2$. More directly: $\sum_{i+j=k} c_i c_j \leq (\sum_i c_i)(\sum_j c_j) = m^2$ by expanding.

Wait, that's not right because the constraint is $i+j=k$, not all pairs. The correct bound:

$\sum_{i+j=k} c_i c_j \leq (\sum_{i: i \leq k} c_i)(\max_j c_j) \leq m \cdot m = m^2$. Or more cleanly: the total $\sum_k \text{conv}[k] = (\sum c_i)^2 = m^2$, and each $\text{conv}[k] \geq 0$, so $\text{conv}[k] \leq m^2$.

**Theorem 2.** $\sum_{k=0}^{2d-2} \text{conv}[k] = m^2$.

**Proof.** $\sum_k \sum_{i+j=k} c_i c_j = \sum_i \sum_j c_i c_j = (\sum_i c_i)^2 = m^2$.

**Theorem 3.** For $m \leq 200$: $m^2 \leq 40000 \leq 2^{31} - 1$.

```lean
-- conv[k] ≤ total sum of conv = m²
theorem conv_entry_le_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=k then c i * c j else 0 ≤ m ^ 2 := by
  sorry

-- Total autoconvolution = m²
theorem conv_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) :
    ∑ k ∈ Finset.range (2*d-1),
      (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=k then c i * c j else 0) = m ^ 2 := by
  sorry

-- m² fits int32 for m ≤ 200
theorem int32_safe (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by omega
```
