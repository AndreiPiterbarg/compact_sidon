# Prompt 7: Composition Enumeration and Child Generation

**Claims 3.1 + 3.2.** Combinatorial completeness. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant via a branch-and-prune cascade. The cascade must enumerate ALL possible step functions at each resolution level. This prompt proves the enumeration is complete.

### Definitions (in output.lean)

- `is_composition n m c` := $\sum_i c_i = m$ for $c : \text{Fin}(2n) \to \mathbb{N}$.

---

## Claim 3.1: Composition Enumeration is Complete (Stars and Bars)

**Theorem.** The number of compositions of $m$ into $d$ nonneg integer parts is $\binom{m+d-1}{d-1}$.

**Proof (stars and bars).** Represent a composition $(c_0, \ldots, c_{d-1})$ with $\sum c_i = m$ as a sequence of $m$ stars and $d-1$ bars: $\underbrace{\star\cdots\star}_{c_0} | \underbrace{\star\cdots\star}_{c_1} | \cdots | \underbrace{\star\cdots\star}_{c_{d-1}}$.

This bijects with choosing positions for $d-1$ bars among $m + d - 1$ total symbols. Count: $\binom{m+d-1}{d-1}$.

Equivalently, this bijects with the set of strictly increasing sequences $0 \leq s_1 < s_2 < \ldots < s_{d-1} \leq m + d - 2$ via $s_k = c_0 + \ldots + c_{k-1} + k$. This is a well-known combinatorial identity.

```lean
-- Number of weak compositions
theorem composition_count (m d : ℕ) (hd : d > 0) :
    Finset.card (Finset.filter (fun c : Fin d → Fin (m + 1) =>
      ∑ i, (c i : ℕ) = m) Finset.univ) = Nat.choose (m + d - 1) (d - 1) := by
  sorry
```

---

## Claim 3.2: Child Generation is Complete

At each refinement level, parent $(c_0, \ldots, c_{d-1})$ generates children by splitting each bin $c_i$ into $(a_i, c_i - a_i)$ where $\max(0, c_i - x_\text{cap}) \leq a_i \leq \min(c_i, x_\text{cap})$.

### 3.2a: Cartesian product structure

**Theorem.** The children form a Cartesian product $\prod_{i=0}^{d-1} S_i$ where $S_i = \{a : \max(0, c_i - x_\text{cap}) \leq a \leq \min(c_i, x_\text{cap})\}$.

The number of children is $\prod_{i=0}^{d-1} |S_i| = \prod_{i=0}^{d-1} (\min(c_i, x_\text{cap}) - \max(0, c_i - x_\text{cap}) + 1)$.

```lean
-- Per-bin choice count
theorem per_bin_choices (c_i x_cap : ℕ) :
    Finset.card (Finset.Icc (Nat.max 0 (c_i - x_cap)) (Nat.min c_i x_cap)) =
    Nat.min c_i x_cap - Nat.max 0 (c_i - x_cap) + 1 := by
  sorry
```

### 3.2b: x_cap soundness (reference to Claim 2.3)

Any child with some bin $> x_\text{cap}$ satisfies $R(f) \geq c_\text{target}$ (proved in prompt 5 as Claim 2.3). So skipping those children is sound.

### 3.2c: Children preserve total mass

**Theorem.** If $\sum_{i=0}^{d-1} c_i = m$ (parent sums to $m$) and child is $(a_0, c_0-a_0, \ldots, a_{d-1}, c_{d-1}-a_{d-1})$ where $0 \leq a_i \leq c_i$, then $\sum_{j=0}^{2d-1} \text{child}_j = m$.

**Proof.** $\sum_{j=0}^{2d-1} \text{child}_j = \sum_{i=0}^{d-1}(a_i + (c_i - a_i)) = \sum_{i=0}^{d-1} c_i = m$.

```lean
theorem child_preserves_mass (d m : ℕ) (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (hp : ∑ i, parent i = m) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j, child j = m := by
  sorry
```

### 3.2d: Child is a valid refinement

**Theorem.** The child layout places sub-bins $(a_i, c_i - a_i)$ at positions $(2i, 2i+1)$. On the continuous grid, bin $i$ of width $\Delta$ splits into two sub-bins of width $\Delta/2$: the left sub-bin gets mass $a_i/m$, the right gets $(c_i - a_i)/m$. Together they have mass $c_i/m$ = parent bin $i$'s mass.

```lean
-- Each parent bin's total mass is preserved in its two child bins
theorem child_bin_sum (d : ℕ) (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i)
    (i : Fin d) :
    child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i := by
  sorry
```
