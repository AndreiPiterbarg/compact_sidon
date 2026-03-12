# Prompt 11: Incremental Autoconvolution Update

**Claim 4.2 only.** This is algebraically substantial. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. The cascade computes the discrete autoconvolution $\text{conv}[t] = \sum_{i+j=t} c_i c_j$ for each child composition. When consecutive children differ in few positions, the autoconvolution is updated incrementally rather than recomputed.

### Definitions (in output.lean)

- `discrete_autoconvolution a k = ∑ i, ∑ j, if i+j=k then a_i * a_j else 0`

---

## Claim 4.2: Incremental Update Correctness

**Theorem.** Let $c, c'$ be two child compositions that differ only in positions belonging to a set $S \subseteq \{0, \ldots, d-1\}$ (i.e., $c'_i = c_i$ for $i \notin S$). Then:

$$\text{conv}'[t] - \text{conv}[t] = \sum_{\substack{i+j=t \\ i \in S \text{ or } j \in S}} (c'_i c'_j - c_i c_j)$$

and this delta can be decomposed into disjoint groups.

### Path 1: Fast path ($|S| = 2$, positions $\{2p, 2p+1\}$)

Only one parent bin $p$ changed: $c'_{2p} \neq c_{2p}$ and $c'_{2p+1} \neq c_{2p+1}$, with $c'_{2p} + c'_{2p+1} = c_{2p} + c_{2p+1} = c_{\text{parent},p}$ (mass preserved within the bin pair).

The delta decomposes as:

$$\Delta\text{conv}[t] = \underbrace{\Delta\text{self}_{2p}[t]}_{\text{Group A}} + \underbrace{\Delta\text{self}_{2p+1}[t]}_{\text{Group B}} + \underbrace{\Delta\text{mutual}[t]}_{\text{Group C}} + \underbrace{\Delta\text{cross}[t]}_{\text{Group D}}$$

where:
- **Group A (self-term, pos $2p$):** $\Delta\text{self}_{2p}[t] = [(c'_{2p})^2 - (c_{2p})^2] \cdot [t = 4p]$
- **Group B (self-term, pos $2p+1$):** $\Delta\text{self}_{2p+1}[t] = [(c'_{2p+1})^2 - (c_{2p+1})^2] \cdot [t = 4p+2]$
- **Group C (mutual term):** $\Delta\text{mutual}[t] = 2[(c'_{2p})(c'_{2p+1}) - (c_{2p})(c_{2p+1})] \cdot [t = 4p+1]$
- **Group D (cross-terms):** $\Delta\text{cross}[t] = \sum_{q \notin \{2p, 2p+1\}} 2[c_q \cdot ((c'_{2p} - c_{2p}) \cdot [t=2p+q] + (c'_{2p+1} - c_{2p+1}) \cdot [t=2p+1+q])]$

**Proof of correctness:**

$\text{conv}'[t] - \text{conv}[t] = \sum_{i+j=t}(c'_i c'_j - c_i c_j)$

Split the sum based on how many of $\{i,j\}$ are in $S = \{2p, 2p+1\}$:

1. **Both in $S$:** $(i,j) \in S \times S$ with $i+j=t$. Possible pairs: $(2p,2p), (2p,2p+1), (2p+1,2p), (2p+1,2p+1)$. These give groups A, B, C.

2. **Exactly one in $S$:** $(i \in S, j \notin S)$ or $(i \notin S, j \in S)$. For $i \in S$: $c'_i c'_j - c_i c_j = (c'_i - c_i) c_j$ (since $c'_j = c_j$). Symmetry gives factor of 2. This is group D.

3. **Neither in $S$:** $c'_i c'_j - c_i c_j = 0$.

Groups are disjoint by the partition of $(i,j)$ pairs. Union is exhaustive (every pair falls in exactly one category).

```lean
-- Delta decomposition for fast path
theorem fast_path_delta {d : ℕ} (c c' : Fin d → ℤ) (p : ℕ) (hp : 2*p+1 < d)
    (h_same : ∀ i : Fin d, i.1 ≠ 2*p → i.1 ≠ 2*p+1 → c' i = c i) (t : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=t then c' i * c' j - c i * c j else 0) =
    -- Group A: self-term at 2p
    (if t = 4*p then c' ⟨2*p, by omega⟩ ^ 2 - c ⟨2*p, by omega⟩ ^ 2 else 0) +
    -- Group B: self-term at 2p+1
    (if t = 4*p+2 then c' ⟨2*p+1, by omega⟩ ^ 2 - c ⟨2*p+1, by omega⟩ ^ 2 else 0) +
    -- Group C: mutual term
    (if t = 4*p+1 then 2 * (c' ⟨2*p, by omega⟩ * c' ⟨2*p+1, by omega⟩ - c ⟨2*p, by omega⟩ * c ⟨2*p+1, by omega⟩) else 0) +
    -- Group D: cross-terms
    (∑ q : Fin d, if q.1 ≠ 2*p ∧ q.1 ≠ 2*p+1 then
      (if q.1 + (2*p) = t then 2 * (c' ⟨2*p, by omega⟩ - c ⟨2*p, by omega⟩) * c q else 0) +
      (if q.1 + (2*p+1) = t then 2 * (c' ⟨2*p+1, by omega⟩ - c ⟨2*p+1, by omega⟩) * c q else 0)
     else 0) := by
  sorry
```

### Path 2: Short carry ($|S| = 2k$ positions, $k$ parent bins changed)

Same structure as fast path, extended. The changed positions are $\{2p_1, 2p_1+1, 2p_2, 2p_2+1, \ldots, 2p_k, 2p_k+1\}$.

The delta decomposes into:
- **(a)** Self + mutual within each changed pair (same as Groups A,B,C above, for each $p_j$)
- **(b)** Cross-terms between different changed pairs
- **(c)** Cross-terms between changed and unchanged bins

These three groups are disjoint (they partition the set of $(i,j)$ pairs with at least one index in $S$) and exhaustive.

```lean
-- General delta decomposition (sketch — the key property)
theorem general_delta_correct {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=t then c' i * c' j else 0) =
    (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=t then c i * c j else 0) +
    (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=t then c' i * c' j - c i * c j else 0) := by
  sorry
```

### Path 3: Deep carry

Full recompute: $\text{conv}'[t] = \sum_{i+j=t} c'_i c'_j$. Trivially correct by definition.
