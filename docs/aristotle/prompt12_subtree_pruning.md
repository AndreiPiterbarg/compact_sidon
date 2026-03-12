# Prompt 12: Subtree Pruning Soundness

**Claim 4.4 only.** Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. The cascade generates children via an odometer. On a deep carry (many positions change), before recomputing the full autoconvolution, we check if the **entire subtree** of remaining children sharing the same prefix can be pruned at once.

### Definitions

- **Child composition:** $d$ bins, $c_i \geq 0$, $\sum c_i = m$.
- **Discrete autoconvolution:** $\text{conv}[t] = \sum_{i+j=t} c_i c_j$.
- **Window sum:** $\text{ws}(\ell, s_\text{lo}) = \sum_{k=s_\text{lo}}^{s_\text{lo}+\ell-2} \text{conv}[k]$.
- **Dynamic threshold:** $\text{dyn\_it}(\ell, W) = \lfloor (\text{base} + 2W) \cdot \text{scale}(\ell) \rfloor$ where base $> 0$, scale $> 0$.
- **Pruning:** Child is pruned if $\exists (\ell, s_\text{lo}): \text{ws} > \text{dyn\_it}$.

### Subtree setup

- **Fixed bins:** $0, \ldots, 2p-1$. These are the same for all children in the subtree.
- **Unfixed bins:** $2p, \ldots, d-1$. These vary across subtree children.
- **Partial autoconvolution:** $\text{partial\_conv}[t] = \sum_{\substack{i+j=t \\ i < 2p,\, j < 2p}} c_i c_j$
- **Partial window sum:** $\text{ws\_partial}(\ell, s_\text{lo}) = \sum_{k} \text{partial\_conv}[k]$

For $W_\text{int,max}$: for fixed parent positions $q < p$, use exact child bin masses. For unfixed positions $q \geq p$, use $\text{parent}[q]$ as upper bound (since $c_{2q} + c_{2q+1} = \text{parent}[q]$ and any subset contributes at most $\text{parent}[q]$).

---

## Three Inequalities

### Inequality 1: $\text{ws\_full}(c') \geq \text{ws\_partial}$ for all children $c'$ in subtree

**Theorem.** For any child $c'$ in the subtree (same fixed bins, any unfixed bins with $c'_i \geq 0$):

$$\text{conv}'[t] = \text{partial\_conv}[t] + \underbrace{\sum_{\substack{i+j=t \\ i \geq 2p \text{ or } j \geq 2p}} c'_i c'_j}_{\geq 0}$$

**Proof.** Split $\sum_{i+j=t} c'_i c'_j$ into three parts:
1. Both $i, j < 2p$: these are the fixed bins, giving $\text{partial\_conv}[t]$.
2. Both $i, j \geq 2p$: $c'_i c'_j \geq 0$ (nonneg integers).
3. One $< 2p$, one $\geq 2p$: $c'_i c'_j \geq 0$ (nonneg integers).

Parts 2+3 are $\geq 0$, so $\text{conv}'[t] \geq \text{partial\_conv}[t]$.

Summing over the window: $\text{ws\_full} \geq \text{ws\_partial}$.

```lean
theorem partial_conv_le_full_conv {d : ℕ} (c : Fin d → ℤ) (hc : ∀ i, 0 ≤ c i)
    (p : ℕ) (hp : 2 * p ≤ d) (t : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i.1 < 2*p ∧ j.1 < 2*p then c i * c j else 0 ≤
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c i * c j else 0 := by
  sorry
```

### Inequality 2: $W_\text{int}(c') \leq W_\text{int,max}$ for all children $c'$ in subtree

**Theorem.** For any window $(\ell, s_\text{lo})$ with contributing bin range $[\text{lo\_bin}, \text{hi\_bin}]$:

$$\sum_{i=\text{lo\_bin}}^{\text{hi\_bin}} c'_i \leq W_\text{int,max}$$

where $W_\text{int,max} = \sum_{\text{fixed } i \in [\text{lo\_bin},\text{hi\_bin}]} c'_i + \sum_{\text{unfixed parent } q : 2q \text{ or } 2q+1 \in [\text{lo\_bin},\text{hi\_bin}]} \text{parent}[q]$.

**Proof.** For each unfixed parent position $q \geq p$: child bins $2q$ and $2q+1$ satisfy $c'_{2q} + c'_{2q+1} = \text{parent}[q]$ and $c'_{2q}, c'_{2q+1} \geq 0$.

If both $2q$ and $2q+1$ are in $[\text{lo\_bin}, \text{hi\_bin}]$: contribute $c'_{2q} + c'_{2q+1} = \text{parent}[q]$. Exact match.

If only one (say $2q$) is in range: contribute $c'_{2q} \leq c'_{2q} + c'_{2q+1} = \text{parent}[q]$. Upper bound holds.

If neither: contribute 0. Upper bound holds.

So the actual $W_\text{int}$ is $\leq W_\text{int,max}$.

```lean
theorem w_int_bounded {d : ℕ} (child : Fin d → ℕ) (parent : Fin (d/2) → ℕ)
    (p : ℕ) (hp : 2*p ≤ d)
    (h_split : ∀ q : Fin (d/2), q.1 ≥ p →
      child ⟨2*q.1, by omega⟩ + child ⟨2*q.1+1, by omega⟩ = parent q)
    (lo hi : ℕ) (hlo : lo ≤ hi) (hhi : hi < d) :
    ∑ i ∈ Finset.Icc lo hi, (child ⟨i, by omega⟩ : ℕ) ≤
    (∑ i ∈ Finset.Icc lo (min hi (2*p-1)), (child ⟨i, by omega⟩ : ℕ)) +
    (∑ q ∈ Finset.filter (fun q => 2*q ≤ hi ∧ lo ≤ 2*q+1)
      (Finset.Icc p (d/2 - 1)), (parent ⟨q, by omega⟩ : ℕ)) := by
  sorry
```

### Inequality 3: $\text{dyn\_it}(W)$ is non-decreasing in $W$

**Theorem.** $W_1 \leq W_2 \implies \text{dyn\_it}(W_1) \leq \text{dyn\_it}(W_2)$.

**Proof.** $\text{dyn\_it}(W) = \lfloor (\text{base} + 2W) \cdot s \rfloor$ where $\text{base} > 0$ and $s > 0$.

$(\text{base} + 2W) \cdot s$ is strictly increasing in $W$ (coefficient $2s > 0$).

$\lfloor \cdot \rfloor$ is non-decreasing. Composition of increasing then non-decreasing is non-decreasing.

```lean
theorem dyn_it_mono (base s : ℝ) (hs : 0 < s) (W1 W2 : ℝ) (hW : W1 ≤ W2) :
    ⌊(base + 2 * W1) * s⌋ ≤ ⌊(base + 2 * W2) * s⌋ := by
  apply Int.floor_le_floor
  apply mul_le_mul_of_nonneg_right
  · linarith
  · exact le_of_lt hs
```

### Chain Conclusion

**Theorem.** If $\text{ws\_partial} > \text{dyn\_it}(W_\text{int,max})$, then for all children $c'$ in the subtree: $\text{ws\_full}(c') > \text{dyn\_it}(W_\text{int}(c'))$.

**Proof.** Chain the three inequalities:

$$\text{ws\_full}(c') \geq \text{ws\_partial} > \text{dyn\_it}(W_\text{int,max}) \geq \text{dyn\_it}(W_\text{int}(c'))$$

The strict inequality is preserved because $\geq$ composed with $>$ gives $>$.

```lean
theorem subtree_pruning_chain (ws_partial ws_full dyn_max dyn_actual : ℤ)
    (h1 : ws_full ≥ ws_partial)
    (h2 : ws_partial > dyn_max)
    (h3 : dyn_max ≥ dyn_actual) :
    ws_full > dyn_actual := by
  omega
```
