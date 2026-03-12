# Prompt 4: Optimization Soundness A — Fused Kernel, Incremental Autoconvolution, Quick-Check, Subtree Pruning

Prove Claims 4.1–4.4: the main algorithmic optimizations do not change the mathematical result. Attach `output.lean` as context — it contains all definitions and foundational lemmas already proved.

---

## Problem Context

We are proving $c \geq 1.4$ where:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

The proof uses a cascade: enumerate step functions, prune those satisfying $R(f) \geq c_{\text{target}}$. The following optimizations speed up the cascade without changing which compositions survive.

### Key Definitions

- **$d = 2n$ bins**, step function with masses $(c_0, \ldots, c_{d-1})$, $\sum c_i = m$.
- **Discrete autoconvolution:** $\text{conv}[k] = \sum_{i+j=k} c_i c_j$ for $k = 0, \ldots, 2d-2$.
- **Test value:** $\text{TV}(\ell, s_{\text{lo}}) = \frac{1}{4n\ell} \sum_{k=s_{\text{lo}}}^{s_{\text{lo}}+\ell-2} \text{conv}[k]$.
- **Window sum (integer):** $\text{ws} = \sum_{k=s_{\text{lo}}}^{s_{\text{lo}}+\ell-2} \text{conv}[k]$ (exact integer).
- **Dynamic threshold (integer):** $\text{dyn\_it}(\ell, W_{\text{int}})$ — floor of scaled threshold.
- **Pruning condition:** A child is pruned if $\exists (\ell, s_{\text{lo}})$ with $\text{ws} > \text{dyn\_it}$.
- **Child generation:** Parent $(c_0, \ldots, c_{d-1})$ produces children $(a_0, c_0-a_0, \ldots, a_{d-1}, c_{d-1}-a_{d-1})$ via Cartesian product of per-bin choices.

### Already Proved in output.lean (available as context)

All formal definitions established. Key lemmas: $\sum c_i = m$, bin masses nonneg, monotonicity of cumulative distribution.

---

## Claim 4.1: Fused Generate-and-Prune Kernel Equivalence

**Theorem to prove:**

The fused kernel (generate one child, immediately test, discard or store) produces the **same set** of survivors as the two-phase approach (generate ALL children, then test each one).

**Formal statement:** Let $\mathcal{C}$ be the set of all valid children (Cartesian product of per-bin choices). Let $P(c) = \text{True}$ iff child $c$ passes all pruning tests. Then:

$$\{c \in \mathcal{C} : P(c)\}_{\text{fused}} = \{c \in \mathcal{C} : P(c)\}_{\text{two-phase}}$$

**What must be shown:**

1. **Completeness:** The odometer iterator visits every element of $\prod_{i=0}^{d-1} [\text{lo}_i, \text{hi}_i]$ exactly once.

   *Proof:* An odometer on ranges $[\text{lo}_0, \text{hi}_0] \times \ldots \times [\text{lo}_{d-1}, \text{hi}_{d-1}]$ increments the last coordinate, carrying to the next when it wraps. This visits $\prod_i (\text{hi}_i - \text{lo}_i + 1)$ distinct tuples in lexicographic order with no repeats.

2. **Determinism:** The pruning predicate $P(c)$ depends only on the child $c$ (and fixed parameters), not on the iteration order.

   *Proof:* $P(c)$ checks (a) asymmetry of left-half mass (deterministic function of $c$) and (b) whether any window sum exceeds the dynamic threshold (deterministic function of $c$). No state from previous iterations affects $P$.

**Lean theorem statement:**

```lean
-- Odometer completeness: visits every element exactly once
theorem odometer_complete {d : ℕ} (lo hi : Fin d → ℕ) (h : ∀ i, lo i ≤ hi i) :
    odometer_elements lo hi = Finset.pi Finset.univ (fun i => Finset.Icc (lo i) (hi i)) := by
  sorry

-- Fused = two-phase
theorem fused_eq_twophase {d : ℕ} (parent : Fin d → ℕ) (x_cap : ℕ) (prune : (Fin (2*d) → ℕ) → Bool) :
    fused_survivors parent x_cap prune = twophase_survivors parent x_cap prune := by
  sorry
```

---

## Claim 4.2: Incremental Autoconvolution Update Correctness

**Theorem to prove:**

When consecutive children in the odometer differ in $k$ parent-bin positions (the last $k$ bins carry), the autoconvolution $\text{conv}[t] = \sum_{i+j=t} c_i c_j$ can be updated incrementally. The result is **bit-exact** (identical to full recomputation).

### Three update paths:

**Path 1: Fast path (1 position changed, positions $2p$ and $2p+1$ change).**

Let $c'$ be the new child, $c$ the old. Only $c'_{2p} \neq c_{2p}$ and $c'_{2p+1} \neq c_{2p+1}$ (with $c'_{2p} + c'_{2p+1} = c_{2p} + c_{2p+1} = c_{\text{parent},p}$). Define:

$$\delta_a = c'_{2p} - c_{2p}, \quad \delta_b = c'_{2p+1} - c_{2p+1} = -\delta_a$$

The update to $\text{conv}[t]$ is:

$$\text{conv}'[t] = \text{conv}[t] + \underbrace{[(c'_{2p})^2 - (c_{2p})^2]}_{\text{self-term } 2p} \cdot [t = 4p]$$
$$+ \underbrace{[(c'_{2p+1})^2 - (c_{2p+1})^2]}_{\text{self-term } 2p+1} \cdot [t = 4p+2]$$
$$+ \underbrace{2[(c'_{2p} c'_{2p+1}) - (c_{2p} c_{2p+1})]}_{\text{mutual term}} \cdot [t = 4p+1]$$
$$+ \underbrace{\sum_{q \neq 2p, 2p+1} 2[\delta_a \cdot c_q \cdot [t = 2p+q] + \delta_b \cdot c_q \cdot [t = 2p+1+q]]}_{\text{cross-terms}}$$

**Path 2: Short carry ($2 \leq k \leq$ threshold).**

Same structure as fast path, extended to multiple changed position pairs. The terms decompose into three disjoint groups:
- (a) Self + mutual within each changed pair
- (b) Cross-terms between different changed pairs
- (c) Cross-terms between changed and unchanged bins

These three groups are **disjoint** (they affect different $(i,j)$ pairs in $\sum_{i+j=t} c_i c_j$) and **exhaustive** (cover all terms that changed from $c$ to $c'$).

**Path 3: Deep carry ($k >$ threshold).**

Full $O(d^2)$ recompute: $\text{conv}[t] = \sum_{i+j=t} c_i c_j$. This is the reference formula — trivially correct.

**What to prove for Paths 1 and 2:** After the incremental update, $\text{conv}'[t] = \sum_{i+j=t} c'_i c'_j$ for all $t$.

**Proof strategy for fast path:**

$\text{conv}'[t] - \text{conv}[t] = \sum_{i+j=t} c'_i c'_j - \sum_{i+j=t} c_i c_j$

Since $c'_q = c_q$ for $q \notin \{2p, 2p+1\}$, the difference only involves terms where at least one of $i, j$ is in $\{2p, 2p+1\}$:

$$= \sum_{\substack{i+j=t \\ i \in \{2p,2p+1\} \text{ or } j \in \{2p,2p+1\}}} (c'_i c'_j - c_i c_j)$$

Expand by cases (both in changed set, one in changed set) to recover the delta formula above.

**Lean theorem statement:**

```lean
-- After incremental update, conv is correct
theorem incremental_autoconv_correct {d : ℕ} (c c' : Fin d → ℤ)
    (conv : ℕ → ℤ) (conv' : ℕ → ℤ)
    (h_conv : ∀ t, conv t = ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c i * c j else 0)
    (h_update : ∀ t, conv' t = conv t + delta_terms c c' t)
    (h_delta_correct : ∀ t, delta_terms c c' t =
      ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c' i * c' j - c i * c j else 0) :
    ∀ t, conv' t = ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c' i * c' j else 0 := by
  sorry
```

---

## Claim 4.3: Quick-Check Soundness

**Theorem to prove:**

After a child is pruned by window $(\ell^*, s^*_{\text{lo}})$, the next child is first tested against that **same** window. If $\text{ws}(\ell^*, s^*_{\text{lo}}) > \text{dyn\_it}(\ell^*, W^*_{\text{int}})$, the child is pruned without running the full window scan.

This is sound because:

1. **Sufficiency:** If any single window exceeds its threshold, the child is prunable (the pruning condition is existential — "there exists a killing window").

2. **Completeness on miss:** If the quick-check does NOT kill, the full scan over all windows runs. No windows are skipped.

3. **$W_{\text{int}}$ correctness:** The contributing-bin mass $W^*_{\text{int}}$ for the quick-check window is maintained exactly:
   - **Fast path:** $O(1)$ update: if the changed parent position $p$ has child bins $2p, 2p+1$ in the window's contributing range $[\text{lo\_bin}, \text{hi\_bin}]$, add/subtract the delta.
   - **Short/deep carry:** Full recompute of $W^*_{\text{int}}$ from the child composition.

**Formal statement:**

$$\text{ws}(\ell^*, s^*_{\text{lo}}, c') > \text{dyn\_it}(\ell^*, W^*_{\text{int}}(c')) \implies \exists (\ell, s_{\text{lo}}): \text{ws}(\ell, s_{\text{lo}}, c') > \text{dyn\_it}(\ell, W_{\text{int}}(c'))$$

which is trivially true by taking $(\ell, s_{\text{lo}}) = (\ell^*, s^*_{\text{lo}})$.

The non-trivial part is proving that $W^*_{\text{int}}(c')$ as tracked by the incremental update equals the true $\sum_{i \in \mathcal{B}(\ell^*, s^*_{\text{lo}})} c'_i$.

**Lean theorem statement:**

```lean
-- Quick-check is a special case of full scan
theorem quick_check_sound {d : ℕ} (c : Fin d → ℤ) (ℓ_star s_lo_star : ℕ)
    (h_kill : window_sum c ℓ_star s_lo_star > dyn_threshold ℓ_star (w_int c ℓ_star s_lo_star)) :
    ∃ ℓ s_lo, window_sum c ℓ s_lo > dyn_threshold ℓ (w_int c ℓ s_lo) := by
  exact ⟨ℓ_star, s_lo_star, h_kill⟩
```

---

## Claim 4.4: Subtree Pruning Soundness

**Theorem to prove:**

On a deep carry (many positions change), if the partial autoconvolution of the fixed bins already exceeds the threshold with a worst-case $W_{\text{int,max}}$, then ALL children in the subtree can be pruned.

### Setup

- **Fixed bins:** $0, \ldots, 2p-1$ (determined by the odometer prefix). These are identical for all children in the subtree.
- **Unfixed bins:** $2p, \ldots, d-1$ (vary across subtree children).
- **Partial autoconvolution:** $\text{partial\_conv}[t] = \sum_{\substack{i+j=t \\ i,j < 2p}} c_i c_j$ (only fixed-bin terms).
- **Partial window sum:** $\text{ws\_partial} = \sum_{k=s_{\text{lo}}}^{s_{\text{lo}}+\ell-2} \text{partial\_conv}[k]$ for windows fully within the fixed range.
- **$W_{\text{int,max}}$:** Upper bound on contributing mass. For fixed parent positions $q < p$: exact child masses. For unfixed positions $q \geq p$: use $\text{parent}[q]$ as the upper bound on any single child bin pair's total mass.

### Three inequalities to prove

**Inequality 1: $\text{ws\_full}(c') \geq \text{ws\_partial}$ for all children $c'$ in subtree.**

*Proof:* For any index $t$ with $i + j = t$ and $i, j < 2p$:
$$\text{conv}[t](c') = \sum_{\substack{i+j=t}} c'_i c'_j = \text{partial\_conv}[t] + \underbrace{\sum_{\substack{i+j=t \\ i \geq 2p \text{ or } j \geq 2p}} c'_i c'_j}_{\geq 0}$$

The additional terms involve at least one unfixed bin. Since all $c'_i \geq 0$, these terms are $\geq 0$. Summing over the window: $\text{ws\_full} \geq \text{ws\_partial}$.

**Inequality 2: $W_{\text{int}}(c') \leq W_{\text{int,max}}$ for all children $c'$ in subtree.**

*Proof:* For each unfixed parent position $q \geq p$: $c'_{2q} + c'_{2q+1} = \text{parent}[q]$. If either child bin $2q$ or $2q+1$ is in the contributing range $\mathcal{B}$, its mass is at most $\text{parent}[q]$. So $W_{\text{int,max}}$ computed using parent masses for unfixed bins is an upper bound.

**Inequality 3: $\text{dyn\_it}(W)$ is non-decreasing in $W$.**

*Proof:* $\text{dyn\_it} = \lfloor (\text{base} + 2W) \cdot \text{scale} \rfloor$ where base $> 0$ and scale $> 0$. The argument of $\lfloor\cdot\rfloor$ is a strictly increasing linear function of $W$, so $\lfloor\cdot\rfloor$ is non-decreasing.

### Chain conclusion

If $\text{ws\_partial} > \text{dyn\_it}(W_{\text{int,max}})$, then for any child $c'$ in the subtree:

$$\text{ws\_full}(c') \geq \text{ws\_partial} > \text{dyn\_it}(W_{\text{int,max}}) \geq \text{dyn\_it}(W_{\text{int}}(c'))$$

So the child would be pruned by the full scan. Pruning the entire subtree is sound.

**Lean theorem statements:**

```lean
-- Inequality 1: full conv >= partial conv (nonneg terms)
theorem partial_conv_le_full {d p : ℕ} (c : Fin d → ℤ) (hp : 2 * p ≤ d)
    (hc_nonneg : ∀ i, 0 ≤ c i) (t : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t ∧ i.1 < 2*p ∧ j.1 < 2*p then c i * c j else 0 ≤
    ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c i * c j else 0 := by
  sorry

-- Inequality 3: dyn_it is monotone in W
theorem dyn_it_monotone (base scale : ℝ) (hscale : 0 < scale) :
    Monotone (fun W => ⌊(base + 2 * W) * scale⌋) := by
  sorry

-- Full subtree pruning soundness
theorem subtree_pruning_sound {d p : ℕ} (hp : 2 * p ≤ d)
    (fixed : Fin (2*p) → ℤ) (parent : Fin (d/2) → ℤ)
    (ws_partial : ℤ) (W_int_max : ℤ)
    (h_ws : ws_partial > dyn_it W_int_max)
    (h_ws_partial_le : ∀ child, is_subtree_child fixed parent child → ws_full child ≥ ws_partial)
    (h_wint_bound : ∀ child, is_subtree_child fixed parent child → w_int child ≤ W_int_max)
    (h_mono : ∀ w1 w2, w1 ≤ w2 → dyn_it w1 ≤ dyn_it w2) :
    ∀ child, is_subtree_child fixed parent child → ws_full child > dyn_it (w_int child) := by
  sorry
```
