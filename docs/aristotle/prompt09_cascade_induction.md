# Prompt 9: Cascade Induction — Complete Coverage

**Claim 3.4.** The cascade proves $c \geq c_\text{target}$ by covering every composition. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

The proof uses a **cascade**: starting from all compositions at a coarse grid, prune those provably satisfying $R(f) \geq c_\text{target}$, then refine survivors to a finer grid and repeat.

### Definitions

- $d_L = 2^L \cdot 2n$: grid resolution at level $L$.
- **Composition at level $L$:** $(c_0, \ldots, c_{d_L-1}) \in \mathbb{Z}_{\geq 0}^{d_L}$ with $\sum c_i = m$.
- **Pruning:** A composition $c$ is pruned at level $L$ if $R(f_c) \geq c_\text{target}$ is provable (via test value exceeding dynamic threshold, or asymmetry, or x_cap).
- **Refinement:** Each surviving parent at level $L$ generates children at level $L+1$ by splitting each bin into two.
- **Canonical symmetry:** At L0, only canonical compositions are enumerated. At L1+, all children are generated, then canonicalized and deduplicated.

### Assumptions (proved in other prompts)

1. **Pruning is sound:** If a composition is pruned, then $R(f) \geq c_\text{target}$ for all $f$ discretizing to it. (Claims 2.1–2.4)
2. **Enumeration is complete:** At L0, all $\binom{m+d_0-1}{d_0-1}$ compositions (or canonical half) are enumerated. (Claim 3.1)
3. **Child generation is complete:** Every child with bins $\leq x_\text{cap}$ is generated; those with bins $> x_\text{cap}$ are safely skipped. (Claims 3.2, 2.3)
4. **Canonical handling is sound:** The symmetry reduction doesn't lose any equivalence class. (Claim 3.3)
5. **Step functions are dense:** The infimum over nonneg $f$ can be approximated by step functions on finer grids. This is a standard result from real analysis.

---

## Claim 3.4: Cascade Completeness

**Theorem.** If the cascade terminates with 0 survivors at level $L$, then $c \geq c_\text{target}$.

### Proof

**Step 1: Every composition at level $L$ is covered.**

By induction on the cascade levels:

**Base ($L = 0$).** All compositions at resolution $d_0 = 2n$ are enumerated (Claim 3.1). Each is either pruned (guaranteed $R \geq c_\text{target}$) or survives to level 1.

**Inductive step ($L-1 \to L$).** Assume every composition at level $L-1$ is either pruned (all descendants safe) or a survivor. For each survivor $p$:
- All children of $p$ at level $L$ with bins $\leq x_\text{cap}$ are generated (Claim 3.2a).
- Children with bins $> x_\text{cap}$ have $R \geq c_\text{target}$ (Claim 2.3).
- Each generated child is either pruned or survives.

If 0 survivors remain at level $L$, every composition at resolution $d_L$ is accounted for.

**Step 2: Every composition at resolution $d_L$ is a descendant of exactly one composition at resolution $d_0$.**

A composition $(c_0, \ldots, c_{d_L-1})$ at level $L$ comes from a unique chain of ancestors:
- Level $L-1$ parent: $(c_0 + c_1, c_2 + c_3, \ldots)$ (merge consecutive pairs).
- Repeat to get level 0 ancestor.

This ancestor is unique because merging is deterministic. So the cascade's coverage at each level propagates down.

**Step 3: From discrete to continuous.**

If every composition at resolution $d_L$ has $R(f_c) \geq c_\text{target}$, we need to show $c \geq c_\text{target}$.

For any nonneg $f$ with $\int f = 1$ on $(-1/4, 1/4)$: discretize $f$ at resolution $d_L$ to get $\hat{c}$. By the correction term (Claim 1.2): $R(f) \geq b_{n_L, m}(\hat{c}) - 2/m - 1/m^2$.

But the cascade didn't just check $b > c_\text{target} + 2/m + 1/m^2$ uniformly — it used the dynamic threshold (Claim 1.3) which is tighter. Either way, if $\hat{c}$ was pruned, $R(f) \geq c_\text{target}$.

**Step 4: Density argument.**

As the grid resolution $d_L \to \infty$, the discretized step functions approximate $f$ in $L^1$. The correction term $2/m + 1/m^2$ is fixed (depends on $m$, not $d_L$). So at any finite level, if all compositions are pruned, the bound $R(f) \geq c_\text{target}$ holds for all $f$ at that discretization, and by the correction term, for the continuous $f$ as well.

**Conclusion.** $c = \inf R(f) \geq c_\text{target}$.

### Lean theorem statements

```lean
-- Ancestor is unique: merging consecutive pairs
def merge_pairs {d : ℕ} (child : Fin (2 * d) → ℕ) : Fin d → ℕ :=
  fun i => child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩

-- merge_pairs preserves total mass
theorem merge_pairs_sum {d m : ℕ} (child : Fin (2 * d) → ℕ) (hc : ∑ i, child i = m) :
    ∑ i, merge_pairs child i = m := by
  sorry

-- Cascade completeness (high-level structure)
-- If all compositions at resolution d_L are pruned, then c >= c_target
theorem cascade_completeness_step (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (L : ℕ)
    (h_all_pruned : ∀ c : Fin (2^(L+1) * n) → ℕ, ∑ i, c i = m →
      ∃ ℓ s_lo, test_value (2^L * n) m c ℓ s_lo > c_target + 2 / m + 1 / m^2) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f ≠ 0 →
      autoconvolution_ratio f ≥ c_target := by
  sorry
```
