# Prompt 3: Cascade Completeness

Prove Claims 3.1–3.4: the branch-and-prune cascade covers every possible step function — no composition escapes without being either pruned or having all its descendants pruned. Attach `output.lean` as context — it contains all definitions and foundational lemmas already proved.

---

## Problem Context

We are proving $c \geq 1.4$ where:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

The proof uses a **cascade** (branch-and-prune): start with all step functions on a coarse grid ($d_0 = 2n$ bins), prune those provably satisfying $R(f) \geq c_{\text{target}}$, then refine survivors to a finer grid ($d_1 = 2d_0$ bins) and repeat.

### Key Definitions

- **$d = 2n$ bins** of width $\Delta = 1/(4n)$ on $(-1/4, 1/4)$.
- **Composition:** $(c_0, \ldots, c_{d-1}) \in \mathbb{Z}_{\geq 0}^d$ with $\sum c_i = m$. Count: $\binom{m+d-1}{d-1}$.
- **Refinement:** Parent $(c_0, \ldots, c_{d-1})$ at resolution $d$ maps to child $(a_0, c_0-a_0, a_1, c_1-a_1, \ldots)$ at resolution $2d$ where $0 \leq a_i \leq c_i$.
- **$x_{\text{cap}}$:** Max allowed bin mass. Children with any bin $> x_{\text{cap}}$ are skipped (proved safe by Claim 2.3 in a separate prompt).
- **Canonical:** $c$ is canonical if $c \leq \text{rev}(c)$ lexicographically, where $\text{rev}(c) = (c_{d-1}, \ldots, c_0)$.
- **Test value:** $b_{n,m}(c) = \max_{\ell, s_{\text{lo}}} \text{TV}(\ell, s_{\text{lo}})$ — lower bound on $\|f*f\|_\infty$.

### Already Proved in output.lean (available as context)

All formal definitions established. Key lemmas: $D(0)=0$, $D(2n)=m$, $\sum c_i = m$, monotonicity, bin masses nonneg.

---

## Claim 3.1: Composition Enumeration is Complete

**Theorem to prove:**

The set of all compositions of $m$ into $d$ nonneg integer parts has cardinality $\binom{m+d-1}{d-1}$ (stars and bars). The enumeration algorithm at level 0 produces exactly this set with no duplicates and no omissions.

**Stars-and-bars proof:**

Each composition $(c_0, \ldots, c_{d-1})$ with $\sum c_i = m$ bijects to a subset of $d-1$ elements from $\{1, \ldots, m+d-1\}$ via: place $m$ stars and $d-1$ bars in a row; $c_i$ = number of stars between bar $i-1$ and bar $i$. The number of such subsets is $\binom{m+d-1}{d-1}$.

**Lean theorem statement:**

```lean
-- The number of compositions of m into d nonneg parts
theorem composition_count (m d : ℕ) (hd : d > 0) :
    Finset.card {c : Fin d → ℕ | ∑ i, c i = m}.toFinset = Nat.choose (m + d - 1) (d - 1) := by
  sorry
```

---

## Claim 3.2: Child Generation is Complete

**Theorem to prove:**

At each refinement level, the set of children of parent $(c_0, \ldots, c_{d-1})$ with all bins $\leq x_{\text{cap}}$ is:

$$\text{Children}(c) = \{(a_0, c_0-a_0, a_1, c_1-a_1, \ldots) : \forall i,\; \max(0, c_i - x_{\text{cap}}) \leq a_i \leq \min(c_i, x_{\text{cap}})\}$$

**Three sub-claims:**

**3.2a: Cartesian product completeness.** The children form a Cartesian product over $d$ independent per-bin choice sets $S_i = \{a_i : \max(0, c_i - x_{\text{cap}}) \leq a_i \leq \min(c_i, x_{\text{cap}})\}$. The enumeration visits every element of $\prod_i S_i$ exactly once.

**3.2b: x_cap pruning is safe.** Any child with some bin $> x_{\text{cap}}$ satisfies $R(f) \geq c_{\text{target}}$ (by Claim 2.3, proved in a separate prompt). So skipping children with bins $> x_{\text{cap}}$ is sound.

**3.2c: Child layout is valid.** The child $(a_0, c_0-a_0, a_1, c_1-a_1, \ldots)$ at positions $(2i, 2i+1)$ represents a valid refinement: the left sub-bin of parent bin $i$ gets mass $a_i$, the right sub-bin gets $c_i - a_i$, and $a_i + (c_i - a_i) = c_i$ preserves total mass. Since $\max(0, c_i - x_{\text{cap}}) \leq a_i$, both sub-bins are $\leq x_{\text{cap}}$.

**Lean theorem statements:**

```lean
-- 3.2a: Number of children
theorem children_count (d m : ℕ) (parent : Fin d → ℕ) (x_cap : ℕ)
    (hp : ∑ i, parent i = m) :
    Finset.card (children_set parent x_cap) =
    ∏ i : Fin d, (min (parent i) x_cap - max 0 (parent i - x_cap) + 1) := by
  sorry

-- 3.2c: Children preserve total mass
theorem child_sum_eq_m (d m : ℕ) (parent : Fin d → ℕ) (hp : ∑ i, parent i = m)
    (child : Fin (2 * d) → ℕ)
    (h_split : ∀ i : Fin d, child ⟨2*i.1, by omega⟩ + child ⟨2*i.1+1, by omega⟩ = parent i) :
    ∑ i, child i = m := by
  sorry
```

---

## Claim 3.3: Canonical Symmetry Reduction is Sound

**Theorem to prove (five sub-claims):**

**3.3a: Test value reversal symmetry.**

$$\text{TV}(\ell, s_{\text{lo}}, c) = \text{TV}(\ell, s'_{\text{lo}}, \text{rev}(c))$$

where $s'_{\text{lo}} = 2d - 2 - (s_{\text{lo}} + \ell - 2) = 2d - \ell - s_{\text{lo}}$.

*Proof:* The autoconvolution satisfies $\text{conv}[k](c) = \text{conv}[2d - 2 - k](\text{rev}(c))$ because $\sum_{i+j=k} c_i c_j = \sum_{i'+j'=2d-2-k} c_{d-1-i'} c_{d-1-j'}$. The window sum over $[s_{\text{lo}}, s_{\text{lo}}+\ell-2]$ for $c$ equals the window sum over the reflected range for $\text{rev}(c)$.

**3.3b:** Consequently, $b_{n,m}(c) = b_{n,m}(\text{rev}(c))$.

**3.3c: Level 0 canonical completeness.** At level 0, for every non-canonical $c$, the canonical partner $\text{rev}(c)$ satisfies $\text{rev}(c) \leq c$ (lex), so $\text{rev}(c)$ IS canonical and IS enumerated. By 3.3b, it has the same max test value, so pruning it is equivalent.

**3.3d: Refinement level handling.** At refinement levels: if parent $P$ is canonical, its children include children of $\text{rev}(P)$ (via the mapping that reverses the child). Since $\text{rev}(P)$ is NOT separately in the parent list, we must generate ALL children of $P$ (not just canonical ones), then canonicalize survivors and deduplicate.

*Proof:* Let $C$ be a child of $\text{rev}(P)$. Define $C' = \text{rev}(C)$. Then $C'$ is a child of $P$ (the reversal of the parent reverses the child layout). So every composition at the finer grid is either a child of some canonical parent or the reverse of such a child. The canonicalize-then-dedup step ensures exactly one representative per $\{c, \text{rev}(c)\}$ pair.

**3.3e: Asymmetry is reversal-symmetric.** $\text{left\_frac}(\text{rev}(c)) = 1 - \text{left\_frac}(c)$, and the asymmetry condition checks both $L \geq \sqrt{c_{\text{target}}/2}$ and $1-L \geq \sqrt{c_{\text{target}}/2}$.

**Lean theorem statements:**

```lean
-- 3.3a: Autoconvolution reversal symmetry
theorem discrete_autoconvolution_rev {d : ℕ} (a : Fin d → ℝ) (k : ℕ) (hk : k ≤ 2 * d - 2) :
    discrete_autoconvolution a k =
    discrete_autoconvolution (fun i => a ⟨d - 1 - i.1, by omega⟩) (2 * d - 2 - k) := by
  sorry

-- 3.3b: Max test value reversal symmetry
theorem max_test_value_rev (n m : ℕ) (c : Fin (2 * n) → ℕ) :
    max_test_value n m c = max_test_value n m (fun i => c ⟨2 * n - 1 - i.1, by omega⟩) := by
  sorry

-- 3.3e: Left fraction reversal
theorem left_frac_rev (n m : ℕ) (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (hm : m > 0) :
    (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) / m +
    (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) / m = 1 := by
  sorry
```

---

## Claim 3.4: Cascade Induction — Complete Coverage

**Theorem to prove:**

The cascade proves $c \geq c_{\text{target}}$ by the following induction:

**Base (Level 0):** Every composition of $m$ into $d_0 = 2n$ parts is either:
- Pruned (proved $R(f) \geq c_{\text{target}}$), OR
- Passed to Level 1 as a surviving parent.

**Step (Level $L \to L+1$):** Every child of every Level-$L$ survivor at resolution $d_L = 2^L \cdot d_0$ is either:
- Pruned, OR
- Passed to Level $L+1$.

**Termination:** If Level $L$ has 0 survivors, then every composition at resolution $d_L$ is covered (either directly pruned or an ancestor was pruned at an earlier level with all its descendants also pruned).

**Formal completeness:** Let $\mathcal{S}_L$ be the survivors at level $L$. Then:
$$\mathcal{S}_0 \supseteq \text{ancestors}(\mathcal{S}_1) \supseteq \text{ancestors}(\mathcal{S}_2) \supseteq \ldots$$

and every composition at resolution $d_L$ that is NOT a descendant of some $\mathcal{S}_L$ survivor was pruned at level $\leq L$.

**Proof sketch:**

1. At level 0, the full set $\mathcal{C}_0$ of $\binom{m+d_0-1}{d_0-1}$ compositions is enumerated (Claim 3.1). Pruning removes a subset; survivors $\mathcal{S}_0 = \mathcal{C}_0 \setminus \text{pruned}$.

2. At level $L$, for each parent $p \in \mathcal{S}_{L-1}$, ALL children of $p$ with bins $\leq x_{\text{cap}}$ are generated (Claim 3.2). Children with bins $> x_{\text{cap}}$ are safely skipped (Claim 2.3). After pruning and deduplication, survivors $\mathcal{S}_L$ emerge.

3. For canonical symmetry (Claim 3.3): the canonical-parent-only enumeration at L0 plus the all-children generation at L1+ ensures every equivalence class $\{c, \text{rev}(c)\}$ has at least one representative tested.

4. If $\mathcal{S}_L = \emptyset$ for some $L$, then every composition at resolution $d_L$ is a descendant of some composition pruned at level $\leq L$. Since pruning is sound (Claims 2.1–2.4), $R(f) \geq c_{\text{target}}$ for all step functions at resolution $d_L$, and by density of step functions, $c \geq c_{\text{target}}$.

**Lean theorem statement (high level):**

```lean
-- If the cascade terminates with 0 survivors at some level,
-- then every nonneg f supported on (-1/4, 1/4) satisfies R(f) >= c_target
theorem cascade_completeness (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (L : ℕ) -- the level where survivors = 0
    (h_zero_survivors : survivors_at_level n m c_target L = ∅)
    (h_pruning_sound : ∀ c, pruned n m c_target c → autoconvolution_ratio_step c ≥ c_target)
    (h_enumeration_complete : ∀ c, is_composition_at_level n m L c →
      c ∈ survivors_at_level n m c_target L ∨ ∃ k ≤ L, ancestor_pruned n m c_target k c) :
    autoconvolution_constant ≥ c_target := by
  sorry
```
