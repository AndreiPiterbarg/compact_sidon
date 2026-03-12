# Prompt 8: Canonical Symmetry Reduction

**Claim 3.3.** Prove that exploiting reversal symmetry is sound. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. The cascade exploits the symmetry $\text{TV}(c) = \text{TV}(\text{rev}(c))$ to halve the search space. We must prove this symmetry is valid and the reduction is complete.

### Definitions (in output.lean)

- `discrete_autoconvolution a k = ∑_{i+j=k} a_i * a_j`
- `test_value n m c ℓ s_lo`
- `max_test_value n m c`

### Key definition (not in output.lean — define it)

```lean
def rev_comp {d : ℕ} (c : Fin d → ℕ) : Fin d → ℕ :=
  fun i => c ⟨d - 1 - i.1, by omega⟩

def is_canonical {d : ℕ} (c : Fin d → ℕ) : Prop :=
  ∀ i : Fin d, c i < c (⟨d - 1 - i.1, by omega⟩) →
    ∃ j : Fin d, j < i ∧ c j > c (⟨d - 1 - j.1, by omega⟩)
-- (c ≤ rev(c) lexicographically)
```

---

## Claim 3.3a: Autoconvolution Reversal Symmetry

**Theorem.** $\text{conv}[k](c) = \text{conv}[2d-2-k](\text{rev}(c))$.

**Proof.** Let $c' = \text{rev}(c)$, so $c'_i = c_{d-1-i}$.

$$\text{conv}[2d-2-k](c') = \sum_{i+j=2d-2-k} c'_i c'_j = \sum_{i+j=2d-2-k} c_{d-1-i} c_{d-1-j}$$

Substitute $i' = d-1-i$, $j' = d-1-j$. Then $i'+j' = 2d-2-(i+j) = 2d-2-(2d-2-k) = k$.

$$= \sum_{i'+j'=k} c_{i'} c_{j'} = \text{conv}[k](c)$$

```lean
theorem autoconv_rev_symmetry {d : ℕ} (c : Fin d → ℝ) (k : ℕ) (hk : k ≤ 2 * d - 2) (hd : d > 0) :
    discrete_autoconvolution c k =
    discrete_autoconvolution (fun i => c ⟨d - 1 - i.1, by omega⟩) (2 * d - 2 - k) := by
  sorry
```

## Claim 3.3b: Max Test Value Reversal Symmetry

**Theorem.** $b_{n,m}(c) = b_{n,m}(\text{rev}(c))$.

**Proof.** From 3.3a, the window sum over $[s_\text{lo}, s_\text{lo}+\ell-2]$ for $c$ equals the window sum over $[2d-2-(s_\text{lo}+\ell-2), 2d-2-s_\text{lo}]$ for $\text{rev}(c)$. The reflected window has the same length $\ell$. As $s_\text{lo}$ ranges over all valid positions, so does the reflected position. The window size $\ell$ is the same. So the set of test values is identical.

```lean
theorem max_test_value_rev_symmetry (n m : ℕ) (hn : n > 0) (c : Fin (2 * n) → ℕ) :
    max_test_value n m c = max_test_value n m (fun i => c ⟨2 * n - 1 - i.1, by omega⟩) := by
  sorry
```

## Claim 3.3c: Level 0 Canonical Completeness

**Theorem.** At level 0, enumerating only canonical compositions (where $c \leq \text{rev}(c)$ lex) is sufficient. For every non-canonical $c$, $\text{rev}(c)$ is canonical and has the same max test value.

**Proof.** For any $c$: either $c = \text{rev}(c)$ (palindrome), $c < \text{rev}(c)$ lex ($c$ is canonical), or $c > \text{rev}(c)$ lex ($\text{rev}(c)$ is canonical). In all cases, exactly one of $\{c, \text{rev}(c)\}$ is canonical (or both, if palindrome). By 3.3b, both have the same test value.

```lean
-- For any composition, min(c, rev(c)) is canonical
theorem canonical_exists {d : ℕ} (c : Fin d → ℕ) :
    ∃ c' : Fin d → ℕ, (c' = c ∨ c' = fun i => c ⟨d - 1 - i.1, by omega⟩) ∧
    -- c' ≤ rev(c') lexicographically
    True := by  -- placeholder for lex condition
  sorry
```

## Claim 3.3d: Refinement-Level Canonical Handling

**Theorem.** If $P$ is canonical and $C$ is a child of $\text{rev}(P)$, then $\text{rev}(C)$ is a child of $P$.

**Proof.** $P$ has bins $(p_0, \ldots, p_{d-1})$. $\text{rev}(P)$ has bins $(p_{d-1}, \ldots, p_0)$.

A child of $\text{rev}(P)$ splits each bin: $\text{rev}(P)_i = p_{d-1-i}$ into $(a_i, p_{d-1-i} - a_i)$.

The child is $C = (a_0, p_{d-1}-a_0, a_1, p_{d-2}-a_1, \ldots)$.

$\text{rev}(C)$ reverses this: $\text{rev}(C)_{2j} = C_{2d-1-2j} = p_j - a_{d-1-j}$ and $\text{rev}(C)_{2j+1} = C_{2d-2-2j} = a_{d-1-j}$.

Define $b_j = p_j - a_{d-1-j}$. Then $\text{rev}(C)$ has bins $(b_0, p_0 - b_0, b_1, p_1 - b_1, \ldots)$ where $b_j + (p_j - b_j) = p_j$. This is a valid child of $P$.

So generating all children of each canonical $P$ covers all children of $\text{rev}(P)$ (up to reversal). After canonicalizing survivors, all equivalence classes are represented.

```lean
-- Reversal of a child of rev(P) is a child of P
theorem rev_child_is_child {d : ℕ} (parent : Fin d → ℕ)
    (a : Fin d → ℕ) (ha : ∀ i, a i ≤ parent ⟨d - 1 - i.1, by omega⟩)
    (child_of_rev : Fin (2 * d) → ℕ)
    (hc : ∀ i : Fin d, child_of_rev ⟨2*i.1, by omega⟩ = a i ∧
          child_of_rev ⟨2*i.1+1, by omega⟩ = parent ⟨d-1-i.1, by omega⟩ - a i) :
    ∃ b : Fin d → ℕ, (∀ i, b i ≤ parent i) ∧
      ∀ i : Fin d,
        (fun j : Fin (2*d) => child_of_rev ⟨2*d-1-j.1, by omega⟩) ⟨2*i.1, by omega⟩ = b i ∧
        (fun j : Fin (2*d) => child_of_rev ⟨2*d-1-j.1, by omega⟩) ⟨2*i.1+1, by omega⟩ = parent i - b i := by
  sorry
```

## Claim 3.3e: Asymmetry is Reversal-Symmetric

**Theorem.** $\text{left\_frac}(\text{rev}(c)) = 1 - \text{left\_frac}(c)$.

**Proof.** $\text{left\_frac}(c) = \frac{1}{m}\sum_{i=0}^{n-1} c_i$ and $\text{left\_frac}(\text{rev}(c)) = \frac{1}{m}\sum_{i=0}^{n-1} c_{d-1-i} = \frac{1}{m}\sum_{i=n}^{d-1} c_i = \frac{1}{m}(m - \sum_{i=0}^{n-1} c_i) = 1 - \text{left\_frac}(c)$.

The asymmetry condition prunes when $\text{left\_frac} \geq \sqrt{c_\text{target}/2}$ OR $1 - \text{left\_frac} \geq \sqrt{c_\text{target}/2}$. Since $\text{left\_frac}(\text{rev}(c)) = 1 - \text{left\_frac}(c)$, the condition is symmetric under reversal.

```lean
theorem left_frac_reversal (n m : ℕ) (hn : n > 0) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℤ)) +
    (∑ i : Fin n, (c ⟨2*n - 1 - i.1, by omega⟩ : ℤ)) = m := by
  sorry
```
