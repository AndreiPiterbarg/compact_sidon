# Prompt 10: Fused Kernel Equivalence and Quick-Check Soundness

**Claims 4.1 + 4.3.** These are relatively simple equivalence/soundness arguments. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. The cascade generates children of each parent and prunes them. Two optimizations: (1) fusing generation with pruning, (2) quick-checking the previous killing window first.

### Definitions

- **Children of parent $(c_0,\ldots,c_{d-1})$:** Cartesian product $\prod_i [\text{lo}_i, \text{hi}_i]$ where $\text{lo}_i = \max(0, c_i - x_\text{cap})$, $\text{hi}_i = \min(c_i, x_\text{cap})$.
- **Pruning predicate $P(c)$:** True iff $\exists (\ell, s_\text{lo})$ with window sum $> $ dynamic threshold, OR left-half mass exceeds asymmetry threshold.
- **Odometer iterator:** Generates elements of $\prod_i [\text{lo}_i, \text{hi}_i]$ in lexicographic order by incrementing the last coordinate and carrying.

---

## Claim 4.1: Fused Kernel ≡ Two-Phase

**Theorem.** The fused kernel (generate child, test, keep/discard) produces the same survivor set as generating all children then testing each.

**Proof.** The pruning predicate $P(c)$ is a deterministic function of $c$ and fixed parameters (it does not depend on processing order or any mutable state from other children). The odometer visits each element of $\prod_i [\text{lo}_i, \text{hi}_i]$ exactly once in lexicographic order:

*Odometer completeness:* An odometer on ranges $[l_0,h_0] \times \cdots \times [l_{d-1},h_{d-1}]$ starts at $(l_0, l_1, \ldots, l_{d-1})$. Each step increments position $d-1$. When position $j$ exceeds $h_j$, reset to $l_j$ and carry (increment position $j-1$). Termination: when position 0 exceeds $h_0$.

This visits exactly $\prod_i (h_i - l_i + 1)$ distinct tuples. Each tuple is visited once (unique lexicographic rank). This is the standard odometer/mixed-radix counter argument.

Since $P$ is deterministic and the odometer visits every child exactly once:

$$\{c \in \prod_i [l_i, h_i] : \neg P(c)\}_\text{fused} = \{c \in \prod_i [l_i, h_i] : \neg P(c)\}_\text{two-phase}$$

```lean
-- Odometer visits every element exactly once (mixed-radix counter)
theorem odometer_bijection {d : ℕ} (lo hi : Fin d → ℕ) (h_valid : ∀ i, lo i ≤ hi i) :
    ∃ (f : Fin (∏ i, (hi i - lo i + 1)) → (∀ i : Fin d, Fin (hi i - lo i + 1))),
      Function.Bijective f := by
  sorry

-- Deterministic predicate + complete enumeration = same result
theorem fused_eq_twophase {α : Type*} [DecidableEq α] (S : Finset α) (P : α → Bool) :
    S.filter (fun x => !P x) = S.filter (fun x => !P x) := by
  rfl
```

The second theorem is literally `rfl` — the point is that the fused kernel applies the same predicate to the same set, just in a different computational pattern.

---

## Claim 4.3: Quick-Check Soundness

**Theorem.** If a child is killed by the quick-check (window $(\ell^*, s^*_\text{lo})$ from the previous kill), then it would also be killed by the full window scan.

**Proof.** The pruning condition is: $\exists (\ell, s_\text{lo})$ such that $\text{ws}(\ell, s_\text{lo}, c) > \text{dyn\_it}(\ell, W_\text{int}(\ell, s_\text{lo}, c))$.

The quick-check tests the specific window $(\ell^*, s^*_\text{lo})$. If it succeeds:

$$\text{ws}(\ell^*, s^*_\text{lo}, c) > \text{dyn\_it}(\ell^*, W^*_\text{int})$$

then the existential is witnessed by $(\ell^*, s^*_\text{lo})$. The full scan would also find this window (among others), so the child would be pruned.

If the quick-check fails, the full scan runs — no windows are skipped.

**The non-trivial part:** The quick-check's $W^*_\text{int}$ (contributing-bin mass for the previous killing window) must be correct for the CURRENT child. This is maintained by:

- **Fast path (1 parent position changed):** If the changed child bins $2p, 2p+1$ are in the contributing range $[\text{lo\_bin}, \text{hi\_bin}]$, update $W^*_\text{int}$ by the delta. O(1).
- **Short/deep carry:** Recompute $W^*_\text{int} = \sum_{i \in [\text{lo\_bin}, \text{hi\_bin}]} c'_i$ from scratch.

In all cases, $W^*_\text{int} = \sum_{i \in \mathcal{B}(\ell^*, s^*_\text{lo})} c'_i$ exactly.

```lean
-- Quick-check: if one window kills, the existential is satisfied
theorem quickcheck_sound {d : ℕ} (ws : ℕ → ℕ → ℤ) (dyn : ℕ → ℕ → ℤ)
    (ℓ_star s_star : ℕ) (h : ws ℓ_star s_star > dyn ℓ_star s_star) :
    ∃ ℓ s, ws ℓ s > dyn ℓ s :=
  ⟨ℓ_star, s_star, h⟩

-- W_int update for fast path is correct
-- (If changed positions are outside window range, W_int unchanged)
-- (If inside, W_int += delta)
theorem w_int_fast_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ) -- changed parent position
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i) -- only 2p, 2p+1 change
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i)
    (delta : ℤ) (hd : delta = (c' (2*p) - c (2*p)) + (c' (2*p+1) - c (2*p+1))) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if 2*p+1 ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  sorry
```
