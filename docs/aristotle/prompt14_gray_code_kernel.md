# Prompt 14: Gray Code Kernel Soundness

**Claims 4.9 + 4.10 + 4.11.** Attach `output.lean` as context.

---

## Problem Context

The cascade kernel was changed from a lexicographic odometer to a mixed-radix Gray code (Knuth TAOCP 7.2.1.1). Three properties must hold for the proof to remain valid.

### Definitions

- **Cartesian product:** $\mathcal{C} = \prod_{i=0}^{d-1} [\text{lo}_i, \text{hi}_i]$, the set of all children of a parent.
- **Pruning predicate $P(c)$:** Deterministic function of child $c$ and fixed parameters. True iff the child is covered by the bound.
- **Survivor set:** $\{c \in \mathcal{C} : \neg P(c)\}$.
- **Mixed-radix Gray code:** A sequence visiting every element of $\mathcal{C}$ exactly once, changing exactly one coordinate by $\pm 1$ per step.

---

## Claim 4.9: Gray Code Enumeration Completeness

**Theorem.** The mixed-radix Gray code on radices $(r_0, \ldots, r_{k-1})$ where $r_j = \text{hi}_{a_j} - \text{lo}_{a_j} + 1$ (active positions with $r_j > 1$) visits exactly $\prod_j r_j$ distinct elements of $\mathcal{C}$, once each. Positions with $r_j = 1$ are fixed throughout.

**Proof sketch.** By induction on $k$ (number of active digits).

*Base case ($k = 0$):* One element, visited once.

*Inductive step:* A mixed-radix Gray code on $(r_0, \ldots, r_{k-1})$ consists of $r_{k-1}$ copies of the Gray code on $(r_0, \ldots, r_{k-2})$, with digit $k-1$ cycling through its $r_{k-1}$ values (alternating direction). Each sub-code visits $\prod_{j<k-1} r_j$ elements, and the $r_{k-1}$ copies cover disjoint digit-$(k-1)$ values. Total: $\prod_{j=0}^{k-1} r_j$.

*Uniqueness:* At each step, exactly one coordinate changes. The sequence of digit values for each coordinate is a path that visits every value in $[0, r_j - 1]$ exactly $\prod_{i \neq j} r_i$ times across the full code, and each full state is visited exactly once (no two steps produce the same tuple because the focus pointer mechanism ensures systematic traversal).

**This replaces the odometer completeness argument in Claim 4.1.** Since $P$ is deterministic and the Gray code visits every child exactly once:

$$\{c \in \mathcal{C} : \neg P(c)\}_\text{gray} = \{c \in \mathcal{C} : \neg P(c)\}_\text{odometer}$$

```lean
theorem gray_code_bijection {k : ℕ} (r : Fin k → ℕ) (hr : ∀ i, 0 < r i) :
    ∃ (f : Fin (∏ i, r i) → (∀ i : Fin k, Fin (r i))),
      Function.Bijective f := by
  sorry
```

---

## Claim 4.10: Incremental Update for Arbitrary Position

**Theorem.** The fast-path incremental autoconvolution update (Claim 4.2, Path 1) is valid when position $p$ is *any* parent position, not just the last one. The cross-term loops must cover bins both before and after $(2p, 2p+1)$.

**Proof.** Claim 4.2 proves the delta decomposition for changed bins $\{2p, 2p+1\}$:

$$\Delta\text{conv}[t] = \Delta\text{self}_{2p} + \Delta\text{self}_{2p+1} + \Delta\text{mutual} + \Delta\text{cross}$$

The decomposition depends only on *which* bins changed, not on their position. The cross-term Group D sums over $q \notin \{2p, 2p+1\}$. In the odometer's fast path, $p = d_\text{parent} - 1$ so all $q < 2p$, and the loop `for q in range(2p)` is sufficient. In the Gray code, $p$ can be any active position, so $q$ ranges over $\{0, \ldots, 2p-1\} \cup \{2p+2, \ldots, d-1\}$.

The code implements this as two loops:
```
for q in range(2p):       # bins before the changed pair
for q in range(2p+2, d):  # bins after the changed pair
```

Both loops iterate over $q \notin \{2p, 2p+1\}$, which is exactly Group D. Total iterations: $d - 2$ regardless of $p$.

```lean
theorem cross_term_split {d : ℕ} (p : ℕ) (hp : 2*p+1 < d)
    (f : Fin d → ℤ) :
    (∑ q : Fin d, if q.1 ≠ 2*p ∧ q.1 ≠ 2*p+1 then f q else 0) =
    (∑ q ∈ Finset.range (2*p), f ⟨q, by omega⟩) +
    (∑ q ∈ Finset.Ico (2*p+2) d, f ⟨q, by omega⟩) := by
  sorry
```

---

## Claim 4.11: Quick-Check $W_\text{int}$ Correctness Under Gray Code

**Theorem.** The quick-check's cached $W^*_\text{int}$ equals $\sum_{i \in \mathcal{B}(\ell^*, s^*)} c_i$ for the current child at every step of the Gray code.

**Proof.** $W^*_\text{int}$ is maintained by two mechanisms:

1. **Initialization:** When the full window scan finds a killing window $(\ell^*, s^*)$, it sets $W^*_\text{int} = W_\text{int}$ where $W_\text{int} = \text{prefix\_c}[\text{hi\_bin}+1] - \text{prefix\_c}[\text{lo\_bin}]$ was just computed from the current child's bin masses. This is exact.

2. **Incremental update:** When the Gray code changes position $p$ (bins $2p, 2p+1$ change by $\delta_1, \delta_2$):

$$W^{*,\text{new}}_\text{int} = W^{*,\text{old}}_\text{int} + \delta_1 \cdot \mathbb{1}[2p \in \mathcal{B}] + \delta_2 \cdot \mathbb{1}[2p+1 \in \mathcal{B}]$$

where $\mathcal{B} = [\text{lo\_bin}, \text{hi\_bin}]$ is the contributing-bin range for window $(\ell^*, s^*)$.

Since only bins $2p$ and $2p+1$ changed, and for all other bins $c'_i = c_i$:

$$\sum_{i \in \mathcal{B}} c'_i = \sum_{i \in \mathcal{B}} c_i + \delta_1 \cdot \mathbb{1}[2p \in \mathcal{B}] + \delta_2 \cdot \mathbb{1}[2p+1 \in \mathcal{B}] = W^{*,\text{new}}_\text{int}$$

This is the same argument as Claim 4.3 (prompt 10), but generalized: it applies to *any* changed position $p$, not just the last one.

**Critical invariant:** $W^*_\text{int}$ is set from scratch (mechanism 1) whenever the killing window changes. Between window changes, it is updated incrementally (mechanism 2). Since the Gray code changes exactly one position per step, mechanism 2 is the only update path (no short/deep carry recomputation needed).

```lean
theorem w_int_gray_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if (2*p+1) ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  sorry
```

---

## Chain: Gray Code Kernel Produces Valid Survivors

The full soundness argument chains:

1. **Completeness** (4.9): Gray code visits every child exactly once.
2. **Incremental correctness** (4.10 + 4.2): `raw_conv` is exact after every step.
3. **Quick-check correctness** (4.11 + 4.3): Quick-check only kills children that would be killed by the full scan.
4. **Pruning soundness** (Claims 1-3 from earlier prompts): Full scan correctly identifies children covered by the bound.

Therefore: the Gray code kernel's survivor set equals the odometer kernel's survivor set, and every pruned child is genuinely covered by the bound.
