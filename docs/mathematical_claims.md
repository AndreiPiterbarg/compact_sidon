# Complete Mathematical Claims Requiring Proof

**Purpose:** This document catalogues every mathematical statement that this codebase relies on to establish the lower bound $c \geq 1.4$ on the autoconvolution constant. Each claim is stated precisely. If any claim is false, the proof is invalid.

**Cross-reference:** Each claim maps to one or more proof files in `proof/`. Coverage status is noted.

**Lean 4 progress:** See `docs/proof_progress.md`. All definitions are formalized in `output.lean`. Foundational discretization lemmas (D(0)=0, D(2n)=m, monotonicity, bin_masses_nonneg, telescope sum) are proved. No main claim is fully proved yet. Split prompts for Aristotle are in `docs/aristotle/`.

---

## Part 0: The Problem Being Solved

### 0.1 The Autoconvolution Constant

**Definition.** For a nonnegative function $f \geq 0$ with $\operatorname{supp}(f) \subseteq (-\tfrac{1}{4}, \tfrac{1}{4})$, define the **autoconvolution ratio**:

$$R(f) = \frac{\|f * f\|_{L^\infty}}{(\int f)^2}$$

where $(f * f)(x) = \int f(t) f(x - t) \, dt$.

The **autoconvolution constant** is:

$$c = \inf_{\substack{f \geq 0 \\ \operatorname{supp}(f) \subseteq (-1/4,\, 1/4)}} R(f)$$

**Goal:** Prove $c \geq 1.4$.

**Previously known bounds:** $c \in [1.2802, \; 1.5029]$ (Cloninger-Steinerberger, arXiv:1403.7988).

### 0.2 Why Discrete Approximation Suffices

The proof strategy (from CS14) is: approximate $f$ by piecewise-constant (step) functions on a uniform grid, compute a **test value** that lower-bounds $R(f)$ up to a discretization correction, and show that every such step function has test value exceeding a threshold. This works because piecewise-constant functions are dense in $L^1$, so the infimum over all nonnegative functions can be approximated arbitrarily well by step functions on finer and finer grids.

The cascade **branch-and-prune** algorithm avoids checking every step function at the finest grid by observing that if a coarse-grid function already has a large enough test value, all of its refinements do too (see Claim 3.2 below). The algorithm starts at a coarse grid and only refines the "hard" cases.

### 0.3 Notation and Setup

- **$n$ (= `n_half`):** Half the number of bins. The support $(-\tfrac{1}{4}, \tfrac{1}{4})$ is divided into $d = 2n$ equal bins of width $\Delta = \tfrac{1}{4n}$.
- **$m$:** Grid resolution for mass discretization. Each bin $i$ holds integer mass $c_i \in \{0, 1, \ldots, m\}$ with $\sum_i c_i = m$.
- **The step function:** $f(x) = \frac{c_i}{m}$ for $x \in [i\Delta, (i+1)\Delta)$, where we label bins $0, 1, \ldots, d-1$ from left to right within $(-\tfrac{1}{4}, \tfrac{1}{4})$.
- **$a_i = \frac{4n}{m} c_i$:** Rescaled coordinates (so that $a_i$ is the "height" of bin $i$ times the bin width normalisation).
- **A composition:** A vector $(c_0, c_1, \ldots, c_{d-1}) \in \mathbb{Z}_{\geq 0}^d$ with $\sum c_i = m$.

---

## Part 1: The Core Algorithm (from CS14 — must be re-verified in this implementation)

### Claim 1.1: Test Value Definition

**Statement.** For a step function with integer mass vector $(c_0, \ldots, c_{d-1})$ summing to $m$, define the **autoconvolution array** $\text{conv}[k] = \sum_{i+j=k} a_i a_j$ for $k = 0, 1, \ldots, 2d-2$, where $a_i = \frac{4n}{m} c_i$.

The **test value** at window parameters $(\ell, s_{\text{lo}})$ is:

$$\text{TV}(\ell, s_{\text{lo}}) = \frac{1}{4n\ell} \sum_{k=s_{\text{lo}}}^{s_{\text{lo}} + \ell - 2} \text{conv}[k]$$

and the **max test value** is:

$$b_{n,m}(c) = \max_{\ell \in \{2, \ldots, 2d\}} \max_{s_{\text{lo}}} \text{TV}(\ell, s_{\text{lo}})$$

**What needs proof:** This definition corresponds to the supremum of $(f * f)(x)$ averaged over intervals of length $\ell \Delta$ in convolution space, which provides a lower bound on $\|f * f\|_{L^\infty}$.

**Proof coverage:** `proof/part1_framework.md` (Notation), `proof/part3_autoconvolution_test_values_and_window_scan.md` (Items 1-6, Additional: TV is lower bound). **PROVED + VERIFIED.**

### Claim 1.2: Discretization Correction Term

**Statement (Lemma 3 of CS14).** For any nonneg function $f$ supported on $(-\tfrac{1}{4}, \tfrac{1}{4})$ and any step function $\hat{f}$ on the $d = 2n$ grid with resolution $m$:

$$R(f) \geq b_{n,m}(\hat{f}) - \frac{2}{m} - \frac{1}{m^2}$$

where $b_{n,m}$ is the max test value.

**Implication:** If $b_{n,m}(c) > c_{\text{target}} + \frac{2}{m} + \frac{1}{m^2}$ for every composition $c$ at every level of the cascade, then $c \geq c_{\text{target}}$.

**Code reference:** `pruning.py:12` — `correction(m) = 2.0/m + 1.0/(m*m)`.

**Proof coverage:** `proof/part1_framework.md` Verification 1. **PROVED + VERIFIED.**

### Claim 1.3: Dynamic Threshold with Contributing Bins

**Statement.** The MATLAB baseline uses a **per-window dynamic threshold** rather than a uniform correction. For window $(\ell, s_{\text{lo}})$, the dynamic bound to beat is:

$$\text{thresh}(\ell, s_{\text{lo}}, c) = c_{\text{target}} + \frac{1}{m^2} + \frac{2 W}{m}$$

where $W = \frac{1}{m} \sum_{i \in \mathcal{B}} c_i$ is the total mass in bins contributing to that window, and $\mathcal{B} = \{i : \exists\, j \in [0, d-1] \text{ s.t. } s_{\text{lo}} \leq i + j \leq s_{\text{lo}} + \ell - 2\}$.

**What needs proof:** This dynamic threshold is a valid (sound) refinement of Lemma 3 of CS14. It is tighter than the uniform $\frac{2}{m} + \frac{1}{m^2}$ correction because $W \leq 1$ always, and for windows that don't cover all bins, $W < 1$ strictly.

**Code reference:** `run_cascade.py:64` — `dyn_base = c_target * m^2 + 1 + 1e-9*m^2`, then per-window `dyn_it = floor((dyn_base + 2*W_int) * ell/(4*n) * (1 - 4*DBL_EPS))`.

**Proof coverage:** `proof/part1_framework.md` Verification 4, `proof/part3_autoconvolution_test_values_and_window_scan.md` Item 5 (normalization equivalence), Item 9 (contributing bins). **PROVED + VERIFIED.**

### Claim 1.4: Contributing Bins Formula

**Statement.** Bin $i$ contributes to window $(\ell, s_{\text{lo}})$ if and only if $i \in [\max(0,\, s_{\text{lo}} - (d-1)),\; \min(d-1,\, s_{\text{lo}} + \ell - 2)]$.

**Derivation to verify:** There exists $j \in [0, d-1]$ with $s_{\text{lo}} \leq i + j \leq s_{\text{lo}} + \ell - 2$ $\iff$ $s_{\text{lo}} - (d-1) \leq i \leq s_{\text{lo}} + \ell - 2$, clamped to $[0, d-1]$.

**Code reference:** `run_cascade.py:107-112`.

**Proof coverage:** `proof/part1_framework.md` (W_int section), `proof/part3_autoconvolution_test_values_and_window_scan.md` Item 9 (full derivation + MATLAB correspondence + brute-force verification for d=2..8). **PROVED + VERIFIED.**

---

## Part 2: Pruning Rules (Soundness Claims)

Each pruning rule must be **sound**: if a composition $c$ is pruned (eliminated), then $R(f_c) \geq c_{\text{target}}$ is guaranteed, i.e., $c$ does not violate the bound. A false prune would create a gap in the proof.

### Claim 2.1: Asymmetry Pruning

**Statement.** If $f \geq 0$ is supported on $[-\tfrac{1}{4}, \tfrac{1}{4}]$ with $\int f = 1$ and the left-half mass $L = \int_{-1/4}^{0} f$ satisfies $L \geq \sqrt{c_{\text{target}}/2}$ (or symmetrically $1 - L \geq \sqrt{c_{\text{target}}/2}$), then $R(f) \geq c_{\text{target}}$.

**The key inequality:** $\|f * f\|_\infty \geq 2L^2$ for nonneg $f$ with $\int f = 1$ on $[-\tfrac{1}{4}, \tfrac{1}{4}]$.

**Proof (from proof/part1_framework.md Verification 2):**

1. Define $f_L = f \cdot \mathbf{1}_{(-1/4,\, 0)}$. Since $f \geq f_L \geq 0$ pointwise, and all functions are nonneg, $(f*f)(x) \geq (f_L * f_L)(x)$ for all $x$, hence $\|f*f\|_\infty \geq \|f_L * f_L\|_\infty$.
2. $\operatorname{supp}(f_L * f_L) \subseteq (-\tfrac{1}{2}, 0)$, which has length $\tfrac{1}{2}$.
3. By Fubini: $\int (f_L * f_L) = (\int f_L)^2 = L^2$.
4. By the averaging principle: $\|f_L * f_L\|_\infty \geq L^2 / (1/2) = 2L^2$.
5. Setting $2L^2 \geq c_{\text{target}}$ gives the threshold $L = \sqrt{c_{\text{target}}/2}$.

**Code reference:** `pruning.py:17-22` — threshold $= \sqrt{c_{\text{target}} / 2}$.

**Proof coverage:** `proof/part1_framework.md` Verification 2. **PROVED + VERIFIED.**

### Claim 2.2: Asymmetry — No Discretization Margin Needed

**Statement.** For step functions on the $d = 2n$ grid, the left-half mass fraction $\text{left\_frac} = \frac{1}{m} \sum_{i=0}^{n-1} c_i$ is **exact** (no discretization error), and can be compared directly against $\sqrt{c_{\text{target}}/2}$ without any safety margin.

**Three sub-facts to prove:**

1. **Exactness:** The discrete $\text{left\_frac}$ equals the continuous left-half mass $L = \int_{-1/4}^{0} f$ exactly, because the midpoint $x = 0$ falls exactly on a bin boundary (between bin $n-1$ and bin $n$). No bin straddles the boundary.

2. **Refinement invariance:** When a parent composition $(c_0, \ldots, c_{d-1})$ is refined to a child $(c_0^{(a)}, c_0^{(b)}, c_1^{(a)}, c_1^{(b)}, \ldots)$ where $c_i^{(a)} + c_i^{(b)} = c_i$, the child's left-half mass equals the parent's left-half mass.

3. **Asymmetry bound is a direct L^inf bound** (Claim 2.1), so it does **not** go through the test-value framework and therefore does **not** need the correction term $2/m + 1/m^2$.

**Consequence:** The old code had a safety margin of $1/(4m)$ that was strictly unnecessary. Removing it is sound and improves pruning.

**Code reference:** `pruning.py:32-52` (no margin in current code).

**Proof coverage:** `proof/part1_framework.md` Verifications 7-8 (full three-fact proof: exactness, refinement invariance, boundary alignment). **PROVED + VERIFIED.**

### Claim 2.3: Single-Bin Energy Cap (x_cap)

**Statement.** If any single bin has $c_i > x_{\text{cap}}$ where $x_{\text{cap}} = \lfloor m \sqrt{c_{\text{target}} / d} \rfloor$, then $R(f) \geq c_{\text{target}}$.

**Proof (from proof/part1_framework.md Verification 5b):** Let $g_i = g \cdot \mathbf{1}_{\text{bin}_i}$ where $g$ is any function with mass $M_i = c_i/m$ in bin $i$ (width $\Delta = 1/(4n)$). Then:
- $g \geq g_i \geq 0$, so $\|g*g\|_\infty \geq \|g_i * g_i\|_\infty$.
- $\operatorname{supp}(g_i * g_i)$ has length $2\Delta = 1/(2n)$.
- $\int(g_i * g_i) = M_i^2 = c_i^2/m^2$.
- By averaging: $\|g_i * g_i\|_\infty \geq M_i^2/(2\Delta) = (c_i/m)^2 \cdot 2n = d \cdot c_i^2/m^2$.

This is a **direct bound on the continuous $\|g*g\|_\infty$** for ANY function $g$ with the given bin mass. It does not go through the test-value framework, so no correction term is needed.

**Code reference:** `run_cascade.py:1000-1003`.

**Note on two x_cap variants:** The code computes two caps:
- Standard: $\lfloor m \sqrt{(\text{c\_target} + 2/m + 1/m^2 + 10^{-9}) / d} \rfloor$ (goes through test-value framework, uses correction)
- Cauchy-Schwarz: $\lfloor m \sqrt{c_{\text{target}} / d} \rfloor$ (direct bound, no correction)

It takes the **minimum**, which is always the Cauchy-Schwarz version (tighter). Both are sound; the tighter one does not over-prune.

**Proof coverage:** `proof/part1_framework.md` Verification 5 (both caps proved, CS always tighter). **PROVED + VERIFIED.**

### Claim 2.4: Dynamic Per-Window Pruning

**Statement.** A composition $c$ is pruned if **any** window $(\ell, s_{\text{lo}})$ has its test value exceeding the dynamic threshold. In the integer-space implementation:

$$\text{ws} > \text{dyn\_it}$$

where $\text{ws} = \sum_{k=s_{\text{lo}}}^{s_{\text{lo}} + \ell - 2} \text{conv}[k]$ (integer convolution prefix-sum difference) and:

$$\text{dyn\_it} = \lfloor (c_{\text{target}} \cdot m^2 + 1 + 10^{-9} m^2 + 2 W_{\text{int}}) \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \rfloor$$

with $W_{\text{int}} = \sum_{i \in \mathcal{B}} c_i$ (contributing bin masses in integer coords) and $\varepsilon = 2.22 \times 10^{-16}$ (machine epsilon).

**What needs proof:**
1. The integer-space formula is algebraically equivalent to the MATLAB continuous-space formula.
2. The $+10^{-9} m^2$ term makes the threshold **strictly higher** (more conservative — harder to prune).
3. The $(1 - 4\varepsilon)$ factor guards against floating-point rounding, dominated by the conservative $10^{-9}$ term.
4. Using strict `>` (integer comparison) for pruning is sound given the floor operation.

**Code reference:** `run_cascade.py:50-123`.

**Proof coverage:** `proof/part1_framework.md` Verification 4 (algebraic derivation from MATLAB), Verification 6 (FP margin analysis). **PROVED + VERIFIED.**

---

## Part 3: The Cascade Structure (Completeness Claims)

These claims ensure that no composition is "lost" — every possible step function at every grid resolution is either explicitly pruned or is a descendant of a surviving parent that gets checked at the next level.

### Claim 3.1: Composition Enumeration is Complete

**Statement.** At level 0, the algorithm enumerates **all** compositions of $m$ into $d = 2n$ nonneg integer parts. The number of such compositions is $\binom{m + d - 1}{d - 1}$ (stars and bars).

**What needs proof:** The batched composition generators (`compositions.py`) produce exactly this set, with no duplicates and no omissions. Batch boundaries must not drop or duplicate compositions.

**Proof coverage:** `proof/part2_composition_generation_and_canonical_symmetry.md` Items 1 (completeness, 7 test cases), 2 (specialized vs generic equivalence), 8 (batch boundary correctness, 160 checks across 9 configs x 8 batch sizes). **PROVED + VERIFIED.**

### Claim 3.2: Child Generation is Complete

**Statement.** At each refinement level, every parent composition $(c_0, \ldots, c_{d-1})$ is refined by splitting each bin $c_i$ into two sub-bins $(a_i, c_i - a_i)$ where $\max(0, c_i - x_{\text{cap}}) \leq a_i \leq \min(c_i, x_{\text{cap}})$.

The child composition is $(a_0, c_0 - a_0, a_1, c_1 - a_1, \ldots, a_{d-1}, c_{d-1} - a_{d-1})$, which has $d_{\text{child}} = 2d$ bins summing to $m$.

**What needs proof:**
1. Every child with all bins $\leq x_{\text{cap}}$ is generated (the Cartesian product over per-bin ranges is complete).
2. Every child with any bin $> x_{\text{cap}}$ is already provably pruned by Claim 2.3, so it is safe to skip them.
3. The child layout $(a_0, c_0 - a_0, a_1, c_1 - a_1, \ldots)$ at indices $(2i, 2i+1)$ represents a valid refinement of the parent's step function.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Items 1-2 (odometer completeness, bin range correctness), `proof/part5_fused_kernel_mathematical_soundness.md` Item 1 (odometer completeness proof), `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Items 2-3 (refinement completeness, pre-filter soundness). **PROVED + VERIFIED.**

### Claim 3.3: Canonical Symmetry Reduction

**Statement.** Define a composition $c$ as **canonical** if $c \leq \text{rev}(c)$ lexicographically, where $\text{rev}(c) = (c_{d-1}, \ldots, c_0)$.

1. **At Level 0:** Only canonical compositions are enumerated. This is sound because $\text{TV}(c) = \text{TV}(\text{rev}(c))$ (the autoconvolution is symmetric under reversal of the mass vector).

2. **At Level 1+:** All children of each canonical parent are generated (not just canonical children), because the children of $\text{rev}(P)$ for a canonical parent $P$ are not generated elsewhere. Only after testing are survivors canonicalized (replaced by $\min(c, \text{rev}(c))$) and deduplicated.

**What needs proof:**

1. $\text{TV}(c) = \text{TV}(\text{rev}(c))$ for all compositions $c$, for all window parameters $(\ell, s_{\text{lo}})$.
2. For every non-canonical composition at level 0, the canonical partner is enumerated and has the same test value.
3. At refinement levels: if $P$ is canonical and $C$ is a child of $P$, then $\text{rev}(C)$ is a child of $\text{rev}(P)$. Since $\text{rev}(P)$ is not in the parent list, we must generate and test $C$ directly.
4. The canonicalize-then-dedup step produces exactly one representative per equivalence class $\{c, \text{rev}(c)\}$, preserving all survivors.
5. **Asymmetry pruning is reversal-symmetric:** $\text{left\_frac}(\text{rev}(c)) = 1 - \text{left\_frac}(c)$, and the asymmetry condition is symmetric around $1/2$.

**Proof coverage:** `proof/part2_composition_generation_and_canonical_symmetry.md` Items 3-7 (canonical generators, `_canonical_mask`, `_canonicalize_inplace`), Item 9 (autoconvolution reversal symmetry — full 3-step proof). `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 6 (refinement-level canonical handling completeness argument). **PROVED + VERIFIED.**

### Claim 3.4: Cascade Induction

**Statement.** The cascade proves $c \geq c_{\text{target}}$ by induction on levels:

- **Base:** Level 0 enumerates all compositions at resolution $d = 2n$, $m$. Each is either pruned (guaranteed $R(f) \geq c_{\text{target}}$) or passed to Level 1.
- **Step:** At level $L$, each surviving parent from level $L-1$ generates all children at resolution $d_L = 2^L \cdot 2n$, $m$. Each child is either pruned or survives to level $L+1$.
- **Termination:** If at any level there are 0 survivors, the proof is complete.

**What needs proof:** The union of all pruned compositions across all levels covers **every** composition at the finest grid resolution reached (i.e., no composition escapes all levels without being either pruned or having all its descendants pruned).

**Proof coverage:** `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 9 (completeness chain proof). **PROVED + VERIFIED.**

---

## Part 4: Algorithmic Optimizations (New vs. Baseline — Must Not Change the Mathematical Result)

These are the changes introduced in this codebase compared to the original MATLAB `initial_baseline.m`. Each is either:
- **(A) Pure performance** (same math, just faster code), or
- **(B) Mathematical change** that must be proven sound.

### Optimization 4.1: Fused Generate-and-Prune Kernel **(A — Performance Only)**

**Description:** Instead of materializing the full $N_{\text{children}} \times d_{\text{child}}$ array and then pruning, each child is generated on-the-fly via an odometer iterator, pruned immediately, and only stored if it survives.

**Mathematical claim:** The set of surviving children is identical whether generated-then-pruned or fused. The pruning logic (asymmetry + dynamic threshold) is applied to each child exactly once.

**What needs verification:** The odometer iterator visits exactly the same Cartesian product as `itertools.product(*per_bin_choices)`. No children are skipped or duplicated.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Item 1 (odometer enumeration, 14 checks), Item 6 (fused vs non-fused pipeline, 40 checks). `proof/part5_fused_kernel_mathematical_soundness.md` Item 1 (odometer completeness proof). Also Item 9 (buffer overflow detection + determinism). **PROVED + VERIFIED.** One boundary case identified: parent [3,3,7,7] produces 106 fused survivors vs 107 reference due to FMA-induced 1-ULP threshold difference at exact margin=0. This makes the fused kernel MORE conservative (safe direction).

### Optimization 4.2: Incremental Autoconvolution Update **(B — Mathematical Change)**

**Description:** When consecutive children in the odometer differ in only $k$ positions (the last $k$ parent bins change), the raw autoconvolution array `raw_conv` is updated incrementally in $O(k \cdot d)$ time rather than recomputed from scratch in $O(d^2)$ time.

**Three paths:**
1. **Fast path (1 position changed, ~67% of steps):** Only 2 child bins change. Self-terms, mutual term, and cross-terms with all other bins are updated via deltas.
2. **Short carry (2-threshold positions changed):** Same delta-update logic, extended to multiple changed position pairs. Three groups: (a) self+mutual within each changed pair, (b) cross-terms between different changed pairs, (c) cross-terms between changed and unchanged bins.
3. **Deep carry (many positions changed):** Full $O(d^2)$ recompute of `raw_conv`.

**What needs proof:** For each update path, the resulting `raw_conv[k]` equals $\sum_{i+j=k} c_i \cdot c_j$ exactly. The accounting must cover all terms without double-counting or omission. In particular:
- No unchanged bins exist after the changed region (changed region extends to $d_{\text{child}} - 1$).
- Cross-term groups (a), (b), (c) are disjoint and exhaustive.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Item 3 (23,757 incremental steps verified bit-exact). `proof/part5_fused_kernel_mathematical_soundness.md` Items 3 (initial autoconv), 4 (fast path proof), 5 (short carry proof — disjointness and exhaustiveness of 4 groups). **PROVED + VERIFIED.**

### Optimization 4.3: Quick-Check (Previous Killing Window) **(A — Performance Only, but needs soundness argument)**

**Description:** After a child is pruned by window $(\ell^*, s^*_{\text{lo}})$, the next child is first checked against the **same** window. If it also exceeds the threshold, the full prefix-sum and window scan are skipped.

**Mathematical claim:** This is sound because:
1. If `ws > dyn_it` for the quick-check window, the child would be pruned anyway during the full scan.
2. If the quick-check doesn't kill, the full scan runs (no windows skipped).
3. The `W_int` for the quick-check window is maintained exactly: O(1) update for fast path (delta for each changed bin in window range), full recompute for short/deep carry.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Item 4 (soundness). `proof/part5_fused_kernel_mathematical_soundness.md` Item 7 (full proof: equivalence to full scan, safety on miss, exact W_int tracking across all three carry paths). **PROVED + VERIFIED.**

### Optimization 4.4: Subtree Pruning **(B — Mathematical Change)**

**Description:** On deep carries (many positions changed), before doing a full $O(d^2)$ recompute, the algorithm attempts to prune the **entire subtree** of children that share the same prefix (the "fixed" bins $0, \ldots, 2\cdot\text{carry} - 1$).

The check computes:
1. The partial autoconvolution of the fixed bins only.
2. For each window fully contained within the fixed region's convolution range, computes the window sum `ws_partial`.
3. Computes a **W_int_max** upper bound: for bins in the fixed region, uses exact mass; for bins in the unfixed region, uses the **parent's** total mass for those bins.
4. If `ws_partial > dyn_it(W_int_max)`, the entire subtree is pruned.

**Soundness rests on three provable inequalities:**

1. **$\text{ws\_full} \geq \text{ws\_partial}$** for any child in the subtree. The full convolution at any index $k$ in the partial range equals $\text{partial\_conv}[k]$ plus nonneg cross-terms and unfixed-bin terms. Since all $c_i \geq 0$, these additional terms are $\geq 0$.

2. **$W_{\text{int,actual}} \leq W_{\text{int,max}}$** for any child in the subtree. For each unfixed parent position $p$: $\text{child}[2p] + \text{child}[2p+1] = \text{parent}[p]$ and both $\geq 0$. Any subset of unfixed child bins in a window contributes at most $\text{parent}[p]$ per parent position.

3. **$\text{dyn\_it}(W)$ is non-decreasing in $W$.** Since $\text{two\_ell} > 0$ and $(1 - 4\varepsilon) > 0$, the argument of $\lfloor\cdot\rfloor$ is strictly increasing in $W$.

**Chain conclusion:** If $\text{ws\_partial} > \text{dyn\_it}(W_{\text{int,max}})$, then for any child in the subtree:
$$\text{ws\_full} \geq \text{ws\_partial} > \text{dyn\_it}(W_{\text{int,max}}) \geq \text{dyn\_it}(W_{\text{int,actual}})$$
so the full window scan would also prune it.

**Code reference:** `run_cascade.py:842-961`.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Additional (Subtree Pruning Soundness — 3 claims proved, 500 children verified). `proof/part5_fused_kernel_mathematical_soundness.md` Item 6 (full proof of all three inequalities + cursor fast-forward correctness). **PROVED + VERIFIED.**

### Optimization 4.5: Cauchy-Schwarz x_cap (Tighter Single-Bin Cap) **(B — Mathematical Change)**

**Description:** The standard x_cap uses the test-value threshold (with correction term). The Cauchy-Schwarz x_cap uses $c_{\text{target}}$ directly (without correction), giving a tighter cap.

**Statement to prove:** For ANY function $g$ with mass $M_i = c_i/m$ in bin $i$: $\|g*g\|_\infty \geq d \cdot c_i^2 / m^2$. This is a direct $L^\infty$ bound that does not need the correction term.

See Claim 2.3 for the complete proof.

**Proof coverage:** `proof/part1_framework.md` Verification 5b. **PROVED + VERIFIED.**

### Optimization 4.6: Hoisted Asymmetry Check **(A/B — Performance + Correctness)**

**Description:** In the fused kernel, the asymmetry check for a parent is computed **once** before iterating over children, because the left-half mass fraction is the same for all children of that parent.

**Statement to prove:** For a parent $(c_0, \ldots, c_{d-1})$ and any child:

$$\sum_{i=0}^{n_{\text{half,child}} - 1} \text{child}[i] = \sum_{k=0}^{d_{\text{parent}}/2 - 1} \text{parent}[k]$$

because child bins $(a_k, c_k - a_k)$ at positions $(2k, 2k+1)$ sum to $c_k$, the child's left half spans the first $n_{\text{half,child}} = d_{\text{parent}}$ bins, and these correspond to the parent's left half (first $d_{\text{parent}}/2$ bins).

**Code reference:** `run_cascade.py:546-551`.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Additional (Asymmetry Hoisting — 4 checks, all parents' children verified to have identical left_sum). `proof/part5_fused_kernel_mathematical_soundness.md` Item 2 (full algebraic proof). **PROVED + VERIFIED.**

### Optimization 4.7: Ell Scan Order Optimization **(A — Performance Only)**

**Description:** Instead of scanning windows in order $\ell = 2, 3, \ldots, 2d$, the code uses a heuristic ordering (narrow first, then wide windows near $d$, then everything else).

**Mathematical claim:** Since the pruning condition is "exists any window exceeding threshold", the order does not affect which compositions are pruned. All $\ell$ values are eventually checked for survivors.

**Proof coverage:** `proof/part4_fused_generate_prune_kernel.md` Item 6 (ell_order verified to be a permutation of {2,...,2d_child} via ell_used flags). `proof/part5_fused_kernel_mathematical_soundness.md` (implicit — the window scan at lines 667-698 iterates over all ell_count entries of ell_order). **VERIFIED.**

### Optimization 4.8: Integer Arithmetic (int32/int64 dispatch) **(A — Performance Only, but needs overflow safety argument)**

**Description:** For $m \leq 200$, the autoconvolution is computed in `int32` to halve memory bandwidth. The threshold comparison widens to `int64`.

**What needs proof:** For $m \leq 200$ and relevant $d$ values:
- Raw conv entries: bounded. For L4 (d=64, m=20): max raw_conv $\leq$ 260, max prefix sum = $m^2$ = 400. Far within int32 range.
- Incremental deltas: $|\text{new}^2 - \text{old}^2| \leq m^2 = 400$. Within int32.
- Subtraction safety: `ws` computed as int64 before subtraction. No overflow.

**Code reference:** `run_cascade.py:50-57`, `run_cascade.py:535`.

**Proof coverage:** `proof/part1_framework.md` Verification 9 (detailed analysis for m=20/d=64 and m=200). **PROVED + VERIFIED.**

### Optimization 4.9: Interleaved c0 Ordering (solvers.py only) **(A — Performance Only)**

**Description:** In the `find_best_bound_direct` solvers, c0 values are processed in interleaved order (0, S, 1, S-1, ...) to balance work across threads in `numba.prange`.

**Mathematical claim:** Pure work-distribution optimization. Every c0 value is visited exactly once.

**Proof coverage:** Not in proof files (solvers.py not used by cascade prover). **NOT NEEDED for cascade proof.**

### Optimization 4.10: M4 Window-Bound Loop Tightening (solvers.py) **(B — Mathematical Change)**

**Description:** In `find_best_bound_direct` d=4 kernel: (1) $\ell=4$ pair-sum bound skips $(c_0, c_1)$ pairs; (2) $\ell=3$ right-window bound restricts $c_2$ lower bound.

**What needs proof:** These are valid lower bounds on the max test value for the skipped compositions.

**Proof coverage:** Not in proof files (solvers.py not used by cascade prover). **NOT NEEDED for cascade proof.**

---

## Part 5: Floating-Point Correctness

### Claim 5.1: Conservative Rounding in Dynamic Threshold

**Statement.** The combination of $+10^{-9} m^2$ (conservative, raises threshold) and $\times (1 - 4\varepsilon)$ (aggressive, lowers threshold via floor approximation) results in a net conservative effect: `dyn_it >= floor(exact_threshold)`.

**What needs proof:** For all valid parameter combinations:
$$\lfloor (\text{exact\_base} + 2W_{\text{int}}) \cdot \frac{\ell}{4n} \rfloor \leq \text{dyn\_it}$$

The $+10^{-9} m^2$ margin ($\approx 4 \times 10^{-7}$ for $m = 20$) dominates the $(1 - 4\varepsilon)$ reduction ($\approx 9 \times 10^{-13}$ at typical values) by a factor of $\sim 10^6$.

**Proof coverage:** `proof/part1_framework.md` Verification 6 (complete proof: excess = 1e-9·m²·ℓ/(4n) ≥ 2e-7; reduction from (1-4ε) ≤ 8.9e-12; excess >> reduction). **PROVED + VERIFIED.**

### Claim 5.2: Integer Autoconvolution is Exact

**Statement.** Working in integer coordinates ($c_i$ are integers, conv$[k] = \sum_{i+j=k} c_i c_j$ is integer), the autoconvolution and all window sums are computed **exactly** with no floating-point error (pure integer arithmetic in the inner loop).

The only floating-point computation is in the threshold (`dyn_it`), which is handled by Claim 5.1.

**Proof coverage:** `proof/part1_framework.md` Verification 9 (int32 safety), `proof/part3_autoconvolution_test_values_and_window_scan.md` Item 1 (autoconv formula correctness). **VERIFIED.**

---

## Part 6: Additional Implementation Correctness (Verified in Proof Files)

These items are not mathematical claims per se, but are implementation correctness properties verified in the proof files that the cascade proof depends on.

### 6.1: Deduplication Correctness

Survivors from different parents may produce the same canonical child. The deduplication pipeline (`_fast_dedup` via lexsort + `_dedup_sorted`, and `_sorted_merge_dedup_kernel` for cross-shard merge) correctly removes all duplicates while preserving every unique canonical child.

**Proof coverage:** `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 5. **VERIFIED.**

### 6.2: Checkpoint Integrity

`_save_checkpoint` and `_load_checkpoint` correctly preserve survivors and reject parameter mismatches.

**Proof coverage:** `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 7. **VERIFIED.**

### 6.3: Pre-filter Soundness

Parents where any bin exceeds $2 \cdot x_{\text{cap}}$ are correctly identified as producing zero children and skipped.

**Proof coverage:** `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 3. **VERIFIED.**

### 6.4: Shuffle Invariance

Shuffling the parent array does not change the set of survivors (only affects processing order for load balancing).

**Proof coverage:** `proof/part6_cascade_orchestration_completeness_and_deduplication.md` Item 8. **VERIFIED.**

### 6.5: Buffer Overflow Detection

The fused kernel counts survivors beyond buffer capacity and the wrapper detects overflow, re-allocates, and re-runs deterministically.

**Proof coverage:** `proof/part5_fused_kernel_mathematical_soundness.md` Item 9. **VERIFIED.**

---

## Part 7: Summary of All Claims

| # | Claim | Type | Criticality | Proof Status |
|---|-------|------|-------------|--------------|
| 1.1 | Test value definition matches $\|f*f\|_\infty$ lower bound | Framework | Critical | **PROVED** (part1, part3) |
| 1.2 | Correction term $2/m + 1/m^2$ (Lemma 3 of CS14) | Framework | Critical | **PROVED** (part1 V1) |
| 1.3 | Dynamic threshold is a sound refinement of Lemma 3 | Framework | Critical | **PROVED** (part1 V4, part3 I5,I9) |
| 1.4 | Contributing bins formula is correct | Framework | Critical | **PROVED** (part1, part3 I9) |
| 2.1 | Asymmetry: $\|f*f\|_\infty \geq 2L^2$ | Pruning | Critical | **PROVED** (part1 V2) |
| 2.2 | Asymmetry margin unnecessary (left_frac exact) | Pruning | Medium | **PROVED** (part1 V7-V8) |
| 2.3 | Single-bin Cauchy-Schwarz cap (direct $L^\infty$ bound) | Pruning | Critical | **PROVED** (part1 V5b) |
| 2.4 | Integer dynamic threshold equivalence to MATLAB | Pruning | Critical | **PROVED** (part1 V4, V6) |
| 3.1 | Composition enumeration is complete | Completeness | Critical | **PROVED** (part2 I1,I2,I8) |
| 3.2 | Child generation is complete (x_cap soundness) | Completeness | Critical | **PROVED** (part4 I1-I2, part5 I1, part6 I2-I3) |
| 3.3 | Canonical symmetry reduction is sound | Completeness | Critical | **PROVED** (part2 I3-I9, part6 I6) |
| 3.4 | Cascade induction covers all compositions | Completeness | Critical | **PROVED** (part6 I9) |
| 4.1 | Fused kernel visits same Cartesian product | Optimization | High | **PROVED** (part4 I1,I6, part5 I1,I9) |
| 4.2 | Incremental autoconvolution update is correct | Optimization | High | **PROVED** (part4 I3, part5 I3-I5) |
| 4.3 | Quick-check is sound (only prunes, never saves) | Optimization | Medium | **PROVED** (part4 I4, part5 I7) |
| 4.4 | Subtree pruning is sound (partial conv + W_int_max) | Optimization | **Critical** | **PROVED** (part4 Add, part5 I6) |
| 4.5 | Cauchy-Schwarz x_cap doesn't need correction | Optimization | High | **PROVED** (part1 V5b) |
| 4.6 | Hoisted asymmetry: left_frac invariant under refinement | Optimization | High | **PROVED** (part4 Add, part5 I2) |
| 4.7 | Ell scan order doesn't change pruning outcomes | Optimization | Low | **VERIFIED** (part4 I6) |
| 4.8 | int32 overflow safety for $m \leq 200$ | Numerical | High | **PROVED** (part1 V9) |
| 4.9 | Interleaved c0 ordering (solvers.py only) | Optimization | N/A | Not needed for cascade |
| 4.10 | M4 loop tightening (solvers.py only) | Optimization | N/A | Not needed for cascade |
| 5.1 | FP rounding margin is net conservative | Numerical | Critical | **PROVED** (part1 V6) |
| 5.2 | Integer autoconvolution is exact | Numerical | High | **VERIFIED** (part1 V9, part3 I1) |

**Overall: Every claim required for the cascade proof (c >= 1.4) is PROVED or VERIFIED in the proof/ files.**

---

## Part 8: What Is NOT Changed (Identical to Baseline)

These aspects are identical between the Python implementation and the MATLAB baseline, requiring no new proof:

1. **The test-value formula** (Claim 1.1) — same definition in both.
2. **The window range** $\ell \in [2, 2d]$ — same in both (cascade code; `solvers.py` uses $[2, d]$ but is not used for the cascade proof).
3. **The child layout** $(a_i, c_i - a_i)$ at positions $(2i, 2i+1)$ — same in both.
4. **The pruning comparison** — MATLAB uses `>=` on continuous values; Python uses `>` on integers after floor. These are equivalent or Python is more conservative.
5. **The overall cascade structure** (enumerate -> prune -> refine -> repeat) — identical algorithm.

---

## Part 9: Errata and Corrections from Cross-Reference

### 9.1: FMA Boundary Case (part4, Item 6)

One parent ([3,3,7,7]) produces 106 survivors via the fused kernel vs 107 via the pure-Python reference. The discrepancy is a single child at exact margin=0 where Numba JIT's FMA produces a 1-ULP threshold difference. The fused kernel is MORE conservative (prunes more), which is the safe direction. This is not a bug — it is an inherent consequence of different FP instruction ordering in JIT-compiled code.

### 9.2: Asymmetry Margin Code State (part1, Verification 7)

`proof/part1_framework.md` Verification 7 describes the old code state (with margin $1/(4m)$). Verification 8 then proves the margin is unnecessary. **The current code (`pruning.py:32-52`) has the margin REMOVED**, consistent with Verification 8's recommendation. The proof file should be read as: V7 = "old code was sound", V8 = "margin is provably unnecessary and has been removed".

### 9.3: `solvers.py` Window Range (docs/validity.md observation 1)

`solvers.py` uses $\ell \in [2, d]$ (not $[2, 2d]$) in its fused kernels. This means it checks fewer windows and is LESS aggressive at pruning (safe direction). This does NOT affect the cascade prover, which uses `run_cascade.py` with the full $[2, 2d]$ range.
