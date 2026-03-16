# Lean 4 Proof Progress

**Lean version:** leanprover/lean4:v4.24.0
**Mathlib version:** f897ebcf72cd16f89ab4577d0c826cd14afaafc7

---

## Combined Proof: `lean/complete_proof.lean`

All completed proofs are merged into a single file: `lean/complete_proof.lean`. This is the canonical proof document. Individual source files are retained in `lean/` for reference.

### Source Files

| File | UUID | Content |
|------|------|---------|
| `output (1).lean` | ca2199a4 | Framework definitions + foundational lemmas (F1–F15) |
| `99433443...lean` | 99433443 | Reversal symmetry (Claims 3.3a, 3.3e) |
| `b66ccc2f...lean` | b66ccc2f | Refinement mass preservation (Claims 3.2c, 4.6) |
| `305874b1...lean` | 305874b1 | Incremental autoconvolution (Claim 4.2) |
| `output (3).lean` | e868a126 | Fused kernel + quick-check (Claims 4.1, 4.3) |
| `output (4).lean` | 31103b4c | Composition enumeration + children (Claims 3.1, 3.2a) |

---

## Completed: Formal Definitions

| Definition | Lean Name |
|-----------|-----------|
| Autoconvolution ratio $R(f) = \|f*f\|_\infty / (\int f)^2$ | `autoconvolution_ratio` |
| Autoconvolution constant $c = \inf R(f)$ | `autoconvolution_constant` |
| Discrete autoconvolution $\text{conv}[k] = \sum_{i+j=k} a_i a_j$ | `discrete_autoconvolution` |
| Test value $\text{TV}(\ell, s_{\text{lo}})$ | `test_value` |
| Max test value $b_{n,m}(c)$ | `max_test_value` |
| Composition (vector summing to $m$) | `is_composition` |
| Bin masses $\int_{\text{bin}_i} f$ | `bin_masses` |
| Canonical discretization (floor-rounding) | `canonical_discretization` |
| Contributing bins $\mathcal{B}(\ell, s_{\text{lo}})$ | `contributing_bins` |
| Cumulative distribution helper $D(k)$ | `canonical_cumulative_distribution` |
| Restriction of $f$ to bin $i$ | `f_restricted` |
| Vector reversal (ℕ) | `rev_vector` |
| Vector reversal (ℝ) | `rev_vector_real` |
| Integer autoconvolution (ℤ) | `int_autoconvolution` |
| Autoconvolution delta | `autoconv_delta` |
| Merge consecutive bin pairs | `merge_pairs` |
| Step function on 2n-bin grid | `step_function` |

## Completed: Foundational Lemmas (F1–F15)

| # | Theorem | Lean Name | Status |
|---|---------|-----------|--------|
| F1 | $c_i = D(i+1) - D(i)$ rewrite | `canonical_discretization_eq` | **PROVED** |
| F2 | $D(0) = 0$ | `canonical_cumulative_distribution_zero` | **PROVED** |
| F3 | $D(2n) = m$ (total mass $\neq 0$) | `canonical_cumulative_distribution_2n` | **PROVED** |
| F4 | Bin masses $\geq 0$ for $f \geq 0$ | `bin_masses_nonneg` | **PROVED** |
| F5 | $\sum c_i = m$ (zero mass edge case) | `canonical_discretization_sum_zero_mass` | **PROVED** |
| F6 | $c_i = D(i+1) - D(i)$ (alt hypothesis) | `canonical_discretization_eq_diff` | **PROVED** |
| F7 | Telescoping sum (AddCommGroup) | `sum_fin_telescope` | **PROVED** |
| F8 | $D$ is monotone for $f \geq 0$ | `canonical_cumulative_distribution_mono` | **PROVED** |
| F9 | $\sum c_i$ = telescope form | `canonical_discretization_sum_eq_telescope` | **PROVED** |
| F10 | $\int f_i = \text{bin\_mass}_i$ | `f_restricted_integral` | **PROVED** |
| F11 | $f \geq f_i \geq 0$ (restriction inequality) | `f_ge_f_restricted` | **PROVED** |
| F12 | Convolution commutativity | `convolution_comm_real` | **PROVED** |
| F13 | $\text{supp}(f) \subseteq (-1/4,1/4) \Rightarrow$ compact support | `f_has_compact_support` | **PROVED** |
| F14 | $f$ integrable $\Rightarrow f_i$ integrable | `f_restricted_integrable` | **PROVED** |
| F15 | $\sum c_i = m$ (positive mass, full proof) | `canonical_discretization_sum_eq_m` | **PROVED** |
| F16 | Nat telescoping sum | `sum_fin_telescope_nat` | **PROVED** |

## Completed: Claim-Level Theorems

| Claim | Description | Lean Name | Status |
|-------|-------------|-----------|--------|
| 3.1 | Composition count (stars-and-bars) | `composition_count` | **PROVED** |
| 3.2a | Per-bin choice count | `per_bin_choices` | **PROVED** |
| 3.2c | Children preserve total mass | `child_preserves_total_mass` | **PROVED** |
| 3.3a | Autoconv reversal symmetry | `autoconv_reversal_symmetry` | **PROVED** |
| 3.3e | Left-half mass reversal | `left_sum_reversal`, `asymmetry_reversal_symmetric` | **PROVED** |
| 4.1 | Odometer bijection (fused kernel) | `odometer_bijection` | **PROVED** |
| 4.2 | Incremental autoconv update | `incremental_update_correct` | **PROVED** |
| 4.3 | Quick-check soundness | `quickcheck_sound` | **PROVED** |
| 4.6 | Left-half sum invariant under refinement | `left_half_sum_invariant`, `left_half_sum_same_for_all_children` | **PROVED** |
| 4.7 | Ell scan order irrelevant | `exists_invariant_under_permutation` | **PROVED** |
| 4.8 | int32 overflow safety ($m \leq 200$) | `int32_safe` | **PROVED** |

### Supporting Lemmas

| Theorem | Lean Name | Status |
|---------|-----------|--------|
| Delta = sum of differences | `delta_eq_sum` | **PROVED** |
| Unchanged terms vanish | `unchanged_terms_zero` | **PROVED** |
| Three-way split by changed set S | `delta_three_way_split` | **PROVED** |
| Cross-term factorization | `cross_term_simplify` | **PROVED** |
| Membership groups exhaustive | `groups_exhaustive` | **PROVED** |
| Membership groups disjoint | `groups_disjoint` | **PROVED** |
| Child even+odd pair = parent | `child_bin_pair_sum` | **PROVED** |
| Fused = two-phase filtering | `fused_eq_twophase` | **PROVED** |
| W_int fast-path update | `w_int_fast_update` | **PROVED** |
| dyn_it monotone in W | `dyn_it_mono` | **PROVED** |
| Subtree pruning chain | `subtree_pruning_chain` | **PROVED** |
| Sliding window pruning equiv | `sliding_window_pruning_equiv` | **PROVED** |
| Zero term vanishes | `zero_term_vanishes` | **PROVED** |
| Sum filter zero | `sum_filter_zero` | **PROVED** |
| Cross-term zero-skip | `cross_term_zero_skip` | **PROVED** |

---

## Main Claims Status

| Claim | Description | Lean Status |
|-------|-------------|-------------|
| 1.1 | Test value = lower bound on $\|f*f\|_\infty$ | **STUB** in complete_proof.lean |
| 1.2 | Correction term $2/m + 1/m^2$ | **STUB** in complete_proof.lean |
| 1.3 | Dynamic threshold soundness | **STUB** in complete_proof.lean |
| 1.4 | Contributing bins formula | **STUB** in complete_proof.lean |
| 2.1 | Asymmetry: $\|f*f\|_\infty \geq 2L^2$ | **IN PROGRESS** — restriction lemmas proved (F10-F14) |
| 2.2 | Asymmetry margin unnecessary | NOT started |
| 2.3 | Single-bin x_cap | **IN PROGRESS** — restriction lemmas apply |
| 2.4 | Integer dynamic threshold | NOT started |
| 3.1 | Composition enumeration complete | **PROVED** |
| 3.2 | Child generation complete | **3.2a, 3.2c PROVED**, full enumeration NOT proved |
| 3.3 | Canonical symmetry sound | **PROVED** (3.3a + 3.3e) |
| 3.4 | Cascade induction | **STUB** in complete_proof.lean |
| 4.1 | Fused kernel equivalence | **PROVED** |
| 4.2 | Incremental autoconvolution | **PROVED** |
| 4.3 | Quick-check soundness | **PROVED** |
| 4.4 | Subtree pruning soundness | **STUB** (helpers proved: dyn_it_mono, subtree_pruning_chain) |
| 4.5 | CS x_cap no correction needed | **STUB** in complete_proof.lean |
| 4.6 | Hoisted asymmetry invariant | **PROVED** |
| 4.7 | Ell scan order irrelevant | **PROVED** |
| 4.8 | int32 overflow safety | **PROVED** (int32_safe); conv bounds **STUB** |
| 4.9 | Gray code bijection | **STUB** in complete_proof.lean |
| 4.10 | Cross-term split | **STUB** in complete_proof.lean |
| 4.11 | W_int Gray code update | **STUB** in complete_proof.lean |
| 4.12 | Sliding window equivalence | **STUB** (pruning equiv proved) |
| 4.13 | Zero-bin skipping | **PARTIALLY PROVED** (helpers done, autoconv_zero_skip stub) |
| 5.1 | FP rounding net conservative | NOT started |
| 5.2 | Integer autoconv exact | NOT started |

---

## Aristotle Prompt Queue

Prompts in `docs/aristotle/`. Attach `lean/complete_proof.lean` to each.

| # | File | Claims | Difficulty | Status |
|---|------|--------|-----------|--------|
| 1 | `prompt01_test_value_lower_bound` | 1.1 | Hard | **STUB** added |
| 2 | `prompt02_correction_term` | 1.2 | Very Hard | **STUB** added |
| 3 | `prompt03_dynamic_threshold_contributing_bins` | 1.3, 1.4 | Medium | **STUB** added |
| 4 | `prompt04_asymmetry_pruning` | 2.1 | Hard | **IN PROGRESS** |
| 5 | `prompt05_xcap_and_no_margin` | 2.2, 2.3 | Medium | **IN PROGRESS** |
| 6 | `prompt06_integer_threshold` | 2.4, 5.1, 5.2 | Medium | **LEAN FILE** added |
| 7 | `prompt07_enumeration_and_children` | 3.1, 3.2 | Medium | **PROVED** (3.1, 3.2a in complete_proof) |
| 9 | `prompt09_cascade_induction` | 3.4 | Medium | **STUB** added |
| 10 | `prompt10_fused_kernel_and_quickcheck` | 4.1, 4.3 | Easy | **PROVED** in complete_proof |
| 12 | `prompt12_subtree_pruning` | 4.4 | Medium-Hard | **STUB** + helpers proved |
| 13 | `prompt13_hoisted_asymmetry_ell_order_int32` | 4.5, 4.7, 4.8 | Easy-Medium | **PARTIALLY PROVED** |
| 14 | `prompt14_gray_code_kernel` | 4.9, 4.10, 4.11 | Medium | **STUB** added |
| 15 | `prompt15_sliding_window_and_zero_skip` | 4.12, 4.13 | Medium | **PARTIALLY PROVED** |

**Removed (fully proved):**
- ~~`prompt08_canonical_symmetry`~~ — Claim 3.3 proved
- ~~`prompt11_incremental_autoconv`~~ — Claim 4.2 proved

**Recommended order:** Prompts 4 and 5 (most Lean infrastructure). Then hard claims: 1, 2, 12. Then remaining stubs: 3, 6, 9, 14, 15.
