/-
Sidon Autocorrelation Project — Threshold Lookup Table (Claims 5.3–5.12)

This file collects the theorems and lemmas that must be proved to
certify the 2D precomputed threshold table optimization implemented in
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization replaces per-window float64 threshold computation:
    dyn_x = dyn_base_ell + 1.0 + eps_margin + 2.0 * float64(W_int)
    dyn_it = int64(dyn_x * one_minus_4eps)
with a single lookup into a precomputed table:
    dyn_it = threshold_table[ell_idx * (m + 1) + W_int]
where threshold_table is built once before the main loop.

The table is indexed by (ell_idx, W_int) where:
  - ell_idx = ell - 2, ranging over [0, 2*d_child - 2]
  - W_int ∈ [0, m], the sum of child masses in a window's bin range

Claims covered:
  5.3       Table construction formula matches runtime formula (rfl)
  5.4       W_int range: partial sum of child masses ∈ [0, m]
  5.6       W_int_max range in subtree pruning ∈ [0, m]
  5.7       Table index is in bounds
  5.11      Pruning soundness is preserved under table lookup
  5.12      Pruning predicate is identical under table lookup (rfl)

Claims removed after audit:
  5.5       Redundant — direct corollary of 5.4 (apply 5.4 to the
            quick-check window range, which is always a partial sum of
            child masses maintained incrementally through Gray code steps)
  5.8–5.10  Redundant — identical universally-quantified statement as 5.3,
            not specialized to any particular call site
  w_int_nonneg  Trivially true (sum of ℕ values is ℕ, hence ≥ 0)

Bridging lemma added:
  runtime_threshold_eq_dyn_it — connects this file's formula to
  DynamicThreshold.lean's dyn_it (syntactic difference: a*(1/b) vs a/b)

NOTE ON IEEE 754 DETERMINISM: The LUT optimization's correctness in the
Python implementation depends on IEEE 754 float64 arithmetic being
deterministic for identical inputs (same operations, same order, same
values → same result). The Lean formalization works over mathematical
reals (ℝ), where this is trivially true. The float64 property is a
standard guarantee of IEEE 754 compliant hardware and is verified
empirically by the test suite (test_gray_code.py: 19 tests comparing
LUT kernel output vs non-LUT kernel output). This is an assumption
external to the formalization.

NOTE ON ELL RANGE: The table is built for ell ∈ [2, 2*d_child], i.e.,
ell_idx ∈ [0, 2*d_child - 2]. All three call sites (main window scan,
quick-check, subtree pruning) iterate ell over this same range. The
quick-check stores a previously-valid (ell, s_lo) pair, so qc_ell is
always in [2, 2*d_child]. This is derivable from the loop structure
of run_cascade.py and does not require a separate theorem.

STATUS: PROOF OBLIGATIONS — proofs marked `sorry` are open obligations.
Claims 5.3 and 5.12 are proved (definitional equality / rewriting).
-/

import Mathlib
import Sidon.Defs
import Sidon.DynamicThreshold

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Table Construction Formula (Claim 5.3)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The runtime threshold formula: given the precomputed per-ℓ constant
    dyn_base_ℓ = c_target · m² · ℓ/(4n), the runtime computes:
      dyn_x = dyn_base_ℓ + 1 + eps_margin + 2 · W_int
      dyn_it = ⌊dyn_x · (1 - 4ε)⌋
    where eps_margin = 10⁻⁹ · m² and ε = 2.220446049250313 × 10⁻¹⁶.

    IMPORTANT: ℓ/(4n) scales ONLY c_target · m², NOT the correction terms
    (1 + eps_margin + 2·W_int). This matches the Python code and is MORE
    conservative than scaling everything by ℓ/(4n), since ℓ/(4n) ≤ 1. -/
noncomputable def runtime_threshold (c_target : ℝ) (m n_half_child ℓ : ℕ) (W_int : ℕ) : ℤ :=
  let m_r := (m : ℝ)
  let inv_4n := 1 / (4 * (n_half_child : ℝ))
  let dyn_base_ell := c_target * m_r ^ 2 * (ℓ : ℝ) * inv_4n
  let eps_margin := 1e-9 * m_r ^ 2
  let one_minus_4eps := 1 - 4 * (2.220446049250313e-16 : ℝ)
  let dyn_x := dyn_base_ell + 1 + eps_margin + 2 * (W_int : ℝ)
  ⌊dyn_x * one_minus_4eps⌋

/-- The table construction formula: identical arithmetic to runtime_threshold.

    Code reference: run_cascade.py lines 1106-1114
      for ell in range(2, 2 * d_child + 1):
          idx = ell - 2
          dyn_base_ell_val = c_target * m_d * m_d * float(ell) * inv_4n
          for w in range(m + 1):
              dyn_x = dyn_base_ell_val + 1.0 + eps_margin + 2.0 * float(w)
              threshold_table[idx * m_plus_1 + w] = int64(dyn_x * one_minus_4eps) -/
noncomputable def table_entry (c_target : ℝ) (m n_half_child ℓ : ℕ) (w : ℕ) : ℤ :=
  let m_r := (m : ℝ)
  let inv_4n := 1 / (4 * (n_half_child : ℝ))
  let dyn_base_ell_val := c_target * m_r ^ 2 * (ℓ : ℝ) * inv_4n
  let eps_margin := 1e-9 * m_r ^ 2
  let one_minus_4eps := 1 - 4 * (2.220446049250313e-16 : ℝ)
  let dyn_x := dyn_base_ell_val + 1 + eps_margin + 2 * (w : ℝ)
  ⌊dyn_x * one_minus_4eps⌋

/-- Claim 5.3: The table entry and runtime threshold are definitionally equal.
    Both compute ⌊(c_target·m²·ℓ/(4n) + 1 + 10⁻⁹·m² + 2·W_int)·(1-4ε)⌋
    with identical Lean terms (differing only in let-binding names, which are
    erased to de Bruijn indices). -/
theorem table_entry_eq_runtime
    (c_target : ℝ) (m n_half_child ℓ : ℕ) (w : ℕ) :
    table_entry c_target m n_half_child ℓ w = runtime_threshold c_target m n_half_child ℓ w :=
  rfl

-- ═══════════════════════════════════════════════════════════════════════════════
-- Bridging Lemma: runtime_threshold = dyn_it
--
-- runtime_threshold uses a * (1/b) while dyn_it uses a / b. Over ℝ these
-- are equal (mul_one_div), but not definitionally so in Lean.
-- This lemma connects the file's formulas to DynamicThreshold.pruning_soundness.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- runtime_threshold and dyn_it (from DynamicThreshold.lean) compute the same
    value. The only syntactic difference is a*(1/b) vs a/b in the dyn_base_ell
    sub-expression:
      runtime_threshold: c_target * m² * ℓ * (1 / (4*n))
      dyn_it:            c_target * m² * ℓ / (4*n)
    These are equal over ℝ by mul_one_div : a * (1/b) = a/b.

    This lemma closes the soundness chain:
      table_entry = runtime_threshold  (Claim 5.3, rfl)
      runtime_threshold = dyn_it       (this lemma)
      dyn_it is conservative           (DynamicThreshold.dyn_it_conservative)
      pruning is sound                 (DynamicThreshold.pruning_soundness) -/
theorem runtime_threshold_eq_dyn_it
    (c_target : ℝ) (m n ℓ W_int : ℕ) :
    runtime_threshold c_target m n ℓ W_int = dyn_it c_target m n ℓ W_int := by
  simp only [runtime_threshold, dyn_it, mul_one_div]

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: W_int Range Bounds (Claims 5.4, 5.6)
--
-- The table is indexed by W_int ∈ [0, m]. We must prove that W_int
-- is always in this range at each call site. Sums use Fin d with
-- Finset.univ.filter to avoid elaboration issues with ℕ-indexed sums
-- requiring i < d proofs (see complete_proof.lean lines 2559-2570 for
-- the same pattern in this codebase).
--
-- The quick-check W_int (Claim 5.5) is a direct corollary of Claim 5.4:
-- qc_W_int is always a partial sum of child masses in a window's bin
-- range (initialized from a window scan, then updated incrementally by
-- delta1/delta2 with delta1+delta2=0 preserving the partial sum
-- invariant). Apply Claim 5.4 to the quick-check window range.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 5.4: In the main window scan, W_int = sum of child masses in
    bins [lo_bin, hi_bin] is in [0, m].

    The child masses are non-negative integers (ℕ) and their total sum
    is m. Any contiguous sub-sum is therefore in [0, m].

    Code reference: run_cascade.py line 1232
      W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin] -/
theorem w_int_main_scan_in_range
    {d : ℕ} (child : Fin d → ℕ) (m : ℕ)
    (h_sum : ∑ i, child i = m)
    (lo_bin hi_bin : ℕ) (hlo : lo_bin ≤ hi_bin) (hhi : hi_bin < d) :
    (∑ i ∈ Finset.univ.filter (fun (i : Fin d) =>
      lo_bin ≤ i.val ∧ i.val ≤ hi_bin), child i) ≤ m := by
  sorry  -- Finset.sum_le_univ_sum_of_nonneg + h_sum: filtered sum ≤ univ sum = m

/-- Claim 5.6: In subtree pruning, W_int_max = W_int_fixed + W_int_unfixed ≤ m.

    W_int_fixed ≤ child masses in [0, 2·p_boundary), which equals parent
    masses in [0, p_boundary) by h_split: child[2q]+child[2q+1] = parent[q].
    W_int_unfixed ≤ parent masses in [p_boundary, d_parent).
    These cover disjoint parent ranges whose union ⊆ [0, d_parent),
    so their sum ≤ total parent mass = m.

    Code reference: run_cascade.py lines 1397-1438 -/
theorem w_int_max_subtree_in_range
    {d_parent : ℕ} (parent : Fin d_parent → ℕ)
    (child : Fin (2 * d_parent) → ℕ) (m : ℕ)
    (h_sum_parent : ∑ i, parent i = m)
    (h_split : ∀ p : Fin d_parent,
      child ⟨2 * p.1, by omega⟩ + child ⟨2 * p.1 + 1, by omega⟩ = parent p)
    (W_int_fixed W_int_unfixed : ℕ)
    (p_boundary : ℕ) (hp : p_boundary ≤ d_parent)
    -- W_int_fixed ≤ sum of child masses in the fixed prefix [0, 2·p_boundary)
    (hWf_bound : W_int_fixed ≤
      ∑ i ∈ Finset.univ.filter (fun (i : Fin (2 * d_parent)) =>
        i.val < 2 * p_boundary), child i)
    -- W_int_unfixed ≤ sum of parent masses for unfixed bins [p_boundary, d_parent)
    (hWu_bound : W_int_unfixed ≤
      ∑ i ∈ Finset.univ.filter (fun (i : Fin d_parent) =>
        p_boundary ≤ i.val), parent i) :
    W_int_fixed + W_int_unfixed ≤ m := by
  sorry  -- Key step: reindex child sums in [0,2p) as parent sums in [0,p) via h_split.
         -- Then: parent[0,p) + parent[p,d) = ∑ parent = m.
         -- Both hWf_bound and hWu_bound are ≤ these respective parent sums.

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Table Index Bounds (Claim 5.7)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 5.7: The flat index ell_idx * (m+1) + W_int < ell_count * (m+1).

    From ell_idx < ell_count and W_int ≤ m:
      ell_idx*(m+1) + W_int < (ell_idx+1)*(m+1) ≤ ell_count*(m+1).

    The first inequality: W_int ≤ m < m+1, so ell_idx*(m+1) + W_int <
    ell_idx*(m+1) + (m+1) = (ell_idx+1)*(m+1).
    The second: ell_idx < ell_count → (ell_idx+1) ≤ ell_count. -/
theorem table_index_in_bounds
    (ell_count m : ℕ) (ell_idx W_int : ℕ)
    (h_ell : ell_idx < ell_count)
    (h_w : W_int ≤ m) :
    ell_idx * (m + 1) + W_int < ell_count * (m + 1) := by
  sorry  -- nlinarith [Nat.lt_succ_of_le h_w, Nat.succ_le_of_lt h_ell,
         --            Nat.mul_le_mul_right (m + 1) (Nat.succ_le_of_lt h_ell)]

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: Pruning Soundness (Claims 5.11, 5.12)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 5.11: Pruning soundness is preserved under table lookup.

    Chain:
    1. h_exceeds_table: ws > table_entry(ℓ, W_int)
    2. table_entry = runtime_threshold         (Claim 5.3, rfl)
    3. runtime_threshold = dyn_it              (runtime_threshold_eq_dyn_it)
    4. → ws > dyn_it(ℓ, W_int)                (from 1-3)
    5. dyn_it ≥ ⌊exact_threshold⌋             (DynamicThreshold.dyn_it_conservative)
    6. → ws > ⌊exact_threshold⌋               (from 4-5)

    The exact threshold is A = c_target·m²·ℓ/(4n) + 1 + 2·W_int, from the
    mathematical pruning condition: if ∑conv[k] > ⌊A⌋ then the composition
    can be soundly pruned. -/
theorem pruning_soundness_with_lut
    (c_target : ℝ) (m n_half_child ℓ : ℕ) (W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n_half_child) (hℓ : 0 < ℓ) (hW : W_int ≤ m)
    (hct : 0 ≤ c_target) (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200)
    (h_exceeds_table : pruning_condition ws
      (table_entry c_target m n_half_child ℓ W_int)) :
    let exact_threshold :=
      ⌊c_target * (m : ℝ) ^ 2 * (ℓ : ℝ) / (4 * (n_half_child : ℝ))
       + 1 + 2 * (W_int : ℝ)⌋
    pruning_condition ws exact_threshold := by
  sorry  -- Rewrite table_entry → runtime_threshold (rfl) → dyn_it
         -- (runtime_threshold_eq_dyn_it), then apply
         -- DynamicThreshold.pruning_soundness with matching hypotheses.

/-- Claim 5.12: The pruning predicate is identical whether the threshold
    comes from the table or the runtime formula. Since table_entry and
    runtime_threshold are definitionally equal (Claim 5.3), the pruning
    decision at every call site is unchanged by the LUT optimization.
    Therefore the set of survivors is identical. -/
theorem lut_pruning_equiv
    (c_target : ℝ) (m n_half_child ℓ : ℕ) (W_int : ℕ) (ws : ℕ) :
    pruning_condition ws (table_entry c_target m n_half_child ℓ W_int) ↔
    pruning_condition ws (runtime_threshold c_target m n_half_child ℓ W_int) :=
  Iff.rfl

end -- noncomputable section
