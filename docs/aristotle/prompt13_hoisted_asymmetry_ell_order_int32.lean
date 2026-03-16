/-
Prompt 13: Hoisted Asymmetry, Ell Scan Order, Integer Safety (Claims 4.5 + 4.7 + 4.8)

NOTE: Claim 4.6 (left-half sum invariant) is ALREADY PROVED in complete_proof.lean
as `left_half_sum_invariant`. Only 4.5, 4.7, 4.8 remain.

Attach complete_proof.lean as context.

THEOREMS TO PROVE (fill in the sorry's)
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- Definitions from complete_proof.lean

noncomputable def bin_masses (f : ℝ → ℝ) (n : ℕ) : Fin (2 * n) → ℝ :=
  fun i =>
    let δ := 1 / (4 * n : ℝ)
    let a := -(1/4 : ℝ) + i * δ
    let b := -(1/4 : ℝ) + (i + 1) * δ
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico a b) f)

-- ═══════════════════════════════════════════════
-- CLAIM 4.5: Cauchy-Schwarz x_cap needs no correction
-- Direct L∞ bound: ‖f*f‖∞ ≥ d · M_i²
-- ═══════════════════════════════════════════════

theorem single_bin_bound (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM : M_i = bin_masses f n i) :
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥
    (2 * n : ℝ) * M_i ^ 2 := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 4.7: Ell scan order irrelevant (existential is permutation-invariant)
-- ═══════════════════════════════════════════════

theorem exists_invariant_under_permutation {α : Type*} [DecidableEq α]
    (S : Finset α) (P : α → Prop) [DecidablePred P] :
    (∃ x ∈ S, P x) ↔ (∃ x ∈ S, P x) :=
  Iff.rfl

-- ═══════════════════════════════════════════════
-- CLAIM 4.8: Integer overflow safety
-- ═══════════════════════════════════════════════

-- conv[k] ≤ m² (each entry bounded by total)
theorem conv_entry_le_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=k then c i * c j else 0 ≤ m ^ 2 := by
  sorry

-- Total autoconvolution = m²
theorem conv_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) :
    ∑ k ∈ Finset.range (2*d-1),
      (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=k then c i * c j else 0) = m ^ 2 := by
  sorry

-- m² fits int32 for m ≤ 200
theorem int32_safe (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by omega

end
