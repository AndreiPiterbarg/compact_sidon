/-
Prompt 03: Dynamic Threshold and Contributing Bins (Claims 1.3 + 1.4)

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

noncomputable def autoconvolution_ratio (f : ℝ → ℝ) : ℝ :=
  let conv := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  let norm_inf := (MeasureTheory.eLpNorm conv ⊤ MeasureTheory.volume).toReal
  let integral := MeasureTheory.integral MeasureTheory.volume f
  norm_inf / (integral ^ 2)

noncomputable def bin_masses (f : ℝ → ℝ) (n : ℕ) : Fin (2 * n) → ℝ :=
  fun i =>
    let δ := 1 / (4 * n : ℝ)
    let a := -(1/4 : ℝ) + i * δ
    let b := -(1/4 : ℝ) + (i + 1) * δ
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico a b) f)

noncomputable def canonical_discretization (f : ℝ → ℝ) (n m : ℕ) : Fin (2 * n) → ℕ :=
  fun i =>
    let masses := bin_masses f n
    let total_mass := ∑ j, masses j
    let cum_mass (k : ℕ) := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
    let target_cum (k : ℕ) := (cum_mass k) / total_mass * m
    let discrete_cum (k : ℕ) := ⌊target_cum k⌋.natAbs
    if i.1 + 1 < 2 * n then discrete_cum (i.1 + 1) - discrete_cum i.1
    else m - discrete_cum i.1

def discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then a i * a j else 0

noncomputable def test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  let d := 2 * n
  let a : Fin d → ℝ := fun i => (4 * n : ℝ) / m * (c i : ℝ)
  let conv := discrete_autoconvolution a
  let sum_conv := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), conv k
  (1 / (4 * n * ℓ : ℝ)) * sum_conv

def contributing_bins (n : ℕ) (ℓ s_lo : ℕ) : Finset (Fin (2 * n)) :=
  let d := 2 * n
  Finset.filter (fun i => ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2) Finset.univ

-- ═══════════════════════════════════════════════
-- CLAIM 1.4: Contributing bins formula
-- ═══════════════════════════════════════════════

theorem contributing_bins_iff (n : ℕ) (hn : n > 0) (ℓ s_lo : ℕ)
    (hℓ : 2 ≤ ℓ) (i : Fin (2 * n)) :
    i ∈ contributing_bins n ℓ s_lo ↔
      Nat.max 0 (s_lo - (2 * n - 1)) ≤ i.1 ∧ i.1 ≤ Nat.min (2 * n - 1) (s_lo + ℓ - 2) := by
  sorry

-- ═══════════════════════════════════════════════
-- CLAIM 1.3: Dynamic threshold soundness
-- ═══════════════════════════════════════════════

theorem dynamic_threshold_sound (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (c i : ℝ)) / m)
    (h_exceeds : test_value n m c ℓ s_lo > c_target + 1 / m ^ 2 + 2 * W / m) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  sorry

end
