/-
Prompt 02: Discretization Correction Term — Lemma 3 of CS14 (Claim 1.2)

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

-- Definitions from complete_proof.lean (repeated for self-containment)

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

noncomputable def max_test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ :=
  let d := 2 * n
  let range_ell := Finset.Icc 2 (2 * d)
  let range_s_lo := Finset.range (2 * d)
  let values := range_ell.biUnion (fun ℓ => range_s_lo.image (fun s_lo => test_value n m c ℓ s_lo))
  if h : values.Nonempty then values.max' h else 0

-- ═══════════════════════════════════════════════
-- CLAIM 1.2: Correction term 2/m + 1/m²
-- ═══════════════════════════════════════════════

theorem correction_term (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    autoconvolution_ratio f ≥
      (max_test_value n m (canonical_discretization f n m) : ℝ) - 2 / m - 1 / m ^ 2 := by
  sorry

end
