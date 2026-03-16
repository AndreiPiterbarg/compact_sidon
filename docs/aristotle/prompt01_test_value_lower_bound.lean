/-
Prompt 01: Test Value is a Lower Bound on ‖f∗f‖∞ (Claim 1.1)

Attach complete_proof.lean as context — it has all definitions and foundational lemmas.

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

-- These definitions are in complete_proof.lean (provided as context).
-- Repeated here so the file is self-contained for Aristotle.

noncomputable def autoconvolution_ratio (f : ℝ → ℝ) : ℝ :=
  let conv := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  let norm_inf := (MeasureTheory.eLpNorm conv ⊤ MeasureTheory.volume).toReal
  let integral := MeasureTheory.integral MeasureTheory.volume f
  norm_inf / (integral ^ 2)

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

-- Step function on the 2n-bin grid
noncomputable def step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ → ℝ :=
  fun x =>
    let d := 2 * n
    let δ := 1 / (4 * n : ℝ)
    if x < -1/4 ∨ x ≥ 1/4 then 0
    else
      let i := ⌊(x + 1/4) / δ⌋.toNat
      if h : i < d then (c ⟨i, h⟩ : ℝ) / m
      else 0

-- ═══════════════════════════════════════════════
-- CLAIM 1.1: Test value ≤ ‖f*f‖∞
-- ═══════════════════════════════════════════════

theorem test_value_le_Linfty (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (max_test_value n m c : ℝ) ≤ autoconvolution_ratio (step_function n m c) := by
  sorry

end
