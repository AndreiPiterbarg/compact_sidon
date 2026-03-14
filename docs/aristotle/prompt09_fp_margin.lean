/-
PROMPT FOR ARISTOTLE: Prove FP margin is net conservative (Claims 2.4 + 5.1).

The dynamic threshold in the cascade is computed as:
  dyn_it = ⌊(A + δ) · (1 - 4ε)⌋
where:
  A = (c_target · m² + 1 + 2·W_int) · ℓ/(4n)    (exact mathematical threshold)
  δ = 1e-9 · m² · ℓ/(4n)                         (conservative additive margin)
  ε = 2.22e-16                                    (IEEE 754 double machine epsilon)

We must prove: dyn_it ≥ ⌊A⌋ (the computed threshold is at least the exact threshold).

Proof: We need (A + δ)·(1 - 4ε) ≥ A, i.e., δ·(1 - 4ε) ≥ 4ε·A.
The additive margin δ ≈ 1e-7 dominates the multiplicative reduction 4ε·A ≈ 1e-12
by a factor of ~10⁵. So the inequality holds with massive margin.

This is a pure real analysis proof about floor functions. No measure theory needed.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

section

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
HELPER: If x ≤ y then ⌊x⌋ ≤ ⌊y⌋ (floor is monotone).
This should be in Mathlib as Int.floor_le_floor or similar.
-/
theorem floor_mono_real {x y : ℝ} (h : x ≤ y) : ⌊x⌋ ≤ ⌊y⌋ := by
  exact Int.floor_le_floor h

/-
CORE LEMMA: If A ≥ 0, δ > 0, 0 < ε < 1/4, and δ·(1 - 4ε) ≥ 4ε·A,
then (A + δ)·(1 - 4ε) ≥ A.

Proof:
  (A + δ)·(1 - 4ε) = A·(1 - 4ε) + δ·(1 - 4ε)
                    = A - 4ε·A + δ·(1 - 4ε)
                    ≥ A - 4ε·A + 4ε·A       (by hypothesis δ·(1-4ε) ≥ 4ε·A)
                    = A.
-/
theorem margin_dominates (A δ ε : ℝ) (hA : 0 ≤ A) (hδ : 0 < δ)
    (hε_pos : 0 < ε) (hε_small : 4 * ε < 1)
    (h_dom : δ * (1 - 4 * ε) ≥ 4 * ε * A) :
    (A + δ) * (1 - 4 * ε) ≥ A := by
  sorry

/-
MAIN THEOREM (Claims 2.4 + 5.1): The computed threshold is conservative.

⌊(A + δ) · (1 - 4ε)⌋ ≥ ⌊A⌋

where A ≥ 0 and the margin dominance condition holds.
-/
theorem dyn_it_conservative (A δ ε : ℝ) (hA : 0 ≤ A) (hδ : 0 < δ)
    (hε_pos : 0 < ε) (hε_small : 4 * ε < 1)
    (h_dom : δ * (1 - 4 * ε) ≥ 4 * ε * A) :
    ⌊(A + δ) * (1 - 4 * ε)⌋ ≥ ⌊A⌋ := by
  sorry

/-
CONCRETE INSTANCE: For our parameters m = 20, n = 2, c_target = 1.4, ε = 2.22e-16.

The margin δ = 1e-9 · m² · ℓ/(4n) ≥ 1e-9 · 400 · 2/8 = 1e-7.
The reduction 4ε·A ≤ 4·2.22e-16 · 9616 ≈ 8.5e-12.
Since 1e-7 >> 8.5e-12, the dominance condition holds.

Verify: 1e-7 · (1 - 4·2.22e-16) ≥ 4·2.22e-16 · 9616.
LHS ≈ 1e-7. RHS ≈ 8.54e-12. LHS/RHS ≈ 11700. ✓
-/
theorem concrete_margin_safe :
    let ε : ℝ := 2.22e-16
    let m : ℝ := 20
    let n : ℝ := 2
    let c_target : ℝ := 1.4
    let A_max : ℝ := (c_target * m^2 + 1 + 2 * m) * (2 * (2 * n) / (4 * n))  -- ℓ=2d, worst case
    let δ_min : ℝ := 1e-9 * m^2 * (2 / (4 * n))  -- ℓ=2, best case
    δ_min * (1 - 4 * ε) ≥ 4 * ε * A_max := by
  sorry

/-
SOUNDNESS: If ws > dyn_it (integer comparison), then the composition is correctly pruned.

ws is an exact integer (autoconvolution of integer masses).
dyn_it = ⌊(threshold)⌋ is an integer.
ws > dyn_it means ws ≥ dyn_it + 1.

Since dyn_it ≥ ⌊exact_threshold⌋ (by dyn_it_conservative),
and ws > dyn_it ≥ ⌊exact_threshold⌋,
the continuous test value exceeds the continuous threshold.
-/
theorem pruning_sound_from_integer_comparison
    (ws : ℤ) (dyn_it : ℤ) (exact_threshold : ℝ)
    (h_ws_exact : True)  -- ws is exact (just a type annotation)
    (h_dyn_conservative : dyn_it ≥ ⌊exact_threshold⌋)
    (h_exceeds : ws > dyn_it) :
    (ws : ℝ) > exact_threshold := by
  sorry

end
