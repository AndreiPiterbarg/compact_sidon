import Mathlib

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
-- Integer Dynamic Threshold (Claims 2.4, 5.1, 5.2)
-- Source: output (5).lean (UUID: d81b0331)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Integer convolution for exact computation. -/
def conv {d : ℕ} (c : Fin d → ℕ) (k : ℕ) : ℕ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0

/-- Window sum of integer convolution. -/
def window_sum {d : ℕ} (c : Fin d → ℕ) (s_lo ℓ : ℕ) : ℕ :=
  ∑ k ∈ Finset.Ico s_lo (s_lo + ℓ - 1), conv c k

/-- Dynamic threshold for pruning.

    Matches the Python code (run_cascade.py:1111-1114):
      dyn_base_ell = c_target * m² * ℓ / (4n)
      dyn_x = dyn_base_ell + 1 + eps_margin + 2 * W_int
      dyn_it = int64(dyn_x * one_minus_4eps)

    IMPORTANT: ℓ/(4n) scales ONLY c_target * m², NOT the correction terms
    (1 + eps_margin + 2*W_int). The correction enters the integer threshold
    comparison directly. This is MORE CONSERVATIVE than scaling everything
    by ℓ/(4n), since ℓ/(4n) ≤ 1.

    The epsilon literal is the exact IEEE 754 float64 machine epsilon. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
   (1 - 4 * (2.220446049250313e-16 : ℝ))⌋

/-- Claim 2.4: Computed threshold is conservative (≥ exact threshold).

    The exact threshold (from the mathematical pruning condition) is:
      A = c_target * m² * ℓ/(4n) + 1 + 2 * W_int
    The computed threshold (dyn_it) adds eps_margin and applies (1-4ε):
      B = (c_target * m² * ℓ/(4n) + 1 + 1e-9*m² + 2*W_int) * (1-4ε)

    We need B ≥ A, i.e., (A + 1e-9*m²) * (1-4ε) ≥ A,
    i.e., 1e-9 * m² * (1-4ε) ≥ 4ε * A.
    For m ≤ 200, c_target ≤ 2, ℓ/(4n) ≤ 1, W_int ≤ m:
      LHS ≈ 1e-9 * m² ≥ 1e-9, RHS ≤ 4ε * 80401 ≈ 7.14e-11.
    So LHS >> RHS. -/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (_hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) :
    let A := c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 2 * (W_int : ℝ)
    let B := (c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 1e-9 * (m : ℝ)^2 +
              2 * (W_int : ℝ)) * (1 - 4 * (2.220446049250313e-16 : ℝ))
    ⌊A⌋ ≤ ⌊B⌋ := by
  sorry  -- Int.floor_mono; need B ≥ A, see docstring for arithmetic sketch

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-- Pruning with computed threshold implies pruning with exact threshold. -/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) :
    let A := c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 2 * (W_int : ℝ)
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  sorry  -- From dyn_it_conservative: computed_threshold ≥ exact_threshold; chain with h_pruning

end -- noncomputable section
