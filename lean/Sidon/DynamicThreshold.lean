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
-- Integer Dynamic Threshold (C&S Lemma 3 + W-refinement)
--
-- C&S Lemma 3 gives a POINTWISE bound: (g*g)(x) ≤ (f*f)(x) + 2/m + 1/m².
-- Since test values are window averages, the correction is window-independent:
--   TV_g(ℓ,s) ≤ TV_f(ℓ,s) + 2/m + 1/m²  (no 4n/ℓ factor)
--
-- With W-refinement (eq. 1 of C&S) and W_f→W_g correction:
--   TV_g(ℓ,s) ≤ TV_f(ℓ,s) + (3 + 2·W_int)/m²
--
-- In integer space, the ENTIRE threshold is scaled by ℓ/(4n):
--   threshold = floor((c_target·m² + 3 + 2·W_int + eps) · ℓ/(4n))
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Integer convolution for exact computation. -/
def conv {d : ℕ} (c : Fin d → ℕ) (k : ℕ) : ℕ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0

/-- Window sum of integer convolution. -/
def window_sum {d : ℕ} (c : Fin d → ℕ) (s_lo ℓ : ℕ) : ℕ :=
  ∑ k ∈ Finset.Ico s_lo (s_lo + ℓ - 1), conv c k

/-- Integer correction: 3 + 2·W_int (C&S Lemma 3 + W-refinement + W_f→W_g).
    The +3 = +1 (|ε*ε| ≤ 1/m²) + 2 (W_f ≤ W_g + 1/m, cumulative rounding). -/
noncomputable def correction_int (_m W_int : ℕ) : ℝ :=
  3 + 2 * (W_int : ℝ)

/-- Exact integer threshold for pruning (C&S Lemma 3).
    The ENTIRE expression is scaled by ℓ/(4n):
      threshold = floor((c_target·m² + 3 + 2·W_int) · ℓ/(4n)) -/
noncomputable def exact_threshold (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 + correction_int m W_int) * ((ℓ : ℝ) / (4 * (n : ℝ)))⌋

/-- Computed threshold with FP safety margins.
    eps_margin = 1e-9·m² is included INSIDE the scaling to match the code. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 + correction_int m W_int + 1e-9 * (m : ℝ)^2) *
   ((ℓ : ℝ) / (4 * (n : ℝ)))⌋

/-- Claim 2.4: Computed threshold is conservative (≥ exact threshold). -/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (_hct : 0 ≤ c_target)
    (_hct_upper : c_target ≤ 2) (_hm_upper : m ≤ 200) :
    exact_threshold c_target m n ℓ W_int ≤ dyn_it c_target m n ℓ W_int := by
  unfold exact_threshold dyn_it
  apply Int.floor_mono
  have heps : (0 : ℝ) ≤ 1e-9 * (m : ℝ) ^ 2 := by positivity
  have hscale : (0 : ℝ) ≤ (ℓ : ℝ) / (4 * (n : ℝ)) := by positivity
  nlinarith

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-- Pruning with computed threshold implies pruning with exact threshold. -/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) :
    pruning_condition ws (dyn_it c_target m n ℓ W_int) →
    pruning_condition ws (exact_threshold c_target m n ℓ W_int) := by
  intro h_pruning
  have h_le := dyn_it_conservative c_target m n ℓ W_int hm hn hℓ hW hct hct_upper hm_upper
  exact h_pruning.trans_le' h_le

/-- The correction_int is always nonneg. -/
theorem correction_int_nonneg (m W_int : ℕ) : 0 ≤ correction_int m W_int := by
  unfold correction_int; positivity

end -- noncomputable section
