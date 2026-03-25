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

/-- Dynamic threshold for pruning. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
   ((ℓ : ℝ) / (4 * (n : ℝ))) * (1 - 4 * 2.22e-16)⌋

/-- Claim 2.4: Computed threshold is conservative (≥ exact threshold). -/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) :
    let A := (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let B := (c_target * (m : ℝ)^2 + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
             ((ℓ : ℝ) / (4 * (n : ℝ))) * (1 - 4 * 2.22e-16)
    ⌊A⌋ ≤ ⌊B⌋ := by
  refine' Int.floor_mono _;
  have h_W_le_m : (W_int : ℝ) ≤ m := by
    norm_cast;
  have h_m_le_200 : (m : ℝ) ≤ 200 := by
    norm_cast;
  field_simp;
  norm_num; nlinarith [ show ( 1 : ℝ ) ≤ m ^ 2 by exact_mod_cast pow_pos hm 2 ] ;

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-- Pruning with computed threshold implies pruning with exact threshold. -/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) :
    let A := (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  intro A exact_threshold computed_threshold h_pruning
  have h_computed_gt_exact : computed_threshold ≥ exact_threshold := by
    convert dyn_it_conservative c_target m n ℓ W_int hm hn hℓ hW hct hct_upper hm_upper using 1
  exact h_pruning.trans_le' h_computed_gt_exact

end -- noncomputable section
