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

    Matches the Python code (run_cascade.py, all code paths):
      cs_corr_base = c_target * m * m + 3.0 + eps_margin
      ct_base_ell = cs_corr_base * ell * inv_4n
      w_scale = 2.0 * ell * inv_4n
      dyn_x = ct_base_ell + w_scale * W_int
      dyn_it = int64(dyn_x * one_minus_4eps)

    The ENTIRE expression (c_target*m² + 3 + 2*W_int + eps) is scaled by
    ℓ/(4n).  The +3 accounts for: +1 from |ε·ε| ≤ 1/m², +2 from
    W_f ≤ W_g + 1/m (cumulative rounding correction).
    Derivation: C&S Lemma 3 + eq(1) W-refinement.

    The epsilon literal is the exact IEEE 754 float64 machine epsilon.

    NOTE: This definition is NOT on the critical proof path. FinalResult.lean
    uses cascade_all_pruned + dynamic_threshold_sound (DiscretizationError.lean)
    which derives a per-window correction (4n/ℓ)·(1/m² + 2W/m) directly.
    This file formalizes the CPU's integer threshold for LUT correctness. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 + 3 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
   ((ℓ : ℝ) / (4 * (n : ℝ))) *
   (1 - 4 * (2.220446049250313e-16 : ℝ))⌋

/-
PROBLEM
Upper bound on A used in the conservativeness proof.

PROVIDED SOLUTION
The exact threshold (matching CPU) is
  A = (c_target*m² + 3 + 2*W_int) * ℓ/(4n).
Since ℓ/(4n) ≤ 1: A ≤ c_target*m² + 3 + 2*W_int ≤ 80000 + 3 + 400 = 80403.
-/
lemma A_upper_bound (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    (c_target * (m : ℝ)^2 + 3 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ))) ≤ 80403 := by
  sorry -- TODO: straightforward bound, needs re-proof after formula change

/-
PROBLEM
The epsilon margin 1e-9 * m^2 * (1 - 4*eps) dominates 4*eps * A_max.
    This is the core numerical inequality.

PROVIDED SOLUTION
Pure numerical inequality. Both sides are concrete real number expressions with no variables. Should be provable by norm_num.
-/
lemma margin_dominates :
    (4 : ℝ) * 2.220446049250313e-16 * 80403 ≤ 1e-9 * (1 - 4 * 2.220446049250313e-16) := by
  norm_num +zetaDelta at *

/-
PROBLEM
Claim 2.4: Computed threshold is conservative (≥ exact threshold).

    The exact threshold (matching CPU, in integer ws-space) is:
      A = (c_target*m² + 3 + 2*W_int) * ℓ/(4n)
    The computed threshold (dyn_it) is:
      B = (c_target*m² + 3 + 1e-9*m² + 2*W_int) * ℓ/(4n) * (1-4ε)

    We need ⌊A⌋ ≤ ⌊B⌋, which follows from A ≤ B.

    B = A*(1-4ε) + 1e-9*m²*ℓ/(4n)*(1-4ε), so:
      B - A = 1e-9*m²*ℓ/(4n)*(1-4ε) - 4ε*A

    Since 1e-9*m²*ℓ/(4n)*(1-4ε) ≥ ... ≥ 4ε*A (from margin_dominates +
    A_upper_bound), we get B ≥ A.

    NOTE: Not on critical proof path (see file header). -/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := (c_target * (m : ℝ)^2 + 3 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let B := (c_target * (m : ℝ)^2 + 3 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
              ((ℓ : ℝ) / (4 * (n : ℝ))) *
              (1 - 4 * (2.220446049250313e-16 : ℝ))
    ⌊A⌋ ≤ ⌊B⌋ := by
  sorry -- TODO: re-prove after formula alignment with CPU code

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-
PROBLEM
Pruning with computed threshold implies pruning with exact threshold.

PROVIDED SOLUTION
From dyn_it_conservative: ⌊A⌋ ≤ ⌊B⌋ = dyn_it.
If ws > dyn_it then ws > ⌊A⌋ by transitivity.

NOTE: The exact threshold here matches the CPU code:
  A = (c_target*m² + 3 + 2*W_int) * ℓ/(4n)
This is the integer-space equivalent of the CPU's pruning condition.
-/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := (c_target * (m : ℝ)^2 + 3 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  sorry -- TODO: re-prove after formula alignment with CPU code

end
