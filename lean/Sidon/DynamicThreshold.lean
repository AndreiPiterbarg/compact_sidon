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
      c_target_m2_ell = c_target * m * m * ell * inv_4n
      dyn_x = c_target_m2_ell + 1.0 + eps_margin + 2.0 * W_int
      dyn_it = int64(dyn_x * one_minus_4eps)

    Only c_target*m² is scaled by ℓ/(4n); the correction (1+eps+2·W_int)
    is NOT scaled.  Derivation: the test-value domain prune condition is
    TV > c_target + (4n/ℓ)·(1/m² + 2·W/m).  Multiplying both sides by
    m²·ℓ/(4n) gives c_target·m²·ℓ/(4n) + 1 + 2·W_int.  The eps_margin
    and (1-4ε) guard are safety margins for FP rounding.
    See Verification 4 in proof/part1_framework.md.

    The epsilon literal is the exact IEEE 754 float64 machine epsilon. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) +
    1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
   (1 - 4 * (2.220446049250313e-16 : ℝ))⌋

/-
PROBLEM
Upper bound on A used in the conservativeness proof.

PROVIDED SOLUTION
The exact threshold is A = c_target * m² * ℓ/(4n) + 1 + 2*W_int.
Since ℓ/(4n) ≤ 1 (from hℓn), c_target*m²*ℓ/(4n) ≤ c_target*m² ≤ 80000.

Key steps:
1. c_target * m² * ℓ/(4n) ≤ c_target * m² ≤ 2 * 200² = 80000.
2. 1 + 2*W_int ≤ 1 + 400 = 401.
3. A = c_target*m²*ℓ/(4n) + 1 + 2*W_int ≤ 80000 + 401 = 80401.
-/
lemma A_upper_bound (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) + 1 + 2 * (W_int : ℝ) ≤ 80401 := by
  have h_div : (ℓ : ℝ) / (4 * (n : ℝ)) ≤ 1 :=
    div_le_one_of_le₀ (by exact_mod_cast hℓn) (by positivity)
  have h_ct_m2 : c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) ≤ c_target * (m : ℝ)^2 :=
    mul_le_of_le_one_right (by positivity) h_div
  nlinarith [show (m : ℝ) ≤ 200 by exact_mod_cast hm_upper,
             show (W_int : ℝ) ≤ (m : ℝ) by exact_mod_cast hW]

/-
PROBLEM
The epsilon margin 1e-9 * m^2 * (1 - 4*eps) dominates 4*eps * A_max.
    This is the core numerical inequality.

PROVIDED SOLUTION
Pure numerical inequality. Both sides are concrete real number expressions with no variables. Should be provable by norm_num.
-/
lemma margin_dominates :
    (4 : ℝ) * 2.220446049250313e-16 * 80401 ≤ 1e-9 * (1 - 4 * 2.220446049250313e-16) := by
  norm_num +zetaDelta at *

/-
PROBLEM
Claim 2.4: Computed threshold is conservative (≥ exact threshold).

    The exact threshold (from the mathematical pruning condition) is:
      A = c_target * m² * ℓ/(4n) + 1 + 2 * W_int
    The computed threshold (dyn_it) is:
      B = (c_target * m² * ℓ/(4n) + 1 + 1e-9*m² + 2*W_int) * (1-4ε)

    We need ⌊A⌋ ≤ ⌊B⌋, which follows from A ≤ B.

    B = (A + 1e-9*m²) * (1-4ε), so:
      B - A = (A + 1e-9*m²)*(1-4ε) - A
            = 1e-9*m²*(1-4ε) - 4ε*A

    Since 1e-9*m²*(1-4ε) ≥ 1e-9*(1-4ε) ≥ 4ε*80401 ≥ 4ε*A
    (from margin_dominates + A_upper_bound), we get B ≥ A.
-/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) + 1 + 2 * (W_int : ℝ)
    let B := (c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) +
              1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
              (1 - 4 * (2.220446049250313e-16 : ℝ))
    ⌊A⌋ ≤ ⌊B⌋ := by
  refine Int.floor_mono ?_
  -- B = (A + 1e-9*m²) * (1-4ε).  Need B ≥ A, i.e., 1e-9*m²*(1-4ε) ≥ 4ε*A.
  have hP : c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) + 1 + 2 * (W_int : ℝ) ≤ 80401 :=
    A_upper_bound c_target m n ℓ W_int hn hW hct hct_upper hm_upper hℓn
  have hM := margin_dominates
  have hm2 : (m : ℝ) ^ 2 ≥ 1 := by exact_mod_cast pow_pos hm 2
  -- 4ε*A ≤ 4ε*80401 ≤ 1e-9*(1-4ε) ≤ 1e-9*m²*(1-4ε)
  nlinarith

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-
PROBLEM
Pruning with computed threshold implies pruning with exact threshold.

PROVIDED SOLUTION
From dyn_it_conservative: ⌊A⌋ ≤ ⌊B⌋ = dyn_it.
If ws > dyn_it then ws > ⌊A⌋ by transitivity.
-/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := c_target * (m : ℝ)^2 * ((ℓ : ℝ) / (4 * (n : ℝ))) + 1 + 2 * (W_int : ℝ)
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  exact fun h => lt_of_le_of_lt (by simpa using dyn_it_conservative c_target m n ℓ W_int hm hn hW hct hct_upper hm_upper hℓn) h

end
