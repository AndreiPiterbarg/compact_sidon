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
      dyn_base = c_target * m² + 1 + eps_margin
      dyn_x = (dyn_base + 2 * W_int) * ℓ / (4n)
      dyn_it = int64(dyn_x * one_minus_4eps)

    ALL terms are scaled by ℓ/(4n).  Derivation: the test-value domain
    prune condition is TV > c_target + 1/m² + 2*W/m.  Multiplying both
    sides by m²·ℓ/(4n) gives the integer-space threshold
    (c_target·m² + 1 + 2·W_int) · ℓ/(4n).  The eps_margin and (1-4ε)
    guard are safety margins for FP rounding.
    See Verification 4 in proof/part1_framework.md.

    The epsilon literal is the exact IEEE 754 float64 machine epsilon. -/
noncomputable def dyn_it (c_target : ℝ) (m n ℓ W_int : ℕ) : ℤ :=
  ⌊(c_target * (m : ℝ)^2 + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
   ((ℓ : ℝ) / (4 * (n : ℝ))) *
   (1 - 4 * (2.220446049250313e-16 : ℝ))⌋

/-
PROBLEM
Upper bound on A used in the conservativeness proof.

PROVIDED SOLUTION
The exact threshold is A = (c_target * m² + 1 + 2*W_int) * ℓ/(4n).
Since ℓ/(4n) ≤ 1 (from hℓn) and the inner sum ≤ 80401, A ≤ 80401.

Key steps:
1. c_target * m² ≤ 2 * 200² = 80000.
2. 1 + 2*W_int ≤ 1 + 400 = 401.
3. Inner = c_target*m² + 1 + 2*W_int ≤ 80401.
4. ℓ/(4n) ≤ 1, so A = Inner * ℓ/(4n) ≤ Inner ≤ 80401.
-/
lemma A_upper_bound (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ))) ≤ 80401 := by
  have h_inner : c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ) ≤ 80401 := by
    nlinarith [show (m : ℝ) ≤ 200 by exact_mod_cast hm_upper,
               show (W_int : ℝ) ≤ (m : ℝ) by exact_mod_cast hW]
  have h_inner_nn : 0 ≤ c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ) := by positivity
  have h_div : (ℓ : ℝ) / (4 * (n : ℝ)) ≤ 1 :=
    div_le_one_of_le₀ (by exact_mod_cast hℓn) (by positivity)
  calc (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
      ≤ (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * 1 :=
        mul_le_mul_of_nonneg_left h_div h_inner_nn
    _ = c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ) := by ring
    _ ≤ 80401 := h_inner

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
      A = (c_target * m² + 1 + 2 * W_int) * ℓ/(4n)
    The computed threshold (dyn_it) is:
      B = (c_target * m² + 1 + 1e-9*m² + 2*W_int) * ℓ/(4n) * (1-4ε)

    We need ⌊A⌋ ≤ ⌊B⌋, which follows from A ≤ B.

    Write P = c_target*m² + 1 + 2*W_int.  Then:
      B = (P + 1e-9*m²) * s * (1-4ε)   where s = ℓ/(4n)
      A = P * s
      B - A = s * [(P + 1e-9*m²)*(1-4ε) - P]
            = s * [1e-9*m²*(1-4ε) - 4ε*P]

    Since s ≥ 0 and 1e-9*m²*(1-4ε) ≥ 1e-9*(1-4ε) ≥ 4ε*80401 ≥ 4ε*P
    (from margin_dominates + A_upper_bound), we get B ≥ A.
-/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let B := (c_target * (m : ℝ)^2 + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
              ((ℓ : ℝ) / (4 * (n : ℝ))) *
              (1 - 4 * (2.220446049250313e-16 : ℝ))
    ⌊A⌋ ≤ ⌊B⌋ := by
  refine Int.floor_mono ?_
  -- Suffices: (P + eps*m²)*(1-4ε) ≥ P, then multiply by s = ℓ/(4n) ≥ 0
  have hs : 0 ≤ (ℓ : ℝ) / (4 * (n : ℝ)) := by positivity
  -- Core inequality: 1e-9*m²*(1-4ε) ≥ 4ε*P where P ≤ 80401
  have hP : c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ) ≤ 80401 := by
    nlinarith [show (m : ℝ) ≤ 200 by exact_mod_cast hm_upper,
               show (W_int : ℝ) ≤ (m : ℝ) by exact_mod_cast hW]
  have hM := margin_dominates
  have hm2 : (m : ℝ) ^ 2 ≥ 1 := by exact_mod_cast pow_pos hm 2
  -- A = P * s, B = (P + eps*m²) * s * q where q = 1-4ε
  -- B - A = s * ((P + eps*m²)*q - P) = s * (eps*m²*q - 4ε*P) ≥ 0
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
    let A := (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  exact fun h => lt_of_le_of_lt (by simpa using dyn_it_conservative c_target m n ℓ W_int hm hn hW hct hct_upper hm_upper hℓn) h

end
