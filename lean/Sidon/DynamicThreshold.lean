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

/-
PROBLEM
Upper bound on A used in the conservativeness proof.

PROVIDED SOLUTION
We need: c_target * m^2 * ℓ/(4*n) + 1 + 2*W_int ≤ 80401.

Key steps:
1. From hℓn: ℓ ≤ 4*n, so (ℓ : ℝ) ≤ 4*(n : ℝ), hence ℓ/(4*n) ≤ 1 (since 4*n > 0 from hn).
2. From hct_upper and hm_upper: c_target * m^2 ≤ 2 * 200^2 = 80000. Combined with step 1: c_target * m^2 * ℓ/(4*n) ≤ 80000.
3. From hW and hm_upper: W_int ≤ m ≤ 200, so 2*W_int ≤ 400.
4. Total: ≤ 80000 + 1 + 400 = 80401.

For step 1, use div_le_one_of_le or similar. The key is that ℓ/(4*n) means real division, and we need cast_le for ℓ ≤ 4*n, then divide both sides by 4*n.

For step 2, use mul_le_mul with hct_upper and sq_le_sq' from hm_upper.

Use nlinarith or gcongr for the final combination.
-/
lemma A_upper_bound (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 2 * (W_int : ℝ) ≤ 80401 := by
  -- Since $\ell \leq 4n$, we have $\frac{\ell}{4n} \leq 1$, thus $c_target * m^2 * \frac{\ell}{4n} \leq c_target * m^2$.
  have h1 : c_target * m^2 * ℓ / (4 * n) ≤ c_target * m^2 := by
    -- Since $\ell \leq 4n$, we have $\ell / (4n) \leq 1$. Therefore, multiplying by $\ell / (4n)$ would make the term smaller than multiplying by 1.
    have h_div : (ℓ : ℝ) / (4 * n) ≤ 1 := by
      exact div_le_one_of_le₀ ( mod_cast hℓn ) ( by positivity );
    simpa only [ mul_div_assoc ] using mul_le_of_le_one_right ( by positivity ) h_div;
  exact le_trans ( add_le_add_three h1 le_rfl ( mul_le_mul_of_nonneg_left ( Nat.cast_le.mpr hW ) zero_le_two ) ) ( by nlinarith [ show ( m : ℝ ) ≤ 200 by norm_cast, show ( c_target : ℝ ) ≤ 2 by norm_cast ] )

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
    The computed threshold (dyn_it) adds eps_margin and applies (1-4ε):
      B = (c_target * m² * ℓ/(4n) + 1 + 1e-9*m² + 2*W_int) * (1-4ε)

    We need ⌊A⌋ ≤ ⌊B⌋, which follows from A ≤ B.

    NOTE: The hypothesis hℓn : ℓ ≤ 4 * n is added because the 1e-9 margin only
    dominates the 4ε rounding for bounded ℓ/n ratios. In the algorithm, ℓ ranges
    over {2, ..., 2*d} = {2, ..., 4*n}, so this always holds.

PROVIDED SOLUTION
We need floor(A) ≤ floor(B). It suffices to show A ≤ B, then apply Int.floor_mono.

Write B = (A + 1e-9 * m^2) * (1 - 4*eps) where eps = 2.220446049250313e-16.
So B = A - 4*eps*A + 1e-9*m^2 - 4*eps*1e-9*m^2 = A + 1e-9*m^2*(1 - 4*eps) - 4*eps*A.
Thus B - A = 1e-9*m^2*(1-4*eps) - 4*eps*A.

We need: 1e-9*m^2*(1-4*eps) ≥ 4*eps*A.

From A_upper_bound: A ≤ 80401.
From margin_dominates: 4*eps*80401 ≤ 1e-9*(1-4*eps).
Since m ≥ 1 (from hm), m^2 ≥ 1, so 1e-9*m^2*(1-4*eps) ≥ 1e-9*(1-4*eps) ≥ 4*eps*80401 ≥ 4*eps*A.

Therefore B ≥ A, and Int.floor_mono gives floor(A) ≤ floor(B).

Steps:
1. intro the lets
2. apply Int.floor_mono (or Int.floor_le_floor)
3. have hA := A_upper_bound ... (to get A ≤ 80401)
4. have hM := margin_dominates (to get 4*eps*80401 ≤ 1e-9*(1-4*eps))
5. Show B ≥ A by nlinarith using hA, hM, and hm (m ≥ 1 so m^2 ≥ 1)
-/
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 2 * (W_int : ℝ)
    let B := (c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 1e-9 * (m : ℝ)^2 +
              2 * (W_int : ℝ)) * (1 - 4 * (2.220446049250313e-16 : ℝ))
    ⌊A⌋ ≤ ⌊B⌋ := by
  refine' Int.floor_mono _;
  have hA := A_upper_bound c_target m n ℓ W_int hn hW hct hct_upper hm_upper hℓn;
  have hM := margin_dominates ; norm_num at * ; nlinarith [ show ( m : ℝ ) ^ 2 ≥ 1 by exact_mod_cast pow_pos hm 2 ] ;

/-- Pruning condition predicate. -/
def pruning_condition (ws : ℕ) (threshold : ℤ) : Prop :=
  (ws : ℤ) > threshold

/-
PROBLEM
Pruning with computed threshold implies pruning with exact threshold.

PROVIDED SOLUTION
Unfold pruning_condition and dyn_it. The goal becomes: if (ws : ℤ) > dyn_it ... then (ws : ℤ) > ⌊A⌋.

From dyn_it_conservative (applied with the same args), we get ⌊A⌋ ≤ dyn_it c_target m n ℓ W_int.
The dyn_it is ⌊B⌋ which equals the computed_threshold.

So chain: ⌊A⌋ ≤ computed_threshold < ws, giving ⌊A⌋ < ws.

Use lt_of_le_of_lt or linarith/omega.
-/
theorem pruning_soundness (c_target : ℝ) (m n ℓ W_int : ℕ) (ws : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hW : W_int ≤ m) (hct : 0 ≤ c_target)
    (hct_upper : c_target ≤ 2) (hm_upper : m ≤ 200) (hℓn : ℓ ≤ 4 * n) :
    let A := c_target * (m : ℝ)^2 * (ℓ : ℝ) / (4 * (n : ℝ)) + 1 + 2 * (W_int : ℝ)
    let exact_threshold := ⌊A⌋
    let computed_threshold := dyn_it c_target m n ℓ W_int
    pruning_condition ws computed_threshold → pruning_condition ws exact_threshold := by
  exact fun h => lt_of_le_of_lt ( by simpa using dyn_it_conservative c_target m n ℓ W_int hm hn hW hct hct_upper hm_upper hℓn ) h

end