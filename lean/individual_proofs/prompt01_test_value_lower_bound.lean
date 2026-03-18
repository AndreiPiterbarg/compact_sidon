/-
PROMPT FOR ARISTOTLE: Prove that the test value is a lower bound on ‖f∗f‖∞ (Claim 1.1).

GOAL: Prove `test_value_le_Linfty` — for a step function with masses c summing to m,
the max test value is at most the autoconvolution ratio.

The proof is decomposed into helper lemmas. The main theorem `test_value_le_Linfty`
is proved modulo two measure-theoretic sorry's:
  - `convolution_at_grid_point`: evaluates the Lebesgue convolution at a grid point
  - `eLpNorm_top_ge_of_continuous_at`: essSup ≥ pointwise value for continuous functions

PROOF STRATEGY:
1. step_function is piecewise constant: f(x) = c_i/m on bin i (width Δ = 1/(4n)).
2. (f*f)(x) is piecewise-linear on intervals of length Δ.
3. At grid points: (f*f)(kΔ - 1/4) = (1/m²) · conv_c[k] where conv_c[k] = ∑_{i+j=k} c_i·c_j.
4. Integral of (f*f) over [kΔ, (k+1)Δ] ≥ Δ · conv_c[k] / m² (trapezoidal lower bound).
5. Test value = windowed average of rescaled convolution ≤ ‖f*f‖∞ / (∫f)².
6. Use max_test_value_le_max to reduce to a specific window.

KEY: The rescaled a_i = (4n/m)·c_i gives conv_a[k] = (4n/m)²·conv_c[k].
TV = (1/(4nℓ))·∑ conv_a[k] = average of f*f over a window ≤ ‖f*f‖∞.
And ∫f = 1 for the step function with total mass 1, so R(f) = ‖f*f‖∞.
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

/-! ## Definitions -/

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

noncomputable def step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ → ℝ :=
  fun x =>
    let d := 2 * n
    let δ := 1 / (4 * n : ℝ)
    if x < -1/4 ∨ x ≥ 1/4 then 0
    else
      let i := ⌊(x + 1/4) / δ⌋.toNat
      if h : i < d then (c ⟨i, h⟩ : ℝ) / m
      else 0

/-! ## Proved helper lemmas -/

/-- The max test value is attained at some window parameters. -/
lemma max_test_value_le_max (n m : ℕ) (hn : n > 0) (c : Fin (2 * n) → ℕ) :
    ∃ ℓ s_lo, ℓ ∈ Finset.Icc 2 (2 * (2 * n)) ∧ s_lo ∈ Finset.range (2 * (2 * n)) ∧
    max_test_value n m c = test_value n m c ℓ s_lo := by
  unfold max_test_value;
  simp +zetaDelta at *;
  split_ifs with h;
  · have := Finset.max'_mem ( Finset.biUnion ( Finset.Icc 2 ( 2 * ( 2 * n ) ) ) fun ℓ => Finset.image ( fun s_lo => test_value n m c ℓ s_lo ) ( Finset.range ( 2 * ( 2 * n ) ) ) ) ; aesop;
  · exact False.elim <| h ⟨ ⟨ 2, by norm_num, by linarith ⟩, hn.ne' ⟩

/-
PROBLEM
Step function is nonneg.

PROVIDED SOLUTION
Unfold step_function. It's an if-then-else: either 0 (nonneg) or c_i/m where c_i : ℕ and m : ℕ with m > 0. Both cases are nonneg by positivity after unfolding and splitting all ifs.
-/
lemma step_function_nonneg (n m : ℕ) (hm : m > 0) (c : Fin (2 * n) → ℕ) :
    ∀ x, 0 ≤ step_function n m c x := by
  -- By definition of $step\_function$, it is nonnegative everywhere.
  intros x
  simp [step_function];
  split_ifs <;> positivity

/-
PROBLEM
The original statement used `Set.Ioo`, but that is false: at x = -1/4 the step function
   can be nonzero.  We weaken to `Set.Ico` (equivalently `Icc` would also work).

PROVIDED SOLUTION
Unfold step_function. If x ∈ support, then f(x) ≠ 0. The outer if gives 0 when x < -1/4 ∨ x ≥ 1/4. So if f(x) ≠ 0, we must have ¬(x < -1/4 ∨ x ≥ 1/4), which means x ≥ -1/4 and x < 1/4. This is exactly x ∈ Ico(-1/4, 1/4).
-/
lemma step_function_support (n m : ℕ) (c : Fin (2 * n) → ℕ) :
    Function.support (step_function n m c) ⊆ Set.Ico (-1/4 : ℝ) (1/4) := by
  intro x hx; unfold step_function at hx; aesop;

/-- Averaging principle: ‖g‖∞ ≥ (∫g) / vol(S) for nonneg g supported on S. -/
theorem averaging_principle (g : ℝ → ℝ) (hg : ∀ x, 0 ≤ g x)
    (hg_int : MeasureTheory.Integrable g MeasureTheory.volume)
    (S : Set ℝ) (hS : Function.support g ⊆ S)
    (v : ℝ) (hS_meas : MeasureTheory.volume S = ENNReal.ofReal v)
    (hv : 0 < v) :
    MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume ≥
      ENNReal.ofReal (MeasureTheory.integral MeasureTheory.volume g / v) := by
  have h_integral_restrict : ∫ x, g x ∂MeasureTheory.volume = ∫ x in S, g x ∂MeasureTheory.volume := by
    rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero fun x hx => by_contra fun hx' => hx <| hS <| by aesop ];
  have h_integral_bound : (∫⁻ x in S, ENNReal.ofReal (g x) ∂MeasureTheory.volume) ≤ (MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume) * (MeasureTheory.MeasureSpace.volume S) := by
    have h_integral_bound : ∀ᵐ x ∂MeasureTheory.Measure.restrict MeasureTheory.volume S, ENNReal.ofReal (g x) ≤ MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume := by
      have h_integral_bound : ∀ᵐ x ∂MeasureTheory.MeasureSpace.volume, ENNReal.ofReal (g x) ≤ MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume := by
        have h_integral_bound : ∀ᵐ x ∂MeasureTheory.MeasureSpace.volume, ‖g x‖ₑ ≤ essSup (fun x => ‖g x‖ₑ) MeasureTheory.MeasureSpace.volume := by
          exact ENNReal.ae_le_essSup _;
        filter_upwards [ h_integral_bound ] with x hx using le_trans ( by simp +decide [ Real.enorm_eq_ofReal ( hg x ) ] ) hx;
      exact MeasureTheory.ae_restrict_of_ae h_integral_bound;
    refine' le_trans ( MeasureTheory.lintegral_mono_ae h_integral_bound ) _ ; aesop;
  simp_all +decide [ ENNReal.ofReal_div_of_pos hv ];
  rw [ ENNReal.div_le_iff_le_mul ] <;> norm_num [ hv ];
  refine' le_trans _ h_integral_bound;
  rw [ MeasureTheory.ofReal_integral_eq_lintegral_ofReal ];
  · exact hg_int.integrableOn;
  · exact Filter.Eventually.of_forall hg

/-- Integral of convolution = (integral)². -/
theorem integral_convolution_square (f : ℝ → ℝ)
    (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) =
    (MeasureTheory.integral MeasureTheory.volume f) ^ 2 := by
  rw [ sq ];
  apply MeasureTheory.integral_convolution;
  · exact hf;
  · exact hf

/-! ## ============================================================
    HELPER LEMMAS FOR DECOMPOSED PROOF
    ============================================================ -/

/-
PROBLEM
The step function is integrable (bounded and compactly supported).

PROVIDED SOLUTION
The step function is bounded (by max_i c_i / m, which is finite) and supported on the bounded set [-1/4, 1/4). A bounded measurable function with support in a set of finite measure is integrable. Use MeasureTheory.Integrable.of_finite_support or show it's bounded by some constant on a set of finite measure. Specifically, step_function is bounded by some M (we can use (Finset.univ.sup' ⟨0, Finset.mem_univ _⟩ c : ℝ) / m + 1 as a bound) and supported on [-1/4, 1/4) which has finite Lebesgue measure 1/2. Use MeasureTheory.memLp_top_of_bound or MeasureTheory.Integrable.mono or show it's bounded and has support of finite measure.
-/
lemma step_function_integrable (n m : ℕ) (c : Fin (2 * n) → ℕ) :
    MeasureTheory.Integrable (step_function n m c) MeasureTheory.volume := by
  -- The step function is bounded, hence integrable.
  have h_bounded : ∃ C, ∀ x, abs (step_function n m c x) ≤ C := by
    use ( ∑ i : Fin ( 2 * n ), ( c i : ℝ ) ) / m;
    intro x; by_cases hx : x < -1 / 4 ∨ x ≥ 1 / 4 <;> simp_all +decide [ step_function ] ;
    · exact div_nonneg ( Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _ ) ( Nat.cast_nonneg _ );
    · split_ifs <;> norm_num at *;
      · exact div_nonneg ( Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _ ) ( Nat.cast_nonneg _ );
      · rw [ abs_of_nonneg ( by positivity ) ] ; exact div_le_div_of_nonneg_right ( mod_cast Finset.single_le_sum ( fun a _ => Nat.zero_le ( c a ) ) ( Finset.mem_univ _ ) ) ( Nat.cast_nonneg _ ) ;
      · exact div_nonneg ( Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _ ) ( Nat.cast_nonneg _ );
  refine' MeasureTheory.Integrable.mono' _ _ _;
  refine' fun x => h_bounded.choose * Set.indicator ( Set.Ico ( -1 / 4 ) ( 1 / 4 ) ) ( fun _ => 1 ) x;
  · exact MeasureTheory.Integrable.const_mul ( MeasureTheory.integrable_indicator_iff ( measurableSet_Ico ) |>.2 ( by norm_num ) ) _;
  · unfold step_function;
    refine' Measurable.aestronglyMeasurable _;
    refine' Measurable.ite ( measurableSet_Iio.union measurableSet_Ici ) measurable_const _;
    fun_prop;
  · filter_upwards [ ] with x ; by_cases hx : x ∈ Set.Ico ( -1 / 4 ) ( 1 / 4 ) <;> simp_all +decide [ Set.indicator ];
    · exact h_bounded.choose_spec x;
    · split_ifs <;> simp_all +decide [ abs_le ];
      · linarith;
      · exact ⟨ by unfold step_function; split_ifs <;> aesop, by unfold step_function; split_ifs <;> aesop ⟩

/-
PROBLEM
Integral of step function equals 1/(4n).
    Proof: ∫f = ∑_i (c_i/m) · Δ = (1/(4nm)) · ∑ c_i = (1/(4nm)) · m = 1/(4n).

PROVIDED SOLUTION
The step function f(x) = c_i/m on bin B_i = [-1/4 + i/(4n), -1/4 + (i+1)/(4n)) for i = 0,...,2n-1, and 0 outside [-1/4, 1/4).

We can write f = ∑_{i : Fin (2n)} (c_i/m) · indicator(B_i).

Then ∫f = ∑_i (c_i/m) · vol(B_i) = ∑_i (c_i/m) · 1/(4n) = (1/(4nm)) · ∑_i c_i = (1/(4nm)) · m = 1/(4n).

The key Mathlib tools are:
- MeasureTheory.integral_finset_sum for sum of integrals
- MeasureTheory.integral_indicator for indicator function integrals
- Or express f as a piecewise function and use MeasureTheory.integral_piecewise

Alternative approach: show that ∫f = ∫_{[-1/4, 1/4)} f by support containment, then decompose the integral over the bins and compute each one using intervalIntegral.integral_const or MeasureTheory.setIntegral_const.

Actually, the simplest approach may be: the step function is equal a.e. to ∑_i (c_i/m) · indicator(B_i), then use linearity of integral and integral_indicator_const/measure of intervals.
-/
lemma integral_step_function (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    ∫ x, step_function n m c x = 1 / (4 * (n : ℝ)) := by
  -- The step function is non-zero only on the interval $[-\frac{1}{4}, \frac{1}{4})$, so we can restrict the integral to this interval.
  have h_restrict : ∫ x, step_function n m c x = ∫ x in Set.Ico (-1 / 4 : ℝ) (1 / 4), step_function n m c x := by
    rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ] ; unfold step_function ; aesop;
  -- The step function is constant on each interval $[-1/4 + i/(4n), -1/4 + (i+1)/(4n))$ for $i = 0, \ldots, 2n-1$.
  have h_const : ∀ i : Fin (2 * n), ∫ x in Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x = (c i : ℝ) / m * (1 / (4 * n)) := by
    intro i
    have h_const_interval : ∀ x ∈ Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x = (c i : ℝ) / m := by
      unfold step_function;
      field_simp;
      intro x hx; split_ifs <;> simp_all +decide [ ne_of_gt, div_lt_iff₀, le_div_iff₀ ] ;
      · cases ‹_› <;> nlinarith [ show ( i : ℝ ) + 1 ≤ 2 * n by norm_cast; linarith [ Fin.is_lt i ], div_mul_cancel₀ ( -n + ( i : ℝ ) ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), div_mul_cancel₀ ( -n + ( i + 1 : ℝ ) ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] ;
      · rw [ mul_div_cancel₀ _ ( by positivity ) ] ; congr ; ring;
        rw [ div_le_iff₀ ( by positivity ), lt_div_iff₀ ( by positivity ) ] at hx ; norm_num [ show ⌊ ( n : ℝ ) + n * x * 4⌋ = i from Int.floor_eq_iff.mpr ⟨ by norm_num; linarith, by norm_num; linarith ⟩ ] at *;
      · rw [ Int.le_floor ] at * ; norm_num at * ; nlinarith [ ( by norm_cast : ( 1 :ℝ ) ≤ n ), mul_div_cancel₀ ( -n + ( i + 1 ) :ℝ ) ( by positivity : ( 4 * n :ℝ ) ≠ 0 ) ] ;
    rw [ MeasureTheory.setIntegral_congr_fun measurableSet_Ico h_const_interval ] ; ring ; norm_num [ hn.ne' ] ; ring;
  -- We can split the integral into a sum of integrals over each interval where the step function is constant.
  have h_split : ∫ x in Set.Ico (-1 / 4 : ℝ) (1 / 4), step_function n m c x = ∑ i : Fin (2 * n), ∫ x in Set.Ico (-1 / 4 + (i : ℝ) / (4 * n)) (-1 / 4 + (i + 1) / (4 * n)), step_function n m c x := by
    rw [ ← MeasureTheory.integral_biUnion_finset ];
    · congr with x ; norm_num [ Finset.mem_univ ];
      constructor <;> intro hx;
      · refine' ⟨ ⟨ ⌊ ( 1 / 4 + x ) * ( 4 * n ) ⌋₊, _ ⟩, _, _ ⟩ <;> norm_num;
        · rw [ Nat.floor_lt ] <;> norm_num <;> nlinarith [ show ( n : ℝ ) ≥ 1 by norm_cast ];
        · rw [ div_le_iff₀ ] <;> nlinarith [ Nat.floor_le ( show 0 ≤ ( 1 / 4 + x ) * ( 4 * n ) by nlinarith [ show ( n : ℝ ) ≥ 1 by norm_cast ] ), show ( n : ℝ ) ≥ 1 by norm_cast ];
        · rw [ lt_div_iff₀ ] <;> first | positivity | linarith [ Nat.lt_floor_add_one ( ( 1 / 4 + x ) * ( 4 * n ) ) ] ;
      · obtain ⟨ i, hi₁, hi₂ ⟩ := hx; constructor <;> nlinarith [ show ( i : ℝ ) + 1 ≤ 2 * n by norm_cast; linarith [ Fin.is_lt i ], div_mul_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), div_mul_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] ;
    · exact fun _ _ => measurableSet_Ico;
    · intros i hi j hj hij; exact Set.disjoint_left.mpr fun x hx₁ hx₂ => hij <| Fin.ext <| Nat.le_antisymm ( Nat.le_of_lt_succ <| by { rw [ ← @Nat.cast_lt ℝ ] ; push_cast; nlinarith [ hx₁.1, hx₁.2, hx₂.1, hx₂.2, show ( n : ℝ ) > 0 by positivity, mul_div_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] } ) ( Nat.le_of_lt_succ <| by { rw [ ← @Nat.cast_lt ℝ ] ; push_cast; nlinarith [ hx₁.1, hx₁.2, hx₂.1, hx₂.2, show ( n : ℝ ) > 0 by positivity, mul_div_cancel₀ ( ( i : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( i + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ), mul_div_cancel₀ ( ( j + 1 : ℝ ) : ℝ ) ( by positivity : ( 4 * n : ℝ ) ≠ 0 ) ] } ) ;
    · intro i hi; specialize h_const i; contrapose! h_const; rw [ MeasureTheory.integral_undef h_const ] ; ring; norm_num [ hn.ne', hm.ne' ] ;
      intro H; simp_all +decide [ Finset.sum_eq_zero_iff_of_nonneg ] ;
      exact h_const <| MeasureTheory.Integrable.integrableOn <| by exact?;
  simp_all +decide [ ← Finset.sum_mul _ _ _, ← Finset.sum_div ];
  rw [ ← Nat.cast_sum, hc, div_self ( by positivity ), one_mul ]

/-
PROBLEM
Discrete autoconvolution values are nonneg (since a_i ≥ 0).

PROVIDED SOLUTION
discrete_autoconvolution a k = ∑ i, ∑ j, if i+j=k then a_i * a_j else 0. Since a_i = 4n/m * c_i ≥ 0 (c_i : ℕ, so c_i ≥ 0, and 4n/m ≥ 0), each term a_i * a_j ≥ 0. Each summand (the if-then-else) is either 0 or a_i * a_j ≥ 0. So the whole sum is nonneg. Use Finset.sum_nonneg twice.
-/
lemma discrete_autoconvolution_nonneg (n m : ℕ) (c : Fin (2 * n) → ℕ) (k : ℕ) :
    0 ≤ discrete_autoconvolution (fun i : Fin (2 * n) => (4 * (n : ℝ)) / m * (c i : ℝ)) k := by
  exact Finset.sum_nonneg fun i hi => Finset.sum_nonneg fun j hj => by positivity;

/-- Key measure-theoretic fact: the L∞ norm of the step function convolution
    at any "grid point" index k is bounded below by the corresponding discrete
    autoconvolution value, scaled by Δ/m² = 1/(4nm²).

    Mathematically: (f⋆f)(y_k) = Δ/m² · conv_c[k-1] for the step function f,
    and ‖f⋆f‖_∞ ≥ (f⋆f)(y_k).

    The proof requires:
    1. Evaluating the Lebesgue convolution integral at grid point y_k = -1/2 + kΔ
    2. Showing essSup ≥ pointwise value (for piecewise-linear continuous functions)
    Both are standard analysis results but require significant Lean formalization.

    Sub-lemma 1: The convolution of the step function with itself, evaluated at
    grid point y_k, equals (1/(4nm²)) · conv_c[k] where conv_c is the discrete
    autoconvolution of the composition vector c.
    Grid point: y_k = -1/2 + (k+1)·Δ where Δ = 1/(4n). -/
lemma convolution_at_grid_point (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    MeasureTheory.convolution (step_function n m c) (step_function n m c)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
      (-1/2 + (↑k + 1) * (1 / (4 * ↑n))) =
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
  sorry

/-- Sub-lemma 2: For any nonneg integrable function g on ℝ, the L∞ norm
    (essential supremum) is ≥ g(x) for any x where g is continuous.
    For our step function convolution, f⋆f is continuous (it's piecewise linear),
    so the L∞ norm ≥ (f⋆f)(y) for any y. -/
lemma eLpNorm_top_ge_of_continuous_at (g : ℝ → ℝ)
    (hg_nn : ∀ x, 0 ≤ g x) (hg_int : MeasureTheory.Integrable g)
    (x₀ : ℝ) (hg_cont : ContinuousAt g x₀) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥ g x₀ := by
  sorry

lemma eLpNorm_conv_ge_discrete (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    (MeasureTheory.eLpNorm
      (MeasureTheory.convolution (step_function n m c) (step_function n m c)
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume).toReal ≥
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
  sorry

/-
PROBLEM
Combinatorial lemma: the windowed sum of discrete autoconvolution is
    bounded by the max discrete autoconv times (ℓ-1).

PROVIDED SOLUTION
PROOF SKETCH using eLpNorm_conv_ge_discrete and integral_step_function.

Let f = step_function n m c, L = ContinuousLinearMap.mul ℝ ℝ.

Step 1: Unfold autoconvolution_ratio.
autoconvolution_ratio f = ‖f⋆f‖_∞.toReal / (∫f)²
From integral_step_function: ∫f = 1/(4n). So (∫f)² = 1/(16n²).
Thus autoconvolution_ratio f = 16n² · ‖f⋆f‖_∞.toReal.

Step 2: From eLpNorm_conv_ge_discrete (applied to the maximizing k*):
‖f⋆f‖_∞.toReal ≥ (1/(4n·m²)) · max_k conv_c[k]
where conv_c[k] = discrete_autoconvolution (fun i => (c i : ℝ)) k.

So autoconvolution_ratio ≥ 16n² · (1/(4nm²)) · max_k conv_c[k] = 4n/m² · max_k conv_c[k].

Step 3: Relate conv_a to conv_c.
conv_a[k] = discrete_autoconvolution (fun i => 4n/m · c_i) k = (4n/m)² · conv_c[k].
So max conv_a = (4n/m)² · max conv_c = 16n²/m² · max conv_c.
Thus 4n/m² · max conv_c = max conv_a / (4n).

So autoconvolution_ratio ≥ max conv_a / (4n).

Step 4: Combinatorial bound.
test_value = (1/(4nℓ)) · ∑_{k∈window} conv_a[k].
The window Icc s_lo (s_lo+ℓ-2) has at most ℓ-1 nonneg terms.
Each conv_a[k] ≤ max conv_a.
So ∑ ≤ (ℓ-1) · max conv_a.
test_value ≤ ((ℓ-1)/(4nℓ)) · max conv_a ≤ max conv_a/(4n) (since (ℓ-1)/ℓ < 1).

Step 5: Combine.
test_value ≤ max conv_a / (4n) ≤ autoconvolution_ratio.

KEY: Use `eLpNorm_conv_ge_discrete` with the k that maximizes conv_c[k].
Use `integral_step_function` for (∫f)² = 1/(16n²).
The rest is algebra and Finset manipulations.
-/
lemma window_sum_le_max_times (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    test_value n m c ℓ s_lo ≤
      autoconvolution_ratio (step_function n m c) := by
  -- By definition of `test_value`, we can express it in terms of the discrete autoconvolution.
  have h_test_value : test_value n m c ℓ s_lo = (1 / (4 * n * ℓ : ℝ)) * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), (4 * n / m : ℝ) ^ 2 * discrete_autoconvolution (fun i => (c i : ℝ)) k := by
    unfold test_value;
    unfold discrete_autoconvolution; norm_num [ Finset.mul_sum _ _ _, mul_pow ] ; ring;
  -- Applying the lemma that bounds the sum of discrete autoconvolutions.
  have h_sum_bound : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ((4 * n / m : ℝ) ^ 2) * discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (ℓ - 1) * ((4 * n / m : ℝ) ^ 2) * (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
    have h_sum_bound : ∀ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
      intros k hk
      have h_discrete_conv : discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
        have := eLpNorm_conv_ge_discrete n m hn hm c hc k
        rw [ div_div, div_mul_eq_mul_div, ge_iff_le, div_le_iff₀ ] at this <;> first | positivity | linarith;
      exact h_discrete_conv;
    convert Finset.sum_le_sum fun k hk => mul_le_mul_of_nonneg_left ( h_sum_bound k hk ) ( sq_nonneg ( 4 * n / m : ℝ ) ) using 1 ; norm_num [ mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ] ; ring; (
    exact Or.inl <| Or.inl <| Or.inl <| by rw [ Nat.cast_sub <| by omega ] ; rw [ Nat.cast_add, Nat.cast_sub <| by omega ] ; push_cast ; ring;);
  -- Substitute the bound from `h_sum_bound` into `h_test_value`.
  rw [h_test_value]
  have h_subst : (1 / (4 * n * ℓ : ℝ)) * ((ℓ - 1) * ((4 * n / m : ℝ) ^ 2) * (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ)) ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal / (1 / (4 * n : ℝ)) ^ 2 := by
    field_simp;
    exact mul_le_mul_of_nonneg_right ( by linarith ) ( ENNReal.toReal_nonneg );
  convert le_trans _ h_subst using 1;
  · unfold autoconvolution_ratio;
    rw [ integral_step_function n m hn hm c hc ];
  · exact mul_le_mul_of_nonneg_left h_sum_bound <| by positivity;

/-! ## ============================================================
    THEOREM TO PROVE
    ============================================================ -/

/-
Claim 1.1: Test value ≤ autoconvolution ratio of step function.

PROOF SKETCH:
1. By max_test_value_le_max, the max is attained at some (ℓ, s_lo) with ℓ ≥ 2.
2. By test_value_le_sup_conv_div, test_value ℓ s_lo ≤ autoconvolution_ratio.
3. Combining: max_test_value = test_value ℓ s_lo ≤ autoconvolution_ratio.
-/
theorem test_value_le_Linfty (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (max_test_value n m c : ℝ) ≤ autoconvolution_ratio (step_function n m c) := by
  obtain ⟨ℓ, s_lo, hℓ_mem, _, h_eq⟩ := max_test_value_le_max n m hn c
  rw [h_eq]
  have hℓ : 2 ≤ ℓ := (Finset.mem_Icc.mp hℓ_mem).1
  exact window_sum_le_max_times n m hn hm c hc ℓ s_lo hℓ

end