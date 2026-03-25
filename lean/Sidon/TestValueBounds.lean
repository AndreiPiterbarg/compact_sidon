/-
Sidon Autocorrelation Project — Test Value Bounds

eLpNorm bounds at grid points, test value ≤ autoconvolution ratio,
and continuous test value ≤ ratio (Fubini set-integral chain).
-/

import Mathlib
import Sidon.Defs
import Sidon.Foundational
import Sidon.CauchySchwarz
import Sidon.AsymmetryBound
import Sidon.StepFunction

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
-- Test Value Bounds (Section 18b)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Sub-lemma: For any nonneg integrable function g on ℝ, the L∞ norm
    (essential supremum) is ≥ g(x) for any x where g is continuous. -/
lemma eLpNorm_top_ge_of_continuous_at (g : ℝ → ℝ)
    (hg_nn : ∀ x, 0 ≤ g x) (hg_int : MeasureTheory.Integrable g)
    (x₀ : ℝ) (hg_cont : ContinuousAt g x₀)
    (h_fin : MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume ≠ ⊤) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥ g x₀ := by
  by_contra h_lt; push_neg at h_lt
  set M := (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal with hM_def
  have h_fin' : MeasureTheory.eLpNormEssSup g MeasureTheory.volume ≠ ⊤ := by
    rwa [← MeasureTheory.eLpNorm_exponent_top]
  have h_ae_le : ∀ᵐ x ∂MeasureTheory.volume, g x ≤ M := by
    filter_upwards [MeasureTheory.enorm_ae_le_eLpNormEssSup g MeasureTheory.volume] with x hx
    rw [← ENNReal.toReal_ofReal (hg_nn x)]
    rw [hM_def, MeasureTheory.eLpNorm_exponent_top]
    exact ENNReal.toReal_mono h_fin' (le_trans (by rw [Real.enorm_eq_ofReal (hg_nn x)]) hx)
  have h_zero : MeasureTheory.volume {x | M < g x} = 0 := by
    have h_ae_neg : ∀ᵐ x ∂MeasureTheory.volume, ¬(M < g x) :=
      h_ae_le.mono fun x hx h => not_le.mpr h hx
    rw [MeasureTheory.ae_iff] at h_ae_neg
    convert h_ae_neg using 2
    ext x; simp [not_not]
  obtain ⟨δ, hδ_pos, hball⟩ := Metric.continuousAt_iff.mp hg_cont (g x₀ - M) (by linarith)
  have h_sub : Metric.ball x₀ δ ⊆ {x | M < g x} := by
    intro x hx; simp only [Set.mem_setOf_eq]
    have := hball hx; rw [Real.dist_eq] at this; linarith [(abs_lt.mp this).1]
  have h_pos : 0 < MeasureTheory.volume (Metric.ball x₀ δ) :=
    Metric.isOpen_ball.measure_pos MeasureTheory.volume ⟨x₀, Metric.mem_ball_self hδ_pos⟩
  exact absurd (le_antisymm (le_of_le_of_eq (MeasureTheory.measure_mono h_sub) h_zero) (zero_le _)) (ne_of_gt h_pos)

/-- The sum of all bin masses equals 1 (for normalized f). -/
lemma sum_bin_masses_eq_one (n : ℕ) (hn : n > 0) (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    ∑ i : Fin (2 * n), bin_masses f n i = 1 := by
  convert hf_int using 1;
  have h_sum_eq_integral : ∑ i : Fin (2 * n), MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico (-(1 / 4 : ℝ) + i.val * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i.val + 1) * (1 / (4 * n : ℝ)))) f) = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + 2 * n * (1 / (4 * n : ℝ))), f x := by
    have h_sum_eq_integral : ∑ i : Fin (2 * n), ∫ x in Set.Ico (-(1 / 4 : ℝ) + i.val * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i.val + 1) * (1 / (4 * n : ℝ))), f x = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + 2 * n * (1 / (4 * n : ℝ))), f x := by
      have h_sum_eq_integral : ∀ m : ℕ, ∑ i ∈ Finset.range m, ∫ x in Set.Ico (-(1 / 4 : ℝ) + i * (1 / (4 * n : ℝ))) (-(1 / 4 : ℝ) + (i + 1) * (1 / (4 * n : ℝ))), f x = ∫ x in Set.Ico (-(1 / 4 : ℝ)) (-(1 / 4 : ℝ) + m * (1 / (4 * n : ℝ))), f x := by
        intro m
        induction' m with m ih;
        · norm_num;
        · rw [ Finset.sum_range_succ, ih, Nat.cast_succ, add_mul, one_mul, ← MeasureTheory.setIntegral_union ] <;> norm_num;
          · rw [ Set.Ico_union_Ico_eq_Ico ] <;> ring <;> norm_num [ hn ];
          · exact MeasureTheory.Integrable.integrableOn ( MeasureTheory.integrable_of_integral_eq_one hf_int );
          · exact MeasureTheory.Integrable.integrableOn ( MeasureTheory.integrable_of_integral_eq_one hf_int );
      simpa [ Finset.sum_range ] using h_sum_eq_integral ( 2 * n );
    convert h_sum_eq_integral using 2;
    rw [ MeasureTheory.integral_indicator ( measurableSet_Ico ) ];
  convert h_sum_eq_integral using 1;
  rw [ MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero ] ; ring_nf ; norm_num [ hn.ne' ];
  exact fun x hx => Classical.not_not.1 fun hx' => by have := hf_supp hx'; exact this.2.not_le <| hx <| by linarith [ this.1 ] ;

/-- The max test value is attained at some window parameters. -/
lemma max_test_value_le_max (n m : ℕ) (hn : n > 0) (c : Fin (2 * n) → ℕ) :
    ∃ ℓ s_lo, ℓ ∈ Finset.Icc 2 (2 * (2 * n)) ∧ s_lo ∈ Finset.range (2 * (2 * n)) ∧
    max_test_value n m c = test_value n m c ℓ s_lo := by
  unfold max_test_value;
  simp +zetaDelta at *;
  split_ifs with h;
  · have := Finset.max'_mem ( Finset.biUnion ( Finset.Icc 2 ( 2 * ( 2 * n ) ) ) fun ℓ => Finset.image ( fun s_lo => test_value n m c ℓ s_lo ) ( Finset.range ( 2 * ( 2 * n ) ) ) ) ; aesop;
  · exact False.elim <| h ⟨ ⟨ 2, by norm_num, by linarith ⟩, hn.ne' ⟩

/-- The step function is ContinuousAt at every point not equal to a bin boundary. -/
private lemma step_function_continuousAt (n m : ℕ) (hn : n > 0)
    (c : Fin (2 * n) → ℕ) (x : ℝ)
    (hx : ∀ k : Fin (2 * n + 1), x ≠ -(1/4 : ℝ) + ↑k.val / (4 * ↑n)) :
    ContinuousAt (step_function n m c) x := by
  have h_n_pos : (0 : ℝ) < ↑n := Nat.cast_pos.mpr hn
  suffices h : ∀ᶠ y in nhds x, step_function n m c y = step_function n m c x from
    continuousAt_const.congr (h.mono fun _ hy => hy.symm)
  by_cases hx_lt : x < -(1/4 : ℝ)
  · exact Filter.eventually_of_mem (IsOpen.mem_nhds isOpen_Iio hx_lt) fun y hy => by
      show step_function n m c y = step_function n m c x
      have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inl hy
      have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inl hx_lt
      simp only [step_function, h1, h2, ↓reduceIte]
  · push_neg at hx_lt
    by_cases hx_ge : x ≥ (1/4 : ℝ)
    · have hx_gt : x > 1/4 := by
        rcases eq_or_lt_of_le hx_ge with h_eq | h_gt
        · exfalso
          have h_bnd := hx ⟨2 * n, by omega⟩
          apply h_bnd
          rw [← h_eq]; push_cast
          have : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.not_eq_zero_of_lt (Nat.zero_lt_of_lt hn))
          field_simp
        · exact h_gt
      exact Filter.eventually_of_mem (IsOpen.mem_nhds isOpen_Ioi hx_gt) fun y hy => by
        show step_function n m c y = step_function n m c x
        have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inr (le_of_lt hy)
        have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inr hx_ge
        simp only [step_function, h1, h2, ↓reduceIte]
    · push_neg at hx_ge
      have hx_lo : -(1/4 : ℝ) < x := by
        rcases eq_or_lt_of_le hx_lt with h_eq | h_lt
        · exfalso; have := hx ⟨0, by omega⟩; apply this; simp [← h_eq]
        · exact h_lt
      set δ := (1 : ℝ) / (4 * ↑n) with hδ_def
      have hδ_pos : (0 : ℝ) < δ := by rw [hδ_def]; positivity
      set α := (x + 1/4) / δ with hα_def
      have hα_pos : 0 < α := div_pos (by linarith) hδ_pos
      have h4n_ne : (4 * (↑n : ℝ)) ≠ 0 := by positivity
      have hα_ne_int : ∀ z : ℤ, α ≠ ↑z := by
        intro z hz
        have hx_eq : x = -(1/4 : ℝ) + (z : ℝ) / (4 * ↑n) := by
          rw [hα_def, hδ_def] at hz
          field_simp [h4n_ne] at hz ⊢
          linarith
        have hz_nn : (0 : ℤ) ≤ z := by
          by_contra h; push_neg at h
          have : (z : ℝ) < 0 := Int.cast_lt_zero.mpr h
          linarith [hx_eq]
        have hz_le : z ≤ 2 * ↑n := by
          by_contra h; push_neg at h
          have hα_lt : α < 2 * ↑n := by
            rw [hα_def, hδ_def, div_lt_iff₀ hδ_pos]
            have hδ_val : 2 * (↑n : ℝ) * δ = 1/2 := by rw [hδ_def]; field_simp; ring
            linarith
          have : (↑z : ℝ) > 2 * (↑n : ℝ) := by exact_mod_cast h
          linarith [hz]
        have h := hx ⟨z.toNat, by omega⟩
        apply h; rw [hx_eq]
        push_cast [Int.toNat_of_nonneg hz_nn]
        ring
      set z := ⌊α⌋
      have hz1 : (↑z : ℝ) < α := lt_of_le_of_ne (Int.floor_le α) (fun h => hα_ne_int z h.symm)
      have hz2 : α < ↑z + 1 := Int.lt_floor_add_one α
      filter_upwards [
        (isOpen_Ioo.preimage (by fun_prop : Continuous fun y => (y + 1/4) / δ)).mem_nhds
          (show (x + 1/4) / δ ∈ Set.Ioo (↑z : ℝ) (↑z + 1) from ⟨hz1, hz2⟩),
        isOpen_Ioo.mem_nhds (show x ∈ Set.Ioo (-(1/4 : ℝ)) (1/4) from ⟨hx_lo, hx_ge⟩)
      ] with y hy_floor hy_range
      have h_floor_eq : ⌊(y + 1/4) / δ⌋ = z := by
        apply le_antisymm
        · have := Int.floor_lt.mpr hy_floor.2; omega
        · exact Int.le_floor.mpr (le_of_lt hy_floor.1)
      have hy_cond : ¬(y < (-1:ℝ)/4 ∨ y ≥ 1/4) := by push_neg; constructor <;> linarith [hy_range.1, hy_range.2]
      have hx_cond : ¬(x < (-1:ℝ)/4 ∨ x ≥ 1/4) := by push_neg; constructor <;> linarith
      simp only [step_function, hy_cond, hx_cond, ↓reduceIte]
      simp only [show (1 : ℝ) / (4 * (↑n : ℝ)) = δ from rfl, h_floor_eq]
      rfl

lemma eLpNorm_conv_ge_discrete (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    (MeasureTheory.eLpNorm
      (MeasureTheory.convolution (step_function n m c) (step_function n m c)
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume).toReal ≥
    (1 / (4 * (n : ℝ)) / (m : ℝ)^2) *
      discrete_autoconvolution (fun i : Fin (2 * n) => (c i : ℝ)) k := by
  have h_grid := convolution_at_grid_point n m hn hm c hc k
  rw [← h_grid]
  set S := step_function n m c
  have h_S_int : MeasureTheory.Integrable S MeasureTheory.volume := step_function_integrable n m c
  have h_S_nn : ∀ x, 0 ≤ S x := step_function_nonneg n m hm c
  have h_conv_nn : ∀ x, 0 ≤ MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x :=
    convolution_nonneg h_S_nn h_S_nn
  have h_S_compact : HasCompactSupport S := by
    apply CompactIccSpace.isCompact_Icc.of_isClosed_subset isClosed_closure
    exact (closure_mono ((step_function_support n m c).trans Set.Ico_subset_Icc_self)).trans
      (by rw [closure_Icc])
  have h_conv_cont : Continuous (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) :=
    h_S_compact.continuous_convolution_right (ContinuousLinearMap.mul ℝ ℝ) h_S_int
  have h_S_le_one : ∀ x, S x ≤ 1 := by
    intro x; simp only [S, step_function]
    split_ifs with h1 h2
    · linarith
    · exact div_le_one_of_le₀ (by exact_mod_cast (hc ▸ Finset.single_le_sum (fun a _ => Nat.zero_le (c a)) (Finset.mem_univ _) : c _ ≤ m)) (by positivity)
    · linarith
  have h_bound : ∀ y, MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume y ≤ ∫ t, S t := by
    intro y
    simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
    apply MeasureTheory.integral_mono
    · exact (h_S_int.comp_sub_left y).bdd_mul' h_S_int.aestronglyMeasurable
        (MeasureTheory.ae_of_all _ (fun x => by
          rw [Real.norm_eq_abs, abs_of_nonneg (h_S_nn x)]
          exact h_S_le_one x))
    · exact h_S_int
    · exact MeasureTheory.ae_of_all _ (fun t =>
        calc S t * S (y - t) ≤ S t * 1 :=
              mul_le_mul_of_nonneg_left (h_S_le_one _) (h_S_nn t)
          _ = S t := mul_one _)
  have h_conv_int : MeasureTheory.Integrable (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) MeasureTheory.volume :=
    h_conv_cont.integrable_of_hasCompactSupport (h_S_compact.convolution h_S_compact)
  have h_memLp := MeasureTheory.memLp_top_of_bound
    h_conv_int.aestronglyMeasurable (∫ t, S t)
    (MeasureTheory.ae_of_all _ (fun y => by
      rw [Real.norm_eq_abs, abs_of_nonneg (h_conv_nn y)]
      exact h_bound y))
  have h_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    h_memLp.2.ne
  exact eLpNorm_top_ge_of_continuous_at
    (MeasureTheory.convolution S S (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
    h_conv_nn h_conv_int _ h_conv_cont.continuousAt h_fin

-- Helper: window sum bound implies test_value ≤ autoconvolution_ratio
lemma window_sum_le_max_times (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    test_value n m c ℓ s_lo ≤
      autoconvolution_ratio (step_function n m c) := by
  have h_test_value : test_value n m c ℓ s_lo = (1 / (4 * n * ℓ : ℝ)) * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), (4 * n / m : ℝ) ^ 2 * discrete_autoconvolution (fun i => (c i : ℝ)) k := by
    unfold test_value;
    unfold discrete_autoconvolution; norm_num [ Finset.mul_sum _ _ _, mul_pow ] ; ring;
  have h_sum_bound : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ((4 * n / m : ℝ) ^ 2) * discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (ℓ - 1) * ((4 * n / m : ℝ) ^ 2) * (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
    have h_sum_bound : ∀ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
      intros k hk
      have h_discrete_conv : discrete_autoconvolution (fun i => (c i : ℝ)) k ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ) := by
        have := eLpNorm_conv_ge_discrete n m hn hm c hc k
        rw [ div_div, div_mul_eq_mul_div, ge_iff_le, div_le_iff₀ ] at this <;> first | positivity | linarith;
      exact h_discrete_conv;
    convert Finset.sum_le_sum fun k hk => mul_le_mul_of_nonneg_left ( h_sum_bound k hk ) ( sq_nonneg ( 4 * n / m : ℝ ) ) using 1 ; norm_num [ mul_assoc, mul_comm, mul_left_comm, Finset.mul_sum _ _ _ ] ; ring; (
    exact Or.inl <| Or.inl <| Or.inl <| by rw [ Nat.cast_sub <| by omega ] ; rw [ Nat.cast_add, Nat.cast_sub <| by omega ] ; push_cast ; ring;);
  rw [h_test_value]
  have h_subst : (1 / (4 * n * ℓ : ℝ)) * ((ℓ - 1) * ((4 * n / m : ℝ) ^ 2) * (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal * (4 * n * m ^ 2 : ℝ)) ≤ (MeasureTheory.eLpNorm (MeasureTheory.convolution (step_function n m c) (step_function n m c) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal / (1 / (4 * n : ℝ)) ^ 2 := by
    field_simp;
    exact mul_le_mul_of_nonneg_right ( by linarith ) ( ENNReal.toReal_nonneg );
  convert le_trans _ h_subst using 1;
  · unfold autoconvolution_ratio;
    rw [ integral_step_function n m hn hm c hc ];
  · exact mul_le_mul_of_nonneg_left h_sum_bound <| by positivity;

/-- Claim 1.1: Test value ≤ ‖f*f‖∞ / (∫f)² for the step function. -/
theorem test_value_le_Linfty (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (max_test_value n m c : ℝ) ≤ autoconvolution_ratio (step_function n m c) := by
  obtain ⟨ℓ, s_lo, hℓ_mem, _, h_eq⟩ := max_test_value_le_max n m hn c
  rw [h_eq]
  have hℓ : 2 ≤ ℓ := (Finset.mem_Icc.mp hℓ_mem).1
  exact window_sum_le_max_times n m hn hm c hc ℓ s_lo hℓ

/-- Bins are disjoint: for any point t, sum_i f_bin(f,n,i)(t) <= f(t). -/
private lemma sum_f_bin_le (n : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (t : ℝ) :
    ∑ i : Fin (2 * n), f_bin f n i t ≤ f t := by
  simp only [f_bin]
  by_cases ht : ∃ i : Fin (2 * n), t ∈ bin_interval n i
  · obtain ⟨i₀, hi₀⟩ := ht
    have h_eq : ∀ i : Fin (2 * n), Set.indicator (bin_interval n i) f t =
        if i = i₀ then f t else 0 := by
      intro i; simp only [Set.indicator_apply]
      split_ifs with h1 h2
      · rfl
      · exfalso; apply h2; simp only [bin_interval] at h1 hi₀
        have hδ : (0 : ℝ) < 1 / (4 * (n : ℝ)) := by
          have : 0 < n := by linarith [i₀.2]
          positivity
        have h1l := (Set.mem_Ico.mp h1).1; have h1r := (Set.mem_Ico.mp h1).2
        have h0l := (Set.mem_Ico.mp hi₀).1; have h0r := (Set.mem_Ico.mp hi₀).2
        have : i.1 = i₀.1 := by
          by_contra h_ne; rcases Nat.lt_or_gt_of_ne h_ne with h | h
          · linarith [mul_le_mul_of_nonneg_right (show (↑i + 1 : ℝ) ≤ ↑↑i₀ from by exact_mod_cast h) (le_of_lt hδ)]
          · linarith [mul_le_mul_of_nonneg_right (show (↑↑i₀ + 1 : ℝ) ≤ ↑↑i from by exact_mod_cast h) (le_of_lt hδ)]
        exact Fin.ext this
      · simp_all
      · rfl
    simp_rw [h_eq]; simp
  · have : ∀ i : Fin (2 * n), Set.indicator (bin_interval n i) f t = 0 := fun i => by
      simp only [Set.indicator_apply]
      split_ifs with h
      · exact absurd ⟨i, h⟩ ht
      · rfl
    simp only [this, Finset.sum_const_zero]; exact hf_nonneg t

/-- **Theorem**: Continuous test value lower bound.
    R(f) >= TV_continuous for admissible f, ell >= 2. -/
theorem continuous_test_value_le_ratio (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    autoconvolution_ratio f ≥ test_value_continuous n f ℓ s_lo := by
  set μ := bin_masses f n
  set conv_ff := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  set N := (MeasureTheory.eLpNorm conv_ff ⊤ MeasureTheory.volume).toReal
  have hf_int' := MeasureTheory.integrable_of_integral_eq_one hf_int
  have hμ_nn : ∀ i, 0 ≤ μ i := fun i => bin_masses_nonneg f hf_nonneg n i
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ℓ := by exact_mod_cast Nat.lt_of_lt_of_le (by norm_num) hℓ
  have hR : autoconvolution_ratio f = N := by
    unfold autoconvolution_ratio; dsimp only []
    rw [hf_int, one_pow, div_one]
  rw [ge_iff_le, hR]; unfold test_value_continuous; simp only [discrete_autoconvolution]
  have h_fac : ∀ k, (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then (4 * ↑n * μ i) * (4 * ↑n * μ j) else 0) =
    (4 * ↑n) ^ 2 * (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then μ i * μ j else 0) := by
    intro k; rw [Finset.mul_sum]; congr 1; ext i; rw [Finset.mul_sum]; congr 1; ext j
    split_ifs <;> ring
  have h_fac' : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then ((4 : ℝ) * ↑n * μ i) * ((4 : ℝ) * ↑n * μ j) else (0 : ℝ)) =
      ((4 : ℝ) * ↑n) ^ 2 * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then μ i * μ j else 0 := by
    rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k
  rw [h_fac']
  set ws := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
    ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then μ i * μ j else 0
  have h_simp : (1 / (4 * ↑n * ↑ℓ)) * ((4 * ↑n) ^ 2 * ws) = (4 * ↑n / ↑ℓ) * ws := by
    field_simp
  rw [h_simp]
  have h_ws_le : ws ≤ 1 := by
    calc ws ≤ ∑ k ∈ Finset.range (2 * (2 * n)),
          ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            (if i.1 + j.1 = k then μ i * μ j else 0) :=
          Finset.sum_le_sum_of_subset_of_nonneg (fun k hk => by simp [Finset.mem_range]; omega)
            (fun k _ _ => Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => by
              split_ifs <;> [exact mul_nonneg (hμ_nn i) (hμ_nn j); exact le_refl 0])
      _ = (∑ i : Fin (2 * n), μ i) ^ 2 := by
          rw [sq, ← Finset.sum_product']; rw [Finset.sum_comm]; congr 1; ext ⟨i, j⟩
          simp only [Finset.sum_ite_eq', Finset.mem_range]
          split_ifs with h; · rfl; · push_neg at h; omega
      _ = 1 := by rw [hμ_sum]; ring
  have hws_nn : 0 ≤ ws := Finset.sum_nonneg fun k _ => Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => by
      split_ifs <;> [exact mul_nonneg (hμ_nn i) (hμ_nn j); exact le_refl 0]
  have hN_nn : 0 ≤ N := ENNReal.toReal_nonneg
  suffices h_key : ws ≤ N * (↑ℓ / (4 * ↑n)) by
    calc (4 * ↑n / ↑ℓ) * ws ≤ (4 * ↑n / ↑ℓ) * (N * (↑ℓ / (4 * ↑n))) :=
          mul_le_mul_of_nonneg_left h_key (div_nonneg (by positivity) (by positivity))
      _ = N := by field_simp
  by_cases h_easy : 1 ≤ N * (↑ℓ / (4 * ↑n))
  · linarith
  · push_neg at h_easy
    set δ := (1 : ℝ) / (4 * n)
    set Z := Set.Ico (-(1/2 : ℝ) + s_lo * δ) (-(1/2 : ℝ) + (↑s_lo + ↑ℓ) * δ) with hZ_def
    have hδ_pos : 0 < δ := by positivity
    have h_supp : ∀ (i j : Fin (2 * n)),
        i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2) → ∀ z, z ∉ Z →
        MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z = 0 := by
      intro i j hij z hz
      simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
      apply MeasureTheory.integral_eq_zero_of_ae
      filter_upwards [] with t
      simp only [f_bin, Set.indicator_apply, bin_interval]
      split_ifs with h1 h2
      · exfalso; apply hz; simp only [Z, hZ_def, Set.mem_Ico, δ]
        have := (Set.mem_Ico.mp h1).1; have := (Set.mem_Ico.mp h1).2
        have := (Set.mem_Ico.mp h2).1; have := (Set.mem_Ico.mp h2).2
        have := (Finset.mem_Icc.mp hij).1; have := (Finset.mem_Icc.mp hij).2
        constructor <;> nlinarith
      · ring
      · ring
      · ring
    have h_pw : ∀ z, (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then
            MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
              (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
          else 0) ≤ conv_ff z := by
      intro z
      simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
      calc ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
            ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
              if i.1 + j.1 = k then ∫ t, f_bin f n i t * f_bin f n j (z - t) else 0
          ≤ ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
              ∫ t, f_bin f n i t * f_bin f n j (z - t) := by
            rw [show ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
              (if i.1 + j.1 = k then ∫ t, f_bin f n i t * f_bin f n j (z - t) else 0) =
              ∑ i : Fin (2 * n), ∑ j : Fin (2 * n), ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
              (if i.1 + j.1 = k then ∫ t, f_bin f n i t * f_bin f n j (z - t) else 0) from by
                rw [Finset.sum_comm]; congr 1; ext i; rw [Finset.sum_comm]]
            apply Finset.sum_le_sum; intro i _; apply Finset.sum_le_sum; intro j _
            have hnn := MeasureTheory.integral_nonneg fun t =>
              mul_nonneg (f_bin_nonneg f hf_nonneg n i t) (f_bin_nonneg f hf_nonneg n j (z - t))
            by_cases h_in : i.1 + j.1 ∈ Finset.Icc s_lo (s_lo + ℓ - 2)
            · simp [Finset.sum_eq_single_of_mem _ h_in (fun k _ hk => by simp [hk])]
            · simp [Finset.sum_eq_zero (fun k hk => by split_ifs with h; · exact absurd (h ▸ hk) h_in; · rfl)]; exact hnn
        _ ≤ ∫ t, f t * f (z - t) := by
            rw [show ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
                ∫ t, f_bin f n i t * f_bin f n j (z - t) =
              ∫ t, ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
                f_bin f n i t * f_bin f n j (z - t) from by
                rw [← MeasureTheory.integral_finset_sum]; congr 1; ext t
                · simp [Finset.sum_comm (f := fun i j => f_bin f n i t * f_bin f n j (z - t))]
                · intro i _; apply MeasureTheory.Integrable.sum; intro j _
                  exact (f_bin_integrable f hf_int' n i).bdd_mul'
                    (f_bin_integrable f hf_int' n j |>.comp_sub_left z).aestronglyMeasurable
                    (MeasureTheory.ae_of_all _ fun x => by
                      rw [Real.norm_eq_abs, abs_of_nonneg (f_bin_nonneg f hf_nonneg n j (z - x))])]
            apply MeasureTheory.integral_mono
            · apply MeasureTheory.Integrable.sum; intro i _
              apply MeasureTheory.Integrable.sum; intro j _
              exact (f_bin_integrable f hf_int' n i).bdd_mul'
                (f_bin_integrable f hf_int' n j |>.comp_sub_left z).aestronglyMeasurable
                (MeasureTheory.ae_of_all _ fun x => by
                  rw [Real.norm_eq_abs, abs_of_nonneg (f_bin_nonneg f hf_nonneg n j (z - x))])
            · exact (hf_int'.comp_sub_left z).bdd_mul' hf_int'.aestronglyMeasurable
                (MeasureTheory.ae_of_all _ fun x => by rw [Real.norm_eq_abs, abs_of_nonneg (hf_nonneg x)])
            · filter_upwards [] with t
              calc ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
                    f_bin f n i t * f_bin f n j (z - t)
                  = (∑ i, f_bin f n i t) * (∑ j, f_bin f n j (z - t)) := Finset.sum_mul_sum.symm
                _ ≤ f t * f (z - t) := mul_le_mul (sum_f_bin_le n f hf_nonneg t)
                    (sum_f_bin_le n f hf_nonneg (z - t))
                    (Finset.sum_nonneg fun j _ => f_bin_nonneg f hf_nonneg n j (z - t))
                    (hf_nonneg t)
    have hconv_nn : ∀ x, 0 ≤ conv_ff x := convolution_nonneg hf_nonneg hf_nonneg
    have hf_compact := f_has_compact_support f hf_supp
    have hconv_cont : Continuous conv_ff :=
      hf_compact.continuous_convolution_right (ContinuousLinearMap.mul ℝ ℝ) hf_int'
    have hconv_int : MeasureTheory.Integrable conv_ff MeasureTheory.volume :=
      hconv_cont.integrable_of_hasCompactSupport (hf_compact.convolution hf_compact)
    have hconv_fin : MeasureTheory.eLpNorm conv_ff ⊤ MeasureTheory.volume ≠ ⊤ := by
      exact (MeasureTheory.memLp_top_of_bound hconv_int.aestronglyMeasurable (∫ t, f t)
        (MeasureTheory.ae_of_all _ fun y => by
          rw [Real.norm_eq_abs, abs_of_nonneg (hconv_nn y)]
          exact MeasureTheory.integral_mono
            ((hf_int'.comp_sub_left y).bdd_mul' hf_int'.aestronglyMeasurable
              (MeasureTheory.ae_of_all _ fun x => by rw [Real.norm_eq_abs, abs_of_nonneg (hf_nonneg x)]))
            hf_int'
            (MeasureTheory.ae_of_all _ fun t => by nlinarith [hf_nonneg t, hf_nonneg (y - t)]))).2.ne
    have h_pw_N : ∀ z, conv_ff z ≤ N := by
      intro z
      exact eLpNorm_top_ge_of_continuous_at conv_ff hconv_nn hconv_int z hconv_cont.continuousAt hconv_fin
    have h_intZ : ∫ z in Z, conv_ff z ≤ N * (↑ℓ * δ) := by
      calc ∫ z in Z, conv_ff z ≤ ∫ z in Z, N := by
            apply MeasureTheory.setIntegral_mono hconv_int.integrableOn
              (MeasureTheory.integrableOn_const.mpr (Or.inr (by
                simp [Z, hZ_def, Real.volume_Ico]; exact ENNReal.ofReal_lt_top)))
            exact fun z _ => h_pw_N z
        _ = N * (↑ℓ * δ) := by
            rw [MeasureTheory.setIntegral_const]
            simp only [Z, hZ_def, Real.volume_Ico, δ]
            rw [ENNReal.toReal_ofReal (by nlinarith)]
            ring
    have h_cross_int : ∀ (i j : Fin (2 * n)),
        ∫ x, MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x = μ i * μ j := by
      intro i j
      rw [MeasureTheory.integral_convolution (ContinuousLinearMap.mul ℝ ℝ)
        (f_bin_integrable f hf_int' n i) (f_bin_integrable f hf_int' n j)]
      simp [ContinuousLinearMap.mul_apply', integral_f_bin]
    set g : ℝ → ℝ := fun z => ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then
          MeasureTheory.convolution (f_bin f n i) (f_bin f n j)
            (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume z
        else 0
    have hg_vanish : ∀ z, z ∉ Z → g z = 0 := by
      intro z hz; simp only [g]
      apply Finset.sum_eq_zero; intro k hk
      apply Finset.sum_eq_zero; intro i _
      apply Finset.sum_eq_zero; intro j _
      split_ifs with h
      · rw [h] at hk; exact h_supp i j hk z hz
      · rfl
    have hg_le : ∀ z, g z ≤ conv_ff z := h_pw
    have hg_nn : ∀ z, 0 ≤ g z := by
      intro z; simp only [g]
      apply Finset.sum_nonneg; intro k _; apply Finset.sum_nonneg; intro i _
      apply Finset.sum_nonneg; intro j _; split_ifs
      · exact MeasureTheory.integral_nonneg fun t =>
          mul_nonneg (f_bin_nonneg f hf_nonneg n i t) (f_bin_nonneg f hf_nonneg n j (z - t))
      · exact le_refl 0
    have hg_int : MeasureTheory.Integrable g MeasureTheory.volume := by
      apply MeasureTheory.Integrable.mono' hconv_int
      · apply Finset.aestronglyMeasurable_sum; intro k _
        apply Finset.aestronglyMeasurable_sum; intro i _
        apply Finset.aestronglyMeasurable_sum; intro j _
        split_ifs
        · exact (MeasureTheory.Integrable.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ)
            (f_bin_integrable f hf_int' n i) (f_bin_integrable f hf_int' n j)).aestronglyMeasurable
        · exact MeasureTheory.aestronglyMeasurable_zero
      · filter_upwards [] with z
        rw [Real.norm_eq_abs, abs_of_nonneg (hg_nn z), Real.norm_eq_abs, abs_of_nonneg (hconv_nn z)]
        exact hg_le z
    have hg_integral : ∫ z, g z = ws := by
      simp only [g, ws]
      rw [MeasureTheory.integral_finset_sum]; congr 1; ext k
      rw [MeasureTheory.integral_finset_sum]; congr 1; ext i
      rw [MeasureTheory.integral_finset_sum]; congr 1; ext j
      split_ifs with h
      · rfl
      · exact MeasureTheory.integral_zero
      all_goals (intro j _; split_ifs
                 · exact MeasureTheory.Integrable.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ)
                     (f_bin_integrable f hf_int' n i) (f_bin_integrable f hf_int' n j)
                 · exact MeasureTheory.integrable_zero _ _ _)
      all_goals (intro i _; apply MeasureTheory.Integrable.sum; intro j _
                 split_ifs
                 · exact MeasureTheory.Integrable.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ)
                     (f_bin_integrable f hf_int' n i) (f_bin_integrable f hf_int' n j)
                 · exact MeasureTheory.integrable_zero _ _ _)
      all_goals (intro k _; apply MeasureTheory.Integrable.sum; intro i _
                 apply MeasureTheory.Integrable.sum; intro j _
                 split_ifs
                 · exact MeasureTheory.Integrable.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ)
                     (f_bin_integrable f hf_int' n i) (f_bin_integrable f hf_int' n j)
                 · exact MeasureTheory.integrable_zero _ _ _)
    have hg_setint : ∫ z, g z = ∫ z in Z, g z := by
      rw [MeasureTheory.setIntegral_eq_integral_of_forall_compl_eq_zero (fun z hz => hg_vanish z hz)]
    have hg_Z_le : ∫ z in Z, g z ≤ ∫ z in Z, conv_ff z := by
      apply MeasureTheory.setIntegral_mono hg_int.integrableOn hconv_int.integrableOn
      exact fun z _ => hg_le z
    calc ws = ∫ z, g z := hg_integral.symm
      _ = ∫ z in Z, g z := hg_setint
      _ ≤ ∫ z in Z, conv_ff z := hg_Z_le
      _ ≤ N * (↑ℓ * δ) := h_intZ
      _ = N * (↑ℓ / (4 * ↑n)) := by simp only [δ]; ring

end -- noncomputable section
