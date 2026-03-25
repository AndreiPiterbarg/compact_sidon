import Mathlib
import Sidon.Defs

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
-- Foundational Lemmas (F1–F15)
-- Source: output (1).lean (UUID: ca2199a4)
-- ═══════════════════════════════════════════════════════════════════════════════

-- F1: c_i = D(i+1) - D(i) rewrite
theorem canonical_discretization_eq (f : ℝ → ℝ) (n m : ℕ) (i : Fin (2 * n)) :
    canonical_discretization f n m i =
    if i.1 + 1 < 2 * n then
      canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1
    else
      m - canonical_cumulative_distribution f n m i.1 := by
        unfold canonical_discretization canonical_cumulative_distribution;
        simp +zetaDelta at *

-- F2: D(0) = 0
theorem canonical_cumulative_distribution_zero (f : ℝ → ℝ) (n m : ℕ) :
    canonical_cumulative_distribution f n m 0 = 0 := by
      unfold canonical_cumulative_distribution; aesop;

-- F3: D(2n) = m
theorem canonical_cumulative_distribution_2n (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0) :
    canonical_cumulative_distribution f n m (2 * n) = m := by
      unfold canonical_cumulative_distribution; aesop;

-- F4: Bin masses ≥ 0 for f ≥ 0
theorem bin_masses_nonneg (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (n : ℕ) (i : Fin (2 * n)) :
    0 ≤ bin_masses f n i := by
      apply_rules [ MeasureTheory.integral_nonneg, Set.indicator_nonneg ] ; aesop

-- F5: ∑ c_i = m (zero mass edge case)
theorem canonical_discretization_sum_zero_mass (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_zero : ∑ j : Fin (2 * n), bin_masses f n j = 0) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i = m := by
      rw [ Finset.sum_eq_single ⟨ 2 * n - 1, Nat.sub_lt ( by positivity ) ( by positivity ) ⟩ ] <;> norm_num [ canonical_discretization ];
      · rw [ Nat.sub_add_cancel ( by linarith ) ] ; aesop;
      · simp_all +decide [ Finset.sum_eq_zero_iff_of_nonneg, bin_masses_nonneg ];
        exact fun i hi₁ hi₂ => False.elim <| hi₁ <| Fin.ext <| by linarith [ Fin.is_lt i, Nat.sub_add_cancel <| show 1 ≤ 2 * n from by linarith ] ;

-- F6: c_i = D(i+1) - D(i) (alt hypothesis, given D(2n) = m)
theorem canonical_discretization_eq_diff (f : ℝ → ℝ) (n m : ℕ)
    (h_D_2n : canonical_cumulative_distribution f n m (2 * n) = m) (i : Fin (2 * n)) :
    canonical_discretization f n m i = canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1 := by
      convert canonical_discretization_eq f n m i using 1;
      grind

-- F7: Telescoping sum (AddCommGroup)
theorem sum_fin_telescope {M : Type*} [AddCommGroup M] (f : ℕ → M) (n : ℕ) :
    ∑ i : Fin n, (f (i + 1) - f i) = f n - f 0 := by
      convert Finset.sum_range_sub ( fun i => f i ) n using 1;
      rw [ Finset.sum_range ]

-- F8: D is monotone for f ≥ 0
theorem canonical_cumulative_distribution_mono (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (n m : ℕ) :
    Monotone (canonical_cumulative_distribution f n m) := by
      have h_floor_nonneg : ∀ k l : ℕ, k ≤ l → ⌊(∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) / (∑ j : Fin (2 * n), bin_masses f n j) * m⌋ ≤ ⌊(∑ j : Fin (2 * n), if j.1 < l then bin_masses f n j else 0) / (∑ j : Fin (2 * n), bin_masses f n j) * m⌋ := by
        intros k l hkl
        have h_sum_le : (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) ≤ (∑ j : Fin (2 * n), if j.1 < l then bin_masses f n j else 0) := by
          exact Finset.sum_le_sum fun i _ => by split_ifs <;> linarith [ show 0 ≤ bin_masses f n i from by exact MeasureTheory.integral_nonneg fun x => by exact Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _ ] ;
        gcongr;
        exact Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _;
      intro k l hkl; specialize h_floor_nonneg k l hkl; simp_all +decide [ canonical_cumulative_distribution ] ;
      rw [ ← Int.ofNat_le, Int.natAbs_of_nonneg ( Int.floor_nonneg.mpr _ ), Int.natAbs_of_nonneg ( Int.floor_nonneg.mpr _ ) ];
      · convert h_floor_nonneg using 1;
      · exact mul_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => by split_ifs <;> [ exact bin_masses_nonneg f hf_nonneg n _ ; norm_num ] ) ( Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _ ) ) ( Nat.cast_nonneg _ );
      · exact mul_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => by split_ifs <;> [ exact bin_masses_nonneg f hf_nonneg n _ ; exact le_rfl ] ) ( Finset.sum_nonneg fun _ _ => bin_masses_nonneg f hf_nonneg n _ ) ) ( Nat.cast_nonneg _ )

-- F9: ∑ c_i = telescope form
theorem canonical_discretization_sum_eq_telescope (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i =
    ∑ i : Fin (2 * n), (canonical_cumulative_distribution f n m (i + 1) - canonical_cumulative_distribution f n m i) := by
      convert Finset.sum_congr rfl fun i _ => canonical_discretization_eq_diff f n m (canonical_cumulative_distribution_2n f n m hn hm h_mass_pos) i

-- Nat telescoping sum (fixed from exact? gap)
theorem sum_fin_telescope_nat (f : ℕ → ℕ) (n : ℕ) (h_mono : Monotone f) :
    ∑ i : Fin n, (f (i + 1) - f i) = f n - f 0 := by
      have h_telescope : ∀ (n : ℕ), ∑ i ∈ Finset.range n, (f (i + 1) - f i) = f n - f 0 := by
        intro n
        induction n with
        | zero => simp
        | succ k ih =>
          rw [Finset.sum_range_succ, ih]
          have h1 : f 0 ≤ f k := h_mono (Nat.zero_le k)
          have h2 : f k ≤ f (k + 1) := h_mono (Nat.le_succ k)
          omega
      rw [ ← h_telescope, Finset.sum_range ]

-- F15: ∑ c_i = m (full proof, positive mass)
theorem canonical_discretization_sum_eq_m (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0)
    (hf_nonneg : ∀ x, 0 ≤ f x) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i = m := by
      rw [canonical_discretization_sum_eq_telescope f n m hn hm h_mass_pos]
      rw [sum_fin_telescope_nat (canonical_cumulative_distribution f n m) (2 * n) (canonical_cumulative_distribution_mono f hf_nonneg n m)]
      rw [canonical_cumulative_distribution_2n f n m hn hm h_mass_pos]
      rw [canonical_cumulative_distribution_zero]
      simp

-- F10: ∫ f_i = bin_mass_i
theorem f_restricted_integral (n : ℕ) (f : ℝ → ℝ) (i : Fin (2 * n)) :
    MeasureTheory.integral MeasureTheory.volume (f_restricted f n i) = bin_masses f n i := by
  unfold f_restricted bin_masses
  rfl

-- F11: f ≥ f_i ≥ 0
theorem f_ge_f_restricted (n : ℕ) (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) (i : Fin (2 * n)) :
    ∀ x, f x ≥ f_restricted f n i x ∧ f_restricted f n i x ≥ 0 := by
      intros x
      simp [f_restricted];
      simp [Set.indicator] at *; aesop;

-- F12: Convolution commutativity
theorem convolution_comm_real (f g : ℝ → ℝ) :
    MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
    MeasureTheory.convolution g f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := by
      ext x;
      have h_subst : ∀ {f g : ℝ → ℝ}, (∫ t, f t * g (x - t)) = (∫ t, g t * f (x - t)) := by
        intro f g; rw [ ← MeasureTheory.integral_sub_left_eq_self ] ; congr; ext; ring;
      exact h_subst

-- F13: supp(f) ⊆ (-1/4, 1/4) ⟹ compact support
theorem f_has_compact_support (f : ℝ → ℝ) (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4)) :
    HasCompactSupport f := by
      have h_closure : closure (Function.support f) ⊆ Set.Icc (-1 / 4 : ℝ) (1 / 4) := by
        have h_closure_subset : closure (Function.support f) ⊆ closure (Set.Ioo (-1 / 4 : ℝ) (1 / 4)) := by
          apply closure_mono hf_supp;
        exact h_closure_subset.trans ( by rw [ closure_Ioo ( by norm_num ) ] );
      exact CompactIccSpace.isCompact_Icc.of_isClosed_subset ( isClosed_closure ) h_closure

-- F14: f integrable ⟹ f_i integrable
theorem f_restricted_integrable (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n))
    (h_int : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.Integrable (f_restricted f n i) MeasureTheory.volume := by
  unfold f_restricted
  apply MeasureTheory.Integrable.indicator h_int measurableSet_Ico

end -- noncomputable section
