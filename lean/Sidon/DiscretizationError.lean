/-
Sidon Autocorrelation Project — Discretization Error and Correction Terms

Discretization error bound, contributing bins characterization,
correction term bound, and dynamic threshold soundness.
(Claims 1.2, 1.3, 1.4)
-/

import Mathlib
import Sidon.Defs
import Sidon.Foundational
import Sidon.StepFunction
import Sidon.TestValueBounds

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
-- Discretization Error and Correction Terms (Section 18c)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Each bin's discretization error is at most 1/m. -/
lemma discretization_error_bound (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    ∀ i : Fin (2 * n), |(canonical_discretization f n m i : ℝ) / m - bin_masses f n i| ≤ 1 / m := by
  intro i;
  have h_diff : ∀ k : ℕ, k ≤ 2 * n → |(⌊(∑ j ∈ Finset.univ.filter (fun j => j.val < k), (bin_masses f n j)) / (∑ j ∈ Finset.univ, (bin_masses f n j)) * m⌋ : ℝ) - (∑ j ∈ Finset.univ.filter (fun j => j.val < k), (bin_masses f n j)) / (∑ j ∈ Finset.univ, (bin_masses f n j)) * m| ≤ 1 := by
    exact fun k hk => abs_sub_le_iff.mpr ⟨ by linarith [ Int.floor_le ( ( ( ∑ j ∈ Finset.univ.filter fun j : Fin ( 2 * n ) => ( j : ℕ ) < k, bin_masses f n j ) / ∑ j : Fin ( 2 * n ), bin_masses f n j ) * m ) ], by linarith [ Int.lt_floor_add_one ( ( ( ∑ j ∈ Finset.univ.filter fun j : Fin ( 2 * n ) => ( j : ℕ ) < k, bin_masses f n j ) / ∑ j : Fin ( 2 * n ), bin_masses f n j ) * m ) ] ⟩;
  have h_ci : (canonical_discretization f n m i : ℝ) = (⌊(∑ j ∈ Finset.univ.filter (fun j => j.val < i.val + 1), (bin_masses f n j)) / (∑ j ∈ Finset.univ, (bin_masses f n j)) * m⌋ : ℝ) - (⌊(∑ j ∈ Finset.univ.filter (fun j => j.val < i.val), (bin_masses f n j)) / (∑ j ∈ Finset.univ, (bin_masses f n j)) * m⌋ : ℝ) := by
    unfold canonical_discretization; norm_num [ Finset.sum_ite ] ;
    split_ifs <;> simp_all +decide [ Finset.sum_ite, Nat.lt_succ_iff ];
    · rw [ Nat.cast_sub ] <;> norm_num [ abs_of_nonneg, Int.floor_nonneg ];
      · rw [ abs_of_nonneg, abs_of_nonneg ] <;> norm_cast;
        · refine' Int.floor_nonneg.mpr _;
          refine' mul_nonneg ( div_nonneg _ _ ) ( Nat.cast_nonneg _ );
          · refine' Finset.sum_nonneg fun _ _ => _;
            exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _;
          · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
            rw [ Set.indicator_apply ] ; aesop;
        · exact Int.floor_nonneg.mpr ( mul_nonneg ( div_nonneg ( Finset.sum_nonneg fun _ _ => by
            exact MeasureTheory.integral_nonneg fun x => by unfold Set.indicator; aesop; ) ( Finset.sum_nonneg fun _ _ => by
            exact MeasureTheory.integral_nonneg fun x => by unfold Set.indicator; aesop; ) ) ( Nat.cast_nonneg _ ) );
      · rw [ show ( Finset.filter ( fun x : Fin ( 2 * n ) => x ≤ i ) Finset.univ : Finset ( Fin ( 2 * n ) ) ) = Finset.filter ( fun x : Fin ( 2 * n ) => x < i ) Finset.univ ∪ { i } from ?_, Finset.sum_union ] <;> norm_num [ Finset.sum_singleton, Finset.sum_union, Finset.sum_filter, Finset.sum_range, Finset.sum_range_succ, Nat.lt_succ_iff ];
        · rw [ ← Int.ofNat_le, Int.natAbs_of_nonneg, Int.natAbs_of_nonneg ] <;> norm_num [ Finset.sum_ite ];
          · gcongr;
            · refine' Finset.sum_nonneg fun _ _ => _;
              exact MeasureTheory.integral_nonneg fun x => by unfold Set.indicator; aesop;
            · refine' le_add_of_nonneg_right _;
              exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _;
          · refine' Int.floor_nonneg.mpr _;
            refine' mul_nonneg ( div_nonneg _ _ ) ( Nat.cast_nonneg _ );
            · refine' add_nonneg ( Finset.sum_nonneg fun _ _ => _ ) _;
              · exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun _ _ => hf_nonneg x ) _;
              · exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _;
            · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
              rw [ Set.indicator_apply ] ; aesop;
          · refine' Int.floor_nonneg.mpr _;
            refine' mul_nonneg ( div_nonneg _ _ ) ( Nat.cast_nonneg _ );
            · refine' Finset.sum_nonneg fun _ _ => _;
              exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun x hx => hf_nonneg x ) _;
            · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
              rw [ Set.indicator_apply ] ; aesop;
        · grind +ring;
    · rw [ Nat.cast_sub ];
      · rw [ show ( ∑ x : Fin ( 2 * n ) with x ≤ i, bin_masses f n x ) = ∑ x : Fin ( 2 * n ), bin_masses f n x from ?_ ];
        · rw [ show ( ∑ x : Fin ( 2 * n ), bin_masses f n x ) = 1 from ?_ ] ; norm_num [ hm.ne' ];
          · refine' Int.floor_nonneg.mpr _;
            refine' mul_nonneg ( Finset.sum_nonneg fun _ _ => _ ) ( Nat.cast_nonneg _ );
            exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun _ _ => hf_nonneg x ) _;
          · convert sum_bin_masses_eq_one n hn f _ _ using 1 <;> aesop;
        · refine' Finset.sum_subset _ _ <;> simp +decide [ Finset.subset_iff ];
          exact fun x hx => False.elim <| hx.not_le <| Nat.le_of_lt_succ <| by linarith [ Fin.is_lt i, Fin.is_lt x ] ;
      · rw [ ← Int.ofNat_le, Int.natAbs_of_nonneg ];
        · refine' Int.le_of_lt_add_one ( Int.floor_lt.mpr _ );
          refine' lt_of_le_of_lt ( mul_le_mul_of_nonneg_right ( div_le_one_of_le₀ _ _ ) ( Nat.cast_nonneg _ ) ) _ <;> norm_num;
          · exact Finset.sum_le_sum_of_subset_of_nonneg ( Finset.filter_subset _ _ ) fun _ _ _ => by
              exact MeasureTheory.integral_nonneg fun x => Set.indicator_nonneg ( fun _ _ => hf_nonneg x ) _;
          · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
            rw [ Set.indicator_apply ] ; aesop;
        · refine' Int.floor_nonneg.mpr _;
          refine' mul_nonneg ( div_nonneg _ _ ) ( Nat.cast_nonneg _ );
          · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
            rw [ Set.indicator_apply ] ; aesop;
          · refine' Finset.sum_nonneg fun _ _ => MeasureTheory.integral_nonneg fun x => _;
            rw [ Set.indicator_apply ] ; aesop;
  have h_sum_bin_masses : ∑ j ∈ Finset.univ, (bin_masses f n j) = 1 := by
    convert sum_bin_masses_eq_one n hn f hf_supp hf_int using 1;
  simp_all +decide [ abs_le, div_le_iff₀, le_div_iff₀ ];
  have := h_diff ( i + 1 ) ( by linarith [ Fin.is_lt i ] ) ; ( have := h_diff i ( by linarith [ Fin.is_lt i ] ) ; simp_all +decide [ Finset.sum_filter, Finset.sum_range, Nat.lt_succ_iff ] ; );
  rw [ show ( ∑ a : Fin ( 2 * n ), if a ≤ i then bin_masses f n a else 0 ) = ( ∑ a : Fin ( 2 * n ), if a < i then bin_masses f n a else 0 ) + bin_masses f n i from ?_ ] at *;
  · field_simp;
    constructor <;> linarith [ Int.floor_le ( ( ( ∑ a : Fin ( 2 * n ), if a < i then bin_masses f n a else 0 ) + bin_masses f n i ) * m ), Int.lt_floor_add_one ( ( ( ∑ a : Fin ( 2 * n ), if a < i then bin_masses f n a else 0 ) + bin_masses f n i ) * m ), Int.floor_le ( ( ∑ a : Fin ( 2 * n ), if a < i then bin_masses f n a else 0 ) * m ), Int.lt_floor_add_one ( ( ∑ a : Fin ( 2 * n ), if a < i then bin_masses f n a else 0 ) * m ) ];
  · rw [ Finset.sum_eq_sum_diff_singleton_add ( Finset.mem_univ i ) ];
    rw [ Finset.sum_congr rfl fun x hx => if_congr ( by exact ⟨ fun h => lt_of_le_of_ne h ( by aesop ), fun h => le_of_lt h ⟩ ) rfl rfl ] ; simp +decide [ Finset.sum_ite, Finset.filter_lt_eq_Ioi ];
    rw [ Finset.sdiff_singleton_eq_erase, Finset.filter_erase ] ; aesop;
  · rw [ Finset.sum_eq_sum_diff_singleton_add ( Finset.mem_univ i ) ];
    rw [ Finset.sum_congr rfl fun x hx => if_congr ( by exact ⟨ fun hx' => lt_of_le_of_ne hx' ( by aesop ), fun hx' => le_of_lt hx' ⟩ ) rfl rfl ] ; simp +decide [ Finset.sum_ite, Finset.filter_lt_eq_Ioi ];
    rw [ Finset.sdiff_singleton_eq_erase, Finset.filter_erase ] ; aesop;
  · rw [ Finset.sum_eq_sum_diff_singleton_add ( Finset.mem_univ i ) ];
    rw [ Finset.sum_congr rfl fun x hx => if_congr ( by exact ⟨ fun hx' => lt_of_le_of_ne hx' ( by aesop ), fun hx' => le_of_lt hx' ⟩ ) rfl rfl ] ; simp +decide [ Finset.sum_ite, Finset.filter_lt_eq_Ioi ];
    rw [ Finset.sdiff_singleton_eq_erase, Finset.filter_erase ] ; aesop

/-- Claim 1.4: Contributing bins characterization. -/
theorem contributing_bins_iff (n : ℕ) (hn : n > 0) (ℓ s_lo : ℕ)
    (hℓ : 2 ≤ ℓ) (i : Fin (2 * n)) :
    i ∈ contributing_bins n ℓ s_lo ↔
      Nat.max 0 (s_lo - (2 * n - 1)) ≤ i.1 ∧ i.1 ≤ Nat.min (2 * n - 1) (s_lo + ℓ - 2) := by
  unfold contributing_bins;
  constructor <;> intro h <;> simp_all +decide [ Fin.exists_iff ];
  · grind +ring;
  · exact ⟨ s_lo - i, by omega, by omega, by omega ⟩

/-- Target cumulative mass (before flooring). -/
noncomputable def target_cum_mass (f : ℝ → ℝ) (n m : ℕ) (k : ℕ) : ℝ :=
  let masses := bin_masses f n
  let total_mass := ∑ j, masses j
  let cum_mass := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
  (cum_mass) / total_mass * m

-- Helper lemmas for cumulative mass bounds
private lemma target_cum_mass_eq (n m : ℕ) (f : ℝ → ℝ)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    target_cum_mass f n m k = (∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0) * ↑m := by
  unfold target_cum_mass; simp only []
  rw [hμ_sum, div_one]

private lemma target_cum_mass_nonneg (n m : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    0 ≤ target_cum_mass f n m k := by
  rw [target_cum_mass_eq n m f hμ_sum k]
  apply mul_nonneg
  · exact Finset.sum_nonneg fun j _ => by split_ifs <;> [exact bin_masses_nonneg f hf_nonneg n j; exact le_refl 0]
  · exact Nat.cast_nonneg _

private lemma ccd_eq_floor_natAbs (n m : ℕ) (f : ℝ → ℝ) (k : ℕ) :
    canonical_cumulative_distribution f n m k = ⌊target_cum_mass f n m k⌋.natAbs := by
  unfold canonical_cumulative_distribution target_cum_mass; rfl

private lemma ccd_cast_eq (n m : ℕ) (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hμ_sum : ∑ j : Fin (2 * n), bin_masses f n j = 1) (k : ℕ) :
    (canonical_cumulative_distribution f n m k : ℝ) = ⌊target_cum_mass f n m k⌋ := by
  rw [ccd_eq_floor_natAbs]
  have h_nn : (0 : ℤ) ≤ ⌊target_cum_mass f n m k⌋ :=
    Int.floor_nonneg.mpr (target_cum_mass_nonneg n m f hf_nonneg hμ_sum k)
  rw [Nat.cast_natAbs, Int.cast_abs, abs_of_nonneg (Int.cast_nonneg.mpr h_nn)]

private lemma partial_sum_discretization (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    (∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ,
      (canonical_discretization f n m i : ℝ)) =
    (canonical_cumulative_distribution f n m k : ℝ) := by
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  have h_mass_nz : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0 := by rw [hμ_sum]; exact one_ne_zero
  have h_D_2n := canonical_cumulative_distribution_2n f n m hn hm h_mass_nz
  have h_mono := canonical_cumulative_distribution_mono f hf_nonneg n m
  -- Each c_i = D(i+1) - D(i)
  have h_eq_diff : ∀ i : Fin (2 * n), canonical_discretization f n m i =
      canonical_cumulative_distribution f n m (i.1 + 1) - canonical_cumulative_distribution f n m i.1 :=
    canonical_discretization_eq_diff f n m h_D_2n
  -- Filter {i : Fin(2n) | i.val < k} is equivalent to Finset.range k mapped into Fin(2n)
  have h_filter_eq : Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ =
      Finset.image (fun i : Fin k => ⟨i.val, by omega⟩) Finset.univ := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro h; exact Finset.mem_image.mpr ⟨⟨i.val, h⟩, Finset.mem_univ _, rfl⟩
    · intro h; obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp h; exact j.isLt
  rw [h_filter_eq]
  rw [Finset.sum_image (by intro a _ b _ hab; ext; simpa using hab)]
  -- Now we sum D(i+1) - D(i) over Fin k, telescoping to D(k) - D(0)
  conv_lhs =>
    arg 2; ext i
    rw [h_eq_diff ⟨i.val, by omega⟩]
  -- Sum of (D(i+1) - D(i)) as ℝ
  have h_sum_telescope : ∑ i : Fin k, ((canonical_cumulative_distribution f n m (i.val + 1) : ℝ) -
      (canonical_cumulative_distribution f n m i.val : ℝ)) =
      (canonical_cumulative_distribution f n m k : ℝ) - (canonical_cumulative_distribution f n m 0 : ℝ) := by
    rw [Fin.sum_univ_eq_sum_range (fun i => (canonical_cumulative_distribution f n m (i + 1) : ℝ) -
        (canonical_cumulative_distribution f n m i : ℝ)) k]
    exact Finset.sum_range_sub (fun i => (canonical_cumulative_distribution f n m i : ℝ)) k
  -- D is monotone, so D(i+1) - D(i) ≥ 0 in ℕ, and (ℕ cast to ℝ) agrees with ℝ subtraction
  have h_cast_sub : ∀ i : Fin k,
      (↑(canonical_cumulative_distribution f n m (i.val + 1) - canonical_cumulative_distribution f n m i.val) : ℝ) =
      (canonical_cumulative_distribution f n m (i.val + 1) : ℝ) -
      (canonical_cumulative_distribution f n m i.val : ℝ) := by
    intro i
    rw [Nat.cast_sub (h_mono (Nat.le_succ _))]
  simp_rw [h_cast_sub]
  rw [h_sum_telescope]
  rw [canonical_cumulative_distribution_zero]
  simp

private lemma partial_sum_mu (n : ℕ) (f : ℝ → ℝ) (k : ℕ) :
    ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ, bin_masses f n i =
    ∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0 := by
  rw [Finset.sum_filter]

private lemma cumulative_delta_upper (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) ≤ 0 := by
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  -- Σ_{i<k} δ_i = D(k)/m - Σ_{j<k} μ_j
  rw [Finset.sum_sub_distrib]
  rw [← Finset.sum_div]
  rw [partial_sum_discretization n m hn hm f hf_nonneg hf_supp hf_int k hk]
  rw [partial_sum_mu]
  -- Now show D(k)/m - Σ_{j<k} μ_j ≤ 0, i.e., D(k)/m ≤ Σ_{j<k} μ_j
  rw [sub_nonpos, div_le_iff₀ hm_pos]
  -- D(k) ≤ Σ_{j<k} μ_j * m  (since D(k) = ⌊(Σ_{j<k} μ_j) * m⌋ ≤ (Σ_{j<k} μ_j) * m)
  rw [ccd_cast_eq n m f hf_nonneg hμ_sum k]
  rw [← target_cum_mass_eq n m f hμ_sum k]
  exact Int.floor_le _

private lemma cumulative_delta_lower (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (k : ℕ) (hk : k ≤ 2 * n) :
    -1 / ↑m ≤ ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < k) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) := by
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
  -- Σ_{i<k} δ_i = D(k)/m - Σ_{j<k} μ_j
  rw [Finset.sum_sub_distrib]
  rw [← Finset.sum_div]
  rw [partial_sum_discretization n m hn hm f hf_nonneg hf_supp hf_int k hk]
  rw [partial_sum_mu]
  -- Now show -1/m ≤ D(k)/m - Σ_{j<k} μ_j
  -- Equivalently: T*m - 1 ≤ D(k), where T = Σ_{j<k} μ_j, D(k) = ⌊T*m⌋
  set T := ∑ j : Fin (2 * n), if j.1 < k then bin_masses f n j else 0
  have hDk : (canonical_cumulative_distribution f n m k : ℝ) = ⌊T * ↑m⌋ := by
    rw [ccd_cast_eq n m f hf_nonneg hμ_sum k, ← target_cum_mass_eq n m f hμ_sum k]
  rw [hDk]
  have h_lb : T * ↑m - 1 < (⌊T * ↑m⌋ : ℝ) := Int.sub_one_lt_floor (T * ↑m)
  -- Goal: -1/m ≤ ⌊T*m⌋/m - T
  -- Rearranges to: T*m - 1 ≤ ⌊T*m⌋, which follows from h_lb (strict implies ≤)
  rw [show (⌊T * ↑m⌋ : ℝ) / ↑m - T = ((⌊T * ↑m⌋ : ℝ) - T * ↑m) / ↑m from by
    field_simp [ne_of_gt hm_pos]]
  rw [show (-1 : ℝ) / ↑m = (-1) / ↑m from rfl]
  rw [div_le_div_iff_of_pos_right hm_pos]
  linarith

private lemma range_sum_delta_le (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (a b : ℕ) (hab : a ≤ b) (hb : b ≤ 2 * n) :
    ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) ≤ 1 / ↑m := by
  -- Range sum = cumulative(b) - cumulative(a)
  -- cumulative(b) ≤ 0 and cumulative(a) ≥ -1/m
  -- So range sum = cumulative(b) - cumulative(a) ≤ 0 - (-1/m) = 1/m
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  have h_split : Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ =
      Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ \
      Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ := by
    ext i; simp [Finset.mem_sdiff, Finset.mem_filter]; omega
  have h_subset : Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ ⊆
      Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ := by
    intro x; simp [Finset.mem_filter]; omega
  have h_sum_eq : ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) =
      ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ,
        ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) -
      ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ,
        ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) := by
    rw [h_split]
    have := Finset.sum_sdiff h_subset (f := fun i => (canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i)
    linarith
  rw [h_sum_eq]
  have h_upper := cumulative_delta_upper n m hn hm f hf_nonneg hf_supp hf_int b hb
  have h_lower := cumulative_delta_lower n m hn hm f hf_nonneg hf_supp hf_int a (le_trans hab hb)
  set S_a := ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i)
  set S_b := ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i)
  -- h_upper : S_b ≤ 0, h_lower : -1/m ≤ S_a, goal : S_b - S_a ≤ 1/m
  have h1 : -1 / (↑m : ℝ) = -(1 / ↑m) := by ring
  rw [h1] at h_lower
  linarith

private lemma range_sum_delta_ge (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (a b : ℕ) (hab : a ≤ b) (hb : b ≤ 2 * n) :
    -1 / ↑m ≤ ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) := by
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  have h_split : Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ =
      Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ \
      Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ := by
    ext i; simp [Finset.mem_sdiff, Finset.mem_filter]; omega
  have h_subset : Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ ⊆
      Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ := by
    intro x; simp [Finset.mem_filter]; omega
  have h_sum_eq : ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => a ≤ i.val ∧ i.val < b) Finset.univ,
      ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) =
      ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < b) Finset.univ,
        ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) -
      ∑ i ∈ Finset.filter (fun i : Fin (2 * n) => i.val < a) Finset.univ,
        ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) := by
    rw [h_split]
    have := Finset.sum_sdiff h_subset (f := fun i => (canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i)
    linarith
  rw [h_sum_eq]
  have h_upper := cumulative_delta_upper n m hn hm f hf_nonneg hf_supp hf_int a (le_trans hab hb)
  have h_lower := cumulative_delta_lower n m hn hm f hf_nonneg hf_supp hf_int b hb
  linarith

/-- Discretization error bound for the autoconvolution test value.
    TV(c) - TV_cont ≤ (4n/ℓ)·(1/m² + 2W/m). -/
theorem discretization_autoconv_error (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / m) :
    test_value n m (canonical_discretization f n m) ℓ s_lo - test_value_continuous n f ℓ s_lo ≤
      (4 * n / ℓ) * (1 / m ^ 2 + 2 * W / m) := by
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  have hW_nn : 0 ≤ W := by
    rw [hW]; exact div_nonneg (Finset.sum_nonneg fun i _ => Nat.cast_nonneg _) (le_of_lt hm_pos)
  -- TV_cont ≥ 0
  have hcont_nn : 0 ≤ test_value_continuous n f ℓ s_lo := by
    unfold test_value_continuous discrete_autoconvolution; simp only []
    apply mul_nonneg (by positivity)
    apply Finset.sum_nonneg; intro k _; apply Finset.sum_nonneg; intro i _
    apply Finset.sum_nonneg; intro j _
    split_ifs <;> [exact mul_nonneg (mul_nonneg (by positivity) (bin_masses_nonneg f hf_nonneg n i))
      (mul_nonneg (by positivity) (bin_masses_nonneg f hf_nonneg n j)); exact le_refl 0]
  -- RHS ≥ 0
  have hRHS : 0 ≤ (4 * (↑n : ℝ) / ↑ℓ) * (1 / ↑m ^ 2 + 2 * W / ↑m) := by positivity
  -- When TV_disc ≤ TV_cont: diff ≤ 0 ≤ RHS. Done.
  by_cases h : test_value n m (canonical_discretization f n m) ℓ s_lo ≤
    test_value_continuous n f ℓ s_lo
  · linarith
  · -- TV_disc > TV_cont. The bound follows from the two-term decomposition
    push_neg at h
    -- Establish the per-bin error, mass sums, and W ≤ 1
    have hδ := discretization_error_bound n m hn hm f hf_nonneg hf_supp hf_int
    have hμ_nn := fun i => bin_masses_nonneg f hf_nonneg n i
    have hμ_sum := sum_bin_masses_eq_one n hn f hf_supp hf_int
    have h_mass_nz : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0 := by rw [hμ_sum]; exact one_ne_zero
    have h_sum_m : (∑ i : Fin (2 * n), (canonical_discretization f n m i : ℝ)) = ↑m :=
      by exact_mod_cast canonical_discretization_sum_eq_m f n m hn hm h_mass_nz hf_nonneg
    have hw_sum : ∑ i : Fin (2 * n), (canonical_discretization f n m i : ℝ) / ↑m = 1 := by
      rw [← Finset.sum_div, h_sum_m, div_self (ne_of_gt hm_pos)]
    have hδ_sum : ∑ i : Fin (2 * n), ((canonical_discretization f n m i : ℝ) / ↑m - bin_masses f n i) = 0 := by
      rw [Finset.sum_sub_distrib, hw_sum, hμ_sum]; ring
    have hW_le : W ≤ 1 := by
      rw [hW]; rw [div_le_one hm_pos]; exact le_of_le_of_eq
        (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun _ _ _ => Nat.cast_nonneg _))
        h_sum_m
    set w : Fin (2 * n) → ℝ := fun i => (canonical_discretization f n m i : ℝ) / ↑m with hw_def
    set μ : Fin (2 * n) → ℝ := bin_masses f n with hμ_def
    set δ_i : Fin (2 * n) → ℝ := fun i => w i - μ i with hδ_def
    -- Per-bin error: |δ_i| ≤ 1/m
    have hδ_bound : ∀ i, |δ_i i| ≤ 1 / ↑m := fun i => by rw [hδ_def, hw_def, hμ_def]; exact hδ i
    -- w_i ≥ 0
    have hw_nn : ∀ i, 0 ≤ w i := fun i => by rw [hw_def]; exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    -- δ_i = w_i - μ_i
    have hδ_eq : ∀ i, δ_i i = w i - μ i := fun i => rfl
    -- The two-term identity: w_i · w_j - μ_i · μ_j = δ_i · w_j + μ_i · δ_j
    have h_two_term : ∀ i j, w i * w j - μ i * μ j = δ_i i * w j + μ i * δ_i j := by
      intro i j; rw [hδ_def]; ring
    -- Crude per-term bound: |w_i · w_j - μ_i · μ_j| ≤ |δ_i| · w_j + μ_i · |δ_j|
    have h_per_term : ∀ i j, w i * w j - μ i * μ j ≤ (w j + μ i) / ↑m := by
      intro i j
      rw [h_two_term]
      have h1 : δ_i i * w j ≤ |δ_i i| * w j := by
        exact mul_le_mul_of_nonneg_right (le_abs_self _) (hw_nn j)
      have h2 : μ i * δ_i j ≤ μ i * |δ_i j| := mul_le_mul_of_nonneg_left (le_abs_self _) (hμ_nn i)
      have h3 : |δ_i i| * w j ≤ (1 / ↑m) * w j := mul_le_mul_of_nonneg_right (hδ_bound i) (hw_nn j)
      have h4 : μ i * |δ_i j| ≤ μ i * (1 / ↑m) := mul_le_mul_of_nonneg_left (hδ_bound j) (hμ_nn i)
      have h5 : 1 / ↑m * w j + μ i * (1 / ↑m) = (w j + μ i) / ↑m := by ring
      linarith
    have hn_pos : (0 : ℝ) < ↑n := Nat.cast_pos.mpr hn
    have hℓ_pos : (0 : ℝ) < ↑ℓ := by exact_mod_cast Nat.lt_of_lt_of_le (by norm_num) hℓ
    have h_aw : ∀ i : Fin (2 * n), (4 * ↑n : ℝ) / ↑m * ↑(canonical_discretization f n m i) =
        (4 * ↑n) * w i := by
      intro i; rw [hw_def]; field_simp
    set Q : ℝ := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then w i * w j - μ i * μ j else 0 with hQ_def
    have h_diff_eq : test_value n m (canonical_discretization f n m) ℓ s_lo -
        test_value_continuous n f ℓ s_lo = ((4 : ℝ) * ↑n / ↑ℓ) * Q := by
      have h_inner : ∀ k, (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 = k then (4 : ℝ) * ↑n / ↑m * ↑(canonical_discretization f n m i) *
              ((4 : ℝ) * ↑n / ↑m * ↑(canonical_discretization f n m j)) else 0) -
          (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 = k then (4 : ℝ) * ↑n * bin_masses f n i *
              ((4 : ℝ) * ↑n * bin_masses f n j) else 0) =
          ((4 : ℝ) * ↑n) ^ 2 * (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
            if i.1 + j.1 = k then w i * w j - μ i * μ j else 0) := by
        intro k
        rw [← Finset.sum_sub_distrib, Finset.mul_sum]
        congr 1; ext i
        rw [← Finset.sum_sub_distrib, Finset.mul_sum]
        congr 1; ext j
        rw [hw_def, hμ_def]; split_ifs <;> field_simp <;> ring
      show test_value n m (canonical_discretization f n m) ℓ s_lo -
        test_value_continuous n f ℓ s_lo = ((4 : ℝ) * ↑n / ↑ℓ) * Q
      simp only [test_value, test_value_continuous, discrete_autoconvolution]
      rw [← mul_sub, ← Finset.sum_sub_distrib]
      simp_rw [h_inner, ← Finset.mul_sum]
      rw [hQ_def]
      have hℓ_ne : (↑ℓ : ℝ) ≠ 0 := ne_of_gt hℓ_pos
      have hn_ne : (↑n : ℝ) ≠ 0 := ne_of_gt hn_pos
      field_simp
    suffices hQ_bound : Q ≤ 1 / ↑m ^ 2 + 2 * W / ↑m by
      rw [h_diff_eq]; exact mul_le_mul_of_nonneg_left hQ_bound (by positivity)
    set Part_A := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then δ_i i * w j else 0
    set Part_B := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
        ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then μ i * δ_i j else 0
    have hQ_eq : Q = Part_A + Part_B := by
      show Q = Part_A + Part_B
      simp only [hQ_def, Part_A, Part_B, ← Finset.sum_add_distrib]
      congr 1; ext k; congr 1; ext i; congr 1; ext j
      split_ifs with h
      · exact h_two_term i j
      · simp
    -- Part A: exchange sums, bound range sums by 1/m, restrict to CB
    have hPartA_exch : Part_A = ∑ j : Fin (2 * n), w j *
        (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ i : Fin (2 * n),
          if i.1 + j.1 = k then δ_i i else (0 : ℝ)) := by
      simp only [Part_A]
      simp_rw [show ∀ (k : ℕ) (i j : Fin (2 * n)),
          (if i.1 + j.1 = k then δ_i i * w j else (0 : ℝ)) =
          w j * (if i.1 + j.1 = k then δ_i i else 0) from
        fun _ _ _ => by split_ifs <;> ring]
      conv_lhs => arg 2; ext k; rw [Finset.sum_comm]
      rw [Finset.sum_comm]
      congr 1; ext j; simp_rw [← Finset.mul_sum]
    set g_fn : Fin (2 * n) → ℝ := fun j =>
      ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ i : Fin (2 * n),
        if i.1 + j.1 = k then δ_i i else 0
    have hg_eq : ∀ j : Fin (2 * n), g_fn j = ∑ i ∈ Finset.filter
        (fun i : Fin (2 * n) => s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 + 2 ≤ s_lo + ℓ)
        Finset.univ, δ_i i := by
      intro j; simp only [g_fn]
      -- Exchange order of summation: Σ_k Σ_i → Σ_i Σ_k
      rw [Finset.sum_comm]
      -- Commute equality in the if-condition, then apply Finset.sum_ite_eq'
      simp_rw [show ∀ (i : Fin (2 * n)) (k : ℕ),
          (if i.1 + j.1 = k then δ_i i else (0 : ℝ)) =
          (if k = i.1 + j.1 then δ_i i else 0) from
        fun i k => by split_ifs with h1 h2 <;> simp_all]
      simp_rw [Finset.sum_ite_eq', Finset.mem_Icc]
      rw [← Finset.sum_filter]
      congr 1
      ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · intro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
      · intro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
    have hg_le : ∀ j, g_fn j ≤ 1 / ↑m := by
      intro j; rw [hg_eq j]
      have h_eq : Finset.filter
          (fun i : Fin (2 * n) => s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 + 2 ≤ s_lo + ℓ)
          Finset.univ =
        Finset.filter (fun i : Fin (2 * n) =>
          (if s_lo ≥ j.1 then s_lo - j.1 else 0) ≤ i.1 ∧
          i.1 < (if s_lo + ℓ ≥ j.1 + 2 then s_lo + ℓ - j.1 - 2 + 1 else 0))
          Finset.univ := by
        ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        constructor <;> intro ⟨h1, h2⟩ <;> (constructor <;> (split_ifs at * <;> omega))
      rw [h_eq]; set a := if s_lo ≥ j.1 then s_lo - j.1 else 0
      set b := if s_lo + ℓ ≥ j.1 + 2 then s_lo + ℓ - j.1 - 2 + 1 else 0
      by_cases hab : a ≤ b
      · by_cases hb2n : b ≤ 2 * n
        · exact range_sum_delta_le n m hn hm f hf_nonneg hf_supp hf_int a b hab hb2n
        · by_cases ha2n : a ≤ 2 * n
          · have h_clip : Finset.filter (fun i : Fin (2 * n) => a ≤ i.1 ∧ i.1 < b) Finset.univ =
                Finset.filter (fun i : Fin (2 * n) => a ≤ i.1 ∧ i.1 < 2 * n) Finset.univ := by
              ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and]
              constructor <;> intro ⟨h1, _⟩ <;> exact ⟨h1, by omega⟩
            rw [h_clip]; exact range_sum_delta_le n m hn hm f hf_nonneg hf_supp hf_int a (2*n) ha2n le_rfl
          · have : Finset.filter (fun i : Fin (2 * n) => a ≤ i.1 ∧ i.1 < b) Finset.univ = ∅ := by
              rw [Finset.filter_eq_empty_iff]; intro i _; simp; omega
            rw [this, Finset.sum_empty]; positivity
      · have : Finset.filter (fun i : Fin (2 * n) => a ≤ i.1 ∧ i.1 < b) Finset.univ = ∅ := by
          rw [Finset.filter_eq_empty_iff]; intro i _; simp; omega
        rw [this, Finset.sum_empty]; positivity
    have hg_zero : ∀ j : Fin (2 * n), j ∉ contributing_bins n ℓ s_lo → g_fn j = 0 := by
      intro j hj; rw [hg_eq j, Finset.sum_eq_zero]; intro i hi
      exfalso; apply hj; unfold contributing_bins
      simp [Finset.mem_filter] at hi ⊢; exact ⟨i, by omega, by omega⟩
    have hPartA_le : Part_A ≤ W / ↑m := by
      rw [hPartA_exch]
      rw [show ∑ j : Fin (2 * n), w j * g_fn j =
        ∑ j ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, w j * g_fn j +
        ∑ j ∈ Finset.filter (· ∉ contributing_bins n ℓ s_lo) Finset.univ, w j * g_fn j from
        (Finset.sum_filter_add_sum_filter_not _ _ _).symm]
      have : ∑ j ∈ Finset.filter (· ∉ contributing_bins n ℓ s_lo) Finset.univ, w j * g_fn j = 0 := by
        apply Finset.sum_eq_zero; intro j hj
        rw [hg_zero j (Finset.mem_filter.mp hj).2, mul_zero]
      rw [this, add_zero]
      calc ∑ j ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, w j * g_fn j
          ≤ ∑ j ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, w j * (1/↑m) := by
            apply Finset.sum_le_sum; intro j _; exact mul_le_mul_of_nonneg_left (hg_le j) (hw_nn j)
        _ = (1/↑m) * ∑ j ∈ contributing_bins n ℓ s_lo, w j := by
            rw [Finset.mul_sum]
            have hfilt : Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ =
                contributing_bins n ℓ s_lo := by ext j; simp [Finset.mem_filter]
            rw [hfilt]; congr 1; ext j; ring
        _ = W / ↑m := by rw [hW, hw_def]; rw [Finset.sum_div]; field_simp
    -- Part B: exchange sums, bound range sums, use CB contiguity for mu bound
    have hPartB_exch : Part_B = ∑ i : Fin (2 * n), μ i *
        (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ j : Fin (2 * n),
          if i.1 + j.1 = k then δ_i j else 0) := by
      have h_factor : ∀ (i j : Fin (2 * n)) (k : ℕ),
          (if i.1 + j.1 = k then μ i * δ_i j else (0 : ℝ)) =
          μ i * (if i.1 + j.1 = k then δ_i j else 0) := by
        intros; split_ifs <;> ring
      calc Part_B
          = ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
              ∑ i : Fin (2 * n), μ i * ∑ j : Fin (2 * n),
                if i.1 + j.1 = k then δ_i j else 0 := by
            apply Finset.sum_congr rfl; intro k _
            apply Finset.sum_congr rfl; intro i _
            conv_lhs => arg 2; ext j; rw [h_factor]
            rw [← Finset.mul_sum]
        _ = ∑ i : Fin (2 * n), μ i *
              (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ j : Fin (2 * n),
                if i.1 + j.1 = k then δ_i j else 0) := by
            rw [Finset.sum_comm]
            congr 1; ext i; rw [Finset.mul_sum]
    set h_fn : Fin (2 * n) → ℝ := fun i =>
      ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then δ_i j else 0
    have hh_eq : ∀ i : Fin (2 * n), h_fn i = ∑ j ∈ Finset.filter
        (fun j : Fin (2 * n) => s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 + 2 ≤ s_lo + ℓ)
        Finset.univ, δ_i j := by
      intro i
      show (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then δ_i j else 0) = _
      rw [Finset.sum_comm]
      have key : ∀ (j : Fin (2*n)) (k : ℕ),
          (if i.1 + j.1 = k then δ_i j else (0:ℝ)) = (if k = i.1 + j.1 then δ_i j else 0) := by
        intros j k; split_ifs with h1 h2 <;> simp_all
      simp_rw [key, Finset.sum_ite_eq', Finset.mem_Icc]
      rw [← Finset.sum_filter]
      congr 1; ext j
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · intro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
      · intro ⟨h1, h2⟩; exact ⟨h1, by omega⟩
    have hh_le : ∀ i, h_fn i ≤ 1 / ↑m := by
      intro i; rw [hh_eq i]
      have h_eq : Finset.filter
          (fun j : Fin (2 * n) => s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 + 2 ≤ s_lo + ℓ)
          Finset.univ =
        Finset.filter (fun j : Fin (2 * n) =>
          (if s_lo ≥ i.1 then s_lo - i.1 else 0) ≤ j.1 ∧
          j.1 < (if s_lo + ℓ ≥ i.1 + 2 then s_lo + ℓ - i.1 - 2 + 1 else 0))
          Finset.univ := by
        ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        constructor
        · intro ⟨h1, h2⟩; constructor <;> split_ifs <;> omega
        · intro ⟨h1, h2⟩; constructor <;> (split_ifs at h1 h2 <;> omega)
      rw [h_eq]; set a := if s_lo ≥ i.1 then s_lo - i.1 else 0
      set b := if s_lo + ℓ ≥ i.1 + 2 then s_lo + ℓ - i.1 - 2 + 1 else 0
      by_cases hab : a ≤ b
      · by_cases hb2n : b ≤ 2 * n
        · exact range_sum_delta_le n m hn hm f hf_nonneg hf_supp hf_int a b hab hb2n
        · have h_clip : Finset.filter (fun j : Fin (2 * n) => a ≤ j.1 ∧ j.1 < b) Finset.univ =
              Finset.filter (fun j : Fin (2 * n) => a ≤ j.1 ∧ j.1 < 2 * n) Finset.univ := by
            ext j; simp only [Finset.mem_filter, Finset.mem_univ, true_and]
            constructor <;> intro ⟨h1, _⟩ <;> exact ⟨h1, by omega⟩
          rw [h_clip]
          by_cases ha2n : a ≤ 2 * n
          · exact range_sum_delta_le n m hn hm f hf_nonneg hf_supp hf_int a (2*n) ha2n le_rfl
          · have : Finset.filter (fun j : Fin (2 * n) => a ≤ j.1 ∧ j.1 < 2 * n) Finset.univ = ∅ := by
              rw [Finset.filter_eq_empty_iff]; intro j _; omega
            rw [this, Finset.sum_empty]; positivity
      · have : Finset.filter (fun j : Fin (2 * n) => a ≤ j.1 ∧ j.1 < b) Finset.univ = ∅ := by
          rw [Finset.filter_eq_empty_iff]; intro j _; omega
        rw [this, Finset.sum_empty]; positivity
    have hh_zero : ∀ i : Fin (2 * n), i ∉ contributing_bins n ℓ s_lo → h_fn i = 0 := by
      intro i hi; rw [hh_eq i, Finset.sum_eq_zero]; intro j hj
      exfalso; apply hi; unfold contributing_bins
      simp [Finset.mem_filter] at hj ⊢; exact ⟨j, by omega, by omega⟩
    have hCB_mu_le : ∑ i ∈ contributing_bins n ℓ s_lo, μ i ≤ W + 1 / ↑m := by
      have h_mu_eq : ∑ i ∈ contributing_bins n ℓ s_lo, μ i =
          W - ∑ i ∈ contributing_bins n ℓ s_lo, δ_i i := by
        have : ∑ i ∈ contributing_bins n ℓ s_lo, μ i =
            ∑ i ∈ contributing_bins n ℓ s_lo, w i -
            ∑ i ∈ contributing_bins n ℓ s_lo, δ_i i := by
          rw [← Finset.sum_sub_distrib]; congr 1; ext i; rw [hδ_def]; ring
        rw [this, hW, hw_def, Finset.sum_div]
      rw [h_mu_eq]
      suffices hd : ∑ i ∈ contributing_bins n ℓ s_lo, δ_i i ≥ -1 / ↑m by
        have : (-1 : ℝ) / ↑m = -(1 / ↑m) := by ring
        linarith
      set cb_lo := Nat.max 0 (s_lo - (2 * n - 1))
      set cb_hi := Nat.min (2 * n - 1) (s_lo + ℓ - 2)
      have hCB_eq : contributing_bins n ℓ s_lo =
          Finset.filter (fun i : Fin (2 * n) => cb_lo ≤ i.1 ∧ i.1 < cb_hi + 1) Finset.univ := by
        ext i; rw [contributing_bins_iff n hn ℓ s_lo hℓ i]
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        constructor <;> intro ⟨h1, h2⟩ <;> exact ⟨h1, by omega⟩
      rw [hCB_eq]
      by_cases h_ne : cb_lo ≤ cb_hi
      · have hcb_hi_bound : cb_hi ≤ 2 * n - 1 := Nat.min_le_left _ _
        exact range_sum_delta_ge n m hn hm f hf_nonneg hf_supp hf_int cb_lo (cb_hi + 1) (by omega) (by omega)
      · have : Finset.filter (fun i : Fin (2 * n) => cb_lo ≤ i.1 ∧ i.1 < cb_hi + 1) Finset.univ = ∅ := by
          rw [Finset.filter_eq_empty_iff]; intro i _; omega
        rw [this, Finset.sum_empty]; simp only [ge_iff_le, neg_div, Left.neg_nonpos_iff]; positivity
    have hPartB_le : Part_B ≤ W / ↑m + 1 / ↑m ^ 2 := by
      rw [hPartB_exch]
      rw [show ∑ i : Fin (2 * n), μ i * h_fn i =
        ∑ i ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, μ i * h_fn i +
        ∑ i ∈ Finset.filter (· ∉ contributing_bins n ℓ s_lo) Finset.univ, μ i * h_fn i from
        (Finset.sum_filter_add_sum_filter_not _ _ _).symm]
      have : ∑ i ∈ Finset.filter (· ∉ contributing_bins n ℓ s_lo) Finset.univ, μ i * h_fn i = 0 := by
        apply Finset.sum_eq_zero; intro i hi
        rw [hh_zero i (Finset.mem_filter.mp hi).2, mul_zero]
      rw [this, add_zero]
      calc ∑ i ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, μ i * h_fn i
          ≤ ∑ i ∈ Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ, μ i * (1/↑m) := by
            apply Finset.sum_le_sum; intro i _; exact mul_le_mul_of_nonneg_left (hh_le i) (hμ_nn i)
        _ = (1/↑m) * ∑ i ∈ contributing_bins n ℓ s_lo, μ i := by
            rw [Finset.mul_sum]
            have hfilt : Finset.filter (· ∈ contributing_bins n ℓ s_lo) Finset.univ =
                contributing_bins n ℓ s_lo := by ext j; simp [Finset.mem_filter]
            rw [hfilt]; congr 1; ext i; ring
        _ ≤ (1/↑m) * (W + 1/↑m) := mul_le_mul_of_nonneg_left hCB_mu_le (by positivity)
        _ = W / ↑m + 1 / ↑m ^ 2 := by ring
    -- Combine: Q = Part_A + Part_B <= W/m + W/m + 1/m^2 = 1/m^2 + 2W/m
    rw [hQ_eq]
    have h_sum_le : Part_A + Part_B ≤ W / ↑m + (W / ↑m + 1 / ↑m ^ 2) :=
      add_le_add hPartA_le hPartB_le
    have h_rearr : W / ↑m + (W / ↑m + 1 / ↑m ^ 2) = 1 / ↑m ^ 2 + 2 * W / ↑m := by ring
    linarith

/-- Window-dependent correction bound (Claim 1.2).
    R(f) ≥ TV(c, ℓ, s_lo) - correction. -/
theorem correction_term_bound (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / m) :
    autoconvolution_ratio f ≥ test_value n m (canonical_discretization f n m) ℓ s_lo - (4 * n / ℓ) * (1 / m ^ 2 + 2 * W / m) := by
  have h_cont : autoconvolution_ratio f ≥ test_value_continuous n f ℓ s_lo :=
    continuous_test_value_le_ratio n hn f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ
  have h_disc : test_value n m (canonical_discretization f n m) ℓ s_lo -
      test_value_continuous n f ℓ s_lo ≤
      (4 * ↑n / ↑ℓ) * (1 / ↑m ^ 2 + 2 * W / ↑m) :=
    discretization_autoconv_error n m hn hm f hf_nonneg hf_supp hf_int ℓ s_lo hℓ W hW
  linarith

/-- Claim 1.2 (corrected): Global correction term bound. -/
theorem correction_term (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥
      (max_test_value n m (canonical_discretization f n m) : ℝ) - 2 * n * (2 / m + 1 / m ^ 2) := by
  -- Step 1: The max test value is attained at some window (ℓ, s_lo) with ℓ ≥ 2
  obtain ⟨ℓ, s_lo, hℓ_mem, _, h_max_eq⟩ :=
    max_test_value_le_max n m hn (canonical_discretization f n m)
  have hℓ : 2 ≤ ℓ := (Finset.mem_Icc.mp hℓ_mem).1
  have hm_pos : (0 : ℝ) < ↑m := Nat.cast_pos.mpr hm
  -- Step 2: Apply correction_term_bound at the maximizing window
  let W : ℝ :=
    (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / ↑m
  have hbound := correction_term_bound n m hn hm f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ W rfl
  -- Step 3: Canonical discretization sums to m (needed for W ≤ 1)
  have h_mass_nz : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0 := by
    rw [sum_bin_masses_eq_one n hn f hf_supp hf_int]; exact one_ne_zero
  have h_sum_m : (∑ i : Fin (2 * n), (canonical_discretization f n m i : ℝ)) = ↑m := by
    exact_mod_cast canonical_discretization_sum_eq_m f n m hn hm h_mass_nz hf_nonneg
  -- Step 4: W ≤ 1
  have hW_le : W ≤ 1 := by
    show (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / ↑m ≤ 1
    rw [div_le_one (by positivity : (0 : ℝ) < ↑m)]
    exact le_of_le_of_eq
      (Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun _ _ _ => Nat.cast_nonneg _))
      h_sum_m
  -- Step 5: Bound (4n/ℓ)·(1/m² + 2W/m) ≤ 2n·(2/m + 1/m²)
  -- Since ℓ ≥ 2: 4n/ℓ ≤ 2n. Since W ≤ 1: 2W/m ≤ 2/m.
  have h_ell_bound : (4 * (n : ℝ) / ℓ) ≤ 2 * n := by
    rw [div_le_iff₀ (by positivity : (0 : ℝ) < ℓ)]; nlinarith [show (2 : ℝ) ≤ ℓ by exact_mod_cast hℓ]
  have h_W_bound : 2 * W / ↑m ≤ 2 / ↑m := by
    apply div_le_div_of_nonneg_right _ (le_of_lt hm_pos)
    linarith
  -- Step 6: Chain the bounds
  have hW_nonneg : (0 : ℝ) ≤ W := by
    show (0 : ℝ) ≤ (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / ↑m
    exact div_nonneg (Finset.sum_nonneg fun _ _ => Nat.cast_nonneg _) (Nat.cast_nonneg _)
  rw [h_max_eq]
  have h_corr : (4 * ↑n / ↑ℓ) * (1 / ↑m ^ 2 + 2 * W / ↑m) ≤ 2 * ↑n * (2 / ↑m + 1 / ↑m ^ 2) := by
    calc (4 * ↑n / ↑ℓ) * (1 / ↑m ^ 2 + 2 * W / ↑m)
        ≤ (2 * ↑n) * (1 / ↑m ^ 2 + 2 * W / ↑m) := by
          apply mul_le_mul_of_nonneg_right h_ell_bound
          positivity
      _ ≤ (2 * ↑n) * (1 / ↑m ^ 2 + 2 / ↑m) := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          linarith [h_W_bound]
      _ = 2 * ↑n * (2 / ↑m + 1 / ↑m ^ 2) := by ring
  linarith

/-- Claim 1.3: Dynamic threshold soundness. -/
theorem dynamic_threshold_sound (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (c i : ℝ)) / m)
    (h_exceeds : test_value n m c ℓ s_lo > c_target + (4 * n / ℓ) * (1 / m ^ 2 + 2 * W / m)) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      MeasureTheory.eLpNorm (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int h_conv_fin hdisc
  have hW' : W = (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / ↑m := by
    rw [hW]; congr 1; congr 1; ext i; rw [hdisc]
  have hbound := correction_term_bound n m hn hm f hf_nonneg hf_supp hf_int h_conv_fin ℓ s_lo hℓ W hW'
  rw [hdisc] at hbound
  linarith

end -- noncomputable section
