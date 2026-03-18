/-
PROMPT FOR ARISTOTLE: Prove dynamic threshold soundness (Claim 1.3).

GOAL: Prove `dynamic_threshold_sound` — if TV exceeds the dynamic threshold
c_target + 1/m² + 2W/m, then R(f) ≥ c_target for all f discretizing to c.

All helper lemmas below are PROVED and can be used freely.
The ONLY sorry is `dynamic_threshold_sound` at the bottom.

Note: `contributing_bins_iff` (Claim 1.4) is PROVED below.
Note: `discretization_error_bound` and `correction_term_bound` are stated as axioms
(proved elsewhere, budget-limited).

PROOF STRATEGY:
This is a refinement of Claim 1.2 with window-dependent correction.
The uniform correction 2/m + 1/m² bounds the error using ∑|δ_i| ≤ d/m.
But for a specific window (ℓ, s_lo), only contributing bins matter.
The error for that window involves ∑_{i ∈ B} |δ_i|·μ_j terms.
Key bound: |TV_continuous - TV_discrete| ≤ 2W/m + 1/m²
where W = (∑_{i ∈ B} c_i)/m ≤ 1.
So: TV > c_target + 2W/m + 1/m² ⟹ TV_continuous > c_target ⟹ R(f) ≥ c_target.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 400000
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

noncomputable def bin_masses (f : ℝ → ℝ) (n : ℕ) : Fin (2 * n) → ℝ :=
  fun i =>
    let δ := 1 / (4 * n : ℝ)
    let a := -(1/4 : ℝ) + i * δ
    let b := -(1/4 : ℝ) + (i + 1) * δ
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico a b) f)

noncomputable def canonical_discretization (f : ℝ → ℝ) (n m : ℕ) : Fin (2 * n) → ℕ :=
  fun i =>
    let masses := bin_masses f n
    let total_mass := ∑ j, masses j
    let cum_mass (k : ℕ) := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
    let target_cum (k : ℕ) := (cum_mass k) / total_mass * m
    let discrete_cum (k : ℕ) := ⌊target_cum k⌋.natAbs
    if i.1 + 1 < 2 * n then discrete_cum (i.1 + 1) - discrete_cum i.1
    else m - discrete_cum i.1

def discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then a i * a j else 0

noncomputable def test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  let d := 2 * n
  let a : Fin d → ℝ := fun i => (4 * n : ℝ) / m * (c i : ℝ)
  let conv := discrete_autoconvolution a
  let sum_conv := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), conv k
  (1 / (4 * n * ℓ : ℝ)) * sum_conv

def contributing_bins (n : ℕ) (ℓ s_lo : ℕ) : Finset (Fin (2 * n)) :=
  let d := 2 * n
  Finset.filter (fun i => ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2) Finset.univ

/-! ## Proved helper lemmas -/

/-- Claim 1.4: Contributing bins formula. PROVED. -/
theorem contributing_bins_iff (n : ℕ) (hn : n > 0) (ℓ s_lo : ℕ)
    (hℓ : 2 ≤ ℓ) (i : Fin (2 * n)) :
    i ∈ contributing_bins n ℓ s_lo ↔
      Nat.max 0 (s_lo - (2 * n - 1)) ≤ i.1 ∧ i.1 ≤ Nat.min (2 * n - 1) (s_lo + ℓ - 2) := by
  unfold contributing_bins;
  constructor <;> intro h <;> simp_all +decide [ Fin.exists_iff ];
  · grind +ring;
  · exact ⟨ s_lo - i, by omega, by omega, by omega ⟩

/-- Sum of bin masses = 1 for normalized f. -/
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
  exact fun x hx => Classical.not_not.1 fun hx' => by have := hf_supp hx'; exact not_le.mpr this.2 (hx (by linarith [this.1])) ;

/-- Per-bin discretization error ≤ 1/m (axiom — proved in output (22), budget-limited). -/
axiom discretization_error_bound (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    ∀ i : Fin (2 * n), |(canonical_discretization f n m i : ℝ) / m - bin_masses f n i| ≤ 1 / m

/-- Window-dependent correction bound (axiom — consequence of Claim 1.2). -/
axiom correction_term_bound (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / m) :
    autoconvolution_ratio f ≥ test_value n m (canonical_discretization f n m) ℓ s_lo - 1 / m ^ 2 - 2 * W / m

/-! ## ============================================================
    THEOREM TO PROVE (fill in the sorry)
    ============================================================ -/

/-
Claim 1.3: Dynamic threshold soundness.

Given: TV(ℓ, s_lo, c) > c_target + 1/m² + 2W/m where W = (∑_{i ∈ B} c_i)/m.
Show: For any f with canonical_discretization f = c, R(f) ≥ c_target.

Proof:
1. By correction_term_bound:
   R(f) ≥ TV(ℓ, s_lo, c) - 1/m² - 2W/m.
2. Since TV > c_target + 1/m² + 2W/m:
   R(f) > c_target + 1/m² + 2W/m - 1/m² - 2W/m = c_target.
-/
theorem dynamic_threshold_sound (n m : ℕ) (c_target : ℝ)
    (hn : n > 0) (hm : m > 0) (hct : 0 < c_target)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (W : ℝ) (hW : W = (∑ i ∈ contributing_bins n ℓ s_lo, (c i : ℝ)) / m)
    (h_exceeds : test_value n m c ℓ s_lo > c_target + 1 / m ^ 2 + 2 * W / m) :
    ∀ f : ℝ → ℝ, (∀ x, 0 ≤ f x) →
      Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4) →
      MeasureTheory.integral MeasureTheory.volume f = 1 →
      canonical_discretization f n m = c →
      autoconvolution_ratio f ≥ c_target := by
  intro f hf_nonneg hf_supp hf_int hdisc
  have hW' : W = (∑ i ∈ contributing_bins n ℓ s_lo, (canonical_discretization f n m i : ℝ)) / ↑m := by
    rw [hW]; congr 1; congr 1; ext i; rw [hdisc]
  have hbound := correction_term_bound n m hn hm f hf_nonneg hf_supp hf_int ℓ s_lo hℓ W hW'
  rw [hdisc] at hbound
  linarith

end
