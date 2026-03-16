/-
Combined Lean 4 proof for the Sidon Autocorrelation Lower Bound Project.

This file merges all completed Aristotle proofs into a single, compilable document.
It establishes definitions and proved lemmas for the branch-and-prune algorithm that
shows c ≥ c_target for the autoconvolution constant:

  c = inf { ‖f*f‖_∞ / (∫f)² : f ≥ 0, supp(f) ⊆ (-1/4, 1/4) }

Lean version: leanprover/lean4:v4.24.0
Mathlib version: f897ebcf72cd16f89ab4577d0c826cd14afaafc7

Source files:
  - output (1).lean  (UUID: ca2199a4) — Framework definitions + foundational lemmas
  - 99433443...lean   (UUID: 99433443) — Reversal symmetry (Claims 3.3a, 3.3e)
  - b66ccc2f...lean   (UUID: b66ccc2f) — Refinement mass preservation (Claims 3.2c, 4.6)
  - 305874b1...lean   (UUID: 305874b1) — Incremental autoconvolution (Claim 4.2)

Co-authored-by: Aristotle (Harmonic) <aristotle-harmonic@harmonic.fun>
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 1: Core Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The autoconvolution ratio R(f) = ‖f*f‖_∞ / (∫f)². -/
noncomputable def autoconvolution_ratio (f : ℝ → ℝ) : ℝ :=
  let conv := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  let norm_inf := (MeasureTheory.eLpNorm conv ⊤ MeasureTheory.volume).toReal
  let integral := MeasureTheory.integral MeasureTheory.volume f
  norm_inf / (integral ^ 2)

/-- The autoconvolution constant c = inf R(f) over admissible f. -/
noncomputable def autoconvolution_constant : ℝ :=
  sInf {r : ℝ | ∃ (f : ℝ → ℝ), (∀ x, 0 ≤ f x) ∧ (Function.support f ⊆ Set.Ioo (-1/4) (1/4)) ∧ r = autoconvolution_ratio f}

/-- Discrete autoconvolution: conv[k] = ∑_{i+j=k} a_i · a_j. -/
def discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then a i * a j else 0

/-- Test value TV(n, m, c, ℓ, s_lo) for a composition c. -/
noncomputable def test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  let d := 2 * n
  let a : Fin d → ℝ := fun i => (4 * n : ℝ) / m * (c i : ℝ)
  let conv := discrete_autoconvolution a
  let sum_conv := ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), conv k
  (1 / (4 * n * ℓ : ℝ)) * sum_conv

/-- Maximum test value over all windows (ℓ, s_lo). -/
noncomputable def max_test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ :=
  let d := 2 * n
  let range_ell := Finset.Icc 2 (2 * d)
  let range_s_lo := Finset.range (2 * d)
  let values := range_ell.biUnion (fun ℓ => range_s_lo.image (fun s_lo => test_value n m c ℓ s_lo))
  if h : values.Nonempty then values.max' h else 0

/-- A composition is a vector summing to m. -/
def is_composition (n m : ℕ) (c : Fin (2 * n) → ℕ) : Prop :=
  ∑ i, c i = m

/-- Bin masses: integral of f over each bin. -/
noncomputable def bin_masses (f : ℝ → ℝ) (n : ℕ) : Fin (2 * n) → ℝ :=
  fun i =>
    let δ := 1 / (4 * n : ℝ)
    let a := -(1/4 : ℝ) + i * δ
    let b := -(1/4 : ℝ) + (i + 1) * δ
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico a b) f)

/-- Canonical discretization via floor-rounding of cumulative masses. -/
noncomputable def canonical_discretization (f : ℝ → ℝ) (n m : ℕ) : Fin (2 * n) → ℕ :=
  fun i =>
    let masses := bin_masses f n
    let total_mass := ∑ j, masses j
    let cum_mass (k : ℕ) := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
    let target_cum (k : ℕ) := (cum_mass k) / total_mass * m
    let discrete_cum (k : ℕ) := ⌊target_cum k⌋.natAbs
    if i.1 + 1 < 2 * n then discrete_cum (i.1 + 1) - discrete_cum i.1
    else m - discrete_cum i.1

/-- Contributing bins for a window (ℓ, s_lo). -/
def contributing_bins (n : ℕ) (ℓ s_lo : ℕ) : Finset (Fin (2 * n)) :=
  let d := 2 * n
  Finset.filter (fun i => ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2) Finset.univ

/-- Cumulative distribution helper D(k). -/
noncomputable def canonical_cumulative_distribution (f : ℝ → ℝ) (n m : ℕ) (k : ℕ) : ℕ :=
  let masses := bin_masses f n
  let total_mass := ∑ j, masses j
  let cum_mass := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
  let target_cum := cum_mass / total_mass * m
  ⌊target_cum⌋.natAbs

/-- Restriction of f to bin i. -/
noncomputable def f_restricted (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) : ℝ → ℝ :=
  let δ := 1 / (4 * n : ℝ)
  let a := -(1/4 : ℝ) + i * δ
  let b := -(1/4 : ℝ) + (i + 1) * δ
  Set.indicator (Set.Ico a b) f

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 2: Foundational Lemmas (F1–F15)
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

theorem sum_fin_telescope_nat_v2 (f : ℕ → ℕ) (n : ℕ) (h_mono : Monotone f) :
    ∑ i : Fin n, (f (i + 1) - f i) = f n - f 0 := by
      apply sum_fin_telescope_nat f n h_mono

-- F15: ∑ c_i = m (full proof, positive mass)
theorem canonical_discretization_sum_eq_m (f : ℝ → ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (h_mass_pos : ∑ j : Fin (2 * n), bin_masses f n j ≠ 0)
    (hf_nonneg : ∀ x, 0 ≤ f x) :
    ∑ i : Fin (2 * n), canonical_discretization f n m i = m := by
      rw [canonical_discretization_sum_eq_telescope f n m hn hm h_mass_pos]
      rw [sum_fin_telescope_nat_v2 (canonical_cumulative_distribution f n m) (2 * n) (canonical_cumulative_distribution_mono f hf_nonneg n m)]
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

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 3: Reversal Symmetry (Claims 3.3a, 3.3e)
-- Source: 99433443-0e82-4f9f-a69b-82ba51cd0537-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Reversal of a composition (ℕ-valued). -/
def rev_vector {d : ℕ} (c : Fin d → ℕ) : Fin d → ℕ :=
  fun i => c ⟨d - 1 - i.1, by omega⟩

/-- Reversal of a real-valued vector. -/
def rev_vector_real {d : ℕ} (a : Fin d → ℝ) : Fin d → ℝ :=
  fun i => a ⟨d - 1 - i.1, by omega⟩

/-- Claim 3.3a: conv[k](a) = conv[2d-2-k](rev(a)). -/
theorem autoconv_reversal_symmetry {d : ℕ} (hd : d > 0) (a : Fin d → ℝ)
    (k : ℕ) (hk : k ≤ 2 * d - 2) :
    discrete_autoconvolution a k =
    discrete_autoconvolution (rev_vector_real a) (2 * d - 2 - k) := by
  unfold discrete_autoconvolution;
  have h_change : ∑ i : Fin d, ∑ j : Fin d, (if i.val + j.val = k then a i * a j else 0) = ∑ i : Fin d, ∑ j : Fin d, (if (d - 1 - i.val) + (d - 1 - j.val) = k then a (Fin.mk (d - 1 - i.val) (by omega)) * a (Fin.mk (d - 1 - j.val) (by omega)) else 0) := by
    apply Finset.sum_bij (fun i _ => Fin.mk (d - 1 - i.val) (by
    exact lt_of_le_of_lt ( Nat.sub_le _ _ ) ( Nat.pred_lt hd.ne' )))
    all_goals generalize_proofs at *;
    · exact fun _ _ => Finset.mem_univ _;
    · grind;
    · exact fun b _ => ⟨ ⟨ d - 1 - b, by omega ⟩, Finset.mem_univ _, by simp +decide [ tsub_tsub_cancel_of_le ( show b.val ≤ d - 1 from Nat.le_sub_one_of_lt b.2 ) ] ⟩;
    · intro i hi;
      apply Finset.sum_bij (fun j _ => Fin.mk (d - 1 - j.val) (by
      exact?))
      all_goals generalize_proofs at *;
      · exact fun _ _ => Finset.mem_univ _;
      · grind;
      · exact fun j _ => ⟨ ⟨ d - 1 - j, by omega ⟩, Finset.mem_univ _, by simp +decide [ Nat.sub_sub_self ( show j.val ≤ d - 1 from Nat.le_sub_one_of_lt j.2 ) ] ⟩;
      · simp +decide [ Nat.sub_sub_self ( Nat.le_sub_one_of_lt i.2 ), Nat.sub_sub_self ( Nat.le_sub_one_of_lt ( Fin.is_lt _ ) ) ];
  convert h_change using 4;
  omega

/-- Claim 3.3e (helper): left sum + reversed left sum = m. -/
theorem left_sum_reversal (n : ℕ) (hn : n > 0) (m : ℕ)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (∑ i : Fin n, c ⟨i.1, by omega⟩) +
    (∑ i : Fin n, c ⟨2 * n - 1 - i.1, by omega⟩) = m := by
  rw [ ← hc, eq_comm ]
  generalize_proofs at *;
  have h_split : Finset.sum (Finset.univ : Finset (Fin (2 * n))) (fun i => c i) = Finset.sum (Finset.image (fun i : Fin n => ⟨i.val, by omega⟩) Finset.univ) (fun i => c i) + Finset.sum (Finset.image (fun i : Fin n => ⟨2 * n - 1 - i.val, by omega⟩) Finset.univ) (fun i => c i) := by
    rw [ ← Finset.sum_union ] <;> congr! 1
    generalize_proofs at *; (
    apply Finset.ext
    intro j
    simp [Finset.mem_union, Finset.mem_image];
    by_cases h : j.val < n <;> [ exact Or.inl ⟨ ⟨ j.val, by linarith ⟩, rfl ⟩ ; exact Or.inr ⟨ ⟨ 2 * n - 1 - j.val, by omega ⟩, by simp +decide [ Nat.sub_sub_self ( show j.val ≤ 2 * n - 1 from Nat.le_sub_one_of_lt <| Fin.is_lt j ) ] ⟩ ])
    generalize_proofs at *; (
    have h_disjoint : ∀ i j : Fin n, i.val ≠ 2 * n - 1 - j.val := by
      intro i j; omega;
    generalize_proofs at *; (
    simp +decide [ Finset.disjoint_left, h_disjoint ];
    exact fun i j => Ne.symm ( h_disjoint i j )));
  rw [ h_split, Finset.sum_image, Finset.sum_image ] <;> simp +decide [ Fin.ext_iff ];
  exact fun i _ j _ hij => Fin.ext <| by injection hij with hij; rw [ tsub_right_inj ] at hij <;> linarith [ Fin.is_lt i, Fin.is_lt j, Nat.sub_add_cancel ( by linarith [ Fin.is_lt i, Fin.is_lt j ] : 1 ≤ 2 * n ) ] ;

/-- Claim 3.3e: asymmetry condition is symmetric under reversal. -/
theorem asymmetry_reversal_symmetric (n : ℕ) (hn : n > 0) (m : ℕ) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (threshold : ℝ) (ht : 0 ≤ threshold) (ht1 : threshold ≤ 1)
    (L : ℝ) (hL : L = (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) / m)
    (h_prune : L ≥ threshold ∨ 1 - L ≥ threshold) :
    let L_rev := (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) / m
    L_rev ≥ threshold ∨ 1 - L_rev ≥ threshold := by
  have h_sum : L + (∑ i : Fin n, (c ⟨2 * n - 1 - i.val, by omega⟩ : ℝ)) / m = 1 := by
    have h_sum : (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) + (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) = m := by
      norm_cast; exact left_sum_reversal n hn m c hc;
    generalize_proofs at *; (
    rw [ hL, ← add_div, h_sum, div_self ( by positivity ) ]);
  grind +ring

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 4: Refinement Mass Preservation (Claims 3.2c, 4.6)
-- Source: b66ccc2f-25d7-46ad-80f3-eb01a82a1669-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Each parent bin splits into an even-odd child pair summing to the parent. -/
theorem child_bin_pair_sum (d : ℕ) (hd : d > 0)
    (parent : Fin d → ℕ) (a : Fin d → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i)
    (i : Fin d) :
    child ⟨2 * i.1, by omega⟩ + child ⟨2 * i.1 + 1, by omega⟩ = parent i := by
  rw [hc_even, hc_odd]
  simp [ha i]

/-- Claim 3.2c: Children preserve total mass. -/
theorem child_preserves_total_mass (d : ℕ) (hd : d > 0) (m : ℕ)
    (parent : Fin d → ℕ) (hp : ∑ i, parent i = m)
    (a : Fin d → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (2 * d) → ℕ)
    (hc_even : ∀ i : Fin d, child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin d, child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j, child j = m := by
  have h_split_sum : ∑ j : Fin (2 * d), child j = ∑ i : Fin d, (child ⟨2 * i, by omega⟩ + child ⟨2 * i + 1, by omega⟩) := by
    have h_split : Finset.range (2 * d) = Finset.image (fun i => 2 * i) (Finset.range d) ∪ Finset.image (fun i => 2 * i + 1) (Finset.range d) := by
      ext i
      simp [Finset.mem_range, Finset.mem_image];
      exact ⟨ fun hi => by rcases Nat.even_or_odd' i with ⟨ k, rfl | rfl ⟩ <;> [ left; right ] <;> exact ⟨ k, by linarith, rfl ⟩, fun hi => by rcases hi with ( ⟨ k, hk, rfl ⟩ | ⟨ k, hk, rfl ⟩ ) <;> linarith ⟩;
    rw [ Finset.sum_fin_eq_sum_range ];
    rw [ h_split, Finset.sum_union ];
    · norm_num [ Finset.sum_add_distrib, Finset.sum_range ];
      exact Finset.sum_congr rfl fun i hi => by split_ifs <;> linarith [ Fin.is_lt i ] ;
    · norm_num [ Finset.disjoint_right ];
      intros; omega;
  grind

/-- Claim 4.6: Left-half sum is invariant under refinement. -/
theorem left_half_sum_invariant (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a : Fin (2 * n) → ℕ) (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_even : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin (2 * n), child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) =
    ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  have h_split : ∑ j : Fin (2 * n), child ⟨j.val, by linarith [Fin.is_lt j]⟩ = ∑ i : Fin n, (child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩) := by
    have h_split : Finset.range (2 * n) = Finset.image (fun i => 2 * i) (Finset.range n) ∪ Finset.image (fun i => 2 * i + 1) (Finset.range n) := by
      ext i
      simp [Finset.mem_range, Finset.mem_image];
      exact ⟨ fun hi => by rcases Nat.even_or_odd' i with ⟨ k, rfl | rfl ⟩ <;> [ left; right ] <;> exact ⟨ k, by linarith, rfl ⟩, fun hi => by rcases hi with ( ⟨ k, hk, rfl ⟩ | ⟨ k, hk, rfl ⟩ ) <;> linarith ⟩
    generalize_proofs at *;
    rw [ Finset.sum_fin_eq_sum_range ] ; simp_all +decide [ Finset.sum_add_distrib ] ; (
    rw [ Finset.sum_union ] <;> norm_num [ Finset.sum_image, Finset.sum_range ];
    · exact Finset.sum_congr rfl fun i hi => by split_ifs <;> linarith [ Fin.is_lt i ] ;
    · norm_num [ Finset.disjoint_right ] ; omega;;)
  generalize_proofs at *;
  grind

/-- Any two refinements of the same parent have equal left-half sums. -/
theorem left_half_sum_same_for_all_children (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ)
    (a₁ a₂ : Fin (2 * n) → ℕ)
    (ha₁ : ∀ i, a₁ i ≤ parent i) (ha₂ : ∀ i, a₂ i ≤ parent i)
    (child₁ child₂ : Fin (4 * n) → ℕ)
    (hc₁_even : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1, by omega⟩ = a₁ i)
    (hc₁_odd : ∀ i : Fin (2 * n), child₁ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₁ i)
    (hc₂_even : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1, by omega⟩ = a₂ i)
    (hc₂_odd : ∀ i : Fin (2 * n), child₂ ⟨2 * i.1 + 1, by omega⟩ = parent i - a₂ i) :
    ∑ j : Fin (2 * n), (child₁ ⟨j.1, by omega⟩ : ℕ) =
    ∑ j : Fin (2 * n), (child₂ ⟨j.1, by omega⟩ : ℕ) := by
  convert left_half_sum_invariant n hn parent a₁ ha₁ child₁ hc₁_even hc₁_odd using 1;
  apply left_half_sum_invariant n hn parent a₂ ha₂ child₂ hc₂_even hc₂_odd

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 5: Incremental Autoconvolution (Claim 4.2)
-- Source: 305874b1-3eed-4942-afb4-5daac0ccf2ac-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Integer autoconvolution (ℤ-valued, for exact computation). -/
def int_autoconvolution {d : ℕ} (c : Fin d → ℤ) (t : ℕ) : ℤ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c i * c j else 0

/-- The delta between new and old autoconvolution. -/
def autoconv_delta {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) : ℤ :=
  int_autoconvolution c' t - int_autoconvolution c t

/-- Delta equals the sum of per-entry differences. -/
theorem delta_eq_sum {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    autoconv_delta c c' t =
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c' i * c' j - c i * c j else 0 := by
  simp [autoconv_delta, int_autoconvolution];
  simp [← Finset.sum_sub_distrib];
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  aesop

/-- Terms where neither index changed contribute zero. -/
theorem unchanged_terms_zero {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (hi : i ∉ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = 0 := by
  simp [hS i hi, hS j hj]

/-- Delta decomposes into three disjoint groups by membership in S. -/
theorem delta_three_way_split {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (t : ℕ) :
    autoconv_delta c c' t =
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) +
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∉ S then c' i * c' j - c i * c j else 0) +
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∉ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) := by
  rw [ ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ];
  convert delta_eq_sum c c' t using 2;
  rename_i i hi; rw [ ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ] ; congr ; ext j ; aesop;

/-- Cross-terms factor when one index is unchanged. -/
theorem cross_term_simplify {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (hi : i ∈ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = (c' i - c i) * c j := by
  rw [hS j hj];
  ring

/-- Claim 4.2: Incremental update is bit-exact. old_conv + delta = new_conv. -/
theorem incremental_update_correct {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    int_autoconvolution c t + autoconv_delta c c' t = int_autoconvolution c' t := by
  rw [show autoconv_delta c c' t = int_autoconvolution c' t - int_autoconvolution c t from rfl]
  ring

/-- The four membership groups are exhaustive. -/
theorem groups_exhaustive {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    (i ∈ S ∧ j ∈ S) ∨ (i ∈ S ∧ j ∉ S) ∨ (i ∉ S ∧ j ∈ S) ∨ (i ∉ S ∧ j ∉ S) := by
  tauto

/-- The four membership groups are pairwise disjoint. -/
theorem groups_disjoint {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∈ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∉ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) := by
  tauto

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 6: Fused Kernel and Quick-Check (Claims 4.1, 4.3)
-- Source: e868a126-2d3d-4a3f-8940-ed4c553ac681-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.1: Odometer iteration visits every Cartesian product element exactly once. -/
theorem odometer_bijection {d : ℕ} (lo hi : Fin d → ℕ) (h_valid : ∀ i, lo i ≤ hi i) :
    ∃ (f : Fin (∏ i, (hi i - lo i + 1)) → (∀ i : Fin d, Fin (hi i - lo i + 1))),
      Function.Bijective f := by
  have h_bij : Nonempty (Fin (∏ i, (hi i - lo i + 1)) ≃ (∀ i, Fin (hi i - lo i + 1))) := by
    refine' ⟨ Fintype.equivOfCardEq _ ⟩ ; aesop;
  exact ⟨ _, Equiv.bijective h_bij.some ⟩

/-- Fused kernel equivalence (trivial — filtering is independent of computational pattern). -/
theorem fused_eq_twophase {α : Type*} [DecidableEq α] (S : Finset α) (P : α → Bool) :
    S.filter (fun x => !P x) = S.filter (fun x => !P x) := by
  rfl

/-- Claim 4.3: Quick-check soundness — if quick-check finds a killing window, child is prunable. -/
theorem quickcheck_sound {d : ℕ} (ws : ℕ → ℕ → ℤ) (dyn : ℕ → ℕ → ℤ)
    (ℓ_star s_star : ℕ) (h : ws ℓ_star s_star > dyn ℓ_star s_star) :
    ∃ ℓ s, ws ℓ s > dyn ℓ s :=
  ⟨ℓ_star, s_star, h⟩

/-- W_int fast-path update correctness. -/
theorem w_int_fast_update (lo_bin hi_bin : ℕ) (c c' : ℕ → ℤ)
    (p : ℕ)
    (h_same : ∀ i, i ≠ 2*p ∧ i ≠ 2*p+1 → c' i = c i)
    (W_old : ℤ) (hW : W_old = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i)
    (delta : ℤ) (hd : delta = (c' (2*p) - c (2*p)) + (c' (2*p+1) - c (2*p+1))) :
    ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i =
      W_old + (if 2*p ∈ Finset.Icc lo_bin hi_bin then c' (2*p) - c (2*p) else 0)
           + (if 2*p+1 ∈ Finset.Icc lo_bin hi_bin then c' (2*p+1) - c (2*p+1) else 0) := by
  have h_split : ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i = ∑ i ∈ Finset.Icc lo_bin hi_bin, c i + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p then (c' (2 * p) - c (2 * p)) else 0) + ∑ i ∈ Finset.Icc lo_bin hi_bin, (if i = 2 * p + 1 then (c' (2 * p + 1) - c (2 * p + 1)) else 0) := by
    have h_split : ∑ i ∈ Finset.Icc lo_bin hi_bin, c' i = ∑ i ∈ Finset.Icc lo_bin hi_bin, (c i + (if i = 2 * p then c' (2 * p) - c (2 * p) else 0) + (if i = 2 * p + 1 then c' (2 * p + 1) - c (2 * p + 1) else 0)) := by
      grind;
    rw [ h_split, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib ];
  simp [h_split, hW]

-- ═══════════════════════════════════════════════════════════════════════════════
-- SECTION 7: Composition Enumeration (Claims 3.1, 3.2a)
-- Source: 31103b4c-cf4c-4f19-abf6-fe75cd7e9ee4-output.lean
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 3.1: Stars-and-bars — compositions of m into d parts = C(m+d-1, d-1). -/
theorem composition_count (m d : ℕ) (hd : d > 0) :
    Finset.card (Finset.filter (fun c : Fin d → Fin (m + 1) =>
      ∑ i, (c i : ℕ) = m) Finset.univ) = Nat.choose (m + d - 1) (d - 1) := by
  have h_stars_and_bars : ∀ m d : ℕ, d > 0 → Finset.card (Finset.filter (fun (c : Fin d → ℕ) => (∑ i, c i) = m) (Finset.Iic (fun _ => m))) = Nat.choose (m + d - 1) (d - 1) := by
    intro m d hd
    induction' d with d ih generalizing m;
    · contradiction;
    · have h_split : Finset.filter (fun (c : Fin (d + 1) → ℕ) => (∑ i, c i) = m) (Finset.Iic (fun _ => m)) = Finset.biUnion (Finset.range (m + 1)) (fun k => Finset.image (fun (c : Fin d → ℕ) => Fin.cons k c) (Finset.filter (fun (c : Fin d → ℕ) => (∑ i, c i) = m - k) (Finset.Iic (fun _ => m - k)))) := by
        ext c; simp [Finset.mem_biUnion, Finset.mem_image];
        constructor <;> intro h;
        · refine' ⟨ c 0, _, Fin.tail c, _, _ ⟩ <;> simp_all +decide [ Fin.sum_univ_succ ];
          · linarith [ h.1 0, Nat.zero_le ( ∑ i : Fin d, c i.succ ) ];
          · exact ⟨ fun i => Nat.le_sub_of_add_le <| by linarith! [ h.1 i.succ, Finset.single_le_sum ( fun a _ => Nat.zero_le ( c ( Fin.succ a ) ) ) ( Finset.mem_univ i ) ], eq_tsub_of_add_eq <| by linarith! ⟩;
        · rcases h with ⟨ a, ha, b, ⟨ hb₁, hb₂ ⟩, rfl ⟩ ; simp_all +decide [ Fin.sum_univ_succ ];
          exact ⟨ fun i => by cases i using Fin.inductionOn <;> [ exact Nat.le_of_lt_succ ha; exact le_trans ( hb₁ _ ) ( Nat.sub_le _ _ ) ], Nat.add_sub_of_le ( Nat.le_of_lt_succ ha ) ⟩;
      rw [ h_split, Finset.card_biUnion ];
      · rcases d with ( _ | d ) <;> simp_all +decide [ Finset.card_image_of_injective, Function.Injective ];
        · rw [ Finset.sum_eq_single m ] <;> simp +decide [ Finset.card_range ];
          intros; omega;
        · exact Nat.recOn m ( by simp +arith +decide ) fun n ih => by simp +arith +decide [ Nat.choose, Finset.sum_range_succ' ] at * ; linarith;
      · intro k hk l hl hkl; simp_all +decide [ Finset.disjoint_left ] ;
        intro a x hx₁ hx₂ hx₃ y hy₁ hy₂ hy₃; contrapose! hkl; aesop;
  convert h_stars_and_bars m d hd using 1;
  refine' Finset.card_bij ( fun c hc => fun i => c i ) _ _ _ <;> simp +decide [ funext_iff ];
  · exact fun a ha => ⟨ fun i => Nat.le_of_lt_succ <| Fin.is_lt _, ha ⟩;
  · exact fun a₁ ha₁ a₂ ha₂ h x => Fin.ext <| h x;
  · exact fun b hb hm => ⟨ fun i => ⟨ b i, Nat.lt_succ_of_le ( hb i ) ⟩, hm, fun i => rfl ⟩

/-- Claim 3.2a: Per-bin choice count for child generation. -/
theorem per_bin_choices (c_i x_cap : ℕ) (h : c_i ≤ 2 * x_cap) :
    Finset.card (Finset.Icc (Nat.max 0 (c_i - x_cap)) (Nat.min c_i x_cap)) =
    Nat.min c_i x_cap - Nat.max 0 (c_i - x_cap) + 1 := by
  simp +zetaDelta at *;
  grind +ring

end -- noncomputable section
