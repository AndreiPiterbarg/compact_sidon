/-
PROMPT FOR ARISTOTLE: Prove x_cap bound and asymmetry no-margin (Claims 2.2 + 2.3).

CLAIM 2.3 (Single-bin energy cap):
  For any nonneg f on (-1/4, 1/4) with ∫f = 1, and any bin i with mass M_i = ∫_{bin_i} f:
    ‖f∗f‖∞ ≥ d · M_i²
  where d = 2n. This is a DIRECT L∞ bound — no discretization correction needed.

  Proof (same "restrict to sub-region" technique as asymmetry):
  1. Define f_i = f · 1_{bin_i}. Since f ≥ 0: f ≥ f_i ≥ 0.
  2. Convolution monotonicity: ‖f∗f‖∞ ≥ ‖f_i∗f_i‖∞.
  3. supp(f_i) ⊆ bin_i (width Δ = 1/(4n)), so supp(f_i∗f_i) has length 2Δ = 1/(2n).
  4. Fubini: ∫(f_i∗f_i) = M_i².
  5. Averaging: ‖f_i∗f_i‖∞ ≥ M_i²/(1/(2n)) = 2n·M_i² = d·M_i².

  Consequence: c_i ≥ m·√(c_target/d) ⟹ R(f) ≥ c_target.
  So x_cap = ⌊m·√(c_target/d)⌋.

CLAIM 2.2 (Asymmetry margin unnecessary, 3 facts):
  Fact 1: The discrete left_frac = (1/m)·∑_{i<n} c_i equals the continuous left-half
          mass L exactly, because x=0 falls on a bin boundary.
  Fact 2: Refinement preserves left-half mass (parent left sum = child left sum).
  Fact 3: The asymmetry bound ‖f∗f‖∞ ≥ 2L² is a direct L∞ bound that does NOT
          go through the test-value framework, so no correction term is needed.

The restriction machinery (f_restricted, f_ge_f_restricted, etc.) is already proved below.
These proofs share the same structure as the asymmetry bound (Claim 2.1) and can
reuse the same helper lemmas.

All definitions and foundational lemmas below are PROVED and can be used freely.
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

/-! ## Definitions (all previously established) -/

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

noncomputable def f_restricted (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) : ℝ → ℝ :=
  let δ := 1 / (4 * n : ℝ)
  let a := -(1/4 : ℝ) + i * δ
  let b := -(1/4 : ℝ) + (i + 1) * δ
  Set.indicator (Set.Ico a b) f

/-! ## Previously proved lemmas (use these freely) -/

theorem bin_masses_nonneg (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (n : ℕ) (i : Fin (2 * n)) :
    0 ≤ bin_masses f n i := by
      apply_rules [ MeasureTheory.integral_nonneg, Set.indicator_nonneg ] ; aesop

theorem f_restricted_integral (n : ℕ) (f : ℝ → ℝ) (i : Fin (2 * n)) :
    MeasureTheory.integral MeasureTheory.volume (f_restricted f n i) = bin_masses f n i := by
  unfold f_restricted bin_masses
  rfl

theorem f_ge_f_restricted (n : ℕ) (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) (i : Fin (2 * n)) :
    ∀ x, f x ≥ f_restricted f n i x ∧ f_restricted f n i x ≥ 0 := by
      intros x
      simp [f_restricted];
      simp [Set.indicator] at *; aesop;

theorem convolution_comm_real (f g : ℝ → ℝ) :
    MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
    MeasureTheory.convolution g f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := by
      ext x;
      have h_subst : ∀ {f g : ℝ → ℝ}, (∫ t, f t * g (x - t)) = (∫ t, g t * f (x - t)) := by
        intro f g; rw [ ← MeasureTheory.integral_sub_left_eq_self ] ; congr; ext; ring;
      exact h_subst

theorem f_has_compact_support (f : ℝ → ℝ) (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4)) :
    HasCompactSupport f := by
      have h_closure : closure (Function.support f) ⊆ Set.Icc (-1 / 4 : ℝ) (1 / 4) := by
        have h_closure_subset : closure (Function.support f) ⊆ closure (Set.Ioo (-1 / 4 : ℝ) (1 / 4)) := by
          apply closure_mono hf_supp;
        exact h_closure_subset.trans ( by rw [ closure_Ioo ( by norm_num ) ] );
      exact CompactIccSpace.isCompact_Icc.of_isClosed_subset ( isClosed_closure ) h_closure

theorem f_restricted_integrable (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n))
    (h_int : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.Integrable (f_restricted f n i) MeasureTheory.volume := by
  unfold f_restricted
  apply MeasureTheory.Integrable.indicator h_int measurableSet_Ico

/-! ## ============================================================
    CLAIM 2.3: Single-bin energy cap (x_cap)
    ============================================================ -/

/-
HELPER 1: Convolution monotonicity for nonneg functions.

If 0 ≤ g ≤ h pointwise, then (g∗g)(x) ≤ (h∗h)(x) for all x.

Proof idea: h = g + e where e = h-g ≥ 0, so
  h∗h = g∗g + g∗e + e∗g + e∗e
and each cross-term is a convolution of nonneg functions, hence nonneg.
-/
theorem convolution_mono_nonneg (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) (hfg : ∀ x, f x ≤ g x) :
    ∀ x, MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
         MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  sorry

/-
HELPER 2: Averaging principle.

For nonneg g supported on S with measure(S) = λ > 0:
  ‖g‖∞ ≥ (∫g) / λ

Proof: if g < (∫g)/λ a.e. on S, integrating gives ∫g < ∫g, contradiction.
-/
theorem averaging_principle (g : ℝ → ℝ) (hg_nonneg : ∀ x, 0 ≤ g x)
    (S : Set ℝ) (hS : Function.support g ⊆ S)
    (hS_meas : MeasureTheory.volume S = ENNReal.ofReal λ_val)
    (hλ : 0 < λ_val) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥
      MeasureTheory.integral MeasureTheory.volume g / λ_val := by
  sorry

/-
HELPER 3: Convolution integral = product of integrals (Fubini).
-/
theorem convolution_integral_eq (f g : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x) (hg_nonneg : ∀ x, 0 ≤ g x)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (hg_int : MeasureTheory.Integrable g MeasureTheory.volume) :
    MeasureTheory.integral MeasureTheory.volume
      (MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) =
    MeasureTheory.integral MeasureTheory.volume f *
    MeasureTheory.integral MeasureTheory.volume g := by
  sorry

/-
HELPER 4: Support of convolution of function restricted to an interval.

If supp(f_i) ⊆ [a, b) (a bin of width Δ = 1/(4n)), then
supp(f_i ∗ f_i) ⊆ [2a, 2b), which has length 2Δ = 1/(2n).
-/
theorem convolution_support_Ico (a b : ℝ) (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ico a b) :
    Function.support (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆
    Set.Ico (2 * a) (2 * b) := by
  sorry

/-
MAIN THEOREM (Claim 2.3): Single-bin energy cap.

For f ≥ 0 with supp(f) ⊆ (-1/4, 1/4), ∫f = 1, and any bin i:
  ‖f∗f‖∞ ≥ d · M_i²
where d = 2n and M_i = ∫_{bin_i} f = bin_masses f n i.

Proof:
1. f_i = f_restricted f n i. By f_ge_f_restricted: f ≥ f_i ≥ 0.
2. By convolution_mono_nonneg: ‖f∗f‖∞ ≥ ‖f_i∗f_i‖∞.
3. supp(f_i) ⊆ bin_i = [a, a+Δ) where Δ = 1/(4n).
   By convolution_support_Ico: supp(f_i∗f_i) ⊆ [2a, 2a+2Δ), length 2Δ = 1/(2n).
4. By convolution_integral_eq: ∫(f_i∗f_i) = (∫f_i)² = M_i².
   (using f_restricted_integral: ∫f_i = bin_masses f n i = M_i)
5. By averaging_principle with λ = 1/(2n):
   ‖f_i∗f_i‖∞ ≥ M_i² / (1/(2n)) = 2n · M_i² = d · M_i².

CONSEQUENCE: If c_i/m ≥ √(c_target/d), i.e. c_i ≥ m·√(c_target/d),
then ‖f∗f‖∞ ≥ d·(c_i/m)² ≥ c_target. So x_cap = ⌊m·√(c_target/d)⌋.
This is a DIRECT bound — no correction term 2/m + 1/m² needed.
-/
theorem single_bin_energy_cap (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (i : Fin (2 * n)) :
    let M_i := bin_masses f n i
    (MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume).toReal ≥ (2 * n : ℝ) * M_i ^ 2 := by
  sorry

/-! ## ============================================================
    CLAIM 2.2: Asymmetry margin is unnecessary (3 facts)
    ============================================================ -/

/-
FACT 1: The midpoint x = 0 falls exactly on a bin boundary.

Bin i covers [-1/4 + i·Δ, -1/4 + (i+1)·Δ) where Δ = 1/(4n).
At i = n: left endpoint = -1/4 + n/(4n) = -1/4 + 1/4 = 0.
So bin n-1 ends at 0 and bin n starts at 0. No bin straddles x = 0.

Therefore the discrete left_frac = (1/m)·∑_{i<n} c_i equals the
continuous left-half mass L = ∫_{-1/4}^{0} f exactly.
-/
theorem bin_boundary_at_zero (n : ℕ) (hn : n > 0) :
    (-1/4 : ℝ) + (n : ℝ) * (1 / (4 * (n : ℝ))) = 0 := by
  field_simp; ring

/-
FACT 2: Refinement preserves left-half mass.

Parent (c₀, ..., c_{d-1}) at resolution d = 2n. Child at resolution 2d = 4n
is (a₀, c₀-a₀, a₁, c₁-a₁, ...) where aᵢ + (cᵢ-aᵢ) = cᵢ.

Parent left half: bins 0..n-1, sum = ∑_{i<n} cᵢ.
Child left half: bins 0..2n-1. Child bins (2i, 2i+1) = (aᵢ, cᵢ-aᵢ).
So child left sum = ∑_{i<n} (aᵢ + cᵢ - aᵢ) = ∑_{i<n} cᵢ = parent left sum.

This holds for ALL choices of aᵢ, so the asymmetry check can be done
once per parent rather than per child.
-/
theorem refinement_preserves_left_sum (n : ℕ) (hn : n > 0)
    (parent : Fin (2 * n) → ℕ) (a : Fin (2 * n) → ℕ)
    (ha : ∀ i, a i ≤ parent i)
    (child : Fin (4 * n) → ℕ)
    (hc_even : ∀ i : Fin (2 * n), child ⟨2 * i.1, by omega⟩ = a i)
    (hc_odd : ∀ i : Fin (2 * n), child ⟨2 * i.1 + 1, by omega⟩ = parent i - a i) :
    ∑ j : Fin (2 * n), (child ⟨j.1, by omega⟩ : ℕ) =
    ∑ i : Fin n, (parent ⟨i.1, by omega⟩ : ℕ) := by
  sorry

/-
FACT 3: No correction term needed for asymmetry.

The bound ‖f∗f‖∞ ≥ 2L² (Claim 2.1, proved in prompt 4) works directly
with ‖f∗f‖∞, NOT via the test-value framework. Therefore the correction
term 2/m + 1/m² does NOT apply.

Combined with Fact 1 (left_frac = L exactly):
  left_frac ≥ √(c_target/2) ⟹ L ≥ √(c_target/2) ⟹ 2L² ≥ c_target ⟹ R(f) ≥ c_target.

No safety margin is needed.
-/
theorem asymmetry_threshold_exact (c_target : ℝ) (hct : 0 < c_target)
    (L : ℝ) (hL : L ≥ Real.sqrt (c_target / 2)) :
    2 * L ^ 2 ≥ c_target := by
  sorry

end
