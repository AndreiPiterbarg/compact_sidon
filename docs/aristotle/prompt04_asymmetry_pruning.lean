/-
PROMPT FOR ARISTOTLE: Prove the asymmetry pruning bound (Claim 2.1).

GOAL: Prove that for any nonneg f supported on [-1/4, 1/4] with ∫f = 1 and
left-half mass L = ∫_{-1/4}^{0} f, we have ‖f∗f‖∞ ≥ 2L².

This is used to prune compositions with asymmetric mass in a branch-and-prune
proof that the autoconvolution constant c ≥ 1.4.

PROOF STRATEGY (4 steps):
1. Restrict f to its left half: f_L = f · 1_{(-1/4, 0)}.
   Since f ≥ f_L ≥ 0 pointwise, convolution monotonicity gives ‖f∗f‖∞ ≥ ‖f_L∗f_L‖∞.
2. supp(f_L∗f_L) ⊆ (-1/2, 0), which has length 1/2.
3. By Fubini: ∫(f_L∗f_L) = (∫f_L)² = L².
4. By averaging: ‖f_L∗f_L‖∞ ≥ L²/(1/2) = 2L².

The restriction machinery (f_restricted, f_ge_f_restricted, etc.) is already proved below.
The main gaps to fill are:
  (a) convolution monotonicity for nonneg functions
  (b) the averaging principle (‖g‖∞ ≥ (∫g)/measure(support))
  (c) combining everything into the main theorem

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
    THEOREMS TO PROVE (fill in the sorry's)
    ============================================================ -/

/-
THEOREM 1: Convolution monotonicity for nonneg functions.

If 0 ≤ g ≤ h pointwise, then (g∗g)(x) ≤ (h∗h)(x) for all x.

Proof idea:
  (h∗h)(x) - (g∗g)(x) = ∫ [h(t)h(x-t) - g(t)g(x-t)] dt
  = ∫ [(h(t)-g(t))·h(x-t) + g(t)·(h(x-t)-g(x-t))] dt ≥ 0
since each factor is ≥ 0.

Alternatively, use that for nonneg functions:
  h = g + (h-g) where h-g ≥ 0
  h∗h = g∗g + g∗(h-g) + (h-g)∗g + (h-g)∗(h-g)
and each cross term is nonneg.
-/
theorem convolution_mono_nonneg (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) (hfg : ∀ x, f x ≤ g x) :
    ∀ x, MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
         MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  sorry

/-
THEOREM 2: Averaging principle.

For any nonneg measurable g supported on a set S of finite measure λ:
  ‖g‖_{L∞} ≥ (∫g) / λ

Proof: If g(x) < (∫g)/λ a.e. on S, then ∫g < (λ/λ)·∫g = ∫g, contradiction.
-/
theorem averaging_principle (g : ℝ → ℝ) (hg_nonneg : ∀ x, 0 ≤ g x)
    (S : Set ℝ) (hS : Function.support g ⊆ S)
    (hS_meas : MeasureTheory.volume S = ENNReal.ofReal λ_val)
    (hλ : 0 < λ_val) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥
      MeasureTheory.integral MeasureTheory.volume g / λ_val := by
  sorry

/-
THEOREM 3: Integral of convolution of nonneg functions = product of integrals.

For nonneg integrable f, g with compact support:
  ∫(f∗g) = (∫f)·(∫g)

This is Fubini's theorem applied to convolution.
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
THEOREM 4: Convolution of nonneg functions is nonneg.
-/
theorem convolution_nonneg (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) :
    ∀ x, 0 ≤ MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  sorry

/-
THEOREM 5: Support of convolution.

If supp(f) ⊆ [a₁, b₁] and supp(g) ⊆ [a₂, b₂], then
supp(f∗g) ⊆ [a₁+a₂, b₁+b₂].

Specifically: if supp(f) ⊆ (-1/4, 0), then supp(f∗f) ⊆ (-1/2, 0).
-/
theorem convolution_support_subset (f : ℝ → ℝ)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) 0) :
    Function.support (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆
    Set.Ioo (-1/2 : ℝ) 0 := by
  sorry

/-
MAIN THEOREM (Claim 2.1): Asymmetry bound.

For f ≥ 0 with supp(f) ⊆ [-1/4, 1/4] and ∫f = 1,
let L = ∫_{-1/4}^{0} f (left-half mass). Then:

  ‖f∗f‖_{L∞} ≥ 2L²

Proof combines the above:
1. Define f_L = f · 1_{(-1/4, 0)}. We have f ≥ f_L ≥ 0.
2. By convolution_mono_nonneg: ‖f∗f‖∞ ≥ ‖f_L∗f_L‖∞.
3. By convolution_support_subset: supp(f_L∗f_L) ⊆ (-1/2, 0), length 1/2.
4. By convolution_integral_eq: ∫(f_L∗f_L) = L².
5. By averaging_principle with λ = 1/2: ‖f_L∗f_L‖∞ ≥ L²/(1/2) = 2L².

CONSEQUENCE: If L ≥ √(c_target/2), then R(f) = ‖f∗f‖∞ ≥ 2L² ≥ c_target.
By symmetry (f(x) → f(-x)), same holds if 1-L ≥ √(c_target/2).
-/
theorem asymmetry_bound (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Icc (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    let L := MeasureTheory.integral MeasureTheory.volume
               (Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f)
    (MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume).toReal ≥ 2 * L ^ 2 := by
  sorry

end
