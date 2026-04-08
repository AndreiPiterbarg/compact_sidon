/-
Sidon Autocorrelation Project — Final Result

The main theorem: autoconvolution_ratio f ≥ 7/5 for all admissible f.
Uses the computational axiom (cascade_all_pruned) plus all preceding theory.
-/

import Mathlib
import Sidon.Defs
import Sidon.Foundational
import Sidon.StepFunction
import Sidon.TestValueBounds
import Sidon.DiscretizationError

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
-- Final Result — Autoconvolution Constant Lower Bound
-- ═══════════════════════════════════════════════════════════════════════════════

-- *** COMPUTATIONAL AXIOM ***
-- The following axiom encodes the result of the branch-and-prune cascade.
-- Every other lemma and theorem in the Lean formalization is fully proved;
-- this axiom and cs_lemma3_per_window (C&S Lemma 3, in DiscretizationError.lean)
-- are the only unverified-in-Lean components.
--
-- The threshold uses the C&S Lemma 3 flat bound (2/m + 1/m²), which is a
-- pointwise bound on |(g*g)(x) - (f*f)(x)| that holds uniformly over all
-- window positions and lengths. This matches dynamic_threshold_sound_cs.
--
-- Reproduction: run the cascade with the flat C&S threshold:
--   python -m cloninger-steinerberger.cpu.run_cascade \
--     --n_half 2 --m 20 --c_target 1.40 --use_flat_threshold
-- and verify that every composition at d=128 has a killing window where
-- the test value exceeds c_target + 2/m + 1/m² = 7/5 + 2/20 + 1/400 = 1.5025.
--
-- The --use_flat_threshold flag ensures the CPU uses the same flat C&S
-- correction (2m+1)/m² as required by this axiom, rather than the tighter
-- W-refined correction (3+W_int/(2n))/m² used by default.
--
-- Fine grid (C&S B_{n,m}): compositions sum to S = 4nm = 4*64*20 = 5120.
-- Heights a_i = c_i/m are multiples of 1/m, giving ||ε||_∞ ≤ 1/m.
-- This parameterization matches the CPU code exactly.
/-- **Computational axiom**: The branch-and-prune cascade with parameters
    n_half=2, m=20, c_target=1.4 terminated with zero survivors.

    Fine grid: compositions of S=4nm=5120 into d=128 bins.
    For every such composition, there exists a window (ℓ, s_lo) where
    the discrete test value exceeds the C&S Lemma 3 flat threshold:
    c_target + 2/m + 1/m² = 7/5 + 2/20 + 1/400 = 1.5025.

    Reproduction:
      python -m cloninger-steinerberger.cpu.run_cascade \
        --n_half 2 --m 20 --c_target 1.40 --use_flat_threshold -/
axiom cascade_all_pruned :
  ∀ c : Fin (2 * 64) → ℕ, ∑ i, c i = 4 * 64 * 20 →
    ∃ ℓ s_lo, 2 ≤ ℓ ∧
      test_value 64 20 c ℓ s_lo > (7/5 : ℝ) + 2 / 20 + 1 / 20 ^ 2

/-- Scale invariance of the autoconvolution ratio.
    R(a·f) = R(f) for a > 0. -/
theorem autoconvolution_ratio_scale_invariant (f : ℝ → ℝ) (a : ℝ) (ha : 0 < a) :
    autoconvolution_ratio (fun x => a * f x) = autoconvolution_ratio f := by
  unfold autoconvolution_ratio
  dsimp only []
  have h_conv : MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
      fun x => a ^ 2 * MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
    ext x; simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply']
    simp only [mul_comm (a) (f _), ← mul_assoc]
    rw [← MeasureTheory.integral_const_mul]
    congr 1; ext t; ring
  rw [h_conv]
  have h_norm : (MeasureTheory.eLpNorm (fun x => a ^ 2 * MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x) ⊤ MeasureTheory.volume).toReal =
      a ^ 2 * (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal := by
    have ha2 : 0 < a ^ 2 := by positivity
    have ha2_ne : a ^ 2 ≠ 0 := ne_of_gt ha2
    have : (fun x => a ^ 2 * MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ)
        MeasureTheory.volume x) = a ^ 2 • (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) := by
      ext x; simp [Pi.smul_apply, smul_eq_mul]
    rw [this, MeasureTheory.eLpNorm_const_smul, ENNReal.toReal_mul,
        Real.enorm_eq_ofReal (le_of_lt ha2), ENNReal.toReal_ofReal (le_of_lt ha2)]
  have h_int : MeasureTheory.integral MeasureTheory.volume (fun x => a * f x) =
      a * MeasureTheory.integral MeasureTheory.volume f := by
    exact MeasureTheory.integral_const_mul a f
  rw [h_norm, h_int]
  have ha2 : a ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt ha)
  have ha_ne : a ≠ 0 := ne_of_gt ha
  field_simp [ha_ne, ha2]

/-- **Main theorem**: Every nonneg function f supported on (-1/4, 1/4) with positive
    integral and finite ‖f*f‖_∞ satisfies ‖f*f‖_∞ / (∫f)² ≥ 7/5 = 1.4.

    The hypothesis h_conv_fin is necessary because autoconvolution_ratio uses
    ENNReal.toReal, which maps ⊤ to 0. For f ∈ L¹ \ L² (e.g., f(x) ~ |x|^{-3/4}),
    ‖f*f‖_∞ = ∞ and the mathematical ratio is ∞ ≥ 7/5, but the Lean-computed ratio
    would be 0. This hypothesis holds for all bounded, L², or step functions.

    Proof: Normalize f to g with ∫g = 1, discretize g at resolution n=64 with m=20,
    apply cascade_all_pruned to find a killing window (ℓ, s_lo) where TV exceeds the
    C&S Lemma 3 threshold, then apply dynamic_threshold_sound_cs to conclude R(g) ≥ 7/5. -/
private lemma eLpNorm_convolution_scale_ne_top (f : ℝ → ℝ) (a : ℝ)
    (h_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    MeasureTheory.eLpNorm (MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ := by
  have h_eq : MeasureTheory.convolution (fun x => a * f x) (fun x => a * f x)
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume =
      a ^ 2 • MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume := by
    ext x; simp only [MeasureTheory.convolution, ContinuousLinearMap.mul_apply',
      Pi.smul_apply, smul_eq_mul]
    simp only [mul_comm a (f _), ← mul_assoc]
    rw [← MeasureTheory.integral_const_mul]; congr 1; ext t; ring
  rw [h_eq, MeasureTheory.eLpNorm_const_smul]
  exact ENNReal.mul_ne_top ENNReal.coe_ne_top h_fin

theorem autoconvolution_ratio_ge_7_5 (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ 7/5 := by
  set I := MeasureTheory.integral MeasureTheory.volume f with hI_def
  set g := fun x => (1/I) * f x with hg_def
  have hI_pos : 0 < I := hf_int_pos
  have h_ratio_eq : autoconvolution_ratio f = autoconvolution_ratio g := by
    rw [hg_def]
    exact (autoconvolution_ratio_scale_invariant f (1/I) (by positivity)).symm
  rw [h_ratio_eq]
  have hg_nonneg : ∀ x, 0 ≤ g x := by
    intro x; simp only [hg_def]; exact mul_nonneg (by positivity) (hf_nonneg x)
  have hg_supp : Function.support g ⊆ Set.Ioo (-1/4 : ℝ) (1/4) := by
    intro x hx; apply hf_supp; rw [Function.mem_support] at hx ⊢
    intro h; exact hx (by simp only [hg_def, h, mul_zero])
  have hg_int : MeasureTheory.integral MeasureTheory.volume g = 1 := by
    simp only [hg_def, MeasureTheory.integral_const_mul]
    rw [← hI_def]; exact div_mul_cancel₀ 1 (ne_of_gt hI_pos)
  set c := canonical_discretization g 64 20
  have h_mass_nz : ∑ j : Fin (2 * 64), bin_masses g 64 j ≠ 0 := by
    rw [sum_bin_masses_eq_one 64 (by norm_num) g hg_supp hg_int]; exact one_ne_zero
  have hc_sum : ∑ i, c i = 4 * 64 * 20 :=
    canonical_discretization_sum_eq_m g 64 20 (by norm_num) (by norm_num) h_mass_nz hg_nonneg
  obtain ⟨ℓ, s_lo, hℓ, h_exceeds⟩ := cascade_all_pruned c hc_sum
  have h_conv_fin_g : MeasureTheory.eLpNorm (MeasureTheory.convolution g g
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤ :=
    eLpNorm_convolution_scale_ne_top f (1/I) h_conv_fin
  exact dynamic_threshold_sound_cs 64 20 (7/5 : ℝ) (by norm_num) (by norm_num) (by norm_num : (0:ℝ) < 7/5)
    c ℓ s_lo hℓ h_exceeds g hg_nonneg hg_supp hg_int h_conv_fin_g rfl

end -- noncomputable section
