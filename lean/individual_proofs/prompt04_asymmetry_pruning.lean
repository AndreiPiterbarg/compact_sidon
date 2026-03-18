/-
PROMPT FOR ARISTOTLE: Prove the asymmetry pruning bound (Claim 2.1).

GOAL: Prove `asymmetry_bound` — for nonneg f on [-1/4, 1/4] with ∫f = 1 and
left-half mass L = ∫_{-1/4}^{0} f, we have ‖f∗f‖∞ ≥ 2L².

All helper lemmas below are PROVED and can be used freely.
The ONLY sorry is `asymmetry_bound` at the bottom.

PROOF STRATEGY:
1. f_L = f · 1_{(-1/4,0)}. Since f ≥ f_L ≥ 0: ‖f∗f‖∞ ≥ ‖f_L∗f_L‖∞ (by convolution_mono_ae).
2. supp(f_L∗f_L) ⊆ (-1/2, 0) (by f_L_conv_supp), length 1/2.
3. ∫(f_L∗f_L) = L² (by integral_convolution_square + f_L_integrable).
4. averaging_principle: ‖f_L∗f_L‖∞ ≥ L²/(1/2) = 2L².

KEY GAP: Convert ENNReal averaging_principle to a toReal bound on ‖f∗f‖∞.
Need finiteness of eLpNorm (given as hypothesis h_bdd).
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

/-- Left-half restriction of f (indicator on (-1/4, 0)). -/
def f_L (f : ℝ → ℝ) : ℝ → ℝ := Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f

/-! ## Proved helper lemmas (use freely) -/

theorem f_L_le_f (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) :
    ∀ x, f_L f x ≤ f x := by
  intros x
  simp [f_L];
  by_cases hx : x ∈ Set.Ioo (-1 / 4 : ℝ) 0 <;> simp [hx, hf]

theorem f_L_supp (f : ℝ → ℝ) :
    Function.support (f_L f) ⊆ Set.Ioo (-1/4 : ℝ) 0 := by
  simp [f_L]

theorem f_L_conv_supp (f : ℝ → ℝ) :
    Function.support (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊆
    Set.Ioo (-1/2 : ℝ) 0 := by
  intro x hx; simp_all +decide [ MeasureTheory.convolution ] ;
  have h_support : ∀ t, f_L f t ≠ 0 → -1 / 4 < t ∧ t < 0 := by
    unfold f_L; aesop;
  contrapose! hx;
  rw [ MeasureTheory.integral_eq_zero_of_ae ];
  filter_upwards [ ] with t ; by_cases ht : f_L f t = 0 <;> by_cases ht' : f_L f ( x - t ) = 0 <;> simp_all +decide [ sub_eq_add_neg ];
  linarith [ h_support t ht, h_support ( x + -t ) ht', hx ( by linarith [ h_support t ht, h_support ( x + -t ) ht' ] ) ]

theorem f_L_nonneg (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x) :
    ∀ x, 0 ≤ f_L f x := by
  exact fun x => Set.indicator_nonneg ( fun _ _ => hf _ ) _

theorem f_L_integrable (f : ℝ → ℝ) (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.Integrable (f_L f) MeasureTheory.volume := by
  convert hf.indicator measurableSet_Ioo using 1

theorem convolution_nonneg_pointwise {f g : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) :
    ∀ x, 0 ≤ MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  intro x
  simp [MeasureTheory.convolution];
  exact MeasureTheory.integral_nonneg fun t => mul_nonneg ( hf t ) ( hg ( x - t ) )

/-- Integral of convolution = (integral)². -/
theorem integral_convolution_square (f : ℝ → ℝ)
    (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) =
    (MeasureTheory.integral MeasureTheory.volume f) ^ 2 := by
  rw [ sq ];
  apply MeasureTheory.integral_convolution;
  · exact hf;
  · exact hf

theorem volume_Ioo_half :
    MeasureTheory.volume (Set.Ioo (-1/2 : ℝ) 0) = ENNReal.ofReal (1/2) := by
  norm_num

/-
PROBLEM
Monotonicity of convolution for nonneg functions (a.e. version).

PROVIDED SOLUTION
For a.e. x, we need (f*f)(x) ≤ (g*g)(x). Since g is L^1, by Fubini/Tonelli, the integrand t ↦ g(t)*g(x-t) is integrable for a.e. x. For those x where g(t)*g(x-t) is integrable, apply MeasureTheory.integral_mono_of_nonneg to get ∫ f(t)*f(x-t) dt ≤ ∫ g(t)*g(x-t) dt, using:
- ae nonneg: f(t)*f(x-t) ≥ 0
- integrability of upper bound: g(t)*g(x-t) is integrable at this x
- ae bound: f(t)*f(x-t) ≤ g(t)*g(x-t) from hfg

To get the a.e. integrability of g(t)*g(x-t): use MeasureTheory.Integrable.mul_prod (or similar) to show (t,x) ↦ g(t)*g(x) is integrable on the product, then change variables to get (t,x) ↦ g(t)*g(x-t) integrable on product, then by MeasureTheory.integrable_prod_iff get a.e. x integrability.

Key API: MeasureTheory.Integrable.convolution_integrand gives exactly the a.e. integrability we need. Or use MeasureTheory.ConvolutionExistsAt.

Alternatively: filter_upwards with the a.e. set from Fubini, then on each element of that set, apply integral_mono_of_nonneg.
-/
theorem convolution_mono_ae (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) (hfg : ∀ x, f x ≤ g x)
    (hg_int : MeasureTheory.Integrable g MeasureTheory.volume) :
    ∀ᵐ x ∂MeasureTheory.volume,
      MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
      MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  have h_conv_integrable : ∀ᵐ x ∂MeasureTheory.volume, MeasureTheory.Integrable (fun t => g t * g (x - t)) MeasureTheory.volume := by
    have h_conv_integrable : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => g p.1 * g p.2) (MeasureTheory.volume.prod MeasureTheory.volume) := by
      exact MeasureTheory.Integrable.mul_prod hg_int hg_int;
    have h_conv_integrable : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => g p.1 * g (p.2 - p.1)) (MeasureTheory.volume.prod MeasureTheory.volume) := by
      have h_conv_integrable : MeasureTheory.MeasurePreserving (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.volume.prod MeasureTheory.volume) (MeasureTheory.volume.prod MeasureTheory.volume) := by
        exact MeasureTheory.measurePreserving_prod_sub MeasureTheory.volume MeasureTheory.volume;
      have h_conv_integrable : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => g p.1 * g p.2) (MeasureTheory.Measure.map (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.volume.prod MeasureTheory.volume)) := by
        rw [ h_conv_integrable.map_eq ] ; assumption;
      rw [ MeasureTheory.integrable_map_measure ] at h_conv_integrable ; aesop;
      · exact h_conv_integrable.1;
      · exact AEMeasurable.prodMk ( measurable_fst.aemeasurable ) ( measurable_snd.sub measurable_fst |> Measurable.aemeasurable );
    rw [ MeasureTheory.integrable_prod_iff' ] at h_conv_integrable ; aesop;
    exact h_conv_integrable.1;
  filter_upwards [ h_conv_integrable ] with x hx;
  apply_rules [ MeasureTheory.integral_mono_of_nonneg ];
  · exact Filter.Eventually.of_forall fun t => mul_nonneg ( hf t ) ( hf ( x - t ) );
  · filter_upwards [ ] with t using mul_le_mul ( hfg t ) ( hfg ( x - t ) ) ( hf _ ) ( hg _ )

/-- Averaging principle: ‖g‖∞ ≥ (∫g) / measure(support). -/
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

/-! ## ============================================================
    THEOREM TO PROVE (fill in the sorry)
    ============================================================ -/

/-
PROBLEM
MAIN THEOREM (Claim 2.1): Asymmetry bound.

For f ≥ 0 with supp(f) ⊆ [-1/4, 1/4] and ∫f = 1,
let L = ∫_{-1/4}^{0} f. Then ‖f∗f‖_{L∞} ≥ 2L².

PROOF PLAN (use a.e. monotonicity + eLpNorm_mono_ae):
1. Let conv_fL = (f_L f) * (f_L f). conv_fL is nonneg a.e. (convolution_nonneg_pointwise).
2. By convolution_mono_ae with f_L ≤ f: conv_fL(x) ≤ (f*f)(x) a.e.
3. So ‖conv_fL(x)‖ ≤ ‖(f*f)(x)‖ a.e., hence eLpNorm(conv_fL, ∞) ≤ eLpNorm(f*f, ∞) by eLpNorm_mono_ae.
4. supp(conv_fL) ⊆ (-1/2, 0) by f_L_conv_supp.
5. conv_fL is integrable by MeasureTheory.Integrable.integrable_convolution.
6. ∫ conv_fL = L² by integral_convolution_square + f_L_integrable.
7. Apply averaging_principle to conv_fL with S = Ioo(-1/2, 0), v = 1/2.
8. Chain and take toReal.

PROVIDED SOLUTION
Proof plan:
1. Let fL = Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f. Note L = ∫ fL.
2. fL is nonneg (f_L_nonneg or Set.indicator_nonneg), fL ≤ f (f_L_le_f), fL integrable (f_L_integrable).
3. Let conv_fL = convolution fL fL. It's nonneg (convolution_nonneg_pointwise).
4. conv_fL is integrable: MeasureTheory.Integrable.integrable_convolution applied to fL.
5. By convolution_mono_ae with fL ≤ f: conv_fL(x) ≤ (f*f)(x) a.e.
6. Both nonneg, so ‖conv_fL(x)‖ ≤ ‖(f*f)(x)‖ a.e. Apply eLpNorm_mono_ae to get eLpNorm(conv_fL, ∞) ≤ eLpNorm(f*f, ∞).
7. supp(conv_fL) ⊆ Ioo(-1/2, 0) by f_L_conv_supp.
8. ∫ conv_fL = L² by integral_convolution_square.
9. averaging_principle with S = Ioo(-1/2,0), v = 1/2: eLpNorm(conv_fL, ∞) ≥ ENNReal.ofReal(L²/(1/2)) = ENNReal.ofReal(2L²).
10. Chain: eLpNorm(f*f, ∞) ≥ eLpNorm(conv_fL, ∞) ≥ ENNReal.ofReal(2L²).
11. toReal both sides: use ENNReal.toReal_mono (h_bdd) and ENNReal.toReal_ofReal (2*L²≥0).
Result: eLpNorm(f*f,∞).toReal ≥ 2*L².
-/
theorem asymmetry_bound (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Icc (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1)
    (h_bdd : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) :
    let L := MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f)
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ 2 * L ^ 2 := by
  -- By the properties of the convolution and the L^p norm, we have that the convolution of f_L with itself is bounded by the convolution of f with itself.
  have h_conv : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≤ MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume := by
    have h_conv_le : ∀ᵐ x ∂MeasureTheory.volume, MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤ MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
      apply convolution_mono_ae (f_L f) f (fun x => f_L_nonneg f hf_nonneg x) hf_nonneg (fun x => f_L_le_f f hf_nonneg x) (by
      exact MeasureTheory.integrable_of_integral_eq_one hf_int);
    apply_rules [ MeasureTheory.eLpNorm_mono_ae ];
    filter_upwards [ h_conv_le ] with x hx using by rw [ Real.norm_of_nonneg ( convolution_nonneg_pointwise ( f_L_nonneg f hf_nonneg ) ( f_L_nonneg f hf_nonneg ) x ), Real.norm_of_nonneg ( convolution_nonneg_pointwise ( hf_nonneg ) ( hf_nonneg ) x ) ] ; exact hx;
  -- By the properties of the convolution and the L^p norm, we have that the convolution of f_L with itself is bounded by the convolution of f with itself, and its integral is (MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2.
  have h_integral : MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) = (MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 := by
    apply integral_convolution_square; exact f_L_integrable f (MeasureTheory.integrable_of_integral_eq_one hf_int);
  -- Apply the averaging principle to the convolution of f_L with itself.
  have h_avg : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal ((MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 / (1 / 2)) := by
    -- Apply the averaging principle to the convolution of f_L with itself, noting that its support is contained in (-1/2, 0).
    have h_avg : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal ((MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution (f_L f) (f_L f) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)) / (1 / 2)) := by
      apply_rules [ averaging_principle ];
      any_goals exact Set.Ioo ( -1 / 2 ) 0;
      · apply_rules [ convolution_nonneg_pointwise, f_L_nonneg ];
      · apply_rules [ MeasureTheory.Integrable.integrable_convolution, f_L_integrable ];
        · exact MeasureTheory.integrable_of_integral_eq_one hf_int;
        · exact MeasureTheory.integrable_of_integral_eq_one hf_int;
      · convert f_L_conv_supp f using 1;
      · norm_num;
      · norm_num;
    aesop;
  -- By combining the results from h_avg and h_conv, we conclude the proof.
  have h_final : (MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ (MeasureTheory.integral MeasureTheory.volume (f_L f)) ^ 2 / (1 / 2) := by
    refine' le_trans _ ( ENNReal.toReal_mono _ <| h_avg.trans h_conv );
    · rw [ ENNReal.toReal_ofReal ( by positivity ) ];
    · assumption;
  convert h_final using 1 ; ring!

end