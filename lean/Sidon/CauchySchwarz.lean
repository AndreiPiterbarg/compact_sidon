/-
Sidon Autocorrelation Project — Cauchy-Schwarz Single-Bin Bound

Integer safety, ell scan order, and the Cauchy-Schwarz single-bin bound
(Claims 4.5, 4.7, 4.8).
-/

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
-- Integer Safety, Ell Scan Order, Cauchy-Schwarz (Claims 4.5, 4.7, 4.8)
-- Source: output (20).lean (UUID: b6236cec) — FULLY PROVED
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The i-th bin interval [-(1/4) + i·δ, -(1/4) + (i+1)·δ). -/
noncomputable def bin_interval (n : ℕ) (i : Fin (2 * n)) : Set ℝ :=
  let δ := 1 / (4 * n : ℝ)
  let a := -(1/4 : ℝ) + i * δ
  let b := -(1/4 : ℝ) + (i + 1) * δ
  Set.Ico a b

/-- Restriction of f to bin i using bin_interval. -/
noncomputable def f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) : ℝ → ℝ :=
  Set.indicator (bin_interval n i) f

lemma f_bin_le_f (f : ℝ → ℝ) (hf : 0 ≤ f) (n : ℕ) (i : Fin (2 * n)) :
  f_bin f n i ≤ f := by
    have h_f_bin_def : ∀ x, f_bin f n i x = if x ∈ bin_interval n i then f x else 0 := by
      simp [f_bin, Set.indicator];
    intros x
    simp [h_f_bin_def];
    aesop

lemma f_bin_nonneg (f : ℝ → ℝ) (hf : 0 ≤ f) (n : ℕ) (i : Fin (2 * n)) :
  0 ≤ f_bin f n i := by
    apply Set.indicator_nonneg; intro x hx; exact hf x

lemma integral_f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) :
  MeasureTheory.integral MeasureTheory.volume (f_bin f n i) = bin_masses f n i := by
    have h_integral : MeasureTheory.integral MeasureTheory.MeasureSpace.volume (f_bin f n i) = MeasureTheory.integral MeasureTheory.MeasureSpace.volume (Set.indicator (bin_interval n i) f) := by
      simp [f_bin];
    convert h_integral using 1

lemma f_bin_integrable (f : ℝ → ℝ) (hf : MeasureTheory.Integrable f MeasureTheory.volume) (n : ℕ) (i : Fin (2 * n)) :
  MeasureTheory.Integrable (f_bin f n i) MeasureTheory.volume := by
  refine' MeasureTheory.Integrable.mono' _ _ _;
  refine' fun x => |f x|;
  · exact hf.norm;
  · exact MeasureTheory.AEStronglyMeasurable.indicator ( hf.1 ) ( measurableSet_Ico );
  · filter_upwards [ ] with x using by rw [ f_bin ] ; rw [ Set.indicator_apply ] ; aesop;

lemma convolution_mono_pointwise (f g : ℝ → ℝ) (hf : 0 ≤ f) (hg : 0 ≤ g) (hfg : g ≤ f)
    (x : ℝ)
    (h_int_f : MeasureTheory.Integrable (fun t => f t * f (x - t)) MeasureTheory.volume) :
    MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
    MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  rw [MeasureTheory.convolution, MeasureTheory.convolution]
  simp only [ContinuousLinearMap.mul_apply']
  by_cases h_int_g : MeasureTheory.Integrable (fun t => g t * g (x - t)) MeasureTheory.volume
  · apply MeasureTheory.integral_mono h_int_g h_int_f
    intro t
    apply mul_le_mul (hfg t) (hfg (x - t)) (hg (x - t)) (hf t)
  · rw [MeasureTheory.integral_undef h_int_g]
    apply MeasureTheory.integral_nonneg
    intro t
    apply mul_nonneg (hf t) (hf (x - t))

lemma support_f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) :
  Function.support (f_bin f n i) ⊆ bin_interval n i := by
    simp [f_bin, Function.support]

lemma measure_support_convolution_f_bin (f : ℝ → ℝ) (n : ℕ) (_hn : n > 0) (i : Fin (2 * n)) :
  MeasureTheory.volume (Function.support (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)) ≤ ENNReal.ofReal (1 / (2 * n : ℝ)) := by
  have h_support_convolution : ∀ x, x ∉ (bin_interval n i + bin_interval n i) → (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.MeasureSpace.volume) x = 0 := by
    intro x hx
    have h_no_t : ∀ t, ¬(t ∈ bin_interval n i ∧ x - t ∈ bin_interval n i) := by
      exact fun t ht => hx <| Set.add_mem_add ht.1 ht.2 |> fun h => by simpa [ add_comm ] using h;
    simp_all +decide [ MeasureTheory.convolution, f_bin ];
    rw [ MeasureTheory.integral_eq_zero_of_ae ] ; filter_upwards [ ] with t ; by_cases ht : t ∈ bin_interval n i <;> by_cases ht' : x - t ∈ bin_interval n i <;> simp_all +decide [ Set.indicator ] ;
  refine' le_trans ( MeasureTheory.measure_mono ( show Function.support ( MeasureTheory.convolution ( f_bin f n i ) ( f_bin f n i ) ( ContinuousLinearMap.mul ℝ ℝ ) MeasureTheory.MeasureSpace.volume ) ⊆ bin_interval n i + bin_interval n i from fun x hx => by contrapose! hx; aesop ) ) _;
  have h_support_convolution : bin_interval n i + bin_interval n i ⊆ Set.Ico (-(1/4 : ℝ) + i * (1 / (4 * n : ℝ)) + (-(1/4 : ℝ) + i * (1 / (4 * n : ℝ)))) (-(1/4 : ℝ) + (i + 1) * (1 / (4 * n : ℝ)) + (-(1/4 : ℝ) + (i + 1) * (1 / (4 * n : ℝ)))) := by
    intro x hx; obtain ⟨ a, ha, b, hb, rfl ⟩ := hx; constructor <;> linarith [ Set.mem_Ico.mp ha, Set.mem_Ico.mp hb ] ;
  refine' le_trans ( MeasureTheory.measure_mono h_support_convolution ) _ ; norm_num ; ring ; norm_num [ _hn.ne' ]

lemma integral_convolution_f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n))
    (hf : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.integral MeasureTheory.volume (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) = (bin_masses f n i) ^ 2 := by
      have h_integral : ∫ x, MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ∂MeasureTheory.volume = (∫ x, f_bin f n i x ∂MeasureTheory.volume) * (∫ x, f_bin f n i x ∂MeasureTheory.volume) := by
        have h_convolution : ∀ (f g : ℝ → ℝ), MeasureTheory.Integrable f MeasureTheory.volume → MeasureTheory.Integrable g MeasureTheory.volume → ∫ x, MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x = (∫ x, f x) * (∫ x, g x) := by
          intro f g hf hg; rw [ MeasureTheory.integral_convolution ] ; aesop;
          · exact hf;
          · exact hg;
        exact h_convolution _ _ ( f_bin_integrable f hf n i ) ( f_bin_integrable f hf n i );
      rw [ h_integral, sq, integral_f_bin ]

lemma convolution_mono_ae_fbin (f : ℝ → ℝ) (hf : 0 ≤ f) (n : ℕ) (i : Fin (2 * n))
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume) :
    ∀ᵐ x ∂MeasureTheory.volume,
      MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
      MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  have h_bin_le : f_bin f n i ≤ f := f_bin_le_f f hf n i
  have h_bin_nonneg : 0 ≤ f_bin f n i := f_bin_nonneg f hf n i
  have h_bin_int : MeasureTheory.Integrable (f_bin f n i) MeasureTheory.volume := f_bin_integrable f hf_int n i
  have h_conv_exists_f : ∀ᵐ x ∂MeasureTheory.volume, MeasureTheory.Integrable (fun t => f t * f (x - t)) MeasureTheory.volume := by
    have h_int_f : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => f p.1 * f p.2) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
      exact hf_int.mul_prod hf_int;
    have h_int_f : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => f p.1 * f (p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
      have h_int_f : MeasureTheory.MeasurePreserving (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume) := by
        exact MeasureTheory.measurePreserving_prod_sub MeasureTheory.volume MeasureTheory.volume;
      have h_int_f : MeasureTheory.Integrable (fun (p : ℝ × ℝ) => f p.1 * f p.2) (MeasureTheory.Measure.map (fun p : ℝ × ℝ => (p.1, p.2 - p.1)) (MeasureTheory.Measure.prod MeasureTheory.volume MeasureTheory.volume)) := by
        rw [ h_int_f.map_eq ] ; assumption;
      convert h_int_f.comp_measurable ( measurable_fst.prodMk ( measurable_snd.sub measurable_fst ) ) using 1;
    rw [ MeasureTheory.integrable_prod_iff' ] at h_int_f ; aesop;
    exact h_int_f.1
  filter_upwards [h_conv_exists_f] with x hx
  apply convolution_mono_pointwise f (f_bin f n i) hf h_bin_nonneg h_bin_le x hx

lemma lintegral_convolution_f_bin (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n))
    (hf : MeasureTheory.Integrable f MeasureTheory.volume)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (M_i : ℝ) (hM : M_i = bin_masses f n i) :
    MeasureTheory.lintegral MeasureTheory.volume (fun x => ENNReal.ofReal (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x)) = ENNReal.ofReal (M_i^2) := by
  rw [← MeasureTheory.ofReal_integral_eq_lintegral_ofReal]
  · rw [integral_convolution_f_bin f n i hf, hM]
  · apply MeasureTheory.Integrable.integrable_convolution (ContinuousLinearMap.mul ℝ ℝ) (f_bin_integrable f hf n i) (f_bin_integrable f hf n i)
  · filter_upwards [] with x
    rw [MeasureTheory.convolution]
    apply MeasureTheory.integral_nonneg
    intro t
    simp only [ContinuousLinearMap.mul_apply']
    apply mul_nonneg
    · apply f_bin_nonneg f hf_nonneg
    · apply f_bin_nonneg f hf_nonneg

lemma lintegral_le_norm_mul_vol (g : ℝ → ℝ) (hg : 0 ≤ g)
    (S : Set ℝ) (hS : Function.support g ⊆ S)
    (_hS_meas : MeasurableSet S) :
    MeasureTheory.lintegral MeasureTheory.volume (fun x => ENNReal.ofReal (g x)) ≤
    MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume * MeasureTheory.volume S := by
      by_cases hS_finite : MeasureTheory.volume S < ⊤;
      · have h_supp : Function.support (fun x => ENNReal.ofReal (g x)) ⊆ S := by
          intro x hx
          rw [Function.mem_support] at hx
          apply hS
          rw [Function.mem_support]
          intro h; exact hx (by simp [h])
        calc ∫⁻ x, ENNReal.ofReal (g x)
            = ∫⁻ x in S, ENNReal.ofReal (g x) :=
              (MeasureTheory.setLIntegral_eq_of_support_subset h_supp).symm
          _ ≤ ∫⁻ _ in S, MeasureTheory.eLpNormEssSup g MeasureTheory.volume := by
              apply MeasureTheory.setLIntegral_mono_ae measurable_const.aemeasurable
              filter_upwards [MeasureTheory.enorm_ae_le_eLpNormEssSup g MeasureTheory.volume] with x hx
              intro _
              rw [← Real.enorm_eq_ofReal (hg x)]
              exact hx
          _ = MeasureTheory.eLpNormEssSup g MeasureTheory.volume * MeasureTheory.volume S := by
              rw [MeasureTheory.setLIntegral_const]
          _ = MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume * MeasureTheory.volume S := by
              rw [MeasureTheory.eLpNorm_exponent_top]
      · by_cases h : MeasureTheory.eLpNorm g ⊤ MeasureTheory.MeasureSpace.volume = 0 <;> simp_all +decide;
        rw [ MeasureTheory.lintegral_congr_ae ( h.mono fun x hx => by aesop ) ] ; norm_num

lemma single_bin_bound_ennreal (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (_hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM : M_i = bin_masses f n i)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume) :
    MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥
    ENNReal.ofReal ((2 * n : ℝ) * M_i ^ 2) := by
      have h_ess_sup_i : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume * ENNReal.ofReal (1 / (2 * n)) ≥ ENNReal.ofReal (M_i^2) := by
        have h_avg_principle : MeasureTheory.eLpNorm (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume * ENNReal.ofReal (1 / (2 * n : ℝ)) ≥ ENNReal.ofReal (M_i^2) := by
          have h_ess_sup_i : ∫⁻ x, ENNReal.ofReal (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x) ∂MeasureTheory.volume ≤ MeasureTheory.eLpNorm (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume * ENNReal.ofReal (1 / (2 * n : ℝ)) := by
            convert lintegral_le_norm_mul_vol _ _ _ _ _ using 1;
            rotate_left;
            rotate_left;
            exact Set.Ico ( - ( 1 / 4 ) + i * ( 1 / ( 4 * n ) ) + ( - ( 1 / 4 ) + i * ( 1 / ( 4 * n ) ) ) ) ( - ( 1 / 4 ) + ( i + 1 ) * ( 1 / ( 4 * n ) ) + ( - ( 1 / 4 ) + ( i + 1 ) * ( 1 / ( 4 * n ) ) ) );
            · intro x hx;
              contrapose! hx; simp_all +decide [ MeasureTheory.convolution ] ;
              rw [ MeasureTheory.integral_eq_zero_of_ae ];
              filter_upwards [ ] with t ; by_cases ht : f_bin f n i t = 0 <;> by_cases ht' : f_bin f n i ( x - t ) = 0 <;> simp_all +decide [ f_bin ];
              unfold bin_interval at * ; norm_num at * ; linarith [ hx ( by linarith ) ] ;
            · exact measurableSet_Ico;
            · norm_num [ add_assoc, mul_add, add_mul, mul_comm, mul_left_comm, hn.ne' ];
              rw [ ← ENNReal.ofReal_mul ( by positivity ) ] ; ring;
            · intro x; exact (by
              refine' MeasureTheory.integral_nonneg fun t => mul_nonneg ( f_bin_nonneg f hf n i t ) ( f_bin_nonneg f hf n i ( x - t ) ));
          refine' le_trans _ h_ess_sup_i;
          rw [ lintegral_convolution_f_bin ] <;> aesop;
        have h_mono : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ MeasureTheory.eLpNorm (MeasureTheory.convolution (f_bin f n i) (f_bin f n i) (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume := by
          apply_rules [ MeasureTheory.eLpNorm_mono_ae ];
          filter_upwards [ convolution_mono_ae_fbin f hf n i hf_int ] with x hx using by rw [ Real.norm_of_nonneg ( show 0 ≤ MeasureTheory.convolution ( f_bin f n i ) ( f_bin f n i ) ( ContinuousLinearMap.mul ℝ ℝ ) MeasureTheory.MeasureSpace.volume x from by
                                                                                                                refine' MeasureTheory.integral_nonneg fun t => _;
                                                                                                                simp [f_bin];
                                                                                                                exact mul_nonneg ( by rw [ Set.indicator_apply ] ; aesop ) ( by rw [ Set.indicator_apply ] ; aesop ) ), Real.norm_of_nonneg ( show 0 ≤ MeasureTheory.convolution f f ( ContinuousLinearMap.mul ℝ ℝ ) MeasureTheory.MeasureSpace.volume x from by
                                                                                                                                                                                                                                                                                              exact MeasureTheory.integral_nonneg fun y => mul_nonneg ( hf _ ) ( hf _ ) ) ] ; exact hx;
        exact h_avg_principle.trans ( mul_le_mul_right' h_mono _ );
      have h_ess_sup_i : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal (M_i^2) * ENNReal.ofReal (2 * n) := by
        refine' le_trans ( mul_le_mul_right' h_ess_sup_i _ ) _;
        rw [ mul_assoc, ← ENNReal.ofReal_mul ( by positivity ), one_div_mul_cancel ( by positivity ), ENNReal.ofReal_one, mul_one ];
      convert h_ess_sup_i using 1 ; rw [ mul_comm, ENNReal.ofReal_mul ( by positivity ) ]

/-- Claim 4.5: Cauchy-Schwarz single-bin bound — ‖f*f‖∞ ≥ d · M_i². PROVED via output (20). -/
theorem single_bin_bound (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM : M_i = bin_masses f n i)
    (hf_int : MeasureTheory.Integrable f MeasureTheory.volume)
    (h_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume < ⊤) :
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥
    (2 * n : ℝ) * M_i ^ 2 := by
      have h_ennreal : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≥ ENNReal.ofReal ((2 * n : ℝ) * M_i ^ 2) := by
        apply single_bin_bound_ennreal n hn f hf hf_supp i M_i hM hf_int;
      have h_real : (MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ (ENNReal.ofReal ((2 * n : ℝ) * M_i ^ 2)).toReal := by
        apply ENNReal.toReal_mono h_fin.ne h_ennreal;
      rwa [ ENNReal.toReal_ofReal ( by positivity ) ] at h_real

-- Claim 4.8: conv[k] ≤ m² (each entry bounded by total)
-- Source: output (14).lean (UUID: 124a8efc) — PROVED
theorem conv_entry_le_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, (if i.1+j.1=k then c i * c j else 0) ≤ m ^ 2 := by
  have h_sum_bound : ∑ i : Fin d, ∑ j : Fin d, (if i.val + j.val = k then c i * c j else 0) ≤ ∑ i : Fin d, ∑ j : Fin d, c i * c j := by
    exact Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun j hj => by split_ifs <;> nlinarith;
  simp_all +decide [ ← Finset.mul_sum _ _ _, ← Finset.sum_mul, sq ]

-- Total autoconvolution = m²
-- Source: output (14).lean (UUID: 124a8efc) — PROVED
theorem conv_total {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) :
    ∑ k ∈ Finset.range (2*d-1),
      (∑ i : Fin d, ∑ j : Fin d, if i.1+j.1=k then c i * c j else 0) = m ^ 2 := by
  have h_fubini : ∑ k ∈ Finset.range (2 * d - 1), ∑ i : Fin d, ∑ j : Fin d, (if i + j = k then c i * c j else 0) = ∑ i : Fin d, ∑ j : Fin d, ∑ k ∈ Finset.range (2 * d - 1), (if i + j = k then c i * c j else 0) := by
    exact Finset.sum_comm.trans ( Finset.sum_congr rfl fun _ _ => Finset.sum_comm );
  simp_all +decide [ Finset.sum_ite, sq ];
  have h_filter : ∀ i j : Fin d, i.val + j.val < 2 * d - 1 := by
    exact fun i j => lt_tsub_iff_left.mpr ( by linarith [ Fin.is_lt i, Fin.is_lt j ] );
  simp +decide [ ← hc, h_filter, Finset.sum_mul _ _ _ ];
  simp +decide only [Finset.mul_sum _ _ _]

-- m² fits int32 for m ≤ 200
theorem int32_safe (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by
  have : m * m ≤ 200 * 200 := Nat.mul_le_mul hm hm
  norm_num [Nat.pow_succ] at *; omega

end -- noncomputable section
