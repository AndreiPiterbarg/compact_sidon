/-
Sidon Autocorrelation Project — f-independent facts about `K_ms`.
=================================================================

This file collects rigorous, `f`-independent facts about the
multi-scale arcsine kernel `K_ms` and its closed-form period-`u`
Fourier coefficients, used by the constructor that discharges the
`ExtremiserPrimitives` bundle.

Proved here (all `sorry`-free, no new axioms):

  * `Ktilde_ms_eq_fourier_lattice` — the reconciliation identity
    `Ktilde_ms j = K_ms_fourier_lattice (j : ℤ)` (Target 5): the two
    closed forms agree, differing only in the grouping
    `π·j·δ/u`  vs  `π·δ·(j/u)`.
  * `besselJ0_ge_one_sub_sq` — the alternating-series lower bound
    `J₀(z) ≥ 1 - (z/2)²` for `|z| ≤ 2`.
  * `besselJ0_pos_of_lt_two` — `J₀(z) > 0` for `|z| < √2 · √2` … in
    practice for the small arguments `π·δᵢ/u` (`δ₁` is largest, and
    `π·δ₁/u ≈ 0.68`).
  * `Ktilde_ms_one_pos` — `Ktilde_ms 1 > 0`.
  * `K_ms_fourier_lattice_one_pos` — `K_ms_fourier_lattice 1 > 0`.
  * `S_1_analytic_pos` — `S_1_analytic > 0` (Target 3).

Blocked targets (see report): the *unconditional* `MemLp K_ms 2 volume`
(Target 1) and its corollary `Integrable (K_ms²)` (Target 2) require an
`Lᵖ` Young convolution inequality or a Bessel `J₀⁴ ∈ L¹` theory, neither
of which exists in mathlib `v4.29.1`.  The fully-general lattice
positivity `∀ j ≠ 0, 0 < K_ms_fourier_lattice j` (part of Target 4)
requires the irrationality / non-coincidence of the `J₀` zero sets,
also absent from mathlib; the finite active-index form is what the
downstream `S_1_analytic` consumer actually needs and is proved here.

No `sorry`, no new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.BundleEq1
import Sidon.BundleEq2Schwartz

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical
open MeasureTheory

namespace Sidon.Constructor

open Sidon.MultiScale

noncomputable section

/-! ## Target 5 — reconciliation of `Ktilde_ms` and `K_ms_fourier_lattice`

`Ktilde_ms (j : ℕ) = ∑ᵢ λᵢ · J₀(π·j·δᵢ/u)²`
`K_ms_fourier_lattice (j : ℤ) = ∑ᵢ λᵢ · J₀(π·δᵢ·(j/u))²`

These are *the same value*: the Bessel arguments agree by `ring`
(`π·j·δᵢ/u = π·δᵢ·(j/u)`), so the only content is a coercion
`((j : ℕ) : ℤ) : ℝ) = ((j : ℕ) : ℝ)`. -/

/-- **Reconciliation identity (Target 5).** The two closed-form
period-`u` Fourier coefficients of `K_ms` agree:
`Ktilde_ms j = K_ms_fourier_lattice (j : ℤ)` for every `j : ℕ`.
The defining Bessel arguments differ only in grouping
(`π · j · δᵢ / u` vs. `π · δᵢ · (j / u)`), which is closed by `ring`. -/
theorem Ktilde_ms_eq_fourier_lattice (j : ℕ) :
    Ktilde_ms j = K_ms_fourier_lattice (j : ℤ) := by
  unfold Ktilde_ms K_ms_fourier_lattice
  have e1 : Real.pi * (j : ℝ) * delta1 / uQ_real
              = Real.pi * delta1 * (((j : ℤ) : ℝ) / uQ_real) := by
    push_cast; ring
  have e2 : Real.pi * (j : ℝ) * delta2 / uQ_real
              = Real.pi * delta2 * (((j : ℤ) : ℝ) / uQ_real) := by
    push_cast; ring
  have e3 : Real.pi * (j : ℝ) * delta3 / uQ_real
              = Real.pi * delta3 * (((j : ℤ) : ℝ) / uQ_real) := by
    push_cast; ring
  rw [e1, e2, e3]

/-! ## A clean alternating-series lower bound for `J₀`

For `|z| ≤ 2` the series `J₀(z) = ∑ₖ (-1)ᵏ gₖ`, with
`gₖ = (z/2)^{2k}/(k!)²`, has `g` antitone (since
`g_{k+1}/g_k = (z/2)²/(k+1)² ≤ 1`), so by the alternating-series
estimate the even partial sum `g₀ - g₁ = 1 - (z/2)²` is a lower bound
for the limit.  This is enough to certify `J₀ > 0` at the small
arguments `π·δᵢ/u`. -/

/-- The nonnegative magnitude factor of the `k`-th Bessel term:
`gₖ(z) = (z/2)^{2k} / (k!)²`, so that `besselJ0Term z k = (-1)ᵏ · gₖ(z)`. -/
private def bMag (z : ℝ) (k : ℕ) : ℝ := (z / 2) ^ (2 * k) / (k.factorial : ℝ) ^ 2

private lemma besselJ0Term_eq (z : ℝ) (k : ℕ) :
    Sidon.Bessel.besselJ0Term z k = (-1) ^ k * bMag z k := by
  unfold Sidon.Bessel.besselJ0Term bMag
  ring

private lemma bMag_nonneg (z : ℝ) (k : ℕ) : 0 ≤ bMag z k := by
  unfold bMag
  apply div_nonneg
  · rw [pow_mul]; positivity
  · positivity

/-- `bMag z` is summable (same majorant as `besselJ0_summable`). -/
private lemma summable_bMag (z : ℝ) : Summable (bMag z) := by
  refine Summable.of_nonneg_of_le (bMag_nonneg z)
    (f := fun k : ℕ => ((z / 2) ^ 2) ^ k / (k.factorial : ℝ))
    (fun k => ?_) (Real.summable_pow_div_factorial _)
  unfold bMag
  have hk : (z / 2) ^ (2 * k) = ((z / 2) ^ 2) ^ k := by rw [pow_mul]
  rw [hk]
  have h_kfact_pos : (0 : ℝ) < (k.factorial : ℝ) := by
    exact_mod_cast Nat.factorial_pos k
  have h_one_le : (1 : ℝ) ≤ (k.factorial : ℝ) := by
    exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.factorial_pos k).ne'
  have hbase : (0 : ℝ) ≤ ((z / 2) ^ 2) ^ k := by positivity
  rw [div_le_div_iff₀ (by positivity) h_kfact_pos]
  nlinarith [hbase, sq_nonneg ((k.factorial : ℝ) - 1), h_one_le, h_kfact_pos,
    mul_nonneg hbase (sub_nonneg.mpr h_one_le)]

/-- For `|z| ≤ 2`, the magnitude sequence `bMag z` is antitone:
`g_{k+1}/g_k = (z/2)²/(k+1)² ≤ 1`. -/
private lemma antitone_bMag {z : ℝ} (hz : |z| ≤ 2) : Antitone (bMag z) := by
  have hzsq : z ^ 2 ≤ 4 := by nlinarith [sq_abs z, abs_nonneg z, hz]
  have hz2 : (z / 2) ^ 2 ≤ 1 := by rw [div_pow]; nlinarith [hzsq]
  refine antitone_nat_of_succ_le (fun k => ?_)
  unfold bMag
  have hk0 : (0 : ℝ) < (k.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos _
  -- (z/2)^(2(k+1)) / ((k+1)!)² ≤ (z/2)^(2k) / (k!)².
  have hfact : ((k + 1).factorial : ℝ) = ((k : ℝ) + 1) * (k.factorial : ℝ) := by
    rw [Nat.factorial_succ]; push_cast; ring
  have hpow : (z / 2) ^ (2 * (k + 1)) = (z / 2) ^ (2 * k) * (z / 2) ^ 2 := by
    rw [show 2 * (k + 1) = 2 * k + 2 by ring, pow_add]
  rw [hpow, hfact]
  have hbase : (0 : ℝ) ≤ (z / 2) ^ (2 * k) := by rw [pow_mul]; positivity
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  have hkp1 : (1 : ℝ) ≤ ((k : ℝ) + 1) ^ 2 := by nlinarith [Nat.cast_nonneg (α := ℝ) k]
  calc (z / 2) ^ (2 * k) * (z / 2) ^ 2 * (k.factorial : ℝ) ^ 2
      ≤ (z / 2) ^ (2 * k) * 1 * (k.factorial : ℝ) ^ 2 := by
        apply mul_le_mul_of_nonneg_right _ (by positivity)
        exact mul_le_mul_of_nonneg_left hz2 hbase
    _ ≤ (z / 2) ^ (2 * k) * (((k : ℝ) + 1) * (k.factorial : ℝ)) ^ 2 := by
        rw [mul_pow]
        nlinarith [hbase, sq_nonneg (k.factorial : ℝ), hkp1,
          mul_nonneg hbase (sq_nonneg (k.factorial : ℝ))]

/-- **Alternating-series lower bound for `J₀`.**  For `|z| ≤ 2`,
`J₀(z) ≥ 1 - (z/2)²`. -/
theorem besselJ0_ge_one_sub_sq {z : ℝ} (hz : |z| ≤ 2) :
    1 - (z / 2) ^ 2 ≤ Sidon.Bessel.besselJ0 z := by
  -- Rewrite J₀ as an alternating series ∑ (-1)^k bMag z k.
  have hsum : Sidon.Bessel.besselJ0 z = ∑' k : ℕ, (-1) ^ k * bMag z k := by
    unfold Sidon.Bessel.besselJ0
    exact tsum_congr (besselJ0Term_eq z)
  rw [hsum]
  -- Apply the alternating-series even-partial-sum lower bound at k = 1.
  have htends := (summable_bMag z).tendsto_alternating_series_tsum
  have hbound := (antitone_bMag hz).alternating_series_le_tendsto htends 1
  -- ∑ i ∈ range 2, (-1)^i bMag z i = bMag z 0 - bMag z 1.
  have hpart : ∑ i ∈ Finset.range (2 * 1), (-1) ^ i * bMag z i
                 = 1 - (z / 2) ^ 2 := by
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, mul_one]
    unfold bMag
    norm_num
    ring
  rw [hpart] at hbound
  exact hbound

/-- For `|z| < √2`, `J₀(z) > 0` (since `1 - (z/2)² > 0` whenever
`(z/2)² < 1`, i.e. `|z| < 2`; we only need the small-argument regime). -/
theorem besselJ0_pos_of_abs_lt_two {z : ℝ} (hz : |z| < 2) :
    0 < Sidon.Bessel.besselJ0 z := by
  have hzle : |z| ≤ 2 := le_of_lt hz
  have hlb := besselJ0_ge_one_sub_sq hzle
  have hzsq : z ^ 2 < 4 := by nlinarith [sq_abs z, abs_nonneg z, hz]
  have hsq : (z / 2) ^ 2 < 1 := by rw [div_pow]; nlinarith [hzsq]
  linarith

/-! ## `Ktilde_ms 1 > 0` and `K_ms_fourier_lattice 1 > 0` (active-index Target 4) -/

/-- The first-scale argument at `j = 1` lies in `(-2, 2)`:
`|π·δ₁/u| < 2`.  (`π·0.138/0.638 ≈ 0.680`.) -/
private lemma abs_arg_delta1_lt_two :
    |Real.pi * (1 : ℝ) * delta1 / uQ_real| < 2 := by
  have hπ : Real.pi < 3.15 := Real.pi_lt_d2
  have hπ0 : (0 : ℝ) < Real.pi := Real.pi_pos
  have hd1 : delta1 = 138 / 1000 := by unfold delta1 delta1Q; push_cast; ring
  have hu : uQ_real = 638 / 1000 := by unfold uQ_real uQ; push_cast; ring
  rw [hd1, hu]
  rw [abs_of_pos (by positivity)]
  rw [div_lt_iff₀ (by norm_num)]
  nlinarith [hπ, hπ0]

/-- `Ktilde_ms 1 > 0`.  Each summand `λᵢ·J₀(π·δᵢ/u)²` is `≥ 0`, and the
first-scale summand is `> 0` because `J₀(π·δ₁/u) > 0` (small argument)
and `λ₁ > 0`. -/
theorem Ktilde_ms_one_pos : 0 < Ktilde_ms 1 := by
  unfold Ktilde_ms
  have hl1 : (0 : ℝ) < lambda1 := by
    unfold lambda1 lambda1Q; push_cast; norm_num
  have hl2 : (0 : ℝ) ≤ lambda2 := by
    unfold lambda2 lambda2Q; push_cast; norm_num
  have hl3 : (0 : ℝ) ≤ lambda3 := by
    unfold lambda3 lambda3Q; push_cast; norm_num
  -- The δ₁ Bessel factor is strictly positive.
  have hJ1 : 0 < Sidon.Bessel.besselJ0 (Real.pi * (1 : ℝ) * delta1 / uQ_real) :=
    besselJ0_pos_of_abs_lt_two abs_arg_delta1_lt_two
  have hterm1 : (0 : ℝ) <
      lambda1 * (Sidon.Bessel.besselJ0 (Real.pi * (1 : ℝ) * delta1 / uQ_real)) ^ 2 :=
    mul_pos hl1 (by positivity)
  have hterm2 : (0 : ℝ) ≤
      lambda2 * (Sidon.Bessel.besselJ0 (Real.pi * (1 : ℝ) * delta2 / uQ_real)) ^ 2 :=
    mul_nonneg hl2 (sq_nonneg _)
  have hterm3 : (0 : ℝ) ≤
      lambda3 * (Sidon.Bessel.besselJ0 (Real.pi * (1 : ℝ) * delta3 / uQ_real)) ^ 2 :=
    mul_nonneg hl3 (sq_nonneg _)
  -- Ktilde_ms's `(j : ℝ)` at `j = 1` is `(1 : ℝ)`.
  have hcast : ((1 : ℕ) : ℝ) = (1 : ℝ) := by norm_num
  rw [hcast]
  linarith

/-- `K_ms_fourier_lattice 1 > 0` (the integer-indexed form), via the
reconciliation identity and `Ktilde_ms_one_pos`. -/
theorem K_ms_fourier_lattice_one_pos : 0 < K_ms_fourier_lattice 1 := by
  have h := Ktilde_ms_one_pos
  rw [Ktilde_ms_eq_fourier_lattice 1, Nat.cast_one] at h
  exact h

/-! ## Target 3 — `S_1_analytic > 0`

`S_1_analytic = ∑_{i<200} (aᵢ/10⁸)² / Ktilde_ms (i+1)`.  In Lean,
division by a nonnegative denominator of a nonnegative numerator is
`≥ 0`, and `Ktilde_ms (i+1) ≥ 0` (Bochner positivity), so every
summand is `≥ 0`.  The `i = 0` summand is *strictly* positive because
`a₀ = 190807542 ≠ 0` and `Ktilde_ms 1 > 0`.  Hence the whole sum is
`> 0`. -/

/-- `Ktilde_ms j ≥ 0` for every `j` (Bochner positivity of the convex
combination of `J₀²`). -/
theorem Ktilde_ms_nonneg (j : ℕ) : 0 ≤ Ktilde_ms j := by
  have h := K_ms_fourier_lattice_nonneg (j : ℤ)
  rw [← Ktilde_ms_eq_fourier_lattice j] at h
  exact h

/-- **`S_1_analytic > 0` (Target 3).** -/
theorem S_1_analytic_pos : 0 < S_1_analytic := by
  unfold S_1_analytic
  -- Each summand is ≥ 0.
  have hnonneg : ∀ i ∈ Finset.range 200,
      (0 : ℝ) ≤ (((qpNumerators.getD i 0 : ℝ) / 100000000)) ^ 2 / Ktilde_ms (i + 1) := by
    intro i _
    exact div_nonneg (sq_nonneg _) (Ktilde_ms_nonneg (i + 1))
  -- The i = 0 summand is > 0.
  have h0mem : (0 : ℕ) ∈ Finset.range 200 := by norm_num
  have ha0 : (qpNumerators.getD 0 0 : ℝ) = 190807542 := by
    have hz : (qpNumerators.getD 0 0 : ℤ) = 190807542 := by rfl
    rw [hz]; norm_num
  have hterm0 : (0 : ℝ) <
      (((qpNumerators.getD 0 0 : ℝ) / 100000000)) ^ 2 / Ktilde_ms (0 + 1) := by
    rw [ha0]
    apply div_pos
    · norm_num
    · simpa using Ktilde_ms_one_pos
  -- Sum of nonneg terms with one strictly positive ⇒ sum > 0.
  exact Finset.sum_pos' hnonneg ⟨0, h0mem, hterm0⟩

end

end Sidon.Constructor
