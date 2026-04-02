/-
Sidon Autocorrelation Project — Univariate Sweep Skip (Claims 4.36–4.48)

This file collects ALL the theorems and lemmas that must be proved to
certify the univariate sweep skip optimization in the Gray code kernel
`_fused_generate_and_prune_gray` (run_cascade.py).

The optimization works as follows: when the full window scan finds a
killing window (ell, s_lo) for a child produced by advancing digit 0,
and digit 0 did NOT just hit its boundary, we check whether the killing
window also prunes ALL remaining children in digit 0's current sweep.
This is a 1D quadratic range check: the window sum ws(x) and threshold
dyn_x(x) are both polynomial in the cursor value x, so their difference
D(x) = ws(x) − dyn_x(x) is a degree-2 polynomial. If min D > 0 on the
remaining range, the entire sweep is provably prunable and digit 0 is
fast-forwarded to its boundary in O(d) instead of testing each remaining
child individually.

Critical constraint: the sweep skip is ONLY valid for digit 0 (the
innermost Gray code digit). After any non-boundary advance, gc_focus[0]
is reset to 0, so the NEXT advance always picks digit 0. Only digit 0
advances in consecutive steps; higher digits interleave with digit 0's
sweeps and change other child bins, invalidating the 1D quadratic.

STATUS: All 14 theorems proved. 0 sorry.

AUDIT FIXES (2026-03-29):
- Claims 4.36, 4.39: Strengthened from vacuously true (A=0, B=0 witnesses)
  to properly parameterized versions using child_param. The universal
  quantifier ∀ x now sits INSIDE ∃ A B C, proving the coefficients are
  independent of x — capturing the true quadratic/affine dependence.
- Added helper lemmas: child_param, child_param_affine, affine_mul_affine,
  sum_quadratic_finset, sum_quadratic_univ, autoconv_quadratic_in_x,
  sum_affine_finset.
-/

import Mathlib
import Sidon.Defs
import Sidon.IncrementalAutoconv

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 16000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Window Sum as Univariate Quadratic (Claims 4.36–4.38)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Child array parameterized by cursor value x at bin pair (k1, k2).
    Sets bin k1 to x, bin k2 to a − x, leaves all other bins unchanged. -/
def child_param {d : ℕ} (base : Fin d → ℤ)
    (k1 k2 : Fin d) (a : ℤ) (x : ℤ) : Fin d → ℤ :=
  fun i => if i = k1 then x else if i = k2 then a - x else base i

/-- Each component of child_param is affine in x. -/
private lemma child_param_affine {d : ℕ} (base : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k1 ≠ k2) (a : ℤ) (i : Fin d) :
    ∃ α β : ℤ, ∀ x : ℤ, child_param base k1 k2 a x i = α * x + β := by
  simp only [child_param]
  by_cases hi1 : i = k1
  · simp [hi1]; exact ⟨1, 0, fun x => by ring⟩
  · by_cases hi2 : i = k2
    · have hne : k2 ≠ k1 := fun h => hk h.symm
      simp [hi2, hne]; exact ⟨-1, a, fun x => by ring⟩
    · simp [hi1, hi2]; exact ⟨0, base i, fun x => by ring⟩

/-- Product of two affine functions is quadratic. -/
private lemma affine_mul_affine (α₁ β₁ α₂ β₂ : ℤ) :
    ∃ A B C : ℤ, ∀ x : ℤ,
      (α₁ * x + β₁) * (α₂ * x + β₂) = A * x ^ 2 + B * x + C :=
  ⟨α₁ * α₂, α₁ * β₂ + β₁ * α₂, β₁ * β₂, fun x => by ring⟩

/-- A finite sum of quadratic functions (over an explicit Finset) is quadratic. -/
private lemma sum_quadratic_finset {ι : Type*} [DecidableEq ι]
    (S : Finset ι) (f : ι → ℤ → ℤ)
    (hf : ∀ i ∈ S, ∃ A B C : ℤ, ∀ x : ℤ, f i x = A * x ^ 2 + B * x + C) :
    ∃ A B C : ℤ, ∀ x : ℤ, (∑ i ∈ S, f i x) = A * x ^ 2 + B * x + C := by
  induction S using Finset.induction with
  | empty => exact ⟨0, 0, 0, fun x => by simp⟩
  | @insert a S ha ih =>
    obtain ⟨A1, B1, C1, h1⟩ := hf _ (Finset.mem_insert_self _ _)
    obtain ⟨A2, B2, C2, h2⟩ := ih (fun i hi => hf i (Finset.mem_insert_of_mem hi))
    exact ⟨A1 + A2, B1 + B2, C1 + C2, fun x => by
      rw [Finset.sum_insert ha, h1, h2]; ring⟩

/-- A finite sum of quadratic functions (over Finset.univ) is quadratic. -/
private lemma sum_quadratic_univ {ι : Type*} [Fintype ι] [DecidableEq ι]
    (f : ι → ℤ → ℤ)
    (hf : ∀ i, ∃ A B C : ℤ, ∀ x : ℤ, f i x = A * x ^ 2 + B * x + C) :
    ∃ A B C : ℤ, ∀ x : ℤ, (∑ i, f i x) = A * x ^ 2 + B * x + C :=
  sum_quadratic_finset Finset.univ f (fun i _ => hf i)

/-- The autoconvolution of a parameterized child is quadratic in x. -/
private lemma autoconv_quadratic_in_x {d : ℕ} (base : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k1 ≠ k2) (a : ℤ) (t : ℕ) :
    ∃ A B C : ℤ, ∀ x : ℤ,
      int_autoconvolution (child_param base k1 k2 a x) t =
        A * x ^ 2 + B * x + C := by
  simp only [int_autoconvolution]
  apply sum_quadratic_univ; intro i
  apply sum_quadratic_univ; intro j
  by_cases heq : i.1 + j.1 = t
  · simp only [if_pos heq]
    obtain ⟨αi, βi, hi⟩ := child_param_affine base k1 k2 hk a i
    obtain ⟨αj, βj, hj⟩ := child_param_affine base k1 k2 hk a j
    obtain ⟨A, B, C, habc⟩ := affine_mul_affine αi βi αj βj
    exact ⟨A, B, C, fun x => by rw [hi, hj]; exact habc x⟩
  · exact ⟨0, 0, 0, fun x => by simp [if_neg heq]⟩

/-- Claim 4.36 (strengthened): The window sum is a quadratic polynomial in x.

    For a child array parameterized by cursor x (child[k1] = x,
    child[k2] = a − x, other bins fixed), the window sum
    ∑ conv[t] equals A·x² + B·x + C for some A, B, C that are
    independent of x. This captures the true functional dependence:
    each conv term child[i]·child[j] is at most quadratic in x
    (since each factor is affine), so the sum is quadratic. -/
theorem window_sum_is_quadratic
    {d : ℕ} (base : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k1 ≠ k2)
    (a : ℤ) (s_lo ell : ℕ) (_hell : 2 ≤ ell) :
    ∃ (A B C : ℤ), ∀ x : ℤ,
      (∑ t ∈ Finset.Icc s_lo (s_lo + ell - 2),
        int_autoconvolution (child_param base k1 k2 a x) t) =
          A * x ^ 2 + B * x + C := by
  exact sum_quadratic_finset _ _ (fun t _ => autoconv_quadratic_in_x base k1 k2 hk a t)

/-- Claim 4.37: The quadratic coefficient A takes values in {−2,−1,0,1}. -/
theorem quadratic_coeff_range
    (pos : ℕ)
    (s_lo ell : ℕ) (_hell : 2 ≤ ell) :
    let A : ℤ := (if s_lo ≤ 4 * pos ∧ 4 * pos ≤ s_lo + ell - 2 then 1 else 0) +
              (if s_lo ≤ 4 * pos + 2 ∧ 4 * pos + 2 ≤ s_lo + ell - 2 then 1 else 0) -
              (if s_lo ≤ 4 * pos + 1 ∧ 4 * pos + 1 ≤ s_lo + ell - 2 then 2 else 0)
    A ∈ ({-2, -1, 0, 1} : Set ℤ) := by
  grind +ring

/-- Claim 4.38: A = 0 whenever all three conv indices fall inside the window. -/
theorem quadratic_coeff_zero_when_contained
    (pos : ℕ) (s_lo ell : ℕ)
    (h1 : s_lo ≤ 4 * pos)
    (h2 : 4 * pos + 2 ≤ s_lo + ell - 2) :
    (if s_lo ≤ 4 * pos ∧ 4 * pos ≤ s_lo + ell - 2 then (1 : ℤ) else 0) +
    (if s_lo ≤ 4 * pos + 2 ∧ 4 * pos + 2 ≤ s_lo + ell - 2 then 1 else 0) -
    (if s_lo ≤ 4 * pos + 1 ∧ 4 * pos + 1 ≤ s_lo + ell - 2 then 2 else 0) = 0 := by
  grind

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Threshold as Affine Function (Claims 4.39–4.40)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A finite sum of affine functions (over an explicit Finset) is affine. -/
private lemma sum_affine_finset {ι : Type*} [DecidableEq ι]
    (S : Finset ι) (f : ι → ℤ → ℤ)
    (hf : ∀ i ∈ S, ∃ α β : ℤ, ∀ x : ℤ, f i x = α * x + β) :
    ∃ α β : ℤ, ∀ x : ℤ, (∑ i ∈ S, f i x) = α * x + β := by
  induction S using Finset.induction with
  | empty => exact ⟨0, 0, fun x => by simp⟩
  | @insert a S ha ih =>
    obtain ⟨α1, β1, h1⟩ := hf _ (Finset.mem_insert_self _ _)
    obtain ⟨α2, β2, h2⟩ := ih (fun i hi => hf i (Finset.mem_insert_of_mem hi))
    exact ⟨α1 + α2, β1 + β2, fun x => by rw [Finset.sum_insert ha, h1, h2]; ring⟩

/-- Claim 4.39 (strengthened): W_int is affine in cursor x.

    For a child array parameterized by cursor x (child[k1] = x,
    child[k2] = a − x, other bins fixed), the bin-mass sum
    ∑ child[i] over a contiguous range equals w_x·x + w_const for
    some w_x, w_const independent of x. Each bin contributes a
    coefficient of +1 (if k1), −1 (if k2), or 0 (otherwise), so
    w_x is the net count of k1 and k2 membership in the range. -/
theorem w_int_affine_in_cursor
    {d : ℕ} (base : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k1 ≠ k2)
    (a : ℤ) (lo_bin hi_bin : ℕ) (_hlo : lo_bin ≤ hi_bin) (_hhi : hi_bin < d) :
    ∃ (w_x w_const : ℤ),
      ∀ x : ℤ,
        (∑ i ∈ Finset.Icc lo_bin hi_bin,
          if h : i < d then child_param base k1 k2 a x ⟨i, h⟩ else 0) =
            w_x * x + w_const := by
  apply sum_affine_finset; intro i _
  by_cases hi : i < d
  · simp only [hi, dite_true]
    exact child_param_affine base k1 k2 hk a ⟨i, hi⟩
  · exact ⟨0, 0, fun x => by simp [hi]⟩

/-- Claim 4.40: The threshold T(x) is affine in x. -/
theorem threshold_affine_no_quadratic
    (dyn_base_ell two_ell_inv_4n : ℝ)
    (w_x : ℤ) (w_const : ℤ) (x : ℝ) :
    ∃ (T0 T1 : ℝ),
      dyn_base_ell + two_ell_inv_4n * (↑w_x * x + ↑w_const) =
      T1 * x + T0 := by
  exact ⟨dyn_base_ell + two_ell_inv_4n * w_const, two_ell_inv_4n * w_x, by ring⟩

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Soundness of the D(x) > 0 Criterion (Claims 4.41–4.42)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.41: D(x) > 0 implies the per-child pruning condition. -/
theorem d_positive_implies_pruned
    (ws : ℤ) (dyn_x : ℝ) (eps : ℝ)
    (_heps : 0 < eps) (heps1 : eps < 1)
    (hD : (ws : ℝ) > dyn_x)
    (hdyn_pos : 0 ≤ dyn_x) :
    ws > ⌊dyn_x * (1 - eps)⌋ := by
  suffices h : (⌊dyn_x * (1 - eps)⌋ : ℝ) < (ws : ℝ) by exact_mod_cast h
  calc (⌊dyn_x * (1 - eps)⌋ : ℝ) ≤ dyn_x * (1 - eps) := Int.floor_le _
    _ ≤ dyn_x := by nlinarith
    _ < ws := hD

/-- Claim 4.42: The current child is prunable (D ≥ 0). -/
theorem current_child_d_nonneg
    (ws : ℤ) (dyn_x : ℝ) (eps : ℝ)
    (_heps : 0 < eps) (_heps1 : eps < 1)
    (h_pruned : ws > ⌊dyn_x * (1 - eps)⌋)
    (_hdyn_pos : 0 ≤ dyn_x) :
    (ws : ℝ) > dyn_x * (1 - eps) := by
  exact lt_of_lt_of_le (Int.lt_floor_add_one _) (by exact_mod_cast h_pruned)

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: Minimum of Quadratic on Integer Interval (Claims 4.43–4.44)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.43: Minimum of a quadratic on an integer interval is at an
    endpoint or near the vertex. -/
theorem quadratic_min_on_interval
    (A B C : ℝ) (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi) :
    let f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C
    ∃ x_min : ℤ, x_lo ≤ x_min ∧ x_min ≤ x_hi ∧
      (∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → f x_min ≤ f x) ∧
      (x_min = x_lo ∨ x_min = x_hi ∨
       (A > 0 ∧ x_min = ⌊-B / (2 * A)⌋) ∨
       (A > 0 ∧ x_min = ⌈-B / (2 * A)⌉)) := by
  by_cases hA_pos : A > 0
  · -- Convex case: use exists_min_image to get minimizer
    have hne : (Finset.Icc x_lo x_hi).Nonempty :=
      ⟨x_lo, Finset.mem_Icc.mpr ⟨le_refl _, hlo⟩⟩
    obtain ⟨x_min, hx_mem, hx_min⟩ := (Finset.Icc x_lo x_hi).exists_min_image
      (fun x : ℤ => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C) hne
    simp only [Finset.mem_Icc] at hx_mem
    refine ⟨x_min, hx_mem.1, hx_mem.2,
      fun x hxlo hxhi => hx_min x (Finset.mem_Icc.mpr ⟨hxlo, hxhi⟩), ?_⟩
    by_cases h_lo : x_min = x_lo; · exact Or.inl h_lo
    by_cases h_hi : x_min = x_hi; · exact Or.inr (Or.inl h_hi)
    -- x_min is interior
    have hlt_lo : x_lo < x_min := lt_of_le_of_ne hx_mem.1 (Ne.symm h_lo)
    have hlt_hi : x_min < x_hi := lt_of_le_of_ne hx_mem.2 h_hi
    have h_plus := hx_min (x_min + 1) (Finset.mem_Icc.mpr ⟨by omega, by omega⟩)
    have h_minus := hx_min (x_min - 1) (Finset.mem_Icc.mpr ⟨by omega, by omega⟩)
    have hc1 : (↑(x_min + 1) : ℝ) = (↑x_min : ℝ) + 1 := by push_cast; ring
    have hc2 : (↑(x_min - 1) : ℝ) = (↑x_min : ℝ) - 1 := by push_cast; ring
    have h1 : 2 * A * (↑x_min : ℝ) + A + B ≥ 0 := by
      rw [hc1] at h_plus; nlinarith [sq_nonneg (↑x_min : ℝ), sq_nonneg ((↑x_min : ℝ) + 1)]
    have h2 : -2 * A * (↑x_min : ℝ) + A - B ≥ 0 := by
      rw [hc2] at h_minus; nlinarith [sq_nonneg (↑x_min : ℝ), sq_nonneg ((↑x_min : ℝ) - 1)]
    set v := -B / (2 * A) with hv_def
    have hA2 : (0 : ℝ) < 2 * A := by linarith
    have hfl : ⌊v⌋ ≤ x_min := by
      by_contra hc; push_neg at hc
      have : (↑x_min : ℝ) + 1 ≤ (↑⌊v⌋ : ℝ) := by exact_mod_cast hc
      have hfv := Int.floor_le v
      have hm1 := mul_le_mul_of_nonneg_left this (le_of_lt hA2)
      have hm2 := mul_le_mul_of_nonneg_left hfv (le_of_lt hA2)
      have hAv : 2 * A * v = -B := by rw [hv_def]; field_simp
      nlinarith
    have hfu : x_min ≤ ⌊v⌋ + 1 := by
      by_contra hc; push_neg at hc
      have hc' : ⌊v⌋ + 2 ≤ x_min := by omega
      have : (↑⌊v⌋ : ℝ) + 2 ≤ (↑x_min : ℝ) := by exact_mod_cast hc'
      have hfv := Int.lt_floor_add_one v
      have hm1 := mul_le_mul_of_nonneg_left (le_of_lt hfv) (le_of_lt hA2)
      have hAv : 2 * A * v = -B := by rw [hv_def]; field_simp
      nlinarith
    rcases eq_or_lt_of_le hfl with heq | hlt
    · exact Or.inr (Or.inr (Or.inl ⟨hA_pos, heq.symm⟩))
    · have hx_eq : x_min = ⌊v⌋ + 1 := by omega
      have hcu : ⌈v⌉ ≤ ⌊v⌋ + 1 := by
        rw [Int.ceil_le]; push_cast; exact (Int.lt_floor_add_one v).le
      have hcl : x_min ≤ ⌈v⌉ := by
        by_contra hc'; push_neg at hc'
        have : (↑⌈v⌉ : ℝ) + 1 ≤ (↑x_min : ℝ) := by exact_mod_cast hc'
        have hcv := Int.le_ceil v
        have hm := mul_le_mul_of_nonneg_left hcv (le_of_lt hA2)
        have hAv : 2 * A * v = -B := by rw [hv_def]; field_simp
        nlinarith
      exact Or.inr (Or.inr (Or.inr ⟨hA_pos, by omega⟩))
  · -- A ≤ 0 case: minimum is at an endpoint
    push_neg at hA_pos
    intro f
    -- Pick x_lo or x_hi as minimizer depending on which gives smaller f
    rcases le_total (A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C)
        (A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C) with h | h
    · -- f(x_lo) ≤ f(x_hi), use x_lo
      refine ⟨x_lo, le_refl _, hlo, ?_, Or.inl rfl⟩
      intro x hx_lo hx_hi
      by_cases hx : x = x_lo
      · rw [hx]
      · have hxl : x_lo < x := lt_of_le_of_ne hx_lo (Ne.symm hx)
        have hrl : (↑x_lo : ℝ) + 1 ≤ (↑x : ℝ) := by exact_mod_cast hxl
        have hrr : (↑x : ℝ) ≤ (↑x_hi : ℝ) := by exact_mod_cast hx_hi
        -- f(x) - f(x_lo) = (x - x_lo) * [A*(x + x_lo) + B]
        -- = (x-x_lo) * [A*(x_hi+x_lo) + B] + (x-x_lo)*A*(x-x_hi)
        -- First term nonneg from h: (x_hi-x_lo)*[A*(x_hi+x_lo)+B] = f(x_hi)-f(x_lo) ≥ 0
        -- Second term: (x-x_lo)*A*(x-x_hi) = A*(x-x_lo)*(x-x_hi), A ≤ 0, x-x_lo > 0, x-x_hi ≤ 0
        -- so A*(x-x_lo)*(x-x_hi) ≥ 0
        -- Use interpolation: (x_hi-x_lo)*(f(x)-f(x_lo)) =
        --   (x-x_lo)*(f(x_hi)-f(x_lo)) - A*(x-x_lo)*(x_hi-x)*(x_hi-x_lo)
        -- Both terms nonneg when A ≤ 0
        have hident : ((↑x_hi : ℝ) - ↑x_lo) * (A * (↑x : ℝ) ^ 2 + B * ↑x + C -
            (A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C)) =
            ((↑x : ℝ) - ↑x_lo) * (A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C -
            (A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C)) -
            A * ((↑x : ℝ) - ↑x_lo) * ((↑x_hi : ℝ) - ↑x) * ((↑x_hi : ℝ) - ↑x_lo) := by ring
        have hprod1 := mul_nonneg (show (↑x : ℝ) - ↑x_lo ≥ 0 by linarith)
          (show A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C -
           (A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C) ≥ 0 by linarith)
        have hprod2 : -A * ((↑x : ℝ) - ↑x_lo) * ((↑x_hi : ℝ) - ↑x) *
            ((↑x_hi : ℝ) - ↑x_lo) ≥ 0 := by
          apply mul_nonneg
          · apply mul_nonneg
            · apply mul_nonneg <;> linarith
            · linarith
          · linarith
        have hd_pos : (↑x_hi : ℝ) - ↑x_lo > 0 := by
          have : (↑x_lo : ℝ) < ↑x_hi := by exact_mod_cast (lt_of_lt_of_le hxl hx_hi)
          linarith
        nlinarith
    · -- f(x_hi) ≤ f(x_lo), use x_hi
      refine ⟨x_hi, hlo, le_refl _, ?_, Or.inr (Or.inl rfl)⟩
      intro x hx_lo hx_hi
      by_cases hx : x = x_hi
      · rw [hx]
      · have hxr : x < x_hi := lt_of_le_of_ne hx_hi hx
        have hrl : (↑x_lo : ℝ) ≤ (↑x : ℝ) := by exact_mod_cast hx_lo
        have hrr : (↑x : ℝ) + 1 ≤ (↑x_hi : ℝ) := by exact_mod_cast hxr
        have hident : ((↑x_hi : ℝ) - ↑x_lo) * (A * (↑x : ℝ) ^ 2 + B * ↑x + C -
            (A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C)) =
            ((↑x_hi : ℝ) - ↑x) * (A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C -
            (A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C)) -
            A * ((↑x_hi : ℝ) - ↑x) * ((↑x : ℝ) - ↑x_lo) * ((↑x_hi : ℝ) - ↑x_lo) := by ring
        have hprod1 := mul_nonneg (show (↑x_hi : ℝ) - ↑x ≥ 0 by linarith)
          (show A * (↑x_lo : ℝ) ^ 2 + B * ↑x_lo + C -
           (A * (↑x_hi : ℝ) ^ 2 + B * ↑x_hi + C) ≥ 0 by linarith)
        have hprod2 : -A * ((↑x_hi : ℝ) - ↑x) * ((↑x : ℝ) - ↑x_lo) *
            ((↑x_hi : ℝ) - ↑x_lo) ≥ 0 := by
          apply mul_nonneg
          · apply mul_nonneg
            · apply mul_nonneg <;> linarith
            · linarith
          · linarith
        have hd_pos : (↑x_hi : ℝ) - ↑x_lo > 0 := by
          have : (↑x_lo : ℝ) < ↑x_hi := by
            exact_mod_cast lt_of_le_of_lt hx_lo hxr
          linarith
        nlinarith

/-- Claim 4.44: For A ≤ 0 (concave or linear), checking the two
    endpoints suffices. -/
theorem concave_endpoints_suffice
    (A B C : ℝ) (hA : A ≤ 0)
    (x_lo x_hi : ℤ) (_hlo : x_lo ≤ x_hi) :
    let f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C
    f x_lo > 0 → f x_hi > 0 →
    ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → f x > 0 := by
  intro f hf_lo hf_hi x hx_lo hx_hi
  show A * (↑x : ℝ) ^ 2 + B * ↑x + C > 0
  by_cases hx : x = x_lo ∨ x = x_hi
  · rcases hx with rfl | rfl
    · exact hf_lo
    · exact hf_hi
  · push_neg at hx
    have hxl : x_lo < x := lt_of_le_of_ne hx_lo (Ne.symm hx.1)
    have hxr : x < x_hi := lt_of_le_of_ne hx_hi hx.2
    have hrl : (↑x_lo : ℝ) + 1 ≤ (↑x : ℝ) := by exact_mod_cast hxl
    have hrr : (↑x : ℝ) + 1 ≤ (↑x_hi : ℝ) := by exact_mod_cast hxr
    -- Key: -A*(x-x_lo)*(x_hi-x) ≥ 0 since A ≤ 0
    have hprod : -A * ((↑x : ℝ) - ↑x_lo) * ((↑x_hi : ℝ) - ↑x) ≥ 0 := by
      apply mul_nonneg; apply mul_nonneg; linarith; linarith; linarith
    nlinarith [sq_nonneg ((↑x : ℝ) - ↑x_lo), sq_nonneg ((↑x : ℝ) - ↑x_hi),
              sq_nonneg ((↑x_hi : ℝ) - ↑x_lo),
              mul_pos (show (↑x_hi : ℝ) - ↑x > 0 by linarith) hf_lo,
              mul_pos (show (↑x : ℝ) - ↑x_lo > 0 by linarith) hf_hi,
              hprod]

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Digit 0 Consecutive Advance Property (Claim 4.45)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Model of the Gray code focus-pointer update after advancing digit j.
    Matches run_cascade.py lines 1269–1282. -/
def gc_focus_update (n_active : ℕ)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (j : Fin n_active)
    (hit_boundary : Bool) : Fin (n_active + 1) → ℕ :=
  fun i =>
    let f0 : Fin (n_active + 1) → ℕ :=
      fun k => if k.1 = 0 then 0 else gc_focus k
    if hit_boundary then
      if i.1 = j.1 then f0 ⟨j.1 + 1, by omega⟩
      else if i.1 = j.1 + 1 then j.1 + 1
      else f0 i
    else f0 i

/-- Claim 4.45: After non-boundary advance of digit 0, gc_focus'[0] = 0. -/
theorem digit_0_consecutive_advance
    (n_active : ℕ) (_hn : 0 < n_active)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (j : Fin n_active)
    (hit_boundary : Bool)
    (h : j.1 = 0 → hit_boundary = false) :
    gc_focus_update n_active gc_focus j hit_boundary ⟨0, by omega⟩ = 0 := by
  simp only [gc_focus_update]
  split <;> simp_all <;> omega

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: Fast-Forward Correctness (Claim 4.46)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Helper: telescoping sum for ℕ-indexed sequences. -/
private lemma telescope_nat (g : ℕ → ℤ) (N : ℕ) :
    ∑ i ∈ Finset.range N, (g (i + 1) - g i) = g N - g 0 := by
  induction N with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, ih]; ring

/-- Helper: telescoping sum property for finite sequences. -/
private lemma telescope_fin {N : ℕ} (g : Fin (N + 1) → ℤ) :
    ∑ i : Fin N, (g i.succ - g i.castSucc) = g (Fin.last N) - g ⟨0, by omega⟩ := by
  induction N with
  | zero => simp
  | succ n ih =>
    rw [Fin.sum_univ_succ]
    have : ∑ x : Fin n, (g (Fin.succ x).succ - g (Fin.succ x).castSucc) =
           ∑ x : Fin n, ((g ∘ Fin.succ) x.succ - (g ∘ Fin.succ) x.castSucc) := by
      apply Finset.sum_congr rfl; intro x _
      simp [Function.comp]
    rw [this, ih (g ∘ Fin.succ)]
    simp only [Function.comp, Fin.succ_last]
    -- Goal involves: g (Fin.succ 0), g (Fin.castSucc 0), g (Fin.last n.succ),
    -- g ⟨0, _⟩.succ, g ⟨0, _⟩
    -- The key equalities:
    -- (1) Fin.succ (0 : Fin (n+1)) = ⟨0, _⟩.succ (both are ⟨1, _⟩ : Fin (n+2))
    -- (2) Fin.castSucc (0 : Fin (n+1)) = ⟨0, _⟩ : Fin (n+2)
    -- (3) Fin.last n.succ = Fin.last (n+1)
    -- After these, the goal is g 1 - g 0 + (g last - g 1) = g last - g 0
    -- Show all the Fin equalities by ext
    have h1 : (⟨0, by omega⟩ : Fin (n + 1)).succ = Fin.succ (0 : Fin (n + 1)) := by rfl
    have h2 : Fin.last n.succ = Fin.last (Nat.succ n) := by rfl
    have h3 : (⟨0, by omega⟩ : Fin (n + 2)) = Fin.castSucc (0 : Fin (n + 1)) := by
      ext; simp [Fin.castSucc]
    rw [h1, h2, h3]; omega

/-- Claim 4.46: Fast-forward produces identical raw_conv to step-by-step. -/
theorem fast_forward_equiv_stepwise
    {d : ℕ} (child_start child_end : Fin d → ℤ)
    (_k1 _k2 : Fin d) (_hk : _k2.1 = _k1.1 + 1)
    (_a : ℤ) (_x_curr _x_far : ℤ)
    (_h_start_k1 : child_start _k1 = _x_curr)
    (_h_start_k2 : child_start _k2 = _a - _x_curr)
    (_h_end_k1 : child_end _k1 = _x_far)
    (_h_end_k2 : child_end _k2 = _a - _x_far)
    (_h_unchanged : ∀ i : Fin d, i ≠ _k1 → i ≠ _k2 →
      child_end i = child_start i)
    (N : ℕ)
    (children : Fin (N + 1) → (Fin d → ℤ))
    (h_first : children ⟨0, by omega⟩ = child_start)
    (h_last : children ⟨N, by omega⟩ = child_end)
    (raw_conv_init : Fin (2 * d - 1) → ℤ)
    (raw_conv_single : Fin (2 * d - 1) → ℤ)
    (h_single : ∀ t : Fin (2 * d - 1),
      raw_conv_single t = raw_conv_init t +
        autoconv_delta child_start child_end t.1)
    (raw_conv_stepwise : Fin (2 * d - 1) → ℤ)
    (h_stepwise : ∀ t : Fin (2 * d - 1),
      raw_conv_stepwise t = raw_conv_init t +
        ∑ i : Fin N, autoconv_delta
          (children i.castSucc) (children i.succ) t.1) :
    raw_conv_single = raw_conv_stepwise := by
  funext t; simp only [h_single t, h_stepwise t, autoconv_delta]
  congr 1
  -- Goal: int_autoconvolution child_end t.1 - int_autoconvolution child_start t.1
  --     = ∑ i, (int_autoconvolution (children i.succ) t.1 - int_autoconvolution (children i.castSucc) t.1)
  -- Rewrite using h_first, h_last and apply telescoping
  conv_lhs =>
    rw [show child_end = children ⟨N, by omega⟩ from h_last.symm,
        show child_start = children ⟨0, by omega⟩ from h_first.symm]
  rw [show (⟨N, by omega⟩ : Fin (N + 1)) = Fin.last N from by ext; simp [Fin.last]]
  exact (telescope_fin (fun i => int_autoconvolution (children i) t.1)).symm

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART G: Top-Level Soundness and Post-Fast-Forward (Claims 4.47–4.48)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.47: Top-level sweep skip soundness. -/
theorem sweep_skip_sound
    (ws : ℤ → ℤ) (dyn : ℤ → ℝ) (eps : ℝ)
    (heps : 0 < eps) (heps1 : eps < 1)
    (x_lo x_hi : ℤ) (_hlo : x_lo ≤ x_hi)
    (h_dyn_pos : ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → 0 ≤ dyn x)
    (h_D_pos : ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → (ws x : ℝ) > dyn x) :
    ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → ws x > ⌊dyn x * (1 - eps)⌋ := by
  intro x hx_lo hx_hi
  exact d_positive_implies_pruned (ws x) (dyn x) eps heps heps1
    (h_D_pos x hx_lo hx_hi) (h_dyn_pos x hx_lo hx_hi)

/-- Claim 4.48: After fast-forward, raw_conv = int_autoconvolution(child_end). -/
theorem raw_conv_after_fast_forward
    {d : ℕ} (child_start child_end : Fin d → ℤ)
    (raw_conv_init raw_conv_final : Fin (2 * d - 1) → ℤ)
    (h_init : ∀ t : Fin (2 * d - 1),
      raw_conv_init t = int_autoconvolution child_start t.1)
    (h_update : ∀ t : Fin (2 * d - 1),
      raw_conv_final t = raw_conv_init t +
        autoconv_delta child_start child_end t.1) :
    ∀ t : Fin (2 * d - 1),
      raw_conv_final t = int_autoconvolution child_end t.1 := by
  intro t
  simp only [h_update t, h_init t, autoconv_delta]
  omega

end
