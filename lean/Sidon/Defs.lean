/-
Sidon Autocorrelation Project — Core Definitions

Core definitions used throughout the proof: autoconvolution ratio, discrete
autoconvolution, test values, bin masses, canonical discretization, etc.

Fine-grid convention (C&S B_{n,m}):
  - d = 2n bins of width δ = 1/(4n)
  - Integer coordinates c_i ≥ 0 with ∑ c_i = S = 4nm
  - Physical heights a_i = c_i / m  (multiples of 1/m)
  - ∫g = ∑ (c_i/m)·δ = S/(4nm) = 1
  - Correction: 2/m + 1/m² (C&S Lemma 3)
-/

import Mathlib

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
-- Core Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- The autoconvolution ratio R(f) = ‖f*f‖_∞ / (∫f)². -/
noncomputable def autoconvolution_ratio (f : ℝ → ℝ) : ℝ :=
  let conv := MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume
  let norm_inf := (MeasureTheory.eLpNorm conv ⊤ MeasureTheory.volume).toReal
  let integral := MeasureTheory.integral MeasureTheory.volume f
  norm_inf / (integral ^ 2)

/-- The autoconvolution constant c = inf R(f) over admissible f with positive integral.
    The condition ∫f > 0 is necessary: without it, zero-integral functions would give
    R(f) = 0/0 = 0, making the infimum trivially ≤ 0. -/
noncomputable def autoconvolution_constant : ℝ :=
  sInf {r : ℝ | ∃ (f : ℝ → ℝ), (∀ x, 0 ≤ f x) ∧ (Function.support f ⊆ Set.Ioo (-1/4) (1/4))
    ∧ MeasureTheory.integral MeasureTheory.volume f > 0 ∧ r = autoconvolution_ratio f}

/-- Discrete autoconvolution: conv[k] = ∑_{i+j=k} a_i · a_j. -/
def discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then a i * a j else 0

/-- Test value TV(n, m, c, ℓ, s_lo) for a composition c on the C&S fine grid.

    Fine grid B_{n,m}: d = 2n bins, integer coordinates c_i summing to S = 4nm.
    Physical heights a_i = c_i / m (multiples of 1/m).
    TV = (1 / (4nℓ)) · ∑_{k in window} ∑_{i+j=k} a_i · a_j
       = (1 / (4nℓm²)) · ∑_{k in window} ∑_{i+j=k} c_i · c_j.
    Matches CPU code: test_values.py uses scale = 1/m, norm = 1/(4n·ℓ). -/
noncomputable def test_value (n m : ℕ) (c : Fin (2 * n) → ℕ) (ℓ s_lo : ℕ) : ℝ :=
  let d := 2 * n
  let a : Fin d → ℝ := fun i => (c i : ℝ) / m
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

/-- A composition on the C&S fine grid B_{n,m}: integer coordinates summing to S = 4nm.
    Physical heights a_i = c_i / m give a step function on 2n bins
    of width 1/(4n) with ∫g = ∑ (c_i/m)·(1/(4n)) = S/(4nm) = 1.
    Matches CPU convention: S = 4 * n * m.
    Palindrome symmetry: c i = c (2n - 1 - i) halves the search space. -/
def is_composition (n m : ℕ) (c : Fin (2 * n) → ℕ) : Prop :=
  ∑ i, c i = 4 * n * m

/-- Bin masses: integral of f over each bin. -/
noncomputable def bin_masses (f : ℝ → ℝ) (n : ℕ) : Fin (2 * n) → ℝ :=
  fun i =>
    let δ := 1 / (4 * n : ℝ)
    let a := -(1/4 : ℝ) + i * δ
    let b := -(1/4 : ℝ) + (i + 1) * δ
    MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ico a b) f)

/-- Canonical discretization via floor-rounding of cumulative masses.

    Fine grid (C&S B_{n,m}): rounds to S = 4nm quanta.  The height quantum is
    1/m, so heights a_i = c_i/m approximate ∫_bin f / δ with ||ε||_∞ ≤ 1/m
    (C&S Lemma 2).  The correction is 2/m + 1/m² (C&S Lemma 3). -/
noncomputable def canonical_discretization (f : ℝ → ℝ) (n m : ℕ) : Fin (2 * n) → ℕ :=
  fun i =>
    let S := 4 * n * m
    let masses := bin_masses f n
    let total_mass := ∑ j, masses j
    let cum_mass (k : ℕ) := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
    let target_cum (k : ℕ) := (cum_mass k) / total_mass * S
    let discrete_cum (k : ℕ) := ⌊target_cum k⌋.natAbs
    if i.1 + 1 < 2 * n then discrete_cum (i.1 + 1) - discrete_cum i.1
    else S - discrete_cum i.1

/-- Contributing bins for a window (ℓ, s_lo). -/
def contributing_bins (n : ℕ) (ℓ s_lo : ℕ) : Finset (Fin (2 * n)) :=
  let d := 2 * n
  Finset.filter (fun i => ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2) Finset.univ

/-- Cumulative distribution helper D(k).
    Fine grid: targets S = 4nm quanta (matching canonical_discretization). -/
noncomputable def canonical_cumulative_distribution (f : ℝ → ℝ) (n m : ℕ) (k : ℕ) : ℕ :=
  let S := 4 * n * m
  let masses := bin_masses f n
  let total_mass := ∑ j, masses j
  let cum_mass := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
  let target_cum := cum_mass / total_mass * S
  ⌊target_cum⌋.natAbs

/-- Restriction of f to bin i. -/
noncomputable def f_restricted (f : ℝ → ℝ) (n : ℕ) (i : Fin (2 * n)) : ℝ → ℝ :=
  let δ := 1 / (4 * n : ℝ)
  let a := -(1/4 : ℝ) + i * δ
  let b := -(1/4 : ℝ) + (i + 1) * δ
  Set.indicator (Set.Ico a b) f

end -- noncomputable section
