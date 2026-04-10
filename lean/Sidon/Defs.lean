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

/-- Cumulative distribution helper D(k).
    Fine grid: targets S = 4nm quanta (matching canonical_discretization). -/
noncomputable def canonical_cumulative_distribution (f : ℝ → ℝ) (n m : ℕ) (k : ℕ) : ℕ :=
  let S := 4 * n * m
  let masses := bin_masses f n
  let total_mass := ∑ j, masses j
  let cum_mass := ∑ j : Fin (2 * n), if j.1 < k then masses j else 0
  let target_cum := cum_mass / total_mass * S
  ⌊target_cum⌋.natAbs

-- ═══════════════════════════════════════════════════════════════════════════════
-- Cascade Pruning Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- A child at resolution 2*n_half is a valid refinement of parent at n_half.
    Allows ±1 deviation per bin pair (from floor rounding in canonical_discretization).
    Total mass is exact: child sums to 2 * parent sum (S doubles when n_half doubles). -/
def is_valid_child (n_half : ℕ) (parent : Fin (2 * n_half) → ℕ)
    (child : Fin (2 * (2 * n_half)) → ℕ) : Prop :=
  (∑ i, child i = 2 * ∑ i, parent i) ∧
  (∀ i : Fin (2 * n_half),
    let pair_sum := child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩
    pair_sum + 1 ≥ 2 * parent i ∧ 2 * parent i + 1 ≥ pair_sum)

/-- A composition is cascade-pruned if either directly pruned (TV > threshold)
    or ALL valid children are cascade-pruned at the next resolution.

    This mirrors the cascade algorithm: at each level, the code either
    prunes a composition (TV > threshold) or refines it into children
    and processes each child recursively. The cascade terminating with
    0 survivors means every root composition is CascadePruned. -/
inductive CascadePruned (m : ℕ) (c_target correction : ℝ) :
    (n_half : ℕ) → (Fin (2 * n_half) → ℕ) → Prop where
  | direct {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∃ ℓ s_lo, 2 ≤ ℓ ∧ test_value n_half m c ℓ s_lo > c_target + correction) :
      CascadePruned m c_target correction n_half c
  | refine {n_half : ℕ} {c : Fin (2 * n_half) → ℕ}
      (h : ∀ child : Fin (2 * (2 * n_half)) → ℕ,
        is_valid_child n_half c child →
        CascadePruned m c_target correction (2 * n_half) child) :
      CascadePruned m c_target correction n_half c

/-- Convolution of nonneg functions is nonneg. -/
theorem convolution_nonneg {f g : ℝ → ℝ} (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) :
    ∀ x, 0 ≤ MeasureTheory.convolution f g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  intro x
  simp [MeasureTheory.convolution]
  exact MeasureTheory.integral_nonneg fun t => mul_nonneg (hf t) (hg (x - t))

end -- noncomputable section
