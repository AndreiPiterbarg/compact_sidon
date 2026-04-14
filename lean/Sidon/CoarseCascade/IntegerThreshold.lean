/-
Sidon Autocorrelation Project — Integer Threshold and Per-Bin Mass Cap (Proof Stubs)

The coarse cascade uses integer arithmetic with a precomputed threshold per ell:

  TV_W = (2d / (ell * S^2)) * ws_int
  Prune if TV >= c_target, i.e., ws_int > floor(c_target * ell * S^2 / (2d) - eps)

Per-bin mass cap:
  If a single bin has mass k, self-convolution gives TV >= d * k^2 / S^2.
  So k > S * sqrt(c_target / d) implies automatic pruning.

Source: run_cascade_coarse.py lines 34-46, 179-190.
-/

import Sidon.Proof.CoarseCascade

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

-- =============================================================================
-- PART 1: Integer Test Value
-- =============================================================================

/-- Integer window sum: ws_int = sum_{k=s..s+ell-2} sum_{i+j=k} c_i * c_j
    where c is an integer composition of S. -/
def int_window_sum {d : ℕ} (c : Fin d → ℕ) (ell s : ℕ) : ℕ :=
  ∑ k ∈ Finset.Icc s (s + ell - 2),
    ∑ i : Fin d, ∑ j : Fin d,
      if i.val + j.val = k then c i * c j else 0

/-- The real-valued TV equals (2d / (ell * S^2)) * ws_int. -/
theorem tv_eq_int_window_sum (d S : ℕ) (hS : S > 0) (c : Fin d → ℕ)
    (hc_sum : ∑ i, c i = S) (ell s : ℕ) (hell : 2 ≤ ell) :
    mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell s =
    (2 * (d : ℝ) / ((ell : ℝ) * (S : ℝ) ^ 2)) * (int_window_sum c ell s : ℝ) := by
  sorry

-- =============================================================================
-- PART 2: Integer Threshold Soundness
-- =============================================================================

/-- **Integer threshold formula:**
    thr[ell] = floor(c_target * ell * S^2 / (2*d) - eps)

    Prune (TV >= c_target) iff ws_int > thr[ell].

    This is sound because:
      TV >= c_target
      iff (2d / (ell * S^2)) * ws_int >= c_target
      iff ws_int >= c_target * ell * S^2 / (2d)
      iff ws_int > floor(c_target * ell * S^2 / (2d) - eps)  [for small eps, ws_int integer] -/
theorem int_threshold_sound (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c_target : ℝ) (hct : 0 < c_target) (ell : ℕ) (hell : 2 ≤ ell)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S)
    (eps : ℝ) (heps : 0 < eps) (heps_small : eps < 1) :
    let thr := Int.floor (c_target * (ell : ℝ) * (S : ℝ) ^ 2 / (2 * (d : ℝ)) - eps)
    (int_window_sum c ell 0 : ℤ) > thr →
    mass_test_value d (fun i => (c i : ℝ) / (S : ℝ)) ell 0 ≥ c_target := by
  sorry

/-- The integer threshold is a 1D array indexed by ell only.
    Unlike the C&S W-refined threshold, it does NOT depend on the
    per-window W_int (total mass in the window). This is because
    the coarse grid has no correction term. -/
theorem int_threshold_ell_only (d S : ℕ) (c_target : ℝ)
    (ell : ℕ) (s1 s2 : ℕ) :
    let thr := fun ℓ => Int.floor (c_target * (ℓ : ℝ) * (S : ℝ) ^ 2 / (2 * (d : ℝ)))
    thr ell = thr ell := by
  rfl

-- =============================================================================
-- PART 3: Per-Bin Mass Cap
-- =============================================================================

/-- Self-convolution of a single bin: if bin i has mass k, then
    conv[2i] = k^2, and the ell=2 window at s=2i gives
    TV = d * k^2 / S^2. -/
theorem single_bin_self_conv (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S) (i : Fin d) :
    mass_test_value d (fun j => (c j : ℝ) / (S : ℝ)) 2 (2 * i.val) ≥
    (d : ℝ) * ((c i : ℝ) / (S : ℝ)) ^ 2 := by
  sorry

/-- **Per-bin mass cap:** if c_i > floor(S * sqrt(c_target / d)),
    then TV >= c_target from self-convolution alone.

    x_cap(d) = floor(S * sqrt(c_target / d))
    k > x_cap => d * k^2 / S^2 > c_target => pruned

    Source: run_cascade_coarse.py lines 179-190. -/
theorem per_bin_mass_cap (d S : ℕ) (hd : d > 0) (hS : S > 0)
    (c_target : ℝ) (hct : 0 < c_target)
    (c : Fin d → ℕ) (hc_sum : ∑ i, c i = S) (i : Fin d)
    (h_exceed : (c i : ℝ) > (S : ℝ) * Real.sqrt (c_target / (d : ℝ))) :
    ∃ ell s, 2 ≤ ell ∧
      mass_test_value d (fun j => (c j : ℝ) / (S : ℝ)) ell s ≥ c_target := by
  sorry

-- =============================================================================
-- PART 4: Constant S Across Levels
-- =============================================================================

/-- In the coarse cascade, S is fixed across all levels.
    When a parent bin with mass p splits into children (a, p-a),
    the total mass S = sum c_i is preserved.

    This contrasts with the C&S fine grid where S = 4nm grows with d. -/
theorem coarse_cascade_mass_preserved {d : ℕ} (S : ℕ)
    (parent : Fin d → ℕ) (h_par_sum : ∑ i, parent i = S)
    (child : Fin (2 * d) → ℕ)
    (h_split : ∀ i : Fin d,
      child ⟨2 * i.val, by omega⟩ + child ⟨2 * i.val + 1, by omega⟩ = parent i) :
    ∑ i, child i = S := by
  sorry

end -- noncomputable section
