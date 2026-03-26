/-
Sidon Autocorrelation Project — Univariate Sweep Skip (Claims 4.36–4.46)

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

STATUS: PROOF OBLIGATIONS ONLY — no proofs are attempted here.
Each `sorry` marks an open obligation. Dependencies on existing modules
(Defs, IncrementalAutoconv, GrayCode, DynamicThreshold) are noted.
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
-- PART A: Window Sum as Univariate Quadratic (Claims 4.36–4.38)
--
-- When only digit 0's cursor changes, the child bins k1 = 2·pos and
-- k2 = 2·pos+1 vary as child[k1] = x, child[k2] = a − x where
-- a = parent[pos] is fixed. All other child bins are constant.
-- The window sum ws(x) = Σ_{t∈W} conv[t] is quadratic in x.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.36: The window sum is a quadratic polynomial in x.

    For a window W = [s_lo, s_lo + ℓ − 2] and cursor variable x,
    ws(x) = A·x² + B·x + C where:
      A = 𝟙[4p ∈ W] + 𝟙[4p+2 ∈ W] − 2·𝟙[4p+1 ∈ W]
      B = −2a·𝟙[4p+2 ∈ W] + 2a·𝟙[4p+1 ∈ W]
          + Σ_{j≠k1,k2} 2·child[j]·(𝟙[k1+j ∈ W] − 𝟙[k2+j ∈ W])
      C = (remaining constant terms)

    This follows from expanding conv[t] = Σ_{i+j=t} c_i·c_j and
    collecting terms by powers of x, noting that only c_{k1} = x
    and c_{k2} = a − x depend on x.

    Depends on: Defs.discrete_autoconvolution -/
theorem window_sum_is_quadratic
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (a : ℤ) (ha : child k1 + child k2 = a)
    (x : ℤ) (hx1 : child k1 = x) (hx2 : child k2 = a - x)
    (s_lo ell : ℕ) (hell : 2 ≤ ell)
    -- Window indicator
    (in_window : ℕ → Prop := fun t => s_lo ≤ t ∧ t ≤ s_lo + ell - 2) :
    -- ∃ A B C : ℤ, ws(x) = A·x² + B·x + C
    ∃ (A B C : ℤ),
      (∑ t ∈ Finset.Icc s_lo (s_lo + ell - 2),
        int_autoconvolution child t) = A * x ^ 2 + B * x + C := by
  sorry

/-- Claim 4.37: The quadratic coefficient A takes values in {−2,−1,0,1}.

    A = 𝟙[4p ∈ W] + 𝟙[4p+2 ∈ W] − 2·𝟙[4p+1 ∈ W]

    Each indicator is 0 or 1, and the three conv indices 4p, 4p+1, 4p+2
    are consecutive. The possible values are:
      all three in W:     A = 1 + 1 − 2 = 0
      none in W:          A = 0
      only 4p in W:       A = 1
      only 4p+2 in W:     A = 1
      only 4p+1 in W:     A = −2
      4p,4p+1 in W:       A = 1 − 2 = −1
      4p+1,4p+2 in W:     A = 1 − 2 = −1

    In particular A = 0 whenever the window fully contains
    {4p, 4p+1, 4p+2}, which holds for most windows of size ≥ 4.

    Depends on: Claim 4.36 -/
theorem quadratic_coeff_range
    (pos : ℕ)
    (s_lo ell : ℕ) (hell : 2 ≤ ell)
    (in_w : ℕ → Bool := fun t => s_lo ≤ t ∧ t ≤ s_lo + ell - 2)
    (A : ℤ := (if s_lo ≤ 4 * pos ∧ 4 * pos ≤ s_lo + ell - 2 then 1 else 0) +
              (if s_lo ≤ 4 * pos + 2 ∧ 4 * pos + 2 ≤ s_lo + ell - 2 then 1 else 0) -
              (if s_lo ≤ 4 * pos + 1 ∧ 4 * pos + 1 ≤ s_lo + ell - 2 then 2 else 0)) :
    A ∈ ({-2, -1, 0, 1} : Set ℤ) := by
  sorry

/-- Claim 4.38: A = 0 whenever all three conv indices fall inside
    the window, i.e., when 4p ≥ s_lo and 4p+2 ≤ s_lo + ℓ − 2.

    This is the common case: the window sum becomes LINEAR in x,
    simplifying the range check to an endpoint evaluation. -/
theorem quadratic_coeff_zero_when_contained
    (pos : ℕ) (s_lo ell : ℕ)
    (h1 : s_lo ≤ 4 * pos)
    (h2 : 4 * pos + 2 ≤ s_lo + ell - 2) :
    (1 : ℤ) + 1 - 2 = 0 := by
  sorry  -- Trivially 0

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Threshold as Affine Function (Claims 4.39–4.40)
--
-- The dynamic threshold dyn_x(x) = dyn_base_ell + two_ell_inv_4n · W_int(x)
-- is affine in x because W_int (the sum of child masses in the window's
-- bin range) depends linearly on x through the variable bins k1 and k2.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.39: W_int is affine in x.

    W_int(x) = Σ_{i ∈ [lo_bin, hi_bin]} child[i]
    The variable bins are k1 (value x) and k2 (value a−x).

    Three cases:
    (a) Both k1 and k2 in [lo_bin, hi_bin]: W_int has x + (a−x) = a,
        constant. Coefficient w_x = 0.
    (b) Only k1 in range: W_int includes x. Coefficient w_x = +1.
    (c) Only k2 in range: W_int includes a−x. Coefficient w_x = −1.
    (d) Neither in range: W_int is constant. Coefficient w_x = 0.

    Depends on: Defs -/
theorem w_int_affine_in_cursor
    {d : ℕ} (child : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (a : ℤ) (x : ℤ)
    (hx1 : child k1 = x) (hx2 : child k2 = a - x)
    (lo_bin hi_bin : ℕ) (hlo : lo_bin ≤ hi_bin) (hhi : hi_bin < d) :
    ∃ (w_x : ℤ) (w_const : ℤ),
      w_x ∈ ({-1, 0, 1} : Set ℤ) ∧
      (∑ i ∈ Finset.Icc lo_bin hi_bin, child ⟨i, by omega⟩) = w_x * x + w_const := by
  sorry

/-- Claim 4.40: The threshold T(x) = dyn_base_ell + two_ell_inv_4n · W_int(x)
    is affine (NOT bilinear) in x. It has no x² term.

    This ensures D(x) = ws(x) − T(x) inherits the quadratic coefficient A
    directly from ws(x), with the threshold only modifying B and C.

    Depends on: Claim 4.39 -/
theorem threshold_affine_no_quadratic
    (dyn_base_ell two_ell_inv_4n : ℝ)
    (w_x : ℤ) (w_const : ℤ) (x : ℝ) :
    ∃ (T0 T1 : ℝ),
      dyn_base_ell + two_ell_inv_4n * (↑w_x * x + ↑w_const) =
      T1 * x + T0 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: Soundness of the D(x) > 0 Criterion (Claims 4.41–4.42)
--
-- We define D(x) = ws(x) − dyn_x(x) and show that D(x) > 0 is a
-- sufficient condition for the per-child pruning test to succeed.
-- This is the core soundness argument: it must account for the floor
-- and the (1−4ε) rounding guard in the integer threshold comparison.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.41: D(x) > 0 implies the per-child pruning condition.

    The per-child test prunes when: ws > ⌊dyn_x · (1 − 4ε)⌋
    We show: D(x) > 0 ⟹ ws(x) > dyn_x(x) ⟹ ws(x) > ⌊dyn_x(x)·(1−4ε)⌋

    The chain holds because:
    (1) dyn_x(x) > dyn_x(x)·(1−4ε)  since 4ε > 0
    (2) dyn_x(x)·(1−4ε) ≥ ⌊dyn_x(x)·(1−4ε)⌋  by definition of floor
    (3) ws(x) > dyn_x(x) ≥ ⌊dyn_x(x)·(1−4ε)⌋

    Note: this is CONSERVATIVE — D(x) > 0 is stricter than needed.
    The gap is at most 1 + 4ε·dyn_x ≈ 1 unit.

    Depends on: DynamicThreshold -/
theorem d_positive_implies_pruned
    (ws : ℤ) (dyn_x : ℝ) (eps : ℝ)
    (heps : 0 < eps) (heps1 : eps < 1)
    (hD : (ws : ℝ) > dyn_x)
    (hdyn_pos : 0 ≤ dyn_x) :
    ws > ⌊dyn_x * (1 - eps)⌋ := by
  sorry

/-- Claim 4.42: The pruning condition at the current child is a
    special case. When the full scan prunes a child with ws > dyn_it,
    we have D(x_current) = ws − dyn_x ≥ 0.

    More precisely: ws > ⌊dyn_x·(1−4ε)⌋ implies ws ≥ ⌊dyn_x·(1−4ε)⌋ + 1,
    and since ⌊dyn_x·(1−4ε)⌋ ≤ dyn_x, we get ws > 0 + dyn_x − 1 = dyn_x − 1.
    Since ws is an integer, ws ≥ ⌈dyn_x⌉ when dyn_x is not an integer,
    giving D(x_current) > 0.

    When dyn_x is exactly an integer, D(x_current) ≥ 0 (possibly zero).
    This edge case means the range check may conservatively fail even
    though the current child was pruned. This is safe — it only means
    the sweep skip doesn't trigger, falling back to per-child testing.

    Depends on: DynamicThreshold -/
theorem current_child_d_nonneg
    (ws : ℤ) (dyn_x : ℝ) (eps : ℝ)
    (heps : 0 < eps) (heps1 : eps < 1)
    (h_pruned : ws > ⌊dyn_x * (1 - eps)⌋)
    (hdyn_pos : 0 ≤ dyn_x) :
    (ws : ℝ) ≥ dyn_x - 1 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART D: Minimum of Quadratic on Integer Interval (Claims 4.43–4.44)
--
-- D(x) = A·x² + B_eff·x + C_eff is a quadratic on the integer interval
-- [x_lo, x_hi]. The minimum over this interval determines whether the
-- entire sweep is prunable. The 3-point check (two endpoints + vertex)
-- correctly finds the minimum for all cases of A.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.43: Minimum of a quadratic on an interval is attained at
    an endpoint or the vertex.

    For A ≤ 0 (concave or linear): the minimum on [a,b] is at an
    endpoint (a or b). No interior check needed.

    For A > 0 (convex): the minimum on [a,b] is either at an endpoint
    or at the vertex x* = −B/(2A). Since we work over integers, we
    check both ⌊x*⌋ and ⌈x*⌉ if they fall within [a,b].

    The total number of candidate points is at most 4 (2 endpoints +
    2 integer neighbors of vertex). -/
theorem quadratic_min_on_interval
    (A B C : ℝ) (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi)
    (f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C) :
    ∃ x_min : ℤ, x_lo ≤ x_min ∧ x_min ≤ x_hi ∧
      (∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → f x_min ≤ f x) ∧
      -- x_min is one of at most 4 candidates
      (x_min = x_lo ∨ x_min = x_hi ∨
       (A > 0 ∧ x_min = ⌊-B / (2 * A)⌋) ∨
       (A > 0 ∧ x_min = ⌈-B / (2 * A)⌉)) := by
  sorry

/-- Claim 4.44: For A ≤ 0 (concave or linear), checking the two
    endpoints suffices. If D(x_lo) > 0 and D(x_hi) > 0, then
    D(x) > 0 for all x ∈ [x_lo, x_hi].

    This is the common case: A = 0 for most windows (Claim 4.38),
    and the range check reduces to two endpoint evaluations. -/
theorem concave_endpoints_suffice
    (A B C : ℝ) (hA : A ≤ 0)
    (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi)
    (f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C)
    (h_lo : f x_lo > 0) (h_hi : f x_hi > 0) :
    ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → f x > 0 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Digit 0 Consecutive Advance Property (Claim 4.45)
--
-- The sweep skip is only valid because digit 0 advances in consecutive
-- steps. We must prove that between a non-boundary advance of digit 0
-- and the next boundary hit, ONLY digit 0 changes.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.45: After a non-boundary advance of digit j in the Knuth
    mixed-radix Gray code, gc_focus[0] = 0, so the next advance picks
    digit 0.

    Proof: at the start of each advance step, gc_focus[0] is reset to 0
    (line 1682 of run_cascade.py). If digit j did NOT hit its boundary,
    the focus chain is not modified. Therefore gc_focus[0] remains 0,
    and the next call to j = gc_focus[0] yields j = 0.

    Consequence: after digit 0 advances without hitting a boundary,
    the next advance is also digit 0. This continues until digit 0
    hits its boundary. Throughout this sweep, all child bins except
    k1 = 2·active_pos[0] and k2 = 2·active_pos[0]+1 are constant.
    This makes the window sum a univariate quadratic in cursor[pos],
    validating the 1D range check.

    Conversely: when digit 0 DOES hit its boundary, gc_focus[0] is set
    to gc_focus[1] ≠ 0, so the next advance picks a higher digit.
    The sweep is over and the range check must NOT be applied (the
    gc_focus[0] == 0 guard in the code ensures this).

    Depends on: GrayCode (Knuth TAOCP 7.2.1.1 Algorithm M) -/
theorem digit_0_consecutive_advance
    (n_active : ℕ) (hn : 0 < n_active)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (radix : Fin n_active → ℕ) (gc_a : Fin n_active → ℕ)
    (gc_dir : Fin n_active → ℤ)
    -- After non-boundary advance of digit 0:
    -- gc_focus[0] was reset to 0, and boundary didn't fire
    (h_not_boundary : gc_a ⟨0, by omega⟩ ≠ 0 ∧
                      gc_a ⟨0, by omega⟩ ≠ radix ⟨0, by omega⟩ - 1)
    (h_focus_reset : gc_focus ⟨0, by omega⟩ = 0) :
    -- Next advance picks digit 0
    gc_focus ⟨0, by omega⟩ = 0 := by
  sorry  -- Immediate from h_focus_reset; real content is the invariant

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART F: Fast-Forward Correctness (Claim 4.46)
--
-- When the range check succeeds, digit 0 is fast-forwarded to its
-- boundary. This involves:
--   (a) Setting gc_a[0] to the boundary value
--   (b) Updating child[k1], child[k2], and raw_conv incrementally
--   (c) Simulating the boundary hit on the Gray code state
-- We must prove the resulting state is identical to what would have
-- been reached by stepping digit 0 one-by-one to the boundary.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.46: The fast-forward produces an identical state to
    step-by-step Gray code advancement.

    Let x_curr be the current cursor value and x_far be the boundary
    value (lo or hi depending on direction). The fast-forward sets
    child[k1] = x_far, child[k2] = a − x_far, and updates raw_conv
    via a single incremental update with delta1 = x_far − x_curr.

    The step-by-step path would apply N = |x_far − x_curr| individual
    ±1 incremental updates. We must show these produce the same raw_conv.

    Key identity: the cross-term contribution of bin j to conv[k1+j] is
      2 · child[k1] · child[j]
    The total delta from x_curr to x_far is:
      2 · (x_far − x_curr) · child[j]
    This is the same whether applied as one step of (x_far−x_curr) or
    N steps of ±1, because child[j] is constant throughout the sweep
    (only bins k1, k2 change) and the updates are linear in delta1.

    Similarly for self-terms:
      new1² − old1² = x_far² − x_curr²
    is the same whether computed in one step or accumulated as
      Σ_{i=1}^{N} (x_curr+i)² − (x_curr+i−1)² = x_far² − x_curr²

    Depends on: IncrementalAutoconv (Claim 4.2) -/
theorem fast_forward_equiv_stepwise
    {d : ℕ} (child_start child_end : Fin d → ℤ)
    (k1 k2 : Fin d) (hk : k2.1 = k1.1 + 1)
    (a : ℤ) (x_curr x_far : ℤ)
    (h_start_k1 : child_start k1 = x_curr)
    (h_start_k2 : child_start k2 = a - x_curr)
    (h_end_k1 : child_end k1 = x_far)
    (h_end_k2 : child_end k2 = a - x_far)
    -- All other bins unchanged
    (h_unchanged : ∀ i : Fin d, i ≠ k1 → i ≠ k2 →
      child_end i = child_start i)
    -- raw_conv updated by single large-step incremental
    (raw_conv_single : Fin (2 * d - 1) → ℤ)
    -- raw_conv updated by N individual ±1 steps
    (raw_conv_stepwise : Fin (2 * d - 1) → ℤ)
    -- Both start from the same raw_conv
    (raw_conv_init : Fin (2 * d - 1) → ℤ) :
    -- Single-step and multi-step produce the same result
    raw_conv_single = raw_conv_stepwise := by
  sorry

end -- noncomputable section
