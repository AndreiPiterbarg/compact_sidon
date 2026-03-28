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

STATUS: PROOF OBLIGATIONS ONLY — no proofs are attempted here.
Each `sorry` marks an open obligation. Dependencies on existing modules
(Defs, IncrementalAutoconv, GrayCode, DynamicThreshold) are noted.
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
    (s_lo ell : ℕ) (hell : 2 ≤ ell) :
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
    (A : ℤ := (if s_lo ≤ 4 * pos ∧ 4 * pos ≤ s_lo + ell - 2 then 1 else 0) +
              (if s_lo ≤ 4 * pos + 2 ∧ 4 * pos + 2 ≤ s_lo + ell - 2 then 1 else 0) -
              (if s_lo ≤ 4 * pos + 1 ∧ 4 * pos + 1 ≤ s_lo + ell - 2 then 2 else 0)) :
    A ∈ ({-2, -1, 0, 1} : Set ℤ) := by
  sorry

/-- Claim 4.38: A = 0 whenever all three conv indices fall inside
    the window, i.e., when 4p ≥ s_lo and 4p+2 ≤ s_lo + ℓ − 2.

    This is the common case: the window sum becomes LINEAR in x,
    simplifying the range check to an endpoint evaluation.

    Under hypotheses h1, h2 all three indicator conditions hold
    (the intermediate index 4p+1 is contained by omega), so the
    A formula from Claim 4.37 reduces to 1 + 1 − 2 = 0. -/
theorem quadratic_coeff_zero_when_contained
    (pos : ℕ) (s_lo ell : ℕ)
    (h1 : s_lo ≤ 4 * pos)
    (h2 : 4 * pos + 2 ≤ s_lo + ell - 2) :
    (if s_lo ≤ 4 * pos ∧ 4 * pos ≤ s_lo + ell - 2 then (1 : ℤ) else 0) +
    (if s_lo ≤ 4 * pos + 2 ∧ 4 * pos + 2 ≤ s_lo + ell - 2 then 1 else 0) -
    (if s_lo ≤ 4 * pos + 1 ∧ 4 * pos + 1 ≤ s_lo + ell - 2 then 2 else 0) = 0 := by
  sorry

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
    (A B C : ℝ) (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi) :
    let f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C
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
    (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi) :
    let f : ℤ → ℝ := fun x => A * (x : ℝ) ^ 2 + B * (x : ℝ) + C
    f x_lo > 0 → f x_hi > 0 →
    ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → f x > 0 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART E: Digit 0 Consecutive Advance Property (Claim 4.45)
--
-- The sweep skip is only valid because digit 0 advances in consecutive
-- steps. We must prove that between a non-boundary advance of digit 0
-- and the next boundary hit, ONLY digit 0 changes.
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Model of the Gray code focus-pointer update after advancing digit j.

    Matches the Knuth mixed-radix Gray code (TAOCP 7.2.1.1) as
    implemented in run_cascade.py lines 1269–1282:
      Step 1: gc_focus[0] = 0                       (reset)
      Step 2: if boundary hit:
                gc_focus[j] = gc_focus[j + 1]        (propagate)
                gc_focus[j + 1] = j + 1              (restore)

    The `hit_boundary` flag corresponds to the code's check
    `gc_a[j] == 0 or gc_a[j] == radix[j] - 1`. -/
def gc_focus_update (n_active : ℕ)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (j : Fin n_active)
    (hit_boundary : Bool) : Fin (n_active + 1) → ℕ :=
  fun i =>
    -- Step 1: reset gc_focus[0] to 0
    let f0 : Fin (n_active + 1) → ℕ :=
      fun k => if k.1 = 0 then 0 else gc_focus k
    -- Step 2: if boundary, propagate focus chain
    if hit_boundary then
      if i.1 = j.1 then f0 ⟨j.1 + 1, by omega⟩
      else if i.1 = j.1 + 1 then j.1 + 1
      else f0 i
    else f0 i

/-- Claim 4.45: After the Gray code advance step, gc_focus'[0] = 0
    whenever digit 0 did NOT hit its boundary.

    Case analysis on (j, hit_boundary):
    • hit_boundary = false: gc_focus'[0] = f0[0] = 0.            ✓
    • hit_boundary = true, j > 0: boundary updates indices j and
      j+1 (both ≥ 1), so gc_focus'[0] = f0[0] = 0.              ✓
    • hit_boundary = true, j = 0: gc_focus'[0] = f0[1] ≠ 0.
      Excluded by hypothesis h.

    Consequence: after digit 0 advances without hitting a boundary,
    the next advance is also digit 0. This continues until digit 0
    hits its boundary. Throughout this sweep, all child bins except
    k1 = 2·active_pos[0] and k2 = 2·active_pos[0]+1 are constant,
    validating the 1D quadratic range check.

    Depends on: GrayCode (Knuth TAOCP 7.2.1.1 Algorithm M) -/
theorem digit_0_consecutive_advance
    (n_active : ℕ) (hn : 0 < n_active)
    (gc_focus : Fin (n_active + 1) → ℕ)
    (j : Fin n_active)
    (hit_boundary : Bool)
    -- If digit 0 was advanced, it did not hit its boundary
    (h : j.1 = 0 → hit_boundary = false) :
    gc_focus_update n_active gc_focus j hit_boundary ⟨0, by omega⟩ = 0 := by
  sorry

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

/-- Claim 4.46: The fast-forward produces an identical raw_conv to
    step-by-step Gray code advancement.

    Let x_curr be the current cursor value and x_far be the boundary
    value. The fast-forward applies a single incremental update with
    delta1 = x_far − x_curr. The step-by-step path applies N = |x_far −
    x_curr| individual ±1 updates through intermediate child configs.

    These produce the same raw_conv because autoconv_delta telescopes:
      Σ_{i=0}^{N-1} autoconv_delta(c_i, c_{i+1})
        = Σ_{i=0}^{N-1} (int_autoconvolution(c_{i+1}) − int_autoconvolution(c_i))
        = int_autoconvolution(c_N) − int_autoconvolution(c_0)
        = autoconv_delta(c_0, c_N)

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
    -- Intermediate child configurations (cursor stepping from x_curr to x_far)
    (N : ℕ)
    (children : Fin (N + 1) → (Fin d → ℤ))
    (h_first : children ⟨0, by omega⟩ = child_start)
    (h_last : children ⟨N, by omega⟩ = child_end)
    -- Starting raw_conv
    (raw_conv_init : Fin (2 * d - 1) → ℤ)
    -- Single-step: one incremental update from child_start to child_end
    (raw_conv_single : Fin (2 * d - 1) → ℤ)
    (h_single : ∀ t : Fin (2 * d - 1),
      raw_conv_single t = raw_conv_init t +
        autoconv_delta child_start child_end t.1)
    -- Multi-step: accumulated N individual incremental updates
    (raw_conv_stepwise : Fin (2 * d - 1) → ℤ)
    (h_stepwise : ∀ t : Fin (2 * d - 1),
      raw_conv_stepwise t = raw_conv_init t +
        ∑ i : Fin N, autoconv_delta
          (children i.castSucc) (children i.succ) t.1) :
    -- Single-step and multi-step produce the same result
    raw_conv_single = raw_conv_stepwise := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART G: Top-Level Soundness and Post-Fast-Forward (Claims 4.47–4.48)
--
-- These claims compose the pieces from Parts A–F into the two top-level
-- properties needed by the cascade prover:
--   (1) The sweep skip correctly prunes all children in the range.
--   (2) After the fast-forward, raw_conv is the autoconvolution of child_end.
--
-- Cross-cutting concerns NOT formalized here (covered elsewhere or by
-- code-level inspection):
--   - Quick-check invalidation: after fast-forward, the code resets
--     qc_ell = -1, so the quick-check cache is not used stale.
--   - nz_list update: the fast-forward applies the same incremental
--     update as a normal Gray code step, triggering the same nz_list
--     maintenance (see SparseCrossTerm.lean Claim 4.33).
--   - W_int update: handled by the same per-bin delta logic as normal
--     steps (see Claim 4.39 for the affine structure).
--   - Edge case x_curr = x_far: when the cursor is already at the
--     boundary, gc_focus[0] ≠ 0 (Claim 4.45), so the sweep skip
--     guard prevents activation.
--   - Single killing window: if one window prunes all children in the
--     range, each child is pruned (only one window is needed per child).
--   - Threshold formula: the sweep skip uses the same dyn_it formula as
--     per-child pruning (see DynamicThreshold.lean).
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 4.47: Top-level sweep skip soundness.

    If ws(x) > dyn(x) for all x ∈ [x_lo, x_hi] (the D(x) > 0 condition),
    then every child in the sweep satisfies the integer pruning condition
    ws(x) > ⌊dyn(x) · (1 − ε)⌋.

    This is a pointwise application of Claim 4.41. The range check
    (Claims 4.43/4.44) establishes the ∀x premise by evaluating D at
    candidate points.

    Depends on: Claims 4.41, 4.43, 4.44 -/
theorem sweep_skip_sound
    (ws : ℤ → ℤ) (dyn : ℤ → ℝ) (eps : ℝ)
    (heps : 0 < eps) (heps1 : eps < 1)
    (x_lo x_hi : ℤ) (hlo : x_lo ≤ x_hi)
    (h_dyn_pos : ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → 0 ≤ dyn x)
    (h_D_pos : ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → (ws x : ℝ) > dyn x) :
    ∀ x : ℤ, x_lo ≤ x → x ≤ x_hi → ws x > ⌊dyn x * (1 - eps)⌋ := by
  sorry

/-- Claim 4.48: After fast-forward, raw_conv equals the autoconvolution
    of child_end (the child at the boundary cursor value).

    This ensures subsequent pruning tests (after the boundary hit triggers
    an outer digit advance) start from a correct raw_conv state.

    Proof:
      raw_conv_final = raw_conv_init + autoconv_delta(start, end)
        = int_autoconvolution(start) + (int_autoconvolution(end) − int_autoconvolution(start))
        = int_autoconvolution(end)

    Depends on: Claim 4.46, IncrementalAutoconv.incremental_update_correct -/
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
  sorry

end -- noncomputable section
