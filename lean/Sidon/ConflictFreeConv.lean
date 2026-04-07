/-
Sidon Autocorrelation Project — Conflict-Free Convolution Update (Claims 6.20–6.24)

This file collects the theorems and lemmas certifying the single-phase
conflict-free incremental convolution update implemented in the GPU kernel
(cascade_kernel.cu: incremental_conv_update, lines 90-171).

The CPU code (run_cascade.py) updates convolution entries sequentially when
a Gray code position changes. The GPU uses a warp-parallel single-phase
approach where each thread writes to a UNIQUE conv address, combining both
delta1 (from child[k1] change) and delta2 (from child[k2] change)
contributions in a single pass with no write conflicts.

The key insight: when position pos changes, bins k1=2*pos and k2=2*pos+1
update. For cross-term conv[k1+j] where j ≠ k1, k2:
  - Thread j writes: conv[k1+j] += 2*delta1*child[j]
  - BUT thread j also handles conv[k2+(j-1)] += 2*delta2*child[j-1]
  - These write to k1+j and k2+j-1 = k1+j (since k2 = k1+1), which is
    the SAME address only when j-1 exists, creating a potential conflict.
  - The GPU resolves this by having each thread compute BOTH contributions
    to its unique address k1+j in a single atomic-free write.

Claims covered:
  6.20  Single-phase write addresses are unique per thread
  6.21  Combined delta at each address equals sum of CPU's two-phase deltas
  6.22  Self-term updates (conv[2k1], conv[2k2], conv[k1+k2]) are correct
  6.23  Thread 0 extra address (conv[k1+d_child]) handling
  6.24  End-to-end: single-phase update produces identical conv as CPU

Cross-cutting dependencies:
  - IncrementalAutoconv.lean (Claim 4.2): bit-exact incremental update spec

STATUS: All sorry stubs — proofs not yet attempted.
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
-- Definitions
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Two-phase CPU delta: the change to conv[idx] from phase 1 (k1 cross-terms)
    and phase 2 (k2 cross-terms), computed separately. -/
def cpu_two_phase_delta {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ)
    (delta1 delta2 : ℤ) (idx : ℕ) : ℤ :=
  -- Phase 1: delta1 contributions (from k1)
  let phase1 := ∑ j : Fin d,
    if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k1 + j.1 = idx then 2 * delta1 * child j else 0
  -- Phase 2: delta2 contributions (from k2)
  let phase2 := ∑ j : Fin d,
    if j.1 ≠ k1 ∧ j.1 ≠ k2 ∧ k2 + j.1 = idx then 2 * delta2 * child j else 0
  phase1 + phase2

/-- Single-phase GPU delta: the combined change to conv[k1+lane] from the
    GPU's merged write, where lane handles both delta1*child[lane] and
    delta2*child[lane-1] contributions. -/
def gpu_single_phase_delta {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ)
    (delta1 delta2 : ℤ) (lane : ℕ) : ℤ :=
  let idx := k1 + lane
  -- Contribution from delta1: child[lane] if lane ∉ {k1, k2}
  let c1 := if lane ≠ k1 ∧ lane ≠ k2 ∧ lane < d
             then 2 * delta1 * child ⟨lane, by omega⟩ else 0
  -- Contribution from delta2: child[lane-1] if lane-1 ∉ {k1, k2} and idx = k2+(lane-1)
  let c2 := if 0 < lane ∧ lane - 1 ≠ k1 ∧ lane - 1 ≠ k2 ∧ lane - 1 < d
             then 2 * delta2 * child ⟨lane - 1, by omega⟩ else 0
  c1 + c2

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART A: Address Uniqueness (Claim 6.20)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.20: Each thread writes to a unique address k1+lane.
    No two threads share the same output address in the single-phase approach.

    This is trivially true because different lanes have different values,
    so k1+lane₁ ≠ k1+lane₂ when lane₁ ≠ lane₂.

    Matches: cascade_kernel.cu incremental_conv_update — each thread writes
    to conv[k1+lane] where lane = threadIdx.x. -/
theorem single_phase_addresses_unique (k1 : ℕ) (lane1 lane2 : ℕ)
    (h_diff : lane1 ≠ lane2) :
    k1 + lane1 ≠ k1 + lane2 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART B: Delta Equivalence (Claims 6.21, 6.22, 6.23)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.21: The GPU's single-phase combined delta at address k1+lane equals
    the CPU's two-phase delta at the same address, for cross-term addresses.

    The CPU computes:
      Phase 1: for each j ≠ k1,k2: conv[k1+j] += 2*delta1*child[j]
      Phase 2: for each j ≠ k1,k2: conv[k2+j] += 2*delta2*child[j]
    Note: k2+j = k1+1+j = k1+(j+1), so phase 2's write to conv[k2+j]
    hits address k1+(j+1), i.e., the GPU lane j+1.

    The GPU computes (for lane ∉ {k1, k2, k1-k1=0 special cases}):
      conv[k1+lane] += 2*delta1*child[lane] + 2*delta2*child[lane-1]
    which combines both phases into a single write.

    Matches: cascade_kernel.cu incremental_conv_update lines 117-156. -/
theorem single_phase_equals_two_phase
    {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ) (hk2 : k2 = k1 + 1)
    (delta1 delta2 : ℤ) (lane : ℕ) (hlane : lane < d)
    (h_not_self : k1 + lane ≠ 2 * k1 ∧ k1 + lane ≠ 2 * k2 ∧ k1 + lane ≠ k1 + k2) :
    gpu_single_phase_delta child k1 k2 delta1 delta2 lane =
    cpu_two_phase_delta child k1 k2 delta1 delta2 (k1 + lane) := by
  sorry

/-- Claim 6.22: Self-term updates are computed correctly.
    The self-term entries conv[2k1], conv[2k2], conv[k1+k2] are handled
    specially by the GPU (not through the cross-term lanes).

    conv[2k1] += (c_old1 + delta1)² - c_old1² = delta1*(2*c_old1 + delta1)
    conv[2k2] += (c_old2 + delta2)² - c_old2² = delta2*(2*c_old2 + delta2)
    conv[k1+k2] += 2*(c_old1+delta1)*(c_old2+delta2) - 2*c_old1*c_old2

    Matches: cascade_kernel.cu incremental_conv_update lines 100-115. -/
theorem self_term_updates_correct
    (c_old1 c_old2 delta1 delta2 : ℤ) :
    ((c_old1 + delta1)^2 - c_old1^2 = delta1 * (2 * c_old1 + delta1)) ∧
    ((c_old2 + delta2)^2 - c_old2^2 = delta2 * (2 * c_old2 + delta2)) ∧
    (2 * (c_old1 + delta1) * (c_old2 + delta2) - 2 * c_old1 * c_old2 =
      2 * (delta1 * c_old2 + delta2 * c_old1 + delta1 * delta2)) := by
  sorry

/-- Claim 6.23: Thread 0's extra address handling.
    Thread 0 additionally handles conv[k1+d_child] = conv[k2+(d_child-1)],
    which is the last k2 cross-term that has no corresponding k1 cross-term
    from a higher lane. Only delta2*child[d_child-1] contributes.

    Matches: cascade_kernel.cu incremental_conv_update lines 158-164
    (thread 0 handles the "spill" address). -/
theorem thread0_extra_address
    {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ) (hk2 : k2 = k1 + 1)
    (hd : 0 < d) (delta2 : ℤ)
    (h_last : d - 1 ≠ k1 ∧ d - 1 ≠ k2) :
    cpu_two_phase_delta child k1 k2 0 delta2 (k1 + d) =
      if d - 1 < d then 2 * delta2 * child ⟨d - 1, by omega⟩ else 0 := by
  sorry

-- ═══════════════════════════════════════════════════════════════════════════════
-- PART C: End-to-End (Claim 6.24)
-- ═══════════════════════════════════════════════════════════════════════════════

/-- Claim 6.24: The GPU's single-phase incremental conv update produces the
    identical conv array as the CPU's sequential two-phase update.

    ∀ idx, conv_gpu[idx] = conv_cpu[idx] after the update.

    Proof sketch: For self-term indices (2k1, 2k2, k1+k2), both compute
    the same algebraic delta (Claim 6.22). For cross-term indices k1+lane,
    the GPU's merged write equals the CPU's two-phase sum (Claim 6.21).
    For indices outside the update range, both leave conv unchanged.

    Matches: cascade_kernel.cu incremental_conv_update — the entire function. -/
theorem single_phase_end_to_end
    {d : ℕ} (child : Fin d → ℤ) (k1 k2 : ℕ) (hk2 : k2 = k1 + 1)
    (hk1d : k1 < d) (hk2d : k2 < d)
    (delta1 delta2 : ℤ)
    (conv_old : ℕ → ℤ)
    (h_conv : ∀ idx, conv_old idx =
      ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = idx then child i * child j else 0)
    (child' : Fin d → ℤ)
    (h_k1 : child' ⟨k1, hk1d⟩ = child ⟨k1, hk1d⟩ + delta1)
    (h_k2 : child' ⟨k2, hk2d⟩ = child ⟨k2, hk2d⟩ + delta2)
    (h_rest : ∀ i : Fin d, i.1 ≠ k1 ∧ i.1 ≠ k2 → child' i = child i) :
    ∀ idx,
      (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = idx then child' i * child' j else 0) =
      conv_old idx + cpu_two_phase_delta child k1 k2 delta1 delta2 idx
      + (if idx = 2 * k1 then delta1 * (2 * child ⟨k1, hk1d⟩ + delta1) else 0)
      + (if idx = 2 * k2 then delta2 * (2 * child ⟨k2, hk2d⟩ + delta2) else 0)
      + (if idx = k1 + k2 then
          2 * (delta1 * child ⟨k2, hk2d⟩ + delta2 * child ⟨k1, hk1d⟩ + delta1 * delta2)
         else 0) := by
  sorry

end -- noncomputable section
