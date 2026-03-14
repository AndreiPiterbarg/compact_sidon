/-
PROMPT FOR ARISTOTLE: Prove the contributing bins formula (Claim 1.4).

Bin i contributes to window (ℓ, s_lo) iff there exists j ∈ [0, d-1] such that
s_lo ≤ i + j ≤ s_lo + ℓ - 2.

CLAIM: This is equivalent to:
  max(0, s_lo - (d-1)) ≤ i ≤ min(d-1, s_lo + ℓ - 2)

Proof (⇒): Given j with s_lo ≤ i+j ≤ s_lo+ℓ-2 and 0 ≤ j ≤ d-1:
  i = (i+j) - j ≥ s_lo - (d-1)      (since j ≤ d-1)
  i ≤ i+j ≤ s_lo + ℓ - 2            (since j ≥ 0)
  Also 0 ≤ i ≤ d-1 by hypothesis.

Proof (⇐): Given max(0, s_lo-(d-1)) ≤ i ≤ min(d-1, s_lo+ℓ-2):
  Choose j = clamp(s_lo - i, 0, d-1).
  Case s_lo ≤ i: j = 0, i+j = i ≥ s_lo. Also i ≤ s_lo+ℓ-2. ✓
  Case s_lo > i and s_lo - i < d: j = s_lo - i, i+j = s_lo. ✓
  Case s_lo - i ≥ d: j = d-1, i+j = i+d-1 ≥ s_lo (since i ≥ s_lo-(d-1)). ✓

This is pure logic/arithmetic over naturals.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

section

/-! ## Definitions -/

/-- Bin i contributes to window (ℓ, s_lo) if there exists j ∈ [0, d-1]
    such that s_lo ≤ i + j ≤ s_lo + ℓ - 2. -/
def bin_contributes (d : ℕ) (i : Fin d) (ℓ s_lo : ℕ) : Prop :=
  ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2

/-- The set of all contributing bins. -/
def contributing_bins_set (d : ℕ) (ℓ s_lo : ℕ) : Finset (Fin d) :=
  Finset.filter (fun i => ∃ j : Fin d, s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 ≤ s_lo + ℓ - 2)
    Finset.univ

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
FORWARD DIRECTION (⇒):
If bin i contributes (∃ j with s_lo ≤ i+j ≤ s_lo+ℓ-2),
then max(0, s_lo - (d-1)) ≤ i ≤ min(d-1, s_lo + ℓ - 2).
-/
theorem contributing_forward (d : ℕ) (hd : d > 0) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (i : Fin d) (h : bin_contributes d i ℓ s_lo) :
    (if s_lo ≤ d - 1 then 0 else s_lo - (d - 1)) ≤ i.1 ∧
    i.1 ≤ min (d - 1) (s_lo + ℓ - 2) := by
  sorry

/-
BACKWARD DIRECTION (⇐):
If max(0, s_lo - (d-1)) ≤ i ≤ min(d-1, s_lo + ℓ - 2),
then bin i contributes (we can construct the witness j).
-/
theorem contributing_backward (d : ℕ) (hd : d > 0) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (i : Fin d)
    (h_lo : (if s_lo ≤ d - 1 then 0 else s_lo - (d - 1)) ≤ i.1)
    (h_hi : i.1 ≤ min (d - 1) (s_lo + ℓ - 2)) :
    bin_contributes d i ℓ s_lo := by
  sorry

/-
COMBINED: The full iff characterization.
-/
theorem contributing_bins_iff (d : ℕ) (hd : d > 0) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ)
    (i : Fin d) :
    bin_contributes d i ℓ s_lo ↔
    (if s_lo ≤ d - 1 then 0 else s_lo - (d - 1)) ≤ i.1 ∧
    i.1 ≤ min (d - 1) (s_lo + ℓ - 2) := by
  sorry

/-
COROLLARY: The contributing bin range is a contiguous interval [lo_bin, hi_bin].
-/
theorem contributing_bins_contiguous (d : ℕ) (hd : d > 0) (ℓ s_lo : ℕ) (hℓ : 2 ≤ ℓ) :
    let lo_bin := if s_lo ≤ d - 1 then 0 else s_lo - (d - 1)
    let hi_bin := min (d - 1) (s_lo + ℓ - 2)
    contributing_bins_set d ℓ s_lo =
    Finset.filter (fun i : Fin d => lo_bin ≤ i.1 ∧ i.1 ≤ hi_bin) Finset.univ := by
  sorry

end
