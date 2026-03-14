/-
PROMPT FOR ARISTOTLE: Prove incremental autoconvolution update (Claim 4.2).

When consecutive children in the odometer differ in only a few positions,
the autoconvolution conv[t] = ∑_{i+j=t} c_i·c_j can be updated incrementally.
The result must be BIT-EXACT (identical to full recomputation).

CORE THEOREM: For c, c' : Fin d → ℤ differing only in positions S:
  conv'[t] - conv[t] = ∑_{i+j=t, i∈S or j∈S} (c'_i·c'_j - c_i·c_j)

The delta decomposes into disjoint groups based on how many indices are in S.

FAST PATH (|S| = 2, positions {2p, 2p+1}):
  Group A: self-term at 2p: (c'_{2p})² - (c_{2p})²  at t = 4p
  Group B: self-term at 2p+1: (c'_{2p+1})² - (c_{2p+1})²  at t = 4p+2
  Group C: mutual term: 2·(c'_{2p}·c'_{2p+1} - c_{2p}·c_{2p+1})  at t = 4p+1
  Group D: cross-terms with unchanged bins

These groups partition all (i,j) pairs where at least one index is in S.

This is pure finite-sum algebra over ℤ. No analysis needed.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

section

/-! ## Definitions -/

def int_autoconvolution {d : ℕ} (c : Fin d → ℤ) (t : ℕ) : ℤ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = t then c i * c j else 0

/-- The delta between new and old autoconvolution. -/
def autoconv_delta {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) : ℤ :=
  int_autoconvolution c' t - int_autoconvolution c t

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
LEMMA 1: The delta splits into terms where at least one index changed.

conv'[t] - conv[t] = ∑_{i+j=t} (c'_i·c'_j - c_i·c_j)

For i,j ∉ S (unchanged): c'_i·c'_j - c_i·c_j = 0.
So only terms with i ∈ S or j ∈ S contribute.

This is just expanding the definition.
-/
theorem delta_eq_sum {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    autoconv_delta c c' t =
    ∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t then c' i * c' j - c i * c j else 0 := by
  sorry

/-
LEMMA 2: Terms where neither index changed contribute zero.

If c'_i = c_i and c'_j = c_j, then c'_i·c'_j - c_i·c_j = 0.
-/
theorem unchanged_terms_zero {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (hi : i ∉ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = 0 := by
  sorry

/-
LEMMA 3: For the fast path (S = {2p, 2p+1}), the delta decomposes as:

  Δconv[t] = (self-term 2p) + (self-term 2p+1) + (mutual) + (cross-terms)

where each group is disjoint.

We prove this by showing: for c, c' agreeing outside {2p, 2p+1}:
  ∑_{i+j=t} (c'_i·c'_j - c_i·c_j)
  = ∑_{i+j=t, i∈S, j∈S} (c'_i·c'_j - c_i·c_j)     [both in S]
  + ∑_{i+j=t, i∈S, j∉S} (c'_i·c'_j - c_i·c_j)     [one in S]
  + ∑_{i+j=t, i∉S, j∈S} (c'_i·c'_j - c_i·c_j)     [one in S, other order]
  + 0                                                 [neither in S]

The "both in S" terms give Groups A, B, C.
The "one in S" terms give Group D (the cross-terms).
-/
theorem delta_three_way_split {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (t : ℕ) :
    autoconv_delta c c' t =
    -- Terms where both i,j ∈ S
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) +
    -- Terms where i ∈ S, j ∉ S
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∈ S ∧ j ∉ S then c' i * c' j - c i * c j else 0) +
    -- Terms where i ∉ S, j ∈ S
    (∑ i : Fin d, ∑ j : Fin d,
      if i.1 + j.1 = t ∧ i ∉ S ∧ j ∈ S then c' i * c' j - c i * c j else 0) := by
  sorry

/-
LEMMA 4: The "one in S" cross-terms simplify.

When i ∈ S and j ∉ S: c'_j = c_j, so
  c'_i·c'_j - c_i·c_j = c'_i·c_j - c_i·c_j = (c'_i - c_i)·c_j

Similarly when i ∉ S and j ∈ S:
  c'_i·c'_j - c_i·c_j = c_i·(c'_j - c_j)
-/
theorem cross_term_simplify {d : ℕ} (c c' : Fin d → ℤ)
    (S : Finset (Fin d)) (hS : ∀ i : Fin d, i ∉ S → c' i = c i)
    (i j : Fin d) (hi : i ∈ S) (hj : j ∉ S) :
    c' i * c' j - c i * c j = (c' i - c i) * c j := by
  sorry

/-
MAIN THEOREM: After applying the incremental delta, we get the correct autoconvolution.

For ANY c, c' (not just fast path):
  int_autoconvolution c t + autoconv_delta c c' t = int_autoconvolution c' t

This is trivially true by definition (delta = new - old, so old + delta = new).
But it confirms the incremental update framework is correct.
-/
theorem incremental_update_correct {d : ℕ} (c c' : Fin d → ℤ) (t : ℕ) :
    int_autoconvolution c t + autoconv_delta c c' t = int_autoconvolution c' t := by
  sorry

/-
COMPLETENESS: The three groups (both-in-S, i-in-S-j-not, i-not-j-in-S) are
exhaustive (together with the zero "neither-in-S" group, they cover all (i,j) pairs).
-/
theorem groups_exhaustive {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    (i ∈ S ∧ j ∈ S) ∨ (i ∈ S ∧ j ∉ S) ∨ (i ∉ S ∧ j ∈ S) ∨ (i ∉ S ∧ j ∉ S) := by
  tauto

/-
DISJOINTNESS: The four groups are mutually exclusive.
-/
theorem groups_disjoint {d : ℕ} (S : Finset (Fin d)) (i j : Fin d) :
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∈ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∈ S)) ∧
    ¬((i ∈ S ∧ j ∉ S) ∧ (i ∉ S ∧ j ∉ S)) ∧
    ¬((i ∉ S ∧ j ∈ S) ∧ (i ∉ S ∧ j ∉ S)) := by
  tauto

end
