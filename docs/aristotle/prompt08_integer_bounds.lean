/-
PROMPT FOR ARISTOTLE: Prove integer autoconvolution bounds (Claims 4.8 + 5.2).

These are bounds on finite sums of natural numbers. No analysis needed.

Setup: c : Fin d → ℕ with ∑ c_i = m.
Autoconvolution: conv[k] = ∑_{i+j=k} c_i · c_j (integer, exact).

CLAIM 4.8a: Each conv entry is bounded by m².
  conv[k] ≤ m²

CLAIM 4.8b: The total autoconvolution equals m².
  ∑_k conv[k] = (∑ c_i)² = m²

CLAIM 4.8c: For m ≤ 200, m² fits in int32.
  m² ≤ 2³¹ - 1

CLAIM 5.2: Integer autoconvolution is exact.
  Since c_i ∈ ℤ, products c_i·c_j ∈ ℤ, and sums of integers are integers.
  (This is trivially true for ℤ-valued functions in Lean.)
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

def nat_autoconvolution {d : ℕ} (c : Fin d → ℕ) (k : ℕ) : ℕ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
CLAIM 4.8b: Total autoconvolution = m².

∑_{k=0}^{2d-2} conv[k] = ∑_k ∑_{i+j=k} c_i·c_j = ∑_i ∑_j c_i·c_j = (∑ c_i)² = m².

Proof: Swap the order of summation. For each (i,j) pair, c_i·c_j appears
in exactly one conv[k] (namely k = i+j). So the total sum over all k
equals ∑_i ∑_j c_i·c_j = (∑_i c_i)·(∑_j c_j) = m·m = m².
-/
theorem conv_total_eq_sq {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) :
    ∑ k ∈ Finset.range (2 * d - 1),
      nat_autoconvolution c k = m ^ 2 := by
  sorry

/-
CLAIM 4.8a: Each conv entry is bounded by m².

conv[k] = ∑_{i+j=k} c_i·c_j ≤ ∑_{all k} conv[k] = m².

Proof: conv[k] is a partial sum of nonneg terms. The total sum = m².
Since all terms are nonneg, each partial sum ≤ total.

Alternatively: conv[k] = ∑_{i+j=k} c_i·c_j.
For each i with i ≤ k and k-i < d: c_i·c_{k-i} ≤ c_i · m (since c_j ≤ m for all j).
So conv[k] ≤ m · ∑_i c_i = m · m = m².
-/
theorem conv_entry_bounded {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    nat_autoconvolution c k ≤ m ^ 2 := by
  sorry

/-
CLAIM 4.8c: For m ≤ 200, m² fits in int32 (2³¹ - 1 = 2147483647).
-/
theorem m_sq_fits_int32 (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by
  omega

/-
Additional: prefix sums are bounded by m².

prefix[k] = ∑_{t=0}^{k} conv[t] ≤ ∑_{all t} conv[t] = m².
-/
theorem prefix_sum_bounded {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m) (k : ℕ) :
    ∑ t ∈ Finset.range (k + 1), nat_autoconvolution c t ≤ m ^ 2 := by
  sorry

/-
Additional: window sums are bounded by m².

ws = prefix[b] - prefix[a] ≤ m².
Since both prefix sums are ≤ m² and ≥ 0, the difference is ≤ m².
-/
theorem window_sum_bounded {d : ℕ} (c : Fin d → ℕ) (m : ℕ) (hc : ∑ i, c i = m)
    (a b : ℕ) (hab : a ≤ b) :
    ∑ t ∈ Finset.Ico a (b + 1), nat_autoconvolution c t ≤ m ^ 2 := by
  sorry

end
