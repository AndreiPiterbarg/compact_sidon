/-
PROMPT FOR ARISTOTLE: Prove autoconvolution reversal symmetry (Claims 3.3a + 3.3e).

These are pure finite-sum index-substitution proofs. No measure theory needed.

CLAIM 3.3a: conv[k](c) = conv[2d-2-k](rev(c))
  where rev(c)_i = c_{d-1-i} and conv[k] = ∑_{i+j=k} c_i · c_j.

  Proof: Substitute i' = d-1-i, j' = d-1-j in the sum.
  Then i'+j' = 2d-2-(i+j) = 2d-2-k, and c_{d-1-i'}·c_{d-1-j'} = rev(c)_{i'}·rev(c)_{j'}.

CLAIM 3.3e: left_frac(rev(c)) = 1 - left_frac(c)
  where left_frac(c) = (∑_{i<n} c_i) / m for c summing to m with d = 2n bins.

  Proof: ∑_{i<n} rev(c)_i = ∑_{i<n} c_{2n-1-i} = ∑_{i=n}^{2n-1} c_i = m - ∑_{i<n} c_i.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-! ## Definitions -/

def discrete_autoconvolution {d : ℕ} (a : Fin d → ℝ) (k : ℕ) : ℝ :=
  ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then a i * a j else 0

def rev_vector {d : ℕ} (c : Fin d → ℕ) : Fin d → ℕ :=
  fun i => c ⟨d - 1 - i.1, by omega⟩

def rev_vector_real {d : ℕ} (a : Fin d → ℝ) : Fin d → ℝ :=
  fun i => a ⟨d - 1 - i.1, by omega⟩

/-! ## ============================================================
    THEOREMS TO PROVE
    ============================================================ -/

/-
CLAIM 3.3a: Autoconvolution reversal symmetry.

conv[k](a) = conv[2d-2-k](rev(a))

Proof:
  conv[2d-2-k](rev(a))
  = ∑_{i+j = 2d-2-k} rev(a)_i · rev(a)_j
  = ∑_{i+j = 2d-2-k} a_{d-1-i} · a_{d-1-j}

  Substitute i' = d-1-i, j' = d-1-j (this is a bijection on Fin d):
  i' + j' = (d-1-i) + (d-1-j) = 2d-2-(i+j)

  When i+j = 2d-2-k: i'+j' = 2d-2-(2d-2-k) = k.

  So the sum becomes ∑_{i'+j' = k} a_{i'} · a_{j'} = conv[k](a).
-/
theorem autoconv_reversal_symmetry {d : ℕ} (hd : d > 0) (a : Fin d → ℝ)
    (k : ℕ) (hk : k ≤ 2 * d - 2) :
    discrete_autoconvolution a k =
    discrete_autoconvolution (rev_vector_real a) (2 * d - 2 - k) := by
  sorry

/-
CLAIM 3.3e: Left-half mass reversal.

For c : Fin (2*n) → ℕ with ∑ c_i = m:
  (∑_{i<n} c_i) + (∑_{i<n} c_{2n-1-i}) = m

Equivalently: left_sum(c) + left_sum(rev(c)) = m.

Proof:
  ∑_{i<n} c_{2n-1-i}  (substitute j = 2n-1-i, j ranges from n to 2n-1)
  = ∑_{j=n}^{2n-1} c_j
  = (∑_{j=0}^{2n-1} c_j) - (∑_{j=0}^{n-1} c_j)
  = m - ∑_{i<n} c_i.
-/
theorem left_sum_reversal (n : ℕ) (hn : n > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) (m : ℕ) :
    (∑ i : Fin n, c ⟨i.1, by omega⟩) +
    (∑ i : Fin n, c ⟨2 * n - 1 - i.1, by omega⟩) = m := by
  sorry

/-
COROLLARY: The asymmetry condition is symmetric under reversal.

If left_frac(c) = L, then left_frac(rev(c)) = 1 - L.
The pruning condition checks BOTH L ≥ threshold AND 1-L ≥ threshold.
So if c is pruned by asymmetry, rev(c) is also pruned by asymmetry.
-/
theorem asymmetry_reversal_symmetric (n : ℕ) (hn : n > 0) (m : ℕ) (hm : 0 < m)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m)
    (threshold : ℝ) (ht : 0 ≤ threshold) (ht1 : threshold ≤ 1)
    (L : ℝ) (hL : L = (∑ i : Fin n, (c ⟨i.1, by omega⟩ : ℝ)) / m)
    (h_prune : L ≥ threshold ∨ 1 - L ≥ threshold) :
    let L_rev := (∑ i : Fin n, (c ⟨2 * n - 1 - i.1, by omega⟩ : ℝ)) / m
    L_rev ≥ threshold ∨ 1 - L_rev ≥ threshold := by
  sorry

end
