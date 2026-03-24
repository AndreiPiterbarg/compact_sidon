"""Submit round 3 of focused prompts to Aristotle API.

Each prompt isolates a specific error cluster in complete_proof.lean
with precise error messages, root cause analysis, and fix suggestions.
"""
import json
import os
import sys
import tarfile
import time

import httpx

API_URL = "https://aristotle.harmonic.fun/api/v2"
API_KEY = os.environ.get("ARISTOTLE_KEY", "").strip()
if not API_KEY:
    # Read from .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ARISTOTLE_KEY"):
                    API_KEY = line.split("=", 1)[1].strip()
                    break

LEAN_DIR = os.path.dirname(os.path.abspath(__file__))
TAR_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "lean_project_v3.tar.gz")
PROJECTS_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "aristotle_projects_v3.json")

# ─────────────────────────────────────────────────────────────────────
# PROMPTS — 9 isolated, focused prompts
# ─────────────────────────────────────────────────────────────────────
PROMPTS = [
    # ════════════════════════════════════════════════════════════════
    # P1: step_function_continuousAt — Cases 1 & 2 (lines 1774-1786)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P1_step_func_cases12",
        "prompt": r"""Fix the proof of `step_function_continuousAt` in complete_proof.lean (line 1765), specifically Cases 1 and 2 (lines 1774–1786). Do NOT touch Case 3 (lines 1787–1822) or any other lemma.

## step_function definition (line 1384):
```lean
noncomputable def step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ → ℝ :=
  fun x =>
    let d := 2 * n
    let δ := 1 / (4 * n : ℝ)
    if x < -1/4 ∨ x ≥ 1/4 then 0
    else
      let i := ⌊(x + 1/4) / δ⌋.toNat
      if h : i < d then (c ⟨i, h⟩ : ℝ) / m
      else 0
```

## Errors in Case 1 (x < -1/4, line 1774):
```
error: line 1774:75: unsolved goals
```
Current code:
```lean
· exact Filter.eventually_of_mem (isOpen_Iio.mem_nhds hx_lt) fun y hy => by
    have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inl hy
    have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inl hx_lt
    simp only [step_function, h1, h2, ↓reduceIte]
```
**Root cause**: `simp only [step_function, ↓reduceIte]` does not close the goal because `step_function` unfolds to a `let`-expression with `if x < -1/4 ∨ x ≥ 1/4 then 0 else ...`. The `↓reduceIte` does not handle hypotheses `h1`/`h2` of disjunction type.

**Fix**: After unfolding, use `if_pos` to resolve the if-condition:
```lean
· exact Filter.eventually_of_mem (IsOpen.mem_nhds isOpen_Iio hx_lt) fun y hy => by
    have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inl hy
    have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inl hx_lt
    simp only [step_function]; rw [if_pos h1, if_pos h2]
```
Or use `simp [step_function, h1, h2]` (without `only`).

## Errors in Case 2 (x ≥ 1/4, lines 1780-1786):
```
error: line 1782:8: Application type mismatch
error: line 1782:31: unsolved goals
error: line 1783:77: unsolved goals
```
Current code:
```lean
· have hx_gt : x > 1/4 := lt_of_le_of_ne hx_ge (Ne.symm
    (hx ⟨2 * n, by omega⟩ (by field_simp; ring)))
  exact Filter.eventually_of_mem (isOpen_Ioi.mem_nhds hx_gt) fun y hy => by
    have h1 : y < -(1/4 : ℝ) ∨ y ≥ 1/4 := Or.inr (le_of_lt hy)
    have h2 : x < -(1/4 : ℝ) ∨ x ≥ 1/4 := Or.inr hx_ge
    simp only [step_function, h1, h2, ↓reduceIte]
```
**Root cause for line 1782**: `hx ⟨2 * n, by omega⟩` has type `x ≠ -(1/4) + ↑(2*n) / (4 * ↑n)`, which is `(x = ...) → False`. The second argument `(by field_simp; ring)` tries to prove `x = -(1/4) + (2*n)/(4*n)`, which is nonsensical. The code structure is wrong.

**Fix for hx_gt**: We need `x ≠ 1/4`. We know `hx ⟨2*n, ...⟩ : x ≠ -(1/4) + (2*n)/(4*n)`. Since `-(1/4) + (2*n)/(4*n) = 1/4`, rewrite:
```lean
· have hx_gt : x > 1/4 := by
    refine lt_of_le_of_ne hx_ge (fun h_eq => ?_)
    have := hx ⟨2 * n, by omega⟩
    apply this; rw [h_eq]; push_cast; field_simp; ring
```
**Fix for line 1786** (same ↓reduceIte issue as Case 1): use `simp [step_function, h1, h2]` or `rw [if_pos h1, if_pos h2]`.

Please provide the corrected proof for Cases 1 and 2 only (the two `by_cases` branches before `· -- Case 3`). Keep the lemma statement and Case 3 unchanged.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P2: step_function_continuousAt — Case 3 (lines 1787-1822)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P2_step_func_case3",
        "prompt": r"""Fix the proof of `step_function_continuousAt` in complete_proof.lean (line 1765), specifically Case 3 (lines 1787–1822). Do NOT touch Cases 1–2 or other lemmas.

## Context
Case 3 handles the interior: `-1/4 < x < 1/4`, x not a bin boundary. The proof shows the floor function `⌊(x+1/4)/δ⌋` (where δ = 1/(4n)) is locally constant at non-integer points, hence step_function is locally constant.

Hypothesis: `hx : ∀ k : Fin (2 * n + 1), x ≠ -(1/4 : ℝ) + ↑k.val / (4 * ↑n)`

## Errors (7 total):

### Error 1 — Line 1790: `hx ⟨0, by omega⟩ (by simp)` — type mismatch
```
error: line 1790:17: Application type mismatch
error: line 1790:36: unsolved goals
```
`hx ⟨0, by omega⟩` has type `x ≠ -(1/4) + 0/(4*n)`, i.e., `(x = -(1/4) + 0/(4*n)) → False`. The `(by simp)` is incorrectly passed as a proof of `x = ...`.

**Fix**: We need `-(1/4) < x`, i.e., `x ≠ -1/4`. Use:
```lean
have hx_lo : -(1/4 : ℝ) < x := lt_of_le_of_ne hx_lt (fun h_eq => by
  have := hx ⟨0, by omega⟩; apply this; rw [← h_eq]; simp [Nat.cast_zero])
```

### Error 2 — Line 1797: `field_simp` made no progress
```
error: line 1797:10: field_simp made no progress at hz
```
`hz : α = ↑z` where `α := (x + 1/4) / δ`, `δ := 1 / (4 * ↑n)`. Goal: `x = -(1/4) + (z : ℝ) / (4 * ↑n)`. The `δ` and `α` are local `set` definitions that field_simp doesn't unfold.

**Fix**: Unfold the definitions first:
```lean
have hx_eq : x = -(1/4 : ℝ) + (z : ℝ) / (4 * ↑n) := by
  have h4n_ne : (4 * (↑n : ℝ)) ≠ 0 := by positivity
  have : α = (x + 1/4) * (4 * ↑n) := by simp [α, δ]; field_simp [h4n_ne]
  rw [hz] at this; linarith
```

### Error 3 — Line 1800: `positivity` fails on `0 < α`
```
error: line 1800:39: failed to prove positivity/nonnegativity/nonzeroness
```
`α = (x + 1/4) / δ`. Positivity can't see through the `set` definitions.

**Fix**: Prove `0 < α` from `hx_lo` and `δ > 0`:
```lean
have hδ_pos : (0 : ℝ) < δ := by simp [δ]; positivity
have hα_pos : 0 < α := div_pos (by linarith [hx_lo]) hδ_pos
```

### Error 4 — Line 1804: `linarith` fails
```
error: line 1804:10: linarith failed to find a contradiction
```
Goal: contradiction from `z > 2*n` and `α < 2*n` and `α = z`. Provide intermediate step:
```lean
have : (z : ℝ) = α := hz.symm
linarith [hα_lt]
```

### Error 5 — Line 1806: `exact_mod_cast` type mismatch
```
error: line 1806:31: mod_cast has type...
```
The proof `hx ⟨z.toNat, by omega⟩ (by rw [hx_eq]; congr 1; exact_mod_cast Int.toNat_of_nonneg hz_nn)` has the same structural problem as Error 1 — passing a proof to `≠`.

**Fix**: Restructure to avoid passing a proof argument to `hx`:
```lean
exact hx ⟨z.toNat, by omega⟩ (by
  rw [hx_eq]; congr 1; push_cast [Int.toNat_of_nonneg hz_nn])
```
But this still has the same bug. The correct pattern is:
```lean
have h := hx ⟨z.toNat, by omega⟩
apply h; rw [hx_eq]; congr 1; push_cast [Int.toNat_of_nonneg hz_nn]
```

### Error 6 — Line 1818: Application type mismatch
```
error: line 1818:61: Application type mismatch
```
`Int.lt_add_one_iff.mp (Int.floor_lt.mpr hy_floor.2)` may fail due to API changes.

**Fix**: Use `omega` or prove directly:
```lean
have h_floor_eq : ⌊(y + 1/4) / δ⌋ = z := by
  apply le_antisymm
  · exact Int.le_of_lt_add_one (Int.floor_lt.mpr hy_floor.2)
  · exact Int.le_floor.mpr (le_of_lt hy_floor.1)
```

### Error 7 — Line 1787: unsolved goals (cascade)
```
error: line 1787:4: unsolved goals
```
This is a cascade from the above errors. Fixing errors 1–6 should resolve this.

Please provide the complete corrected Case 3 proof body (from `· -- Case 3: -1/4 < x < 1/4` through `simp only [step_function, ...]`). Keep Cases 1–2 and the lemma statement unchanged.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P3: eLpNorm_conv_ge_discrete (lines 1824-1913)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P3_eLpNorm_conv",
        "prompt": r"""Fix errors in lemma `eLpNorm_conv_ge_discrete` in complete_proof.lean (line 1824). This lemma shows the L∞ norm of the self-convolution of a step function is at least the discrete autoconvolution value.

## Error 1 — Line 1861: Type mismatch in `ae_of_all`
```
error: line 1861:6: Type mismatch
```
Current code (argument to `MeasureTheory.integral_mono`):
```lean
· exact MeasureTheory.ae_of_all MeasureTheory.volume (fun t => by
    calc S t * S (y - t) ≤ S t * 1 :=
          mul_le_mul_of_nonneg_left (h_S_le_one _) (h_S_nn t)
      _ = S t := by ring)
```
**Root cause**: The `calc` block proves `S t * S (y - t) ≤ S t`, but `integral_mono` expects the ae-bound to have type `∀ᵐ t, ‖S t * S(y-t)‖ ≤ bound` or `∀ᵐ t, S t * S(y-t) ≤ S t`. The issue is that `integral_mono` needs both functions to be integrable AND the ae-pointwise bound. The third argument should be an ae-statement matching the exact types that `integral_mono` expects.

**Fix**: Check the signature of `MeasureTheory.integral_mono`. It may need `∀ᵐ t, f t ≤ g t` where `f t = S t * S(y-t)` and `g t = S t`. Or it may be that the integrability hypotheses are wrong. Try:
```lean
· exact fun t => by
    calc S t * S (y - t) ≤ S t * 1 :=
          mul_le_mul_of_nonneg_left (h_S_le_one _) (h_S_nn t)
      _ = S t := mul_one _
```
(Use `mul_one` instead of `by ring`, and drop the `ae_of_all` wrapper since `integral_mono` may accept pointwise bounds directly.)

## Error 2 — Line 1895: rewrite pattern not found
```
error: line 1895:18: Tactic `rewrite` failed: Did not find an occurrence of the pattern
```
Current code:
```lean
(by rw [Real.norm_eq_abs, abs_of_nonneg (h_S_nn _)]; exact h_S_le_one _) (norm_nonneg _)
```
The `rw [Real.norm_eq_abs]` can't find `‖_‖` because the norm may have already been simplified or uses a different form.

**Fix**: Use `simp [Real.norm_eq_abs, abs_of_nonneg (h_S_nn _)]` or just `norm_num` / `simp [norm_nonneg]`. Or try:
```lean
exact le_trans (norm_le_of_nonneg (h_S_nn _) (h_S_le_one _)) (le_refl _)
```

## Error 3 — Line 1904: Application type mismatch
```
error: line 1904:65: Application type mismatch
```
Current: `exact hx_disc (step_function_continuousAt n m hn c x h_none)`
`h_none` from `push_neg at h_none` has type `∀ k : Fin(2*n+1), -(1/4) + ↑k.val/(4*↑n) ≠ x` but `step_function_continuousAt` expects `∀ k, x ≠ -(1/4) + ↑k.val/(4*↑n)`.

**Fix**: Flip the inequality: `fun k => Ne.symm (h_none k)` or `fun k => (h_none k).symm`.

## Error 4 — Line 1907: unsolved goals
```
error: line 1907:62: unsolved goals
```
Current: `(fun t ht => (Set.mem_image _ _ _).mpr ⟨y₀ - t, ht, by ring⟩)`
The `ring` goal is `y₀ - (y₀ - t) = t` which `ring` should handle. But `Set.mem_image` may have a different form.

**Fix**: Try `⟨y₀ - t, ht, by ring⟩` or `⟨y₀ - t, ht, sub_sub_cancel y₀ t⟩`. If the image is `(fun t => y₀ - t) '' S`, then the preimage point is `y₀ - t` and we need `y₀ - (y₀ - t) = t`. Also try `by simp` or `by abel`.

Fix ONLY the 4 errors above. Do not change other parts of the file.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P4: continuous_test_value_le_ratio — h_fac' (lines 2013-2019)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P4_h_fac_prime",
        "prompt": r"""Fix the `h_fac'` statement in theorem `continuous_test_value_le_ratio` in complete_proof.lean (line 1989). Lines 2013–2019 only.

## Errors (all same root cause):
```
error: line 2014:25: failed to synthesize HMul ℕ ℝ ℝ
error: line 2014:6: failed to synthesize AddCommMonoid Prop
error: line 2013:16: failed to synthesize AddCommMonoid ℝ (but actually Prop)
error: line 2015:31: failed to synthesize HMul ℕ ℝ ℝ
error: line 2015:48: failed to synthesize HMul ℕ ℝ ℝ
```

Current code:
```lean
have h_fac' : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
    ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then (4 * ↑n * μ i) * (4 * ↑n * μ j) else 0 =
    (4 * ↑n) ^ 2 * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then μ i * μ j else 0 := by
  rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k
```

**Root cause**: The `↑n` in `4 * ↑n` is ambiguous — Lean tries to interpret `4 * ↑n` as `(4 : ℕ) * (↑n : ℕ)` (Nat multiplication), but then `(4 * ↑n) * μ i` requires `HMul ℕ ℝ ℝ` which doesn't exist. This causes the `if ... then ... else 0` to have type `Prop` (since the branches can't be unified), cascading to `AddCommMonoid Prop` for the sum.

The working `h_fac` (line 2007) uses `∀ k` (universally quantified) which somehow gives Lean enough context. But `h_fac'` uses `∑ k ∈ Finset.Icc ...` where `k : ℕ`, and this changes how `i.1 + j.1 = k` is interpreted (now comparing Fin.val + Fin.val with ℕ).

**Fix**: Add explicit type annotations to force ℝ:
```lean
have h_fac' : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
    (∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then ((4 : ℝ) * ↑n * μ i) * ((4 : ℝ) * ↑n * μ j) else (0 : ℝ)) =
    ((4 : ℝ) * ↑n) ^ 2 * ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
      ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
        if i.1 + j.1 = k then μ i * μ j else (0 : ℝ) := by
  rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k
```

Or more simply, annotate just the `0`:
```lean
if i.1 + j.1 = k then (4 * ↑n * μ i) * (4 * ↑n * μ j) else (0 : ℝ)
```
This forces the `if` to have type ℝ, which should resolve all coercion issues.

Fix ONLY h_fac' (lines 2013–2019). The proof body `rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k` should remain the same if the types match h_fac.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P5: range_sum_delta_le — linarith (line 2588)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P5_range_sum_delta",
        "prompt": r"""Fix the proof of `range_sum_delta_le` in complete_proof.lean (line 2558). Only the final step at line 2588.

## Error:
```
error: line 2588:52: linarith failed to find a contradiction
```

Current code:
```lean
  rw [h_sum_eq]
  have h_upper := cumulative_delta_upper n m hn hm f hf_nonneg hf_supp hf_int b hb
  have h_lower := cumulative_delta_lower n m hn hm f hf_nonneg hf_supp hf_int a (le_trans hab hb)
  exact sub_le_iff_le_add.mpr (le_trans h_upper (by linarith))
```

## Context:
- `h_upper : ∑ i with i.val < b, δ_i ≤ 0` (cumulative sum up to b is ≤ 0)
- `h_lower : -1/↑m ≤ ∑ i with i.val < a, δ_i` (cumulative sum up to a is ≥ -1/m)
- After `rw [h_sum_eq]`, the goal is: `(∑[<b] δ) - (∑[<a] δ) ≤ 1/↑m`

**Root cause**: `sub_le_iff_le_add.mpr` converts the goal to `∑[<b] δ ≤ 1/↑m + ∑[<a] δ`. Then `le_trans h_upper` makes the goal `0 ≤ 1/↑m + ∑[<a] δ`. The `linarith` should close this using `h_lower: -1/↑m ≤ ∑[<a] δ` (which gives `0 ≤ 1/↑m + ∑[<a] δ`). But linarith may not see h_lower because it's defined BEFORE the `exact` chain.

**Fix**: Simply use `linarith` directly with both hypotheses in scope:
```lean
  rw [h_sum_eq]
  have h_upper := cumulative_delta_upper n m hn hm f hf_nonneg hf_supp hf_int b hb
  have h_lower := cumulative_delta_lower n m hn hm f hf_nonneg hf_supp hf_int a (le_trans hab hb)
  linarith
```
If that doesn't work, try `linarith [h_upper, h_lower]` or add explicit steps:
```lean
  have : ∑[<b] δ ≤ 0 := h_upper
  have : -1/↑m ≤ ∑[<a] δ := h_lower
  linarith
```

Fix ONLY line 2588 (the last line of the proof). Everything else is correct.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P6: discretization_autoconv_error — h_diff_eq + hQ_eq
    #     (lines 2738-2762)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P6_disc_h_diff_hQ",
        "prompt": r"""Fix the proofs of `h_diff_eq` and `hQ_eq` inside theorem `discretization_autoconv_error` in complete_proof.lean (line 2640). Lines 2738–2762 only.

## Error 1 — h_diff_eq (lines 2740-2748):
```
error: line 2741:10: Tactic `rewrite` failed: Did not find an occurrence of the pattern
error: line 2742:45: unsolved goals
```
Current code:
```lean
have h_diff_eq : test_value n m (canonical_discretization f n m) ℓ s_lo -
    test_value_continuous n f ℓ s_lo = (4 * ↑n / ↑ℓ) * Q := by
  rw [hQ_def]; unfold test_value test_value_continuous discrete_autoconvolution; simp only []
  rw [show (1 / (4 * ↑n * ↑ℓ)) * _ - (1 / (4 * ↑n * ↑ℓ)) * _ =
      (1 / (4 * ↑n * ↑ℓ)) * (_ - _) from by ring]
  rw [← Finset.sum_sub_distrib]; congr 1
  · rw [show 4 * (↑n : ℝ) / ↑ℓ = 1 / (4 * ↑n * ↑ℓ) * (4 * ↑n) ^ 2 from by field_simp; ring]
  · apply Finset.sum_congr rfl; ...
```

**Root cause**: After `unfold test_value test_value_continuous discrete_autoconvolution; simp only []`, the goal is a complex expression. The `rw [show ...]` pattern `(1/(4*↑n*↑ℓ)) * A - (1/(4*↑n*↑ℓ)) * B` doesn't match because the actual expression has different variable ordering or additional casts.

**Fix approach**: Use `ring_nf` or `field_simp` to normalize, then work with the normalized form:
```lean
have h_diff_eq : test_value n m (canonical_discretization f n m) ℓ s_lo -
    test_value_continuous n f ℓ s_lo = (4 * ↑n / ↑ℓ) * Q := by
  rw [hQ_def]; unfold test_value test_value_continuous discrete_autoconvolution
  simp only []
  -- Factor out 1/(4nℓ) using mul_sub
  have hn_pos : (0 : ℝ) < ↑n := Nat.cast_pos.mpr hn
  have hℓ_pos : (0 : ℝ) < ↑ℓ := by exact_mod_cast Nat.lt_of_lt_of_le (by norm_num) hℓ
  ring_nf
  congr 1; ext k; congr 1; ext i; congr 1; ext j
  split_ifs with heq
  · rw [h_aw i, h_aw j]; ring
  · ring
```

Or try a cleaner approach using `calc` and `Finset.sum_sub_distrib`:
```lean
  simp only [test_value, test_value_continuous, discrete_autoconvolution]
  simp_rw [← Finset.mul_sum]
  ring_nf
  ...
```

Note: `Finset.sum_sub_distrib` may not exist — try `Finset.sum_sub_distrib` → `← Finset.sum_sub_distrib` or the correct name in current Mathlib (maybe `Finset.sum_sub`).

## Error 2 — hQ_eq (lines 2757-2762):
```
error: line 2759:49: Tactic `rewrite` failed: Did not find an occurrence of the pattern
```
Current code:
```lean
have hQ_eq : Q = Part_A + Part_B := by
  simp only [hQ_def, Part_A, Part_B, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro k _; rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _; rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _
  split_ifs with heq; · rw [h_two_term]; · ring
```

**Root cause**: `simp only [... ← Finset.sum_add_distrib]` at the outer level fails because the types are ambiguous (same AddCommMonoid Prop issue as P4).

**Fix**: Use `simp_rw` which works inside binders, and add explicit type annotations:
```lean
have hQ_eq : Q = Part_A + Part_B := by
  simp only [hQ_def, Part_A, Part_B]
  simp_rw [← Finset.sum_add_distrib]
  congr 1; ext k; congr 1; ext i; congr 1; ext j
  split_ifs with heq
  · exact h_two_term i j
  · ring
```

Fix ONLY h_diff_eq and hQ_eq. Do not change variable definitions above or proofs below.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P7: discretization_autoconv_error — Part A
    #     (hPartA_exch, hg_eq, hg_le, lines 2763-2810)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P7_disc_partA",
        "prompt": r"""Fix the Part A sub-proofs inside `discretization_autoconv_error` in complete_proof.lean (line 2640). Lines 2763–2810.

## Error cluster 1 — hPartA_exch (lines 2764-2775):
```
error: line 2768:27: failed to synthesize AddCommMonoid ...
error: line 2768:8: failed to synthesize ...
error: line 2767:27: failed to synthesize ...
```
Current code uses `show _ = _; rw [show ... from by ...]` which fails because the `show` pattern with nested sums triggers type inference issues.

**Fix**: Rewrite the exchange proof step by step:
```lean
have hPartA_exch : Part_A = ∑ j : Fin (2 * n), w j *
    (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ i : Fin (2 * n),
      if i.1 + j.1 = k then δ_i i else (0 : ℝ)) := by
  simp only [Part_A]
  -- Step 1: Factor w j out of inner sum
  conv_lhs =>
    arg 1; ext k; arg 1; ext i
    rw [Finset.sum_comm]  -- swap j and (inner)
  -- Alternative cleaner approach:
  simp_rw [Finset.sum_comm (f := fun j i => if i.1 + j.1 = _ then δ_i i * w j else 0)]
  ...
```

Actually, the cleanest fix is likely:
```lean
have hPartA_exch : Part_A = ∑ j : Fin (2 * n), w j *
    (∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2), ∑ i : Fin (2 * n),
      if i.1 + j.1 = k then δ_i i else (0 : ℝ)) := by
  simp only [Part_A]
  rw [Finset.sum_comm]
  congr 1; ext j
  rw [← Finset.mul_sum]
  congr 1; ext k
  rw [Finset.sum_comm]
  congr 1; ext i
  split_ifs <;> ring
```

## Error cluster 2 — hg_eq (lines 2779-2787):
```
error: line 2782:31: Tactic `rewrite` failed: Did not find an occurrence of the pattern
```
Current code:
```lean
intro j; show _ = _; rw [Finset.sum_comm]
```

**Fix**: Use `simp only` to unfold g_fn, then rewrite:
```lean
have hg_eq : ∀ j : Fin (2 * n), g_fn j = ∑ i ∈ Finset.filter
    (fun i : Fin (2 * n) => s_lo ≤ i.1 + j.1 ∧ i.1 + j.1 + 2 ≤ s_lo + ℓ)
    Finset.univ, δ_i i := by
  intro j; simp only [g_fn]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl; intro i _
  rw [Finset.sum_ite_eq']; simp only [Finset.mem_Icc]
  constructor
  · intro ⟨h1, h2⟩; simp [Finset.mem_filter]; exact ⟨h1, by omega⟩
  · intro h; simp [Finset.mem_filter] at h; exact ⟨h.1, by omega⟩
```

## Error cluster 3 — hg_le filter equivalence (lines 2796-2807):
```
error: line 2798:60: Tactic `split_ifs` failed: no if-then-else conditions
error: line 2806:45: Tactic `constructor` failed: target is not an inductive datatype
error: line 2807:97: omega could not prove the goal
```

**Root cause**: After `ext j; simp [Finset.mem_filter]`, the if-then-else conditions are already simplified away, so `split_ifs` finds nothing. And the goal after `simp` may not be a conjunction for `constructor`.

**Fix**: Replace the filter equivalence with `omega`:
```lean
ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and]; omega
```
And for the `by omega` in line 2807, provide explicit intermediate bounds or use `Nat.le_of_lt_succ` etc.

Fix ONLY the Part A sub-proofs (hPartA_exch, hg_eq, hg_le, and the filter proof in hg_le). Part B has the same structure — do NOT fix Part B here.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P8: discretization_autoconv_error — Part B + CB + final
    #     (lines 2833-2928)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P8_disc_partB_CB_final",
        "prompt": r"""Fix the Part B, CB (contributing bins), and final bound sub-proofs inside `discretization_autoconv_error` in complete_proof.lean (line 2640). Lines 2833–2928.

**IMPORTANT**: Part B (hPartB_exch, hh_eq, hh_le) has the EXACT same structure as Part A (hPartA_exch, hg_eq, hg_le). Apply the same fix pattern. The errors are identical:

## Part B errors (same pattern as Part A):
```
error: line 2839:27: failed to synthesize (hPartB_exch, same as hPartA_exch)
error: line 2839:8: failed to synthesize
error: line 2838:27: failed to synthesize
error: line 2852:31: Tactic `rewrite` failed (hh_eq, same as hg_eq)
error: line 2868:60: split_ifs failed (same as hg_le filter)
error: line 2876:45: constructor failed (same as hg_le filter)
error: line 2877:97: omega failed (same as hg_le bound)
```

Use the same fix strategy as Part A: use `simp_rw`, `Finset.sum_comm`, explicit `congr`/`ext`, and replace `split_ifs; omega` with just `omega` in the filter proofs.

## CB contiguity argument (lines 2893-2906):
```
error: line 2894:73: linarith failed
error: line 2900:34: omega failed
error: line 2903:117: omega failed
error: line 2906:37: linarith failed
```

### Line 2894: `linarith` for `∑ δ_i ≥ -1/m`
Current: `suffices hd : ∑ i ∈ contributing_bins n ℓ s_lo, δ_i i ≥ -1 / ↑m by linarith`
This should work if hd is proved. The issue may be in the sufficiency direction. Try adding explicit hypotheses.

### Line 2900: `omega` for contributing_bins_iff filter
```lean
ext i; rw [contributing_bins_iff n hn ℓ s_lo hℓ i]
simp [Finset.mem_filter]; omega
```
`contributing_bins_iff` rewrites membership as `Nat.max 0 (s_lo - (2*n-1)) ≤ i.1 ∧ i.1 ≤ Nat.min (2*n-1) (s_lo+ℓ-2)`. The filter has `cb_lo ≤ i.1 ∧ i.1 < cb_hi + 1` where `cb_lo = Nat.max 0 (s_lo-(2*n-1))` and `cb_hi = Nat.min (2*n-1) (s_lo+ℓ-2)`.

The `omega` may fail because `Nat.max` and `Nat.min` don't simplify with `simp [Finset.mem_filter]`.

**Fix**: Use `simp only [Finset.mem_filter, Finset.mem_univ, true_and]` then `constructor <;> intro ⟨h1, h2⟩ <;> exact ⟨by omega, by omega⟩`, or unfold Nat.max/Nat.min explicitly.

### Line 2903: `omega` for `cb_hi + 1 ≤ 2 * n`
```lean
exact range_sum_delta_ge ... cb_lo (cb_hi + 1) (by omega) (by simp [cb_hi]; omega)
```
The `by simp [cb_hi]; omega` needs `cb_hi + 1 ≤ 2*n`. Since `cb_hi = Nat.min (2*n-1) (s_lo+ℓ-2)`, we have `cb_hi ≤ 2*n-1`, so `cb_hi + 1 ≤ 2*n`. But `omega` may not simplify `Nat.min`.

**Fix**: Add `have : cb_hi ≤ 2 * n - 1 := Nat.min_le_left _ _` before the `omega`.

### Line 2906: `linarith` for empty filter case
```lean
rw [this, Finset.sum_empty]; linarith [show (0:ℝ) < 1/↑m from by positivity]
```
Goal after `Finset.sum_empty`: `0 ≥ -1 / ↑m`. This is `0 ≥ -1/m` which is true for m > 0. Try `linarith [show (0:ℝ) < 1/↑m from by positivity]` or `positivity` (0 ≥ -positive = -(negative) is nonneg).

Actually, `-1/↑m` for `m > 0` gives `-1/m < 0`, so `0 ≥ -1/m` is `1/m ≥ 0`, which follows from positivity.

**Fix**: `rw [this, Finset.sum_empty]; positivity` or `linarith [Nat.cast_pos.mpr hm]`.

## Final linarith (line 2928):
```
error: line 2928:16: linarith failed
```
Goal: `Q ≤ 1/m² + 2*W/m` from `hQ_eq : Q = Part_A + Part_B`, `hPartA_le : Part_A ≤ W/m`, `hPartB_le : Part_B ≤ W/m + 1/m²`.

This should be: `Q = Part_A + Part_B ≤ W/m + W/m + 1/m² = 2W/m + 1/m²`. The `linarith` should close this.

**Fix**: Make sure hQ_eq, hPartA_le, hPartB_le are all in scope:
```lean
rw [hQ_eq]; linarith [hPartA_le, hPartB_le]
```

Fix Part B, the CB argument, and the final bound. Use the same patterns as Part A for Part B.""",
    },
    # ════════════════════════════════════════════════════════════════
    # P9: autoconvolution_ratio_scaling — eLpNorm rewrite (line 3089)
    # ════════════════════════════════════════════════════════════════
    {
        "name": "P9_scaling_eLpNorm",
        "prompt": r"""Fix the eLpNorm rewrite in the `autoconvolution_ratio_scaling` lemma in complete_proof.lean, around line 3089.

## Error:
```
error: line 3089:14: Tactic `rewrite` failed: Did not find an occurrence of the pattern
```

Current code:
```lean
have : (fun x => a ^ 2 * MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ)
    MeasureTheory.volume x) = (fun x => a ^ 2 • MeasureTheory.convolution f f
    (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x) := by ext x; rfl
rw [this, MeasureTheory.eLpNorm_const_smul, ENNReal.toReal_mul,
    Real.enorm_eq_ofReal (le_of_lt ha2), ENNReal.toReal_ofReal (le_of_lt ha2)]
```

**Root cause**: After `rw [this]`, the goal has `eLpNorm (fun x => a^2 • conv_ff x) ⊤ volume`. But `MeasureTheory.eLpNorm_const_smul` expects `eLpNorm (c • f) p μ` where `c • f` is the scalar multiplication on functions (i.e., `c • f = fun x => c • f x`), NOT `fun x => c • f x`. The `rw` can't match because `fun x => c • f x` is not definitionally `c • f` for rewriting purposes.

Also, `Real.enorm_eq_ofReal` may not exist in current Mathlib. The correct name might be different.

**Fix approach 1**: Convert to function-level smul:
```lean
have h_eq : (fun x => a ^ 2 • MeasureTheory.convolution f f
    (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x) =
  a ^ 2 • (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) := by
  ext x; simp [Pi.smul_apply]
rw [this, h_eq]
```
Then:
```lean
rw [MeasureTheory.eLpNorm_const_smul]
```
should work since the argument is now `c • f` at the function level.

**Fix approach 2**: Use `simp_rw` instead of `rw`:
```lean
simp_rw [this]
rw [show (fun x => a ^ 2 • conv_ff x) = (a ^ 2 • conv_ff) from rfl]
rw [MeasureTheory.eLpNorm_const_smul]
```

**Fix approach 3**: Avoid the smul route entirely and use `eLpNorm_const_mul`:
```lean
-- If eLpNorm_const_mul exists in Mathlib
rw [MeasureTheory.eLpNorm_const_mul]
```

For the `ENNReal.toReal_mul` and `Real.enorm_eq_ofReal` steps, try:
- `Real.enorm_eq_ofReal` → `enorm_ofReal` or `ENNReal.ofReal_eq_coe_nnabs` or similar
- Or just use `simp [ENNReal.toReal_mul, abs_of_pos ha2]` to simplify

Fix ONLY the h_norm proof (lines 3078–3090). The surrounding code is correct.""",
    },
]


def create_tar():
    """Create tar.gz of the lean project (without .lake)."""
    print(f"Creating tar.gz at {TAR_PATH}...")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        for fname in ["complete_proof.lean", "lakefile.lean", "lean-toolchain", "lake-manifest.json"]:
            fpath = os.path.join(LEAN_DIR, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)
                print(f"  + {fname}")
            else:
                print(f"  ! {fname} not found, skipping")
    size_mb = os.path.getsize(TAR_PATH) / (1024 * 1024)
    print(f"  tar.gz size: {size_mb:.2f} MB\n")


def submit_project(prompt_data):
    """Submit a prompt + tar.gz to Aristotle API."""
    body = json.dumps({"prompt": prompt_data["prompt"]})
    with open(TAR_PATH, "rb") as f:
        tar_content = f.read()
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{API_URL}/project",
            headers={"X-API-Key": API_KEY},
            data={"body": body},
            files=[("input", ("lean_project.tar.gz", tar_content, "application/x-tar"))],
        )
        response.raise_for_status()
        return response.json()


def check_status(project_id):
    """Check status of a submitted project."""
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{API_URL}/project/{project_id}",
            headers={"X-API-Key": API_KEY},
        )
        response.raise_for_status()
        return response.json()


def download_result(project_id, name):
    """Download result tar.gz for a completed project."""
    with httpx.Client(timeout=120) as client:
        resp = client.get(
            f"{API_URL}/project/{project_id}/result",
            headers={"X-API-Key": API_KEY},
        )
        if resp.status_code == 200:
            out_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"aristotle_v3_{name}.tar.gz")
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
    return None


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: ARISTOTLE_KEY not found in environment or .env file")
        sys.exit(1)

    # --- STATUS ---
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        try:
            with open(PROJECTS_FILE, "r") as f:
                projects = json.load(f)
        except FileNotFoundError:
            print("No projects file found. Run without arguments to submit.")
            sys.exit(1)
        for p in projects:
            if p.get("project_id"):
                try:
                    status = check_status(p["project_id"])
                    pct = status.get("percent_complete", "?")
                    print(f"  {p['name']:30s} {status['status']:25s} ({pct}%)")
                except Exception as e:
                    print(f"  {p['name']:30s} ERROR: {e}")
        sys.exit(0)

    # --- RESULTS ---
    if len(sys.argv) > 1 and sys.argv[1] == "results":
        try:
            with open(PROJECTS_FILE, "r") as f:
                projects = json.load(f)
        except FileNotFoundError:
            print("No projects file found.")
            sys.exit(1)
        for p in projects:
            if not p.get("project_id"):
                continue
            try:
                status = check_status(p["project_id"])
            except Exception as e:
                print(f"\n=== {p['name']} ERROR: {e} ===")
                continue
            s = status["status"]
            print(f"\n=== {p['name']} ({s}) ===")
            if s in ("COMPLETE", "COMPLETE_WITH_ERRORS", "OUT_OF_BUDGET"):
                out_path = download_result(p["project_id"], p["name"])
                if out_path:
                    out_dir = os.path.join(os.environ.get("TEMP", "/tmp"), f"aristotle_v3_out_{p['name']}")
                    os.makedirs(out_dir, exist_ok=True)
                    with tarfile.open(out_path, "r:gz") as t:
                        t.extractall(out_dir)
                    # Find the lean file in output
                    for root, dirs, files in os.walk(out_dir):
                        for fn in files:
                            if fn == "complete_proof.lean":
                                lean_out = os.path.join(root, fn)
                                lean_cur = os.path.join(LEAN_DIR, "complete_proof.lean")
                                import difflib
                                with open(lean_out, encoding="utf-8") as a, open(lean_cur, encoding="utf-8") as b:
                                    diff = list(difflib.unified_diff(
                                        b.readlines(), a.readlines(),
                                        fromfile="current", tofile="aristotle", n=3))
                                if diff:
                                    print(f"  {len(diff)} diff lines. First 50:")
                                    for line in diff[:50]:
                                        print(f"    {line.rstrip()}")
                                else:
                                    print("  No changes (identical)")
                                break
                else:
                    print("  Failed to download result")
        sys.exit(0)

    # --- SUBMIT ---
    create_tar()

    results = []
    for i, prompt_data in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] Submitting: {prompt_data['name']}...")
        try:
            result = submit_project(prompt_data)
            pid = result.get("project_id", "???")
            print(f"  -> project_id={pid} status={result.get('status', '?')}")
            results.append({"name": prompt_data["name"], "project_id": pid})
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"name": prompt_data["name"], "project_id": None, "error": str(e)})
        time.sleep(2)  # Rate limit

    with open(PROJECTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} projects to: {PROJECTS_FILE}")
    print(f"Check status: python lean/submit_to_aristotle_v3.py status")
    print(f"Get results:  python lean/submit_to_aristotle_v3.py results")
