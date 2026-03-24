#!/bin/bash
# Submit focused prompts to Aristotle API
ARISTOTLE_KEY=$(grep ARISTOTLE_KEY "C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\.env" | cut -d'=' -f2 | tr -d ' ')
TAR_FILE="/tmp/lean_project.tar.gz"
API_URL="https://aristotle.harmonic.fun/api/v2/project"

submit_prompt() {
    local name="$1"
    local prompt="$2"
    echo "=== Submitting: $name ==="
    local escaped_prompt=$(echo "$prompt" | python -c "import sys,json; print(json.dumps(sys.stdin.read()))")
    local body="{\"prompt\": $escaped_prompt}"
    local result=$(curl -s -X POST "$API_URL" \
        -H "X-API-Key: $ARISTOTLE_KEY" \
        -F "body=$body" \
        -F "input=@$TAR_FILE;type=application/x-tar;filename=lean_project.tar.gz")
    local pid=$(echo "$result" | python -c "import sys,json; print(json.loads(sys.stdin.read())['project_id'])" 2>/dev/null)
    echo "Project ID: $pid"
    echo "$name|$pid" >> /tmp/aristotle_projects.txt
    echo "$result" | python -m json.tool 2>/dev/null || echo "$result"
    echo ""
}

> /tmp/aristotle_projects.txt

# PROMPT 1: step_function_continuousAt
submit_prompt "step_function_continuousAt" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of the private lemma \`step_function_continuousAt\` starting at line 1765.

The lemma states: a step function (piecewise constant on bins [-1/4 + k/(4n), -1/4 + (k+1)/(4n))) is ContinuousAt at every point x that is not a bin boundary.

Current errors in the proof:
1. Lines 1774, 1783: \`isOpen_Iio.mem_nhds hx_lt\` and \`isOpen_Ioi.mem_nhds hx_gt\` fail with 'Invalid field notation'. The Lean 4 / Mathlib API may have changed. Use \`IsOpen.mem_nhds isOpen_Iio hx_lt\` or the appropriate current API.
2. Lines 1782, 1790: \`hx ⟨2*n, by omega⟩ (by field_simp; ring)\` fails because the argument to hx has wrong type. The hypothesis hx says x ≠ -(1/4) + k.val/(4*n) for k : Fin(2*n+1). The proof needs to instantiate k correctly and show the equality.
3. Line 1797: \`field_simp\` made no progress. The goal involves \`(x + 1/4) / (1/(4*n)) = z\`. Try \`have h4n_ne : (4 * (n:ℝ)) ≠ 0 := by positivity\` then \`field_simp [h4n_ne] at hz\`.
4. Line 1800: \`positivity\` fails to show \`0 < α\` where α = (x+1/4)/δ. Need to establish δ > 0 and x > -1/4 first.
5. Line 1804: \`linarith\` fails to show z ≤ 2*n from α < 2*n and α = z. Provide intermediate steps.
6. Line 1806: \`exact_mod_cast\` type mismatch. The cast from Int.toNat to Fin index needs explicit handling.
7. Line 1818: \`Int.lt_add_one_iff\` application fails. Use \`Int.floor_lt\` and \`Int.le_floor\` directly.

Please provide a complete corrected proof body for this lemma. Keep the same theorem statement."

# PROMPT 2: eLpNorm_conv_ge_discrete (integral_mono + continuity)
submit_prompt "eLpNorm_conv_ge_discrete" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of lemma \`eLpNorm_conv_ge_discrete\` starting at line 1824.

This lemma states: the L∞ norm of the autoconvolution of a step function is at least (1/(4n))/m² times the discrete autoconvolution at any index k.

Current errors:
1. Line 1861: \`MeasureTheory.ae_of_all _ (fun t => ...)\` causes 'typeclass instance problem is stuck: MeasureTheory.OuterMeasureClass'. The issue is that \`MeasureTheory.ae_of_all\` needs the measure argument explicitly. Try \`MeasureTheory.ae_of_all MeasureTheory.volume\` instead of \`MeasureTheory.ae_of_all _\`.
2. Lines 1895: \`rw [Real.norm_eq_abs, abs_of_nonneg ...]\` fails with 'Did not find occurrence of ‖?r‖'. The norm notation may have changed. Try using \`norm_num\` or \`simp [Real.norm_eq_abs]\` instead.
3. Lines 1904: \`step_function_continuousAt n m hn c x h_none\` has argument type mismatch. The \`h_none\` has type \`∀ x₁ : Fin(2*n+1), -(1/4) + x₁.val/(4*n) ≠ x\` but the lemma expects \`x ≠ -(1/4) + k.val/(4*n)\`. Need to flip the inequality with \`fun k => Ne.symm (h_none k)\` or similar.
4. Lines 1907: unsolved goals after the above fix.

Please provide a complete corrected proof body for this lemma."

# PROMPT 3: sum_f_bin_le (rename_i issue)
submit_prompt "sum_f_bin_le" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of lemma \`sum_f_bin_le\` starting at line 1951.

This lemma states: for a nonneg function f, the sum of indicator functions over all bins is at most f(t) at each point.

The error is at line 1974:
\`\`\`
rename_i h_not_mem h_eq_i₀; subst h_eq_i₀; exact absurd hi₀ h_not_mem
\`\`\`
Error: 'too many variable names provided' and 'Unknown identifier h_not_mem'.

This is in the third case of a split_ifs (t ∉ bin i, i = i₀). The \`rename_i\` is providing 2 names but the goal only has 1 unnamed hypothesis. The fix should be:
- Use \`rename_i h_eq_i₀\` (just one name for the equality i = i₀)
- Then \`subst h_eq_i₀\`
- Then the negated membership should already be named \`h\` from split_ifs, so use \`exact absurd hi₀ h\` or \`simp_all\`.

Please provide a corrected proof for just the affected case branch, or the complete proof body if needed."

# PROMPT 4: continuous_test_value_le_ratio (AddCommMonoid Prop)
submit_prompt "continuous_test_value_le_ratio" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of theorem \`continuous_test_value_le_ratio\` starting at line 1989.

The errors are at lines 2013-2015:
\`\`\`
have h_fac' : ∑ k ∈ Finset.Icc s_lo (s_lo + ℓ - 2),
    ∑ i : Fin (2 * n), ∑ j : Fin (2 * n),
      if i.1 + j.1 = k then (4 * ↑n * μ i) * (4 * ↑n * μ j) else 0 = ...
\`\`\`
Error: 'failed to synthesize AddCommMonoid Prop' and 'failed to synthesize HMul ℕ ℝ'.

The root cause is a type coercion issue: \`↑n\` is being interpreted as a natural number cast, but the expression \`4 * ↑n * μ i\` needs n to be cast to ℝ. The multiplication \`(4 : ℕ) * (↑n : ℕ)\` produces a ℕ, but then multiplying by \`μ i : ℝ\` fails.

Fix: ensure the cast is explicit: use \`(4 * (↑n : ℝ))\` or \`(4 * n : ℝ)\` consistently. Or add type annotations: \`((4 : ℝ) * ↑n * μ i)\`.

The \`h_fac'\` proof body is:
\`\`\`
rw [Finset.mul_sum]; congr 1; ext k; exact h_fac k
\`\`\`
which should work if the types are fixed. Please provide the corrected statement and proof for h_fac'."

# PROMPT 5: range_sum_delta_le (linarith)
submit_prompt "range_sum_delta_le" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of lemma \`range_sum_delta_le\` (around line 2555).

The error is at line 2588:
\`\`\`
exact sub_le_iff_le_add.mpr (le_trans h_upper (by linarith))
\`\`\`
Error: 'linarith failed to find a contradiction'.

Context: The goal after \`sub_le_iff_le_add.mpr\` is approximately:
  cumulative_upper_bound ≤ 1/m + cumulative_lower_bound_complement
where h_upper and h_lower provide bounds on cumulative delta sums.

The \`linarith\` likely needs additional hypotheses made explicit. Try:
1. Adding \`have := h_lower\` and \`have := h_upper\` before the linarith
2. Or rewriting the chain more explicitly using \`calc\`
3. The issue may be that the lower bound gives ≥ -1/m while upper gives ≤ 1/m, and the difference of two such bounds needs 1/m + 1/m = 2/m, but the target is 1/m.

Please examine the exact goal state and provide a corrected proof for this specific step."

# PROMPT 6: discretization_autoconv_error (h_per_term linarith)
submit_prompt "discretization_autoconv_error_part1" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of theorem \`discretization_autoconv_error\` starting at line 2640. Focus specifically on these errors:

1. Line 2727: \`linarith\` fails in \`h_per_term\` proof. The goal is to show:
   \`w i * w j - μ i * μ j ≤ (w j + μ i) / ↑m\`
   from hypotheses h1: δ_i i * w j ≤ |δ_i i| * w j, h2: μ i * δ_i j ≤ μ i * |δ_i j|, h3: |δ_i i| * w j ≤ (1/m) * w j, h4: μ i * |δ_i j| ≤ μ i * (1/m), and h_two_term: w i * w j - μ i * μ j = δ_i i * w j + μ i * δ_i j.
   The chain is: LHS = δ_i·w_j + μ_i·δ_j ≤ |δ_i|·w_j + μ_i·|δ_j| ≤ w_j/m + μ_i/m = (w_j + μ_i)/m.
   Try: \`calc w i * w j - μ i * μ j = δ_i i * w j + μ i * δ_i j := h_two_term i j; _ ≤ ... := by linarith\`

2. Lines 2740-2741: \`rw [show 1/(4*↑n*↑ℓ) * _ - 1/(4*↑n*↑ℓ) * _ = ...]\` rewrite fails. The pattern doesn't match because the actual expression uses different variable representations. Try using \`ring_nf\` or \`field_simp\` followed by \`ring\` instead.

3. Line 2742: \`Finset.sum_sub_distrib\` doesn't exist in current Mathlib. Use \`Finset.sum_sub_distrib\` → try \`← Finset.sum_sub_distrib\` or look for the correct lemma name (maybe \`Finset.sum_sub_distrib\` was renamed to something else).

Please provide corrected proof steps for these specific issues within discretization_autoconv_error."

# PROMPT 7: discretization_autoconv_error (Parts A & B)
submit_prompt "discretization_autoconv_error_part2" "In the uploaded Lean 4 project (complete_proof.lean), fix the proof of theorem \`discretization_autoconv_error\` starting at line 2640. Focus on the Part A and Part B sub-proofs (lines 2756-2925):

1. Lines 2758, 2766-2768: \`Finset.sum_add_distrib\` rewrite fails with 'Did not find occurrence'. Also 'failed to synthesize AddCommMonoid Prop'. Same coercion issue as prompt 4 - the if-then-else expression types may be Prop instead of ℝ. Ensure the split produces ℝ values, not Props.

2. Lines 2781, 2850: \`rw [Finset.sum_comm]\` fails for g_fn/h_fn definition. The pattern \`∑ x ∈ S, ∑ y ∈ T, f x y\` is not found. This may be because the inner sum is over Finset.univ (implicit) vs explicit range. Try \`simp only [Finset.sum_comm]\` or restructure.

3. Lines 2797, 2866: \`split_ifs\` fails with 'no if-then-else conditions to split'. The goal may have already been simplified so there are no ifs left. Check whether the filter condition needs omega or decide instead.

4. Lines 2805, 2874: \`constructor\` fails with 'target is not an inductive datatype'. The goal is likely an And (∧) but may have been transformed. Use \`exact ⟨..., ...⟩\` or \`refine ⟨?_, ?_⟩\` instead.

5. Lines 2822, 2913: \`hj.2\` / \`hi.2\` fail with 'Projections cannot be used on functions'. The hypothesis hj has type \`j ∈ CB → False\` (a function), not a conjunction. Use \`hj (by ...)\` to apply it, or fix the Finset.mem_filter destructuring.

6. Lines 2828, 2919: rewrite with Finset.sum_filter fails. The pattern doesn't match.

7. Lines 2892, 2904, 2925: various linarith/omega failures in the CB contiguity and final bound.

Please provide corrected proof steps for the Part A (hPartA_le) and Part B (hPartB_le) sub-proofs."

echo ""
echo "=== All prompts submitted ==="
echo "Project IDs saved to /tmp/aristotle_projects.txt"
cat /tmp/aristotle_projects.txt
