"""Submit round 2 of focused prompts to Aristotle API."""
import json
import os
import time
import httpx

API_URL = "https://aristotle.harmonic.fun/api/v2"
API_KEY = "arstl_jCrlrl_A2qiRoMb-vWH3CLRTBW_gK_AHhVCD6b2dyj8"
TAR_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "lean_project_v2.tar.gz")

PROMPTS = [
    # ── 1. step_function_continuousAt: full rewrite ──
    {
        "name": "P1_step_func_case1",
        "prompt": (
            "In complete_proof.lean, the private lemma `step_function_continuousAt` (line 1765) "
            "has errors in Case 1 (x < -1/4) and Case 2 (x >= 1/4). "
            "The proof uses `Filter.eventually_of_mem (isOpen_Iio.mem_nhds hx_lt) fun y hy => by ...` "
            "which compiles but then the body has unsolved goals.\n\n"
            "The issue: after Filter.eventually_of_mem, the hypothesis `hy` has type `y ∈ Set.Iio (-(1/4))` "
            "i.e. `y < -(1/4)`, but the simp needs `y < -(1/4) ∨ y ≥ 1/4` which is `Or.inl hy`. "
            "Check that `simp only [step_function, ...]` actually closes the goal — it may need "
            "`if_pos` / `if_neg` or `dite` handling instead of `↓reduceIte`.\n\n"
            "Also in Case 2, `hx ⟨2 * n, by omega⟩ (by field_simp; ring)` fails. "
            "The goal is to show `x ≠ -(1/4) + (2*n)/(4*n)` i.e. `x ≠ 1/4`. "
            "Try: `by push_cast; field_simp; linarith`.\n\n"
            "Fix ONLY Case 1 and Case 2 (lines 1774-1786). Do not change Case 3."
        ),
    },
    {
        "name": "P2_step_func_case3",
        "prompt": (
            "In complete_proof.lean, the private lemma `step_function_continuousAt` (line 1765) "
            "has errors in Case 3 (the interior case, -1/4 < x < 1/4, lines 1787-1822).\n\n"
            "Errors:\n"
            "- Line 1790: `hx ⟨0, by omega⟩ (by simp)` fails. Goal: show x ≠ -1/4. "
            "  The hypothesis hx says `∀ k : Fin(2n+1), x ≠ -(1/4) + k.val/(4n)`. "
            "  For k=0: -(1/4) + 0/(4n) = -1/4. Try `simp [Fin.val]` or `push_cast; ring_nf`.\n"
            "- Line 1797: `field_simp` made no progress on `α = ↑z` where α = (x+1/4)/δ, δ = 1/(4n). "
            "  Try: unfold δ, then `field_simp [show (4*(n:ℝ)) ≠ 0 from by positivity]`.\n"
            "- Line 1800: `positivity` fails on `0 < α`. Establish `0 < x + 1/4` (from hx_lo) and `0 < δ` first.\n"
            "- Line 1804: `linarith` fails. Add `have := hz` (α = z) and `have : α < 2*n` explicitly.\n"
            "- Line 1806: `exact_mod_cast Int.toNat_of_nonneg hz_nn` wrong type. Try `push_cast [Int.toNat_of_nonneg hz_nn]`.\n"
            "- Line 1818: `Int.lt_add_one_iff.mp (Int.floor_lt.mpr hy_floor.2)` fails because "
            "  `hy_floor.2` has a lambda wrapper. Beta-reduce with `show` or extract the bound first.\n\n"
            "Fix ONLY Case 3 (lines 1787-1822). Do not change Cases 1-2 or other lemmas."
        ),
    },
    # ── 2. eLpNorm_conv_ge_discrete ──
    {
        "name": "P3_eLpNorm_h_bound",
        "prompt": (
            "In complete_proof.lean, lemma `eLpNorm_conv_ge_discrete` (line 1824) has an error at line 1861:\n"
            "```\n"
            "· exact MeasureTheory.ae_of_all _ (fun t => by\n"
            "    calc S t * S (y - t) ≤ S t * 1 := ...\n"
            "      _ = S t := by ring)\n"
            "```\n"
            "Error: 'typeclass instance problem is stuck: MeasureTheory.OuterMeasureClass ?m ℝ'\n\n"
            "The `_` in `ae_of_all _` can't be inferred. This is the third argument to "
            "`MeasureTheory.integral_mono`. The issue may be that `integral_mono` expects "
            "an ae-condition but we're providing a pointwise one.\n\n"
            "Try replacing the whole `· exact MeasureTheory.ae_of_all _ (fun t => ...)` with:\n"
            "```\n"
            "· exact MeasureTheory.ae_of_all MeasureTheory.volume (fun t => ...)\n"
            "```\n"
            "Or try `Filter.Eventually.of_forall` or just `fun t => ...` without the ae wrapper.\n\n"
            "Fix ONLY this one error at line 1861. Do not change other parts of the proof."
        ),
    },
    {
        "name": "P4_eLpNorm_continuity",
        "prompt": (
            "In complete_proof.lean, lemma `eLpNorm_conv_ge_discrete` (line 1824), "
            "the sub-proof of `h_conv_cont` (lines 1880-1910) has errors:\n\n"
            "1. Line 1895: `rw [Real.norm_eq_abs, ...]` fails — 'Did not find ‖?r‖'. "
            "   The norm may already be simplified or use a different form.\n"
            "2. Line 1904: `step_function_continuousAt n m hn c x h_none` — type mismatch. "
            "   `h_none` has `∀ k, -(1/4) + k.val/(4n) ≠ x` but lemma expects `x ≠ -(1/4) + k.val/(4n)`. "
            "   Use `fun k => (h_none k).symm` or `fun k => Ne.symm (h_none k)`.\n"
            "3. Line 1907: image membership proof has unsolved goals. "
            "   `⟨y₀ - t, ht, by ring⟩` — the `ring` goal is `y₀ - (y₀ - t) = t`. "
            "   Maybe needs `sub_sub_cancel` or explicit type annotation.\n\n"
            "Fix ONLY h_conv_cont and the filter_upwards block (lines 1880-1913)."
        ),
    },
    # ── 3. continuous_test_value_le_ratio ──
    {
        "name": "P5_test_value_coercion",
        "prompt": (
            "In complete_proof.lean, theorem `continuous_test_value_le_ratio` (line 1989), "
            "lines 2007-2019 define h_fac and h_fac'. The h_fac' statement at lines 2013-2018 "
            "causes 'failed to synthesize AddCommMonoid Prop' and 'HMul ℕ ℝ'.\n\n"
            "h_fac (lines 2007-2012) compiles fine, but h_fac' (lines 2013-2018) does not. "
            "Both have the same structure: `∑ k, ∑ i, ∑ j, if i.1+j.1=k then (4*↑n*μ i)*(4*↑n*μ j) else 0`. "
            "The difference: h_fac uses `∀ k` while h_fac' uses `∑ k ∈ Finset.Icc ...`.\n\n"
            "The Finset.Icc sum somehow breaks type inference for the inner if-then-else. "
            "Lean can't determine that the body is ℝ.\n\n"
            "Possible fixes:\n"
            "- Convert h_fac' to use h_fac via `Finset.sum_congr` and `Finset.mul_sum`\n"
            "- Add a type ascription: `(∑ i, ∑ j, ... : ℝ)` on the inner sums\n"
            "- Use `show` to annotate the expected type\n"
            "- Rewrite h_fac' proof differently\n\n"
            "Fix ONLY h_fac' (lines 2013-2019). The rest of the theorem is fine."
        ),
    },
    # ── 4. range_sum_delta_le ──
    {
        "name": "P6_range_sum_delta",
        "prompt": (
            "In complete_proof.lean, lemma `range_sum_delta_le` (around line 2555), "
            "the final step at line 2588 fails:\n"
            "```\n"
            "exact sub_le_iff_le_add.mpr (le_trans h_upper (by linarith))\n"
            "```\n\n"
            "Context: We're showing that a range sum of δ_i (discretization errors) over bins [a,b) "
            "is ≤ 1/m. We have:\n"
            "- h_upper: cumulative sum over [0,b) ≤ 1/m (from cumulative_delta_upper)\n"
            "- h_lower: cumulative sum over [0,a) ≥ -1/m (from cumulative_delta_lower)\n"
            "- h_sum_eq: range sum = [0,b) sum - [0,a) sum\n\n"
            "The bound should be: range sum = [0,b) - [0,a) ≤ 1/m - (-1/m) = 2/m. "
            "But we need ≤ 1/m, so the approach may need adjustment. "
            "Perhaps use a tighter bound or a different decomposition.\n\n"
            "Fix this one linarith step. You may restructure the final proof chain if needed."
        ),
    },
    # ── 5. discretization_autoconv_error: h_diff_eq ──
    {
        "name": "P7_disc_h_diff_eq",
        "prompt": (
            "In complete_proof.lean, theorem `discretization_autoconv_error` (line 2640), "
            "the sub-proof `h_diff_eq` (lines 2738-2748) shows:\n"
            "  test_value(discrete) - test_value_continuous = (4n/ℓ) * Q\n\n"
            "Current proof uses `rw [show 1/(4nℓ)*_ - 1/(4nℓ)*_ = 1/(4nℓ)*(_-_) from by ring]` "
            "which fails because the pattern doesn't match the goal after unfold.\n\n"
            "Try a different approach:\n"
            "- Use `← mul_sub` to factor 1/(4nℓ) from the difference\n"
            "- Use `simp_rw [← Finset.sum_sub_distrib]` to push subtraction inside sums\n"
            "- Define a per-term lemma showing each (if p then disc else 0) - (if p then cont else 0) "
            "  = (4n)² * (if p then w*w-μ*μ else 0), using `by_cases hp : i.1+j.1=k <;> simp [hp] <;> ring`\n"
            "- Then use `simp_rw` with that lemma, factor with `Finset.mul_sum`, and `field_simp; ring`\n\n"
            "Fix ONLY h_diff_eq. The variable definitions (w, μ, δ_i, etc.) above it are fine."
        ),
    },
    # ── 6. discretization_autoconv_error: hQ_eq ──
    {
        "name": "P8_disc_hQ_eq",
        "prompt": (
            "In complete_proof.lean, theorem `discretization_autoconv_error` (line 2640), "
            "the sub-proof `hQ_eq` (lines 2757-2762) shows Q = Part_A + Part_B where:\n"
            "- Q = ∑_k ∑_i ∑_j (if i+j=k then w_i·w_j - μ_i·μ_j else 0)\n"
            "- Part_A = ∑_k ∑_i ∑_j (if i+j=k then δ_i·w_j else 0)\n"
            "- Part_B = ∑_k ∑_i ∑_j (if i+j=k then μ_i·δ_j else 0)\n\n"
            "Current proof:\n```\n"
            "simp only [hQ_def, Part_A, Part_B, ← Finset.sum_add_distrib]\n"
            "```\n"
            "This fails with 'AddCommMonoid Prop' because Lean can't determine the sum type.\n\n"
            "Fix: try `simp_rw` instead of `simp only` (works inside binders), or use:\n"
            "```\n"
            "rw [hQ_def]; simp_rw [← Finset.sum_add_distrib]\n"
            "congr 1; ext k; congr 1; ext i; congr 1; ext j\n"
            "split_ifs <;> [exact h_two_term i j; ring]\n"
            "```\n\n"
            "Fix ONLY hQ_eq."
        ),
    },
    # ── 7. discretization_autoconv_error: hPartA_exch + hg_eq ──
    {
        "name": "P9_disc_partA_exch",
        "prompt": (
            "In complete_proof.lean, theorem `discretization_autoconv_error` (line 2640), "
            "the sub-proofs `hPartA_exch` and `hg_eq` (lines 2763-2788) have errors.\n\n"
            "hPartA_exch shows: Part_A = ∑_j w_j * (∑_k ∑_i (if i+j=k then δ_i else 0))\n"
            "Current proof uses `show _ = _; rw [show ... from by ...]` which fails with "
            "'Tactic rewrite failed' and 'failed to synthesize AddCommMonoid Prop'.\n\n"
            "The mathematical steps are: (1) factor w_j out of inner sum via split_ifs/ring, "
            "(2) swap ∑_i ∑_j via Finset.sum_comm, (3) factor w_j from ∑_k via Finset.mul_sum.\n\n"
            "Try using `conv` blocks to target specific sub-expressions, or `simp_rw` for rewrites "
            "inside binders. Also try adding `with hPartA_def` when defining Part_A to get an equation.\n\n"
            "hg_eq (lines 2778-2787) converts g_fn from a double sum to a filtered single sum. "
            "It uses `Finset.sum_ite_eq'` and `split_ifs` which fail. Try `simp [Finset.sum_ite]` "
            "or `Finset.sum_comm` + `Finset.sum_ite_eq'` + omega.\n\n"
            "Fix hPartA_exch and hg_eq."
        ),
    },
    # ── 8. discretization_autoconv_error: hPartA_le (contributing bins) ──
    {
        "name": "P10_disc_partA_le",
        "prompt": (
            "In complete_proof.lean, theorem `discretization_autoconv_error` (line 2640), "
            "the sub-proof `hPartA_le` (lines 2814-2832) has errors at:\n\n"
            "- Line 2823: `hj.2` fails — 'Projections cannot be used on functions'. "
            "  `hj` comes from `simp [Finset.mem_filter] at hj` and has type `j ∈ CB → False` "
            "  (an implication, not a conjunction). Fix: destructure differently, or use "
            "  `have hjm := (Finset.mem_filter.mp hj).2` to extract the membership.\n"
            "- Line 2829: `rw [Finset.sum_filter]` fails. The pattern involves "
            "  `∑ a ∈ S with ?p a, ?f a` but the target has a different form. "
            "  Try `Finset.sum_filter_of_ne` or rewrite the contributing_bins membership differently.\n\n"
            "The same patterns repeat for hPartB at lines 2914, 2920. Fix both hPartA_le and hPartB_le."
        ),
    },
    # ── 9. discretization_autoconv_error: hPartB + hCB ──
    {
        "name": "P11_disc_partB_hCB",
        "prompt": (
            "In complete_proof.lean, theorem `discretization_autoconv_error` (line 2640), "
            "the sub-proof for hPartB_exch (lines 2833-2856) has the same pattern as hPartA_exch "
            "(rewrite failures, AddCommMonoid Prop). Apply the same fix as for hPartA_exch.\n\n"
            "Also, the CB (contributing bins) contiguity argument (lines 2893-2905) has:\n"
            "- Line 2893: `linarith` fails for `∑ δ_i ≥ -1/m`\n"
            "- Line 2899: `omega` fails in `simp [Finset.mem_filter]; omega`\n"
            "- Line 2902: `omega` fails in the Nat.min bound\n"
            "- Line 2905: `linarith` fails for empty filter case\n\n"
            "These are subtle Nat arithmetic issues. The `contributing_bins_iff` rewrite "
            "may produce different filter forms than expected. Try `simp only [Finset.mem_filter, "
            "Finset.mem_univ, true_and]` instead of `simp [Finset.mem_filter]`, and provide "
            "explicit intermediate bounds for omega/linarith.\n\n"
            "Also line 2926: final `linarith` for Q ≤ 1/m²+2W/m. May need explicit `have` steps.\n\n"
            "Fix hPartB_exch, hh_eq, the CB argument, and the final linarith."
        ),
    },
    # ── 10. line 3087 rewrite ──
    {
        "name": "P12_final_rewrite",
        "prompt": (
            "In complete_proof.lean, at line 3087, there is a rewrite failure:\n"
            "'Tactic `rewrite` failed: Did not find an occurrence of the pattern'\n\n"
            "Please examine what theorem this is in, what the rewrite target is, "
            "and fix it. It may be a simple pattern mismatch that can be fixed with "
            "`simp_rw` instead of `rw`, or by providing explicit type annotations.\n\n"
            "Fix ONLY this one error."
        ),
    },
]

def submit_project(prompt_data):
    body = json.dumps({"prompt": prompt_data["prompt"]})
    with open(TAR_PATH, "rb") as f:
        tar_content = f.read()
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{API_URL}/project",
            headers={"X-API-Key": API_KEY},
            data={"body": body},
            files=[("input", ("lean_project.tar.gz", tar_content, "application/x-tar"))],
        )
        response.raise_for_status()
        return response.json()

def check_status(project_id):
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{API_URL}/project/{project_id}",
            headers={"X-API-Key": API_KEY},
        )
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        try:
            with open(os.path.join(os.environ.get("TEMP", "/tmp"), "aristotle_projects_v2.json"), "r") as f:
                projects = json.load(f)
        except FileNotFoundError:
            print("No projects file found. Run without arguments to submit.")
            sys.exit(1)
        for p in projects:
            if p.get("project_id"):
                status = check_status(p["project_id"])
                print(f"  {p['name']}: {status['status']} ({status.get('percent_complete', '?')}%)")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "results":
        import tarfile
        try:
            with open(os.path.join(os.environ.get("TEMP", "/tmp"), "aristotle_projects_v2.json"), "r") as f:
                projects = json.load(f)
        except FileNotFoundError:
            print("No projects file found.")
            sys.exit(1)
        for p in projects:
            if not p.get("project_id"):
                continue
            status = check_status(p["project_id"])
            s = status["status"]
            print(f"\n=== {p['name']} ({s}) ===")
            if s in ("COMPLETE", "COMPLETE_WITH_ERRORS", "OUT_OF_BUDGET"):
                with httpx.Client(timeout=60) as client:
                    resp = client.get(
                        f"{API_URL}/project/{p['project_id']}/result",
                        headers={"X-API-Key": API_KEY},
                    )
                    if resp.status_code == 200:
                        out_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"aristotle_v2_{p['name']}.tar.gz")
                        with open(out_path, "wb") as f2:
                            f2.write(resp.content)
                        # Extract and diff
                        out_dir = os.path.join(os.environ.get("TEMP", "/tmp"), f"aristotle_v2_out_{p['name']}")
                        os.makedirs(out_dir, exist_ok=True)
                        with tarfile.open(out_path, "r:gz") as t:
                            t.extractall(out_dir)
                        lean_out = os.path.join(out_dir, "lean_project_aristotle", "complete_proof.lean")
                        lean_cur = os.path.join("C:", os.sep, "Users", "andre", "OneDrive - PennO365",
                                                "Desktop", "compact_sidon", "lean", "complete_proof.lean")
                        import difflib
                        with open(lean_out, encoding="utf-8") as a, open(lean_cur, encoding="utf-8") as b:
                            diff = list(difflib.unified_diff(b.readlines(), a.readlines(), n=2))
                        print(f"  Downloaded. Diff: {len(diff)} lines changed")
                    else:
                        print(f"  Failed to download: {resp.status_code}")
        sys.exit(0)

    # Submit all prompts
    results = []
    for i, prompt_data in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] Submitting: {prompt_data['name']}...")
        try:
            result = submit_project(prompt_data)
            pid = result["project_id"]
            print(f"  -> {pid} ({result['status']})")
            results.append({"name": prompt_data["name"], "project_id": pid})
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"name": prompt_data["name"], "project_id": None, "error": str(e)})
        time.sleep(1)

    out_file = os.path.join(os.environ.get("TEMP", "/tmp"), "aristotle_projects_v2.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_file}")
    print(f"Check: python submit_to_aristotle.py status")
    print(f"Pull:  python submit_to_aristotle.py results")
