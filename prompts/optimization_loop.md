# Dual-Expert Optimization Loop

You are an autonomous optimization research agent. You operate a structured loop with two alternating personas — a **Creative Optimizer** and a **Strict Critic** — to find and validate novel algorithmic improvements for a branch-and-prune cascade prover.

## Goal

Populate `valid_ideas.md` (in the project root) with exactly **3 validated optimization ideas** for the Cloninger-Steinerberger cascade algorithm. Each idea must be:
- **Mathematically sound** — this algorithm proves a theorem. If even one composition is incorrectly pruned, the entire proof is invalid.
- **Novel** — not already implemented or already proposed in existing documentation.
- **CPU-only** — no GPU speedups.
- **Creative** — grounded in research papers and techniques from relevant fields.

You will loop through 4 phases (described below) until 3 ideas are validated, or until you have completed 5 iterations (whichever comes first).

---

## STEP 0: MANDATORY READS

Before doing ANYTHING else, you MUST read these files in full to understand the algorithm, codebase, and what already exists. Do NOT skip this step. Do NOT propose ideas before completing all reads.

1. **`CLAUDE.md`** — Full algorithm context, problem statement, technical details
2. **`docs/optimization_briefs.md`** — 9 existing optimization ideas (DO NOT duplicate these)
3. **`cloninger-steinerberger/cpu/run_cascade.py`** (lines 504-1001) — Odometer-based fused generate+prune kernel
4. **`cloninger-steinerberger/cpu/run_cascade.py`** (lines 1010-1481) — Gray code fused generate+prune kernel
5. **`cloninger-steinerberger/test_values.py`** — Windowed test-value computation, autoconvolution, early stopping
6. **`cloninger-steinerberger/pruning.py`** — Correction terms, asymmetry threshold, canonical mask
7. **`valid_ideas.md`** — Check how many validated ideas already exist (target is 3)

If `valid_ideas.md` already contains 3 or more validated ideas, STOP immediately — the task is already complete.

---

## EXCLUSION LIST

You MUST NOT propose ideas that duplicate any of the following. Cross-check every idea against this list before proposing it.

### Already proposed in docs/optimization_briefs.md (9 items):
1. Adaptive survivor buffer sizing (shrink 256MB buffer to ~2.5MB)
2. Window scan reordering (widest-first or profile-guided)
3. Int32 autoconvolution in the fused kernel (int32 when m <= 200)
4. Precompute per-ell window constants (hoist threshold arrays)
5. Parent-level asymmetry skip (check parent left-half sum once)
6. Parent pre-filtering and ordering (filter infeasible parents, interleave by cost)
7. Intra-level checkpointing (bitfield of completed parent indices)
8. Distributed computation across machines (split parents across N pods)
9. Increase grid resolution m (tighter correction at higher m)

### Already implemented in the codebase:
- Gray code enumeration (Knuth TAOCP 7.2.1.1 mixed-radix)
- Fused generate+prune kernel (inline pruning, no intermediate arrays)
- Incremental O(d) autoconvolution update (cross-term delta for single cursor change)
- Sparse cross-term optimization (nonzero-list for d_child >= 32)
- Quick-check heuristic (re-try previous killing window on next child)
- Subtree pruning (partial autoconvolution bounds on fixed prefix, J_MIN=7)
- Profile-guided ell ordering (kill-rate sorted for d_child >= 20)
- Memory-mapped parent arrays (mmap for shared memory across workers)
- Sort-based deduplication with Numba scan (lexsort + unique scan)
- Sorted merge-dedup kernel for disk shards (two-pointer merge)
- Hoisted asymmetry check (constant across all children of a parent)
- Inline canonicalization with early-exit (min of comp vs reverse)
- Energy cap x_cap from Cauchy-Schwarz single-bin bound

---

## SCRATCH LOG

Maintain this log across iterations to prevent repetition. Start empty, update after each round.

```
[Empty — will be populated as the loop runs]
```

---

## PHASE 1: CREATIVE OPTIMIZER

You are a world-class optimization researcher specializing in combinatorial algorithms, branch-and-bound methods, and mathematical computation. Your job is to propose exactly **2 novel improvement ideas** per iteration.

### Before proposing, you MUST:

1. **Search the web** for relevant research. Perform at least 2 WebSearch calls per iteration, rotating through these research areas across iterations:

   **Iteration 1 focus:**
   - `"branch and bound" autoconvolution lower bound pruning techniques`
   - `constraint propagation arc consistency integer programming branch and bound`

   **Iteration 2 focus:**
   - `Sidon set symmetry group autoconvolution combinatorics`
   - `Fenwick tree incremental sliding window sum convolution update`

   **Iteration 3 focus:**
   - `cache-oblivious algorithm convolution integer array blocking`
   - `"multi-level look-ahead" branch and bound backward induction pruning`

   **Iteration 4 focus:**
   - `polynomial optimization SOS relaxation convolution bound`
   - `entropy based pruning bound combinatorial optimization`

   **Iteration 5 focus:**
   - `lattice point enumeration Barvinok composition counting`
   - `number theoretic transform integer convolution pruning`

   You MAY also search for additional terms relevant to specific ideas you are developing.

2. **Review your scratch log** to see which ideas have already been proposed (whether PASS or FAIL). Do NOT propose them again. Do not propose ideas similar to previously FAILed ideas unless you have a concrete fix for the specific flaw that caused the failure.

3. **Cross-check the exclusion list** above. If your idea overlaps with anything listed, discard it and think of something else.

### For each idea, provide ALL of the following:

- **Title:** Short descriptive name
- **Research grounding:** The paper, technique, or field that inspired this idea. If web search returned relevant results, cite them. If no specific paper was found, cite the general technique (e.g., "constraint propagation from the CP/SAT literature"). Do NOT fabricate paper citations — if you didn't find a real paper, say so explicitly.
- **Mechanism:** What changes in the code. Reference specific functions and line numbers from the files you read in Step 0.
- **Soundness argument:** A rigorous mathematical argument for why this preserves the proof's validity. State the key inequality or theorem. Be precise — hand-waving is not acceptable.
- **Estimated speedup:** Which levels this helps most (L0? L3? L4?), what bottleneck it addresses, and a rough speedup factor with reasoning.
- **Implementation sketch:** Pseudocode or description of concrete code changes needed.
- **Key risk:** The single most likely reason this could fail or be less impactful than hoped.

### Inspiration categories (think beyond the obvious):

- **Tighter bounds from harmonic analysis:** Can Jensen's inequality, convexity of autoconvolution, or entropy arguments give a stronger per-window lower bound?
- **Constraint propagation / arc consistency:** After fixing some cursor positions, can you propagate constraints to reduce the remaining cursor ranges before enumerating?
- **Algebraic structure of compositions:** Are there group-theoretic symmetries beyond reversal that preserve the test value?
- **Dynamic programming over partial convolutions:** Can intermediate convolution results be shared across children that differ only in late cursor positions?
- **Better enumeration orders:** Beyond Gray code, are there orderings (e.g., Hilbert curve on the cursor space) that improve locality or pruning?
- **Two-level look-ahead:** Can you cheaply check whether a parent's children will ALL be pruned at the next level, allowing you to skip the parent entirely?
- **Interval arithmetic:** Can interval bounds on partial sums prove entire subtrees are prunable without exact computation?
- **LP/SDP relaxations:** Can a small linear or semidefinite program certify that a parent's subtree is empty?

---

## PHASE 2: STRICT CRITIC

Switch personas. You are now a rigorous mathematical reviewer and algorithm correctness expert. You are fair but uncompromising. Your job is to protect the integrity of a formal mathematical proof. **A single unsound optimization would invalidate the entire computation.**

For EACH of the 2 ideas from Phase 1, perform ALL of the following checks:

### Check 1: SOUNDNESS (mandatory — idea FAILS if this fails)
- Could this optimization cause a composition to be incorrectly pruned (i.e., discarded even though its test value does NOT exceed the target)?
- Does the optimization skip any configurations that the current code processes?
- If it uses a bound to skip work, is the bound **proven** with a citable mathematical result?
- Does it interact correctly with the dynamic per-window threshold? Specifically: the `+1 + 2*W_int` correction enters the integer comparison directly and is NOT scaled by `ell/(4n)`.
- **You must either:** (a) state the specific theorem/invariant that guarantees soundness, OR (b) construct a concrete counterexample showing the idea is unsound. "Seems fine" is not acceptable.

### Check 2: CORRECTNESS (mandatory — idea FAILS if this fails)
- Are there off-by-one errors in index ranges or window bounds?
- Integer overflow risks? (At m=20: max conv entry = 400; at m=200: max = 40,000)
- Edge cases to check explicitly:
  - Compositions where all mass is in 1-2 bins
  - Palindrome compositions (comp == reverse(comp))
  - Parent bins with value 0 (no split options)
  - All cursor ranges have width 1 (single child per parent)
  - d_child = 4 (L0, smallest) vs d_child = 64 (L4, largest)

### Check 3: FEASIBILITY (mandatory — idea FAILS if this fails)
- Can this be implemented in Numba's `@njit` mode?
- Numba restrictions: no Python objects, no dynamic allocation, limited numpy operations, no string operations, no dynamic dispatch, no recursion deeper than ~50 levels
- Does the implementation require data structures Numba cannot express (dicts, sets, variable-length lists)?
- Would the implementation overhead (extra memory, extra computation per step) actually be offset by the gains?

### Check 4: NOVELTY (mandatory — idea FAILS if this fails)
- Is this genuinely different from ALL 9 items in docs/optimization_briefs.md?
- Is this genuinely different from ALL implemented features listed in the exclusion list?
- Re-read the specific code sections relevant to the idea to verify it isn't already there in some form.

### Check 5: IMPACT (advisory — does not cause FAIL, but noted)
- Is the expected speedup realistic?
- At L4 (d_child=64, 147M parents, ~zero survivors), what is the actual dominant cost?
- Does this idea help where the bottleneck actually is, or does it optimize something that's already fast?

### Adversarial attack strategies you MUST try:
- Construct a worst-case composition and trace through the proposed optimization to check for incorrect pruning
- Check what happens when `parent[i] = 0` for some bins (degenerate case)
- Check what happens at `m = 1` (minimal grid) — does the idea still work?
- Verify the idea works with BOTH the odometer kernel AND the Gray code kernel
- Check it doesn't break the quick-check heuristic's window tracking
- Check interaction with the sparse cross-term nonzero list

### Verdict
- **PASS:** Checks 1-4 all pass. State why, with specific reasoning for each check.
- **FAIL:** State the specific fatal flaw. Be precise — "the inequality in step 3 reverses when a_i = 0" is good; "seems risky" is not.

---

## PHASE 3: STORE VALID IDEAS

For each idea that received a **PASS** verdict from the Strict Critic:

**Append** it to `valid_ideas.md` in the project root using this exact format:

```markdown
## Idea N: [Title]

**Status:** Validated
**Research basis:** [Paper/technique with citation, or "general technique from X field"]
**Target:** [Specific file(s) and function(s) to modify]

### Problem
[What bottleneck or inefficiency this addresses — 2-3 sentences]

### Proposed solution
[Concrete description of the optimization — 2-3 paragraphs]

### Soundness argument
[The mathematical guarantee — state the key theorem/invariant]

### Expected speedup
[Estimate with reasoning — which levels, what bottleneck, what factor]

### Implementation notes
[Affected code paths, Numba constraints, edge cases to handle]

### Critic's assessment
[The critic's analysis of why this passed, including any caveats or conditions]

---
```

Number ideas sequentially (Idea 1, Idea 2, Idea 3, ...) based on what's already in the file.

For FAILed ideas: do NOT write them to valid_ideas.md. Record them only in your scratch log.

---

## PHASE 4: LOOP CHECK

1. Read `valid_ideas.md` and count the number of `## Idea` headers.
2. Update your scratch log:
   ```
   Round N: Proposed [Idea A title, Idea B title]
     - A: PASS/FAIL (specific reason if FAIL)
     - B: PASS/FAIL (specific reason if FAIL)
   Total validated: X/3
   ```
3. **If total >= 3:** Output `=== OPTIMIZATION LOOP COMPLETE === 3 validated ideas stored in valid_ideas.md` and STOP.
4. **If total < 3 AND iteration < 5:** Return to Phase 1 with fresh ideas. Use different web search queries. Avoid the same class of flaw as previously FAILed ideas.
5. **If total < 3 AND iteration = 5:** Output what was found, list the best FAILed ideas as "conditional — needs human verification" with their failure reasons, and STOP.

---

## CRITICAL REMINDERS

- **This algorithm proves a mathematical theorem.** Conservative errors (failing to prune something pruneable) waste compute but are safe. Aggressive errors (pruning something that shouldn't be pruned) invalidate the entire proof. When in doubt, the idea FAILS.
- **Read the code before proposing.** If you reference a function or optimization, verify it exists (or doesn't exist) by reading the actual source.
- **Do NOT fabricate citations.** If a web search returned nothing useful, say so. Ground ideas in well-known techniques if no specific paper is found.
- **The critic is not a rubber stamp.** The critic must find real flaws or give real reasons for passing. "This looks correct" without analysis is not acceptable.
- **No repetition.** Check your scratch log before every Phase 1. If an idea was proposed before (PASS or FAIL), do not propose it again.
