"""Submit SparseCrossTerm.lean to Aristotle for automated theorem proving."""
import asyncio
from aristotlelib import set_api_key, Project

set_api_key("arstl_jCrlrl_A2qiRoMb-vWH3CLRTBW_gK_AHhVCD6b2dyj8")

PROMPT = r"""
Prove ALL sorry's in Sidon/SparseCrossTerm.lean. This file formalizes Claims 4.26-4.35
of the Sidon autocorrelation project, certifying the sparse cross-term optimization
in a branch-and-prune cascade prover.

TARGET FILE: Sidon/SparseCrossTerm.lean

ALREADY PROVED (no sorry): nz_list_invariant (Claim 4.26), rebuild_nz_list_correct
(Claim 4.33), sparse_gate_correctness (Claim 4.34), sparse_cross_term_sound (Claim 4.35).

REMAINING sorry's TO PROVE (3 theorems):

1. swap_remove_preserves_invariant (Claim 4.28, ~line 143):
   When removing index i from nz_list via swap-remove (swap position p with the last
   element, decrement count), the resulting set equals the original set minus {i}.
   PROOF STRATEGY:
   - (→) If nz_list'[k'] = j for some k' < nz_count', trace k' through h_swap/h_rest
     to find a witness in the original nz_list. Then j ≠ i by h_distinct (if j = i,
     the witness would force k'' = p, but position p now holds nz_list[nz_count-1]).
   - (←) If nz_list[k0] = j and j ≠ i: if k0 < nz_count' and k0 ≠ p, use h_rest;
     if k0 = nz_count-1, use h_swap (nz_list'[p] = nz_list[nz_count-1] = j);
     k0 = p gives j = i, contradiction.
   KEY HYPOTHESES: h_distinct (injectivity), h_at_p (nz_list[p] = i),
   h_swap (nz_list'[p] = nz_list[nz_count-1]), h_rest (other positions unchanged),
   h_count' (nz_count' = nz_count - 1).

2. append_preserves_invariant (Claim 4.29, ~line 176):
   When appending index i at position nz_count and incrementing count, the resulting
   set equals the original set union {i}.
   PROOF STRATEGY:
   - (→) If k' < nz_count, use h_rest to get nz_list[k'] = j (left disjunct).
     If k' = nz_count, use h_append to get j = i (right disjunct).
   - (←) Left disjunct: nz_list[k0] = j implies nz_list'[k0] = j via h_rest,
     and k0 < nz_count < nz_count'. Right disjunct: j = i, use h_append with
     k' = nz_count < nz_count'.
   KEY HYPOTHESES: h_not_in (i not already present), h_append (nz_list'[nz_count] = i),
   h_rest (positions < nz_count unchanged), h_count' (nz_count' = nz_count + 1).

3. incremental_nz_update_correct (Claim 4.30, ~line 217):
   After the four-case update for bins k1 and k2, the nz_list invariant is restored.
   PROOF STRATEGY: Case split on whether i = k1, i = k2, or neither.
   - Case i = k1: use h_k1.symm (biconditional linking membership to child' k1 ≠ 0).
   - Case i = k2: use h_k2.symm.
   - Case i ≠ k1, i ≠ k2: chain h_unchanged (child' i = child i) with
     h_inv_before (old invariant) and h_rest (set membership unchanged).

4. sparse_cross_term_eq_dense (Claim 4.31, ~line 297):
   The sparse cross-term sum equals the dense cross-term sum for both delta1 and delta2.
   PROOF STRATEGY: For each component (k1 and k2), prove in two steps:
   Step 1 (zero-filtering): Show ∑_{j : Fin d} f(j) = ∑_{j : Fin d, child j ≠ 0} f(j)
   because when child j = 0, the term 2 * delta * child j = 2 * delta * 0 = 0.
   Step 2 (bijection): The nz_list invariant (h_inv) plus h_distinct plus h_valid
   gives a bijection between Fin nz_count and {j : Fin d | child j ≠ 0}.
   Use Finset.sum_nbij or Finset.sum_bij to reindex the sum.
   The conjunction ∧ is introduced by ⟨proof1, proof2⟩ where proof1 handles k1
   and proof2 handles k2 (identical structure with delta2 replacing delta1).

5. raw_conv_sparse_eq_dense (Claim 4.32, ~line 352):
   The full raw_conv arrays are equal after sparse vs dense cross-term updates.
   PROOF STRATEGY: Apply funext t, then rewrite using h_dense and h_sparse at t.
   The two cross-term sums are equal by sparse_cross_term_eq_dense (applied at t.1).
   Obtain ⟨h₁, h₂⟩ from the conjunction, rewrite h₁ and h₂, done.

MATHEMATICAL CONTEXT:
- This is about a branch-and-prune algorithm for proving lower bounds on the
  autoconvolution constant C_{1a}.
- The sparse optimization maintains an explicit list of nonzero child bins (nz_list)
  instead of iterating all d_child bins. Since zero bins contribute 0 to cross-term
  sums (2·δ·0 = 0), both approaches give identical results.
- The nz_list is maintained incrementally: when Gray code advances and bins k1, k2
  change, at most 2 add/remove operations keep it in sync.
- k1 and k2 are consecutive (k2 = k1 + 1), corresponding to a parent bin split.

LEAN-SPECIFIC HINTS:
- All arrays use Fin d indexing. nz_list maps Fin d → ℕ but only the first nz_count
  entries are meaningful.
- The file imports Mathlib and Sidon.Defs. Useful Mathlib lemmas include:
  Finset.sum_bij, Finset.sum_congr, Fin.ext, and standard omega/aesop tactics.
- For Claims 4.28/4.29, the proof involves careful case analysis on whether indices
  equal p (the swap position) or nz_count (the append position).
- For Claim 4.31, the key insight is that summing over all Fin d with a filter on
  child j ≠ 0 is the same as summing over the nz_list entries, because:
  (a) zero entries contribute 0 (mul_zero), and
  (b) the nz_list bijects onto exactly the nonzero entries (h_inv + h_distinct).
- For Claim 4.30, use Decidable.em or by_cases for the three-way case split on i.
- maxHeartbeats is set to 8000000, so the prover has generous resources.

Do NOT modify any theorem statement, hypothesis, or type signature.
Only replace `sorry` with valid proof terms.
Do NOT add or remove any theorems. Do NOT modify the already-proved theorems
(nz_list_invariant, rebuild_nz_list_correct, sparse_gate_correctness,
sparse_cross_term_sound).
"""

async def main():
    project = await Project.create_from_directory(
        prompt=PROMPT,
        project_dir=r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\lean",
    )
    print(f"Project ID: {project.project_id}")
    print(f"Status: {project.status}")

asyncio.run(main())
