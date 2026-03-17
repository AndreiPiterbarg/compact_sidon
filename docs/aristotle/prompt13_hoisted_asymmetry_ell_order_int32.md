# Prompt 13: Cauchy-Schwarz Single-Bin Bound

**Claim 4.5 only.** Attach `complete_proof.lean` as context.

> **NOTE:** Claims 4.6, 4.7, 4.8 are **ALREADY PROVED** in `complete_proof.lean`. Only Claim 4.5 remains.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. Grid: $d = 2n$ bins, compositions summing to $m$. Parameters: $n = 2, m = 20$.

### Definitions (in complete_proof.lean)

- `bin_masses f n i`, `discrete_autoconvolution`, `is_composition`
- `f_L` (left-half restriction), `convolution_mono_ae`, `averaging_principle`
- `support_convolution_subset_add`, `measure_support_convolution_bound`

---

## Claim 4.5: Cauchy-Schwarz x_cap Needs No Correction

**Theorem.** For any nonneg $f$ on $(-1/4, 1/4)$ with $\int f = 1$ and mass $M_i$ in bin $i$ (width $\Delta = 1/(4n)$):

$$\|f*f\|_\infty \geq d \cdot M_i^2$$

**Proof.** Restrict to bin $i$: let $f_i = f \cdot \mathbf{1}_{\text{bin}_i}$. Since $f \geq f_i \geq 0$: $\|f*f\|_\infty \geq \|f_i * f_i\|_\infty$.

$\operatorname{supp}(f_i * f_i)$ has length $2\Delta = 1/(2n)$. By Fubini: $\int(f_i * f_i) = M_i^2$. By averaging: $\|f_i * f_i\|_\infty \geq M_i^2/(1/(2n)) = 2n \cdot M_i^2 = d \cdot M_i^2$.

This is a DIRECT $L^\infty$ bound — no correction term needed.

**Hint:** The proof structure is identical to the asymmetry bound (Claim 2.1) already formalized in `complete_proof.lean` via `f_L`, `convolution_mono_ae`, `averaging_principle`, and `measure_support_convolution_bound`. The key difference is restricting to a single bin (width $1/(4n)$) instead of the left half (width $1/4$).

```lean
-- Same structure as Claim 2.1 but restricted to a single bin
theorem single_bin_bound (n : ℕ) (hn : n > 0)
    (f : ℝ → ℝ) (hf : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Ioo (-1/4 : ℝ) (1/4))
    (i : Fin (2 * n)) (M_i : ℝ) (hM : M_i = bin_masses f n i) :
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥
    (2 * n : ℝ) * M_i ^ 2 := by
  sorry
```
