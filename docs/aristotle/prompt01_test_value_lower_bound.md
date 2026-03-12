# Prompt 1: Test Value is a Lower Bound on ‖f∗f‖∞

**Claim 1.1 only.** Attach `output.lean` as context — it has all definitions and foundational lemmas.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

The support $(-1/4, 1/4)$ is divided into $d = 2n$ equal bins of width $\Delta = 1/(4n)$. A step function has height $c_i / m$ on bin $i$, where $(c_0, \ldots, c_{d-1})$ is a composition summing to $m$. The rescaled coordinates are $a_i = (4n/m) \cdot c_i$.

### Definitions (already in output.lean)

- **Discrete autoconvolution:** `discrete_autoconvolution a k = ∑_{i+j=k} a_i * a_j`
- **Test value:** `test_value n m c ℓ s_lo = (1/(4n·ℓ)) · ∑_{k=s_lo}^{s_lo+ℓ-2} conv[k]`
- **Max test value:** `max_test_value n m c = max over (ℓ, s_lo) of test_value`

### Already proved in output.lean

`canonical_cumulative_distribution_zero`, `canonical_cumulative_distribution_2n`, `bin_masses_nonneg`, `canonical_cumulative_distribution_mono`, `sum_fin_telescope`, `canonical_discretization_sum_zero_mass`, `canonical_discretization_eq`, `canonical_discretization_eq_diff`.

---

## Claim 1.1: Test Value ≤ ‖f∗f‖∞

**Theorem.** For any nonneg step function $f$ on the $d = 2n$ grid with integer masses $(c_0, \ldots, c_{d-1})$ summing to $m$ and $\int f = 1$:

$$b_{n,m}(c) \leq \|f * f\|_{L^\infty}$$

**Proof.**

Step 1: For a step function $f$ with $f(x) = c_i/m$ on bin $[i\Delta, (i+1)\Delta)$, the autoconvolution $(f*f)(x) = \int f(t)f(x-t)\,dt$ is a piecewise-linear function on intervals of length $\Delta$.

Step 2: For any interval $I$ of length $|I|$, $\|g\|_\infty \geq \frac{1}{|I|}\int_I g$ for nonneg $g$ (averaging principle). So $\|f*f\|_\infty \geq \frac{1}{\ell\Delta} \int_{s_\text{lo}\Delta}^{(s_\text{lo}+\ell-1)\Delta} (f*f)(x)\,dx$.

Step 3: Compute $\int_{s_\text{lo}\Delta}^{(s_\text{lo}+\ell-1)\Delta} (f*f)(x)\,dx$. For a step function, this integral equals (up to normalization):

$$\frac{\Delta}{(4n)^2 m^2} \cdot \sum_{k=s_\text{lo}}^{s_\text{lo}+\ell-2} \text{conv}[k] \cdot (\text{trapezoidal terms})$$

The key fact: on the interval $[k\Delta, (k+1)\Delta]$, the integral of $(f*f)$ is at least $\Delta \cdot \text{conv}[k] / (4n \cdot m)^2$ (since the piecewise-linear function on this interval has average $\geq$ its value at the midpoint, and the discrete convolution gives the value at grid points).

Step 4: More precisely, for a step function with bins of width $\Delta$:
$(f*f)(k\Delta - 1/4) = \frac{1}{m^2} \sum_{i+j=k} c_i c_j = \text{conv}[k]/(4n)^2$ (since $a_i = 4n \cdot c_i/m$).

Actually, the exact relationship is:
$$\int_{s_\text{lo}\Delta}^{(s_\text{lo}+\ell-1)\Delta} (f*f)(x)\,dx \geq \frac{\Delta}{(4n/m)^2} \sum_{k=s_\text{lo}}^{s_\text{lo}+\ell-2} \text{conv}_c[k]$$

where $\text{conv}_c[k] = \sum_{i+j=k} c_i c_j$ (in integer $c_i$ coordinates).

Step 5: Dividing by $\ell\Delta$ and noting $\int f = 1$:
$$\|f*f\|_\infty \geq \frac{1}{\ell \cdot (4n)^2/m^2} \sum_{k} \text{conv}_c[k] = \frac{m^2}{(4n)^2 \ell} \sum_{k} \text{conv}_c[k]$$

The test value in rescaled $a_i$ coordinates is $\text{TV} = \frac{1}{4n\ell}\sum_k \text{conv}_a[k]$ where $\text{conv}_a[k] = \sum_{i+j=k} a_i a_j = (4n/m)^2 \text{conv}_c[k]$.

So $\text{TV} = \frac{(4n/m)^2}{4n\ell} \sum_k \text{conv}_c[k] = \frac{4n}{m^2 \ell} \sum_k \text{conv}_c[k]$.

And $\|f*f\|_\infty \geq \frac{m^2}{(4n)^2 \ell} \sum_k \text{conv}_c[k] \cdot \frac{1}{\Delta} = \ldots$

The precise derivation requires careful tracking of the normalization. The key point is that the test value is the average of $(f*f)$ over an interval, and the $L^\infty$ norm is at least any average.

Step 6: Maximizing over $(\ell, s_\text{lo})$ gives $b_{n,m}(c) \leq \|f*f\|_\infty$.

**Lean theorem to prove:**

```lean
-- Define the step function
noncomputable def step_function (n m : ℕ) (c : Fin (2 * n) → ℕ) : ℝ → ℝ :=
  fun x =>
    let d := 2 * n
    let δ := 1 / (4 * n : ℝ)
    if x < -1/4 ∨ x ≥ 1/4 then 0
    else
      let i := ⌊(x + 1/4) / δ⌋.toNat
      if h : i < d then (c ⟨i, h⟩ : ℝ) / m
      else 0

-- Test value is a lower bound on the L^infty norm of the autoconvolution
theorem test_value_le_Linfty (n m : ℕ) (hn : n > 0) (hm : m > 0)
    (c : Fin (2 * n) → ℕ) (hc : ∑ i, c i = m) :
    (max_test_value n m c : ℝ) ≤ autoconvolution_ratio (step_function n m c) := by
  sorry
```
