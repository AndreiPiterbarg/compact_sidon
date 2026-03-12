# Prompt 4: Asymmetry Pruning

**Claim 2.1 only.** Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

$(f*f)(x) = \int f(t)f(x-t)\,dt$ is the autoconvolution (self-convolution).

### Definitions (in output.lean)

- `autoconvolution_ratio f` = $\|f*f\|_{L^\infty} / (\int f)^2$

---

## Claim 2.1: $\|f*f\|_\infty \geq 2L^2$

**Theorem.** Let $f \geq 0$ with $\operatorname{supp}(f) \subseteq [-1/4, 1/4]$ and $\int f = 1$. Let $L = \int_{-1/4}^{0} f$ (left-half mass). Then:

$$\|f * f\|_{L^\infty} \geq 2L^2$$

**Consequence.** If $L \geq \sqrt{c_\text{target}/2}$ then $R(f) = \|f*f\|_\infty \geq 2L^2 \geq c_\text{target}$. By symmetry (replacing $f(x)$ with $f(-x)$), same holds if $1 - L \geq \sqrt{c_\text{target}/2}$.

So any function with sufficiently asymmetric mass is automatically pruned.

### Proof

**Step 1: Restriction to left half.**

Define $f_L(x) = f(x) \cdot \mathbf{1}_{(-1/4, 0)}(x)$.

Since $f \geq 0$, we have $f(x) \geq f_L(x) \geq 0$ for all $x$.

For nonneg functions, convolution is monotone: if $0 \leq g \leq h$ pointwise, then $(g*g)(x) \leq (h*h)(x)$ for all $x$. This is because:
$$(h*h)(x) - (g*g)(x) = \int [h(t)h(x-t) - g(t)g(x-t)]\,dt$$
$$= \int [(h(t)-g(t))h(x-t) + g(t)(h(x-t)-g(x-t))]\,dt \geq 0$$

since each factor is $\geq 0$.

Therefore $\|f*f\|_\infty \geq \|f_L * f_L\|_\infty$.

**Step 2: Support of $f_L * f_L$.**

$\operatorname{supp}(f_L) \subseteq (-1/4, 0)$.

$\operatorname{supp}(f_L * f_L) \subseteq \{x + y : x, y \in (-1/4, 0)\} = (-1/2, 0)$.

This is an interval of length $1/2$.

**Step 3: Total mass by Fubini.**

$$\int (f_L * f_L)(x)\,dx = \int\int f_L(t) f_L(x-t)\,dt\,dx = \left(\int f_L\right)^2 = L^2$$

This uses Fubini's theorem: swap the order of integration in $\iint f_L(t) f_L(x-t)\,dx\,dt$, and the inner integral over $x$ gives $\int f_L = L$ for each $t$.

**Step 4: Averaging principle.**

For any nonneg measurable function $g$ supported on a set of measure $\lambda$:

$$\|g\|_{L^\infty} \geq \frac{1}{\lambda} \int g$$

(Proof: if $g(x) < \frac{1}{\lambda}\int g$ a.e. on the support, then $\int g < \frac{\lambda}{\lambda}\int g = \int g$, contradiction.)

Apply with $g = f_L * f_L$ (which is nonneg since it's a convolution of nonneg functions) and $\lambda = 1/2$:

$$\|f_L * f_L\|_\infty \geq \frac{L^2}{1/2} = 2L^2$$

**Combining:** $\|f*f\|_\infty \geq \|f_L * f_L\|_\infty \geq 2L^2$.

### Lean theorems to prove

```lean
-- Monotonicity of convolution for nonneg functions
theorem convolution_mono_nonneg (f g : ℝ → ℝ)
    (hf : ∀ x, 0 ≤ f x) (hg : ∀ x, 0 ≤ g x) (hfg : ∀ x, f x ≤ g x) :
    ∀ x, MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x ≤
         MeasureTheory.convolution g g (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume x := by
  sorry

-- Averaging principle: ‖g‖∞ ≥ (∫g) / measure(support)
theorem averaging_principle (g : ℝ → ℝ) (hg : ∀ x, 0 ≤ g x)
    (S : Set ℝ) (hS : Function.support g ⊆ S) (hS_meas : MeasureTheory.volume S = ENNReal.ofReal λ_val)
    (hλ : 0 < λ_val) :
    (MeasureTheory.eLpNorm g ⊤ MeasureTheory.volume).toReal ≥
      MeasureTheory.integral MeasureTheory.volume g / λ_val := by
  sorry

-- Main result
theorem asymmetry_bound (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_supp : Function.support f ⊆ Set.Icc (-1/4 : ℝ) (1/4))
    (hf_int : MeasureTheory.integral MeasureTheory.volume f = 1) :
    let L := MeasureTheory.integral MeasureTheory.volume (Set.indicator (Set.Ioo (-1/4 : ℝ) 0) f)
    (MeasureTheory.eLpNorm (MeasureTheory.convolution f f
      (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume).toReal ≥ 2 * L ^ 2 := by
  sorry
```
