# Prompt 6: Integer Dynamic Threshold and FP Margins

**Claims 2.4 + 5.1 + 5.2.** All about the integer-space threshold computation. Attach `output.lean` as context.

---

## Problem Context

We are proving $c \geq 1.4$ on the autoconvolution constant. The cascade prunes compositions using a dynamic threshold computed in integer/floating-point arithmetic. We must show the computation is sound: the threshold is conservative (at least as high as the exact mathematical value), so pruning never incorrectly eliminates a composition.

Parameters: $n = 2$ (half-bins), $m = 20$ (resolution), $c_\text{target} = 1.4$.

### Setup

- Integer masses: $c_i \in \mathbb{Z}_{\geq 0}$, $\sum c_i = m$.
- Integer autoconvolution: $\text{conv}[k] = \sum_{i+j=k} c_i c_j$ (exact integer).
- Window sum: $\text{ws} = \sum_{k=s_\text{lo}}^{s_\text{lo}+\ell-2} \text{conv}[k]$ (exact integer).
- Contributing mass: $W_\text{int} = \sum_{i \in \mathcal{B}} c_i$ (exact integer).

The dynamic threshold in floating-point:
$$\text{dyn\_it} = \lfloor (c_\text{target} \cdot m^2 + 1 + 10^{-9} m^2 + 2 W_\text{int}) \cdot \frac{\ell}{4n} \cdot (1 - 4\varepsilon) \rfloor$$

where $\varepsilon = 2.22 \times 10^{-16}$ (IEEE 754 double machine epsilon).

A composition is pruned when $\text{ws} > \text{dyn\_it}$.

---

## Claim 2.4: Integer Threshold is Conservative

**Theorem.** The computed `dyn_it` satisfies:

$$\text{dyn\_it} \geq \left\lfloor (c_\text{target} \cdot m^2 + 1 + 2 W_\text{int}) \cdot \frac{\ell}{4n} \right\rfloor$$

That is, the computed threshold is at least as large as the exact mathematical threshold (conservative = harder to prune = safe direction).

**Proof.** Let $A = (c_\text{target} \cdot m^2 + 1 + 2W_\text{int}) \cdot \ell/(4n)$ (exact). Let $\delta = 10^{-9} m^2 \cdot \ell/(4n)$ (additive margin). Then:

$$\text{dyn\_it} = \lfloor (A + \delta) \cdot (1 - 4\varepsilon) \rfloor = \lfloor A + \delta - 4\varepsilon(A + \delta) \rfloor$$

We need $\delta > 4\varepsilon(A + \delta)$, i.e., $\delta(1 - 4\varepsilon) > 4\varepsilon A$.

Since $4\varepsilon \approx 8.88 \times 10^{-16}$ is negligible: $\delta(1 - 4\varepsilon) \approx \delta > 4\varepsilon A$.

**Concrete bounds for $m = 20, n = 2$:**
- $\delta \geq 10^{-9} \cdot 400 \cdot 2/8 = 10^{-7}$ (minimum at $\ell = 2$).
- $A \leq (1.4 \cdot 400 + 1 + 40) \cdot 128/8 = 601 \cdot 16 = 9616$.
- $4\varepsilon \cdot A \leq 8.88 \times 10^{-16} \cdot 9616 \approx 8.54 \times 10^{-12}$.
- $\delta \geq 10^{-7} \gg 8.54 \times 10^{-12}$. ✓ (margin of $\sim 10^4$)

**General bound for $m \leq 200$:**
- $\delta \geq 10^{-9} \cdot m^2 \cdot 2/(4n) = 10^{-9} \cdot m^2 / (2n)$.
- $A \leq (c_\text{target} \cdot m^2 + 1 + 2m) \cdot 2d/(4n) = (c_\text{target} \cdot m^2 + 2m + 1) \cdot d/2n$.
- Ratio: $\delta / (4\varepsilon A) \geq 10^{-9} m^2 / (4\varepsilon(c_\text{target} m^2 + 2m + 1)) \geq 10^{-9}/(4 \cdot 2.22 \times 10^{-16} \cdot 2) \approx 10^6$. Always safe.

```lean
-- The additive margin dominates the multiplicative reduction
theorem dyn_it_conservative (c_target : ℝ) (m n ℓ W_int : ℕ)
    (hm : 0 < m) (hn : 0 < n) (hℓ : 0 < ℓ) (hW : W_int ≤ m) (hct : 0 ≤ c_target) :
    let A := (c_target * (m : ℝ)^2 + 1 + 2 * (W_int : ℝ)) * ((ℓ : ℝ) / (4 * (n : ℝ)))
    let B := (c_target * (m : ℝ)^2 + 1 + 1e-9 * (m : ℝ)^2 + 2 * (W_int : ℝ)) *
             ((ℓ : ℝ) / (4 * (n : ℝ))) * (1 - 4 * 2.22e-16)
    ⌊A⌋ ≤ ⌊B⌋ := by
  sorry
```

---

## Claim 5.1: FP Margin is Net Conservative (same as above)

This is the same result as Claim 2.4 stated differently. The $+10^{-9}m^2$ raises the threshold (conservative); the $\times(1-4\varepsilon)$ lowers it (aggressive). Net effect is conservative because the additive margin dominates by $\sim 10^6\times$.

(Already covered by the theorem above.)

---

## Claim 5.2: Integer Autoconvolution is Exact

**Theorem.** When $c_i \in \mathbb{Z}_{\geq 0}$:

1. $\text{conv}[k] = \sum_{i+j=k} c_i c_j \in \mathbb{Z}$.
2. All prefix sums and window sums are exact integers.
3. The comparison $\text{ws} > \text{dyn\_it}$ compares an exact integer against a floored value.

**Proof.** Integers are closed under multiplication and addition. Since $c_i$ are integers, each product $c_i c_j$ is integer, and the sum is integer. Prefix sums are cumulative sums of integers. Window sums are differences of prefix sums (integers).

The threshold $\text{dyn\_it} = \lfloor \cdot \rfloor$ is also an integer. So $\text{ws} > \text{dyn\_it}$ is an integer comparison.

```lean
-- Conv entries bounded by m²
theorem conv_bounded {d : ℕ} (c : Fin d → ℕ) (hc : ∑ i, c i = m) (m : ℕ) (k : ℕ) :
    ∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0 ≤ m ^ 2 := by
  sorry

-- Total sum of conv = m²
theorem conv_total_eq_m_squared {d : ℕ} (c : Fin d → ℕ) (hc : ∑ i, c i = m) (m : ℕ) :
    ∑ k ∈ Finset.range (2 * d - 1),
      (∑ i : Fin d, ∑ j : Fin d, if i.1 + j.1 = k then c i * c j else 0) = m ^ 2 := by
  sorry

-- For m ≤ 200: m² fits int32
theorem m_sq_int32 (m : ℕ) (hm : m ≤ 200) : m ^ 2 ≤ 2 ^ 31 - 1 := by
  omega
```
