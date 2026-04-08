# 9. Correctness Guarantees

## 9.1 Mathematical Rigor Requirements

This kernel is part of a **rigorous mathematical proof**. A missed survivor invalidates the proof entirely. Requirements:

1. **Exact integer arithmetic for autoconvolution** — no floating-point approximation in conv values
2. **Conservative threshold comparison** — the `one_minus_4eps` guard and `eps_margin` must be preserved exactly
3. **Complete enumeration** — every child in the Cartesian product must be tested
4. **Correct Gray code traversal** — must visit every element exactly once
5. **Correct canonicalization** — `min(child, reverse(child))` must be lexicographically correct

## 9.2 Verification Strategy

1. **Small-case exact match:** Run L0-L2 on both CPU and GPU, verify identical survivor sets
2. **Random parent spot-check:** For L3/L4, select 1000 random parents, verify GPU survivors ⊇ CPU survivors (GPU may have more pre-dedup due to ordering, but after dedup must be identical)
3. **Inject known survivors:** Insert compositions that should survive, verify they appear in output
4. **Boundary case testing:** Test parents with 0, 1, and maximum children
5. **Gray code completeness:** Verify `total_tested == product(hi-lo+1)` for each parent

## 9.3 Floating-Point Reproducibility

The threshold computation involves float64:
```
dyn_x = (c_target * m * m + 3.0 + (double)W_int / (2.0 * n) + eps_margin) * 4.0 * n * ell
dyn_it = (int64_t)(dyn_x * one_minus_4eps)
```

CUDA float64 operations are IEEE 754 compliant on H100. The `one_minus_4eps` guard provides 4 ULPs of margin, which is sufficient for any rounding mode. **However, ensure `--ftz=false --prec-div=true --prec-sqrt=true` compiler flags are set** to avoid flush-to-zero or reduced-precision optimizations.

**CRITICAL: Use `-fmad=false` for the threshold computation to ensure the multiply-add isn't fused** (FMA can change rounding). Alternatively, compute thresholds on CPU and upload as a precomputed table (already done via `threshold_table`).
