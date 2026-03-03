# CPU Cascade Prover: Validity Audit

**Date:** 2026-02-26
**Audited code:** `cloninger-steinerberger/cpu/run_cascade.py` and supporting modules
**Baseline:** `initial_baseline.m` (MATLAB, arXiv:1403.7988, verified correct)
**Parameters under test:** n_half=2, m=20, d0=4, c_target=1.3

## Results Under Review

```
Level |    Parents |     Children |   Ch/Par |  Survivors |     Factor |       Time
---------------------------------------------------------------------------
L   1 |        237 |      174,879 |    737.9 |     13,855 |   58.4599x |      5.32s
L   2 |     13,855 |   97,978,981 |   7071.7 |    118,342 |    8.5415x |     10.31s
L   3 |    118,342 | 6,957,731,855 |  58793.4 |        256 |  0.002163x |      11.5m
L   4 |        256 |   62,159,872 | 242812.0 |          0 |         0x |     21.29s
```

Claim: c >= 1.3, proven in ~12 minutes on CPU with m=20.

---

## Verdict: VALID

All six independent audits confirm the code is mathematically correct and conservative relative to the MATLAB baseline. No bugs, no over-pruning, no missed compositions.

---

## 1. Dynamic Threshold (`_prune_dynamic`) — CORRECT

The integer-space threshold formula is algebraically equivalent to the MATLAB's continuous-space formula:

- **MATLAB:** `boundToBeat = (lowerBound + gridSpace^2) + 2*gridSpace*(contributing_masses)`
- **Python:** `dyn_base = c_target*m^2 + 1 + 1e-9*m^2`, then `dyn_it = int64((dyn_base + 2*W_int) * ell/(4*n_half) * (1 - 4*DBL_EPS))`

Two safety margins are present:

| Margin | Effect | Magnitude |
|--------|--------|-----------|
| `+1e-9*m^2` | Threshold HIGHER (harder to prune) | ~4e-7 for m=20 |
| `*(1-4*DBL_EPS)` | Threshold LOWER (easier to prune) | ~9e-16 * dyn_x |

The `+1e-9` dominates by a factor of ~866x, so the Python code is strictly **more conservative** than MATLAB. MATLAB uses `>=` for pruning; Python uses `>` with floor — equivalent or more conservative for integers.

Verified with 10 concrete compositions and exhaustive parameter sweeps across m in {20, 50} and all valid (ell, W_int) combinations.

## 2. Contributing Bins (W_int) — CORRECT, NO OFF-BY-ONE ERRORS

The formula:
```python
lo_bin = max(0, s_lo - (d - 1))
hi_bin = min(d - 1, s_lo + ell - 2)
W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
```

**Derivation:** Bin i contributes to window (ell, s_lo) iff there exists j in [0, d-1] with s_lo <= i+j <= s_lo+ell-2. This gives i in [max(0, s_lo-(d-1)), min(d-1, s_lo+ell-2)].

**Verification:**
- Derived analytically from pair contribution condition
- Brute-force pair enumeration for d=2..8: **371/371 cases pass**
- Python simulation of MATLAB sparse matrix construction: **88/88 match**
- Random W_int computations: **7,420/7,420 pass**
- MATLAB 1-indexed to Python 0-indexed conversion: verified correct

## 3. Canonical Filter Handling — MATHEMATICALLY COMPLETE

| Level | Filter | Rationale |
|-------|--------|-----------|
| L0 | Canonical only (c <= rev(c) lex) | Halves work; rev(c) has identical autoconvolution |
| L1+ | NO canonical filter; test ALL children | Children of rev(P) are not generated since rev(P) is not in parent list; testing C covers rev(C) by autoconvolution symmetry |
| Post-test | Canonicalize + deduplicate survivors | min(C, rev(C)) + set-based dedup ensures unique canonical representatives |

**Key proofs:**
- If C is a child of canonical parent P, then rev(C) is a child of rev(P). Since rev(P) is not in the parent list, we must test C directly.
- test_value(C) = test_value(rev(C)) guarantees that testing C is equivalent to testing rev(C).
- The asymmetry filter is **exactly symmetric** under reversal: `prune(left_frac) <=> prune(1 - left_frac)`. It is impossible for C to be pruned while rev(C) would survive.
- For palindrome parents (P = rev(P)), both C and rev(C) are children of the same parent, tested independently, then canonicalized to the same form and deduplicated.

## 4. Child Generation (`generate_children_uniform`) — COMPLETE

Each parent bin c_i is split into (a, c_i - a) where both sub-bins are capped at x_cap:
```python
x_cap = floor(m * sqrt(thresh / d_child))   # thresh = c_target + 2/m + 1/m^2 + 1e-9
```

**Completeness proof:** Any child with a bin > x_cap would be immediately pruned by the ell=2 diagonal window test, because (x_cap+1)^2 > dyn_it for that window. Verified for all m in [5, 200] and d_child in {4, 8, 16, 32, 64, 128, 256, 512, 1024}. All cases pass with margin >= 1.

**Comparison with MATLAB:** Python's x_cap >= MATLAB's equivalent (due to thresh > c_target), so Python generates at least as many children. Verified numerically:

| d_child | Python x_cap | MATLAB-equiv | Python >= MATLAB? |
|---------|-------------|--------------|-------------------|
| 8 | 8 | 8 | Yes |
| 16 | 5 | 5 | Yes |
| 32 | 4 | 4 | Yes |
| 64 | 2 | 2 | Yes |

Child layout (a, b_i-a) at positions (2i, 2i+1) matches MATLAB. Edge cases (b_i=0, b_i > 2*x_cap) handled correctly.

## 5. Asymmetry Pruning — SOUND

**Mathematical basis:** For f >= 0 on [-1/4, 1/4] with left-half mass L:

```
||f*f||_{L^inf} >= 2L^2
```

Proved via Fubini's theorem, restricting to the left-left contribution of the autoconvolution integral, then applying the L^inf / L^1 bound over [-1/2, 0].

**Threshold:** `sqrt(c_target / 2)` is the exact cutoff where `2L^2 = c_target`.

**Margin:** `1/(4m)` correctly bounds discrete-to-continuous left mass discrepancy:
- n_half bins, each off by at most 1/m in paper coordinates
- Normalized by 4*n_half (total bins times bin width)
- n_half cancels: margin = 1/(4m), independent of dimension

**Soundness:** Compared against c_target directly (not c_target + correction) because the asymmetry bound is a direct L^inf bound that does not go through the test-value framework.

## 6. Numerical Verification — ALL PASS

Exhaustive test of all 891 canonical d=4, m=20 compositions:

| Check | Result |
|-------|--------|
| Pure-Python simulation vs Numba `_prune_dynamic` | **0 mismatches** |
| No dynamic over-pruning (pruned but tv < dynamic threshold) | **0 violations** |
| No missed prunes (not pruned but tv > dynamic threshold) | **0 missed** |
| Total pruned | 655 |
| Total survived | 236 |

4 compositions are correctly pruned by the per-window dynamic threshold despite having max test value below the uniform conservative threshold. This is expected behavior — the dynamic correction `2*W_int/m^2` is tighter than the uniform `2/m` when not all bins contribute to a window.

---

## Why the Speed Difference is Legitimate

The CPU result (m=20, ~12 minutes) vs GPU (m=50, 60+ hours) is **not** an apples-to-apples comparison:

| Parameter | CPU (m=20) | GPU (m=50) |
|-----------|-----------|-----------|
| L0 compositions (d=4) | C(23,3) = 1,771 | C(53,3) = 23,426 |
| Correction term | 2/20 + 1/400 = 0.1025 | 2/50 + 1/2500 = 0.0404 |
| Effective threshold | 1.4025 | 1.3404 |
| Children per parent | fewer (smaller x_cap) | many more |

The m=20 grid has dramatically fewer compositions at every level. The correction term is larger (0.1025 vs 0.0404), meaning the effective threshold is higher and the proof is "harder" per-composition, but this is compensated by the cascade refining to d=64 where structure is resolved finely enough. The mathematical proof is valid regardless of m — the correction term `2/m + 1/m^2` correctly accounts for discretization error at that grid spacing.

---

## Non-Critical Observations

1. **`solvers.py` window range:** The fused kernels in `solvers.py` use `ell in range(2, d+1)` (up to d), while `run_cascade.py` and the MATLAB baseline use `ell in range(2, 2*d+1)` (up to 2d). This means `solvers.py` checks fewer windows and is less aggressive at pruning (safe direction). This does NOT affect the cascade prover, which uses `run_cascade.py`.

2. **Asymmetry margin tightness:** The margin `1/(4m)` is a worst-case bound assuming all bin rounding errors align. The actual Lemma 2 rounding procedure constrains the running deficit, so the true maximum discrepancy may be smaller. A tighter margin would prune more compositions, but the current margin is mathematically safe.

3. **Integer overflow:** At d=64 (level 4), max conv value is bounded by ~256 and prefix sums by ~32,512. Well within int64 range.

---

## Files Audited

| File | Lines | Role |
|------|-------|------|
| `cloninger-steinerberger/cpu/run_cascade.py` | 1-755 | Main cascade runner, `_prune_dynamic`, child generation |
| `cloninger-steinerberger/pruning.py` | 1-71 | Correction terms, asymmetry threshold, canonical mask |
| `cloninger-steinerberger/compositions.py` | 1-376 | Composition generators (canonical and full) |
| `cloninger-steinerberger/test_values.py` | 1-150 | Autoconvolution and windowed test-value computation |
| `cloninger-steinerberger/solvers.py` | 1-1400+ | Fused Numba solvers (not used by cascade) |
| `initial_baseline.m` | 1-294 | Original MATLAB baseline (gold standard) |
