# Tightest Valid Pruning Bound for the Coarse-Grid Cascade

## 0. Setup and Notation

Bins are indexed `i = 0, ..., d-1`. Integer mass coordinate `k_i >= 0`, `sum k_i = S`,
and physical mass `mu_i = k_i / S`. The grid cell around `mu*_i = k_i / S` is

    Cell = { mu : mu_i = mu*_i + delta_i, |delta_i| <= h, sum delta_i = 0 },
    where h := 1 / (2S).

For window `W = (ell, s)` (ell-1 contiguous convolution indices `s, s+1, ..., s+ell-2`),

    TV_W(mu) = (2d/ell) * sum_{k=s}^{s+ell-2} sum_{i+j=k} mu_i mu_j
            = (2d/ell) * mu^T A_W mu,
    A_W in R^{dxd}, (A_W)_{i,j} = 1{ s <= i+j <= s+ell-2 }.

The quadratic Taylor expansion is exact:

    TV_W(mu* + delta) = TV_W(mu*) + grad_W . delta + (2d/ell) Q_W(delta),     (*)
    grad_W = 2 (2d/ell) A_W mu*,    Q_W(delta) = delta^T A_W delta.

A pruning bound `B_W(mu*) >= 0` is **valid** if

    TV_W(mu*) - c_target > B_W(mu*)  =>  TV_W(mu) > c_target  for all mu in Cell,

i.e., `B_W(mu*)` upper-bounds the worst-case decrement
`max_{delta} [ -grad_W.delta - (2d/ell) Q_W(delta) ]` over `delta` in Cell.

The current v2 bound (`run_cascade_coarse_v2.py`) uses the triangle inequality:

    B_v2 = cell_var + quad_corr,
    cell_var = (1/(2S)) * sum_{k=0}^{d/2-1} ( g_(d-1-k) - g_(k) ),                  (1)
    quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4 S^2),                       (2)

where `g_(0) <= ... <= g_(d-1)` is the sorted gradient,
`N_W = sum_{k in window} n_k`, `n_k = #{(i,j) : 0 <= i,j < d, i+j = k}`,
`M_W = #{k in window : k even and k/2 < d}`, `cross_W = N_W - M_W`.

---

## 1. Rejected proposals (filtered out)

After review, the following candidates are either unsound or no tighter than v2 and
were dropped. They are listed once for the record.

| Proposal | Why dropped |
|----------|-------------|
| MATLAB Formula B `1/S^2 + 2W/S` | UNSOUND for narrow windows (counterexample at d=4, S=10, c=(2,1,1,6), window (ell=2, s=6); per-window error 0.44 vs correction 0.13 — a 3.4x deficit, predicted by the missing 2d/ell factor) |
| Per-window Formula A `(2d/ell)(1/S^2 + 2W/S)` | Sound, but uses only aggregated mass `W` and ignores gradient shape. Strictly looser than v2 by 2-5x on representative windows |
| Sum-zero LP refinement of `cell_var` | Sound, but `h * sum |g_i - median(g)|` collapses to identity (1) for even `d`. Even `d` is the only case the cascade hits (d_child = 2 d_parent). Zero improvement |
| Eigenvalue bound on quad_corr `(2d/ell)(-lambda_min^{(0)}) d h^2` | Sound, but a Rayleigh relaxation. Ties v2 when the most-negative eigenvector of `A_W` restricted to the sum-zero subspace happens to be a `+/-h` box vertex; otherwise looser. No worked window in the cascade range improves on v2 |
| Mass-space `|B|/S^2` refinement | Sound, but `|B_W| >= 1` always, so it is strictly looser than Formula A — and Formula A is already looser than v2 |

---

## 2. The only surviving proposal: joint QP

**Bound.**

    B_QP(mu*) := max{ -grad_W . delta - (2d/ell) delta^T A_W delta
                       : -h <= delta_i <= h, sum delta_i = 0 }.                     (3)

This is the *exact* worst-case decrement over Cell and dominates every other valid
per-window bound:

    B_QP <= cell_var + quad_corr,

with equality only when the linear and quadratic maximizers coincide (rare; the
common case is anti-correlation, where the triangle inequality strictly overestimates).

**Soundness.** Direct from the exact Taylor identity (*): if (3) is `< TV_W(mu*) - c_target`,
then `TV_W(mu) > c_target` for every `mu` in Cell.

**No tighter per-window bound exists.** (3) is the worst-case decrement *by definition*;
any sound certificate must upper-bound it.

**Computational cost.** `A_W` is generally indefinite (e.g., d=2, ell=2, s=1 gives
eigenvalues +/-1), so (3) is an indefinite QP. The maximum of a quadratic over a
polytope is attained at a **vertex**, so for `d <= 16` solve (3) by complete vertex
enumeration of the cell `{|delta_i| <= h, sum delta_i = 0}`:

- Each vertex fixes `d-1` coordinates at `+/-h` and the d-th by sum-zero (clip to [-h, h]).
- d=8: 2^7 = 128 vertices, ~1 us per binding window.
- d=16: 2^15 = 32768 vertices, ~50 us per binding window.

For `d >= 32`, vertex enumeration is infeasible (2^31+ vertices). At those scales the
cascade falls back to v2's `cell_var + quad_corr` triangle bound, which is the
tightest closed-form expression that survives this filtering.

**Concrete tightness gain (d=4, S=10, c=(2,1,1,6), window ell=2 s=3):**

`A_W` is the anti-diagonal of pairs (0,3), (1,2), (2,1), (3,0); eigenvalues `(-1,-1,1,1)`.
`TV(mu*) = 1.04`, `grad_W = (4.8, 0.8, 0.8, 1.6)`.

| Bound      | Value | Notes |
|-----------|-------|-------|
| v2 triangle | 0.28 | cell_var=0.24, quad_corr=0.04 |
| **Joint QP (3)** | **0.20** | delta = (-0.05, -0.045, +0.05, +0.045) |

**Improvement: 28.6 %.** The gain comes from anti-correlation: the linear-maximizing
delta (+h on high-`g` bin, -h on low) and the quadratic-maximizing delta (largest
swap across the off-diagonal `A_W` block) work against each other. The triangle
inequality is blind to this. Empirically (see [tests/box_cert_tightness.py](../tests/box_cert_tightness.py)),
the v2 / QP ratio reaches 1.38x on symmetric compositions over middle-width windows
and is typically 1.0-1.05x elsewhere.

In `S`-budget terms, certifying the worked example with QP needs `S >= ~36` instead
of `S >= ~50`. Since L0 enumeration scales like `S^{d-1}` after canonical reduction,
this is a substantial pruning gain at small d.

---

## 3. Strict mathematical statement of the tightest valid bound

For any composition `c` with grid point `mu*_i = c_i / S`, and any window `W = (ell, s)`,
`c` is **box-certified** at `c_target` if and only if

    TV_W(mu*) - c_target  >  max{ -grad_W . delta - (2d/ell) delta^T A_W delta
                                : delta in [-h, h]^d, 1^T delta = 0 }                (4)

with `grad_W = (4d/ell) A_W mu*`, `(A_W)_{i,j} = 1{s <= i+j <= s+ell-2}`, `h = 1/(2S)`.

(4) is equivalent to `TV_W(mu) > c_target` for every continuous `mu` in the cell,
by the exact Taylor identity (*) and the cell definition. **No tighter sound bound
exists at the per-window level.**

(A cross-window min-max could in principle beat per-window certification, since one
delta may defeat one window while boosting another. This is a much harder min-max QP,
not natural to the cascade's window-by-window pruning, and is not pursued here.)

---

## 4. Recommendation

**For `d_child <= 16` (the cascade's small-d levels):** replace v2's
`cell_var + quad_corr` with vertex-enumeration QP (3) inside `_fused_coarse` of
`run_cascade_coarse_v2.py`. Estimated overhead: ~10% wall-time. Estimated effective
`S`-budget reduction at the cliff windows: ~30%.

**For `d_child >= 32`:** keep v2 unchanged. Vertex enumeration is infeasible and no
other proposal in this filter survives as both sound and strictly tighter.
