# Prompt 2: Discretization Correction Term (Claim 1.2)

**Claim 1.2 only.** The `.lean` file is self-contained. Optionally also attach `output (22).lean` for partial progress on the discretization error proof.

---

## Problem Context

We are proving $c \geq 1.4$ where $c = \inf_{f \geq 0,\, \text{supp}(f) \subseteq (-1/4, 1/4)} \|f*f\|_{L^\infty} / (\int f)^2$.

## Claim 1.2: Correction Term $(4n/\ell)(2/m + 1/m^2)$

**Theorem (Lemma 3 of CS14).** For any nonneg $f$ on $(-1/4, 1/4)$ with $\int f = 1$ and canonical discretization $\hat{c}$, the per-window correction is:

$$R(f) \geq b_{n,m}(\hat{c}) - \frac{4n}{\ell}\left(\frac{2}{m} + \frac{1}{m^2}\right)$$

where $\ell$ is the window length. Globally (since $\ell \geq 2$, so $4n/\ell \leq 2n$): $R(f) \geq b_{n,m}(\hat{c}) - 2n(2/m + 1/m^2)$.

### Proof

Let $\mu_i = \int_{\text{bin}_i} f$, $w_i = c_i/m$, $\delta_i = w_i - \mu_i$.

1. $|\delta_i| \leq 1/m$ (by `discretization_error_bound`, stated as axiom in attached file).
2. $\sum \delta_i = 0$ (since $\sum w_i = \sum \mu_i = 1$).
3. The convolution error decomposes as: $\sum_{i+j=k}(w_iw_j - \mu_i\mu_j) = \sum_{i+j=k}(\delta_i\mu_j + \mu_i\delta_j + \delta_i\delta_j)$.
4. Window-averaging and bounding: use `sum_mul_bound_succ` (Abel summation, proved in file) with $A = 1/m$, yielding the raw $2/m + 1/m^2$ bound per convolution index. After window normalization by $1/(4n\ell)$, the effective per-window correction is $(4n/\ell)(2/m + 1/m^2)$.

### What's proved vs what needs proving

**Proved helpers** in attached file:
- `sum_bin_masses_eq_one` — $\sum \mu_i = 1$
- `bin_masses_nonneg` — $\mu_i \geq 0$
- `sum_mul_bound_succ` — Abel summation inequality
- `discretization_error_bound` — $|c_i/m - \mu_i| \leq 1/m$ (axiom)

**Only `correction_term` has sorry.** This is the hardest single claim in the project. The correction is $(4n/\ell)(2/m + 1/m^2)$ per window, or $2n(2/m + 1/m^2)$ globally.
