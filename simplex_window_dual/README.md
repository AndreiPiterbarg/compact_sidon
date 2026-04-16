# Simplex Window Dual

This directory contains a new lower-bound proof route for the discrete window problem

\[
\mathrm{val}(d) = \min_{\mu \in \Delta_d} \max_W \mu^\top M_W \mu.
\]

The core idea is dual: instead of searching over pseudo-moments or enumerating
grid children, search for a certificate proving that every simplex point forces
at least one window value above a target \(\alpha\).

## Implemented certificate families

### 1. Constant window mixtures

This is the baseline family

\[
\sum_W \lambda_W (\mu^\top M_W \mu - \alpha),
\qquad \lambda_W \ge 0,\ \sum_W \lambda_W = 1.
\]

It is easy to evaluate exactly, but empirically it is too weak.

### 2. Degree-1 nonnegative window multipliers

This is the main implemented family. For fixed \(\alpha\), search for

\[
\sum_W \Lambda_W(x)\,(x^\top M_W x - \alpha)
  = N(x) + (1-\mathbf{1}^\top x)\,H(x),
\]

where

- each window multiplier \(\Lambda_W(x)\) has nonnegative coefficients and degree at most 1,
- \(N(x)\) has nonnegative coefficients and degree at most 3,
- \(H(x)\) is a free polynomial of degree at most 2.

On the simplex \(\{x \ge 0,\ \mathbf{1}^\top x = 1\}\), the right-hand side
reduces to \(N(x)\), which is nonnegative coefficientwise and therefore
nonnegative for all \(x \ge 0\). If such an identity exists, then every simplex
point satisfies \(\max_W x^\top M_W x \ge \alpha\).

This feasibility problem is a linear program at fixed \(\alpha\).

## Status

This directory is intended to be a real development base for the dual proof
route. The immediate goal is to see whether degree-1 certificates can exceed
the current record \(1.2802\) at some moderate dimension, before moving to
higher-degree multipliers or cloud runs.
