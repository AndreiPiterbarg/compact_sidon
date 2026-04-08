# Vacuity Analysis: Upper Bound vs. Provable Lower Bound

## The Fundamental Constraint

The pruning rule is (fine grid, compositions summing to $S = 4nm$):

```
prune if TV_discrete > c_target + (3 + W_int/(2n)) / m^2
```

where `W_int` = integer mass in the window's bins (0 <= W_int <= 4nm), and `W_g = W_int/(4nm)` is the physical mass fraction.

For pruning to be **non-vacuous**, the threshold must be **below C_upper** (the best known upper bound on C_{1a}). Otherwise, even the near-optimal function (TV ~ C_upper) passes through unpruned, and the cascade can never reach 0 survivors.

```
Non-vacuous requires:  c_target + (3 + W_int/(2n)) / m^2  <  C_upper
Solving for c_target:  c_target  <  C_upper - (3 + W_int/(2n)) / m^2
```

Two cases:
- **Best window (W_int=0):** `c_target_max = C_upper - 3/m^2`
- **All windows (W_int=4nm, i.e., W_g=1):** `c_target_max = C_upper - 2/m - 3/m^2`

The correction term `2/m + 3/m^2` (worst case over all windows) comes from Lemma 3 (discretization error bound). It must **fit in the gap** between c_target and C_upper. If there's no room, pruning cannot reject anything.

## Three Nested Constraints

```
|<-- provable -->|<-- non-vacuous but diverges -->|<-- vacuous -->|<-- impossible -->|
1.28            ~1.35                          ~1.40-1.50        C_1a            C_upper
     Wall 3              Wall 2                       Wall 1
  (convergence)        (vacuity)                 (mathematical)
```

1. **Wall 1 -- Mathematical impossibility:** `c_target > C_1a` (the true constant). No algorithm can prove this. There exists a function with ||f\*f||_inf < c_target.

2. **Wall 2 -- Vacuity:** `c_target > C_upper - 2/m - 3/m^2`. The pruning threshold exceeds C_upper for windows with W_g near 1. Near-optimal configs escape pruning. Increasing m pushes this wall back, but more m = more compositions = more survivors.

3. **Wall 3 -- Convergence (empirical):** Even when non-vacuous, the cascade may diverge (expansion factor > 1 at every level). From benchmarks: convergence only observed at c_target <= 1.35 with n_half=3, m=15.

## Effect of a New Upper Bound

- **Past proofs are NOT invalidated.** A completed proof (0 survivors) is a fact about an exhaustive enumeration. The correction term comes from Lemma 3 (a theorem about approximation error), not from C_upper.
- **Future proofs are constrained.** A tighter C_upper shrinks the gap that the correction must fit in, requiring larger m, which produces more survivors and makes convergence harder.
- **The only thing that matters is the gap** `C_upper - c_target`. The minimum m depends solely on this gap (approximately m > 2/gap).

## Table 1: Minimum m for Non-Vacuous Pruning (All Windows)

Formula: `2/m + 3/m^2 < C_upper - c_target`

`'.' = impossible (c_target >= C_upper)`

| C_upper | 1.28 | 1.29 | 1.30 | 1.31 | 1.32 | 1.33 | 1.34 | 1.35 | 1.36 | 1.37 | 1.38 | 1.39 | 1.40 | 1.41 | 1.42 | 1.43 | 1.44 | 1.45 | 1.46 | 1.47 | 1.48 | 1.49 | 1.50 |
|---------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 1.5029 | 11 | 11 | 12 | 12 | 13 | 13 | 14 | 15 | 16 | 17 | 18 | 20 | 21 | 23 | 26 | 29 | 34 | 40 | 49 | 63 | 89 | 157 | 692 |
| 1.5000 | 11 | 11 | 12 | 12 | 13 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . |
| 1.4900 | 11 | 12 | 12 | 13 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . |
| 1.4800 | 12 | 12 | 13 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . |
| 1.4700 | 12 | 13 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . |
| 1.4600 | 13 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . |
| 1.4500 | 14 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . |
| 1.4400 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . |
| 1.4300 | 15 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . |
| 1.4200 | 16 | 17 | 19 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . |
| 1.4100 | 17 | 19 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . |
| 1.4000 | 19 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . |
| 1.3900 | 20 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . | . |
| 1.3800 | 22 | 24 | 27 | 31 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| 1.3700 | 24 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| 1.3600 | 27 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| 1.3500 | 30 | 35 | 42 | 52 | 69 | 102 | 202 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |

### How to read Table 1

Pick a row (hypothetical C_upper) and a column (desired c_target). The cell gives the minimum m for the pruning threshold to be below C_upper at all windows.

- **m <= 20:** Feasible with current code (cascade may or may not converge)
- **m = 20-50:** Hard zone -- maybe feasible with GPU, but more survivors
- **m > 50:** Likely impractical -- too many compositions for cascade convergence

**Example:** If C_upper drops to 1.42 and you want c_target = 1.35, you need m >= 31. But m=31 produces far more survivors than m=20, and the cascade already diverges at m=20.

## Table 2: Best Non-Vacuous c_target for Each (C_upper, m) Pair

Formula: `c_target_max = C_upper - 2/m - 3/m^2`

`* = below current best known lower bound (1.2802)`

| C_upper | m=10 | m=12 | m=15 | m=18 | m=20 | m=22 | m=25 | m=30 | m=35 | m=40 | m=50 | m=60 | m=75 | m=100 | m=150 | m=200 |
|---------|------|------|------|------|------|------|------|------|------|------|------|------|------|-------|-------|-------|
| 1.5029 | 1.273* | 1.315 | 1.356 | 1.383 | 1.395 | 1.406 | 1.418 | 1.433 | 1.443 | 1.451 | 1.462 | 1.469 | 1.476 | 1.483 | 1.489 | 1.493 |
| 1.5000 | 1.270* | 1.312 | 1.353 | 1.380 | 1.392 | 1.403 | 1.415 | 1.430 | 1.440 | 1.448 | 1.459 | 1.466 | 1.473 | 1.480 | 1.487 | 1.490 |
| 1.4900 | 1.260* | 1.302 | 1.343 | 1.370 | 1.382 | 1.393 | 1.405 | 1.420 | 1.430 | 1.438 | 1.449 | 1.456 | 1.463 | 1.470 | 1.477 | 1.480 |
| 1.4800 | 1.250* | 1.292 | 1.333 | 1.360 | 1.372 | 1.383 | 1.395 | 1.410 | 1.420 | 1.428 | 1.439 | 1.446 | 1.453 | 1.460 | 1.467 | 1.470 |
| 1.4700 | 1.240* | 1.282 | 1.323 | 1.350 | 1.362 | 1.373 | 1.385 | 1.400 | 1.410 | 1.418 | 1.429 | 1.436 | 1.443 | 1.450 | 1.457 | 1.460 |
| 1.4600 | 1.230* | 1.272* | 1.313 | 1.340 | 1.352 | 1.363 | 1.375 | 1.390 | 1.400 | 1.408 | 1.419 | 1.426 | 1.433 | 1.440 | 1.447 | 1.450 |
| 1.4500 | 1.220* | 1.262* | 1.303 | 1.330 | 1.342 | 1.353 | 1.365 | 1.380 | 1.390 | 1.398 | 1.409 | 1.416 | 1.423 | 1.430 | 1.437 | 1.440 |
| 1.4400 | 1.210* | 1.252* | 1.293 | 1.320 | 1.332 | 1.343 | 1.355 | 1.370 | 1.380 | 1.388 | 1.399 | 1.406 | 1.413 | 1.420 | 1.427 | 1.430 |
| 1.4300 | 1.200* | 1.242* | 1.283 | 1.310 | 1.322 | 1.333 | 1.345 | 1.360 | 1.370 | 1.378 | 1.389 | 1.396 | 1.403 | 1.410 | 1.417 | 1.420 |
| 1.4200 | <1.2 | 1.232* | 1.273* | 1.300 | 1.312 | 1.323 | 1.335 | 1.350 | 1.360 | 1.368 | 1.379 | 1.386 | 1.393 | 1.400 | 1.407 | 1.410 |
| 1.4100 | <1.2 | 1.222* | 1.263* | 1.290 | 1.302 | 1.313 | 1.325 | 1.340 | 1.350 | 1.358 | 1.369 | 1.376 | 1.383 | 1.390 | 1.397 | 1.400 |
| 1.4000 | <1.2 | 1.212* | 1.253* | 1.280* | 1.292 | 1.303 | 1.315 | 1.330 | 1.340 | 1.348 | 1.359 | 1.366 | 1.373 | 1.380 | 1.387 | 1.390 |
| 1.3900 | <1.2 | 1.202* | 1.243* | 1.270* | 1.282 | 1.293 | 1.305 | 1.320 | 1.330 | 1.338 | 1.349 | 1.356 | 1.363 | 1.370 | 1.377 | 1.380 |
| 1.3800 | <1.2 | <1.2 | 1.233* | 1.260* | 1.272* | 1.283 | 1.295 | 1.310 | 1.320 | 1.328 | 1.339 | 1.346 | 1.353 | 1.360 | 1.367 | 1.370 |
| 1.3700 | <1.2 | <1.2 | 1.223* | 1.250* | 1.262* | 1.273* | 1.285 | 1.300 | 1.310 | 1.318 | 1.329 | 1.336 | 1.343 | 1.350 | 1.357 | 1.360 |
| 1.3600 | <1.2 | <1.2 | 1.213* | 1.240* | 1.252* | 1.263* | 1.275* | 1.290 | 1.300 | 1.308 | 1.319 | 1.326 | 1.333 | 1.340 | 1.347 | 1.350 |
| 1.3500 | <1.2 | <1.2 | 1.203* | 1.230* | 1.242* | 1.253* | 1.265* | 1.280* | 1.290 | 1.298 | 1.309 | 1.316 | 1.323 | 1.330 | 1.337 | 1.340 |

### How to read Table 2

Pick a row (hypothetical C_upper) and a column (m value). The cell gives the highest c_target you can attempt without vacuity.

**Example readings:**
- C_upper=1.45, m=20: best non-vacuous c_target = 1.342
- C_upper=1.40, m=30: best non-vacuous c_target = 1.330
- C_upper=1.35, m=50: best non-vacuous c_target = 1.309

## Key Observations

1. **Every cell depends only on the gap** `C_upper - c_target`. The same minimum m appears along diagonals wherever the gap is equal. A tighter upper bound shifts the entire table, demanding larger m for the same c_target.

2. **Larger m is a double-edged sword.** It shrinks the correction (pushing back the vacuity wall), but produces exponentially more compositions and survivors. Empirically, m=20 has the lowest expansion factors. m=15 with n_half=3 is the only config that achieves cascade convergence.

3. **If C_upper drops to 1.40:** With m=20, the best non-vacuous c_target is only 1.292 -- you can't even beat the current record of 1.2802 without going to m >= 22.

4. **If C_upper drops to 1.35:** You need m >= 35 just to attempt c_target = 1.29, and m=35 almost certainly diverges.

5. **Non-vacuity is necessary but not sufficient.** Even when the threshold is below C_upper, the cascade may diverge (expansion > 1 at every level). The convergence wall is strictly below the vacuity wall.

## Empirical Convergence Data (from benchmark sweeps)

For reference, the only configurations that achieved cascade convergence (0 survivors):

| c_target | Config | Converged at | Min m for non-vacuity |
|----------|--------|-------------|-----------------------|
| 1.33 | n_half=3, m=15 | L6 (d=384) | 13 |
| 1.35 | n_half=3, m=15 | L6 (d=384) | 15 |

All other tested configurations (n_half=2 or m=20) diverge at every c_target.

## Reproducing These Tables

```bash
python tests/critical_vacuity_table.py
python tests/upper_bound_vs_provable_lower.py
```
