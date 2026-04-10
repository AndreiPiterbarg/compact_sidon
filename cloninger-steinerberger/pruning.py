"""Pruning utilities for branch-and-prune algorithm.

Correction terms, asymmetry thresholds, composition counting,
and symmetry masks.
"""
import numpy as np
import numba
from math import comb


def correction(m, n_half=None, ell_min=2):
    """Discretization error bound (C&S Lemma 3).

    C&S Lemma 3 bounds ||g*g - f*f||_∞ ≤ 2/m + 1/m² (pointwise on the
    autoconvolution, NOT per-window on the test value).  Valid because
    the code uses the paper's fine grid B_{n,m} where heights are
    multiples of 1/m (integer compositions sum to 4nm), giving
    ||ε||_∞ ≤ 1/m.

    The n_half and ell_min parameters are accepted for API compatibility
    but are no longer used in the correction formula.
    """
    return 2.0 / m + 1.0 / (m * m)


def asymmetry_threshold(c_target):
    """Minimum left-mass fraction for the asymmetry argument to give >= c_target.

    If left_mass_frac >= threshold (or right_mass_frac >= threshold),
    then ||f*f||_inf >= 2 * threshold^2 >= c_target.
    """
    return np.sqrt(c_target / 2.0)


def count_compositions(d, S):
    """Number of non-negative integer vectors of length d summing to S.
    Equals C(S + d - 1, d - 1).
    """
    return comb(S + d - 1, d - 1)


def asymmetry_prune_mask(batch_int, n_half, m, c_target):
    """Return boolean mask: True for configs NOT covered by asymmetry argument.

    Returns True for configs that NEED test-value checking.
    n_half can be a float (e.g. 1.5 for d=3 starting dimension).
    """
    d = batch_int.shape[1]
    total = float(2 * d * m)  # Fine grid: integer coords sum to 2*d*m
    threshold = asymmetry_threshold(c_target)

    # No discretization margin needed: left_frac = sum(c_i for left bins) / m
    # is exact for piecewise-constant functions on the discrete grid, and is
    # preserved exactly under refinement (child bins sum to parent bins).
    # See docs/verification_part1_framework.md, Verification 8 for full proof.

    left_bins = d // 2  # floor(d/2) bins as "left half"
    left = batch_int[:, :left_bins].sum(axis=1).astype(np.float64)
    left_frac = left / total

    # Asymmetry covers: left_frac >= threshold or left_frac <= 1 - threshold
    # Need checking: everything in between
    needs_check = (left_frac > 1 - threshold) & (left_frac < threshold)
    return needs_check


@numba.njit(parallel=True, cache=True)
def _canonical_mask(batch_int):
    """Return bool mask: True for b where b <= rev(b) lexicographically."""
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    result = np.ones(B, dtype=numba.boolean)
    for b in numba.prange(B):
        for i in range(d // 2):
            j = d - 1 - i
            if batch_int[b, i] < batch_int[b, j]:
                break  # canonical
            elif batch_int[b, i] > batch_int[b, j]:
                result[b] = False
                break
    return result
