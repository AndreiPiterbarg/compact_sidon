"""Core mathematical primitives for the Lasserre SDP hierarchy.

Provides monomial enumeration, vectorized hash-based indexing, window
matrix construction, and moment collection.  No solver logic — only
reusable infrastructure imported by all other modules.
"""
import numpy as np


# =====================================================================
# Mersenne prime modular hashing (overflow-safe for d >= 28)
# =====================================================================

MERSENNE_61 = (1 << 61) - 1


def _hash_add(a, b, prime=None):
    """Add two hash values with optional modular reduction."""
    result = a + b
    if prime is not None:
        result = result % prime
    return result


# =====================================================================
# Known val(d) values for gap-closure reporting
# =====================================================================

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}


# =====================================================================
# Monomial enumeration
# =====================================================================

def enum_monomials(d, max_deg):
    """All multi-indices alpha in N^d with |alpha| <= max_deg."""
    result = []
    def gen(pos, remaining, current):
        if pos == d:
            result.append(tuple(current))
            return
        for v in range(remaining + 1):
            current.append(v)
            gen(pos + 1, remaining - v, current)
            current.pop()
    gen(0, max_deg, [])
    return result


def _add_mi(a, b, d):
    return tuple(a[i] + b[i] for i in range(d))


def _unit(d, i):
    return tuple(1 if k == i else 0 for k in range(d))


# =====================================================================
# Vectorized monomial hashing
# =====================================================================

def _make_hash_bases(d, max_comp):
    """Hash bases for monomial indexing.  Linear in components.

    For small d: exact mixed-radix (max_comp+1)^k — perfectly injective.
    Returns (bases, None).

    For large d: random coefficients mod Mersenne prime 2^61-1.
    Returns (bases, MERSENNE_61).

    The hash linearity h(a+b) = (h(a)+h(b)) % p is preserved.
    """
    base = max_comp + 1
    if base ** (d - 1) < 2**62:
        return np.array([base**k for k in range(d)], dtype=np.int64), None
    rng = np.random.RandomState(seed=d * 1000 + max_comp)
    bases = rng.randint(1, MERSENNE_61, size=d, dtype=np.int64)
    return bases, MERSENNE_61


def _hash_monos(arr, bases, prime=None):
    """Hash array of monomials. arr shape (..., d) -> (...) int64.

    When prime is not None, uses split-base technique to avoid int64
    overflow: bases are split into hi/lo halves, dot products computed
    separately (each fits int64), then combined mod prime using the
    Mersenne identity 2^61 ≡ 1 (mod 2^61-1).
    """
    if prime is None:
        return np.tensordot(arr.astype(np.int64), bases, axes=([-1], [0]))
    # Split bases: bases = hi * 2^30 + lo
    lo = bases & ((1 << 30) - 1)
    hi = bases >> 30
    arr64 = arr.astype(np.int64)
    # dot_lo: max per term = max_comp * 2^30 ~ 4*2^30 = 2^32
    #         sum of d=128 terms: 128*2^32 = 2^39. Fits int64.
    dot_lo = np.tensordot(arr64, lo, axes=([-1], [0]))
    # dot_hi: max per term = max_comp * 2^31 ~ 4*2^31 = 2^33
    #         sum of d=128 terms: 128*2^33 = 2^40. Fits int64.
    dot_hi = np.tensordot(arr64, hi, axes=([-1], [0]))
    # Combine: result = (dot_hi * 2^30 + dot_lo) mod p
    # Using Mersenne identity: for p = 2^61 - 1, 2^61 ≡ 1 (mod p)
    # So (x * 2^30) mod p: split x into x_hi (bits 31+) and x_lo (bits 0-30)
    #   x * 2^30 = x_hi * 2^61 * 2^(k-31) + x_lo * 2^30
    #            ≡ x_hi * 2^(k-31) + x_lo * 2^30  (mod p)
    # For simplicity: (x * 2^30) mod p = ((x >> 31) + ((x & 0x7FFFFFFF) << 30)) mod p
    dot_hi_mod = dot_hi % prime
    x_hi = dot_hi_mod >> 31
    x_lo = dot_hi_mod & np.int64(0x7FFFFFFF)
    shifted = x_hi + (x_lo << np.int64(30))
    return (shifted + dot_lo) % prime


def _build_hash_table(mono_list, bases, prime=None):
    """Sorted hashes + original indices for vectorized searchsorted lookup."""
    mono_arr = np.array(mono_list, dtype=np.int64)
    hashes = _hash_monos(mono_arr, bases, prime)
    order = np.argsort(hashes)
    return hashes[order], order


def _hash_lookup(query_hashes, sorted_hashes, sort_order):
    """Vectorized lookup: query_hashes -> moment indices (or -1)."""
    flat = query_hashes.ravel()
    pos = np.searchsorted(sorted_hashes, flat)
    pos = np.clip(pos, 0, len(sorted_hashes) - 1)
    found = sorted_hashes[pos] == flat
    result = np.where(found, sort_order[pos], -1)
    return result.reshape(query_hashes.shape)


# =====================================================================
# Window matrices
# =====================================================================

def build_window_matrices(d):
    """Build all window matrices for d bins.

    Returns (windows, M_mats) where windows is a list of (ell, s_lo)
    tuples and M_mats[w] is the d x d matrix with M_W[i,j] = (2d/ell)
    if s_lo <= i+j <= s_lo+ell-2, else 0.
    """
    conv_len = 2 * d - 1
    windows = [(ell, s) for ell in range(2, 2 * d + 1)
               for s in range(conv_len - ell + 2)]
    ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    sums = ii + jj
    M_mats = []
    for ell, s_lo in windows:
        mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)
        M_mats.append((2.0 * d / ell) * mask.astype(np.float64))
    return windows, M_mats


# =====================================================================
# Collect all moment indices
# =====================================================================

def collect_moments(d, order, basis, loc_basis, consist_mono):
    r"""Collect all moment multi-indices needed for the Lasserre SDP.

    Returns the set {α ∈ ℕ^d : |α| ≤ 2·order} as a sorted list + index map.

    Proof of completeness: every sub-structure of the degree-2k Lasserre
    relaxation generates monomials of degree ≤ 2k:
      - Moment matrix M_k: basis[a]+basis[b], |·| ≤ k+k = 2k  ✓
      - Localizing M_{k-1}(μ_i·y): loc[a]+loc[b]+e_i, |·| ≤ 2(k-1)+1 = 2k-1  ✓
      - Window localizing: loc[a]+loc[b]+e_i+e_j, |·| ≤ 2(k-1)+2 = 2k  ✓
      - Consistency α+e_i: |α| ≤ 2k-1 ⟹ |α+e_i| ≤ 2k  ✓
    All are subsets of enum_monomials(d, 2k). QED.
    """
    all_monos = np.array(enum_monomials(d, 2 * order), dtype=np.int64)
    sort_idx = np.lexsort(all_monos.T[::-1])
    all_monos = all_monos[sort_idx]

    mono_list = [tuple(row) for row in all_monos]
    idx = {m: i for i, m in enumerate(mono_list)}
    return mono_list, idx
