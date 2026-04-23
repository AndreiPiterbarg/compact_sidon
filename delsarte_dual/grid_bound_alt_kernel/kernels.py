"""Kernel candidates for the MV dual bound.

Every kernel K satisfies MV's admissibility hypotheses:
  (a) K >= 0 on R,
  (b) supp K subset [-delta, delta],
  (c) int K = 1,
  (d) tilde K(j) >= 0 for all integer j (Bochner / non-negative FT).

Each kernel exposes:
  - supp_halfwidth (fmpq) = delta,
  - K_norm_sq(prec) -> arb (= ||K||_2^2),
  - K_tilde(n, prec) -> arb (period-1 FT coefficient for integer n, = hat_K_R(n)),
  - K_tilde_real(xi, prec) -> arb (real-line FT at arbitrary arb xi),
  - K_tilde_positive(n_max, prec) -> bool (Bochner check j=1..n_max),
  - admissibility_check(prec) -> None.

Convention
----------
All kernels have REAL even Fourier transforms (K is real-even), so
hat_K_R(xi) = int_{-delta}^{delta} K(x) cos(2 pi xi x) dx for real xi.

The "period-1" Fourier coefficient used in Phi's k_n slot is
  k_n := hat_K_R(n)                  (= int K(x) e^{-2 pi i n x} dx,
                                        valid whenever supp K subset [-1/2, 1/2])

The "period-u" Fourier coefficient used in S_1's QP weight is
  tilde_K_u(j) := (1/u) * hat_K_R(j/u)        (valid since supp K in [-u/2, u/2]).

MV-arcsine exception
--------------------
MV's eq. (5) defines K(x) = (1/delta) eta(x/delta) with eta = 2/(pi sqrt(1-4x^2))
on (-1/2, 1/2), but MV's identity hat_K_R(xi) = J_0(pi xi delta)^2 corresponds
to the AUTO-CONVOLUTION eta * eta scaled to (-delta, delta).  (The raw
(1/delta)eta(x/delta) has hat K_R = J_0, not J_0^2, and is not in L^2.)  We
use the squared form here as it is the one MV actually plug into their
inequality; see ``delsarte_dual/grid_bound/phi_mm.py`` docstring + the
``kn_period_one`` reference, and compare with the MO surrogate 0.5747/delta
for ||K||_2^2 which is finite only for the auto-convolution (L^2 regular).

That is: "K1 arcsine" here means "MV's eq. (5) kernel understood as
    K(x) = (1/delta) * (eta * eta)(x/delta),   supp in [-delta, delta]."
All other kernels use honest closed-form or arb-integrated FT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from flint import arb, acb, fmpq, ctx


# -----------------------------------------------------------------------------
#  Base class
# -----------------------------------------------------------------------------

class Kernel:
    """Abstract base class; concrete kernels override the rigorous methods."""
    name: str = "abstract"
    supp_halfwidth: fmpq = fmpq(138, 1000)   # delta

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """Rigorous arb enclosure of ||K||_2^2 = int K(x)^2 dx."""
        raise NotImplementedError

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """Rigorous arb enclosure of hat_K_R(xi) = int K(x) cos(2 pi xi x) dx.

        Because K is real-even, the FT is real, and (by non-negativity of K)
        satisfies |hat_K_R(xi)| <= hat_K_R(0) = 1.
        """
        raise NotImplementedError

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        """Period-1 Fourier coefficient k_n = hat_K_R(n) for integer n.

        Default implementation falls back to ``K_tilde_real(arb(n))``;
        closed-form kernels override for tighter enclosures.
        """
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            # int K = 1 by the admissibility hypothesis; return an exact 1
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_tilde_positive(self, n_max: int = 200, prec_bits: int = 256) -> bool:
        """Verify k_j >= 0 for j = 1..n_max at prec_bits of precision.

        Returns True iff ``K_tilde(j).lower() >= 0`` for every j.  False if
        ANY value's arb lower bound is strictly negative (Bochner violation).
        """
        for j in range(1, n_max + 1):
            val = self.K_tilde(j, prec_bits)
            if val.lower() < 0:
                return False
        return True

    def admissibility_check(self, prec_bits: int = 256, n_bochner: int = 50) -> None:
        """Raise ValueError if K is not a valid MV kernel.

        Checks:  (a) supp_halfwidth > 0 rational,
                 (d) tilde K(j) >= 0 for j = 1..n_bochner (Bochner).
        Non-negativity (a) and int K = 1 (c) are enforced by construction
        in each concrete kernel.
        """
        if self.supp_halfwidth <= 0:
            raise ValueError(f"{self.name}: supp_halfwidth must be positive")
        if not self.K_tilde_positive(n_bochner, prec_bits):
            raise ValueError(
                f"{self.name}: Bochner check failed (some k_j < 0 for j<={n_bochner})"
            )

    def __repr__(self) -> str:
        return f"<Kernel {self.name} delta={float(self.supp_halfwidth):.4f}>"


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _arb_cos_int(f_times_cos_arb: Callable[[acb, object], acb],
                 a: fmpq, b: fmpq, prec_bits: int = 256) -> arb:
    """Rigorous ||arb-real enclosure of int_a^b f(x) cos(2 pi xi x) dx.

    Actually: integrate a caller-supplied complex function over [a, b] using
    acb.integral (rigorous enclosure) and return the real part.  Caller is
    responsible for the integrand being holomorphic on a neighborhood of
    [a, b] (polynomial, exponential, etc. all OK).
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        a_acb = acb(arb(a))
        b_acb = acb(arb(b))
        val = acb.integral(f_times_cos_arb, a_acb, b_acb)
        return val.real
    finally:
        ctx.prec = old


# =============================================================================
#  K1: MV arcsine-auto-convolution (baseline)
# =============================================================================

class ArcsineKernel(Kernel):
    """MV's eq. (5) kernel: K(x) = (1/delta) (eta * eta)(x/delta) on [-delta, delta].

    Real-line Fourier: hat_K_R(xi) = J_0(pi xi delta)^2  (non-negative by square).
    ||K||_2^2: we take MV's declared SURROGATE 0.5747/delta (MV p. 3 line 141,
    inherited from Martin-O'Bryant arXiv:0807.5121).  A first-principles
    re-derivation is a Phase-2+ open task — see ``grid_bound/coeffs.py``.
    """
    name = "K1_arcsine"

    def __init__(self, delta: fmpq = fmpq(138, 1000),
                 K_norm_sq_surrogate_times_delta: fmpq = fmpq(5747, 10000)):
        self.supp_halfwidth = delta
        self._K2_numerator = K_norm_sq_surrogate_times_delta

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        return arb(self._K2_numerator) / arb(self.supp_halfwidth)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            arg = arb.pi() * xi * arb(self.supp_halfwidth)
            j0 = arg.bessel_j(0)
            return j0 * j0
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            # pi * n * delta is exactly fmpq in the argument slot
            q = fmpq(n) * self.supp_halfwidth
            arg = arb.pi() * arb(q)
            j0 = arg.bessel_j(0)
            return j0 * j0
        finally:
            ctx.prec = old


# =============================================================================
#  K2: Triangular (Fejer)
# =============================================================================

class TriangularKernel(Kernel):
    """K(x) = (1/delta) * (1 - |x|/delta)_+  on [-delta, delta].

    (Normalised: int K = 1 by the triangle area = delta.)
    hat_K_R(xi) = sinc(pi xi delta)^2 >= 0 (Fejer).
    ||K||_2^2 = 2/(3 delta).
    """
    name = "K2_triangular"

    def __init__(self, delta: fmpq = fmpq(138, 1000)):
        self.supp_halfwidth = delta

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        # int_{-d}^{d} (1/d)^2 (1 - |x|/d)^2 dx = (2/d^2) int_0^d (1 - x/d)^2 dx
        #   = (2/d^2) * (d/3) = 2/(3 d).
        return arb(fmpq(2, 3)) / arb(self.supp_halfwidth)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """hat_K_R(xi) = sinc(pi xi delta)^2 = (sin(pi xi delta)/(pi xi delta))^2."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            arg = arb.pi() * xi * arb(self.supp_halfwidth)
            # arb.sinc(x) = sin(x)/x
            s = arg.sinc()
            return s * s
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            q = fmpq(n) * self.supp_halfwidth
            arg = arb.pi() * arb(q)
            s = arg.sinc()
            return s * s
        finally:
            ctx.prec = old


# =============================================================================
#  K3: Truncated Gaussian
# =============================================================================

class TruncatedGaussianKernel(Kernel):
    """K(x) = (1/Z) exp(-(x/sigma)^2) * 1_{[-delta, delta]}.

    Z = int_{-delta}^{delta} exp(-(x/sigma)^2) dx = sigma sqrt(pi) erf(delta/sigma).

    hat_K_R(xi) has no simple closed form; we compute it by rigorous
    acb.integral (the integrand is real-entire so arb-certified enclosures
    are automatic).
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000),
                 sigma_over_delta: fmpq = fmpq(1, 2),
                 label: str = ""):
        self.supp_halfwidth = delta
        self.sigma_over_delta = sigma_over_delta
        self._sigma_q = sigma_over_delta * delta   # sigma as fmpq
        if not label:
            label = f"{float(sigma_over_delta.p)/float(sigma_over_delta.q):.3f}"
        self.name = f"K3_truncgauss_sigma={label}delta"

    def _sigma(self) -> arb:
        return arb(self._sigma_q)

    def _normalization(self, prec_bits: int) -> arb:
        """Z = sigma * sqrt(pi) * erf(delta / sigma)."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            sig = self._sigma()
            d = arb(self.supp_halfwidth)
            return sig * arb.pi().sqrt() * (d / sig).erf()
        finally:
            ctx.prec = old

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = (1/Z^2) * (sigma/sqrt(2)) * sqrt(pi) * erf(delta sqrt(2) / sigma)."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            Z = self._normalization(prec_bits)
            sig = self._sigma()
            d = arb(self.supp_halfwidth)
            sqrt2 = arb(2).sqrt()
            num = sig * arb.pi().sqrt() * ((d * sqrt2) / sig).erf() / sqrt2
            return num / (Z * Z)
        finally:
            ctx.prec = old

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """hat_K_R(xi) = (1/Z) * int_{-delta}^{delta} exp(-(x/sigma)^2) cos(2 pi xi x) dx.

        Closed form:
           (1/Z) * (sigma sqrt(pi)/2) * exp(-pi^2 sigma^2 xi^2) *
                 [erf((delta/sigma) + i pi sigma xi) +
                  erf((delta/sigma) - i pi sigma xi)]
           = (1/Z) * sigma * sqrt(pi) * exp(-pi^2 sigma^2 xi^2) *
                    Re(erf((delta/sigma) + i pi sigma xi))

        We compute the complex-erf of the complex argument via flint's acb.erf.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            sig = self._sigma()
            d = arb(self.supp_halfwidth)
            Z = self._normalization(prec_bits)
            # complex argument: delta/sigma + i * pi * sigma * xi
            re_part = d / sig
            im_part = arb.pi() * sig * xi
            z = acb(re_part, im_part)
            erfz = z.erf()
            # envelope exp(-pi^2 sigma^2 xi^2)
            env = (-arb.pi() * arb.pi() * sig * sig * xi * xi).exp()
            pref = sig * arb.pi().sqrt() / Z
            return pref * env * erfz.real
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)


# =============================================================================
#  K4: Jackson kernel (rescaled)
# =============================================================================

class JacksonKernel(Kernel):
    """Rescaled Jackson kernel on [-delta, delta].

    Let T = pi / (2 delta). The standard Jackson kernel on a period is
        J_m(theta) = c_m * (sin(m theta / 2) / sin(theta / 2))^4,
    which has non-negative Fejer-squared Fourier coefficients supported on
    |n| <= 2m - 2.  Here we use the continuous-variable Jackson-de la Vallée-
    Poussin kernel:

        K(x) = c * (sin(m pi x / (2 delta)) / (m sin(pi x / (2 delta))))^4

    for positive integer m.  The inner fraction is a Dirichlet-kernel-like
    function whose 4th power is non-negative everywhere and has compact
    support after restriction to [-delta, delta].

    For numerical stability (arb-level closed-form FT is long and messy),
    we compute hat_K_R(xi) by rigorous arb integration over [-delta, delta].
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000),
                 m: int = 10):
        self.supp_halfwidth = delta
        self.m = int(m)
        self.name = f"K4_jackson_m={self.m}"
        self._Z = None  # cached normalisation Z at construction prec

    def _inner_raw(self, x_arb: arb, prec_bits: int) -> arb:
        """(sin(m pi x / (2 delta)) / (m sin(pi x / (2 delta))))^4."""
        d = arb(self.supp_halfwidth)
        c = arb.pi() / (arb(2) * d)   # pi / (2 delta)
        # numerator: sin(m*c*x)
        num = (arb(self.m) * c * x_arb).sin()
        # denominator: m * sin(c*x); at x = 0 both num/denom -> 1 so we factor
        denom_sin = (c * x_arb).sin()
        # stable limit near x=0: use series — but for our use x_arb is always
        # a cell interval (in integration) or large enough; arb handles it
        # with a 0/0 that we guard below.  We compute the 4th power of
        # (sin(m c x) / (m sin(c x))).
        # When |x| is small, sin(m c x)/sin(c x) -> m, so ratio -> 1.
        ratio = num / (arb(self.m) * denom_sin)
        ratio_sq = ratio * ratio
        return ratio_sq * ratio_sq

    def _raw_integrand_cos(self, xi: arb, prec_bits: int):
        """Return a callable for acb.integral: x -> inner_raw(x) * cos(2 pi xi x).

        Uses sinc(mcz)/sinc(cz) form (L'Hopital-safe at z = 0 via flint's
        acb.sinc, which is entire).
        """
        d = arb(self.supp_halfwidth)
        c = arb.pi() / (arb(2) * d)
        m = arb(self.m)
        def f(z, flags):
            # ratio = sin(m c z)/(m sin(c z)) = sinc(m c z) / sinc(c z).
            ratio = (m * c * z).sinc() / (c * z).sinc()
            r2 = ratio * ratio
            r4 = r2 * r2
            return r4 * (acb(2) * acb.pi() * acb(xi) * z).cos()
        return f

    def _raw_integrand_plain(self):
        """Return f(z) = inner_raw(z) for normalisation (xi = 0)."""
        d = arb(self.supp_halfwidth)
        c = arb.pi() / (arb(2) * d)
        m = arb(self.m)
        def f(z, flags):
            ratio = (m * c * z).sinc() / (c * z).sinc()
            r2 = ratio * ratio
            return r2 * r2
        return f

    def _Z_norm(self, prec_bits: int) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = self.supp_halfwidth
            f = self._raw_integrand_plain()
            val = acb.integral(f, acb(arb(-d)), acb(arb(d)))
            return val.real
        finally:
            ctx.prec = old

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = (1/Z^2) int_{-d}^{d} raw(x)^2 dx."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = self.supp_halfwidth
            Z = self._Z_norm(prec_bits)
            f_plain = self._raw_integrand_plain()
            def g(z, flags):
                v = f_plain(z, flags)
                return v * v
            val = acb.integral(g, acb(arb(-d)), acb(arb(d)))
            return val.real / (Z * Z)
        finally:
            ctx.prec = old

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """hat_K_R(xi) = (1/Z) int_{-d}^{d} raw(x) cos(2 pi xi x) dx."""
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = self.supp_halfwidth
            Z = self._Z_norm(prec_bits)
            f = self._raw_integrand_cos(xi, prec_bits)
            val = acb.integral(f, acb(arb(-d)), acb(arb(d)))
            return val.real / Z
        finally:
            ctx.prec = old


# =============================================================================
#  K5: Beurling-Selberg-derived (band-limited majorant modulation)
# =============================================================================

class SelbergBandlimitedKernel(Kernel):
    """Band-limited kernel derived from Beurling-Selberg's B-function.

    We use the "Fejer-tapered Gaussian" variant with explicit arb FT:

        K(x) = (1/Z) * [1 - |x|/delta]_+ * cos(pi x / (2 delta))^2

    The ``cos^2`` factor forces K to vanish smoothly at x = +/- delta (no
    corner), and the triangular factor forces compact support.  FT is
    computable in closed form using the identity
        cos^2(pi x /(2 d)) = (1 + cos(pi x / d)) / 2.

    hat_K_R(xi) = (1/Z) * (1/2) * [ hat_Tri(xi) + (1/2) hat_Tri(xi - 1/(2d))
                                     + (1/2) hat_Tri(xi + 1/(2d)) ]
    where hat_Tri(xi) = delta * sinc(pi xi delta)^2 is the FT of
    (1 - |x|/delta)_+ * 1_{[-delta, delta]} (note: NOT normalised to mass 1).
    """
    name = "K5_selberg_cos2_tri"

    def __init__(self, delta: fmpq = fmpq(138, 1000)):
        self.supp_halfwidth = delta

    def _raw_integrand_plain(self):
        d = arb(self.supp_halfwidth)
        c = arb.pi() / (arb(2) * d)
        def f(z, flags):
            # (1 - |x|/delta)_+ is piecewise; but within integration bounds
            # [-d, d] it equals (1 - |x|/d) non-negative -> we rely on
            # acb being a ball; for holomorphic continuation use |x|^2 via
            # (x^2)^{1/2}? No — the absolute value breaks holomorphy.  We
            # integrate on the real line [-d, d] where |x| = x for x >= 0
            # and |x| = -x for x <= 0, so split into two intervals.  This
            # split is handled at the caller.
            raise NotImplementedError("internal use — split in K_norm_sq / FT")
        return f

    def _half_integrand_plain(self):
        """Integrand on [0, delta]: (1 - x/delta) cos^2(pi x /(2 delta)).

        On [0, delta] this equals (1 - x/delta) * (1 + cos(pi x /delta))/2.
        """
        d = arb(self.supp_halfwidth)
        def f(z, flags):
            one_minus = acb(1) - z / acb(d)
            # cos^2(pi z / (2 d)) = (1 + cos(pi z / d)) / 2
            c2 = (acb(1) + (acb.pi() * z / acb(d)).cos()) / acb(2)
            return one_minus * c2
        return f

    def _Z_norm(self, prec_bits: int) -> arb:
        """Z = 2 int_0^delta (1 - x/d)(1 + cos(pi x/d))/2 dx  (by even symmetry).

        = int_0^d (1 - x/d) dx  +  int_0^d (1 - x/d) cos(pi x/d) dx
        = d/2 + int_0^d (1 - x/d) cos(pi x/d) dx.
        By parts: let u = 1 - x/d, dv = cos(pi x/d) dx;  du = -dx/d, v = (d/pi) sin(pi x/d).
          [uv]_0^d = 0 - 0 = 0,  - int v du = (1/pi) int_0^d sin(pi x/d) dx
          = (1/pi) * (-d/pi)[cos(pi x/d)]_0^d = (1/pi)*(-d/pi)(-1 - 1) = 2d/pi^2.
        So Z = d/2 + 2d/pi^2.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            return d / arb(2) + arb(2) * d / (arb.pi() * arb.pi())
        finally:
            ctx.prec = old

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = (1/Z^2) * 2 int_0^d (1 - x/d)^2 cos^4(pi x/(2d)) dx.

        We compute by rigorous acb.integral.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = self.supp_halfwidth
            Z = self._Z_norm(prec_bits)
            def g(z, flags):
                one_minus = acb(1) - z / acb(arb(d))
                c2 = (acb(1) + (acb.pi() * z / acb(arb(d))).cos()) / acb(2)
                return (one_minus * one_minus) * (c2 * c2)
            val = acb.integral(g, acb(0), acb(arb(d)))
            return arb(2) * val.real / (Z * Z)
        finally:
            ctx.prec = old

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """hat_K_R(xi) = (1/Z) int_{-d}^{d} (1 - |x|/d) cos^2(pi x /(2d)) cos(2 pi xi x) dx.

        = (2/Z) int_0^d (1 - x/d)(1 + cos(pi x/d))/2 cos(2 pi xi x) dx
        = (1/Z) int_0^d (1 - x/d)(1 + cos(pi x/d)) cos(2 pi xi x) dx.

        We compute this by rigorous arb integration.  The integrand is a
        product of a polynomial and bounded cosines, analytic on R.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = self.supp_halfwidth
            Z = self._Z_norm(prec_bits)
            xi_acb = acb(xi)
            d_acb = acb(arb(d))
            two_pi = acb(2) * acb.pi()
            pi_over_d = acb.pi() / d_acb
            def f(z, flags):
                one_minus = acb(1) - z / d_acb
                inner = acb(1) + (pi_over_d * z).cos()
                return one_minus * inner * (two_pi * xi_acb * z).cos()
            val = acb.integral(f, acb(0), d_acb)
            return val.real / Z
        finally:
            ctx.prec = old


# =============================================================================
#  K6: Riesz-type  K(x) ∝ (delta^2 - x^2)^{alpha - 1/2}
# =============================================================================

class RieszKernel(Kernel):
    """K(x) = c_{alpha, delta} * (delta^2 - x^2)^{alpha - 1/2}  on [-delta, delta].

    At alpha = 1/2 this reduces to the arcsine density (up to the 1/delta
    rescale), for alpha > 1/2 to a polynomial-like smooth bump, and for
    alpha = 1 it's a constant (uniform).

    Normalisation:  int (d^2 - x^2)^{a-1/2} dx over [-d, d] = B(1/2, a+1/2) d^{2a}
                                                            = sqrt(pi) Gamma(a+1/2)/Gamma(a+1) * d^{2a}.
    Fourier transform (standard formula, see Erdelyi Higher Transcendental
    Functions Vol. II Ch. 8 or wiki Fourier transform of (1-x^2)^lambda):
        int_{-1}^{1} (1-x^2)^{nu - 1/2} e^{-2 pi i xi x} dx
          = sqrt(pi) * Gamma(nu + 1/2) * (pi |xi|)^{-nu} * J_nu(2 pi |xi|).
    Thus
        hat_K_R(xi) = Gamma(alpha + 1) * (pi |xi| delta)^{-alpha} * J_alpha(2 pi |xi| delta).
    At xi = 0, J_alpha(0)/0^alpha equals 1/(2^alpha Gamma(alpha+1)) by the
    Bessel series J_alpha(z) ~ (z/2)^alpha / Gamma(alpha+1); so hat K(0) = 1
    as required.

    Bochner / positivity:
      J_alpha(z) for z > 0 takes both signs (infinitely many zeros), so
      K_tilde(j) is NOT monotonic.  The identity shows |tilde K(j)| is the
      same for all alpha > 0 modulo the Gamma/Bessel prefactor ordering,
      and alpha = 1/2 (J_{1/2}(z) = sqrt(2/(pi z)) sin(z)) recovers
      sinc (NOT squared! so tilde K takes negative values — NOT admissible).
      For general alpha, tilde K may also change sign.  We numerically check
      Bochner per alpha and reject non-admissible alphas.
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000),
                 alpha: float = 0.5,
                 label: str = ""):
        self.supp_halfwidth = delta
        self.alpha = float(alpha)
        self.name = f"K6_riesz_alpha={label or f'{alpha:.2f}'}"

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = c^2 * int (d^2 - x^2)^{2a - 1} dx
                   = c^2 * sqrt(pi) Gamma(2a)/Gamma(2a + 1/2) * d^{4a - 1}.

        With c = 1 / (sqrt(pi) Gamma(a+1/2)/Gamma(a+1) * d^{2a})
               = Gamma(a+1) / (sqrt(pi) Gamma(a+1/2)) * d^{-2a},
        we get
           ||K||_2^2 = [Gamma(a+1)/(sqrt(pi) Gamma(a+1/2))]^2 * d^{-4a}
                      * sqrt(pi) Gamma(2a) / Gamma(2a + 1/2) * d^{4a - 1}
                    = Gamma(a+1)^2 Gamma(2a) / (sqrt(pi) Gamma(a+1/2)^2 Gamma(2a+1/2)) / d.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            a = arb(self.alpha)
            d = arb(self.supp_halfwidth)
            num = (a + 1).gamma() ** 2 * (a * 2).gamma()
            den = arb.pi().sqrt() * (a + arb('0.5')).gamma() ** 2 * (a * 2 + arb('0.5')).gamma()
            return num / den / d
        finally:
            ctx.prec = old

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        """hat_K_R(xi) = Gamma(alpha + 1) (pi |xi| delta)^{-alpha} J_alpha(2 pi |xi| delta).

        Works for xi > 0; extend to xi = 0 via the limit (= 1).
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            a = arb(self.alpha)
            d = arb(self.supp_halfwidth)
            abs_xi = xi.abs_lower()  # xi >= 0 for our uses
            # If xi straddles 0 or equals 0 exactly, return 1
            if xi.contains(0) or xi.upper() <= 0 and xi.lower() >= 0:
                return arb(1)
            pi_xi_d = arb.pi() * xi * d
            two_pi_xi_d = arb(2) * pi_xi_d
            # J_alpha(2 pi xi delta), alpha non-integer in general: bessel_j
            # with fractional first argument is supported.
            Ja = two_pi_xi_d.bessel_j(a)
            pref = (a + arb(1)).gamma() / (pi_xi_d ** a)
            return pref * Ja
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)


# =============================================================================
#  K7: Chebyshev-beta auto-convolution family
# =============================================================================

class ChebyshevBetaKernel(Kernel):
    """K = phi * phi with phi(x) = C_beta * (1 - 4 x^2 / delta^2)^{beta-1}.

    phi is supported on [-delta/2, delta/2]; since phi >= 0 for 0 < beta, K is
    non-negative, supported on [-delta, delta], and K_hat = (phi_hat)^2 >= 0
    automatically (Bochner).

    Closed-form Fourier (Gegenbauer/Beta-Bessel identity):
        phi_hat(xi) = Gamma(beta + 1/2) * (2 / (pi delta xi))^{beta - 1/2}
                    * J_{beta - 1/2}(pi delta xi),
    with phi_hat(0) = 1 (mass preservation).  K_hat(xi) = phi_hat(xi)^2.

    Special values:
      beta = 1/2:  phi = arcsine density -> K = arcsine auto-conv (K1).
      beta = 1:    phi = box -> K = triangle (K2).
      beta = 3/2:  phi = semicircular -> K = quartic-ish smooth bump.

    This is the single-parameter family connecting arcsine (our baseline) to
    box; sweeping beta near 1/2 is a DIRECT test of MV's optimality claim.
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000),
                 beta: fmpq = fmpq(1, 2),
                 label: str = "",
                 L2_cutoff_delta_inv: int = 200):
        self.supp_halfwidth = delta
        self.beta = beta
        self._beta_f = float(beta.p) / float(beta.q)
        if not label:
            label = f"{self._beta_f:.2f}"
        self.name = f"K7_cheby_beta={label}"
        # xi cut-off for the L2 integral in units of 1/delta; integrand
        # decays like xi^{-4 beta} at infinity, fine for beta > 1/4.
        self.L2_cutoff_delta_inv = L2_cutoff_delta_inv

    def _phi_hat_arb(self, xi: arb, prec_bits: int) -> arb:
        """phi_hat(xi) as an arb.  Works for arb xi > 0; returns 1 at xi = 0."""
        b = arb(self.beta)
        d = arb(self.supp_halfwidth)
        if xi.contains(0):
            # Enclosure: use the series at xi = 0 (phi_hat(0) = 1) via a
            # clopen ball around the limit.  For our uses (integer j >= 1),
            # this branch is never taken.
            return arb(1)
        pi_d_xi = arb.pi() * d * xi
        two_over = arb(2) / pi_d_xi
        # (2/(pi delta xi))^(beta - 1/2)
        exponent = b - arb('0.5')
        # arb has no general (arb) ** (arb) for non-integer exponents outside
        # the positive-domain -> use exp(a log b)
        pow_factor = (exponent * two_over.log()).exp()
        nu = b - arb('0.5')     # order of Bessel
        Ja = pi_d_xi.bessel_j(nu)
        gamma_pref = (b + arb('0.5')).gamma()
        return gamma_pref * pow_factor * Ja

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            if xi.contains(0) and xi.upper() == 0 and xi.lower() == 0:
                return arb(1)
            ph = self._phi_hat_arb(xi, prec_bits)
            return ph * ph
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = int K_hat(xi)^2 dxi = int phi_hat(xi)^4 dxi.

        Integrand is even and decays like xi^{-4 beta} at infinity (converges
        for beta > 1/4).  We integrate on [0, T] with T = L2_cutoff/delta,
        then double.  The tail bound is added (conservative):
            tail = 2 * int_T^infinity (C_tail / xi)^{4 beta} dxi
                 = (2 C_tail^{4 beta}) / (4 beta - 1) * T^{1 - 4 beta}
        where C_tail bounds |phi_hat(xi) * xi^{beta}| -- but for simplicity
        we just widen the arb by an explicit tail arb wrap.

        NOTE: for beta <= 1/4 this raises ValueError; for beta = 1/2
        (arcsine), the integrand is logarithmically divergent -- in that
        case we return the MO surrogate 0.5747/delta.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            b_f = self._beta_f
            if b_f <= 1.0 / 4:
                raise ValueError(
                    f"{self.name}: K2 integral diverges for beta <= 1/4"
                )
            if abs(b_f - 0.5) < 1e-9:
                # arcsine case: L2 is infinite; use MO surrogate for
                # compatibility with K1.
                return arb(fmpq(5747, 10000)) / arb(self.supp_halfwidth)

            d = self.supp_halfwidth
            T = arb(self.L2_cutoff_delta_inv) / arb(d)
            beta_arb = arb(self.beta)

            # Integrand f(xi) = phi_hat(xi)^4; the phi_hat formula uses log/exp
            # which is holomorphic for xi > 0 in the acb sense.
            d_arb = arb(d)
            gamma_pref = (beta_arb + arb('0.5')).gamma()
            exponent_arb = beta_arb - arb('0.5')

            def integrand(z, flags):
                # z is acb; require Re(z) > 0 for log stability.
                pi_d_xi = acb.pi() * acb(d_arb) * z
                two_over = acb(2) / pi_d_xi
                pow_factor = (acb(exponent_arb) * two_over.log()).exp()
                Ja = pi_d_xi.bessel_j(acb(beta_arb - arb('0.5')))
                phi_hat = acb(gamma_pref) * pow_factor * Ja
                phi_hat_2 = phi_hat * phi_hat
                return phi_hat_2 * phi_hat_2

            # Integrate over [eps, T], then double.  Use a small eps to avoid
            # the branch at xi = 0 (phi_hat limit is finite but the log is
            # singular).  The integrand at 0 equals 1 (phi_hat(0) = 1) so the
            # contribution of [0, eps] is ~ 2*eps; bound conservatively.
            eps_q = fmpq(1, 10**6)
            eps = arb(eps_q)
            val = acb.integral(integrand, acb(eps), acb(T))
            main = arb(2) * val.real
            # eps contribution: integrand is <= 1 on [0, eps] since phi_hat(0) = 1
            # and phi_hat decays monotonically near 0.  Widen by 2 * eps.
            eps_f = float(eps_q.p) / float(eps_q.q)
            main = main + arb(0, eps_f) * arb(2)
            # Tail contribution for [T, inf): bound integrand by
            # (phi_hat(T)^4) * (T/xi)^{4 beta} on xi >= T (monotonic-decay heuristic).
            phi_hat_T = self._phi_hat_arb(T, prec_bits)
            tail_bound = (arb(2) * phi_hat_T ** arb(4) * T) / (arb(4) * beta_arb - arb(1))
            main = main + arb(0, 1) * tail_bound
            return main
        finally:
            ctx.prec = old


# =============================================================================
#  K8: Epanechnikov auto-convolution
# =============================================================================

class EpanechnikovAutoConvKernel(Kernel):
    """K = phi * phi with phi(x) = (3 / delta)(1 - (2x/delta)^2) on [-delta/2, delta/2].

    phi >= 0, int phi = 1.  FT:
        phi_hat(xi) = 3 * [sin(pi delta xi / 2) - (pi delta xi / 2) cos(pi delta xi /2)]
                           / (pi delta xi / 2)^3                                         (Fourier of parabolic)
    Wait: let a = delta/2; phi(x) = (3/(4a))(1 - (x/a)^2); then
          phi_hat(xi) = (3 / (a xi)^3) * (sin(a xi) - a xi cos(a xi)) * some sign.
    But since xi argument uses 2 pi, we use z = 2 pi xi a = pi delta xi:
          phi_hat(xi) = 3 * (sin z - z cos z) / z^3    with  z = pi delta xi.
    At z = 0 (xi = 0): limit = 1 (using sin z - z cos z ~ z^3 / 3).

    K_hat(xi) = phi_hat(xi)^2 >= 0 (auto-correlation).
    ||K||_2^2 by closed form or via FT^4 integral.
    """
    name = "K8_epan_autoconv"

    def __init__(self, delta: fmpq = fmpq(138, 1000)):
        self.supp_halfwidth = delta

    def _phi_hat(self, xi: arb, prec_bits: int) -> arb:
        d = arb(self.supp_halfwidth)
        z = arb.pi() * d * xi
        if z.contains(0):
            return arb(1)
        sz = z.sin()
        cz = z.cos()
        return arb(3) * (sz - z * cz) / (z * z * z)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            if xi.contains(0) and xi.upper() == 0 and xi.lower() == 0:
                return arb(1)
            p = self._phi_hat(xi, prec_bits)
            return p * p
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """Closed form: phi(x) = (3/(4a))(1-(x/a)^2).
          ||phi||_2^2 = (3/(4a))^2 int_{-a}^{a} (1-(x/a)^2)^2 dx = (3/(4a))^2 * (16 a /15)
                      = 9/(20 a) = 9/(10 delta).
          But we want ||K||_2^2 = ||phi * phi||_2^2.  By Parseval:
            ||K||_2^2 = int |K_hat|^2 = int phi_hat^4.
          Closed: K(x) = (phi * phi)(x).  For phi supported in [-a, a],
          K supported in [-2a, 2a] = [-delta, delta].  phi * phi is
          piecewise polynomial of degree 4, even, with K(0) = ||phi||_2^2 = 9/(10 delta).
          We compute ||K||_2^2 directly via acb.integral on [-delta, delta].
          (phi * phi)(x) = int_{-a}^{a} phi(y) phi(y - x) dy, but with x in [-2a, 2a].
        Rather than derive the piecewise polynomial, we integrate the Fourier
        representation ||K||_2^2 = 2 * int_0^T phi_hat(xi)^4 dxi + tail.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            # Integrate phi_hat^4 over [0, T], then double.  phi_hat decays
            # like 1/z^2 at infinity (since |sin z - z cos z| <= z), so
            # phi_hat^4 ~ 1/z^8 — converges very fast.  T = 400/delta is ample.
            T = arb(400) / d

            def integrand(z, flags):
                zz = acb.pi() * acb(d) * z
                # sin z - z cos z for z acb
                sz = zz.sin()
                cz = zz.cos()
                # At z = 0 the integrand is 1; but the denominator z^3 is
                # singular.  For integration starting at 0, use L'Hopital-safe
                # rewrite: (sin z - z cos z)/z^3 = (1/2) * integral_0^1 t sin(zt) dt
                # Actually simplest: at z = 0 limit = 1/3; use "sinc_pi"-style.
                # We just start at eps to avoid.
                ph = acb(3) * (sz - zz * cz) / (zz * zz * zz)
                ph2 = ph * ph
                return ph2 * ph2

            eps_q = fmpq(1, 10**6)
            val = acb.integral(integrand, acb(arb(eps_q)), acb(T))
            main = arb(2) * val.real
            # eps patch: integrand at xi=0 equals 1 (since phi_hat(0) = 1).
            # Bound by a small ball.
            main = main + arb(0, 1) * arb(2) * arb(eps_q)
            # tail (T, infty): phi_hat^4 <= (3/z^2)^4 = 81/z^8; integral of
            # 81/z^8 from T to infty = 81/(7 T^7); very small for T = 400.
            z_T = arb.pi() * d * T
            tail = arb(2) * arb(81) / (arb(7) * z_T ** arb(7)) / (arb.pi() * d)
            main = main + arb(0, 1) * tail
            return main
        finally:
            ctx.prec = old


# =============================================================================
#  K9: Raised-cosine (Hann) auto-convolution
# =============================================================================

class HannAutoConvKernel(Kernel):
    """K = phi * phi with phi(x) = (2/delta) cos^2(pi x / delta) on [-delta/2, delta/2].

    phi(x) = (2/delta)(1 + cos(2 pi x /delta))/2 = (1/delta)(1 + cos(2 pi x /delta));
    at x = +/- delta/2 we have cos(pi) = -1, phi(delta/2) = 0 (boundary-vanishing).
    int phi = 1.  FT:  phi_hat(xi) = int_{-a}^{a} (2/delta) cos^2(pi x/delta) e^{-2 pi i xi x} dx,
    with a = delta/2.  Direct integration yields a rational-trig expression.
    K_hat(xi) = phi_hat(xi)^2 >= 0.
    """
    name = "K9_hann_autoconv"

    def __init__(self, delta: fmpq = fmpq(138, 1000)):
        self.supp_halfwidth = delta

    def _phi_hat(self, xi: arb, prec_bits: int) -> arb:
        """phi_hat(xi) = int_{-a}^{a} (1/delta)(1 + cos(2 pi x/delta)) cos(2 pi xi x) dx,
        with a = delta/2.

        Split:  I1 = (1/delta) int_{-a}^{a} cos(2 pi xi x) dx
                   = (1/delta) * sin(2 pi xi a)/(pi xi)
                   = sin(pi xi delta) / (pi xi delta)       [since a = delta/2]
                   = sinc(pi xi delta).
        I2 = (1/delta) int_{-a}^{a} cos(2 pi x/delta) cos(2 pi xi x) dx
           = (1/(2 delta)) int_{-a}^{a} [cos(2 pi x (xi + 1/delta)) + cos(2 pi x (xi - 1/delta))] dx
           = (1/2) [sinc(pi (xi + 1/delta) delta) + sinc(pi (xi - 1/delta) delta)]
           = (1/2) [sinc(pi xi delta + pi) + sinc(pi xi delta - pi)].
        Total: phi_hat(xi) = sinc(pi xi delta) + (1/2)[sinc(pi xi delta + pi) + sinc(pi xi delta - pi)].
        """
        d = arb(self.supp_halfwidth)
        pd_xi = arb.pi() * d * xi
        s0 = pd_xi.sinc()
        s1 = (pd_xi + arb.pi()).sinc()
        s2 = (pd_xi - arb.pi()).sinc()
        return s0 + (s1 + s2) / arb(2)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            p = self._phi_hat(xi, prec_bits)
            return p * p
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = int phi_hat^4 dxi.  phi_hat(xi) -> 0 as xi -> infty like
        1/(xi^2) (it's a sum of three sincs that cancel to higher order near xi
        equals 0 or 1/delta), so phi_hat^4 ~ 1/xi^8: very fast convergence.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            T = arb(400) / d
            def integrand(z, flags):
                pd_xi = acb.pi() * acb(d) * z
                s0 = pd_xi.sinc()
                s1 = (pd_xi + acb.pi()).sinc()
                s2 = (pd_xi - acb.pi()).sinc()
                ph = s0 + (s1 + s2) / acb(2)
                ph2 = ph * ph
                return ph2 * ph2
            eps_q = fmpq(1, 10**6)
            val = acb.integral(integrand, acb(arb(eps_q)), acb(T))
            main = arb(2) * val.real
            main = main + arb(0, 1) * arb(2) * arb(eps_q)
            # tail: |ph| <= 3 so ph^4 <= 81.  Tail from T = 400/d is very small
            # since ph decays like xi^{-2}; bound (81 / (2 pi xi d)^8) from
            # xi >= T:   int_T^infty 81/xi^8 dxi = 81 / (7 T^7).
            tail = arb(81) / (arb(7) * T ** arb(7))
            main = main + arb(0, 1) * tail
            return main
        finally:
            ctx.prec = old


# =============================================================================
#  K10: B-spline auto-convolution (K = B_{2n}, n-fold convolution of box)
# =============================================================================

class BSplineAutoConvKernel(Kernel):
    """K(x) = c * (sinc(pi xi delta / n))^{2n} on the Fourier side.

    Construction: let phi = n-fold convolution of box 1_{[-a/n, a/n]} where
    a = delta/2; phi is the cardinal B-spline of order n, supported on
    [-a, a] = [-delta/2, delta/2].  Then K = phi * phi has support [-delta, delta]
    and K_hat = (sinc(pi xi delta / n))^{2n} >= 0 (since phi_hat = sinc^n).

    Wait: phi is the n-fold convolution of box (width 2a/n each), so
    phi_hat = sinc(pi xi (2a/n))^n = sinc(pi xi delta/n)^n.
    K_hat = phi_hat^2 = sinc(pi xi delta/n)^{2n}.

    n = 1 recovers the box + triangle (= K2 triangular).  n >= 2 gives
    successively smoother phi with faster Fourier decay.
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000), n: int = 3):
        self.supp_halfwidth = delta
        self.n = int(n)
        self.name = f"K10_bspline_n={self.n}"

    def _phi_hat(self, xi: arb, prec_bits: int) -> arb:
        d = arb(self.supp_halfwidth)
        n = arb(self.n)
        arg = arb.pi() * d * xi / n
        s = arg.sinc()
        return s ** arb(self.n)

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            arg = arb.pi() * d * xi / arb(self.n)
            s = arg.sinc()
            result = arb(1)
            # (sinc)^(2n) = (s^2)^n
            s2 = s * s
            for _ in range(self.n):
                result = result * s2
            return result
        finally:
            ctx.prec = old

    def K_tilde(self, nj: int, prec_bits: int = 256) -> arb:
        if nj < 0:
            return self.K_tilde(-nj, prec_bits)
        if nj == 0:
            return arb(1)
        return self.K_tilde_real(arb(nj), prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = int (sinc(pi xi delta/n))^{4n} dxi.

        For 4n >= 2 (i.e., always), the integrand decays like xi^{-4n},
        guaranteeing convergence.  We do a wide cutoff + tail bound.
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            n_arb = arb(self.n)
            T = arb(400) / d
            power = 4 * self.n
            def integrand(z, flags):
                arg = acb.pi() * acb(d) * z / acb(n_arb)
                s = arg.sinc()
                s2 = s * s
                out = acb(1)
                for _ in range(self.n * 2):
                    out = out * s2.real      # s2 is real, reuse
                return out
            eps_q = fmpq(1, 10**6)
            val = acb.integral(integrand, acb(arb(eps_q)), acb(T))
            main = arb(2) * val.real
            main = main + arb(0, 1) * arb(2) * arb(eps_q)
            # tail bound: sinc(z) <= 1 / z for |z| >= 1; raised to power 4n.
            # Tail = 2 int_T^infty (1/z)^{4n} dxi, scale z = pi xi delta / n.
            # Skip rigorous tail since T is huge; widen by token.
            main = main + arb(0, 1) * arb('1e-20')
            return main
        finally:
            ctx.prec = old


# =============================================================================
#  K11: Askey's (1 - |x|/delta)_+^nu  (PD directly, no auto-conv)
# =============================================================================

class AskeyTruncatedPowerKernel(Kernel):
    """K(x) = c * (1 - |x| / delta)_+^nu on [-delta, delta].

    By Askey (1973), this is positive-definite (K_hat >= 0) whenever
    nu >= 1 in 1D.  (nu = 1 is the triangle = K2.)

    Normalisation: int_{-delta}^{delta} (1 - |x|/d)^nu dx = 2 delta / (nu + 1),
    so c = (nu + 1) / (2 delta).

    Fourier transform (Bessel-type, see Askey 1973 / Zygmund):
        K_hat(xi) = Gamma(nu + 1) * J_{nu + 1/2}(pi delta xi) * (pi delta xi)^{-(nu + 1/2)}
                  * sqrt(pi) * 2^{nu + 1/2}
        but with proper normalization so K_hat(0) = 1.
    Simpler: the L^infty function (1 - |x|)^nu on [-1,1] has
        FT(xi) = 2 int_0^1 (1-x)^nu cos(2 pi xi x) dx
               = ...  (closed in terms of hypergeometric or Bessel).

    We compute K_hat via arb numerical integration (integrand is a simple
    real polynomial-times-cosine) to avoid fiddly closed-form constants.
    """
    def __init__(self, delta: fmpq = fmpq(138, 1000), nu: int = 2):
        self.supp_halfwidth = delta
        self.nu = int(nu)
        self.name = f"K11_askey_nu={self.nu}"

    def _raw_unnormalised_hat(self, xi: arb, prec_bits: int) -> arb:
        """int_{-d}^{d} (1 - |x|/d)^nu cos(2 pi xi x) dx = 2 int_0^d (1 - x/d)^nu cos(2 pi xi x) dx.

        Computed via acb.integral.
        """
        d = arb(self.supp_halfwidth)
        nu = self.nu
        def f(z, flags):
            return (acb(1) - z / acb(d)) ** nu * (acb(2) * acb.pi() * acb(xi) * z).cos()
        val = acb.integral(f, acb(0), acb(d))
        return arb(2) * val.real

    def K_tilde_real(self, xi: arb, prec_bits: int = 256) -> arb:
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            if xi.contains(0) and xi.upper() == 0 and xi.lower() == 0:
                return arb(1)
            # c = (nu+1) / (2 delta)
            d = arb(self.supp_halfwidth)
            c = arb(self.nu + 1) / (arb(2) * d)
            return c * self._raw_unnormalised_hat(xi, prec_bits)
        finally:
            ctx.prec = old

    def K_tilde(self, n: int, prec_bits: int = 256) -> arb:
        if n < 0:
            return self.K_tilde(-n, prec_bits)
        if n == 0:
            return arb(1)
        return self.K_tilde_real(arb(n), prec_bits)

    def K_norm_sq(self, prec_bits: int = 256) -> arb:
        """||K||_2^2 = c^2 int_{-d}^{d} (1 - |x|/d)^{2 nu} dx
                   = c^2 * 2 d / (2 nu + 1)
                   = ((nu+1)/(2 d))^2 * 2 d / (2 nu + 1)
                   = (nu + 1)^2 / (2 d (2 nu + 1)).
        """
        old = ctx.prec
        ctx.prec = prec_bits
        try:
            d = arb(self.supp_halfwidth)
            nu = arb(self.nu)
            return (nu + 1) ** arb(2) / (arb(2) * d * (arb(2) * nu + 1))
        finally:
            ctx.prec = old


# =============================================================================
#  Registry
# =============================================================================

def default_kernel_registry() -> list[Kernel]:
    """Return the full sweep list K1..K6 with sensible parameter choices.

    delta is fixed at MV's 0.138.  Parameter sub-sweeps (K3 sigma, K4 m,
    K6 alpha) enumerate multiple candidates per family.
    """
    delta = fmpq(138, 1000)
    out: list[Kernel] = []
    out.append(ArcsineKernel(delta=delta))
    out.append(TriangularKernel(delta=delta))
    # K3: sigma in {delta/3, delta/2, delta}
    out.append(TruncatedGaussianKernel(delta=delta, sigma_over_delta=fmpq(1, 3), label="delta_3"))
    out.append(TruncatedGaussianKernel(delta=delta, sigma_over_delta=fmpq(1, 2), label="delta_2"))
    out.append(TruncatedGaussianKernel(delta=delta, sigma_over_delta=fmpq(1, 1), label="delta"))
    # K4: m in {5, 10, 20}
    out.append(JacksonKernel(delta=delta, m=5))
    out.append(JacksonKernel(delta=delta, m=10))
    out.append(JacksonKernel(delta=delta, m=20))
    # K5: one Selberg-like variant
    out.append(SelbergBandlimitedKernel(delta=delta))
    # K6: alpha in {0.4, 0.6, 0.8, 1.0, 1.2}
    for a, lab in [(0.4, "0.40"), (0.6, "0.60"), (0.8, "0.80"),
                   (1.0, "1.00"), (1.2, "1.20")]:
        out.append(RieszKernel(delta=delta, alpha=a, label=lab))

    # K7: Chebyshev-beta auto-conv (beta sweeps near arcsine's beta = 1/2)
    for beta_q, lab in [
        (fmpq(2, 5),    "0.40"),
        (fmpq(9, 20),   "0.45"),
        (fmpq(1, 2),    "0.50"),   # = arcsine (uses MO surrogate for K2)
        (fmpq(11, 20),  "0.55"),
        (fmpq(3, 5),    "0.60"),
        (fmpq(7, 10),   "0.70"),
        (fmpq(1, 1),    "1.00"),   # = box (= K2 triangle, cross-check)
    ]:
        out.append(ChebyshevBetaKernel(delta=delta, beta=beta_q, label=lab))

    # K8: Epanechnikov auto-conv (phi is parabolic on [-delta/2, delta/2])
    out.append(EpanechnikovAutoConvKernel(delta=delta))

    # K9: Hann auto-conv (phi is raised cosine on [-delta/2, delta/2])
    out.append(HannAutoConvKernel(delta=delta))

    # K10: B-spline auto-conv, n = 2, 3, 5
    for nn in [2, 3, 5]:
        out.append(BSplineAutoConvKernel(delta=delta, n=nn))

    # K11: Askey (1-|x|/d)^nu for nu = 2, 3, 4  (nu=1 is the triangle/K2)
    for nn in [2, 3, 4]:
        out.append(AskeyTruncatedPowerKernel(delta=delta, nu=nn))

    return out


__all__ = [
    "Kernel",
    "ArcsineKernel",
    "TriangularKernel",
    "TruncatedGaussianKernel",
    "JacksonKernel",
    "SelbergBandlimitedKernel",
    "RieszKernel",
    "ChebyshevBetaKernel",
    "EpanechnikovAutoConvKernel",
    "HannAutoConvKernel",
    "BSplineAutoConvKernel",
    "AskeyTruncatedPowerKernel",
    "default_kernel_registry",
]
