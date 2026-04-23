"""Alternative-kernel sweep (grid_bound_alt_kernel).

This package replaces MV's arcsine-derived kernel K with other admissible
kernels (triangular, truncated Gaussian, Jackson, Beurling-Selberg, Riesz)
and recomputes the N=1 Phi bisection bound M_cert for each one.

MV claim "are quite convinced that the choice of K in [MO] is optimal" but
never prove it. This sweep looks for counterexamples.
"""
