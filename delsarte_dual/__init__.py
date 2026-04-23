"""Delsarte-type dual lower bound on the Sidon autocorrelation constant.

Goal: prove a rigorous lower bound on
    C_{1a} = inf { ||f*f||_infty : f>=0, supp(f) ⊂ [-1/4,1/4], int f = 1 }
via admissible positive-definite test functions g, using interval
arithmetic (mpmath) for the verification step.

Current target: beat the Matolcsi-Vinuesa record 1.2802.

Public pipeline
---------------
    from delsarte_dual.run_all import run_all
    run_all()

Produces a two-sided ball [lb_low, lb_high] for each family and reports
the best rigorous lb_low.
"""

from . import family_f1_selberg  # noqa: F401
from . import family_f2_gauss_poly  # noqa: F401
from . import family_f3_vaaler  # noqa: F401
from . import rigorous_max  # noqa: F401
