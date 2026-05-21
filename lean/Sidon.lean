/-
Sidon Autocorrelation Constant: rigorous lower bound `C_{1a} ≥ 1.292`.

This is the top-level entry point of the Lean formalisation of the
paper *A New Lower Bound for the Supremum of Autoconvolutions*.  The
proof of the headline theorem lives in `Sidon.MultiScale`.

Construction.  Three-scale arcsine kernel applied to the
Matolcsi–Vinuesa (2010) master inequality, with all numerical anchors
discharged by a `flint.arb` certifier at 256-bit precision (see
`delsarte_dual/grid_bound_alt_kernel/`).

Headline theorems.
  * `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`
    (and equivalently `Sidon.MultiScale.autoconvolution_ratio_ge_1_292`,
    `Sidon.MultiScale.C1a_ge_1292`) — the bundle headline, parameterised
    on an `ExtremiserPrimitives f` bundle of analytic primitives.
  * `Sidon.MultiScale.C1a_ge_1292_unconditional`
    (in `Sidon.Constructor.Assembly`) — the unconditional headline:
    the bundle is constructed from raw admissibility hypotheses
    (`ExtremiserPrimitives.of_admissible`), so it carries only
    admissibility hypotheses, no bundle.

Axioms.    The bundle headline `autoconvolution_ratio_ge_1292_1000`
           reaches exactly **two** numerical-only user axioms in its
           dependency closure:
             * `Sidon.MultiScale.K2_analytic_le_K2UpperQ`
               (`K_2(K_ms) ≤ 47897/10000`, paper Lemma 4.2), and
             * `Sidon.MultiScale.gain_analytic_ge_gainLowerQ`
               (`gain_analytic ≥ 20925/100000`, paper Lemmas 4.3–4.5).
           The unconditional headline `C1a_ge_1292_unconditional`
           additionally reaches the two certifier facts discharged inside
           the constructor:
             * `Sidon.MultiScale.min_G_analytic_ge_minGLowerQ`
               (`min_G_analytic ≥ 998/1000`, paper Lemma 4.3), and
             * `Sidon.Constructor.LatticePositivity.K_ms_fourier_lattice_pos_active`
               (active-set lattice positivity, paper Lemma 4.6).
           All four are certifier outputs of the form
           "`flint.arb` evaluated this functional".

           `Sidon.MultiScale.MV_master_inequality_for_extremiser` is
           now a Lean **theorem**, composed from the zero-axiom
           wire-ups `MV_master_via_slack_monotonicity` and
           `MV_master_inequality_from_MV_lemmas`.  The quadratic
           inversion `master_inequality_M_lower` and the five
           slack-soundness statements (`K_two_upper_bound`,
           `k_one_lower_bound`, `S_one_upper_bound`,
           `min_G_lower_bound`, `gain_lower_bound`) are likewise
           Lean *theorems*.

Layout.    The formalisation totals ≈ 15.6 kLoC across 30 modules:
           this root entry, the 13 core modules in `Sidon/` (Defs,
           Bessel, BilinearParseval, BundleDefs, BundleEq1,
           BundleEq2Schwartz, BundleEq3Schwartz, BundleEq4, FourierAux,
           MasterFromLemmas, MultiScale, MVLemmas, TorusParseval), and
           the 17 modules in `Sidon/Constructor/` that construct the
           `ExtremiserPrimitives` bundle from raw admissibility for the
           unconditional headline.

No `sorry`, no conjectural axioms.  Per-module axiom inventories can
be printed by running e.g. `lake env lean AxiomCheckBundleDefs.lean`,
`lake env lean AxiomCheckFourier.lean`, `lake env lean AxiomCheckMV.lean`,
or `lake env lean AxiomCheckTorus.lean`.
-/

import Sidon.Defs
import Sidon.Bessel
import Sidon.FourierAux
import Sidon.TorusParseval
import Sidon.MVLemmas
import Sidon.MasterFromLemmas
import Sidon.BundleDefs
import Sidon.BundleEq1
import Sidon.BundleEq2Schwartz
import Sidon.BundleEq3Schwartz
import Sidon.BundleEq4
import Sidon.BilinearParseval
import Sidon.MultiScale
import Sidon.Constructor.CauchySchwarzFloor
import Sidon.Constructor.Assembly
import Sidon.Constructor.LatticePositivity
import Sidon.Constructor.PoissonSummable
