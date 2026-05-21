/-
Sidon Autocorrelation Project — Constructor assembly (`of_admissible`).
=======================================================================

This file assembles the constructor

  `Sidon.MultiScale.ExtremiserPrimitives.of_admissible`

which builds the bundle `ExtremiserPrimitives f` for a general admissible
`f` (nonneg, `L¹ ∩ L²`, supported in `(-1/4, 1/4)`, mass `1`) from the
already-proven field lemmas in `Sidon.Constructor.*`, collecting every
prerequisite that is **not** discharge-able from admissibility (nor
chainable from another field lemma) as an EXPLICIT, clearly-labelled
hypothesis.

The anchor fields `m_G, S_G, S_cos, LHS1, LHS2` are set to their canonical
analytic values, so their binding equations are `rfl`; the gain identity
`gain_eq` and `S_G_pos` are proved inline from `uQ_real_pos` and
`S_1_analytic_pos`.  The four MV Lemma 3.1 outputs are chained from
`field_hEq1`, `field_hEq2_canonical`, `field_hEq3_ge`, `field_hEq4`, with
`field_K2_ge_1`/`field_R_ge_1` feeding the `1 ≤ K₂` / `1 ≤ R(f)` slots of
`field_hEq2_canonical` (chained, not re-listed).

From `of_admissible` the unconditional headline
`Sidon.MultiScale.C1a_ge_1292_unconditional` follows by feeding the bundle
into `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`, with the
finiteness side-condition supplied by `conv_eLpNorm_top_ne_top` (the
`FieldsEasy` helper) and `∫ f > 0` derived from `∫ f = 1`.

The residual analytic hypotheses are HYPOTHESES of `of_admissible`, not
axioms; hence `#print axioms` for the headline lists only the two
verifiable-by-computation numerical axioms (plus the Lean-core trio).

No `sorry`, no new `axiom`.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.FieldsEasy
import Sidon.Constructor.FieldsParseval
import Sidon.Constructor.FieldEq4
import Sidon.Constructor.KernelFacts
import Sidon.Constructor.KernelL2
import Sidon.Constructor.Glue
import Sidon.Constructor.CauchySchwarzFloor
import Sidon.Constructor.LatticePositivity
import Sidon.Constructor.PoissonSampling
import Sidon.Constructor.PoissonSummable
import Sidon.Constructor.Eq2Split
import Sidon.Constructor.Eq2Period1

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option linter.deprecated false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical
open MeasureTheory

namespace Sidon.MultiScale

open Sidon.Constructor
open Sidon.FourierAux (autocorr)

noncomputable section

/-- **Constructor for the extremiser primitives bundle.**

For a general admissible `f` (nonneg, `L¹ ∩ L²`, supported in
`(-1/4, 1/4)`, mass `1`), `of_admissible` builds `ExtremiserPrimitives f`
from the proven `Sidon.Constructor.*` field lemmas.  Every prerequisite
that cannot be discharged from admissibility (nor chained from another
field) is carried as an explicit hypothesis, grouped and labelled below:

  * `[MemLp]` — `Integrable (K_ms²)` is PROVED unconditionally
    (`Sidon.Constructor.K_ms_sq_integrable`, via the Young
    `L^{4/3} ⋆ L^{4/3} → L²` inequality); discharged inside.
  * `[Glue]` — `hProd_int`, `hConv_prod_int`, `h_joint` PROVED from
    admissibility (`Sidon.Constructor.Glue` / `Sidon.BundleEq1`).
  * `[MV Eq.(2)]` — `hEq2` is now FULLY DISCHARGED from admissibility
    via the period-1 Parseval split
    (`Sidon.Constructor.Eq2Period1.field_hEq2_unconditional`), which
    takes ONLY the five admissibility hypotheses.  There are NO residual
    `J2`, `hJ2_0`, `h_parseval_split`, or `h_F_bound` hypotheses.
  * `[MV Eq.(3)]` — `h_torus_eq_full` (the TRUE identity
    `LHS1 + LHS2 = 2/u + (2/u)·S_cos f`, full-line `S_cos`, coefficient
    `2/u`) is now DISCHARGED hypothesis-free via the periodisation Poisson
    sampling
    (`Sidon.Constructor.PoissonSampling.field_P3_periodUParsevalTrue_via_periodization`,
    summability from `Sidon.Constructor.PoissonSummable`).  The bundle
    anchor rebinds `S_cos := S_cos f / u³` so the struct's `2u²`-convention
    field `hEq3_ge` is exactly the `≤`-direction (`field_hEq3_ge`).
  * `[MV Eq.(4)]` — `h_cs` : the genuine full-tsum Cauchy–Schwarz floor
    `S_cos f ≥ 2·min_G_analytic²/S_1_analytic`, PROVED from admissibility
    + the certifier sign `0 ≤ min_G_analytic` (itself DISCHARGED from the
    sanctioned `min_G_analytic_ge_minGLowerQ` axiom via
    `Sidon.MultiScale.min_G_analytic_nonneg`); TRUE-as-stated and
    `flint.arb`-certified.

The `1 ≤ K2_analytic` and `1 ≤ autoconvolution_ratio f` slots needed by
`field_hEq2_canonical` are CHAINED from `field_K2_ge_1`/`field_R_ge_1`,
never re-listed as residuals.

This is a `def` (not a `theorem`) because `ExtremiserPrimitives f` is a
data-bearing structure (`Type`, not `Prop`); it carries the canonical
analytic anchor values `m_G, S_G, S_cos, LHS1, LHS2`. -/
def ExtremiserPrimitives.of_admissible (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MeasureTheory.MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    ExtremiserPrimitives f := by
  -- `Integrable (K_ms²)` is now PROVED unconditionally (Young `L^{4/3}⋆L^{4/3}→L²`),
  -- not assumed: see `Sidon.Constructor.K_ms_sq_integrable`.
  have hK_sq_int : Integrable (fun x => Sidon.MultiScale.K_ms x ^ 2) volume :=
    Sidon.Constructor.K_ms_sq_integrable
  -- `hProd_int` (= `hAuto_prod_int`), `h_joint` PROVED from admissibility
  -- (`Sidon.Constructor.Glue`).
  have hProd_int : Integrable (fun x => autocorr f x * Sidon.MultiScale.K_ms x) volume :=
    Sidon.Constructor.Glue.hProd_int f hf_int hf_L2
  have h_joint : Integrable
      (Function.uncurry (fun x t : ℝ => f t * f (x + t)))
      (volume.prod volume) :=
    Sidon.Constructor.Glue.h_joint f hf_int
  -- `hConv_prod_int` : `Integrable ((f*f) · K_ms)` PROVED from admissibility
  -- (`Sidon.BundleEq1.conv_K_ms_integrable`, as in `field_hEq1`); needed by the
  -- period-`u` Parseval-equality discharge `field_P3_periodUParsevalTrue_via_periodization`.
  have hConv_prod_int : Integrable
      (fun x => (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) x
          * Sidon.MultiScale.K_ms x) volume := by
    have h_conv_fin :
        eLpNorm (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ volume ≠ ⊤ :=
      Sidon.Constructor.conv_eLpNorm_top_ne_top hf_L2
    have h_conv_int :
        Integrable (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) volume :=
      hf_int.integrable_convolution _ hf_int
    exact Sidon.BundleEq1.conv_K_ms_integrable hf_nonneg
      h_conv_int.aestronglyMeasurable h_conv_fin
  -- The certifier sign `0 ≤ min_G_analytic` is now DISCHARGED from the sanctioned
  -- `min_G_analytic_ge_minGLowerQ` axiom (no longer a residual hypothesis).
  have hmin_nonneg : 0 ≤ Sidon.MultiScale.min_G_analytic :=
    Sidon.MultiScale.min_G_analytic_nonneg
  -- The MV Eq.(3) full period-`u` Parseval EQUALITY, DISCHARGED hypothesis-free
  -- via the periodisation Poisson sampling
  -- (`Sidon.Constructor.PoissonSampling.field_P3_periodUParsevalTrue_via_periodization`),
  -- with summability supplied by `Sidon.Constructor.PoissonSummable`.
  have h_summable := Sidon.Constructor.PoissonSummable.summable_fourier_product
    f hf_int hf_L2 hf_supp
  have h_summable_re := Sidon.Constructor.PoissonSummable.summable_fourier_re
    f hf_L2 hf_supp
  have h_torus_eq_full :
      Sidon.MultiScale.LHS1 f + Sidon.MultiScale.LHS2 f
        = 2 / Sidon.MultiScale.uQ_real
            + (2 / Sidon.MultiScale.uQ_real) * Sidon.MultiScale.S_cos f :=
    Sidon.Constructor.PoissonSampling.field_P3_periodUParsevalTrue_via_periodization
      f hf_int hf_L2 hf_one hf_supp h_joint hConv_prod_int hProd_int
      h_summable h_summable_re
  -- MV Eq.(4) full-tsum Cauchy–Schwarz floor `h_cs`, PROVED from admissibility
  -- + the discharged certifier sign `hmin_nonneg` (finite CS + period-`u`
  -- Plancherel summability — NO Poisson summation;
  -- `Sidon.Constructor.cauchy_schwarz_floor`).
  have h_cs : Sidon.MultiScale.S_cos f
              ≥ 2 * Sidon.MultiScale.min_G_analytic ^ 2
                  / Sidon.MultiScale.S_1_analytic :=
    Sidon.Constructor.cauchy_schwarz_floor f hf_int hf_L2 hf_supp hf_nonneg hf_one hmin_nonneg
  -- Chained positivity facts (NOT residuals).
  have hK2_ge_1 : 1 ≤ Sidon.MultiScale.K2_analytic := field_K2_ge_1 hK_sq_int
  have hR_ge_1 : 1 ≤ autoconvolution_ratio f :=
    field_R_ge_1 f hf_int hf_L2 hf_supp hf_nonneg hf_one
  -- Inline `S_G > 0` from `uQ_real_pos` + `S_1_analytic_pos`.
  have hSG_pos : 0 < uQ_real * S_1_analytic / 2 := by
    have h1 : 0 < uQ_real := uQ_real_pos
    have h2 : 0 < S_1_analytic := S_1_analytic_pos
    positivity
  refine
    { m_G := min_G_analytic
      S_G := uQ_real * S_1_analytic / 2
      S_cos := Sidon.MultiScale.S_cos f / uQ_real ^ 3
      LHS1 := Sidon.MultiScale.LHS1 f
      LHS2 := Sidon.MultiScale.LHS2 f
      m_G_eq := rfl
      S_G_eq := rfl
      S_cos_eq := rfl
      LHS1_eq := rfl
      LHS2_eq := rfl
      K2_ge_1 := hK2_ge_1
      gain_eq := ?_
      R_ge_1 := hR_ge_1
      S_G_pos := hSG_pos
      hEq1 := field_hEq1 f hf_int hf_L2 hf_supp hf_nonneg hf_one
      hEq2 := ?_
      hEq3_ge := ?_
      hEq4 := ?_ }
  · -- `gain_eq` : `gain_analytic = 2 · min_G² / (u · S_1 / 2)`.
    unfold gain_analytic
    have hu : uQ_real ≠ 0 := ne_of_gt uQ_real_pos
    have hS1 : S_1_analytic ≠ 0 := ne_of_gt S_1_analytic_pos
    field_simp
    ring
  · -- `hEq2` via the PERIOD-1 Parseval split, FULLY DISCHARGED from admissibility
    -- (no `h_parseval_split` / `h_F_bound` / `h_K_bound` residuals):
    -- `Sidon.Constructor.Eq2Period1.field_hEq2_unconditional`.
    exact Sidon.Constructor.Eq2Period1.field_hEq2_unconditional
      f hf_int hf_L2 hf_supp hf_nonneg hf_one
  · -- `hEq3_ge` via the *true* (`2/u`-coefficient) period-`u` Parseval
    -- equality; `field_hEq3_ge` rewrites `2u²·(S_cos/u³) = (2/u)·S_cos`.
    exact field_hEq3_ge f h_torus_eq_full
  · -- `hEq4` via the genuine full-tsum Cauchy–Schwarz floor `h_cs`.
    exact field_hEq4 f h_cs

/-- **Unconditional headline.**  For a general admissible `f` (nonneg,
`L¹ ∩ L²`, supported in `(-1/4, 1/4)`, mass `1`), `R(f) ≥ 1292/1000`.
The theorem takes ONLY the five admissibility hypotheses (`hf_int`,
`hf_L2`, `hf_supp`, `hf_nonneg`, `hf_one`); there are NO residual
analytic hypotheses.

Note on "unconditional": this refers to the absence of an
`ExtremiserPrimitives` bundle hypothesis (the bundle is *constructed*
from admissibility), NOT to the `L²` membership `hf_L2`, which is part
of admissibility.  The literature constant `C₁ₐ = inf R(f)` over the
full nonnegative `L¹` class reduces to this `L¹ ∩ L²` statement by the
Schinzel–Schmidt 2002 step-function argument — the *identical* step
Matolcsi–Vinuesa 2010 invoke (their reference [4], §2); see
`lower_bound_proof.tex` §1 and `literature_verification.md`.

ALL former residuals — `hProd_int`, `h_joint`, the certifier sign
`0 ≤ min_G_analytic`, the MV Eq.(4) Cauchy–Schwarz floor `h_cs`, the MV
Eq.(2) data, and the MV Eq.(3) full period-`u` Parseval equality
`h_torus_eq_full` — are now DISCHARGED inside `of_admissible`:
`hmin_nonneg` from `Sidon.MultiScale.min_G_analytic_nonneg` (sanctioned
`min_G_analytic_ge_minGLowerQ` axiom), `hEq2` hypothesis-free from the
period-1 Parseval split
`Sidon.Constructor.Eq2Period1.field_hEq2_unconditional`, and
`h_torus_eq_full` hypothesis-free from the periodisation Poisson sampling
`Sidon.Constructor.PoissonSampling.field_P3_periodUParsevalTrue_via_periodization`
(summability from `Sidon.Constructor.PoissonSummable`).

`#print axioms` for this theorem lists the Lean-core trio plus the four
sanctioned verifiable-by-computation numerical axioms
`K2_analytic_le_K2UpperQ`, `gain_analytic_ge_gainLowerQ`,
`K_ms_fourier_lattice_pos_active`, and `min_G_analytic_ge_minGLowerQ`. -/
theorem C1a_ge_1292_unconditional (f : ℝ → ℝ)
    (hf_int : Integrable f volume)
    (hf_L2 : MeasureTheory.MemLp f 2 volume)
    (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (hf_one : ∫ x, f x ∂volume = 1) :
    autoconvolution_ratio f ≥ (1292 / 1000 : ℝ) := by
  -- `∫ f > 0` from `∫ f = 1`.
  have hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0 := by
    rw [hf_one]; norm_num
  -- Finiteness of `‖f*f‖_∞` (PRE-G helper from `FieldsEasy`).
  have h_conv_fin : MeasureTheory.eLpNorm
      (MeasureTheory.convolution f f
        (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
      ⊤ MeasureTheory.volume ≠ ⊤ :=
    Sidon.Constructor.conv_eLpNorm_top_ne_top hf_L2
  -- Build the bundle and feed the headline.  ALL residuals (including the
  -- hEq2-family) are now discharged INSIDE `of_admissible` (from admissibility
  -- + the sanctioned numerical axioms).
  exact autoconvolution_ratio_ge_1292_1000 f hf_nonneg hf_supp hf_int_pos h_conv_fin
    (ExtremiserPrimitives.of_admissible f hf_int hf_L2 hf_supp hf_nonneg hf_one)

end -- noncomputable section

end Sidon.MultiScale
