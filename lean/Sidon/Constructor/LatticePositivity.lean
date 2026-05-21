/-
Sidon Autocorrelation Project — active-set lattice positivity `hK4`.
=====================================================================

This file is the canonical home for the residual

  `hK4 : ∀ j, 1 ≤ j → j ≤ 200 → 0 < K_ms_fourier_lattice j`

(the QP active-set period-`u` denominators are strictly positive),
where

  `K_ms_fourier_lattice j = ∑ᵢ λᵢ · J₀(π δᵢ j / u)²`,

`(δ₁,δ₂,δ₃) = (0.138, 0.055, 0.025)`, `(λ₁,λ₂,λ₃) = (0.85, 0.10, 0.05)`,
`u = 0.638`.

Two regimes:

  * **Structural prefix `1 ≤ j ≤ 16` (proved here, no axioms).**  The
    smallest scale `δ₃ = 25/1000` keeps the Bessel argument
    `|π δ₃ j / u| < 2` whenever `|j| < 2u/(π δ₃) = 1276/(25π) ≈ 16.246`,
    so `J₀(π δ₃ j / u) > 0` by the alternating-series bound
    `J₀(z) ≥ 1 - (z/2)²` (`Sidon.Constructor.besselJ0_pos_of_abs_lt_two`),
    and `λ₃ > 0` makes that summand strictly positive while the other
    two are `≥ 0`.  The cutoff is sharp for this *clean* bound: at
    `j = 17`, `|π δ₃·17/u| ≈ 2.093 > 2` and `1 - (z/2)² ≈ -0.095 < 0`,
    so the `1 - (z/2)²` certificate fails.

  * **Tail `17 ≤ j ≤ 200` (sanctioned verifiable-by-computation
    axiom).**  For `|j| ≥ 17` every scale's `J₀` argument can sit near
    a zero of `J₀`, so positivity rests on the three transcendental
    zero-sets of `J₀(π δ₁ ·), J₀(π δ₂ ·), J₀(π δ₃ ·)` not coinciding at
    the lattice point `j/u`.  This is a *logically decidable* finite
    conjunction of strict `J₀`-sum inequalities, but mathlib `v4.29.1`
    has no numerical-Bessel / interval-arithmetic library to discharge
    it.  We therefore record it as a single sanctioned
    verifiable-by-computation axiom `K_ms_fourier_lattice_pos_active`,
    backed by the same `flint.arb` certificate at 256-bit precision as
    the existing two numerical axioms — an analogue *in role* of MV's
    Mathematica citation, but strictly more rigorous (proven interval
    enclosures).  See `delsarte_dual/grid_bound_alt_kernel/`
    (`audit_lattice_positivity.py`).

The combined `K_ms_fourier_lattice_pos_active` ranges over all
`1 ≤ j ≤ 200`; the structural prefix is folded in as a *theorem* so the
axiom's effective content is only the genuinely-transcendental tail.

No `sorry`.  One new sanctioned verifiable-by-computation axiom
(documented to the same standard as the two existing numerical axioms);
no other new axioms.
-/

import Mathlib
import Sidon.MultiScale
import Sidon.Constructor.KernelFacts

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false
set_option maxHeartbeats 4000000

open scoped BigOperators Real Classical

namespace Sidon.Constructor.LatticePositivity

open Sidon.MultiScale
open Sidon.Constructor

noncomputable section

/-! ## Part 1 — structural prefix `1 ≤ j ≤ 16` (axiom-free)

The smallest scale `δ₃` drives a strictly-positive Bessel summand
whenever the argument stays inside `(-2, 2)`.  We re-export the generic
small-argument fragment and the explicit `|j| ≤ 16` corollary as the
canonical statements (the underlying Bessel positivity machinery lives
in `Sidon.Constructor.KernelFacts`). -/

/-- **Generic structural positivity (smallest scale).**
`K_ms_fourier_lattice j > 0` whenever the `δ₃` Bessel argument
`π · δ₃ · (j/u)` lies strictly inside `(-2, 2)`, via the strictly
positive smallest-scale (`δ₃`) Bessel term `J₀(π δ₃ j/u) > 0` and
`λ₃ > 0`, with the `δ₁`, `δ₂` summands `≥ 0`. -/
theorem pos_of_delta3_small {j : ℤ}
    (hj : |Real.pi * delta3 * ((j : ℝ) / uQ_real)| < 2) :
    0 < K_ms_fourier_lattice j := by
  unfold K_ms_fourier_lattice
  have hl1 : (0 : ℝ) ≤ lambda1 := by unfold lambda1; exact_mod_cast lambdas_nonneg.1
  have hl2 : (0 : ℝ) ≤ lambda2 := by unfold lambda2; exact_mod_cast lambdas_nonneg.2.1
  have hl3 : (0 : ℝ) < lambda3 := by unfold lambda3 lambda3Q; push_cast; norm_num
  have hJ3 : 0 < Sidon.Bessel.besselJ0 (Real.pi * delta3 * ((j : ℝ) / uQ_real)) :=
    Sidon.Constructor.besselJ0_pos_of_abs_lt_two hj
  have ht1 : (0 : ℝ) ≤ lambda1 *
      (Sidon.Bessel.besselJ0 (Real.pi * delta1 * ((j : ℝ) / uQ_real))) ^ 2 :=
    mul_nonneg hl1 (sq_nonneg _)
  have ht2 : (0 : ℝ) ≤ lambda2 *
      (Sidon.Bessel.besselJ0 (Real.pi * delta2 * ((j : ℝ) / uQ_real))) ^ 2 :=
    mul_nonneg hl2 (sq_nonneg _)
  have ht3 : (0 : ℝ) < lambda3 *
      (Sidon.Bessel.besselJ0 (Real.pi * delta3 * ((j : ℝ) / uQ_real))) ^ 2 :=
    mul_pos hl3 (by positivity)
  linarith

/-- **Structural prefix.** For `1 ≤ |j| ≤ 16`, `|π · δ₃ · (j/u)| < 2`,
hence `K_ms_fourier_lattice j > 0`.  (`2u/(π·δ₃) = 1276/(25π) ≈ 16.246`,
so the largest integer that fits is `16`.) -/
theorem pos_of_abs_le_16 {j : ℤ} (hj_le : |j| ≤ 16) :
    0 < K_ms_fourier_lattice j := by
  apply pos_of_delta3_small
  have hπ : Real.pi < 3.15 := Real.pi_lt_d2
  have hπ0 : (0 : ℝ) < Real.pi := Real.pi_pos
  have hd3 : delta3 = 25 / 1000 := by unfold delta3 delta3Q; push_cast; ring
  have hu : uQ_real = 638 / 1000 := by unfold uQ_real uQ; push_cast; ring
  have hjabs : (|(j : ℝ)|) ≤ 16 := by
    rw [← Int.cast_abs]; exact_mod_cast hj_le
  rw [hd3, hu]
  rw [abs_mul, abs_mul, abs_of_pos hπ0]
  have habs_div : |(j : ℝ) / (638 / 1000)| = |(j : ℝ)| / (638 / 1000) := by
    rw [abs_div]; congr 1; rw [abs_of_pos (by norm_num)]
  rw [abs_of_pos (by norm_num : (0:ℝ) < 25/1000), habs_div]
  have hkey : Real.pi * (25 / 1000) * (|(j : ℝ)| / (638 / 1000))
                = Real.pi * |(j : ℝ)| * (25 / 638) := by ring
  rw [hkey]
  nlinarith [hπ, hπ0, hjabs, abs_nonneg (j : ℝ)]

/-! ## Part 2 — the sanctioned verifiable-by-computation axiom

### Why option (ii-a) (a standalone axiom), not (ii-b) (fold into the gain axiom)

`hK4` cannot be soundly *folded into* `gain_analytic_ge_gainLowerQ`.
Over `ℝ` with genuine division, the certifier's finite value
`S_1 = ∑ⱼ aⱼ²/K̃(j) ≤ 29.84 < ∞` does witness every active `K̃(j) > 0`
(a zero denominator would diverge), and the arb run that produces the
gain certificate would itself error/overflow on a zero denominator.
**But the Lean axiom records only a rational *lower bound on the gain***
`(4/u)·min²/S_1 ≥ gainLowerQ`, equivalently an *upper* bound
`S_1_analytic ≤ (4/u)·min²/gainLowerQ`.  In Lean's `x/0 = 0`
convention a zero denominator makes its summand `0`, which only
*decreases* `S_1_analytic` and so still satisfies that upper bound.
Hence the gain axiom is logically insufficient to *force* positivity:
folding (ii-b) would be unsound as stated.  We therefore take **option
(ii-a)** — a single standalone axiom whose statement is exactly the
positivity fact the certifier verifies — keeping it logically faithful
to what `flint.arb` decides.

### The axiom

The structural prefix above already discharges `1 ≤ j ≤ 16` as a
*theorem*; the genuinely-axiomatic content is only the transcendental
tail `17 ≤ j ≤ 200`.  We state the axiom over the full active range
`1 ≤ j ≤ 200` for downstream convenience (the consumer threads a single
hypothesis), and immediately re-derive the prefix from the theorem so
no double-counting occurs. -/

/-- (NUMERICAL AXIOM 3, paper Lemma 4.6 — active-set lattice positivity)
`K_ms_fourier_lattice j > 0` for every active QP index `1 ≤ j ≤ 200`.

The `flint.arb` certifier evaluates each
`K̃_ms(j) = ∑ᵢ λᵢ · J₀(π δᵢ j / u)²` as an arb interval at 256-bit
precision and verifies `K̃_ms(j).lower() > 0` for all `200` indices.
The minimum certified value is
`min_{1 ≤ j ≤ 200} K̃_ms(j) ≈ 0.000208174` (certified arb lower
endpoint, attained at `j = 147`, comfortably bounded away from `0`);
see `delsarte_dual/grid_bound_alt_kernel/audit_lattice_positivity.py`.

This is a *logically decidable* statement: a finite conjunction of `200`
strict `J₀`-sum inequalities about specific real numbers, exactly the
kind any sufficient interval-arithmetic implementation can decide.  It
appears as an `axiom` rather than a `theorem` only because mathlib
`v4.29.1` provides no numerical-Bessel / interval-arithmetic library to
discharge it mechanically (`Sidon.Bessel` supplies only the
alternating-series bound `J₀(z) ≥ 1 - (z/2)²`, useful for `|z| < 2`,
i.e. the `1 ≤ j ≤ 16` prefix, which is a *theorem* — see
`pos_of_abs_le_16`).

*Provenance.* This plays the same role as MV (2010)'s Mathematica
citations for `J₀`-evaluations, but is strictly more rigorous: a
proven `flint.arb` interval enclosure at 256-bit precision rather than
heuristic CAS numerics, anchored to the same SHA-256-stamped
reproducible certificate as the other two numerical axioms.  It is a
*new* numerical fact specific to the three-scale kernel `K_ms` and the
200-point active set, not a statement contained in MV 2010.

*Relation to the gain axiom.*  It is **not** derivable from
`gain_analytic_ge_gainLowerQ` (which records only a rational *upper*
bound on `S_1` — see the discussion above), so it is recorded as a
distinct axiom rather than folded in. -/
axiom K_ms_fourier_lattice_pos_active :
    ∀ j : ℤ, 1 ≤ j → j ≤ 200 → 0 < K_ms_fourier_lattice j

/-- **Active-set lattice positivity `hK4`.**  `K_ms_fourier_lattice j > 0`
for every active QP index `1 ≤ j ≤ 200`.

This is the form `field_hEq4` / the bundle constructor consume.  The
prefix `1 ≤ j ≤ 16` is discharged by the structural theorem
`pos_of_abs_le_16` (axiom-free); the tail `17 ≤ j ≤ 200` uses the
sanctioned verifiable-by-computation axiom
`K_ms_fourier_lattice_pos_active`. -/
theorem hK4_active (j : ℤ) (hj1 : 1 ≤ j) (hj2 : j ≤ 200) :
    0 < K_ms_fourier_lattice j := by
  by_cases hsmall : |j| ≤ 16
  · -- Structural prefix, axiom-free.
    exact pos_of_abs_le_16 hsmall
  · -- Transcendental tail `17 ≤ j ≤ 200`, sanctioned axiom.
    exact K_ms_fourier_lattice_pos_active j hj1 hj2

end

end Sidon.Constructor.LatticePositivity
