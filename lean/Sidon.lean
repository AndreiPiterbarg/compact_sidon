/-
Sidon Autocorrelation Project — Root Module

Imports all proof sections. This file serves as the single entry point
that combines all split proof modules into one compilable unit.
-/

-- Core definitions
import Sidon.Defs

-- Foundational lemmas (F1-F15)
import Sidon.Foundational

-- Reversal symmetry (Claims 3.3a, 3.3e)
import Sidon.ReversalSymmetry

-- Refinement mass preservation (Claims 3.2c, 4.6)
import Sidon.RefinementMass

-- Incremental autoconvolution (Claim 4.2)
import Sidon.IncrementalAutoconv

-- Fused kernel and quick-check (Claims 4.1, 4.3)
import Sidon.FusedKernel

-- Composition enumeration (Claims 3.1, 3.2a)
import Sidon.CompositionEnum

-- Subtree pruning (Claim 4.4)
import Sidon.SubtreePruning

-- Cauchy-Schwarz single-bin bound (Claims 4.5, 4.7, 4.8)
import Sidon.CauchySchwarz

-- Cascade induction (Claim 3.4)
import Sidon.CascadeInduction

-- Gray code kernel (Claims 4.9, 4.10, 4.11)
import Sidon.GrayCode

-- Sliding window and zero-bin skip (Claims 4.12, 4.13)
import Sidon.SlidingWindow

-- Gray code subtree pruning (Claims 4.14–4.25)
import Sidon.GrayCodeSubtreePruning

-- Asymmetry bound (Claim 2.1)
import Sidon.AsymmetryBound

-- Refinement & support properties (Claims 2.2, 2.3)
import Sidon.RefinementSupport

-- Correction term support lemmas
import Sidon.CorrectionSupport

-- Essential supremum bounds
import Sidon.EssSup

-- Step function and grid convolution
import Sidon.StepFunction

-- Test value bounds (Claims 1.1, continuous test value)
import Sidon.TestValueBounds

-- Discretization error and correction terms (Claims 1.2, 1.3, 1.4)
import Sidon.DiscretizationError

-- Sparse cross-term optimization (Claims 4.26–4.35)
import Sidon.SparseCrossTerm

-- Arc consistency / range tightening (Claims 6.30–6.36)
import Sidon.ArcConsistency

-- Final result: c ≥ 133/100 = 1.33
import Sidon.FinalResult
