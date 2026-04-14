/-
Sidon Autocorrelation Project — Root Module

Main theorem: autoconvolution_ratio_ge_32_25 (C₁ₐ ≥ 32/25 = 1.28)

Three components:
  • Proof/          — the main theorem and its proof chain (10 files, 2 axioms)
  • Algorithm/      — correctness of the cascade algorithm (17 files, 0 axioms)
  • CoarseCascade/  — proof stubs for coarse-grid cascade method (13 files, all sorry)
-/

-- Shared definitions
import Sidon.Defs

-- Main proof: C₁ₐ ≥ 32/25
import Sidon.Proof

-- Algorithm correctness: cascade is sound
import Sidon.Algorithm

-- Coarse cascade: no-correction method proof stubs
import Sidon.CoarseCascade
