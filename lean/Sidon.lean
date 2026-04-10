/-
Sidon Autocorrelation Project — Root Module

Main theorem: autoconvolution_ratio_ge_32_25 (C₁ₐ ≥ 32/25 = 1.28)

Two components:
  • Proof/    — the main theorem and its proof chain (8 files, 2 axioms)
  • Algorithm/ — correctness of the cascade algorithm (15 files, 0 axioms)
-/

-- Shared definitions
import Sidon.Defs

-- Main proof: C₁ₐ ≥ 32/25
import Sidon.Proof

-- Algorithm correctness: cascade is sound
import Sidon.Algorithm
