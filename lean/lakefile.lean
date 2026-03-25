import Lake
open Lake DSL

package sidon where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

-- Split proof modules (default build target)
@[default_target]
lean_lib Sidon where
  srcDir := "."
  roots := #[`Sidon]

-- Legacy monolithic proof (kept for reference)
lean_lib SidonMonolithic where
  srcDir := "."
  roots := #[`complete_proof]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "f897ebcf72cd16f89ab4577d0c826cd14afaafc7"
