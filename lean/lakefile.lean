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

-- Lasserre SDP proof inventory (stub-only audit of proof obligations)
lean_lib LasserreAudit where
  srcDir := "."
  roots := #[`lasserre]

-- Interval BnB: rigorous derivation C_{1a} ≥ val(d) from Python definitions
lean_lib IntervalBnB where
  srcDir := "."
  roots := #[`IntervalBnB]

-- SOS-dual Farkas LP verification (soundness of lasserre/dual_sdp.py output)
lean_lib SOSDual where
  srcDir := "."
  roots := #[`SOSDual]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "f897ebcf72cd16f89ab4577d0c826cd14afaafc7"
