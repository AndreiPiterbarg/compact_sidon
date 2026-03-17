Once those print versions, you can build the proof:

  cd "C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\lean"
  lake update
  lake exe cache get
  lake build

  The lake update + cache get steps will take a while the first time (downloading Mathlib). lake build is what actually type-checks the proof.