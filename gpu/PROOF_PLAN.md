# GPU Proof Plan: Proving C₁ₐ ≥ 1.40

## Measured L3→L4 Throughput (1× H100 80GB HBM3, 132 SMs)

| Parents | Kernel Time | Throughput | Raw Survivors |
|---------|------------|------------|---------------|
| 100 | 32s | 3.1/s | 21M |
| 1,000 | 33s | 30/s | 196M |
| 10,000 | 154s | **65 parents/s** | 2.01 billion |

## Time Estimate for 700M Parents at L3→L4

| GPUs | Time |
|------|------|
| 1× H100 | **~125 days** |
| 8× H100 | ~16 days |
| 64× H100 | ~2 days |

**Bottom line: 65 parents/second is too slow.** The kernel works correctly but needs to be faster by 10-100× to be practical on a reasonable GPU budget.

---

## Corrected Survivor Counts (c_target=1.4, m=20, n_half=2)

The CLAUDE.md table is outdated. Actual numbers from GPU runs:

| Level | d_child | Parents In | Unique Survivors | Raw GPU Output |
|-------|---------|------------|-----------------|----------------|
| L0→L1 | 8 | 467 | 167,313 | 169,918 |
| L1→L2 | 16 | ~170K | ~7.5M | 433M (58× duplication) |
| L2→L3 | 32 | ~7.5M | ~500M (user-reported) | unknown |
| L3→L4 | 64 | ~500-700M | billions (user-reported) | 2B per 10K parents |

---

## Why 65 Parents/Second is Slow

Each L3→L4 parent has d_parent=32, d_child=64. With x_cap=2, a typical parent with 16 active positions produces 2^16 = 65,536 children. Each child requires:
1. Incremental conv update: O(d_child) = 64 operations across 2 warps
2. Quick-check: warp-cooperative sum over ~32 conv values
3. Full window scan (if not quick-killed): prefix sum + 127 ell values × multiple windows

At 65 parents/s × 65K children/parent = ~4.2M children/s across 660 blocks. That's ~6,400 children/s per block, or ~10 μs per child. The window scan dominates.

## Performance Improvement Opportunities

### 1. Subtree Pruning (J_MIN=7) — potentially 10-100× speedup

The subtree pruning code exists but barely triggers at d_child=32 (few parents have >7 active positions). At d_child=64 with 16+ active positions, it SHOULD trigger frequently but needs verification. Check: are most L3 parents hitting the subtree prune path? If not, J_MIN may need tuning for d=64.

From the trace run: parent 0 had `n_active=16, expected=65536, skipped=0`. **Zero subtree prunes despite 16 active positions.** This means subtree pruning is not triggering effectively. The partial window scan may not be finding kills because the fixed prefix is too small relative to d_child=64.

**Action:** Profile which parents have subtree pruning, and investigate whether J_MIN=7 is optimal for d=64 or if the partial conv check is too conservative.

### 2. Warp Utilization — 2× potential

At d_child=64, we launch 64 threads (2 warps) per block. The incremental conv update and window scan use all 64 threads. But the Gray code advance, subtree pruning, and child-bin updates are lane-0 only. With 2 warps, 62 of 64 threads are idle during these serial phases.

### 3. Quick-Check Hit Rate

The quick-check reuses the previous killing window. At d=32, it kills ~85% of children. At d=64, the hit rate may be different. If it's lower, every child falls through to the expensive full window scan.

### 4. Window Scan Optimization

The full window scan iterates 127 ell values, each with up to ~127 window positions. This is O(ell_count × max_windows / blockDim) per child. At d=64, ell_count=127 and max_windows can be large. The scan exits early on first kill, but if kills happen late in the ell order, it's slow.

**Action:** Profile which ell values are killing at d=64 and reorder `build_ell_order()` accordingly. The current profile-guided order is tuned for d=32.

### 5. Multiple Children Per Gray Code Step

Currently each Gray code step produces 1 child. Some GPU architectures benefit from processing multiple children simultaneously (e.g., advancing the Gray code on lane 0 while other lanes finish scanning the previous child). This is a major architectural change.

---

## Current Limitations

### 1. Duplicate Survivors (all levels)

The GPU kernel does not deduplicate. Raw output is 58× larger than unique at L1→L2 and ~200K× at L3→L4. This makes single-run output infeasible for levels with many survivors.

**Solution needed:** Chunked processing wrapper or on-GPU sort+unique (Thrust).

### 2. Survivor Buffer OOM

`max_survivors` allocates `N × d_child × 4` bytes in GPU global memory. For L3→L4 with 50M cap: 50M × 64 × 4 = 12.8 GB. This crashed (likely competing with PyTorch docker overhead). 1M cap (256 MB) worked fine but truncates output.

With chunked processing, this is manageable — each chunk uses a small buffer.

### 3. Throughput Too Low for Production

65 parents/s means 700M parents takes 125 days on 1 H100. Even with 64 H100s, that's 2 days of GPU time (~$12,000 at RunPod rates). Algorithmic improvements (better subtree pruning, tuned ell order, higher quick-check hit rate) could reduce this by 10-100×.

### 4. CLAUDE.md Table is Wrong

The survivor counts and CPU timings in CLAUDE.md do not match actual runs with c_target=1.4. The table appears to be for c_target=1.30 or an older code version. Must be updated.

---

## What Needs to Happen Next

**Priority 1: Profile the kernel at d=64**
- Why is subtree pruning not triggering (0 skipped out of 65K)?
- What is the quick-check hit rate?
- Which ell values are killing at d=64?
- How much time is in conv update vs window scan vs other?

**Priority 2: Optimize for d=64**
- Tune J_MIN and partial window scan for d=64
- Re-profile ell kill rates and rebuild ell_order for d=64
- Consider hierarchical window scan (E4 from alternatives doc)

**Priority 3: Implement chunked processing**
- Write `run_chunked.py` for dedup across chunks
- Or add Thrust sort+unique to cascade_host.cu

**Priority 4: Multi-GPU**
- Trivially parallel: partition parents across GPUs
- Pre-sorted round-robin (J1 from alternatives doc)

---

## Files

| File | Status | Purpose |
|------|--------|---------|
| `gpu/cascade_kernel.cu` | Done, needs d=64 optimization | CUDA kernel |
| `gpu/cascade_host.cu` | Done | Host driver |
| `gpu/cascade_kernel.h` | Done | Header |
| `gpu/build.sh` | Done | Linux build (auto-detects arch) |
| `gpu/run.sh` | Done | Single-run convenience script |
| `gpu/run_chunked.py` | **TODO** | Chunked processing with dedup |
| `gpupod/` | Done | RunPod H100 pod management |
