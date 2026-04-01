# 13. Detailed Data Layout & Memory Map

## Global Memory Layout

```
parents[num_parents × d_parent]     int32, row-major, read-only
lo_arrays[num_parents × d_parent]   int32, row-major, read-only
hi_arrays[num_parents × d_parent]   int32, row-major, read-only
threshold_table[max_ell × (m+1)]    int64, row-major, read-only (also copied to smem)
ell_order[max_ell]                  int32, read-only (also copied to smem)
survivors[max_survivors × d_child]  int32, row-major, write-only
survivor_count                      int32, atomic counter

// Precomputed per-parent data (optional, saves per-block recomputation):
parent_prefix[num_parents × (d_parent+1)]  int64, read-only
total_children[num_parents]                 int64, read-only (for load balancing)
```

## Shared Memory Layout (per block, d_child=64)

```
Offset 0:      parent[32]              128B   int32
Offset 128:    child[64]               256B   int32
Offset 384:    raw_conv[127]           508B   int32
Offset 892:    prefix_c[65]            520B   int64  (built on demand)
Offset 1412:   cursor[32]              128B   int32
Offset 1540:   lo_arr[32]              128B   int32
Offset 1668:   hi_arr[32]              128B   int32
Offset 1796:   gc_a[32]                128B   int32
Offset 1924:   gc_dir[32]              128B   int32
Offset 2052:   gc_focus[33]            132B   int32
Offset 2184:   active_pos[32]          128B   int32
Offset 2312:   radix[32]               128B   int32
Offset 2440:   ell_order[63]           252B   int32
Offset 2692:   threshold_table[63×21]  10,584B int64
Offset 13276:  surv_buf[64×64]         16,384B int32 (64 staging slots)
Offset 29660:  surv_count              4B     int32
Offset 29664:  misc (qc_ell, etc.)     64B
                                       ─────────
Total:                                 ~29.7KB
```

**Fits easily in 228KB shared memory.** Can run up to 7 blocks/SM (29.7KB × 7 = 208KB < 228KB).
At 64 threads/block × 7 blocks/SM = 448 threads/SM (22% occupancy).

**To increase occupancy:** Reduce shared memory usage by moving threshold_table to constant memory or L2.
Without threshold_table: ~19KB/block → 11 blocks/SM → 704 threads/SM (34% occupancy).

**Alternative: 32 threads/block (one warp for d=64 too, with loop):**
- Halves shared memory per block
- Doubles blocks/SM potential
- But inner loop is 2x longer per thread

## Register Usage Estimate

```
Per thread (64 threads/block):
  lane          1 reg
  parent_idx    1 reg
  pos, k1, k2  3 regs
  old1, old2    2 regs
  new1, new2    2 regs
  delta1, delta2 2 regs
  cj            1 reg
  ws, ws_qc     2 regs (int64 = 2 regs each = 4)
  dyn_it        2 regs (int64)
  pruned, quick_killed  2 regs
  loop counters 4 regs
  temps         4 regs
  ─────────
  ~28 regs/thread
```

At 28 regs × 64 threads × 7 blocks = 12,544 regs/SM out of 65,536. **Ample headroom.**

---

# 14. Potential Pitfalls & Mitigations

| Pitfall | Risk | Mitigation |
|---------|------|------------|
| Gray code visits element twice/skips one | Proof invalid | Verify total_tested == product(ranges); run exhaustive test for small cases |
| Integer overflow in conv accumulation | Wrong pruning | m ≤ 200 → max conv value ~80K, fits int32. Add debug assertions. |
| Float64 threshold rounding | Miss survivor | Use precomputed int64 threshold table (eliminates runtime float64) |
| Shared memory race condition | Wrong conv values | Careful analysis of write patterns; use `__syncwarp()` / `__syncthreads()` at correct points |
| Warp divergence in window scan | Performance loss | Use `__ballot_sync` + early exit; all lanes exit together |
| Survivor buffer overflow | Lost survivors | Track count; if overflow, re-run parent with larger buffer (same as CPU) |
| Atomic contention on survivor_count | Performance loss | Block-level batching (flush 64 at a time, not 1 at a time) |
| Parent load imbalance across GPUs | Idle GPUs | Pre-sort by estimated children, round-robin assign |
| Memory limits (80GB) | OOM | Parents: 147M × 128B = 18.8GB. Survivors buffer: pre-allocate based on estimated rate. Total: ~25GB fits easily. |
| Threshold table size at larger m | Shared memory overflow | For m > 200: use int64 conv, threshold table in L2 cache |

---

# 15. Compilation & Build

```bash
# CUDA compilation for H100 (SM 9.0)
nvcc -arch=sm_90 -O3 \
     --use_fast_math=false \      # CRITICAL: no fast math for correctness
     -ftz=false \                  # no flush-to-zero
     -prec-div=true \              # precise division
     -prec-sqrt=true \             # precise sqrt
     -Xptxas -dlcm=ca \           # cache all loads in L1
     -maxrregcount=32 \            # limit registers for occupancy
     -I/path/to/cub \
     cascade_kernel.cu -o cascade_kernel

# For multi-GPU:
# Link with NCCL for inter-GPU communication (optional)
# Link with CUDA runtime, driver API
```

## Python Integration

```python
# ctypes wrapper for calling CUDA kernel from Python
import ctypes
lib = ctypes.CDLL('./cascade_kernel.so')

# Or use pybind11/nanobind for cleaner interface
# Or CuPy's RawKernel for direct .cu compilation
```
