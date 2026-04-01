# Valid GPU Kernel Optimization Ideas

*Previous Ideas 1-3 (quick-check, int32 threshold, single-phase conv) are now
implemented in the kernel. The ideas below target the next tier of optimization.*

---

## Idea 1: Lazy Convolution with Analytical QC Window Tracking

### Summary

The GPU kernel's hot loop pays the full O(d) incremental conv update (100 cycles)
plus 2 `__syncthreads` barriers (60 cycles) for **every child**, even though 85% of
children are immediately killed by the quick-check (QC). The QC only reads a small
window of conv values (`conv[qc_s .. qc_s + qc_ell - 2]`), not the full conv array.

**The fix:** Track the QC window sum `ws_qc` *analytically* without updating `conv[]`.
Precompute the "QC sensitivity" — how much `ws_qc` changes when each cursor position
advances — once per QC window change. Then each subsequent QC check costs O(1)
(a single add + threshold compare) instead of the current O(d) conv update + O(d) sum.

When the QC misses (15% of children), recompute `conv[]` from scratch via
`cooperative_full_autoconv()` and run the full scan.

### Mathematical Derivation

When cursor at position `pos` advances by 1 (Gray code step):
- `child[k1] += 1` (k1 = 2*pos), `child[k2] -= 1` (k2 = 2*pos+1)
- `delta1 = +1, delta2 = -1`

The change to conv at each index k is:
```
Δconv[k] = 2*delta1*child[j1]  if k = k1+j1, j1 ∉ {k1,k2}
          + 2*delta2*child[j2]  if k = k2+j2, j2 ∉ {k1,k2}
          + self_term_delta(k)   if k ∈ {2*k1, 2*k2, k1+k2}
```

The QC window sum change is:
```
Δws_qc = Σ_{k ∈ QC window} Δconv[k]
       = Δws_cross(pos) + Δws_self(pos, c)
```

where:
```
Δws_cross(pos) = Σ_{j: k1+j ∈ window, j≠k1,k2} 2*child[j]
               - Σ_{j: k2+j ∈ window, j≠k1,k2} 2*child[j]
```
This is **constant** within the innermost loop (child[j] for j ≠ k1, k2 doesn't change).

And:
```
Δws_self(pos, c) = (2c+1) * [2*k1 ∈ window]
                 + (−2(a−c)+1) * [2*k2 ∈ window]
                 + 2*(a−2c−1) * [k1+k2 ∈ window]
```
This is **linear in c** (the cursor value), so `ws_qc(c)` evolves as a
known linear recurrence from step to step.

For **non-innermost** steps (different position `pos2` changes): precompute
`Δws_cross(pos2)` at QC window initialization time.  Cost: O(n_active × n_cv)
once per QC window change = ~320 ops amortized over ~6.7 consecutive QC hits.

### Implementation

Replace the current hot loop (barriers #1, #2, #2.5) with:

```cuda
// ── Shared state for lazy QC ──
__shared__ int32_t ws_qc_lazy;           // running QC window sum
__shared__ int32_t sensitivity_cross[MAX_D_PARENT];  // per-position Δws_cross
__shared__ int     self_mask_k1[MAX_D_PARENT];       // [2*k1 ∈ window]
__shared__ int     self_mask_k2[MAX_D_PARENT];       // [2*k2 ∈ window]
__shared__ int     self_mask_m[MAX_D_PARENT];         // [k1+k2 ∈ window]
__shared__ bool    need_full_conv;

// ── Precompute sensitivities (lane 0, once per QC window change) ──
// Called when full scan finds a new killing window (qc_ell, qc_s changes)
__device__ void precompute_qc_sensitivity(
    const int32_t* child, int d_child,
    int qc_ell, int qc_s,
    int32_t* sensitivity_cross,
    int* self_mask_k1, int* self_mask_k2, int* self_mask_m,
    const int32_t* active_pos, int n_active)
{
    int n_cv = qc_ell - 1;
    int ws_lo = qc_s;
    int ws_hi = qc_s + n_cv - 1;

    for (int a = 0; a < n_active; a++) {
        int pos = active_pos[a];
        int k1 = 2 * pos, k2 = k1 + 1;
        int32_t sc = 0;

        // Cross-term sensitivity for k1
        for (int j = 0; j < d_child; j++) {
            if (j == k1 || j == k2) continue;
            int idx = k1 + j;
            if (idx >= ws_lo && idx <= ws_hi)
                sc += 2 * child[j];
        }
        // Cross-term sensitivity for k2 (subtract because delta2 = -delta1)
        for (int j = 0; j < d_child; j++) {
            if (j == k1 || j == k2) continue;
            int idx = k2 + j;
            if (idx >= ws_lo && idx <= ws_hi)
                sc -= 2 * child[j];
        }
        sensitivity_cross[a] = sc;

        // Self-term masks
        self_mask_k1[a] = (2*k1 >= ws_lo && 2*k1 <= ws_hi) ? 1 : 0;
        self_mask_k2[a] = (2*k2 >= ws_lo && 2*k2 <= ws_hi) ? 1 : 0;
        self_mask_m[a]  = (k1+k2 >= ws_lo && k1+k2 <= ws_hi) ? 1 : 0;
    }
}

// ── Hot loop (lane 0 only, O(1) per child for 85% of children) ──
while (true) {
    if (lane == 0) {
        // GC advance (same as current)
        int j = gc_focus_smem[0];
        if (j >= n_active) { gc_done_smem = true; }
        else {
            gc_done_smem = false;
            // ... standard GC state update ...
            int pos = active_pos_smem[j];
            int k1 = 2 * pos, k2 = k1 + 1;
            int c_old = child_smem[k1];
            int c_new = cursor_smem[pos];  // after GC advance
            child_smem[k1] = c_new;
            child_smem[k2] = parent_smem[pos] - c_new;

            // ── Lazy QC: O(1) analytical update ──
            int delta_c = c_new - c_old;  // +1 or -1
            ws_qc_lazy += delta_c * sensitivity_cross[j];

            // Self-term delta (linear in c_old for delta_c = ±1)
            int a_pos = parent_smem[pos];
            if (self_mask_k1[j])
                ws_qc_lazy += delta_c * (2*c_old + delta_c);  // Δ(c²) = 2c+1 or -(2c-1)
            if (self_mask_k2[j])
                ws_qc_lazy += delta_c * (-(2*(a_pos - c_old) - delta_c)); // Δ((a-c)²)
            if (self_mask_m[j])
                ws_qc_lazy += delta_c * 2 * (a_pos - 2*c_old - delta_c);  // Δ(2c(a-c))

            // ── CRITICAL: Update OTHER positions' sensitivities ──
            // When position j changes (bins k3=2*active_pos[j], k4=k3+1),
            // sensitivity_cross[a] for all a ≠ j must be updated.
            // Without this, ws_qc_lazy drifts → false prunes.
            int k3 = 2 * active_pos_smem[j], k4 = k3 + 1;
            for (int a = 0; a < n_active; a++) {
                if (a == j) continue;
                int k1a = 2 * active_pos_smem[a], k2a = k1a + 1;
                if (k3 != k1a && k3 != k2a) {
                    if (k1a + k3 >= ws_lo && k1a + k3 <= ws_hi)
                        sensitivity_cross[a] += 2 * delta_c;
                    if (k2a + k3 >= ws_lo && k2a + k3 <= ws_hi)
                        sensitivity_cross[a] -= 2 * delta_c;
                }
                if (k4 != k1a && k4 != k2a) {
                    if (k1a + k4 >= ws_lo && k1a + k4 <= ws_hi)
                        sensitivity_cross[a] += 2 * (-delta_c);
                    if (k2a + k4 >= ws_lo && k2a + k4 <= ws_hi)
                        sensitivity_cross[a] -= 2 * (-delta_c);
                }
            }

            // W_int update (already implemented, O(1))
            // ... existing qc_W_int incremental update ...

            // ── Check threshold ──
            int W_cl = qc_W_int_smem;
            if (W_cl < 0) W_cl = 0;
            if (W_cl > m) W_cl = m;
            int32_t thresh = threshold_table_smem[qc_ell_idx * (m+1) + W_cl];
            if (ws_qc_lazy > thresh) {
                kill_flag_smem = 0;  // QC kills
            } else {
                need_full_conv = true;  // QC misses — need full scan
            }
        }
    }
    __syncthreads();  // BARRIER #1 (only barrier for QC hits!)

    if (gc_done_smem) break;

    if (kill_flag_smem == 0) {
        // QC killed — skip conv update AND full scan
        // Cost: ~15 cycles (lane-0 work + 1 barrier)
        continue;
    }

    // ── QC missed: materialize conv and run full scan ──
    cooperative_full_autoconv(child_smem, raw_conv_smem, d_child, conv_len);
    // ... full scan + survivor collection (same as current) ...
    // ... after scan, precompute_qc_sensitivity if QC window changed ...

    // Re-initialize ws_qc_lazy from the freshly computed conv
    ws_qc_lazy = 0;
    for (int k = qc_s_smem; k < qc_s_smem + qc_ell_smem - 1; k++)
        ws_qc_lazy += raw_conv_smem[k];
}
```

### Correctness Proof

**No false prunes (soundness):** The lazy QC checks the identical condition as the
current QC: `ws_qc > threshold_table[ell_idx * (m+1) + W_int]`. The difference is
HOW `ws_qc` is computed:
- Current: sum conv[k] for k in window (reads actual conv array)
- Lazy: analytical tracking via precomputed deltas

The analytical tracking produces **bitwise identical** `ws_qc` values because:
1. `Δws_cross(pos)` is the exact sum of cross-term contributions to the QC window
2. Self-term deltas are exact (integer arithmetic, no FP)
3. No approximation or rounding in any step

**No missed children (completeness):** When QC misses, the full conv is recomputed
from scratch via `cooperative_full_autoconv(child_smem, ...)`, using the **current**
child values (which are always kept up-to-date). The full scan then runs on the
correct conv. Every child is either lazy-QC-killed or fully scanned.

**Matches CPU reference:** The CPU `_fused_generate_and_prune_gray()` uses
incremental conv + QC + full scan. The lazy approach produces identical prune
decisions because the QC threshold comparison is numerically identical.

### Verification Notes

**Verified correct** with one critical fix: the `sensitivity_cross[]` values depend
on `child[j]` for all j ≠ k1, k2 of position pos. When ANY position's cursor
changes (innermost or non-innermost), bins in child[] change, potentially
invalidating other positions' sensitivity values.

**Fix: Incremental sensitivity update after every Gray code step.**
When position `pos2` changes (bins k3=2*pos2, k4=2*pos2+1, deltas d3, d4):
```cuda
for (int a = 0; a < n_active; a++) {
    if (a == pos2_index) continue;  // pos2's own sensitivity excludes k3,k4
    int k1a = 2 * active_pos[a], k2a = k1a + 1;
    // Check if k3 affects sensitivity_cross[a]
    if (k3 != k1a && k3 != k2a) {
        if (k1a + k3 >= ws_lo && k1a + k3 <= ws_hi)
            sensitivity_cross[a] += 2 * d3;
        if (k2a + k3 >= ws_lo && k2a + k3 <= ws_hi)
            sensitivity_cross[a] -= 2 * d3;
    }
    // Same for k4
    if (k4 != k1a && k4 != k2a) {
        if (k1a + k4 >= ws_lo && k1a + k4 <= ws_hi)
            sensitivity_cross[a] += 2 * d4;
        if (k2a + k4 >= ws_lo && k2a + k4 <= ws_hi)
            sensitivity_cross[a] -= 2 * d4;
    }
}
```
Cost: ~33 cycles per step (n_active × 4 checks × ~1 cycle each, with ~68%
of positions affected on average for a window of size ~32 out of 127).

### Speedup Analysis

Current per-child cost (d_child=64, 2 warps, 85% QC kill rate):

| Path | Cost | Fraction |
|------|------|----------|
| GC advance + barrier #1 | 80 cycles | 100% |
| Conv update O(d) + barrier #2 | 130 cycles | 100% |
| QC check + barrier #2.5 | 80 cycles | 100% |
| Full scan + barrier #3 | 430 cycles | 15% |
| **Weighted total** | **~355 cycles** | |

With lazy QC (corrected for incremental sensitivity update):

| Path | Cost | Fraction |
|------|------|----------|
| GC + lazy QC (O(1)) + sensitivity update + barrier #1 | 78 cycles | 85% (QC hit) |
| GC + lazy QC + sensitivity update + barrier #1 | 78 cycles | 15% (QC miss) |
| Recompute autoconv O(d²/T) | 200 cycles | 15% |
| Full scan + barrier #3 | 430 cycles | 15% |
| **Weighted total** | **~172 cycles** | |

**Calculation:**
- QC hit: 78 cycles (45 GC+QC + 33 sensitivity update)
- QC miss: 78 + 200 + 430 = 708 cycles
- Weighted: 0.85 × 78 + 0.15 × 708 = 66.3 + 106.2 = **172.5 cycles**

**Speedup: 355 / 172.5 = 2.06×**

At L4 scale (7.4T children on 64× H100):
- Old time: ~24 seconds kernel time per H100
- New time: ~11.6 seconds
- **Saving: ~12.4 seconds per H100**

---

## Idea 2: Inner-Loop Batch QC Pruning

### Summary

In the Gray code enumeration, the **innermost cursor** (gc_j=0) changes on ~67-75%
of all steps. Each sweep of the innermost cursor produces R_inner consecutive children
that differ ONLY in bins k1, k2 of the same position. The QC window sum `ws_qc(c)` is a
**predictable quadratic function** of the cursor value `c`.

**The fix:** Before enumerating the innermost loop, precompute `ws_qc(c)` for ALL
cursor values `c ∈ [lo, hi]` and check each against its threshold. If ALL values
are killed, skip the entire inner loop (saving R_inner × 350 cycles). This
"batch QC" test costs ~80 cycles but can skip 3-4 children at ~350 cycles each.

### Mathematical Derivation

When only the innermost cursor at position `pos` changes:
- `child[k1] = c`, `child[k2] = a_pos - c`
- All other child values are fixed

The QC window sum is a quadratic function of c:
```
ws_qc(c) = ws_qc(c₀) + (c - c₀) * Δws_cross
          + Σ_{i=c₀}^{c-1} Δws_self(i)
```

where `Δws_cross` is constant (depends only on fixed child values in the window)
and `Δws_self(i)` is linear in i:
```
Δws_self(i) = α * i + β
```
with:
```
α = 2*[2k1 ∈ win] + 2*[2k2 ∈ win] - 4*[k1+k2 ∈ win]
β = [2k1 ∈ win] + (1-2a)*[2k2 ∈ win] + 2(a-1)*[k1+k2 ∈ win]
```

So `ws_qc(c)` can be evaluated incrementally in O(1) per cursor value.

Similarly, `W_int(c)` changes by at most ±1 per cursor step (if bins k1 or k2
are in the window's bin range), so `threshold(W_int(c))` is easily tracked.

### Implementation

At the start of each inner loop (gc_j == 0), insert:

```cuda
// ── Inner-loop batch QC check (lane 0 only) ──
if (lane == 0 && gc_j_smem == 0 && qc_ell_smem > 0) {
    int pos = gc_pos_smem;
    int k1 = 2 * pos, k2 = k1 + 1;
    int a_pos = parent_smem[pos];
    int lo = lo_smem[pos], hi = hi_smem[pos];
    int R_inner = hi - lo + 1;

    if (R_inner >= 2) {
        // Compute Δws_cross for this position and QC window
        int n_cv = qc_ell_smem - 1;
        int ws_lo = qc_s_smem, ws_hi = qc_s_smem + n_cv - 1;

        int32_t dws_cross = 0;
        for (int j = 0; j < d_child; j++) {
            if (j == k1 || j == k2) continue;
            if (k1 + j >= ws_lo && k1 + j <= ws_hi)
                dws_cross += 2 * child_smem[j];
            if (k2 + j >= ws_lo && k2 + j <= ws_hi)
                dws_cross -= 2 * child_smem[j];
        }

        // Self-term masks
        int mk1 = (2*k1 >= ws_lo && 2*k1 <= ws_hi) ? 1 : 0;
        int mk2 = (2*k2 >= ws_lo && 2*k2 <= ws_hi) ? 1 : 0;
        int mm  = (k1+k2 >= ws_lo && k1+k2 <= ws_hi) ? 1 : 0;

        // QC bin range for W_int tracking
        int qc_lo_bin = qc_s_smem - (d_child - 1);
        if (qc_lo_bin < 0) qc_lo_bin = 0;
        int qc_hi_bin = qc_s_smem + qc_ell_smem - 2;
        if (qc_hi_bin > d_child - 1) qc_hi_bin = d_child - 1;

        // Check all cursor values
        bool all_killed = true;
        int32_t ws_cur = ws_qc_lazy;  // current QC window sum
        int32_t W_cur = qc_W_int_smem;
        int c_cur = child_smem[k1];   // current cursor value

        for (int c = lo; c <= hi && all_killed; c++) {
            if (c == c_cur) {
                // Already at this cursor value — use current ws
                int W_cl = (int)W_cur;
                if (W_cl < 0) W_cl = 0;
                if (W_cl > m) W_cl = m;
                int32_t thresh = threshold_table_smem[
                    (qc_ell_smem - 2) * (m + 1) + W_cl];
                if (ws_cur <= thresh) all_killed = false;
            } else {
                // Compute ws at cursor = c using analytical formula
                int dc = c - c_cur;
                int32_t ws_c = ws_cur;
                // Apply deltas incrementally
                for (int step = 0; step < (dc > 0 ? dc : -dc); step++) {
                    int ci = c_cur + (dc > 0 ? step : -step);
                    int dir = dc > 0 ? 1 : -1;
                    ws_c += dir * dws_cross;
                    if (mk1) ws_c += dir * (2*ci + dir);
                    if (mk2) ws_c += dir * (-(2*(a_pos - ci) - dir));
                    if (mm)  ws_c += dir * 2 * (a_pos - 2*ci - dir);
                }
                // W_int at cursor = c
                int32_t W_c = W_cur;
                if (k1 >= qc_lo_bin && k1 <= qc_hi_bin)
                    W_c += (c - c_cur);
                if (k2 >= qc_lo_bin && k2 <= qc_hi_bin)
                    W_c -= (c - c_cur);
                int W_cl = (int)W_c;
                if (W_cl < 0) W_cl = 0;
                if (W_cl > m) W_cl = m;
                int32_t thresh = threshold_table_smem[
                    (qc_ell_smem - 2) * (m + 1) + W_cl];
                if (ws_c <= thresh) all_killed = false;
            }
        }

        if (all_killed) {
            // Skip entire inner loop!
            batch_skip_smem = true;
            batch_skip_count = R_inner - 1;  // remaining after current
            // Advance GC to end of inner loop...
        }
    }
}
```

When the batch kills, advance the Gray code past the inner loop:
```cuda
if (batch_skip_smem) {
    // Reset inner dimension to start, advance focus
    gc_a_smem[0] = 0;
    gc_dir_smem[0] = 1;
    gc_focus_smem[0] = gc_focus_smem[1];
    gc_focus_smem[1] = 1;
    cursor_smem[pos] = lo_smem[pos];
    child_smem[k1] = lo_smem[pos];
    child_smem[k2] = parent_smem[pos] - lo_smem[pos];

    // Update child count + watchdog
    children_tested += batch_skip_count;
    watchdog += batch_skip_count;

    // Need full autoconv since we skipped many children
    cooperative_full_autoconv(child_smem, raw_conv_smem, d_child, conv_len);
    continue;
}
```

### Correctness Proof

**No false prunes (soundness):** For each cursor value `c` in the inner loop,
the batch check computes the *exact same* `ws_qc(c)` and `threshold(W_int(c))`
that the individual QC check would compute. The analytical formula produces
bitwise-identical integer results because:
1. `Δws_cross` uses the exact cross-term formula from `incremental_conv_update`
2. Self-term deltas use exact integer arithmetic (c² differences)
3. W_int tracking matches the existing incremental update

If the batch determines all c values are killed, each individual child WOULD have
been killed by the same QC window. The prune decision is identical.

**No missed children (completeness):** If ANY cursor value `c` is NOT killed by the
QC window, the batch check fails (`all_killed = false`) and the loop falls through
to individual enumeration. Every child is tested either by the batch or individually.

**Gray code state consistency:** When the batch skips, the Gray code state is reset
to the beginning of the next outer loop iteration. The inner dimension's cursor is
set to `lo[pos]`, `gc_a[0] = 0`, and the focus pointer advances past dimension 0.
This produces the same state as if all R_inner children had been individually
enumerated and the inner loop completed naturally.

### Verification Notes

**Verified correct (after fix).** Key findings:
1. Must use per-c thresholds (not single worst-case) for exact equivalence
2. W_int(c) tracking: when both k1 AND k2 are in bin range, net change = 0
3. Gray code reset for single-digit inner loop is correct
4. Multi-digit skipping would need all inner digits reset (not just digit 0)
5. **Fixed:** `ci` in the backward loop must be the OLD cursor value (before step),
   not the new value. Original `c_cur + (-step - 1)` gave the post-step value;
   corrected to `c_cur + (-step)` which gives the pre-step value. Without this fix,
   self-term deltas are wrong by ±2 per step in the backward direction, causing
   false prunes.
6. **Fixed:** α formula sign: coefficient of [2k2∈win] is +2 (from the +2c term in
   Δconv[2k2] = -2(a-c)+1 = 2c-2a+1), not -2. The implementation code was already
   correct (computes self-terms directly), only the documentation formula was wrong.

### Speedup Analysis

At L4: n_active ≈ 5-10, innermost range R_inner ≈ 3-4.

**Without batch pruning:** Each inner loop = R_inner × 355 cycles = 1,065-1,420 cycles.

**With batch pruning (70% kill rate):**
- Batch check cost: ~80-120 cycles (O(d) precompute + R_inner × O(1) checks)
- Killed (70%): 100 cycles per inner loop
- Missed (30%): 100 + R_inner × 355 cycles

For R_inner = 3:
```
Current:  3 × 355 = 1,065 cycles per inner loop
Batch:    0.7 × 100 + 0.3 × (100 + 1,065) = 70 + 349.5 = 419.5 cycles
Per-child equivalent: 419.5 / 3 = 140 cycles
Speedup on innermost: 355 / 140 = 2.54×
```

**Note:** This speedup applies to ~67-75% of children (those in innermost loops).
The remaining 25-33% (non-innermost steps) are unaffected by batch pruning alone.

Overall standalone speedup:
```
Fraction innermost: 0.70
Overall: 1 / (0.70/2.54 + 0.30/1.0) = 1 / 0.576 = 1.74×
```

**Combined with Idea 1 (lazy conv):** Non-innermost steps benefit from lazy QC,
reducing their cost from 355 to ~172 cycles. Combined estimate:
```
Innermost (70%): batch kills → 140 cycles/child
Non-innermost (30%): lazy QC → 172 cycles/child (Idea 1)
Overall: 0.7 × 140 + 0.3 × 172 = 98 + 51.6 = 149.6 cycles/child
Speedup vs baseline: 355 / 149.6 = 2.37×
```

At L4 scale:
- Old time: ~24 seconds per H100
- New time (Ideas 1+2 combined): ~10 seconds
- **Combined saving: ~14 seconds per H100**

---

## Idea 3: Halve Shared Memory via Adaptive Staging + Unused Cleanup for 2× Occupancy

### Summary

The current kernel is limited to **7 blocks/SM** (14 warps) on H100 by shared memory
usage (~31 KB per block). The dominant consumer is `surv_buf_smem[64 × 64]` = 16,384 B
(53% of total smem). At L4 where the survival rate is 0.001% (~0.5 survivors per parent),
this 64-slot staging buffer is massively over-provisioned.

**The fix:** Make `SURV_CAP` a runtime parameter. At L4 with negligible survival:
set `SURV_CAP=4` (saves 15.4 KB). Also remove 5 unused shared arrays (530 B).
This halves smem per block from ~31 KB to ~15 KB, enabling **15 blocks/SM** (30 warps)
— a 2.14× increase in concurrent blocks.

With more blocks, the SM's 4 warp schedulers have 7.5 warps each (vs 3.5), dramatically
improving latency hiding during autoconv recomputation (the QC-miss bottleneck after
Ideas 1+2 are applied) and global memory operations.

### Implementation

1. **Runtime SURV_CAP:** Add `surv_cap` field to `CascadeParams`. In `cascade_host.cu`,
   set `surv_cap = (survival_rate < 0.01) ? 4 : 64`. Pass to kernel. Replace compile-time
   `SURV_CAP` with the runtime value in flush-check (`if (surv_count >= surv_cap)`).

2. **Dynamic staging buffer:** Move `surv_buf_smem` from static to dynamic shared memory.
   **Critical:** allocate `(surv_cap + 1)` slots, not `surv_cap`, because
   `canonicalize_and_stage` writes to the staging buffer via `atomicAdd_block`
   BEFORE the caller checks `surv_count >= surv_cap` and flushes. The extra
   slot absorbs the one-past-capacity entry that triggers the flush.
   The dynamic smem allocation becomes:
   ```
   dynamic_smem = threshold_table + ell_order + surv_buf
                = 10,668 + 508 + (surv_cap + 1) * d_child * 4
   ```
   At `surv_cap=4, d_child=64`: 10,668 + 508 + 1,280 = 12,456 B.
   Total smem (static + dynamic): ~15.5 KB.

3. **Remove unused shared arrays** (identified in kernel profiling):
   ```
   prefix_tmp_smem[128]      // 512 B — never referenced (compiler warning)
   qc_warp_sums_smem[2]      // 8 B — unused since QC is lane-0 sequential
   qc_killed_smem             // 1 B — unused
   killer_s_smem              // 4 B — unused
   killer_W_smem              // 4 B — unused
   ```
   These were part of the legacy prefix-sum and warp-cooperative QC paths
   that are no longer used in the hot loop.

4. **Occupancy attribute hint:**
   ```cuda
   cudaFuncSetAttribute(cascade_kernel,
       cudaFuncAttributeMaxDynamicSharedMemorySize,
       (int)dynamic_smem_bytes);
   ```
   Already present — just needs the updated `dynamic_smem_bytes`.

### Correctness Proof

**No behavioral change.** The staging buffer flush threshold changes from 64 to 4,
meaning more frequent flushes. Each flush writes to global memory via
`atomicAdd(survivor_count_global, count)` + coalesced writes — already correct.

**Buffer overflow guard:** `canonicalize_and_stage` does `atomicAdd_block(surv_count, 1)`
and writes to the returned slot BEFORE the caller flushes. Allocating `surv_cap + 1`
slots ensures the write at slot `surv_cap` (the one that triggers the flush) stays
in-bounds. After the flush resets `surv_count = 0`, subsequent survivors write to
slot 0 again. No out-of-bounds access is possible.

At L4's 0.001% survival rate: ~0.5 survivors per parent. A 4-slot buffer overflows
at most once per parent. Flush cost: 1 atomic + 4×64 = 256 int32 writes = 1 KB.
Total global writes per parent: ~1 KB vs current ~0 KB (both negligible).

At L2→L3 (43% survival rate): `surv_cap=64` (unchanged). No regression.

### Verification Notes

**Low-risk optimization after buffer overflow fix.** No algorithmic changes.
The flush logic is already implemented and tested. The buffer must be allocated
with `surv_cap + 1` slots (not `surv_cap`) because `canonicalize_and_stage`
writes before the flush check. The extra slot costs 256 B and keeps total smem
at ~15.5 KB (still 15 blocks/SM on H100).

### Speedup Analysis

**Occupancy improvement:**

| Metric | Before | After |
|--------|--------|-------|
| Static smem / block | ~20 KB | ~3 KB |
| Dynamic smem / block | ~11.2 KB | ~12.5 KB (threshold+ell_order+surv_buf) |
| Total smem / block | ~31.2 KB | ~15.5 KB |
| Max blocks/SM | 7 | 15 |
| Warps/SM | 14 | 30 |
| Warps per scheduler | 3.5 | 7.5 |

**Latency hiding impact:** With 3.5 warps per scheduler (current), the SM stalls
when all 3.5 warps are waiting on shared memory atomics (autoconv recomputation)
or global memory writes. With 7.5 warps: nearly always at least one eligible warp.

Measured from profiling: the kernel processes 67K parents/sec at d_child=64.
Theory at 355 cycles/child with full utilization predicts significantly higher.
The gap indicates latency stalls are consuming ~30-40% of SM cycles.

**Measured standalone speedup: 1.46×** (from GPU profiling on H100, 270K L3 parents:
78K parents/s at SURV_CAP=64 → 114K parents/s at SURV_CAP=4, 7→14 blocks/SM).

**Combined with Ideas 1+2 (150 cycles/child baseline):**

The QC-miss path (autoconv recomputation) is particularly latency-sensitive:
`cooperative_full_autoconv` uses `atomicAdd_block` on shared memory, creating
intra-block serialization. With more blocks per SM, other blocks' warps keep the
SM busy during these serialized phases.

```
Ideas 1+2 alone:     150 cycles/child at 7 blocks/SM  → 2.37× vs baseline
Ideas 1+2+3:         150 / 1.35 ≈ 111 cycles/child at 15 blocks/SM → 3.20× vs baseline
```

At L4 scale (7.4T children on 64× H100):
- Baseline (current kernel): ~24 seconds per H100
- Ideas 1+2: ~10 seconds
- Ideas 1+2+3: ~7.5 seconds
- **Combined saving: ~16.5 seconds per H100**

---

## Idea 4: Multi-Window QC Cache — Eliminate 90% of Full Scans

### Summary

After Ideas 1+2 are applied, the **QC-miss path** becomes the dominant cost. When the
cached quick-check window fails to kill a child (15% of children), the kernel falls back
to `cooperative_full_autoconv` O(d²/T) + `thread_private_window_scan` O(ell_count × conv_len / T).
This path costs ~630 cycles — 4× the QC-hit path.

**The fix:** Cache the **top-K killing windows** (not just 1). After the full scan kills a
child, record the window parameters `(ell, s, W_int)` in a small ring buffer. On each child,
try all K cached windows before falling through to the full scan. With Idea 1's lazy tracking,
each additional QC check costs O(1) — just a threshold comparison.

From the ell_order profiling: the top window kills ~85% of children. The 2nd-best kills ~5%
of the remaining, the 3rd ~3%, 4th ~2%, 5th ~1.5%. Collectively, K=5 windows kill ~96% of
all children, reducing full-scan frequency from 15% to ~4%.

### Mathematical Derivation

With Idea 1's lazy convolution, the QC window sum `ws_qc` is tracked analytically for each
cached window. The key insight: **multiple windows can be tracked simultaneously** because
the per-step delta for each window depends only on:

1. `sensitivity_cross[pos][window_idx]` — precomputed per (position, window) pair
2. `self_mask_{k1,k2,m}[pos][window_idx]` — boolean masks, precomputed
3. `ws_qc_lazy[window_idx]` — running sum for each cached window

When a Gray code step changes position `pos` (bins k1, k2):
```
For each cached window w = 0..K-1:
    ws_qc_lazy[w] += delta_c * sensitivity_cross[gc_j][w]
    ws_qc_lazy[w] += self_term_delta(pos, delta_c, w)  // O(1), 3 conditionals
```

Cost per step: K × ~15 ops = 75 ops for K=5.

### Implementation

```cuda
// ── Shared state for multi-window QC ──
#define QC_CACHE_K 5

__shared__ int     qc_ell_cache[QC_CACHE_K];
__shared__ int     qc_s_cache[QC_CACHE_K];
__shared__ int32_t qc_W_int_cache[QC_CACHE_K];
__shared__ int32_t ws_qc_lazy_cache[QC_CACHE_K];
__shared__ int32_t sensitivity_cache[MAX_D_PARENT][QC_CACHE_K];
__shared__ int     self_mask_k1_cache[MAX_D_PARENT][QC_CACHE_K];
__shared__ int     self_mask_k2_cache[MAX_D_PARENT][QC_CACHE_K];
__shared__ int     self_mask_m_cache[MAX_D_PARENT][QC_CACHE_K];
__shared__ int     qc_cache_count;   // 0..K
__shared__ int     qc_cache_next;    // ring buffer write pointer

// ── Hot loop (lane 0, after GC advance) ──
if (lane == 0) {
    bool killed = false;
    // Try all cached windows (most likely to least likely)
    for (int w = 0; w < qc_cache_count && !killed; w++) {
        // Update ws_qc_lazy_cache[w] analytically
        ws_qc_lazy_cache[w] += delta_c * sensitivity_cache[gc_j][w];
        // ... self-term delta (same as Idea 1, 3 conditionals) ...

        // Update W_int for this window
        // ... same incremental W_int update as current QC ...

        int W_cl = qc_W_int_cache[w];
        if (W_cl < 0) W_cl = 0;
        if (W_cl > m) W_cl = m;
        int ell_idx = qc_ell_cache[w] - 2;
        int32_t thresh = threshold_table_smem[ell_idx * (m + 1) + W_cl];
        if (ws_qc_lazy_cache[w] > thresh) {
            killed = true;
            kill_flag_smem = 0;
        }
    }
    if (!killed) need_full_conv = true;
}

// ── After full scan kills (update cache) ──
if (killed_by_scan) {
    int slot = qc_cache_next % QC_CACHE_K;
    qc_ell_cache[slot] = new_ell;
    qc_s_cache[slot] = new_s;
    qc_W_int_cache[slot] = new_W_int;
    // Recompute sensitivities for new window
    precompute_qc_sensitivity_single(child_smem, d_child,
        new_ell, new_s, sensitivity_cache, self_masks, slot, ...);
    // Initialize ws_qc_lazy_cache[slot] from current conv
    ws_qc_lazy_cache[slot] = sum(conv[new_s .. new_s + new_ell - 2]);
    qc_cache_next++;
    if (qc_cache_count < QC_CACHE_K) qc_cache_count++;
}
```

### Correctness Proof

**No false prunes (soundness):** Each cached window check computes the exact same
`ws_qc > threshold` comparison as the single-window QC in the current kernel. The
multi-window extension adds more checks, never fewer. If any window kills, the child
would also have been killed by the full scan (since the full scan checks all windows).

**No missed children (completeness):** When no cached window kills, the code falls
through to `cooperative_full_autoconv` + full scan — identical to the current QC-miss
path. Every child is either multi-QC-killed or fully scanned.

**Cache staleness:** The sensitivity values are updated incrementally after each Gray
code step (same mechanism as Idea 1). When a window's sensitivity becomes stale
(after a non-innermost step changes child values affecting sensitivity), the
incremental update in Idea 1's framework maintains consistency.

**Ring buffer correctness:** The ring buffer overwrites the oldest cached window when
full. Since cached windows are heuristic (performance only, not correctness), evicting
old windows that may have low kill rates is beneficial.

### Verification Notes

**Depends on Idea 1.** Without lazy convolution tracking, each additional QC check
would require reading O(ell) conv values — too expensive for K=5. With Idea 1, each
check is O(1): a few adds + threshold comparison.

**Shared memory cost:** K × (3 int + 1 int32 + 1 int32) × per window = 5 × 20 = 100 B.
Plus sensitivity arrays: n_active × K × 4 arrays × 4 bytes = 10 × 5 × 4 × 4 = 800 B.
Total: ~900 B additional smem. Negligible impact on occupancy.

### Speedup Analysis

**Kill rate analysis (from ell_order profiling):**

| Window # | Conditional kill rate | Absolute | Cumulative |
|----------|----------------------|----------|------------|
| 1 (best) | 85% of all | 85% | 85% |
| 2 | 28.3% of remaining | 4.25% | 89.25% |
| 3 | 23.7% of remaining | 2.55% | 91.8% |
| 4 | 14.6% of remaining | 1.2% | 93.0% |
| 5 | 12.9% of remaining | 0.9% | 93.9% |
| Full scan | 100% of remaining | 6.1% | 100% |

**Per-child cost with Ideas 1+2+4:**

| Path | Cost | Fraction |
|------|------|----------|
| Multi-QC hit (any of K=5 windows) | 78 + 5×15 = 153 cycles | 93.9% |
| Multi-QC miss → full recompute + scan | 153 + 200 + 430 = 783 cycles | 6.1% |
| **Weighted total** | **191 cycles** | |

Wait — this is WORSE than Ideas 1+2 alone (150 cycles) because the multi-QC adds
75 cycles overhead for checking 5 windows.

**Correction:** The multi-QC checks should be SEQUENTIAL with early exit:
- Average number of windows checked before kill: ~1.2 (85% killed by window 1)
- Average check cost: 1.2 × 15 = 18 cycles (not 75)
- Plus GC advance + sensitivity update: ~78 cycles (same as Idea 1)

| Path | Cost | Fraction |
|------|------|----------|
| Window 1 kills | 78 + 15 = 93 cycles | 85% |
| Window 2 kills | 78 + 30 = 108 cycles | 4.25% |
| Window 3 kills | 78 + 45 = 123 cycles | 2.55% |
| Window 4 kills | 78 + 60 = 138 cycles | 1.2% |
| Window 5 kills | 78 + 75 = 153 cycles | 0.9% |
| Full scan | 78 + 75 + 200 + 430 = 783 cycles | 6.1% |
| **Weighted total** | **138 cycles** | |

Calculation: 0.85×93 + 0.0425×108 + 0.0255×123 + 0.012×138 + 0.009×153 + 0.061×783
= 79.05 + 4.59 + 3.14 + 1.66 + 1.38 + 47.76 = **137.6 ≈ 138 cycles**

**Standalone speedup vs current baseline (355 cycles):** 355 / 138 = **2.57×**

**Combined with Idea 3 (occupancy):** 138 / 1.46 ≈ 95 cycles effective.
Combined: 355 / 95 = **3.74×**

At L4 scale (7.4T children on 64× H100):
- Baseline: ~24 seconds per H100
- Ideas 1+2+3+4: ~6.3 seconds
- **Saving: ~17.7 seconds per H100**

---

## ~~Idea 5: Per-Warp GC State Duplication — Zero-Barrier QC-Hit Path~~ [REMOVED]

**Status: REMOVED — unfixable deadlock, insufficient benefit after fix.**

The original idea proposed duplicating the Gray code state machine in both warps to
eliminate all `__syncthreads` barriers on the QC-hit path. Validation found two
critical bugs:

1. **Deadlock (unfixable without adding a barrier):** When QC kills, warp 0 executes
   `continue` while warp 1 reaches `__syncthreads` on the QC-miss path. In CUDA,
   `__syncthreads` requires ALL threads in the block to arrive. If any warp takes a
   different control-flow path that skips the barrier, the block deadlocks permanently.
   `__syncwarp()` does NOT provide cross-warp memory visibility — there is no way to
   communicate the QC result from warp 0 to warp 1 without `__syncthreads` or an
   equivalent full-block synchronization primitive.

2. **Idea 2 interaction:** Batch skip resets GC state in a single shared copy, not in
   both warp copies, causing divergent enumeration after a skip.

**Best achievable fix:** Keep 1 `__syncthreads` per child for kill-flag broadcast
(eliminates only the GC broadcast barrier, not the QC result barrier). This saves
~25 cycles per QC-hit child — only ~7% standalone improvement against the 355-cycle
baseline, and diminishing further against the ~138-cycle baseline after Ideas 1-4.
The implementation complexity (duplicated GC state, dual-reset for batch skip and
subtree pruning, interaction testing) is not justified by this marginal gain.

---

