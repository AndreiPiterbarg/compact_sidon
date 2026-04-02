/*
 * cascade_kernel.cu — CUDA kernel for the Sidon autocorrelation cascade prover.
 *
 * This kernel is computing a RIGOROUS MATHEMATICAL PROOF.  Correctness
 * requirements:
 *   1. No false prunes (soundness).
 *   2. Complete enumeration (completeness).
 *   3. Exact integer arithmetic throughout (no FP in hot path).
 *
 * Target: NVIDIA H100 SXM 80GB (sm_90).
 *
 * Build:
 *   nvcc -arch=sm_90 -O3 -ftz=false -prec-div=true -prec-sqrt=true \
 *        -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu    \
 *        -o cascade_prover
 */

#include "cascade_kernel.h"
#include <cassert>
#include <cstdio>

/* g_next_parent is now passed as a kernel parameter (int32_t* in global
 * memory) rather than a __device__ variable.  This avoids cross-TU
 * symbol issues with cudaMemcpyToSymbol and lets the host monitor
 * progress by reading it with cudaMemcpy. */

/* ═══════════════════════════════════════════════════════════════════
 * Shared-memory layout (declared inside the kernel as a union).
 *
 * The layout matches Section 3.3 of final_architecture.md.
 * We use a struct for clarity; the total must fit in 228 KB / blocks_per_SM.
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * We declare shared memory inside the kernel with explicit arrays
 * rather than a monolithic byte array, for readability and type safety.
 * The template parameter D selects the child dimension (32 or 64).
 */

/* ═══════════════════════════════════════════════════════════════════
 *  cooperative_full_autoconv — O(d^2) initial autoconvolution
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void cooperative_full_autoconv(
    const int32_t* child,
    int32_t*       conv,
    int d_child, int conv_len)
{
    const int lane = threadIdx.x;

    /* Zero conv cooperatively. */
    for (int k = lane; k < conv_len; k += blockDim.x)
        conv[k] = 0;
    __syncthreads();

    /* Distribute the triangular loop across lanes.
     * Each lane handles rows i = lane, lane+blockDim, ... */
    for (int i = lane; i < d_child; i += blockDim.x) {
        int ci = child[i];
        if (ci == 0) continue;
        atomicAdd_block(&conv[2 * i], ci * ci);          /* self-term */
        for (int j = i + 1; j < d_child; j++) {
            int cj = child[j];
            if (cj != 0)
                atomicAdd_block(&conv[i + j], 2 * ci * cj); /* cross-term */
        }
    }
    __syncthreads();
}

/* ═══════════════════════════════════════════════════════════════════
 *  incremental_conv_update — single-phase conflict-free update
 *
 *  When cursor at parent position `pos` changes, child bins
 *  k1=2*pos and k2=2*pos+1 are updated.  Cross-terms with all
 *  other bins are recomputed incrementally.
 *
 *  Each thread j writes ONLY to conv[k1+j], combining both the
 *  delta1 contribution (from child[j]) and the delta2 contribution
 *  (from child[j-1], which maps to conv[k2+(j-1)] = conv[k1+j]).
 *  Since each thread writes to a unique address, no write conflicts
 *  occur and only 1 barrier is needed (was 2 in the two-phase approach).
 *
 *  Thread 0 additionally handles the "extra" address conv[k1+d_child]
 *  = conv[k2+d_child-1] which is the delta2 contribution from
 *  child[d_child-1].
 *
 *  CORRECTNESS: Verified by exhaustive enumeration of all conv index
 *  contributions — produces bitwise identical results to the two-phase
 *  approach.  See valid_ideas.md, Idea 3.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void incremental_conv_update(
    int32_t*       conv,
    const int32_t* child,
    int k1, int k2,
    int old1, int old2, int new1, int new2,
    int d_child, int conv_len)
{
    const int lane = threadIdx.x;
    int delta1 = new1 - old1;
    int delta2 = new2 - old2;
    int idx = k1 + lane;

    if (idx < conv_len && lane <= d_child) {
        int32_t delta_total = 0;

        /* Self-terms at specific indices:
         *   conv[2*k1]   = conv[k1+k1]   → lane == k1
         *   conv[k1+k2]  = conv[k1+k1+1] → lane == k2  (= k1+1)
         *   conv[2*k2]   = conv[k1+k1+2] → lane == k1+2 */
        if (lane == k1)
            delta_total += new1 * new1 - old1 * old1;
        if (lane == k2)           /* k2 = k1+1 */
            delta_total += 2 * (new1 * new2 - old1 * old2);
        if (lane == k1 + 2)
            delta_total += new2 * new2 - old2 * old2;

        /* delta1 cross-term: child[lane] contributes to conv[k1+lane].
         * Excluded when lane == k1 or lane == k2 (handled by self/mutual). */
        if (lane < d_child && lane != k1 && lane != k2) {
            int cj = child[lane];
            if (cj != 0)
                delta_total += 2 * delta1 * cj;
        }

        /* delta2 cross-term: child[lane-1] contributes to
         * conv[k2+(lane-1)] = conv[k1+lane].
         * Excluded when (lane-1) == k1 or (lane-1) == k2. */
        {
            int jm1 = lane - 1;
            if (jm1 >= 0 && jm1 < d_child && jm1 != k1 && jm1 != k2) {
                int cj = child[jm1];
                if (cj != 0)
                    delta_total += 2 * delta2 * cj;
            }
        }

        if (delta_total != 0)
            conv[idx] += delta_total;
    }

    /* Extra address: conv[k1+d_child] = conv[k2+d_child-1].
     * Only the delta2 contribution from child[d_child-1].
     * Handled by lane 0 (which writes to conv[k1+0], a different address). */
    if (lane == 0) {
        int extra_idx = k1 + d_child;
        int jlast = d_child - 1;
        if (extra_idx < conv_len && jlast != k1 && jlast != k2) {
            int cj = child[jlast];
            if (cj != 0)
                conv[extra_idx] += 2 * delta2 * cj;
        }
    }

    __syncthreads();   /* single barrier (was 2 in two-phase approach) */
}

/* ═══════════════════════════════════════════════════════════════════
 *  warp_cooperative_quick_check — retry previous killing window
 *
 *  Returns true if the child is killed by the cached (ell, s, W_int).
 *  Only warp 0 participates in the sum; result is broadcast to all
 *  threads via shared memory.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ __noinline__ bool warp_cooperative_quick_check(
    const int32_t* conv,
    const int32_t* threshold_table,
    int qc_ell, int qc_s, int32_t qc_W_int, int m,
    bool* qc_killed_smem,
    int32_t* qc_warp_sums)          /* [2] in shared mem for multi-warp */
{
    if (qc_ell == 0) return false;

    const int lane = threadIdx.x;
    const int warp_id = lane / WARP_SIZE;
    const int warp_lane = lane % WARP_SIZE;
    int n_cv_qc = qc_ell - 1;

    /* Each thread accumulates conv values strided by blockDim.x
     * (not WARP_SIZE) so all threads cooperate over the full range.
     * int32 safe: max window sum = m^2 = 400 for m=20. */
    int32_t partial = 0;
    for (int k = qc_s + lane; k < qc_s + n_cv_qc; k += (int)blockDim.x)
        partial += conv[k];

    /* Intra-warp reduction. */
    unsigned mask = 0xFFFFFFFF;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(mask, partial, offset);

    /* Multi-warp: combine via shared memory. */
    if (blockDim.x > WARP_SIZE) {
        if (warp_lane == 0)
            qc_warp_sums[warp_id] = partial;
        __syncthreads();
        if (lane == 0) {
            int32_t ws = qc_warp_sums[0];
            for (int w = 1; w < (int)blockDim.x / WARP_SIZE; w++)
                ws += qc_warp_sums[w];
            int ell_idx = qc_ell - 2;
            int W_int_clamped = (int)qc_W_int;
            if (W_int_clamped < 0) W_int_clamped = 0;
            if (W_int_clamped > m) W_int_clamped = m;
            int32_t thresh = threshold_table[ell_idx * (m + 1) + W_int_clamped];
            *qc_killed_smem = (ws > thresh);
        }
        __syncthreads();
    } else {
        /* Single warp: lane 0 already has the total. */
        if (lane == 0) {
            int32_t ws = partial;
            int ell_idx = qc_ell - 2;
            int W_int_clamped = (int)qc_W_int;
            if (W_int_clamped < 0) W_int_clamped = 0;
            if (W_int_clamped > m) W_int_clamped = m;
            int32_t thresh = threshold_table[ell_idx * (m + 1) + W_int_clamped];
            *qc_killed_smem = (ws > thresh);
        }
        __syncthreads();
    }
    return *qc_killed_smem;
}

/* ═══════════════════════════════════════════════════════════════════
 *  precompute_qc_sensitivity — Idea 1: compute analytical QC deltas
 *
 *  For each active position a, compute the cross-term sensitivity
 *  (how much ws_qc changes when that position's cursor advances by 1)
 *  and the self-term masks (which self/mutual conv indices fall in the
 *  QC window).
 *
 *  Lane 0 only.  Called when the QC window changes or conv is recomputed.
 *  For the multi-window cache (Idea 4), w selects which cache slot.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void precompute_qc_sensitivity(
    const int32_t* child, int d_child,
    int qc_ell, int qc_s,
    const int32_t* active_pos, int n_active,
    /* Single-window output (w == -1) or cache slot (w >= 0) */
    int32_t* sens_cross,       /* [MAX_D_PARENT] or [MAX_D_PARENT*K+w] */
    int*     mask_k1,
    int*     mask_k2,
    int*     mask_m,
    int      stride)           /* 1 for single, QC_CACHE_K for cache */
{
    int n_cv = qc_ell - 1;
    int ws_lo = qc_s;
    int ws_hi = qc_s + n_cv - 1;

    for (int a = 0; a < n_active; a++) {
        int pos = active_pos[a];
        int k1 = 2 * pos, k2 = k1 + 1;
        int32_t sc = 0;

        /* Cross-term sensitivity for k1 (delta1 = +1) */
        for (int j = 0; j < d_child; j++) {
            if (j == k1 || j == k2) continue;
            int idx = k1 + j;
            if (idx >= ws_lo && idx <= ws_hi)
                sc += 2 * child[j];
        }
        /* Cross-term sensitivity for k2 (delta2 = -1, subtract) */
        for (int j = 0; j < d_child; j++) {
            if (j == k1 || j == k2) continue;
            int idx = k2 + j;
            if (idx >= ws_lo && idx <= ws_hi)
                sc -= 2 * child[j];
        }
        sens_cross[a * stride] = sc;

        /* Self-term masks: which self/mutual conv indices in window */
        mask_k1[a * stride] = (2 * k1 >= ws_lo && 2 * k1 <= ws_hi) ? 1 : 0;
        mask_k2[a * stride] = (2 * k2 >= ws_lo && 2 * k2 <= ws_hi) ? 1 : 0;
        mask_m[a * stride]  = (k1 + k2 >= ws_lo && k1 + k2 <= ws_hi) ? 1 : 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  update_sensitivities_after_step — Idea 1: incremental update
 *
 *  After position j changes (bins k3=2*pos, k4=k3+1, delta_c = ±1),
 *  update ALL other positions' sensitivity_cross values for a given
 *  cached window.  Without this, ws_qc_lazy drifts → false prunes.
 *
 *  For the multi-window cache (Idea 4), called once per cached window.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void update_sensitivities_after_step(
    int j_changed,            /* active index that changed */
    int k3, int k4,           /* bins: k3=2*pos, k4=k3+1 */
    int delta_c,              /* +1 or -1 */
    int qc_s, int qc_ell,
    const int32_t* active_pos, int n_active,
    int32_t* sens_cross,
    int      stride)
{
    int n_cv = qc_ell - 1;
    int ws_lo = qc_s;
    int ws_hi = qc_s + n_cv - 1;
    int d3 = delta_c;         /* delta for k3 = child[k3] change */
    int d4 = -delta_c;        /* delta for k4 = child[k4] change */

    for (int a = 0; a < n_active; a++) {
        if (a == j_changed) continue;
        int k1a = 2 * active_pos[a], k2a = k1a + 1;

        if (k3 != k1a && k3 != k2a) {
            if (k1a + k3 >= ws_lo && k1a + k3 <= ws_hi)
                sens_cross[a * stride] += 2 * d3;
            if (k2a + k3 >= ws_lo && k2a + k3 <= ws_hi)
                sens_cross[a * stride] -= 2 * d3;
        }
        if (k4 != k1a && k4 != k2a) {
            if (k1a + k4 >= ws_lo && k1a + k4 <= ws_hi)
                sens_cross[a * stride] += 2 * d4;
            if (k2a + k4 >= ws_lo && k2a + k4 <= ws_hi)
                sens_cross[a * stride] -= 2 * d4;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  lazy_qc_update_and_check — Idea 1+4: O(1) analytical QC per child
 *
 *  Updates ws_qc_lazy and W_int for ALL cached windows, checks each
 *  against its threshold with early exit.  Returns true if any window
 *  kills the child.
 *
 *  Lane 0 only.  Called from the hot loop after GC advance + child
 *  update.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ bool lazy_qc_update_and_check(
    int gc_j,                 /* active index that changed */
    int delta_c,              /* +1 or -1 */
    int c_old,                /* old child[k1] value */
    int a_pos,                /* parent[pos] */
    int k3, int k4,           /* bins that changed */
    int d_child, int m,
    const int32_t* active_pos, int n_active,
    const int32_t* threshold_table,
    /* Single-window lazy state */
    int32_t* ws_lazy,
    int* qc_ell_p, int* qc_s_p, int32_t* qc_W_int_p,
    int32_t* sens_cross, int* mask_k1, int* mask_k2, int* mask_m,
    /* Multi-window cache state */
    int32_t* cache_ws_lazy, int* cache_ell, int* cache_s,
    int32_t* cache_W_int,
    int32_t* cache_sens, int* cache_mk1, int* cache_mk2, int* cache_mm,
    int cache_count)
{
    bool killed = false;

    /* === Check ALL cached windows (sequential, early exit) === */
    for (int w = 0; w < cache_count; w++) {
        int qc_ell = cache_ell[w];
        int qc_s   = cache_s[w];
        if (qc_ell == 0) continue;

        /* Analytical ws update */
        cache_ws_lazy[w] += delta_c * cache_sens[gc_j * QC_CACHE_K + w];

        /* Self-term deltas */
        if (cache_mk1[gc_j * QC_CACHE_K + w])
            cache_ws_lazy[w] += delta_c * (2 * c_old + delta_c);
        if (cache_mk2[gc_j * QC_CACHE_K + w])
            cache_ws_lazy[w] += delta_c * (-(2 * (a_pos - c_old) - delta_c));
        if (cache_mm[gc_j * QC_CACHE_K + w])
            cache_ws_lazy[w] += delta_c * 2 * (a_pos - 2 * c_old - delta_c);

        /* W_int incremental update */
        int qc_lo_bin = qc_s - (d_child - 1);
        if (qc_lo_bin < 0) qc_lo_bin = 0;
        int qc_hi_bin = qc_s + qc_ell - 2;
        if (qc_hi_bin > d_child - 1) qc_hi_bin = d_child - 1;
        if (k3 >= qc_lo_bin && k3 <= qc_hi_bin)
            cache_W_int[w] += delta_c;
        if (k4 >= qc_lo_bin && k4 <= qc_hi_bin)
            cache_W_int[w] += (-delta_c);

        /* Threshold check */
        int ell_idx = qc_ell - 2;
        int W_cl = (int)cache_W_int[w];
        if (W_cl < 0) W_cl = 0;
        if (W_cl > m) W_cl = m;
        int32_t thresh = threshold_table[ell_idx * (m + 1) + W_cl];
        if (cache_ws_lazy[w] > thresh) {
            killed = true;
            break;
        }
    }

    if (!killed) {
        /* Update sensitivity for ALL cached windows (even though not killed,
         * the values must stay current for the next child's check). */
    }

    /* Update sensitivities for all cached windows after this GC step.
     * Must happen whether killed or not — keeps state consistent. */
    for (int w = 0; w < cache_count; w++) {
        if (cache_ell[w] == 0) continue;
        update_sensitivities_after_step(
            gc_j, k3, k4, delta_c,
            cache_s[w], cache_ell[w],
            active_pos, n_active,
            &cache_sens[w], QC_CACHE_K);
    }

    return killed;
}

/* ═══════════════════════════════════════════════════════════════════
 *  thread_private_window_scan — barrier-free sliding-window scan
 *
 *  Each thread independently scans a subset of ell values using a
 *  sliding window over the raw conv[] and child[] arrays.  No prefix
 *  sum needed.  Kill detection via atomicMin_block on a shared flag.
 *
 *  This eliminates the 254 __syncthreads per surviving child that
 *  the old prefix-sum approach required (2 barriers per ell × 127
 *  ells for cross-warp reduction).
 *
 *  CORRECTNESS: Checks every (ell, s) pair.  The sliding window
 *  produces the same ws and W_int values as the prefix-sum approach.
 *  W_int update derived from exact bin-range formulas:
 *    lo_bin(s) = max(0, s - d_child + 1)
 *    hi_bin(s) = min(d_child - 1, s + ell - 2)
 *
 *  Caller must init *kill_flag to blockDim.x and __syncthreads
 *  BEFORE calling.  Caller must __syncthreads AFTER return to
 *  read the kill flag.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ __noinline__ bool thread_private_window_scan(
    const int32_t* conv,
    const int32_t* child,
    const int32_t* threshold_table,
    const int32_t* ell_order,
    int ell_count, int conv_len, int d_child, int m,
    int* kill_flag,
    /* Quick-check state output: killing (ell, s, W_int) written on prune. */
    int* qc_ell_out,
    int* qc_s_out,
    int32_t* qc_W_int_out)
{
    const int lane = threadIdx.x;
    const int bd = (int)blockDim.x;

    for (int ell_oi = lane; ell_oi < ell_count; ell_oi += bd) {
        int ell = ell_order[ell_oi];
        int n_cv = ell - 1;
        int n_windows = conv_len - n_cv + 1;
        if (n_windows <= 0) continue;

        /* ── Initial window sum: ws = sum(conv[0..n_cv-1]) ──
         * int32 safe: max ws = sum(all conv) = m^2 = 400 for m=20. */
        int32_t ws = 0;
        for (int k = 0; k < n_cv; k++)
            ws += conv[k];

        /* ── Initial W_int: sum(child[0..hi_bin_0]) ── */
        int hi_bin_0 = ell - 2;
        if (hi_bin_0 > d_child - 1) hi_bin_0 = d_child - 1;
        int32_t W_int = 0;
        for (int b = 0; b <= hi_bin_0; b++)
            W_int += child[b];

        /* ── Check first window (s=0) ── */
        int ell_idx = ell - 2;
        {
            int W_cl = (int)W_int;
            if (W_cl < 0) W_cl = 0;
            if (W_cl > m) W_cl = m;
            if (ws > threshold_table[ell_idx * (m + 1) + W_cl]) {
                int prev = atomicMin_block(kill_flag, lane);
                if (prev >= bd) {
                    /* First killer — record the killing window. */
                    *qc_ell_out = ell;
                    *qc_s_out = 0;
                    *qc_W_int_out = W_int;
                }
                goto done_ell;
            }
        }

        /* ── Sliding window for s = 1..n_windows-1 ── */
        for (int s = 1; s < n_windows; s++) {
            /* Update ws: add right edge, subtract left edge. */
            ws += conv[s + n_cv - 1];
            ws -= conv[s - 1];

            /* Update W_int via bin-range sliding window.
             * hi_bin increases by 1 when s + ell - 2 < d_child.
             * lo_bin increases by 1 when s >= d_child (old_lo = s - d_child). */
            if (s + ell - 2 < d_child)
                W_int += child[s + ell - 2];
            if (s >= d_child)
                W_int -= child[s - d_child];

            int W_cl = (int)W_int;
            if (W_cl < 0) W_cl = 0;
            if (W_cl > m) W_cl = m;
            if (ws > threshold_table[ell_idx * (m + 1) + W_cl]) {
                int prev = atomicMin_block(kill_flag, lane);
                if (prev >= bd) {
                    /* First killer — record the killing window. */
                    *qc_ell_out = ell;
                    *qc_s_out = s;
                    *qc_W_int_out = W_int;
                }
                goto done_ell;
            }
        }

        done_ell:
        /* Early exit if any thread already found a kill. */
        if (*kill_flag < bd) return true;
    }

    return false;  /* actual result checked via kill_flag after sync */
}

/* ═══════════════════════════════════════════════════════════════════
 *  parallel_window_scan — prefix-sum based full window scan (LEGACY)
 *
 *  Kept for d_child<=32 subtree pruning path.  The hot loop uses
 *  thread_private_window_scan instead (60× fewer barriers).
 *
 *  Builds prefix sums of conv and child, then tests all (ell, s_lo)
 *  pairs in parallel across lanes with early exit per ell.
 *
 *  Returns true if the child is pruned.  On prune, writes the
 *  killing (ell, s, W_int) to the output pointers for quick-check
 *  state update.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ __noinline__ bool parallel_window_scan(
    const int32_t* conv,
    const int32_t* child,
    const int32_t* threshold_table,
    const int32_t* ell_order,
    int ell_count, int conv_len, int d_child, int m,
    int* qc_ell_out,
    int* qc_s_out,
    int32_t* qc_W_int_out,
    /* scratch shared memory: */
    int32_t* prefix_conv,   /* [128] */
    int32_t* prefix_tmp,    /* [128] */
    int32_t* prefix_c,      /* [d_child+1] */
    /* shared temporaries for cross-warp reduction: */
    int* killer_s_smem,
    int* killer_W_smem)
{
    const int lane = threadIdx.x;

    /* ── Build prefix_conv (inclusive prefix sum of raw_conv) ──
     * int32 safe: max prefix = sum(all conv) = m^2 ≤ 400. */
    for (int i = lane; i < conv_len; i += blockDim.x)
        prefix_conv[i] = conv[i];
    /* Zero-pad to power-of-2 alignment (128). */
    for (int i = conv_len + lane; i < 128; i += blockDim.x)
        prefix_conv[i] = 0;
    __syncthreads();

    /* Kogge-Stone inclusive prefix sum with ping-pong buffers.
     * All threads participate in every __syncthreads to avoid deadlock. */
    int32_t* src = prefix_conv;
    int32_t* dst = prefix_tmp;
    for (int stride = 1; stride < conv_len; stride <<= 1) {
        for (int idx = lane; idx < conv_len; idx += blockDim.x) {
            int32_t val = src[idx];
            if (idx >= stride)
                val += src[idx - stride];
            dst[idx] = val;
        }
        __syncthreads();
        /* swap src and dst */
        int32_t* swap = src; src = dst; dst = swap;
    }
    /* Ensure final result is in prefix_conv. */
    if (src != prefix_conv) {
        for (int idx = lane; idx < conv_len; idx += blockDim.x)
            prefix_conv[idx] = src[idx];
        __syncthreads();
    }

    /* ── Build prefix_c (inclusive prefix sum of child masses) ──
     * int32 safe: max prefix = sum(all child) = m ≤ 20. */
    if (lane == 0) prefix_c[0] = 0;
    __syncthreads();
    if (lane < d_child)
        prefix_c[lane + 1] = child[lane];
    __syncthreads();
    /* In-place Kogge-Stone on prefix_c[1..d_child].
     * We work on indices 1..d_child, so the effective length is d_child. */
    for (int stride = 1; stride < d_child; stride <<= 1) {
        int32_t val = 0;
        if (lane < d_child) {
            int idx = lane + 1;
            val = prefix_c[idx];
            if (idx > stride)
                val += prefix_c[idx - stride];
        }
        __syncthreads();
        if (lane < d_child)
            prefix_c[lane + 1] = val;
        __syncthreads();
    }

    /* ── Scan ell values in optimised order ── */
    for (int ell_oi = 0; ell_oi < ell_count; ell_oi++) {
        int ell  = ell_order[ell_oi];
        int n_cv = ell - 1;
        int n_windows = conv_len - n_cv + 1;
        if (n_windows <= 0) continue;

        bool lane_pruned   = false;
        int  lane_killer_s = -1;
        int  lane_killer_W = -1;

        for (int s_lo = lane; s_lo < n_windows; s_lo += blockDim.x) {
            /* Window sum from prefix_conv. */
            int32_t ws = prefix_conv[s_lo + n_cv - 1];
            if (s_lo > 0) ws -= prefix_conv[s_lo - 1];

            int lo_bin = s_lo - (d_child - 1);
            if (lo_bin < 0) lo_bin = 0;
            int hi_bin = s_lo + ell - 2;
            if (hi_bin > d_child - 1) hi_bin = d_child - 1;
            int W_int = (int)(prefix_c[hi_bin + 1] - prefix_c[lo_bin]);

            int ell_idx = ell - 2;
            int32_t thresh = threshold_table[ell_idx * (m + 1) + W_int];
            if (ws > thresh) {
                lane_pruned   = true;
                lane_killer_s = s_lo;
                lane_killer_W = W_int;
                break;
            }
        }

        /* ── Reduce across block: did any lane find a kill? ── */
        bool any_killed = false;

        if (blockDim.x == 32) {
            /* Single warp: use __ballot_sync for fast reduction. */
            uint32_t kill_mask = __ballot_sync(0xFFFFFFFF, lane_pruned);
            any_killed = (kill_mask != 0);
            if (any_killed) {
                int winner = __ffs((int)kill_mask) - 1;
                if (lane == winner) {
                    *killer_s_smem = lane_killer_s;
                    *killer_W_smem = lane_killer_W;
                }
            }
            __syncthreads();
        } else {
            /* Multi-warp: shared-memory coordination. */
            if (lane == 0) *killer_W_smem = (int)blockDim.x;  /* sentinel */
            __syncthreads();
            if (lane_pruned)
                atomicMin_block(killer_W_smem, lane);
            __syncthreads();
            any_killed = (*killer_W_smem < (int)blockDim.x);
            if (any_killed) {
                int winner = *killer_W_smem;
                __syncthreads();
                if ((int)lane == winner) {
                    *killer_s_smem = lane_killer_s;
                    *killer_W_smem = lane_killer_W;
                }
                __syncthreads();
            }
        }

        if (any_killed) {
            if (lane == 0) {
                *qc_ell_out = ell;
                *qc_s_out = *killer_s_smem;
                *qc_W_int_out = (int32_t)*killer_W_smem;
            }
            __syncthreads();
            return true;
        }
    }

    return false;   /* survived all windows */
}

/* ═══════════════════════════════════════════════════════════════════
 *  canonicalize_and_stage — determine canonical (min of fwd/rev)
 *  and stage the survivor in shared memory buffer.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void canonicalize_and_stage(
    const int32_t* child,
    int32_t*       surv_buf,
    int*           surv_count,
    int d_child,
    int*           cmp_array,     /* [MAX_D_CHILD] shared */
    bool*          use_rev_smem,   /* shared */
    int*           slot_smem)      /* shared */
{
    const int lane = threadIdx.x;
    bool use_rev = false;

    if (d_child <= 32) {
        /* Single warp: warp-ballot comparison. */
        int fwd = (lane < d_child) ? child[lane] : 0;
        int rev = (lane < d_child) ? child[d_child - 1 - lane] : 0;
        int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
        uint32_t lt_mask = __ballot_sync(0xFFFFFFFF, cmp < 0);
        uint32_t gt_mask = __ballot_sync(0xFFFFFFFF, cmp > 0);
        int first_lt = lt_mask ? __ffs((int)lt_mask) : 33;
        int first_gt = gt_mask ? __ffs((int)gt_mask) : 33;
        use_rev = (first_lt < first_gt);
    } else {
        /* Two warps: comparison via shared memory. */
        int fwd = (lane < d_child) ? child[lane] : 0;
        int rev = (lane < d_child) ? child[d_child - 1 - lane] : 0;
        int cmp = (rev < fwd) ? -1 : (rev > fwd) ? 1 : 0;
        cmp_array[lane] = cmp;
        __syncthreads();
        if (lane == 0) {
            use_rev = false;
            for (int i = 0; i < d_child; i++) {
                if (cmp_array[i] < 0) { use_rev = true; break; }
                if (cmp_array[i] > 0) { break; }
            }
            *use_rev_smem = use_rev;
        }
        __syncthreads();
        use_rev = *use_rev_smem;
    }

    /* Allocate a slot in the staging buffer. */
    if (lane == 0)
        *slot_smem = atomicAdd_block(surv_count, 1);
    __syncthreads();
    int slot = *slot_smem;

    /* Write survivor (canonical form) to staging buffer. */
    if (lane < d_child) {
        if (use_rev)
            surv_buf[slot * d_child + lane] = child[d_child - 1 - lane];
        else
            surv_buf[slot * d_child + lane] = child[lane];
    }
    __syncthreads();
}

/* ═══════════════════════════════════════════════════════════════════
 *  flush_survivors_to_global — copy staging buffer to global output
 * ═══════════════════════════════════════════════════════════════════ */
__device__ void flush_survivors_to_global(
    const int32_t* surv_buf,
    int*           surv_count_smem,
    int32_t*       survivors_global,
    int32_t*       survivor_count_global,
    int d_child,
    int max_survivors,
    int*           base_smem)         /* shared */
{
    const int lane = threadIdx.x;
    int count = *surv_count_smem;
    if (count == 0) return;

    if (lane == 0)
        *base_smem = atomicAdd(survivor_count_global, count);
    __syncthreads();
    int base = *base_smem;

    /* Overflow guard: don't write past the global buffer. */
    if (base + count > max_survivors) {
        /* Silently clamp.  Host checks survivor_count > max_survivors. */
        int room = max_survivors - base;
        if (room <= 0) {
            if (lane == 0) *surv_count_smem = 0;
            __syncthreads();
            return;
        }
        count = room;
    }

    int total_elements = count * d_child;
    for (int i = lane; i < total_elements; i += blockDim.x)
        survivors_global[base * d_child + i] = surv_buf[i];

    __syncthreads();
    if (lane == 0) *surv_count_smem = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  partial_window_scan_max_threshold — subtree pruning window check
 *
 *  Checks whether partial conv (fixed prefix) + guaranteed minimum
 *  contributions from unfixed bins already exceeds the threshold.
 *  Scans ALL window positions (not just within the fixed prefix)
 *  using min_contrib_prefix to cover the unfixed region (Idea 02).
 *
 *  Runs on lane 0 only (sequential).  See Section 3.9.
 * ═══════════════════════════════════════════════════════════════════ */
__device__ bool partial_window_scan_max_threshold(
    const int32_t* partial_conv_prefix,  /* inclusive prefix sum */
    int partial_conv_len,
    int fixed_len,
    const int32_t* threshold_table,
    const int32_t* ell_order,
    int ell_count, int m, int d_child,
    const int32_t* prefix_c_fixed,   /* prefix sum of child[0..fixed_len-1] */
    const int32_t* parent_prefix,     /* prefix sum of parent masses */
    int first_unfixed_parent, int d_parent,
    const int32_t* min_contrib_prefix, /* inclusive prefix sum of guaranteed min contributions */
    int full_conv_len)                 /* 2*d_child - 1 */
{
    for (int ell_oi = 0; ell_oi < ell_count; ell_oi++) {
        int ell  = ell_order[ell_oi];
        int n_cv = ell - 1;
        /* Scan ALL window positions, not just within fixed prefix. */
        int n_windows = full_conv_len - n_cv + 1;
        if (n_windows <= 0) continue;

        for (int s_lo = 0; s_lo < n_windows; s_lo++) {
            int s_hi = s_lo + n_cv - 1;

            /* Fixed part: sum partial_conv[k] for k in window ∩ [0, partial_conv_len). */
            int32_t ws = 0;
            {
                int k_start = s_lo;
                int k_end = s_hi;
                if (k_end >= partial_conv_len) k_end = partial_conv_len - 1;
                if (k_end >= k_start) {
                    ws = partial_conv_prefix[k_end];
                    if (k_start > 0) ws -= partial_conv_prefix[k_start - 1];
                }
            }

            /* Unfixed part: guaranteed minimum contributions (Idea 02). */
            {
                int32_t mc_sum = min_contrib_prefix[s_hi];
                if (s_lo > 0) mc_sum -= min_contrib_prefix[s_lo - 1];
                ws += mc_sum;
            }

            int lo_bin = s_lo - (d_child - 1);
            if (lo_bin < 0) lo_bin = 0;
            int hi_bin = s_lo + ell - 2;
            if (hi_bin > d_child - 1) hi_bin = d_child - 1;

            /* W_int_fixed: actual child masses in fixed prefix bins. */
            int32_t W_int_fixed = 0;
            {
                int fhi = hi_bin;
                if (fhi > fixed_len - 1) fhi = fixed_len - 1;
                if (fhi >= lo_bin) {
                    int flo = lo_bin < 0 ? 0 : lo_bin;
                    W_int_fixed = prefix_c_fixed[fhi + 1] - prefix_c_fixed[flo];
                }
            }

            /* W_int_unfixed: parent upper bound for bins right of fixed prefix. */
            int32_t W_int_unfixed = 0;
            {
                int uflo = lo_bin;
                if (uflo < fixed_len) uflo = fixed_len;
                if (uflo <= hi_bin) {
                    int p_lo = uflo / 2;
                    int p_hi = hi_bin / 2;
                    if (p_lo < first_unfixed_parent) p_lo = first_unfixed_parent;
                    if (p_hi >= d_parent) p_hi = d_parent - 1;
                    if (p_lo <= p_hi)
                        W_int_unfixed = parent_prefix[p_hi + 1] - parent_prefix[p_lo];
                }
            }

            int32_t W_int_max = W_int_fixed + W_int_unfixed;
            /* Clamp to m to avoid out-of-bounds table access. */
            if (W_int_max > m) W_int_max = m;
            int ell_idx = ell - 2;
            int32_t thresh = threshold_table[ell_idx * (m + 1) + (int)W_int_max];
            if (ws > thresh)
                return true;
        }
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════════
 *
 *  CASCADE KERNEL — main entry point
 *
 * ═══════════════════════════════════════════════════════════════════ */
__global__ void cascade_kernel(
    const int32_t* __restrict__ g_parents,
    const int32_t* __restrict__ g_lo_arrays,
    const int32_t* __restrict__ g_hi_arrays,
    const int32_t* __restrict__ g_threshold_table,
    const int32_t* __restrict__ g_ell_order,
    int32_t*       __restrict__ g_survivors,
    int32_t*       __restrict__ g_survivor_count,
    int32_t*       __restrict__ g_next_parent,
    int32_t*       __restrict__ g_done_parent,
    int num_parents, int d_parent, int d_child, int m,
    int ell_count, int conv_len,
    double threshold_asym,
    int max_survivors,
    int surv_cap)
{
    const int lane = threadIdx.x;

    /* blockDim.x must be >= d_child and a multiple of WARP_SIZE.
     * For d_child < 32, we launch 32 threads with extra lanes idle.
     * For d_child >= 32, blockDim.x == d_child. */
    assert(blockDim.x >= (unsigned)d_child && blockDim.x % WARP_SIZE == 0);

    /* Diagnostic trace. Enabled with -DTRACE (off by default for production).
     * Device printf is extremely slow and can fill the 1MB default buffer,
     * causing threads to block.  Only enable for debugging small runs. */
    #ifdef TRACE
    #define TRACE_PARENT0(msg) \
        do { if (lane == 0 && pid == 0) printf("[TRACE] %s\n", msg); } while(0)
    #define TRACE_PARENT0_VAL(msg, val) \
        do { if (lane == 0 && pid == 0) printf("[TRACE] %s = %lld\n", msg, (long long)(val)); } while(0)
    #else
    #define TRACE_PARENT0(msg)           do {} while(0)
    #define TRACE_PARENT0_VAL(msg, val)  do {} while(0)
    #endif

    /* ────────── Shared memory declarations ────────── */

    /* Parent, child, and cursor arrays. */
    __shared__ int32_t parent_smem[MAX_D_PARENT];
    __shared__ int32_t child_smem[MAX_D_CHILD];
    __shared__ int32_t cursor_smem[MAX_D_PARENT];
    __shared__ int32_t lo_smem[MAX_D_PARENT];
    __shared__ int32_t hi_smem[MAX_D_PARENT];

    /* Autoconvolution. */
    __shared__ int32_t raw_conv_smem[MAX_CONV_LEN];

    /* Gray code state. */
    __shared__ int32_t active_pos_smem[MAX_D_PARENT];
    __shared__ int32_t radix_smem[MAX_D_PARENT];
    __shared__ int32_t gc_a_smem[MAX_D_PARENT];
    __shared__ int32_t gc_dir_smem[MAX_D_PARENT];
    __shared__ int32_t gc_focus_smem[MAX_D_PARENT + 1];
    __shared__ int     n_active_smem;

    /* Prefix sums (for subtree pruning, d_child<=32 only).
     * Downsized from int64 to int32: max prefix = m^2 = 400 (conv)
     * or m = 20 (child). Both trivially fit int32. */
    __shared__ int32_t prefix_conv_smem[128];
    __shared__ int32_t prefix_c_smem[MAX_D_CHILD + 1];

    /* Threshold table, ell order, and survivor staging buffer in dynamic
     * shared memory.  Downsized thresholds from int64 to int32: max = 601
     * for m=20.  Survivor buffer uses (surv_cap+1) slots to absorb the
     * one-past-capacity write that triggers the flush (Idea 3). */
    extern __shared__ char dynamic_smem[];
    int32_t* threshold_table_smem = (int32_t*)dynamic_smem;
    int32_t* ell_order_smem = (int32_t*)(dynamic_smem +
                               ell_count * (m + 1) * sizeof(int32_t));
    int32_t* surv_buf_smem = (int32_t*)(dynamic_smem +
                               ell_count * (m + 1) * sizeof(int32_t) +
                               ell_count * sizeof(int32_t));

    /* Survivor staging buffer — in dynamic shared memory (Idea 3).
     * Pointer set up below after threshold_table + ell_order. */
    __shared__ int     surv_count_smem;

    /* Quick-check state (single-window, used as primary cache slot). */
    __shared__ int     qc_ell_smem;
    __shared__ int     qc_s_smem;
    __shared__ int32_t qc_W_int_smem;

    /* Lazy QC state — Idea 1: analytical window-sum tracking. */
    __shared__ int32_t ws_qc_lazy_smem;
    __shared__ int32_t sensitivity_cross_smem[MAX_D_PARENT];
    __shared__ int     self_mask_k1_smem[MAX_D_PARENT];
    __shared__ int     self_mask_k2_smem[MAX_D_PARENT];
    __shared__ int     self_mask_m_smem[MAX_D_PARENT];
    __shared__ bool    need_full_conv_smem;

    /* Multi-window QC cache — Idea 4: K=5 killing windows. */
    __shared__ int     qc_cache_ell[QC_CACHE_K];
    __shared__ int     qc_cache_s[QC_CACHE_K];
    __shared__ int32_t qc_cache_W_int[QC_CACHE_K];
    __shared__ int32_t qc_cache_ws_lazy[QC_CACHE_K];
    __shared__ int32_t qc_cache_sensitivity[MAX_D_PARENT * QC_CACHE_K];
    __shared__ int     qc_cache_mask_k1[MAX_D_PARENT * QC_CACHE_K];
    __shared__ int     qc_cache_mask_k2[MAX_D_PARENT * QC_CACHE_K];
    __shared__ int     qc_cache_mask_m[MAX_D_PARENT * QC_CACHE_K];
    __shared__ int     qc_cache_count_smem;
    __shared__ int     qc_cache_next_smem;

    /* Batch QC skip state — Idea 2. */
    __shared__ bool    batch_skip_smem;
    __shared__ int     batch_skip_count_smem;

    __shared__ int     cmp_array_smem[MAX_D_CHILD];
    __shared__ bool    use_rev_smem;
    __shared__ int     slot_smem;
    __shared__ int     flush_base_smem;

    /* Gray code loop communication. */
    __shared__ int     gc_pos_smem;
    __shared__ int     gc_j_smem;
    __shared__ bool    gc_done_smem;
    __shared__ bool    skip_parent_smem;
    __shared__ int     parent_idx_smem;

    /* (old1/old2/new1/new2_smem removed — no longer needed with lazy conv) */

    /* Thread-private window scan kill flag. */
    __shared__ int     kill_flag_smem;

    /* Subtree pruning. */
    __shared__ bool    subtree_killed_smem;

    /* Guaranteed minimum contributions from unfixed bins (Idea 02).
     * Used during subtree pruning to tighten the lower bound on
     * window sums by accounting for the guaranteed minimum mass
     * in unfixed child bins.  Stored as inclusive prefix sum. */
    __shared__ int32_t min_contrib_smem[MAX_CONV_LEN];

    /* Parent prefix sum for subtree pruning W_int_unfixed. */
    __shared__ int32_t parent_prefix_smem[MAX_D_PARENT + 1];

    /* Subtree sizes: subtree_size[j] = product(radix[0..j-1]).
     * subtree_size[0]=1.  Used for cost/benefit and skip counting. */
    __shared__ int64_t subtree_size_smem[MAX_D_PARENT + 1];

    /* ────────── Load threshold table + ell_order into shared mem ────────── */
    {
        int table_size = ell_count * (m + 1);
        for (int i = lane; i < table_size; i += blockDim.x)
            threshold_table_smem[i] = g_threshold_table[i];
        for (int i = lane; i < ell_count; i += blockDim.x)
            ell_order_smem[i] = g_ell_order[i];
    }
    __syncthreads();

    /* Initialise staging buffer counter. */
    if (lane == 0) surv_count_smem = 0;
    __syncthreads();

    /* ═══════════════════════════════════════════════════════════════
     *  PERSISTENT BLOCK LOOP — claim one parent at a time
     * ═══════════════════════════════════════════════════════════════ */
    while (true) {
        /* ── Claim next parent (atomic work-stealing) ── */
        if (lane == 0)
            parent_idx_smem = atomicAdd(g_next_parent, 1);
        __syncthreads();
        if (parent_idx_smem >= num_parents) break;
        int pid = parent_idx_smem;

        /* ── Phase 0: Load parent data from global memory ── */
        if (lane < d_parent) {
            parent_smem[lane] = g_parents[pid * d_parent + lane];
            lo_smem[lane]     = g_lo_arrays[pid * d_parent + lane];
            hi_smem[lane]     = g_hi_arrays[pid * d_parent + lane];
        }
        __syncthreads();

#ifdef DEBUG
        if (lane == 0 && pid == 0) {
            printf("[block %d] parent %d: [%d %d %d %d] lo=[%d %d %d %d] hi=[%d %d %d %d]\n",
                   blockIdx.x, pid,
                   parent_smem[0], parent_smem[1],
                   d_parent > 2 ? parent_smem[2] : -1,
                   d_parent > 3 ? parent_smem[3] : -1,
                   lo_smem[0], lo_smem[1],
                   d_parent > 2 ? lo_smem[2] : -1,
                   d_parent > 3 ? lo_smem[3] : -1,
                   hi_smem[0], hi_smem[1],
                   d_parent > 2 ? hi_smem[2] : -1,
                   d_parent > 3 ? hi_smem[3] : -1);
        }
#endif

        TRACE_PARENT0("Phase 0: parent loaded");

        /* ── Phase 0b: Asymmetry pre-filter ──
         * Skip parents whose left-mass fraction is outside
         * [1-thresh, thresh].  Performance optimisation, not soundness. */
        if (lane == 0) {
            int left_sum = 0;
            for (int i = 0; i < d_parent / 2; i++)
                left_sum += parent_smem[i];
            double left_frac = (double)left_sum / (double)m;
            skip_parent_smem = (left_frac >= threshold_asym ||
                                left_frac <= 1.0 - threshold_asym);
        }
        __syncthreads();
        if (skip_parent_smem) {
            TRACE_PARENT0("Phase 0b: SKIPPED by asymmetry filter");
            continue;
        }

        /* ── Phase 1: Build active positions (bins with range > 1) ──
         * Right-to-left so inner (fastest) Gray code digits correspond
         * to rightmost parent bins.  Fixed region = left prefix. */
        if (lane == 0) {
            n_active_smem = 0;
            for (int i = d_parent - 1; i >= 0; i--) {
                cursor_smem[i] = lo_smem[i];
                if (hi_smem[i] > lo_smem[i]) {
                    active_pos_smem[n_active_smem] = i;
                    radix_smem[n_active_smem] = hi_smem[i] - lo_smem[i] + 1;
                    n_active_smem++;
                }
            }
        }
        __syncthreads();
        int n_active = n_active_smem;

        /* If no active positions, the parent has exactly one child.
         * Still need to test it. */

        /* ── Phase 1b: Build parent prefix sum + subtree sizes ── */
        if (lane == 0) {
            parent_prefix_smem[0] = 0;
            for (int i = 0; i < d_parent; i++)
                parent_prefix_smem[i + 1] = parent_prefix_smem[i] +
                                             parent_smem[i];
            /* Subtree sizes: subtree_size[j] = product(radix[0..j-1]) */
            subtree_size_smem[0] = 1;
            for (int j = 0; j < n_active; j++)
                subtree_size_smem[j + 1] = subtree_size_smem[j] *
                                            (int64_t)radix_smem[j];
        }
        __syncthreads();

        TRACE_PARENT0_VAL("Phase 1: n_active", n_active);

        /* ── Phase 2: Build initial child from cursor ── */
        if (lane < d_parent) {
            int c = cursor_smem[lane];
            child_smem[2 * lane]     = c;
            child_smem[2 * lane + 1] = parent_smem[lane] - c;
        }
        __syncthreads();

        TRACE_PARENT0("Phase 2: initial child built");

        /* ── Phase 3: Full autoconvolution of initial child O(d^2) ── */
        cooperative_full_autoconv(child_smem, raw_conv_smem,
                                  d_child, conv_len);

        TRACE_PARENT0("Phase 3: autoconv done");

        /* ── Phase 4: Initialise Gray code state ── */
        if (lane == 0) {
            for (int j = 0; j < n_active; j++) {
                gc_a_smem[j]     = 0;
                gc_dir_smem[j]   = 1;
                gc_focus_smem[j] = j;
            }
            gc_focus_smem[n_active] = n_active;  /* sentinel */
            qc_ell_smem   = 0;      /* no quick-check history */
            qc_s_smem     = 0;
            qc_W_int_smem = 0;
        }
        __syncthreads();

#ifdef DEBUG
        if (lane == 0 && pid == 0) {
            printf("[block %d] parent %d: n_active=%d, initial child=[",
                   blockIdx.x, pid, n_active);
            for (int ii = 0; ii < d_child; ii++)
                printf("%d%s", child_smem[ii], ii < d_child-1 ? " " : "");
            printf("]\n");
            printf("[block %d] parent %d: Phase 3 done. conv=[",
                   blockIdx.x, pid);
            for (int ii = 0; ii < conv_len; ii++)
                printf("%d%s", raw_conv_smem[ii], ii < conv_len-1 ? " " : "");
            printf("]\n");
        }
        __syncthreads();
#endif

        TRACE_PARENT0("Phase 4: Gray code init done");

        /* ── Phase 5: Test initial child ── */
        TRACE_PARENT0("Phase 5: testing initial child");
        {
            if (lane == 0) kill_flag_smem = (int)blockDim.x;
            __syncthreads();

            thread_private_window_scan(
                raw_conv_smem, child_smem,
                threshold_table_smem, ell_order_smem,
                ell_count, conv_len, d_child, m,
                &kill_flag_smem,
                &qc_ell_smem, &qc_s_smem, &qc_W_int_smem);
            __syncthreads();

            bool pruned = (kill_flag_smem < (int)blockDim.x);
            if (!pruned) {
                canonicalize_and_stage(
                    child_smem, surv_buf_smem, &surv_count_smem,
                    d_child, cmp_array_smem, &use_rev_smem, &slot_smem);
                if (surv_count_smem >= surv_cap) {
                    flush_survivors_to_global(
                        surv_buf_smem, &surv_count_smem,
                        g_survivors, g_survivor_count,
                        d_child, max_survivors, &flush_base_smem);
                }
            }
        }

        TRACE_PARENT0("Phase 5: initial child tested");

        /* ── Phase 5b: Initialize multi-window QC cache (Ideas 1+4) ── */
        if (lane == 0) {
            qc_cache_count_smem = 0;
            qc_cache_next_smem = 0;
            ws_qc_lazy_smem = 0;

            /* If Phase 5 found a killing window, seed the cache. */
            if (qc_ell_smem > 0) {
                qc_cache_ell[0] = qc_ell_smem;
                qc_cache_s[0]   = qc_s_smem;
                qc_cache_W_int[0] = qc_W_int_smem;

                /* Compute ws_qc_lazy from conv for this window. */
                int32_t ws0 = 0;
                int n_cv = qc_ell_smem - 1;
                for (int k = qc_s_smem; k < qc_s_smem + n_cv; k++)
                    ws0 += raw_conv_smem[k];
                qc_cache_ws_lazy[0] = ws0;

                /* Precompute sensitivity for cache slot 0. */
                precompute_qc_sensitivity(
                    child_smem, d_child,
                    qc_ell_smem, qc_s_smem,
                    active_pos_smem, n_active,
                    &qc_cache_sensitivity[0],
                    &qc_cache_mask_k1[0],
                    &qc_cache_mask_k2[0],
                    &qc_cache_mask_m[0],
                    QC_CACHE_K);

                qc_cache_count_smem = 1;
                qc_cache_next_smem = 1;
            }
        }
        __syncthreads();

        TRACE_PARENT0("Phase 5b: QC cache initialized");

        /* ── Phase 6: Gray code enumeration loop ──
         *
         * OPTIMIZED HOT LOOP (Ideas 1+2+4):
         *   1. Gray code advance + child update (lane 0).
         *   2. Lazy QC: O(1) analytical check against K cached
         *      windows (lane 0).  Kills ~94% of children.
         *   3. ONE __syncthreads barrier.
         *   4. If QC killed: continue (no conv, no scan!).
         *   5. If QC missed: cooperative_full_autoconv + full scan.
         *   6. Batch QC (Idea 2): skip entire inner loop if all
         *      cursor values killed.
         * Per-child barriers: QC hit = 1, QC miss = ~4.
         *   Plus ~2 for canonicalize if survivor.
         */
        int64_t children_tested = 1;
        int64_t n_skipped = 0;
        int64_t watchdog = 0;
        int64_t expected_total = 1;
        if (lane == 0) {
            for (int j = 0; j < n_active; j++)
                expected_total *= radix_smem[j];
        }
        TRACE_PARENT0_VAL("Phase 6: expected_total", expected_total);

        while (true) {
            /* ═══ STEP 1: Gray code advance + child update + lazy QC ═══
             * Lane 0: advance GC, update child, then run analytical
             * QC check against ALL cached windows (Ideas 1+4).
             * If any window kills: set kill_flag=0, skip conv+scan.
             * If none kills: set need_full_conv=true. */
            if (lane == 0) {
                int j = gc_focus_smem[0];
                if (j >= n_active || watchdog > expected_total + 10) {
                    gc_done_smem = true;
                    need_full_conv_smem = false;
                    batch_skip_smem = false;
                } else {
                    gc_done_smem = false;
                    batch_skip_smem = false;
                    gc_focus_smem[0] = 0;

                    int pos = active_pos_smem[j];
                    gc_a_smem[j] += gc_dir_smem[j];
                    cursor_smem[pos] = lo_smem[pos] + gc_a_smem[j];

                    if (gc_a_smem[j] == 0 ||
                        gc_a_smem[j] == radix_smem[j] - 1)
                    {
                        gc_dir_smem[j] = -gc_dir_smem[j];
                        gc_focus_smem[j] = gc_focus_smem[j + 1];
                        gc_focus_smem[j + 1] = j + 1;
                    }

                    gc_pos_smem = pos;
                    gc_j_smem   = j;

                    /* Read old child[k1], compute new values. */
                    int k1 = 2 * pos;
                    int k2 = k1 + 1;
                    int old1 = child_smem[k1];
                    int new_cursor = cursor_smem[pos];
                    int new1 = new_cursor;
                    int new2 = parent_smem[pos] - new_cursor;
                    int delta_c = new1 - old1;  /* +1 or -1 */

                    /* Write new child values. */
                    child_smem[k1] = new1;
                    child_smem[k2] = new2;

                    /* ── Lazy QC: O(1) analytical check (Ideas 1+4) ── */
                    int cache_count = qc_cache_count_smem;
                    bool killed = false;

                    if (cache_count > 0) {
                        killed = lazy_qc_update_and_check(
                            j, delta_c, old1,
                            parent_smem[pos],
                            k1, k2,
                            d_child, m,
                            active_pos_smem, n_active,
                            threshold_table_smem,
                            &ws_qc_lazy_smem,
                            &qc_ell_smem, &qc_s_smem, &qc_W_int_smem,
                            sensitivity_cross_smem,
                            self_mask_k1_smem, self_mask_k2_smem,
                            self_mask_m_smem,
                            qc_cache_ws_lazy, qc_cache_ell, qc_cache_s,
                            qc_cache_W_int,
                            qc_cache_sensitivity,
                            qc_cache_mask_k1, qc_cache_mask_k2,
                            qc_cache_mask_m,
                            cache_count);
                    }

                    if (killed) {
                        kill_flag_smem = 0;
                        need_full_conv_smem = false;

                        /* ═══ Idea 2: Batch QC — try to skip entire inner loop ═══
                         * If gc_j==0 (innermost digit), check all remaining cursor
                         * values analytically.  If ALL killed by the best cached
                         * window, skip the entire inner sweep. */
                        if (j == 0 && cache_count > 0 && n_active > 0) {
                            int ipos = active_pos_smem[0];
                            int ik1 = 2 * ipos, ik2 = ik1 + 1;
                            int a_ipos = parent_smem[ipos];
                            int lo_i = lo_smem[ipos], hi_i = hi_smem[ipos];
                            int R_inner = hi_i - lo_i + 1;

                            if (R_inner >= 3) {
                                /* Use best cached window (slot 0) for batch check. */
                                int bw = 0;
                                int b_ell = qc_cache_ell[bw];
                                int b_s   = qc_cache_s[bw];
                                if (b_ell > 0) {
                                    int b_n_cv = b_ell - 1;
                                    int b_ws_lo = b_s;
                                    int b_ws_hi = b_s + b_n_cv - 1;

                                    /* Precompute cross-term sensitivity for innermost pos */
                                    int32_t dws_cross = 0;
                                    for (int jj = 0; jj < d_child; jj++) {
                                        if (jj == ik1 || jj == ik2) continue;
                                        if (ik1 + jj >= b_ws_lo && ik1 + jj <= b_ws_hi)
                                            dws_cross += 2 * child_smem[jj];
                                        if (ik2 + jj >= b_ws_lo && ik2 + jj <= b_ws_hi)
                                            dws_cross -= 2 * child_smem[jj];
                                    }

                                    int mk1 = (2*ik1 >= b_ws_lo && 2*ik1 <= b_ws_hi) ? 1 : 0;
                                    int mk2 = (2*ik2 >= b_ws_lo && 2*ik2 <= b_ws_hi) ? 1 : 0;
                                    int mm  = (ik1+ik2 >= b_ws_lo && ik1+ik2 <= b_ws_hi) ? 1 : 0;

                                    /* QC bin range for W_int tracking */
                                    int b_lo_bin = b_s - (d_child - 1);
                                    if (b_lo_bin < 0) b_lo_bin = 0;
                                    int b_hi_bin = b_s + b_ell - 2;
                                    if (b_hi_bin > d_child - 1) b_hi_bin = d_child - 1;

                                    int32_t ws_cur = qc_cache_ws_lazy[bw];
                                    int32_t W_cur  = qc_cache_W_int[bw];
                                    int c_cur = child_smem[ik1];

                                    bool all_killed = true;
                                    for (int c = lo_i; c <= hi_i && all_killed; c++) {
                                        if (c == c_cur) {
                                            int W_cl = (int)W_cur;
                                            if (W_cl < 0) W_cl = 0;
                                            if (W_cl > m) W_cl = m;
                                            int ell_idx = b_ell - 2;
                                            int32_t thresh = threshold_table_smem[ell_idx * (m + 1) + W_cl];
                                            if (ws_cur <= thresh) all_killed = false;
                                        } else {
                                            int dc = c - c_cur;
                                            int32_t ws_c = ws_cur;
                                            int abs_dc = dc > 0 ? dc : -dc;
                                            int dir = dc > 0 ? 1 : -1;
                                            for (int step = 0; step < abs_dc; step++) {
                                                int ci = c_cur + (dc > 0 ? step : -step);
                                                ws_c += dir * dws_cross;
                                                if (mk1) ws_c += dir * (2*ci + dir);
                                                if (mk2) ws_c += dir * (-(2*(a_ipos - ci) - dir));
                                                if (mm)  ws_c += dir * 2 * (a_ipos - 2*ci - dir);
                                            }
                                            int32_t W_c = W_cur;
                                            if (ik1 >= b_lo_bin && ik1 <= b_hi_bin)
                                                W_c += (c - c_cur);
                                            if (ik2 >= b_lo_bin && ik2 <= b_hi_bin)
                                                W_c -= (c - c_cur);
                                            int W_cl = (int)W_c;
                                            if (W_cl < 0) W_cl = 0;
                                            if (W_cl > m) W_cl = m;
                                            int ell_idx = b_ell - 2;
                                            int32_t thresh = threshold_table_smem[ell_idx * (m + 1) + W_cl];
                                            if (ws_c <= thresh) all_killed = false;
                                        }
                                    }

                                    if (all_killed) {
                                        batch_skip_smem = true;
                                        /* Count remaining children in inner sweep.
                                         * Current child already counted; skip rest. */
                                        batch_skip_count_smem = R_inner - 1 -
                                            (gc_a_smem[0] > 0 ? gc_a_smem[0] : 0);

                                        /* Reset inner GC digit. */
                                        gc_a_smem[0] = 0;
                                        gc_dir_smem[0] = 1;
                                        gc_focus_smem[0] = gc_focus_smem[1];
                                        gc_focus_smem[1] = 1;
                                        cursor_smem[ipos] = lo_i;
                                        child_smem[ik1] = lo_i;
                                        child_smem[ik2] = a_ipos - lo_i;
                                    }
                                }
                            }
                        }
                    } else {
                        kill_flag_smem = (int)blockDim.x;
                        need_full_conv_smem = true;
                    }
                }
            }
            __syncthreads();   /* ── BARRIER #1 ── */
            if (gc_done_smem) break;

            int gc_j = gc_j_smem;

            children_tested++;
            watchdog++;

            /* ═══ Idea 2: Handle batch skip ═══
             * If the batch QC killed all remaining inner children,
             * account for them, recompute conv from new child state,
             * refresh the cache, and continue to next outer step. */
            if (batch_skip_smem) {
                if (lane == 0) {
                    children_tested += batch_skip_count_smem;
                    watchdog += batch_skip_count_smem;
                }
                /* Child state was reset by lane 0 in the batch check.
                 * Recompute full conv for the new starting point. */
                cooperative_full_autoconv(child_smem, raw_conv_smem,
                                          d_child, conv_len);
                /* Refresh ALL QC cache slots from fresh conv. */
                if (lane == 0) {
                    for (int w = 0; w < qc_cache_count_smem; w++) {
                        int ell_w = qc_cache_ell[w];
                        int s_w   = qc_cache_s[w];
                        if (ell_w == 0) continue;
                        int32_t ws = 0;
                        int n_cv = ell_w - 1;
                        for (int kk = s_w; kk < s_w + n_cv; kk++)
                            ws += raw_conv_smem[kk];
                        qc_cache_ws_lazy[w] = ws;
                        int qc_lo2 = s_w - (d_child - 1);
                        if (qc_lo2 < 0) qc_lo2 = 0;
                        int qc_hi2 = s_w + ell_w - 2;
                        if (qc_hi2 > d_child - 1) qc_hi2 = d_child - 1;
                        int32_t W = 0;
                        for (int ii = qc_lo2; ii <= qc_hi2; ii++)
                            W += child_smem[ii];
                        qc_cache_W_int[w] = W;
                        precompute_qc_sensitivity(
                            child_smem, d_child, ell_w, s_w,
                            active_pos_smem, n_active,
                            &qc_cache_sensitivity[w],
                            &qc_cache_mask_k1[w],
                            &qc_cache_mask_k2[w],
                            &qc_cache_mask_m[w],
                            QC_CACHE_K);
                    }
                }
                __syncthreads();
                continue;
            }

#ifdef TRACE
            if (lane == 0 && pid == 0 && (watchdog <= 3 || watchdog % 10000 == 0)) {
                printf("[TRACE] gc step %lld/%lld, pos=%d, j=%d, qc=%s\n",
                       (long long)watchdog, (long long)expected_total, pos, gc_j,
                       need_full_conv_smem ? "MISS" : "HIT");
            }
#endif

            /* ═══ STEP 2: QC killed → skip conv + scan entirely ═══ */
            if (!need_full_conv_smem) {
                /* QC killed this child.  No conv update needed because:
                 * - Lazy QC tracks ws analytically (Idea 1), never reads conv[]
                 * - On next QC miss, cooperative_full_autoconv recomputes from scratch
                 * Cost: just 1 barrier for the entire child. */
                goto step_subtree;
            }

            /* ═══ STEP 3: QC missed → recompute conv + full scan ═══ */
            cooperative_full_autoconv(child_smem, raw_conv_smem,
                                      d_child, conv_len);
            /* ── BARRIER #2+#3 inside cooperative_full_autoconv ── */

            {
                /* Full window scan on fresh conv. */
                if (lane == 0) kill_flag_smem = (int)blockDim.x;
                __syncthreads();

                thread_private_window_scan(
                    raw_conv_smem, child_smem,
                    threshold_table_smem, ell_order_smem,
                    ell_count, conv_len, d_child, m,
                    &kill_flag_smem,
                    &qc_ell_smem, &qc_s_smem, &qc_W_int_smem);
                __syncthreads();   /* ── BARRIER #4 ── */
            }

            {
                bool pruned = (kill_flag_smem < (int)blockDim.x);

                /* Update QC cache with new killing window (Idea 4). */
                if (lane == 0 && pruned && qc_ell_smem > 0) {
                    /* Check if this window is already in the cache. */
                    bool already_cached = false;
                    for (int w = 0; w < qc_cache_count_smem; w++) {
                        if (qc_cache_ell[w] == qc_ell_smem &&
                            qc_cache_s[w] == qc_s_smem) {
                            already_cached = true;
                            /* Refresh this slot's ws_lazy and W_int. */
                            int32_t ws0 = 0;
                            int n_cv = qc_ell_smem - 1;
                            for (int kk = qc_s_smem; kk < qc_s_smem + n_cv; kk++)
                                ws0 += raw_conv_smem[kk];
                            qc_cache_ws_lazy[w] = ws0;
                            qc_cache_W_int[w] = qc_W_int_smem;
                            precompute_qc_sensitivity(
                                child_smem, d_child,
                                qc_ell_smem, qc_s_smem,
                                active_pos_smem, n_active,
                                &qc_cache_sensitivity[w],
                                &qc_cache_mask_k1[w],
                                &qc_cache_mask_k2[w],
                                &qc_cache_mask_m[w],
                                QC_CACHE_K);
                            break;
                        }
                    }
                    if (!already_cached) {
                        int slot = qc_cache_next_smem % QC_CACHE_K;
                        qc_cache_ell[slot] = qc_ell_smem;
                        qc_cache_s[slot]   = qc_s_smem;
                        qc_cache_W_int[slot] = qc_W_int_smem;

                        int32_t ws0 = 0;
                        int n_cv = qc_ell_smem - 1;
                        for (int kk = qc_s_smem; kk < qc_s_smem + n_cv; kk++)
                            ws0 += raw_conv_smem[kk];
                        qc_cache_ws_lazy[slot] = ws0;

                        precompute_qc_sensitivity(
                            child_smem, d_child,
                            qc_ell_smem, qc_s_smem,
                            active_pos_smem, n_active,
                            &qc_cache_sensitivity[slot],
                            &qc_cache_mask_k1[slot],
                            &qc_cache_mask_k2[slot],
                            &qc_cache_mask_m[slot],
                            QC_CACHE_K);

                        qc_cache_next_smem++;
                        if (qc_cache_count_smem < QC_CACHE_K)
                            qc_cache_count_smem++;
                    }
                }

                /* Also refresh ALL existing cache slots' ws_lazy from
                 * fresh conv (we just did full autoconv, so conv is exact). */
                if (lane == 0) {
                    for (int w = 0; w < qc_cache_count_smem; w++) {
                        int ell_w = qc_cache_ell[w];
                        int s_w   = qc_cache_s[w];
                        if (ell_w == 0) continue;
                        int32_t ws = 0;
                        int n_cv = ell_w - 1;
                        for (int kk = s_w; kk < s_w + n_cv; kk++)
                            ws += raw_conv_smem[kk];
                        qc_cache_ws_lazy[w] = ws;

                        /* Recompute W_int from child for this window. */
                        int qc_lo = s_w - (d_child - 1);
                        if (qc_lo < 0) qc_lo = 0;
                        int qc_hi = s_w + ell_w - 2;
                        if (qc_hi > d_child - 1) qc_hi = d_child - 1;
                        int32_t W = 0;
                        for (int ii = qc_lo; ii <= qc_hi; ii++)
                            W += child_smem[ii];
                        qc_cache_W_int[w] = W;

                        /* Recompute sensitivity from current child. */
                        precompute_qc_sensitivity(
                            child_smem, d_child,
                            ell_w, s_w,
                            active_pos_smem, n_active,
                            &qc_cache_sensitivity[w],
                            &qc_cache_mask_k1[w],
                            &qc_cache_mask_k2[w],
                            &qc_cache_mask_m[w],
                            QC_CACHE_K);
                    }
                }

                /* ═══ STEP 4: Collect survivor ═══ */
                if (!pruned) {
                    canonicalize_and_stage(
                        child_smem, surv_buf_smem, &surv_count_smem,
                        d_child, cmp_array_smem, &use_rev_smem, &slot_smem);
                    if (surv_count_smem >= surv_cap) {
                        flush_survivors_to_global(
                            surv_buf_smem, &surv_count_smem,
                            g_survivors, g_survivor_count,
                            d_child, max_survivors, &flush_base_smem);
                    }
                }
            }

            step_subtree:

            /* ═══ STEP 6: Multi-level subtree pruning (Idea 01) ═══
             *
             * Check at EVERY Gray code level gc_j >= J_MIN_LOWEST (=2),
             * not just at a single fixed level.  When digit j advances,
             * digits 0..j-1 are about to sweep through subtree_size[j]
             * children.  If the partial autoconvolution of the fixed
             * prefix (child bins 0..fixed_len-1) already exceeds the
             * threshold for all possible inner configurations, the
             * entire subtree is skipped.
             *
             * Cost/benefit: the check costs O(fixed_len^2).  Only
             * performed if fixed_len >= 2 and either fixed_len >= 4
             * (always worth it) or subtree_size > 4 * fixed_len^2.
             *
             * Correctness: identical to the CPU multi-level check in
             * _fused_generate_and_prune_gray (line 1637).  The partial
             * conv is a lower bound on the full conv; W_int_max is an
             * upper bound on W_int; threshold is monotone in W_int.
             */
            {
                if (lane == 0)
                    subtree_killed_smem = false;
                __syncthreads();

                if (gc_j >= 2 && n_active > gc_j) {
                    if (lane == 0) {
                        int fixed_parent_boundary = active_pos_smem[gc_j - 1];
                        int fixed_len = 2 * fixed_parent_boundary;

                        /* Cost/benefit: match CPU J_MIN_LOWEST=2, SUBTREE_COST_MULT=4 */
                        bool do_check = (fixed_len >= 2);
                        if (do_check && fixed_len < 4) {
                            int64_t st_size = subtree_size_smem[gc_j];
                            int64_t cost = (int64_t)fixed_len * (int64_t)fixed_len;
                            do_check = (st_size > 4 * cost);
                        }

                        if (do_check) {
                            int pconv_len = 2 * fixed_len - 1;
                            for (int kk = 0; kk < pconv_len; kk++)
                                prefix_conv_smem[kk] = 0;
                            for (int ii = 0; ii < fixed_len; ii++) {
                                int ci = child_smem[ii];
                                if (ci == 0) continue;
                                prefix_conv_smem[2 * ii] += ci * ci;
                                for (int jj = ii + 1; jj < fixed_len; jj++) {
                                    int cj = child_smem[jj];
                                    if (cj != 0)
                                        prefix_conv_smem[ii + jj] += 2 * ci * cj;
                                }
                            }

                            for (int kk = 1; kk < pconv_len; kk++)
                                prefix_conv_smem[kk] += prefix_conv_smem[kk - 1];

                            prefix_c_smem[0] = 0;
                            for (int ii = 0; ii < fixed_len; ii++)
                                prefix_c_smem[ii + 1] = prefix_c_smem[ii] +
                                                        child_smem[ii];

                            int first_unfixed_parent = fixed_parent_boundary;

                            /* ── Idea 02: guaranteed minimum contributions ──
                             *
                             * Compute lower-bound conv contributions from
                             * unfixed bins using their guaranteed minimum
                             * child masses.  Matches CPU min_contrib logic
                             * in _fused_generate_and_prune_gray.
                             */
                            for (int kk = 0; kk < conv_len; kk++)
                                min_contrib_smem[kk] = 0;

                            /* (A) Inner active positions (digits 0..gc_j-1) */
                            for (int kk = 0; kk < gc_j; kk++) {
                                int p_unf = active_pos_smem[kk];
                                int k1u = 2 * p_unf;
                                int k2u = 2 * p_unf + 1;
                                int ml = lo_smem[p_unf];
                                int mh = parent_smem[p_unf] - hi_smem[p_unf];

                                /* Self-terms */
                                min_contrib_smem[2 * k1u] += ml * ml;
                                min_contrib_smem[2 * k2u] += mh * mh;
                                min_contrib_smem[k1u + k2u] += 2 * ml * mh;

                                /* Cross-terms with fixed bins */
                                for (int ii = 0; ii < fixed_len; ii++) {
                                    int ci = child_smem[ii];
                                    if (ci > 0) {
                                        if (ml > 0)
                                            min_contrib_smem[ii + k1u] += 2 * ci * ml;
                                        if (mh > 0)
                                            min_contrib_smem[ii + k2u] += 2 * ci * mh;
                                    }
                                }

                                /* Cross-terms with other inner unfixed */
                                for (int kk2 = kk + 1; kk2 < gc_j; kk2++) {
                                    int p2 = active_pos_smem[kk2];
                                    int k1u2 = 2 * p2;
                                    int k2u2 = 2 * p2 + 1;
                                    int ml2 = lo_smem[p2];
                                    int mh2 = parent_smem[p2] - hi_smem[p2];
                                    if (ml > 0 && ml2 > 0)
                                        min_contrib_smem[k1u + k1u2] += 2 * ml * ml2;
                                    if (ml > 0 && mh2 > 0)
                                        min_contrib_smem[k1u + k2u2] += 2 * ml * mh2;
                                    if (mh > 0 && ml2 > 0)
                                        min_contrib_smem[k2u + k1u2] += 2 * mh * ml2;
                                    if (mh > 0 && mh2 > 0)
                                        min_contrib_smem[k2u + k2u2] += 2 * mh * mh2;
                                }
                            }

                            /* (B) Non-active unfixed parents (range==1,
                             *     beyond fixed prefix).  Their child values
                             *     are fixed but not in partial_conv. */
                            for (int pp = first_unfixed_parent; pp < d_parent; pp++) {
                                /* Skip if this is an inner active position */
                                bool is_inner = false;
                                for (int kk = 0; kk < gc_j; kk++) {
                                    if (active_pos_smem[kk] == pp) {
                                        is_inner = true;
                                        break;
                                    }
                                }
                                if (is_inner) continue;

                                int k1na = 2 * pp;
                                int k2na = 2 * pp + 1;
                                int cv1 = lo_smem[pp];
                                int cv2 = parent_smem[pp] - cv1;

                                /* Self-terms */
                                min_contrib_smem[2 * k1na] += cv1 * cv1;
                                min_contrib_smem[2 * k2na] += cv2 * cv2;
                                min_contrib_smem[k1na + k2na] += 2 * cv1 * cv2;

                                /* Cross with fixed prefix */
                                for (int ii = 0; ii < fixed_len; ii++) {
                                    int ci = child_smem[ii];
                                    if (ci > 0) {
                                        if (cv1 > 0)
                                            min_contrib_smem[ii + k1na] += 2 * ci * cv1;
                                        if (cv2 > 0)
                                            min_contrib_smem[ii + k2na] += 2 * ci * cv2;
                                    }
                                }

                                /* Cross with inner active unfixed */
                                for (int kk = 0; kk < gc_j; kk++) {
                                    int p_unf = active_pos_smem[kk];
                                    int k1u = 2 * p_unf;
                                    int k2u = 2 * p_unf + 1;
                                    int ml = lo_smem[p_unf];
                                    int mh = parent_smem[p_unf] - hi_smem[p_unf];
                                    if (cv1 > 0 && ml > 0)
                                        min_contrib_smem[k1na + k1u] += 2 * cv1 * ml;
                                    if (cv1 > 0 && mh > 0)
                                        min_contrib_smem[k1na + k2u] += 2 * cv1 * mh;
                                    if (cv2 > 0 && ml > 0)
                                        min_contrib_smem[k2na + k1u] += 2 * cv2 * ml;
                                    if (cv2 > 0 && mh > 0)
                                        min_contrib_smem[k2na + k2u] += 2 * cv2 * mh;
                                }

                                /* Cross with other non-active unfixed */
                                for (int pp2 = pp + 1; pp2 < d_parent; pp2++) {
                                    bool is_inner2 = false;
                                    for (int kk = 0; kk < gc_j; kk++) {
                                        if (active_pos_smem[kk] == pp2) {
                                            is_inner2 = true;
                                            break;
                                        }
                                    }
                                    if (is_inner2) continue;
                                    int k1na2 = 2 * pp2;
                                    int k2na2 = 2 * pp2 + 1;
                                    int cv12 = lo_smem[pp2];
                                    int cv22 = parent_smem[pp2] - cv12;
                                    if (cv1 > 0 && cv12 > 0)
                                        min_contrib_smem[k1na + k1na2] += 2 * cv1 * cv12;
                                    if (cv1 > 0 && cv22 > 0)
                                        min_contrib_smem[k1na + k2na2] += 2 * cv1 * cv22;
                                    if (cv2 > 0 && cv12 > 0)
                                        min_contrib_smem[k2na + k1na2] += 2 * cv2 * cv12;
                                    if (cv2 > 0 && cv22 > 0)
                                        min_contrib_smem[k2na + k2na2] += 2 * cv2 * cv22;
                                }
                            }

                            /* Convert min_contrib to inclusive prefix sum */
                            for (int kk = 1; kk < conv_len; kk++)
                                min_contrib_smem[kk] += min_contrib_smem[kk - 1];

                            bool sub_killed = partial_window_scan_max_threshold(
                                prefix_conv_smem, pconv_len, fixed_len,
                                threshold_table_smem, ell_order_smem,
                                ell_count, m, d_child,
                                prefix_c_smem, parent_prefix_smem,
                                first_unfixed_parent, d_parent,
                                min_contrib_smem, conv_len);

                            if (sub_killed) {
                                subtree_killed_smem = true;

                                n_skipped += subtree_size_smem[gc_j] - 1;

                                int next_focus = gc_focus_smem[gc_j];
                                for (int kk = 0; kk < gc_j; kk++) {
                                    gc_a_smem[kk]     = 0;
                                    gc_dir_smem[kk]   = 1;
                                    gc_focus_smem[kk]  = kk;
                                }
                                gc_focus_smem[0]    = next_focus;
                                gc_focus_smem[gc_j] = gc_j;

                                for (int kk = 0; kk < gc_j; kk++) {
                                    int p = active_pos_smem[kk];
                                    cursor_smem[p] = lo_smem[p];
                                    child_smem[2 * p]     = lo_smem[p];
                                    child_smem[2 * p + 1] = parent_smem[p] -
                                                             lo_smem[p];
                                }
                            }
                        }
                    }
                    __syncthreads();

                    if (subtree_killed_smem) {
                        cooperative_full_autoconv(child_smem, raw_conv_smem,
                                                 d_child, conv_len);

                        /* Refresh ALL QC cache slots from fresh conv. */
                        if (lane == 0) {
                            for (int w = 0; w < qc_cache_count_smem; w++) {
                                int ell_w = qc_cache_ell[w];
                                int s_w   = qc_cache_s[w];
                                if (ell_w == 0) continue;
                                int32_t ws = 0;
                                int n_cv = ell_w - 1;
                                for (int kk = s_w; kk < s_w + n_cv; kk++)
                                    ws += raw_conv_smem[kk];
                                qc_cache_ws_lazy[w] = ws;
                                int qc_lo2 = s_w - (d_child - 1);
                                if (qc_lo2 < 0) qc_lo2 = 0;
                                int qc_hi2 = s_w + ell_w - 2;
                                if (qc_hi2 > d_child - 1) qc_hi2 = d_child - 1;
                                int32_t W = 0;
                                for (int ii = qc_lo2; ii <= qc_hi2; ii++)
                                    W += child_smem[ii];
                                qc_cache_W_int[w] = W;
                                precompute_qc_sensitivity(
                                    child_smem, d_child, ell_w, s_w,
                                    active_pos_smem, n_active,
                                    &qc_cache_sensitivity[w],
                                    &qc_cache_mask_k1[w],
                                    &qc_cache_mask_k2[w],
                                    &qc_cache_mask_m[w],
                                    QC_CACHE_K);
                            }
                        }
                        __syncthreads();
                        continue;
                    }
                }
            } /* end multi-level subtree pruning */

#ifdef DEBUG
            if (lane == 0 && pid == 0 && watchdog <= 3)
                printf("[block %d] step %lld: iteration complete\n",
                       blockIdx.x, (long long)watchdog);
            __syncthreads();
#endif

        } /* end Gray code enumeration */

#ifdef TRACE
        if (lane == 0) {
            printf("[TRACE] parent %d DONE: tested=%lld skipped=%lld "
                   "expected=%lld surv_pending=%d\n",
                   pid, (long long)children_tested, (long long)n_skipped,
                   (long long)expected_total, surv_count_smem);
        }
        __syncthreads();
#endif

        /* ── Phase 7: Flush remaining survivors ── */
        flush_survivors_to_global(
            surv_buf_smem, &surv_count_smem,
            g_survivors, g_survivor_count,
            d_child, max_survivors, &flush_base_smem);

#ifdef TRACE
        /* Verify enumeration completeness. */
        if (lane == 0) {
            int64_t expected = 1;
            for (int j = 0; j < n_active; j++)
                expected *= radix_smem[j];
            if (children_tested + n_skipped != expected) {
                printf("[ERROR] parent %d: ENUMERATION MISMATCH tested=%lld "
                       "skipped=%lld expected=%lld\n",
                       pid, (long long)children_tested,
                       (long long)n_skipped, (long long)expected);
            }
        }
        __syncthreads();
#endif

        /* Signal parent completion for host progress monitor. */
        if (lane == 0)
            atomicAdd(g_done_parent, 1);
    } /* end persistent block loop */
}
