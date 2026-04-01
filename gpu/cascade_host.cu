/*
 * cascade_host.cu — Host-side helpers and kernel launcher.
 *
 * Provides:
 *   - build_threshold_table(): precompute int64 thresholds on CPU
 *   - build_ell_order():       precompute optimised ell scan order
 *   - launch_cascade_kernel(): grid config + launch
 *   - main():                  end-to-end driver (load parents, launch, save)
 *
 * Build (combined with cascade_kernel.cu):
 *   nvcc -arch=sm_90 -O3 -ftz=false -prec-div=true -prec-sqrt=true \
 *        -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu    \
 *        -o cascade_prover
 */

#include "cascade_kernel.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>

/* Forward declaration of the kernel (defined in cascade_kernel.cu). */
extern __global__ void cascade_kernel(
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
    int max_survivors);

/* g_next_parent is a global-memory int32 counter, allocated alongside
 * survivor_count.  The host monitors it for progress reporting. */

/* ═══════════════════════════════════════════════════════════════════
 *  CUDA error checking
 * ═══════════════════════════════════════════════════════════════════ */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return -1;                                                    \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_VOID(call)                                             \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
        }                                                                 \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════
 *  build_threshold_table
 *
 *  Precomputes int64 thresholds on the CPU.  All terms are scaled
 *  by ell/(4n):
 *
 *    dyn_x = (c_target * m^2 + 1.0 + eps_margin + 2.0 * W_int) * ell / (4*n)
 *    dyn_it = (int64_t)(dyn_x * one_minus_4eps)
 *
 *  Derivation: test-value prune condition TV > c_target + 1/m^2 + 2*W/m,
 *  multiplied by m^2*ell/(4n).  See Verification 4, proof/part1_framework.md.
 *  Matches the CPU fused kernel (_fused_generate_and_prune_gray).
 *
 *  where:
 *    n = n_half_child = d_child / 2
 *    eps_margin   = 1e-9 * m^2
 *    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
 *    ell ranges from 2 to 2*d_child (ell_idx = ell - 2)
 *    W_int ranges from 0 to m
 * ═══════════════════════════════════════════════════════════════════ */
void build_threshold_table(int32_t* table,
                           int d_child, int m, double c_target)
{
    int n_half_child = d_child / 2;
    double m_d = (double)m;
    double inv_4n = 1.0 / (4.0 * (double)n_half_child);
    double DBL_EPS = 2.220446049250313e-16;
    double one_minus_4eps = 1.0 - 4.0 * DBL_EPS;
    double eps_margin = 1e-9 * m_d * m_d;
    double dyn_base = c_target * m_d * m_d + 1.0 + eps_margin;

    for (int ell = 2; ell <= 2 * d_child; ell++) {
        int ell_idx = ell - 2;
        double ell_scale = (double)ell * inv_4n;
        double dyn_base_ell = dyn_base * ell_scale;
        double two_ell_inv_4n = 2.0 * ell_scale;
        for (int w = 0; w <= m; w++) {
            double dyn_x = dyn_base_ell + two_ell_inv_4n * (double)w;
            /* int32 is safe: max threshold = (c_target*m^2 + 1 + eps + 2*m)
             * * max_ell/(4*n) ~ 601 for m=20, ~56401 for m=200.
             * All values << INT32_MAX (2,147,483,647). */
            table[ell_idx * (m + 1) + w] = (int32_t)(dyn_x * one_minus_4eps);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  build_ell_order
 *
 *  Matches CPU reference (_fused_generate_and_prune_gray, lines 1119-1160):
 *    - For d_child >= 20: profile-guided order centred at hc = d_child/2
 *    - For d_child < 20: sequential 2..16 then wide windows
 *    - Remaining ells in ascending order
 * ═══════════════════════════════════════════════════════════════════ */
int build_ell_order(int32_t* ell_order, int d_child)
{
    int ell_count = 2 * d_child - 1;
    std::vector<bool> used(ell_count, false);
    int oi = 0;

    auto try_add = [&](int ell) {
        if (ell >= 2 && ell <= 2 * d_child && !used[ell - 2]) {
            ell_order[oi++] = ell;
            used[ell - 2] = true;
        }
    };

    if (d_child >= 40) {
        /* Extended profile-guided order for d_child=64.
         * At d=64, hc=32.  Kill rates shift: wider spread around center,
         * and medium-width windows (ell ~ d_child/2 to d_child) contribute
         * more than at d=32.  Phase 1 covers a broader range around hc
         * and adds medium windows earlier. */
        int hc = d_child / 2;
        /* Phase 1: core killing range — wider than d=32 */
        int phase1[] = { hc+1, hc+2, hc+3, hc, hc-1, hc+4, hc+5,
                         hc-2, hc+6, hc-3, hc+7, hc+8, hc-4, hc+9,
                         hc-5, hc+10, hc-6, hc+11, hc-7, hc+12 };
        for (int e : phase1) try_add(e);

        /* Phase 2: medium windows — these kill at d=64 more than d=32 */
        int phase2[] = { d_child*3/4, d_child*3/4+1, d_child*3/4-1,
                         d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2,
                         d_child/4, d_child/4+1, d_child/4-1 };
        for (int e : phase2) try_add(e);
    } else if (d_child >= 20) {
        int hc = d_child / 2;
        /* Profile-guided order: kill-rate descending at d_child=32 */
        int phase1[] = { hc+1, hc+2, hc+3, hc, hc-1, hc+4, hc+5,
                         hc-2, hc+6, hc-3, hc+7, hc+8 };
        for (int e : phase1) try_add(e);

        /* Phase 2: wide windows around d_child */
        int phase2[] = { d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2 };
        for (int e : phase2) try_add(e);
    } else {
        int phase1_end = std::min(16, 2 * d_child);
        for (int ell = 2; ell <= phase1_end; ell++)
            try_add(ell);
        int phase2[] = { d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2,
                         d_child/2 };
        for (int e : phase2) try_add(e);
    }

    /* Phase 3: everything remaining in ascending order. */
    for (int ell = 2; ell <= 2 * d_child; ell++)
        try_add(ell);

    return oi;  /* should == ell_count */
}

/* ═══════════════════════════════════════════════════════════════════
 *  launch_cascade_kernel
 * ═══════════════════════════════════════════════════════════════════ */
int launch_cascade_kernel(const CascadeParams* p)
{
    /* Allocate the parent counter and survivor counter as host-mapped
     * (zero-copy) memory so the host can read them directly without
     * cudaMemcpy (which would block on the running kernel). */
    int32_t *h_next_parent, *d_next_parent;
    int32_t *h_progress_surv, *d_progress_surv;
    CUDA_CHECK(cudaHostAlloc(&h_next_parent, sizeof(int32_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_next_parent, h_next_parent, 0));
    *h_next_parent = 0;

    CUDA_CHECK(cudaHostAlloc(&h_progress_surv, sizeof(int32_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress_surv, h_progress_surv, 0));
    *h_progress_surv = 0;

    /* Grid configuration.
     * Always at least 32 threads (one full warp) so that warp-level
     * primitives (__ballot_sync, __shfl_down_sync) work correctly. */
    int block_size = (p->d_child < 32) ? 32 : p->d_child;

    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int sm_count = prop.multiProcessorCount;
    printf("  GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, sm_count, prop.major, prop.minor);

    /* Dynamic shared memory = threshold_table + ell_order. */
    size_t smem_threshold = (size_t)p->ell_count * (p->m + 1) * sizeof(int32_t);
    size_t smem_ell_order = (size_t)p->ell_count * sizeof(int32_t);
    size_t dynamic_smem_bytes = smem_threshold + smem_ell_order;

    /* Increase CUDA printf buffer.  The default 1MB fills quickly even
     * without explicit TRACE, because watchdog/error printfs exist.
     * A full buffer causes device threads to block on printf calls. */
#ifdef DEBUG
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
    printf("  DEBUG: printf buffer set to 64 MB\n");
#else
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024));
#endif

    /* Request max shared memory per block. */
    CUDA_CHECK(cudaFuncSetAttribute(
        cascade_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)dynamic_smem_bytes));

    /* Use CUDA occupancy API to determine optimal blocks per SM.
     * This accounts for shared memory, registers, and thread limits
     * on whatever GPU we're actually running on. */
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, cascade_kernel, block_size, dynamic_smem_bytes));
    if (blocks_per_sm <= 0) blocks_per_sm = 1;
    int grid_size = sm_count * blocks_per_sm;
    if (grid_size > p->num_parents)
        grid_size = p->num_parents;

    /* ── WDDM TDR detection ──
     * On Windows with WDDM driver model, the OS kills GPU kernels
     * exceeding TdrDelay (default ~2s).  Warn the user. */
    {
        #ifdef _WIN32
        /* cudaDeviceGetAttribute doesn't expose driver model directly.
         * prop.tccDriver == 0 means WDDM on Windows. */
        if (!prop.tccDriver) {
            printf("\n");
            printf("  *** WARNING: GPU is in WDDM mode (display driver). ***\n");
            printf("  Windows TDR will kill kernels running longer than ~2 seconds.\n");
            printf("  For large workloads, either:\n");
            printf("    1. Increase TdrDelay: HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay (DWORD, seconds)\n");
            printf("    2. Use a TCC-mode GPU (Tesla/datacenter) or Linux\n");
            printf("    3. Build without -DDEBUG to reduce kernel runtime\n");
            printf("\n");
            fflush(stdout);
        }
        #endif
    }

    printf("Launching cascade kernel:\n");
    printf("  parents:       %d\n", p->num_parents);
    printf("  d_parent=%d  d_child=%d  m=%d\n",
           p->d_parent, p->d_child, p->m);
    printf("  grid=%d  block=%d  blocks/SM=%d  dynamic_smem=%zu B\n",
           grid_size, block_size, blocks_per_sm, dynamic_smem_bytes);
    fflush(stdout);

    /* Launch the kernel on a separate stream so that cudaMemcpy on the
     * default stream can read progress counters while the kernel runs. */
    cudaStream_t kernel_stream;
    CUDA_CHECK(cudaStreamCreate(&kernel_stream));

    auto t0 = std::chrono::high_resolution_clock::now();

    cascade_kernel<<<grid_size, block_size, dynamic_smem_bytes, kernel_stream>>>(
        p->parents, p->lo_arrays, p->hi_arrays,
        p->threshold_table, p->ell_order,
        p->survivors, p->survivor_count,
        d_next_parent,
        d_progress_surv,        /* g_done_parent: completed parent counter */
        p->num_parents, p->d_parent, p->d_child, p->m,
        p->ell_count, p->conv_len,
        p->threshold_asym,
        p->max_survivors);

    CUDA_CHECK(cudaGetLastError());

    /* ── Progress monitor loop (main thread) ──
     * Reads host-mapped (zero-copy) counters directly — no cudaMemcpy
     * needed, so it doesn't block on the running kernel.
     * volatile reads ensure we see the GPU's atomic writes.
     *
     * CRITICAL: Also polls cudaStreamQuery() to detect kernel completion
     * or failure (e.g., WDDM TDR killing the kernel).  Without this,
     * a TDR-killed kernel leaves the progress counter frozen and the
     * monitor loop spins forever. */
    int num_parents_copy = p->num_parents;
    volatile int32_t* vol_done = (volatile int32_t*)h_progress_surv;
    int32_t prev_progress = -1;
    const int POLL_MS = 1000;              /* poll every 1s for kernel completion */
    const int PRINT_INTERVAL_MS = 30000;  /* print progress every 30s */
    const int MAX_STALL_MS = 120000;      /* 120s with no progress → stall */
    int ms_since_print = PRINT_INTERVAL_MS; /* print immediately on first poll */
    int ms_stalled = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));

        /* Check if the kernel has finished (or been killed by TDR). */
        cudaError_t query_err = cudaStreamQuery(kernel_stream);
        if (query_err == cudaSuccess) {
            /* Kernel finished — print final progress. */
            int32_t done = *vol_done;
            if (done > num_parents_copy) done = num_parents_copy;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t0).count();
            printf("\r     [%d/%d] (100.0%%) done  [%.1fs elapsed]\n",
                   done, num_parents_copy, elapsed_s);
            fflush(stdout);
            break;
        } else if (query_err != cudaErrorNotReady) {
            /* Kernel failed — likely TDR or device error. */
            fprintf(stderr, "\n  *** KERNEL FAILED: %s ***\n",
                    cudaGetErrorString(query_err));
            fprintf(stderr, "  This is likely caused by Windows WDDM TDR "
                    "(kernel exceeded ~2s timeout).\n");
            fprintf(stderr, "  Fix: increase TdrDelay registry key or "
                    "build without -DDEBUG.\n\n");
            fflush(stderr);
            cudaGetLastError();
            cudaStreamDestroy(kernel_stream);
            cudaFreeHost(h_next_parent);
            cudaFreeHost(h_progress_surv);
            return -1;
        }

        /* Kernel still running — read completed-parent counter. */
        int32_t progress = *vol_done;
        if (progress > num_parents_copy) progress = num_parents_copy;

        ms_since_print += POLL_MS;
        if (ms_since_print >= PRINT_INTERVAL_MS) {
            ms_since_print = 0;

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t0).count();
            double rate = (elapsed_s > 0) ? progress / elapsed_s : 0;
            double eta_s = (rate > 0) ? (num_parents_copy - progress) / rate : 0;
            double pct = (double)progress / num_parents_copy * 100.0;

            int eta_h = (int)(eta_s / 3600);
            int eta_m = (int)((eta_s - eta_h * 3600) / 60);
            int eta_sec = (int)(eta_s - eta_h * 3600 - eta_m * 60);

            printf("\r     [%d/%d] (%.1f%%) "
                   "%.0f parents/s, ETA %02d:%02d:%02d  [%.1fs elapsed]",
                   progress, num_parents_copy, pct,
                   rate, eta_h, eta_m, eta_sec, elapsed_s);
            fflush(stdout);
        }

        /* Detect stalls. */
        if (progress == prev_progress) {
            ms_stalled += POLL_MS;
            if (ms_stalled >= MAX_STALL_MS) {
                fprintf(stderr, "\n\n  *** TIMEOUT: kernel stalled for %.0fs "
                        "with no progress. ***\n",
                        (double)ms_stalled / 1000.0);
                fprintf(stderr, "  Progress frozen at %d/%d completed.\n",
                        progress, num_parents_copy);
                fprintf(stderr, "  Attempting cudaDeviceSynchronize() to "
                        "diagnose...\n");
                fflush(stderr);

                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    fprintf(stderr, "  cudaDeviceSynchronize returned: %s\n",
                            cudaGetErrorString(sync_err));
                    fprintf(stderr, "  ==> WDDM TDR killed the kernel. "
                            "Increase TdrDelay registry key.\n\n");
                } else {
                    fprintf(stderr, "  cudaDeviceSynchronize returned success "
                            "(kernel completed during sync).\n");
                }
                fflush(stderr);
                cudaGetLastError();
                cudaStreamDestroy(kernel_stream);
                cudaFreeHost(h_next_parent);
                cudaFreeHost(h_progress_surv);
                return -1;
            }
        } else {
            ms_stalled = 0;
            prev_progress = progress;
        }
    }

    /* cudaStreamQuery returned cudaSuccess above, so no sync needed,
     * but call it anyway to be safe and to surface any deferred errors. */
    CUDA_CHECK(cudaStreamSynchronize(kernel_stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  kernel time: %.1f ms (%.2f s)\n", elapsed_ms, elapsed_ms / 1000.0);

    cudaStreamDestroy(kernel_stream);
    cudaFreeHost(h_next_parent);
    cudaFreeHost(h_progress_surv);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Simple NumPy .npy loader for int32 2D arrays
 *
 *  Reads the standard .npy v1.0 format produced by np.save().
 *  Only supports int32, C-contiguous, 2D arrays.
 * ═══════════════════════════════════════════════════════════════════ */
static bool load_npy_int32(const char* path, std::vector<int32_t>& data,
                           int& rows, int& cols)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    /* Read 10-byte header prefix: \x93NUMPY + major + minor + header_len. */
    uint8_t prefix[10];
    if (fread(prefix, 1, 10, f) != 10) {
        fclose(f); return false;
    }
    if (prefix[0] != 0x93 || memcmp(prefix + 1, "NUMPY", 5) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path);
        fclose(f); return false;
    }
    uint16_t header_len = prefix[8] | ((uint16_t)prefix[9] << 8);
    std::vector<char> header(header_len + 1, '\0');
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f); return false;
    }

    /* Parse shape from header string.
     * Expected format: "{'descr': '<i4', 'fortran_order': False, 'shape': (R, C), }" */
    const char* sp = strstr(header.data(), "'shape': (");
    if (!sp) { fprintf(stderr, "%s: cannot parse shape\n", path); fclose(f); return false; }
    sp += strlen("'shape': (");

    rows = atoi(sp);
    const char* comma = strchr(sp, ',');
    if (!comma) {
        /* 1D array: shape (N,) — treat as (N, 1). */
        cols = 1;
    } else {
        cols = atoi(comma + 1);
    }

    size_t total = (size_t)rows * cols;
    data.resize(total);
    size_t nread = fread(data.data(), sizeof(int32_t), total, f);
    fclose(f);
    if (nread != total) {
        /* Truncated file (e.g. interrupted save).  Use what we have. */
        size_t actual_rows = nread / cols;
        fprintf(stderr, "%s: header says %d rows but file has %zu rows "
                "(truncated). Using %zu rows.\n",
                path, rows, actual_rows, actual_rows);
        rows = (int)actual_rows;
        data.resize(actual_rows * cols);
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Simple NumPy .npy saver for int32 2D arrays
 * ═══════════════════════════════════════════════════════════════════ */
static bool save_npy_int32(const char* path, const int32_t* data,
                           int rows, int cols)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return false; }

    /* Build header string. */
    char hdr_str[256];
    snprintf(hdr_str, sizeof(hdr_str),
             "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, %d), }",
             rows, cols);
    int hdr_len = (int)strlen(hdr_str);
    /* Pad to multiple of 64 (NumPy convention). */
    int total_hdr = 10 + hdr_len + 1;  /* +1 for newline */
    int pad = (64 - (total_hdr % 64)) % 64;
    hdr_len += pad + 1;  /* pad spaces + trailing newline */

    /* Write 10-byte prefix. */
    uint8_t prefix[10] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0 };
    prefix[8] = (uint8_t)(hdr_len & 0xFF);
    prefix[9] = (uint8_t)((hdr_len >> 8) & 0xFF);
    fwrite(prefix, 1, 10, f);

    /* Write header string + padding + newline. */
    fprintf(f, "%s", hdr_str);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);

    /* Write data. */
    fwrite(data, sizeof(int32_t), (size_t)rows * cols, f);
    fclose(f);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  _compute_bin_ranges — host-side range computation
 *
 *  Matches CPU reference (_compute_bin_ranges, lines 1516-1553).
 * ═══════════════════════════════════════════════════════════════════ */
static bool compute_bin_ranges(
    const int32_t* parent, int d_parent, int d_child,
    int m, double c_target,
    int32_t* lo_out, int32_t* hi_out, int64_t* total_children)
{
    /* correction(m, n_half_child):
     * Matches pruning.py correction() exactly:
     *   base = 2.0/m + 1.0/(m*m)
     *   factor = max(1.0, 4.0*n_half_child/ell_min)  where ell_min=2
     *   correction = factor * base */
    int n_half_child = d_child / 2;
    double base_corr = 2.0 / (double)m + 1.0 / ((double)m * (double)m);
    double factor = std::max(1.0, 4.0 * (double)n_half_child / 2.0);
    double corr = factor * base_corr;
    double thresh = c_target + corr + 1e-9;
    int x_cap = (int)floor((double)m * sqrt(thresh / (double)d_child));
    int x_cap_cs = (int)floor((double)m * sqrt(c_target / (double)d_child));
    x_cap = std::min(x_cap, x_cap_cs);
    x_cap = std::min(x_cap, m);
    x_cap = std::max(x_cap, 0);

    *total_children = 1;
    for (int i = 0; i < d_parent; i++) {
        int b_i = parent[i];
        int lo = std::max(0, b_i - x_cap);
        int hi = std::min(b_i, x_cap);
        if (lo > hi) return false;
        lo_out[i] = lo;
        hi_out[i] = hi;
        *total_children *= (int64_t)(hi - lo + 1);
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  main — End-to-end driver
 *
 *  Usage:
 *    ./cascade_prover <parents.npy> <output.npy> \
 *        --d_parent 32 --m 20 --c_target 1.4 [--max_survivors 200000]
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <parents.npy> <output.npy> "
            "--d_parent D --m M --c_target C [--max_survivors N]\n",
            argv[0]);
        return 1;
    }

    const char* parents_path = argv[1];
    const char* output_path  = argv[2];

    int    d_parent      = 32;
    int    m             = 20;
    double c_target      = 1.4;
    int    max_survivors = 200000;

    for (int i = 3; i < argc - 1; i++) {
        if (strcmp(argv[i], "--d_parent") == 0)
            d_parent = atoi(argv[++i]);
        else if (strcmp(argv[i], "--m") == 0)
            m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--c_target") == 0)
            c_target = atof(argv[++i]);
        else if (strcmp(argv[i], "--max_survivors") == 0)
            max_survivors = atoi(argv[++i]);
    }

    int d_child  = 2 * d_parent;
    int ell_count = 2 * d_child - 1;
    int conv_len  = 2 * d_child - 1;

    printf("Sidon Cascade GPU Prover\n");
    printf("  d_parent=%d  d_child=%d  m=%d  c_target=%.4f\n",
           d_parent, d_child, m, c_target);
    printf("  ell_count=%d  conv_len=%d  max_survivors=%d\n",
           ell_count, conv_len, max_survivors);

    /* ── Load parents ── */
    std::vector<int32_t> h_parents;
    int nrows, ncols;
    if (!load_npy_int32(parents_path, h_parents, nrows, ncols)) return 1;
    if (ncols != d_parent) {
        fprintf(stderr, "Parents array has %d columns, expected %d\n",
                ncols, d_parent);
        return 1;
    }
    int num_parents = nrows;
    printf("  Loaded %d parents from %s\n", num_parents, parents_path);

    /* ── Compute lo/hi arrays on host ── */
    printf("  Computing bin ranges...\n");
    std::vector<int32_t> h_lo(num_parents * d_parent);
    std::vector<int32_t> h_hi(num_parents * d_parent);
    std::vector<int>     valid_indices;
    valid_indices.reserve(num_parents);

    for (int i = 0; i < num_parents; i++) {
        int64_t tc;
        if (compute_bin_ranges(&h_parents[i * d_parent], d_parent, d_child,
                               m, c_target,
                               &h_lo[i * d_parent], &h_hi[i * d_parent],
                               &tc))
        {
            valid_indices.push_back(i);
        }
    }
    printf("  Valid parents (non-empty range): %zu / %d\n",
           valid_indices.size(), num_parents);

    /* Pack valid parents into contiguous arrays. */
    int n_valid = (int)valid_indices.size();
    std::vector<int32_t> pack_parents(n_valid * d_parent);
    std::vector<int32_t> pack_lo(n_valid * d_parent);
    std::vector<int32_t> pack_hi(n_valid * d_parent);
    for (int i = 0; i < n_valid; i++) {
        int src = valid_indices[i];
        memcpy(&pack_parents[i * d_parent],
               &h_parents[src * d_parent],
               d_parent * sizeof(int32_t));
        memcpy(&pack_lo[i * d_parent],
               &h_lo[src * d_parent],
               d_parent * sizeof(int32_t));
        memcpy(&pack_hi[i * d_parent],
               &h_hi[src * d_parent],
               d_parent * sizeof(int32_t));
    }

    /* ── Build threshold table ── */
    printf("  Building threshold table (%d x %d)...\n", ell_count, m + 1);
    std::vector<int32_t> h_threshold(ell_count * (m + 1));
    build_threshold_table(h_threshold.data(), d_child, m, c_target);

    /* ── Build ell order ── */
    std::vector<int32_t> h_ell_order(ell_count);
    int ell_written = build_ell_order(h_ell_order.data(), d_child);
    printf("  ell_order: %d entries\n", ell_written);

    /* ── Allocate GPU memory ── */
    printf("  Allocating GPU memory...\n");
    int32_t *d_parents, *d_lo, *d_hi, *d_survivors, *d_count;
    int32_t *d_threshold;
    int32_t *d_ell_order;

    size_t parent_bytes = (size_t)n_valid * d_parent * sizeof(int32_t);
    size_t thresh_bytes = (size_t)ell_count * (m + 1) * sizeof(int32_t);
    size_t ell_bytes    = (size_t)ell_count * sizeof(int32_t);
    size_t surv_bytes   = (size_t)max_survivors * d_child * sizeof(int32_t);

    CUDA_CHECK_VOID(cudaMalloc(&d_parents,   parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_lo,        parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_hi,        parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_threshold, thresh_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_ell_order, ell_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_survivors, surv_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_count,     sizeof(int32_t)));

    /* ── Copy to device ── */
    printf("  Copying data to GPU...\n");
    CUDA_CHECK_VOID(cudaMemcpy(d_parents, pack_parents.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_lo, pack_lo.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_hi, pack_hi.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_threshold, h_threshold.data(),
                               thresh_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_ell_order, h_ell_order.data(),
                               ell_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemset(d_count, 0, sizeof(int32_t)));

    /* ── Launch kernel ── */
    CascadeParams params;
    params.parents         = d_parents;
    params.lo_arrays       = d_lo;
    params.hi_arrays       = d_hi;
    params.threshold_table = d_threshold;
    params.ell_order       = d_ell_order;
    params.survivors       = d_survivors;
    params.survivor_count  = d_count;
    params.num_parents     = n_valid;
    params.d_parent        = d_parent;
    params.d_child         = d_child;
    params.m               = m;
    params.ell_count       = ell_count;
    params.conv_len        = conv_len;
    params.threshold_asym  = sqrt(c_target / 2.0);
    params.max_survivors   = max_survivors;

    int rc = launch_cascade_kernel(&params);
    if (rc != 0) {
        fprintf(stderr, "Kernel launch failed\n");
        return 1;
    }

    /* ── Read back results ── */
    int32_t h_count = 0;
    CUDA_CHECK_VOID(cudaMemcpy(&h_count, d_count,
                               sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("  Survivors found: %d\n", h_count);

    if (h_count > max_survivors) {
        fprintf(stderr, "WARNING: survivor_count (%d) > max_survivors (%d); "
                "output is truncated!\n", h_count, max_survivors);
        h_count = max_survivors;
    }

    std::vector<int32_t> h_survivors(h_count * d_child);
    if (h_count > 0) {
        CUDA_CHECK_VOID(cudaMemcpy(h_survivors.data(), d_survivors,
                                   (size_t)h_count * d_child * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
    }

    /* ── Save output ── */
    if (h_count > 0) {
        if (!save_npy_int32(output_path, h_survivors.data(), h_count, d_child))
            return 1;
        printf("  Saved %d survivors to %s\n", h_count, output_path);
    } else {
        printf("  No survivors — proof complete at this level!\n");
    }

    /* ── Cleanup ── */
    cudaFree(d_parents);
    cudaFree(d_lo);
    cudaFree(d_hi);
    cudaFree(d_threshold);
    cudaFree(d_ell_order);
    cudaFree(d_survivors);
    cudaFree(d_count);

    return 0;
}
