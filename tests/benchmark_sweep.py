"""Empirical cascade benchmark for optimal (m, n_half) selection.

Runs L0 fully, samples a fixed number of survivors, then cascades through
refinement levels (L1 -> L2 -> L3 ...) until convergence (0 survivors).
Records time and expansion factor at each level.  Projects full runtime
by linear scaling from the sample.

This is how we found m=20 to be optimal for proving c_target=1.4.

Usage:
    python tests/benchmark_sweep.py
    python tests/benchmark_sweep.py --m_values 10,20,30,40,50
    python tests/benchmark_sweep.py --dry_run
    python tests/benchmark_sweep.py --resume
    python tests/benchmark_sweep.py --sample_size 5000
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

import numpy as np

# Add the cloninger-steinerberger package directories to the path
# (the package uses direct imports without __init__.py)
_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions
from run_cascade import run_level0, process_parent_fused


# Known upper bound on c (Yu 2021). If c_target + correction exceeds this,
# the proof is vacuously true (the discretization error is too large
# to distinguish near-optimal functions) and the benchmark is meaningless.
C_UPPER_BOUND = 1.5029


# ---------------------------------------------------------------------------
# x_cap and refinement count
# ---------------------------------------------------------------------------

def compute_x_cap(m, d_child, c_target, n_half_child):
    """Compute the single-bin energy cap matching run_cascade logic.

    Fine grid: bin height = c_i / (4*n*m), energy = sum c_i^2 / (4*n*m^2).
    Single-bin Cauchy-Schwarz: c_i <= m * sqrt(4 * d_child * thresh).
    """
    corr = correction(m, n_half_child)
    thresh = c_target + corr + 1e-9
    x_cap = int(math.floor(m * math.sqrt(4 * d_child * thresh)))
    # Cauchy-Schwarz bound: +1 for canonical rounding adjustment
    x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child * c_target))) + 1
    x_cap = min(x_cap, x_cap_cs)
    x_cap = max(x_cap, 0)
    return x_cap


def compute_refs_per_parent(survivors, m, c_target, d_child, n_half_child):
    """Per-parent refinement counts (vectorized numpy).

    Parameters
    ----------
    survivors : np.ndarray of shape (N, d_parent), int32
    m : int
    c_target : float
    d_child : int
    n_half_child : int

    Returns
    -------
    np.ndarray of shape (N,), dtype=int64 : refinement count per parent.
    """
    if len(survivors) == 0:
        return np.array([], dtype=np.int64)

    x_cap = compute_x_cap(m, d_child, c_target, n_half_child)

    # For each parent bin b_i, child bins range from max(0, 2*b_i - x_cap)
    # to min(2*b_i, x_cap).  Factor of 2 because bin width halves while
    # mass is conserved: parent bin b_i splits into (c, 2*b_i - c).
    B = survivors.astype(np.int64)
    lo = np.maximum(0, 2 * B - x_cap)
    hi = np.minimum(2 * B, x_cap)
    eff = np.maximum(hi - lo + 1, 0)

    return np.prod(eff, axis=1)


def compute_total_refs(survivors, m, c_target, d_child, n_half_child):
    """Sum of per-parent refinement counts."""
    return int(np.sum(compute_refs_per_parent(
        survivors, m, c_target, d_child, n_half_child)))


# ---------------------------------------------------------------------------
# Refinement level runner (CPU)
# ---------------------------------------------------------------------------

def run_refinement_level(parent_configs, m, c_target, n_half_child,
                         time_budget_sec=0.0):
    """Run one refinement level on the CPU.

    Processes each parent via process_parent_fused().

    Parameters
    ----------
    parent_configs : np.ndarray of shape (N, d_parent), int32
    m : int
    c_target : float
    n_half_child : int
    time_budget_sec : float
        Per-level time budget in seconds (0 = no limit).

    Returns
    -------
    dict with: parents_in, survivors_out, survivor_configs,
               elapsed, total_children, throughput, expansion_factor,
               timed_out
    """
    parents_in = len(parent_configs)
    d_parent = parent_configs.shape[1]
    d_child = 2 * d_parent

    # Pre-filter infeasible parents (any bin > x_cap).
    # Cursor range for parent bin b_i is [max(0, 2*b_i - x_cap), min(2*b_i, x_cap)].
    # Empty when b_i > x_cap (since lo > hi iff 2*b_i > 2*x_cap).
    x_cap = compute_x_cap(m, d_child, c_target, n_half_child)
    max_bin_val = x_cap
    feasible_mask = np.all(parent_configs <= max_bin_val, axis=1)
    parent_configs = parent_configs[feasible_mask]
    n_filtered = parents_in - len(parent_configs)
    parents_in_eff = len(parent_configs)

    t0 = time.time()
    all_survivors = []
    total_children = 0
    timed_out = False

    for i, parent in enumerate(parent_configs):
        if time_budget_sec > 0 and (time.time() - t0) > time_budget_sec:
            timed_out = True
            break

        surv, n_children = process_parent_fused(
            parent, m, c_target, n_half_child)
        total_children += n_children
        if len(surv) > 0:
            all_survivors.append(surv)

        # Progress every 10% or 60s
        if (i + 1) % max(1, parents_in_eff // 10) == 0:
            elapsed_so_far = time.time() - t0
            rate = (i + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (parents_in_eff - i - 1) / rate if rate > 0 else 0
            n_surv = sum(len(s) for s in all_survivors)
            print(f"       {i+1:,}/{parents_in_eff:,} parents "
                  f"| {n_surv:,} survivors | ETA {_fmt_time(eta)}")

    elapsed = time.time() - t0

    if all_survivors:
        survivor_configs = np.vstack(all_survivors)
    else:
        survivor_configs = np.empty((0, d_child), dtype=np.int32)

    survivors_out = len(survivor_configs)
    # When timed out, break happens BEFORE processing parent i,
    # so parents 0..i-1 were completed → count = i.
    # When not timed out, all parents_in_eff were completed.
    if parents_in_eff == 0:
        feasible_completed = 0
    elif timed_out:
        feasible_completed = i  # parents 0..i-1 completed
    else:
        feasible_completed = parents_in_eff

    # Include filtered (infeasible) parents proportionally in expansion.
    # Infeasible parents produce 0 survivors, so the overall expansion
    # must account for them.  Scale: completed_including_infeasible =
    # feasible_completed * (parents_in / parents_in_eff).
    if parents_in_eff > 0 and feasible_completed > 0:
        scale = parents_in / parents_in_eff
        parents_completed_adj = feasible_completed * scale
    else:
        parents_completed_adj = max(feasible_completed, 1)

    throughput = total_children / elapsed if elapsed > 0 else 0
    expansion = (survivors_out / parents_completed_adj
                 if parents_completed_adj > 0 else 0)

    return {
        'parents_in': parents_in,
        'parents_completed': feasible_completed,
        'parents_completed_adj': parents_completed_adj,
        'parents_filtered': n_filtered,
        'survivors_out': survivors_out,
        'survivor_configs': survivor_configs,
        'elapsed': elapsed,
        'total_children': total_children,
        'throughput': throughput,
        'expansion_factor': expansion,
        'timed_out': timed_out,
    }


# ---------------------------------------------------------------------------
# Cascade pipeline
# ---------------------------------------------------------------------------

MAX_PARENTS_PER_LEVEL = 10_000
LEVEL_TIME_BUDGET = 300  # seconds per level (0 = no limit)


def run_cascade(n_half, m, c_target, max_levels=6):
    """Run L0 fully, then cascade refinement levels on sampled parents.

    Algorithm:
      1. Run L0 fully to get exact survivor count N0.
      2. At each subsequent level L_k:
         a. From actual survivors of the previous sample, randomly select
            min(SAMPLE_SIZE, n_actual) parents.
         b. Process those parents, measure wall time and count survivors.
         c. Compute expansion_factor = survivors_out / parents_processed.
         d. Project full-run survivors: projected_pop * expansion_factor.
         e. Project full-run time: elapsed * (projected_pop / parents_processed).
         f. The projected_pop for level L_k is the projected survivors from L_{k-1}.
      3. Pass actual survivors (not projected) to the next level as the
         sampling pool, so each level's sample is drawn from real data.

    This gives an unbiased estimate of both runtime and survivor count at
    each level, using only SAMPLE_SIZE operations per level.

    Returns
    -------
    dict with cascade results.
    """
    d0 = 2 * n_half
    corr = correction(m, n_half)
    n_total = count_compositions(d0, m)

    info = {
        'n_half': n_half,
        'm': m,
        'd0': d0,
        'correction': corr,
        'l0_compositions': n_total,
        'sample_size': MAX_PARENTS_PER_LEVEL,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"CONFIG: n_half={n_half}, m={m}, d0={d0}, "
          f"corr={corr:.6f}, L0 comps={n_total:,}", flush=True)
    print(f"{'='*60}", flush=True)

    # ---- Step 1: Level 0 (run fully — need exact survivor count) ----
    print(f"\n  [L0] Running Level 0...", flush=True)
    l0 = run_level0(n_half, m, c_target, verbose=True)

    info['l0_time'] = l0['elapsed']
    info['l0_survivors'] = l0['n_survivors']
    info['l0_pruned_asym'] = l0['n_pruned_asym']
    info['l0_pruned_test'] = l0['n_pruned_test']

    print(f"  [L0] Done in {l0['elapsed']:.2f}s: "
          f"{l0['n_survivors']:,} survivors", flush=True)

    if l0['proven']:
        info['proven_at'] = 'L0'
        info['levels'] = []
        info['projected_total'] = l0['elapsed']
        print(f"  [L0] PROVEN at Level 0!", flush=True)
        return info

    survivors = l0['survivors']
    N0 = len(survivors)

    # ---- Step 2: Cascade through refinement levels ----
    levels = []
    current_configs = survivors       # actual survivors available for sampling
    d_parent = d0
    n_half_parent = n_half
    level_num = 1
    proven_at = None

    # Use OS entropy for unbiased sampling; print seed for reproducibility
    seed = int.from_bytes(os.urandom(4), 'big')
    rng = np.random.RandomState(seed)
    print(f"  [Sampling] RNG seed={seed} (for reproducibility)", flush=True)

    # projected_population tracks the estimated number of parents entering
    # each level if we were running the full (non-sampled) cascade.
    projected_population = N0

    while level_num <= max_levels:
        d_child = 2 * d_parent
        n_half_child = 2 * n_half_parent

        n_available = len(current_configs)
        if n_available == 0:
            # No actual survivors to sample from — if projected_population > 0,
            # the sample was too small to capture any survivors.  Record this.
            if projected_population > 0:
                print(f"\n  [L{level_num}] WARNING: 0 actual survivors in "
                      f"sample but projected_population="
                      f"{_fmt_count(projected_population)}. "
                      f"Sample may be too small.", flush=True)
            break

        # Sample min(SAMPLE_SIZE, n_available) parents uniformly at random
        if n_available > MAX_PARENTS_PER_LEVEL:
            idx = rng.choice(n_available, size=MAX_PARENTS_PER_LEVEL,
                             replace=False)
            sampled = current_configs[idx]
            print(f"\n  [L{level_num}] Sampling {MAX_PARENTS_PER_LEVEL:,} / "
                  f"{n_available:,} actual survivors "
                  f"(projected pop: {_fmt_count(projected_population)})",
                  flush=True)
        else:
            sampled = current_configs
            print(f"\n  [L{level_num}] Using all {n_available:,} actual "
                  f"survivors (projected pop: "
                  f"{_fmt_count(projected_population)})", flush=True)

        parents_requested = len(sampled)

        # Compute median children per parent for display only
        refs = compute_refs_per_parent(sampled, m, c_target,
                                       d_child, n_half_child)
        median_refs = int(np.median(refs)) if len(refs) > 0 else 0

        # NO sorting — random order preserves unbiased expansion factor.
        # Sorting light-first would bias expansion LOW when timeout occurs,
        # because light parents have systematically fewer survivors/parent.

        print(f"  [L{level_num}] Refining: d={d_parent}->{d_child}, "
              f"{parents_requested:,} parents "
              f"(median {_fmt_count(median_refs)} children/parent)...",
              flush=True)

        level_result = run_refinement_level(
            sampled, m, c_target, n_half_child,
            time_budget_sec=LEVEL_TIME_BUDGET)

        # Use parents_completed_adj (includes proportional infeasible parents)
        parents_completed_adj = level_result['parents_completed_adj']
        parents_completed = level_result['parents_completed']

        # Expansion factor already accounts for infeasible parents
        expansion = level_result['expansion_factor']

        # Project what the FULL (non-sampled) run would produce:
        #   - Survivors: projected_population * expansion_factor
        #   - Time: measured_time * (projected_population / parents_completed_adj)
        # We scale by projected_population (what the full run would feed in),
        # not by n_available (actual sample survivors from previous level).
        projected_survivors = projected_population * expansion
        projected_time = (level_result['elapsed']
                          * (projected_population / parents_completed_adj)
                          if parents_completed_adj > 0 else float('inf'))

        timed_out = level_result['timed_out']
        lvl = {
            'level': level_num,
            'd_parent': d_parent,
            'd_child': d_child,
            'parents_processed': parents_completed,
            'parents_requested': parents_requested,
            'parents_available': n_available,
            'projected_parents': int(projected_population),
            'survivors_out': level_result['survivors_out'],
            'projected_survivors': int(projected_survivors),
            'elapsed': level_result['elapsed'],
            'projected_time': projected_time,
            'total_children': level_result['total_children'],
            'throughput': level_result['throughput'],
            'expansion_factor': expansion,
            'timed_out': timed_out,
        }
        levels.append(lvl)

        timeout_note = " [TIMED OUT]" if timed_out else ""
        print(f"  [L{level_num}] Done in {level_result['elapsed']:.2f}s: "
              f"{level_result['survivors_out']:,} survivors / "
              f"{level_result['total_children']:,} children "
              f"({parents_completed:,}/{parents_requested:,} parents), "
              f"expansion={expansion:.4f}x{timeout_note}", flush=True)
        print(f"  [L{level_num}] Projected full: "
              f"~{_fmt_count(projected_survivors)} survivors, "
              f"~{_fmt_time(projected_time)} time", flush=True)

        if level_result['survivors_out'] == 0:
            proven_at = f'L{level_num}'
            print(f"  [L{level_num}] PROVEN at Level {level_num} "
                  f"(0 survivors from {parents_requested:,} sampled "
                  f"parents)!", flush=True)
            break

        # Check for convergence: expansion < 1 means cascade is shrinking
        if expansion < 1.0 and not timed_out:
            print(f"  [L{level_num}] CONVERGING: expansion={expansion:.6f}x < 1 "
                  f"— cascade will eventually prove.", flush=True)

        # Advance to next level
        projected_population = projected_survivors
        current_configs = level_result['survivor_configs']
        d_parent = d_child
        n_half_parent = n_half_child
        level_num += 1

    info['levels'] = levels
    info['proven_at'] = proven_at

    # Project total time across all levels
    projected_total = info['l0_time']
    for lvl in levels:
        projected_total += lvl['projected_time']
    info['projected_total'] = projected_total

    print(f"\n  [Total] Projected full runtime: {_fmt_time(projected_total)}",
          flush=True)

    return info


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_cascade_table(info):
    """Print detailed cascade results for a single config."""
    n_half = info['n_half']
    m = info['m']
    d0 = info['d0']

    print(f"\nCONFIG: n_half={n_half}, m={m}, d0={d0}")
    print(f"  L0: {_fmt_time(info['l0_time'])}, "
          f"{_fmt_count(info['l0_survivors'])} survivors (full)")

    levels = info.get('levels', [])
    if not levels:
        if info.get('proven_at') == 'L0':
            print(f"  PROVEN at L0")
        return

    print()
    header = (f"  {'Level':>5} | {'Processed':>10} | {'Proj Parents':>12} | "
              f"{'Survivors':>10} | {'Factor':>10} | {'Time':>8} | "
              f"{'Proj Surv':>12} | {'Proj Time':>12}")
    print(header)
    print(f"  {'-'*100}")

    for lvl in levels:
        lnum = f"L{lvl['level']}"
        ef = lvl['expansion_factor']
        if ef == 0:
            factor = '0x'
        elif ef < 0.01:
            factor = f"{ef:.6f}x"
        else:
            factor = f"{ef:.4f}x"

        parents_key = ('parents_processed' if 'parents_processed' in lvl
                       else 'parents_sampled')
        proj_parents = lvl.get('projected_parents',
                               lvl.get('parents_available', 0))
        row = (f"  {lnum:>5} | "
               f"{_fmt_count(lvl[parents_key]):>10} | "
               f"{_fmt_count(proj_parents):>12} | "
               f"{_fmt_count(lvl['survivors_out']):>10} | "
               f"{factor:>10} | "
               f"{_fmt_time(lvl['elapsed']):>8} | "
               f"{_fmt_count(lvl.get('projected_survivors', 0)):>12} | "
               f"{_fmt_time(lvl.get('projected_time', 0)):>12}")
        print(row)

    # Final status
    last_level = levels[-1] if levels else None
    projected = info.get('projected_total')

    if info.get('proven_at'):
        if projected is not None:
            print(f"\n  Projected total: {_fmt_time(projected)}")
        print(f"  PROVEN — cascade converges at {info['proven_at']}")
    else:
        proj_surv = last_level.get('projected_survivors', 0) if last_level else 0
        last_d = last_level['d_child'] if last_level else 0
        if projected is not None:
            print(f"\n  Projected total time: {_fmt_time(projected)}")
        print(f"  NOT PROVEN — ~{_fmt_count(proj_surv)} projected survivors "
              f"remain at d={last_d}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    """Format seconds as human-readable string."""
    if seconds is None:
        return 'N/A'
    if seconds == float('inf'):
        return 'inf'
    if seconds < 60:
        return f'{seconds:.2f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}m'
    if seconds < 86400:
        return f'{seconds/3600:.2f}h'
    return f'{seconds/86400:.2f}d'


def _fmt_count(n):
    """Format large count with SI suffix."""
    if n is None:
        return 'N/A'
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f'{n/1000:.1f}K'
    if n < 1_000_000_000:
        return f'{n/1_000_000:.2f}M'
    return f'{n/1_000_000_000:.2f}B'


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def build_sweep_configs(m_values, n_half_values, c_target, force=False):
    """Build configs sorted cheapest-first by L0 composition count.

    Configs where the flat correction alone (legacy, without window scaling)
    makes the effective threshold exceed C_UPPER_BOUND are flagged as vacuous.
    We use the flat correction (2/m + 1/m^2) for this check because the actual
    per-window thresholds are tighter — the scaled correction(m, n_half) is a
    worst-case upper bound that overstates the effective threshold.
    """
    configs = []
    skipped = []
    for n_half in n_half_values:
        d = 2 * n_half
        for m in m_values:
            # Use flat correction for vacuity check (conservative)
            flat_corr = correction(m)  # 2/m + 1/m^2
            scaled_corr = correction(m, n_half)
            if not force and c_target + flat_corr >= C_UPPER_BOUND:
                skipped.append((n_half, m, flat_corr, c_target + flat_corr))
                continue
            n_total = count_compositions(d, m)
            configs.append({
                'n_half': n_half,
                'm': m,
                'd': d,
                'n_total': n_total,
                'correction_flat': flat_corr,
                'correction_scaled': scaled_corr,
            })
    if skipped:
        print(f"Skipping {len(skipped)} vacuous configs "
              f"(c_target + flat_correction >= {C_UPPER_BOUND}):")
        for n_half, m, corr, eff in skipped:
            print(f"  n_half={n_half}, m={m}: "
                  f"flat_correction={corr:.4f}, threshold={eff:.4f}")
    configs.sort(key=lambda c: c['n_total'])
    return configs


def run_sweep(configs, c_target, max_levels, output_dir, resume):
    """Run the full sweep, checkpointing after each config."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'benchmark_checkpoint.json')

    # Load checkpoint if resuming
    completed = {}
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        for r in ckpt.get('results', []):
            key = (r['n_half'], r['m'])
            completed[key] = r
        print(f"Resumed: {len(completed)} configs already completed",
              flush=True)

    results = list(completed.values())

    for i, cfg in enumerate(configs):
        key = (cfg['n_half'], cfg['m'])
        if key in completed:
            print(f"\n[{i+1}/{len(configs)}] SKIP n_half={cfg['n_half']}, "
                  f"m={cfg['m']} (already completed)", flush=True)
            continue

        print(f"\n[{i+1}/{len(configs)}] Running n_half={cfg['n_half']}, "
              f"m={cfg['m']} (max {MAX_PARENTS_PER_LEVEL:,} parents/level)...",
              flush=True)

        try:
            result = run_cascade(
                cfg['n_half'], cfg['m'], c_target,
                max_levels=max_levels)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            result = {
                'n_half': cfg['n_half'],
                'm': cfg['m'],
                'd0': cfg['d'],
                'error': str(e),
            }

        results.append(result)
        format_cascade_table(result)

        # Checkpoint
        _save_checkpoint(checkpoint_path, results, c_target,
                         [c['m'] for c in configs],
                         list(set(c['n_half'] for c in configs)))

    return results


def _save_checkpoint(path, results, c_target, m_values, n_half_values):
    """Save checkpoint JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "inf"
        return obj

    def convert_dict(d):
        return {k: convert(v) for k, v in d.items()}

    def convert_result(r):
        out = {}
        for k, v in r.items():
            if k == 'levels' and isinstance(v, list):
                out[k] = [convert_dict(lvl) for lvl in v]
            elif k == 'survivor_configs':
                continue  # Don't serialize large arrays to checkpoint
            else:
                out[k] = convert(v)
        return out

    ckpt = {
        'sweep_params': {
            'c_target': c_target,
            'max_parents_per_level': MAX_PARENTS_PER_LEVEL,
            'm_values': m_values,
            'n_half_values': n_half_values,
        },
        'results': [convert_result(r) for r in results],
    }
    with open(path, 'w') as f:
        json.dump(ckpt, f, indent=2, default=convert)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_summary_table(results):
    """Print cross-config comparison table with per-level expansion factors."""
    ok = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    def sort_key(r):
        t = r.get('projected_total')
        if t is None or t == float('inf') or t == 'inf':
            return float('inf')
        return t

    ok.sort(key=sort_key)

    max_lvls = max((len(r.get('levels', [])) for r in ok), default=0)

    print(f"\n{'='*130}")
    print(f"BENCHMARK SWEEP RESULTS  (max {MAX_PARENTS_PER_LEVEL:,} "
          f"parents/level, sorted by projected total time)")
    print(f"{'='*130}")

    # Build header
    hdr = f"{'n_half':>6} | {'m':>5} | {'L0 surv':>10} | {'L0 time':>8}"
    for i in range(1, max_lvls + 1):
        hdr += f" | {'L'+str(i)+' exp':>8} | {'L'+str(i)+' proj':>10}"
    hdr += f" | {'Status':>10} | {'Proj Total':>12}"
    print(hdr)
    print('-' * 130)

    for r in ok:
        l0_surv = r.get('l0_survivors', 0)
        l0_time = r.get('l0_time', 0)
        levels = r.get('levels', [])

        row = (f"{r['n_half']:>6} | {r['m']:>5} | "
               f"{_fmt_count(l0_surv):>10} | "
               f"{_fmt_time(l0_time):>8}")

        for i in range(max_lvls):
            if i < len(levels):
                lvl = levels[i]
                ef = lvl['expansion_factor']
                if ef == 0:
                    fac = '0x'
                elif ef < 0.01:
                    fac = f"{ef:.4f}x"
                else:
                    fac = f"{ef:.2f}x"
                proj_surv = _fmt_count(lvl.get('projected_survivors', 0))
            else:
                fac = '-'
                proj_surv = '-'
            row += f" | {fac:>8} | {proj_surv:>10}"

        proven_at = r.get('proven_at')
        if proven_at:
            status = proven_at
        else:
            if levels:
                ps = levels[-1].get('projected_survivors', 0)
                status = f'{_fmt_count(ps)} left'
            else:
                status = '-'
        projected = r.get('projected_total')
        row += f" | {status:>10} | {_fmt_time(projected):>12}"
        print(row)

    if failed:
        print(f"\nFAILED CONFIGS:")
        for r in failed:
            print(f"  n_half={r['n_half']}, m={r['m']}: {r['error']}")


def save_results(results, output_dir):
    """Write JSON + flattened CSV output files."""
    ts = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj == float('inf'):
            return "inf"
        return obj

    def convert_dict(d):
        return {k: convert(v) for k, v in d.items()}

    def convert_result(r):
        out = {}
        for k, v in r.items():
            if k == 'levels' and isinstance(v, list):
                out[k] = [convert_dict(lvl) for lvl in v]
            elif k == 'survivor_configs':
                continue
            else:
                out[k] = convert(v)
        return out

    # JSON (with nested levels)
    json_path = os.path.join(output_dir, f'benchmark_sweep_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump([convert_result(r) for r in results],
                  f, indent=2, default=convert)
    print(f"\nJSON saved to {json_path}")

    # CSV (flattened: L1_*, L2_*, L3_* columns)
    max_lvls = max((len(r.get('levels', [])) for r in results), default=0)

    base_fields = [
        'n_half', 'm', 'd0', 'correction', 'l0_compositions',
        'l0_survivors', 'l0_time', 'l0_pruned_asym', 'l0_pruned_test',
        'sample_size',
    ]
    level_suffixes = [
        'parents_processed', 'parents_available', 'projected_parents',
        'survivors_out', 'projected_survivors', 'expansion_factor',
        'elapsed', 'projected_time', 'total_children', 'throughput',
    ]
    level_fields = []
    for i in range(1, max_lvls + 1):
        for suf in level_suffixes:
            level_fields.append(f'L{i}_{suf}')

    tail_fields = ['proven_at', 'projected_total', 'error']
    fieldnames = base_fields + level_fields + tail_fields

    csv_path = os.path.join(output_dir, f'benchmark_sweep_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {k: convert(r.get(k, '')) for k in base_fields + tail_fields}
            for i, lvl in enumerate(r.get('levels', []), 1):
                for suf in level_suffixes:
                    row[f'L{i}_{suf}'] = convert(lvl.get(suf, ''))
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(configs, c_target):
    """Print theoretical quantities and dimension schedules (no computation)."""
    print(f"\nDRY RUN: c_target={c_target}")
    print(f"{'='*120}")

    header = (f"{'#':>3} | {'n_half':>6} | {'m':>5} | {'d0':>3} | "
              f"{'corr':>8} | {'eff_thresh':>10} | {'L0 comps':>12} | "
              f"{'Dim Schedule':>50}")
    print(header)
    print('-' * 120)

    for i, cfg in enumerate(configs):
        n_half = cfg['n_half']
        m = cfg['m']
        d0 = cfg['d']
        corr_flat = cfg.get('correction_flat', correction(m))
        corr_scaled = cfg.get('correction_scaled', correction(m, n_half))
        eff_thresh = c_target + corr_flat

        # Build dimension schedule with x_cap info
        dims = []
        d_parent = d0
        n_half_p = n_half
        while True:
            d_child = 2 * d_parent
            n_half_c = 2 * n_half_p
            x_cap = compute_x_cap(m, d_child, c_target, n_half_c)
            dims.append(f"d{d_child}(xcap={x_cap})")
            if d_child >= 128:  # Stop at reasonable depth
                break
            d_parent = d_child
            n_half_p = n_half_c
        schedule = (f"d{d0} -> " + " -> ".join(dims)
                    if dims else f"d{d0} (no refinement)")

        row = (f"{i+1:>3} | {n_half:>6} | {m:>5} | {d0:>3} | "
               f"{corr_flat:>8.4f} | {eff_thresh:>10.6f} | "
               f"{cfg['n_total']:>12,} | {schedule}")
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Empirical cascade benchmark for optimal (m, n_half)')
    parser.add_argument('--m_values', type=str,
                        default='10,15,20,25,30,40,50',
                        help='Comma-separated m values (default: 10,15,20,25,30,40,50)')
    parser.add_argument('--n_half_values', type=str, default='2,3',
                        help='Comma-separated n_half values (default: 2,3)')
    parser.add_argument('--c_target', type=float, default=1.40,
                        help='Target lower bound (default: 1.40)')
    parser.add_argument('--max_parents', type=int, default=10_000,
                        help='Max parents per level (default: 10000)')
    parser.add_argument('--max_levels', type=int, default=6,
                        help='Max refinement levels after L0 (default: 6)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--time_budget', type=int, default=300,
                        help='Per-level time budget in seconds (0=no limit, default: 300)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print theoretical quantities only (no computation)')
    parser.add_argument('--force', action='store_true',
                        help='Include vacuous configs (skip vacuity check)')
    return parser.parse_args()


def main():
    global MAX_PARENTS_PER_LEVEL, LEVEL_TIME_BUDGET
    args = parse_args()
    MAX_PARENTS_PER_LEVEL = args.max_parents
    LEVEL_TIME_BUDGET = args.time_budget

    m_values = [int(x) for x in args.m_values.split(',')]
    n_half_values = [int(x) for x in args.n_half_values.split(',')]

    configs = build_sweep_configs(m_values, n_half_values, args.c_target,
                                  force=args.force)

    print(f"Benchmark sweep: {len(configs)} configs", flush=True)
    print(f"  m values: {m_values}", flush=True)
    print(f"  n_half values: {n_half_values}", flush=True)
    print(f"  c_target: {args.c_target}", flush=True)
    print(f"  max parents/level: {MAX_PARENTS_PER_LEVEL:,}", flush=True)
    print(f"  time budget/level: {LEVEL_TIME_BUDGET}s", flush=True)
    print(f"  max levels: {args.max_levels}", flush=True)

    if args.dry_run:
        dry_run(configs, args.c_target)
        return

    results = run_sweep(configs, args.c_target,
                        args.max_levels, args.output_dir, args.resume)

    format_summary_table(results)
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
