"""Generate publication-quality comparison figures.

Reads baseline/comparison_results.json (produced by run_comparison.py)
and generates research-standard figures including Octave/MATLAB baseline data.

Estimated time: ~5-15 seconds for all 6 figures in PDF+PNG.

Figures:
  1. Per-parent throughput: Octave [CS14] vs Python Naive vs Optimized (grouped bars)
  2. Speedup scaling with dimension (line + markers)
  3. Per-parent latency distribution at L4 (box plots)
  4. Optimization contribution breakdown (horizontal stacked bar)
  5. End-to-end projected time comparison (log-scale bars)
  6. Combined 2x2 summary panel

Usage:
    python -m baseline.plot_comparison                  # all figures (~10s)
    python -m baseline.plot_comparison --fig 1 3 5      # selected (~5s)
    python -m baseline.plot_comparison --format png      # PNG only (~5s)
"""
import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'comparison_results.json')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
def setup_style():
    """Configure matplotlib for publication-quality output."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif',
                        'Times New Roman'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'mathtext.fontset': 'cm',
        'text.usetex': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


# Colorblind-friendly palette
COLOR_OCTAVE = '#B2182B'      # Red — MATLAB/Octave baseline
COLOR_NAIVE = '#BBBBBB'       # Gray — Python naive
COLOR_OPTIMIZED = '#2166AC'   # Blue — Python optimized
COLOR_MATLAB_FAITHFUL = '#F4A582'  # Light red — MATLAB-faithful Python
COLOR_ACCENT = '#4DAF4A'      # Green — accents
COLOR_DARK = '#333333'


def load_results():
    """Load comparison results JSON."""
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found. Run baseline.run_comparison first.")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save_fig(fig, name, formats):
    """Save figure in requested formats."""
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f'{name}.{fmt}')
        fig.savefig(path)
        print(f"  Saved {path}")


def _has_octave(levels):
    """Check if any level has Octave data."""
    return any('octave' in lev for lev in levels)


# =====================================================================
# Figure 1: Throughput comparison (Hero figure)
# =====================================================================

def fig1_throughput(data, formats):
    """Grouped bar chart: Octave vs Naive vs Optimized throughput at each level."""
    import matplotlib.pyplot as plt

    levels = data['levels']
    has_octave = _has_octave(levels)
    has_mf = any('matlab_faithful' in lev for lev in levels)
    n = len(levels)
    names = [lev['name'] for lev in levels]
    d_labels = [f"$d={lev['d_parent']}\\to{lev['d_child']}$" for lev in levels]

    naive_rates = [lev['naive']['parents_per_sec'] for lev in levels]
    opt_rates = [lev['optimized']['parents_per_sec'] for lev in levels]
    octave_rates = [lev.get('octave', {}).get('parents_per_sec', 0) for lev in levels]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(n)

    if has_octave:
        n_bars = 3
        w = 0.25
        offsets = [-w, 0, w]
        bar_data = [
            (octave_rates, 'MATLAB/Octave [CS14]', COLOR_OCTAVE),
            (naive_rates, 'Python naive', COLOR_NAIVE),
            (opt_rates, 'Python optimized', COLOR_OPTIMIZED),
        ]
    else:
        n_bars = 2
        w = 0.35
        offsets = [-w/2, w/2]
        bar_data = [
            (naive_rates, 'Naive (materialize + batch prune)', COLOR_NAIVE),
            (opt_rates, 'Optimized (fused kernel)', COLOR_OPTIMIZED),
        ]

    for idx, (rates, label, color) in enumerate(bar_data):
        bars = ax.bar(x + offsets[idx], rates, w, label=label,
                      color=color, edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_yscale('log')
    ax.set_ylabel('Throughput (parents / sec)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nm}\n{dl}' for nm, dl in zip(names, d_labels)])
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7)

    # Annotate speedup (optimized vs slowest baseline)
    for i in range(n):
        if has_octave and octave_rates[i] > 0:
            sp = opt_rates[i] / octave_rates[i]
            label_str = f'{sp:.0f}x vs Octave'
        else:
            sp = opt_rates[i] / naive_rates[i] if naive_rates[i] > 0 else 1
            label_str = f'{sp:.0f}x' if sp >= 2 else f'{sp:.1f}x'
        y_top = max(naive_rates[i], opt_rates[i],
                    octave_rates[i] if has_octave else 0)
        ax.annotate(label_str,
                    xy=(x[i], y_top), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=COLOR_OPTIMIZED)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    save_fig(fig, 'fig1_throughput', formats)
    plt.close(fig)


# =====================================================================
# Figure 2: Speedup scaling with dimension
# =====================================================================

def fig2_speedup_scaling(data, formats):
    """Line plot: speedup factor vs child dimension."""
    import matplotlib.pyplot as plt

    levels = data['levels']
    has_octave = _has_octave(levels)
    d_child = [lev['d_child'] for lev in levels]
    speedups_naive = [lev['speedup']['wall_time'] for lev in levels]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))

    ax.plot(d_child, speedups_naive, 's--', color=COLOR_NAIVE,
            label='vs. Python naive', markersize=6, zorder=4, alpha=0.8)

    if has_octave:
        speedups_octave = []
        d_octave = []
        for lev in levels:
            if 'octave' in lev and lev['speedup'].get('octave_vs_optimized', 0) > 0:
                speedups_octave.append(lev['speedup']['octave_vs_optimized'])
                d_octave.append(lev['d_child'])
        if speedups_octave:
            ax.plot(d_octave, speedups_octave, 'o-', color=COLOR_OPTIMIZED,
                    label='vs. MATLAB/Octave [CS14]', markersize=7, zorder=5)
            for dc, sp in zip(d_octave, speedups_octave):
                ax.annotate(f'{sp:.0f}x',
                            xy=(dc, sp), xytext=(6, 4),
                            textcoords='offset points', fontsize=7,
                            color=COLOR_OPTIMIZED)

    # Annotate naive speedups
    for dc, sp in zip(d_child, speedups_naive):
        ax.annotate(f'{sp:.0f}x' if sp >= 2 else f'{sp:.1f}x',
                    xy=(dc, sp), xytext=(-6, -12),
                    textcoords='offset points', fontsize=7, color=COLOR_NAIVE)

    ax.set_xlabel('Child dimension $d_{\\mathrm{child}}$')
    ax.set_ylabel('Speedup factor')
    ax.set_xscale('log', base=2)
    ax.set_xticks(d_child)
    ax.set_xticklabels([str(d) for d in d_child])
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    fig.tight_layout()
    save_fig(fig, 'fig2_speedup_scaling', formats)
    plt.close(fig)


# =====================================================================
# Figure 3: Per-parent latency distribution (L4)
# =====================================================================

def fig3_latency_distribution(data, formats):
    """Box plots: Octave vs Naive vs Optimized per-parent latency at L4."""
    import matplotlib.pyplot as plt

    l4 = next((lev for lev in data['levels'] if lev['name'] == 'L4'), None)
    if l4 is None:
        print("  Skipping fig3: no L4 data")
        return

    has_octave = 'octave' in l4

    naive_ms = np.array(l4['naive']['per_parent_times_ms'])
    opt_ms = np.array(l4['optimized']['per_parent_times_ms'])

    # Filter near-zero (asymmetry-skipped)
    mask = naive_ms > 0.1
    naive_ms = naive_ms[mask]
    opt_ms = opt_ms[mask]

    box_data = []
    box_colors = []
    box_labels = []

    if has_octave:
        oct_ms = np.array(l4['octave']['per_parent_times_ms'])
        # Octave may have different n_parents; use all available
        oct_mask = oct_ms > 0.1
        oct_ms = oct_ms[oct_mask]
        box_data.append(oct_ms)
        box_colors.append(COLOR_OCTAVE)
        box_labels.append('MATLAB/Octave\n[CS14]')

    box_data.append(naive_ms)
    box_colors.append(COLOR_NAIVE)
    box_labels.append('Python\nnaive')

    box_data.append(opt_ms)
    box_colors.append(COLOR_OPTIMIZED)
    box_labels.append('Python\noptimized')

    n_boxes = len(box_data)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    positions = list(range(1, n_boxes + 1))
    bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=True, zorder=3,
                    flierprops=dict(marker='.', markersize=3, alpha=0.4),
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    for i, color in enumerate(box_colors):
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.8)

    ax.set_yscale('log')
    ax.set_ylabel('Per-parent latency (ms)')
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, fontsize=7)

    # Annotate medians
    for i, (pos, vals) in enumerate(zip(positions, box_data)):
        med = np.median(vals)
        unit = 'ms' if med < 1000 else 's'
        val_str = f'{med:.1f} ms' if med < 1000 else f'{med/1000:.1f} s'
        ax.annotate(val_str, xy=(pos, med), xytext=(28, 0),
                    textcoords='offset points', fontsize=7, va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Speedup annotation
    if has_octave and len(oct_ms) > 0 and len(opt_ms) > 0:
        speedup = np.median(oct_ms) / np.median(opt_ms)
        geo_mean = np.sqrt(np.median(oct_ms) * np.median(opt_ms))
        ax.annotate(f'{speedup:.0f}x',
                    xy=(1.5, geo_mean), ha='center',
                    fontsize=10, fontweight='bold', color=COLOR_OPTIMIZED)
    elif len(naive_ms) > 0 and len(opt_ms) > 0:
        speedup = np.median(naive_ms) / np.median(opt_ms)
        geo_mean = np.sqrt(np.median(naive_ms) * np.median(opt_ms))
        mid_pos = (positions[0] + positions[-1]) / 2
        ax.annotate(f'{speedup:.0f}x speedup',
                    xy=(mid_pos, geo_mean), ha='center',
                    fontsize=9, fontweight='bold', color=COLOR_OPTIMIZED)

    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, 'fig3_latency_distribution', formats)
    plt.close(fig)


# =====================================================================
# Figure 4: Optimization breakdown (instrumentation)
# =====================================================================

def fig4_optimization_breakdown(data, formats):
    """Horizontal stacked bar showing where work is saved."""
    import matplotlib.pyplot as plt

    inst = data.get('instrumentation')
    if inst is None:
        print("  Skipping fig4: no instrumentation data")
        return

    cart = inst['total_cartesian']
    visited = inst['total_visited']
    skipped_subtree = inst['subtree_pruning']['children_skipped']
    total_advances = inst['total_advances']

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 3.5),
                              gridspec_kw={'height_ratios': [1, 1.2]})

    # --- Panel A: Work reduction ---
    ax = axes[0]
    frac_subtree = skipped_subtree / max(1, cart) * 100
    frac_visited = visited / max(1, cart) * 100
    ax.barh(['Cartesian\nproduct'], [frac_visited], height=0.5,
            color=COLOR_OPTIMIZED, label=f'Visited ({frac_visited:.1f}%)', zorder=3)
    ax.barh(['Cartesian\nproduct'], [frac_subtree], left=[frac_visited], height=0.5,
            color=COLOR_ACCENT, label=f'Subtree-pruned ({frac_subtree:.1f}%)', zorder=3)
    remaining = 100 - frac_visited - frac_subtree
    if remaining > 0.5:
        ax.barh(['Cartesian\nproduct'], [remaining],
                left=[frac_visited + frac_subtree], height=0.5,
                color=COLOR_NAIVE, label=f'Asymmetry-skipped ({remaining:.1f}%)', zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Fraction of Cartesian product (%)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title('(a) Work reduction', fontsize=9, loc='left', fontweight='bold')

    # --- Panel B: Per-child optimization paths ---
    ax = axes[1]
    if total_advances > 0:
        categories = ['Autoconv\nupdate cost', 'Window\nscan cost']
        fast_pct = inst['carry_paths']['fast_n1_pct']
        short_pct = inst['carry_paths']['short_carry_pct']
        deep_pct = inst['carry_paths']['deep_carry_pct']
        qc_pct = inst['quick_check']['hit_pct_of_visited']
        full_pct = inst['quick_check']['full_scan_pct']

        ax.barh([0], [fast_pct], height=0.4, color='#2166AC',
                label=f'Fast path $O(d)$ ({fast_pct:.0f}%)', zorder=3)
        ax.barh([0], [short_pct], left=[fast_pct], height=0.4, color='#67A9CF',
                label=f'Short carry ({short_pct:.0f}%)', zorder=3)
        ax.barh([0], [deep_pct], left=[fast_pct + short_pct], height=0.4,
                color='#D1E5F0', label=f'Deep carry $O(d^2)$ ({deep_pct:.0f}%)', zorder=3)

        surv_pct = 100 - qc_pct - full_pct
        ax.barh([1], [qc_pct], height=0.4, color='#4DAF4A',
                label=f'Quick-check hit ({qc_pct:.0f}%)', zorder=3)
        ax.barh([1], [full_pct], left=[qc_pct], height=0.4, color='#FDB863',
                label=f'Full scan ({full_pct:.0f}%)', zorder=3)
        if surv_pct > 0.5:
            ax.barh([1], [surv_pct], left=[qc_pct + full_pct], height=0.4,
                    color='#E0E0E0', label=f'Survived ({surv_pct:.0f}%)', zorder=3)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Fraction of visited children (%)')
    ax.legend(loc='upper right', fontsize=6.5, framealpha=0.9, ncol=2)
    ax.set_title('(b) Per-child optimization paths', fontsize=9, loc='left',
                 fontweight='bold')

    fig.tight_layout(h_pad=1.5)
    save_fig(fig, 'fig4_optimization_breakdown', formats)
    plt.close(fig)


# =====================================================================
# Figure 5: End-to-end projected time comparison
# =====================================================================

def fig5_end_to_end(data, formats):
    """Log-scale bar chart: MATLAB published vs Octave measured vs Ours."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ee = data.get('end_to_end')
    if ee is None:
        print("  Skipping fig5: no end-to-end data")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    matlab_hours = ee['matlab']['cpu_hours']
    naive_sc = ee['ours']['l4_naive_single_core_hours'] + ee['ours']['l0_l3_hours']
    opt_sc = ee['ours']['l4_optimized_single_core_hours'] + ee['ours']['l0_l3_hours']

    labels = []
    values = []
    colors = []

    # Always include MATLAB published
    labels.append(f'MATLAB [CS14]\n$c \\geq {ee["matlab"]["c_target"]}$')
    values.append(matlab_hours)
    colors.append(COLOR_OCTAVE)

    # Include Octave measured if available
    has_octave = 'l4_octave_single_core_hours' in ee.get('ours', {})
    if has_octave:
        oct_sc = ee['ours']['l4_octave_single_core_hours'] + ee['ours']['l0_l3_hours']
        labels.append(f'Octave (measured)\n$c \\geq {ee["ours"]["c_target"]}$')
        values.append(oct_sc)
        colors.append(COLOR_MATLAB_FAITHFUL)

    # Naive baseline
    labels.append(f'Python naive\n$c \\geq {ee["ours"]["c_target"]}$')
    values.append(naive_sc)
    colors.append(COLOR_NAIVE)

    # Optimized
    labels.append(f'This work\n$c \\geq {ee["ours"]["c_target"]}$')
    values.append(opt_sc)
    colors.append(COLOR_OPTIMIZED)

    n_bars = len(values)
    bars = ax.bar(range(n_bars), values, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.65, zorder=3)

    ax.set_yscale('log')
    ax.set_ylabel('Single-core CPU-hours')
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=7)

    for bar, val in zip(bars, values):
        if val >= 1000:
            label = f'{val/1000:.1f}K'
        else:
            label = f'{val:.0f}'
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Speedup annotation: optimized vs MATLAB published
    if opt_sc > 0:
        speedup_vs_matlab = matlab_hours / opt_sc
        ax.annotate(f'{speedup_vs_matlab:.0f}x',
                    xy=(n_bars - 1, opt_sc), xytext=(0.3, matlab_hours * 0.5),
                    fontsize=10, fontweight='bold', color=COLOR_OPTIMIZED,
                    arrowprops=dict(arrowstyle='->', color=COLOR_OPTIMIZED, lw=1.2),
                    ha='center')

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x:,.0f}' if x >= 1 else f'{x:.1f}'))

    fig.tight_layout()
    save_fig(fig, 'fig5_end_to_end', formats)
    plt.close(fig)


# =====================================================================
# Figure 6: Combined 2x2 summary panel
# =====================================================================

def fig6_summary_panel(data, formats):
    """2x2 summary panel combining key results."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    levels = data['levels']
    if len(levels) < 2:
        print("  Skipping fig6: need at least 2 levels")
        return

    has_octave = _has_octave(levels)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    # --- Panel (a): Throughput bars ---
    ax = axes[0, 0]
    n = len(levels)
    names = [lev['name'] for lev in levels]
    naive_rates = [lev['naive']['parents_per_sec'] for lev in levels]
    opt_rates = [lev['optimized']['parents_per_sec'] for lev in levels]
    x = np.arange(n)

    if has_octave:
        octave_rates = [lev.get('octave', {}).get('parents_per_sec', 0)
                        for lev in levels]
        w = 0.25
        ax.bar(x - w, octave_rates, w, color=COLOR_OCTAVE,
               label='Octave', zorder=3)
        ax.bar(x, naive_rates, w, color=COLOR_NAIVE,
               label='Naive', zorder=3)
        ax.bar(x + w, opt_rates, w, color=COLOR_OPTIMIZED,
               label='Optimized', zorder=3)
    else:
        w = 0.35
        ax.bar(x - w/2, naive_rates, w, color=COLOR_NAIVE,
               label='Naive', zorder=3)
        ax.bar(x + w/2, opt_rates, w, color=COLOR_OPTIMIZED,
               label='Optimized', zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel('Parents / sec')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('(a) Throughput comparison', fontsize=9, loc='left',
                 fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # --- Panel (b): Speedup scaling ---
    ax = axes[0, 1]
    d_child = [lev['d_child'] for lev in levels]
    speedups = [lev['speedup']['wall_time'] for lev in levels]
    ax.plot(d_child, speedups, 's--', color=COLOR_NAIVE,
            label='vs. naive', markersize=5, zorder=4, alpha=0.8)

    if has_octave:
        sp_oct = []
        d_oct = []
        for lev in levels:
            if 'octave' in lev:
                sp_oct.append(lev['speedup'].get('octave_vs_optimized', 0))
                d_oct.append(lev['d_child'])
        if sp_oct:
            ax.plot(d_oct, sp_oct, 'o-', color=COLOR_OPTIMIZED,
                    label='vs. Octave', markersize=7, zorder=5)
            for dc, sp in zip(d_oct, sp_oct):
                ax.annotate(f'{sp:.0f}x', xy=(dc, sp), xytext=(5, 5),
                            textcoords='offset points', fontsize=7,
                            color=COLOR_OPTIMIZED)

    ax.set_xlabel('$d_{\\mathrm{child}}$')
    ax.set_ylabel('Speedup')
    ax.set_xscale('log', base=2)
    ax.set_xticks(d_child)
    ax.set_xticklabels([str(d) for d in d_child])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('(b) Speedup vs. dimension', fontsize=9, loc='left',
                 fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # --- Panel (c): Latency distribution (last level) ---
    ax = axes[1, 0]
    last = levels[-1]
    naive_ms = np.array(last['naive']['per_parent_times_ms'])
    opt_ms = np.array(last['optimized']['per_parent_times_ms'])
    mask = naive_ms > 0.1

    box_data = []
    box_colors = []
    box_labels_short = []

    if 'octave' in last:
        oct_ms = np.array(last['octave']['per_parent_times_ms'])
        oct_mask = oct_ms > 0.1
        box_data.append(oct_ms[oct_mask])
        box_colors.append(COLOR_OCTAVE)
        box_labels_short.append('Octave')

    if np.any(mask):
        box_data.append(naive_ms[mask])
        box_colors.append(COLOR_NAIVE)
        box_labels_short.append('Naive')
        box_data.append(opt_ms[mask])
        box_colors.append(COLOR_OPTIMIZED)
        box_labels_short.append('Optimized')

    if box_data:
        positions = list(range(1, len(box_data) + 1))
        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=True, zorder=3,
                        flierprops=dict(marker='.', markersize=2, alpha=0.4),
                        medianprops=dict(color='black', linewidth=1))
        for i, c in enumerate(box_colors):
            bp['boxes'][i].set_facecolor(c)
            bp['boxes'][i].set_alpha(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels_short, fontsize=7)

    ax.set_yscale('log')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'(c) Per-parent latency ({last["name"]})', fontsize=9,
                 loc='left', fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # --- Panel (d): End-to-end ---
    ax = axes[1, 1]
    ee = data.get('end_to_end')
    if ee:
        matlab_h = ee['matlab']['cpu_hours']
        naive_h = ee['ours']['l4_naive_single_core_hours'] + ee['ours']['l0_l3_hours']
        opt_h = ee['ours']['l4_optimized_single_core_hours'] + ee['ours']['l0_l3_hours']

        bar_labels = []
        bar_values = []
        bar_colors = []

        bar_labels.append(f'MATLAB\n$c\\geq{ee["matlab"]["c_target"]}$')
        bar_values.append(matlab_h)
        bar_colors.append(COLOR_OCTAVE)

        if 'l4_octave_single_core_hours' in ee.get('ours', {}):
            oct_h = ee['ours']['l4_octave_single_core_hours'] + ee['ours']['l0_l3_hours']
            bar_labels.append(f'Octave\n$c\\geq{ee["ours"]["c_target"]}$')
            bar_values.append(oct_h)
            bar_colors.append(COLOR_MATLAB_FAITHFUL)

        bar_labels.append(f'Naive\n$c\\geq{ee["ours"]["c_target"]}$')
        bar_values.append(naive_h)
        bar_colors.append(COLOR_NAIVE)

        bar_labels.append(f'Ours\n$c\\geq{ee["ours"]["c_target"]}$')
        bar_values.append(opt_h)
        bar_colors.append(COLOR_OPTIMIZED)

        bars = ax.bar(range(len(bar_values)), bar_values,
                      color=bar_colors, edgecolor='white',
                      linewidth=0.5, width=0.6, zorder=3)
        ax.set_yscale('log')
        ax.set_ylabel('CPU-hours (1 core)')
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=6.5)
        for bar, val in zip(bars, bar_values):
            lbl = f'{val/1000:.1f}K' if val >= 1000 else f'{val:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.2, lbl,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title('(d) End-to-end projected time', fontsize=9, loc='left',
                 fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_fig(fig, 'fig6_summary_panel', formats)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

FIGURE_FUNCS = {
    1: ('Throughput comparison', fig1_throughput),
    2: ('Speedup scaling', fig2_speedup_scaling),
    3: ('Latency distribution', fig3_latency_distribution),
    4: ('Optimization breakdown', fig4_optimization_breakdown),
    5: ('End-to-end comparison', fig5_end_to_end),
    6: ('Summary panel', fig6_summary_panel),
}


def main():
    parser = argparse.ArgumentParser(description='Generate comparison figures')
    parser.add_argument('--fig', type=int, nargs='*', default=None,
                        help='Figure numbers to generate (default: all)')
    parser.add_argument('--format', type=str, default='pdf,png',
                        help='Output formats, comma-separated (default: pdf,png)')
    args = parser.parse_args()

    formats = [f.strip() for f in args.format.split(',')]
    figs_to_make = args.fig if args.fig else list(FIGURE_FUNCS.keys())

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)

    setup_style()
    data = load_results()

    print(f"Generating figures from {RESULTS_PATH}")
    print(f"Output directory: {FIGURES_DIR}")
    print(f"Formats: {formats}\n")

    for fig_num in figs_to_make:
        if fig_num not in FIGURE_FUNCS:
            print(f"  Unknown figure {fig_num}, skipping")
            continue
        desc, func = FIGURE_FUNCS[fig_num]
        print(f"  Figure {fig_num}: {desc}")
        try:
            func(data, formats)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
