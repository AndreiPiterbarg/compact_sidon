"""Generate figures for the STAT 4830 presentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#0b3d91"
TEAL = "#138086"
GOLD = "#d4a017"
CORAL = "#c44536"
GREY = "#6b7280"
LIGHT = "#e5e7eb"


def autoconv_intuition():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.linspace(-0.25, 0.25, 800)
    f = np.where(np.abs(x) < 0.25, np.exp(-18 * x**2), 0.0)
    f = f / np.trapezoid(f, x)
    t = np.linspace(-0.5, 0.5, 1600)
    dx = x[1] - x[0]
    fc = np.convolve(f, f) * dx
    tc = np.linspace(2 * x[0], 2 * x[-1], len(fc))

    axes[0].fill_between(x, f, color=NAVY, alpha=0.25)
    axes[0].plot(x, f, color=NAVY, lw=2.2)
    axes[0].set_xlim(-0.35, 0.35)
    axes[0].set_title(r"$f$ on $[-1/4,\,1/4]$,  $\int f = 1$")
    axes[0].set_xlabel("x")
    axes[0].axvline(-0.25, color=GREY, lw=0.7, ls=":")
    axes[0].axvline(0.25, color=GREY, lw=0.7, ls=":")

    peak = fc.max()
    axes[1].fill_between(tc, fc, color=CORAL, alpha=0.25)
    axes[1].plot(tc, fc, color=CORAL, lw=2.2)
    axes[1].axhline(peak, color=CORAL, lw=1.2, ls="--", alpha=0.7)
    axes[1].annotate(r"$\|f*f\|_\infty$", xy=(0.0, peak), xytext=(0.25, peak * 0.95),
                     fontsize=14, color=CORAL, ha="center")
    axes[1].set_xlim(-0.6, 0.6)
    axes[1].set_title(r"$f * f$ on $[-1/2,\,1/2]$")
    axes[1].set_xlabel("t")
    axes[1].axvline(-0.5, color=GREY, lw=0.7, ls=":")
    axes[1].axvline(0.5, color=GREY, lw=0.7, ls=":")

    for ax in axes:
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "fig_autoconv_intuition.png")
    plt.close(fig)


def bounds_number_line():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(1.23, 1.55)
    ax.set_ylim(-1.3, 1.5)

    ax.hlines(0, 1.24, 1.54, color="black", lw=1.2, zorder=1)
    for v in np.arange(1.25, 1.56, 0.05):
        ax.plot([v, v], [-0.04, 0.04], color="black", lw=0.9)
        ax.text(v, -0.15, f"{v:.2f}", ha="center", va="top", fontsize=11, color=GREY)

    feasible_l, feasible_r = 1.4, 1.5029
    ax.fill_betweenx([-0.04, 0.04], feasible_l, feasible_r, color=LIGHT, zorder=0)
    ax.text((feasible_l + feasible_r) / 2, -0.55, "remaining uncertainty",
            ha="center", fontsize=10, color=GREY, style="italic")

    # (value, title, subtitle, color, y_top_of_label, ha, x_offset)
    markers = [
        (1.2802, "C&S 2017",            "lower bound",          NAVY,  0.70, "right",  -0.002),
        (1.3,    "SDP (ours)",          "Lasserre, certified",  TEAL,  1.30, "left",    0.002),
        (1.4,    "Cascade (ours)",      "branch-and-prune GPU", GOLD,  0.70, "center",  0.0),
        (1.5029, "Matolcsi–Vinuesa",    "upper bound",          CORAL, 1.30, "center",  0.0),
    ]
    for v, title, sub, c, y, ha, dx in markers:
        ax.plot([v, v], [0.05, y - 0.18], color=c, lw=1.4)
        ax.scatter([v], [0], color=c, s=70, zorder=3)
        ax.text(v + dx, y, f"{title}\n{v}", ha=ha, va="bottom", fontsize=11,
                color=c, fontweight="bold")
        ax.text(v + dx, y - 0.17, sub, ha=ha, va="top", fontsize=9, color=GREY)

    ax.set_xlabel(r"$C_{1a}$", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bounds_number_line.png")
    plt.close(fig)


def correction_gap():
    """Show how the MATLAB code's Lemma-3 correction stays flat at ~0.04
    while the true correction grows with the cascade depth."""
    ds = np.array([3, 6, 12, 24, 48, 96])
    eps_mass = 0.02
    # Code: 2·eps + eps² (with W ≈ 1 window mass)
    code = 2 * eps_mass + eps_mass ** 2 * np.ones_like(ds, dtype=float)
    # Correct: 2·(2d·eps) + (2d·eps)² using height step = 2d·eps
    true_lin = 2 * (2 * ds * eps_mass)
    true_quad = (2 * ds * eps_mass) ** 2
    true = true_lin + true_quad

    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.38
    idx = np.arange(len(ds))
    ax.bar(idx - width / 2, code, width, color=CORAL,
           label="Code budget  (ε = mass step = 0.02)")
    ax.bar(idx + width / 2, true, width, color=NAVY,
           label="Required   (ε = height step = 2d × 0.02)")

    for i, (c, t) in enumerate(zip(code, true)):
        ratio = t / c
        ax.text(i + width / 2, t * 1.12, f"×{ratio:.0f}", ha="center",
                fontsize=11, color=NAVY, fontweight="bold")
        ax.text(i - width / 2, c * 1.6, f"{c:.3f}", ha="center",
                fontsize=8.5, color=CORAL)

    ax.set_xticks(idx)
    ax.set_xticklabels([f"d={d}" for d in ds])
    ax.set_ylabel("Lemma-3 correction  (log scale, W ≈ 1)")
    ax.set_title("MATLAB treats the mass step as if it were the height step —\n"
                 "the true correction grows as 2d·ε, so it is too small by ≈ 2d")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_correction_gap.png")
    plt.close(fig)


def lasserre_ladder():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [
        (r"$\mathrm{val}^{(k,b)}(d)$", "sparse Lasserre SDP", 0, TEAL),
        (r"$\mathrm{val}^{(k)}(d)$",   "dense Lasserre SDP",  1, NAVY),
        (r"$\mathrm{val}(d)$",          "simplex polynomial",  2, GOLD),
        (r"$C_{1a}$",                  "autoconvolution constant", 3, CORAL),
    ]
    y_base = 0.3
    for text, sub, i, c in labels:
        x = 0.5 + i * 2.3
        ax.add_patch(plt.Rectangle((x - 0.9, y_base), 1.8, 0.9,
                                   color=c, alpha=0.18, zorder=1))
        ax.text(x, y_base + 0.6, text, ha="center", va="center",
                fontsize=17, color=c, fontweight="bold", zorder=2)
        ax.text(x, y_base + 0.22, sub, ha="center", va="center",
                fontsize=10, color=GREY, zorder=2)
        if i < 3:
            ax.annotate("", xy=(x + 1.0, y_base + 0.6),
                        xytext=(x + 0.9, y_base + 0.6),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color=GREY))
            ax.text(x + 0.95, y_base + 0.95, r"$\leq$", ha="center",
                    fontsize=18, color=GREY)

    ax.text(4.0, 0.08,
            "solve at $k=3,\\ d=16$",
            ha="center", fontsize=11, color=NAVY, style="italic")
    ax.set_xlim(-0.5, 8.4)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lasserre_ladder.png")
    plt.close(fig)


def effective_m_plot():
    """Show that the effective fine-grid m in the MATLAB code drops below 1
    as d grows — i.e., Lemma 3 is vacuous at higher cascade levels."""
    ds = np.arange(2, 100)
    eps_mass = 0.02
    m_eff = 1.0 / (2 * ds * eps_mass)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(ds, m_eff, color=NAVY, lw=2.2, label=r"effective $m$ in C&S grid")
    ax.axhline(1, color=CORAL, lw=1.4, ls="--", label="Lemma 3 vacuous below here")
    ax.axhline(50, color=GREY, lw=1.0, ls=":", label="C&S paper's $m=50$")

    ax.fill_between(ds, 0, 1, where=(m_eff < 1), color=CORAL, alpha=0.12)
    ax.set_xlabel(r"cascade resolution $d$")
    ax.set_ylabel(r"effective $m_{\mathrm{CS}} = 1/(2d\cdot\varepsilon_{\mathrm{mass}})$")
    ax.set_title("The deeper the cascade, the less the Lemma-3 correction says")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 80)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_effective_m.png")
    plt.close(fig)


def three_tracks():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    tracks = [
        ("Track 1", "Cascade",
         "branch-and-prune\non GPU",
         r"$C_{1a}\geq 1.4$", NAVY),
        ("Track 2", "Lasserre SDP",
         "continuous → polynomial\n→ semidefinite",
         r"$C_{1a}\geq 1.3$ certified", TEAL),
        ("Track 3", "MATLAB audit",
         "soundness check\nof the C&S artifact",
         "bug identified", CORAL),
    ]
    for i, (tag, head, body, result, c) in enumerate(tracks):
        x = 1.2 + i * 3.3
        ax.add_patch(plt.Rectangle((x - 1.3, 0.1), 2.6, 2.8,
                                   color=c, alpha=0.08, zorder=0, ec=c, lw=1.3))
        ax.text(x, 2.55, tag, ha="center", fontsize=11, color=c, fontweight="bold")
        ax.text(x, 2.15, head, ha="center", fontsize=17, fontweight="bold", color="black")
        ax.text(x, 1.45, body, ha="center", fontsize=12, color=GREY)
        ax.text(x, 0.55, result, ha="center", fontsize=13, color=c, fontweight="bold")
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_tracks.png")
    plt.close(fig)


def cascade_schematic():
    fig, ax = plt.subplots(figsize=(9, 4.0))
    # Multi-level tree: each level has exponentially more nodes, many pruned
    levels = 5
    ys = np.linspace(3.8, 0.4, levels)
    prev_positions = [(5.0, ys[0])]
    ax.scatter(*prev_positions[0], s=120, color=NAVY, zorder=3)
    ax.text(prev_positions[0][0], prev_positions[0][1] + 0.3, "root",
            ha="center", fontsize=10, color=NAVY)

    rng = np.random.default_rng(3)
    for lv in range(1, levels):
        new_positions = []
        for (px, py) in prev_positions:
            n_children = rng.integers(2, 4)
            xs = np.linspace(px - 1.0, px + 1.0, n_children) if n_children > 1 else [px]
            for cx in xs:
                cy = ys[lv]
                survive = rng.random() < (0.55 if lv < levels - 1 else 0.35)
                color = NAVY if survive else CORAL
                alpha = 0.95 if survive else 0.45
                ax.plot([px, cx], [py, cy], color=GREY, lw=0.6, alpha=0.5, zorder=1)
                ax.scatter([cx], [cy], s=55 if survive else 35,
                           color=color, alpha=alpha, zorder=3,
                           marker="o" if survive else "x")
                if survive:
                    new_positions.append((cx, cy))
        prev_positions = new_positions or prev_positions

    ax.text(0.6, 3.9, "refine each surviving\nconfiguration",
            fontsize=11, color=NAVY, ha="left", va="top")
    ax.text(0.6, 0.3, "prune when test value\nexceeds threshold",
            fontsize=11, color=CORAL, ha="left", va="top")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    ax.set_title("Branch-and-prune cascade: refine, enumerate, prune", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cascade_schematic.png")
    plt.close(fig)


if __name__ == "__main__":
    autoconv_intuition()
    bounds_number_line()
    correction_gap()
    lasserre_ladder()
    effective_m_plot()
    three_tracks()
    cascade_schematic()
    print("figures written to", OUT)
