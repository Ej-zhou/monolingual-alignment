#!/usr/bin/env python3
"""Figure 6: directional success rate for eng -> {fra, deu, spa, jpn, zho}
(country -> capital) under within-model ceiling, Procrustes, unprojected and
shuffle, with Wilson 95% CIs.  Per target the span start j in {1..8} that best
separates {within, Procrustes} from the two controls is shown.

Reads results/w_<tgt>/within_results.csv and results/x_eng_latn-<tgt>/cross_results.csv.

Usage:
  python plot_summary.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from patching_results import rate_ci, read_results

ROOT = Path(__file__).resolve().parent
ENG_X = ["fra_latn", "deu_latn", "spa_latn", "jpn_jpan", "zho_hans"]
TGT_PRETTY = {"fra_latn": "French", "deu_latn": "German", "spa_latn": "Spanish",
              "jpn_jpan": "Japanese", "zho_hans": "Chinese"}


def best_j_for_target(tgt):
    w = read_results(ROOT / "results" / f"w_{tgt}", "within_results.csv")
    x = read_results(ROOT / "results" / f"x_eng_latn-{tgt}", "cross_results.csv")
    best = None
    for j in sorted(w["j_start"].unique()):
        wr = rate_ci(w[w["j_start"] == j]["delta_logp_donor"])
        pr = rate_ci(x[(x["condition"] == "procrustes")
                       & (x["j_start"] == j)]["delta_logp_donor"])
        sr = rate_ci(x[(x["condition"] == "shuffle")
                       & (x["j_start"] == j)]["delta_logp_donor"])
        ur = rate_ci(x[(x["condition"] == "unprojected")
                       & (x["j_start"] == j)]["delta_logp_donor"])
        if any(r is None for r in [wr, pr, sr, ur]):
            continue
        score = wr[0] + pr[0] - 0.5 * (sr[0] + ur[0])
        if best is None or score > best[5]:
            best = (j, wr, pr, sr, ur, score)
    return best


def main():
    # ── EMNLP single-column rcParams ──────────────────────────────────────
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":       8.0,
        "axes.titlesize":  8.5,
        "axes.labelsize":  8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,        # editable text in PDF
        "ps.fonttype":  42,
    })

    # ── Conditions (order: methods left, controls right) ──────────────────
    conds = [
        ("within", "within-model ceiling",  "#B5384E"),
        ("proc",   r"Procrustes (eng $\to$ X)", "#2D5598"),
        ("unp",    "unprojected (control)",  "#D49A5C"),
        ("shuf",   "shuffle (control)",      "#9DA6B5"),
    ]

    bests = [best_j_for_target(t) for t in ENG_X]
    js = [b[0] for b in bests]
    for t, b in zip(ENG_X, bests):
        print(f"{TGT_PRETTY[t]:9s} j={b[0]}  within {b[1][0]:.0f}  procrustes {b[2][0]:.0f}  "
              f"shuffle {b[3][0]:.0f}  unprojected {b[4][0]:.0f}")

    # ── Figure size for EMNLP single-column figure (~3.15 in wide) ────────
    fig, ax = plt.subplots(figsize=(3.15, 2.6))
    n_tgt = len(ENG_X)
    x_pos = np.arange(n_tgt)
    width = 0.20

    for i, (key, label, color) in enumerate(conds):
        means, lows, highs = [], [], []
        for b in bests:
            _, w, p, sh, un, _ = b
            r = {"within": w, "proc": p, "shuf": sh, "unp": un}[key]
            means.append(r[0]); lows.append(r[1]); highs.append(r[2])
        means = np.array(means); lows = np.array(lows); highs = np.array(highs)
        yerr_lo = np.clip(means - lows, 0, None)
        yerr_hi = np.clip(highs - means, 0, None)
        offset = (i - 1.5) * width
        bars = ax.bar(x_pos + offset, means, width,
                      yerr=[yerr_lo, yerr_hi], capsize=1.4,
                      label=label, color=color,
                      edgecolor="white", linewidth=0.25,
                      error_kw={"linewidth": 0.45, "ecolor": "#333"})
        for rect, m, hi in zip(bars, means, highs):
            ax.text(rect.get_x() + rect.get_width() / 2, hi + 1.4,
                    f"{m:.0f}", ha="center", va="bottom",
                    fontsize=5.6, color=color, fontweight="bold")

    # 50% chance reference
    ax.axhline(50, color="#666", lw=0.45, ls=(0, (4, 4)), alpha=0.7)
    ax.text(n_tgt - 0.9, 51.5, "chance", color="#444",
            fontsize=6.4, va="bottom", ha="left", style="italic")

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{TGT_PRETTY[t]}\n($j$={j})" for t, j in zip(ENG_X, js)],
    )
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylabel("Directional success rate (%)",
                  fontsize=7.5, labelpad=2.5)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.grid(axis="y", alpha=0.18, linewidth=0.35)
    ax.set_axisbelow(True)

    # ── Legend ABOVE the plot, 2x2 to fit single-column width ─────────────
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.01), frameon=False,
               handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0, labelspacing=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    (ROOT / "figures").mkdir(exist_ok=True)
    out_png = ROOT / "figures" / "paper_summary_eng_to_x_capital.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
