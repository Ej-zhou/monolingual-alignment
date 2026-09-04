#!/usr/bin/env python3
"""Figure 3: single-column 9x9 last-layer CKA heatmap (mean pooling, Goldfish
1000 MB, averaged over the four parallel datasets).  Lower triangle = matched,
upper triangle = shuffled.

Usage:
  python plot_heatmap_singlecol.py --results-dir results/main
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from cka_results import DATASETS, build_matrix, lang_of, models_at

DISPLAY = {"eng_latn": "EN", "zho_hans": "ZH", "spa_latn": "ES", "arb_arab": "AR",
           "hin_deva": "HI", "fra_latn": "FR", "rus_cyrl": "RU", "deu_latn": "DE",
           "jpn_jpan": "JA"}
CMAP, BG_COLOR, DIAG_COLOR = "YlGnBu", "#F2F2F2", "#E8E8E8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/main")
    ap.add_argument("--setting", default="mean_pool")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default="figures/cka_singlecol_meanpool_1000mb.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    models = models_at("1000mb")
    labels = [DISPLAY[lang_of(m)] for m in models]
    mat_m, mat_s = build_matrix(args.results_dir, models, models, args.setting, args.datasets)

    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    all_vals = np.concatenate([mat_m[~np.isnan(mat_m)], mat_s[~np.isnan(mat_s)]])
    vmin = float(np.floor(np.nanmin(all_vals) * 10) / 10)
    vmax = float(np.ceil(np.nanmax(all_vals) * 10) / 10)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(3.15, 3.15))
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=DIAG_COLOR,
                                           edgecolor="white", linewidth=0.7))
                continue
            val = mat_m[i, j] if i > j else mat_s[i, j]   # lower = matched, upper = shuffled
            fc = BG_COLOR if np.isnan(val) else cmap(norm(val))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white", linewidth=0.7))
            if not np.isnan(val):
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                        fontsize=4.8, color="#222222" if lum > 0.55 else "white")
    ax.plot([0, n], [0, n], color="white", linewidth=1.2, zorder=3)
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, fontsize=7.0); ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(axis="both", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(n * 0.22, n * 0.78, "Matched", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color="#1A4E6E", zorder=5, bbox=bbox)
    ax.text(n * 0.78, n * 0.22, "Shuffled", ha="center", va="center", fontsize=7.2,
            fontweight="bold", color="#6A4C93", zorder=5, bbox=bbox)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.10, aspect=28)
    cb.set_label("CKA (last layer, mean-pool, avg over datasets)", fontsize=6.8, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.outline.set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
