#!/usr/bin/env python3
"""Appendix Figures 7-9: cross-size CKA (1000 MB vs 100 MB and 10 MB vs 5 MB)
for each representation.  Rows = languages at size A, columns = languages at
size B; the diagonal is the same language at two sizes.

Usage:
  python plot_cross_size_heatmaps.py --results-dir results/size_scaling
"""
import argparse
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from cka_results import LANG_NAMES, SETTINGS, SIZE_LABEL, build_matrix, lang_of, models_at
from plot_heatmaps_all_sizes import FILE_TAG, SETTING_TITLE, publication_style

CROSS_SIZE_PAIRS = [("1000mb", "100mb"), ("10mb", "5mb")]


def plot_setting(results_dir, setting, out_dir, dpi):
    n_pairs = len(CROSS_SIZE_PAIRS)
    fig = plt.figure(figsize=(5.5 * n_pairs + 0.5, 2 * 5.2 + 0.8))
    gs = gridspec.GridSpec(2, n_pairs + 1, width_ratios=[1] * n_pairs + [0.03],
                           wspace=0.10, hspace=0.18, left=0.07, right=0.94, top=0.91, bottom=0.04)
    cmap, vmin, vmax = "RdYlBu_r", 0.0, 1.0
    for col, (sA, sB) in enumerate(CROSS_SIZE_PAIRS):
        rows_m, cols_m = models_at(sA), models_at(sB)
        labels_r = [LANG_NAMES[lang_of(m)] for m in rows_m]
        labels_c = [LANG_NAMES[lang_of(m)] for m in cols_m]
        mat_m, mat_s = build_matrix(results_dir, rows_m, cols_m, setting, ["flores"])
        for row, (mat, cond) in enumerate([(mat_m, "Matched"), (mat_s, "Shuffled")]):
            ax = fig.add_subplot(gs[row, col])
            sns.heatmap(mat, ax=ax, xticklabels=labels_c, yticklabels=labels_r if col == 0 else False,
                        vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", square=True,
                        cbar=False, annot_kws={"size": 8, "weight": "medium"},
                        linewidths=0.5, linecolor="white")
            if row == 0:
                ax.set_title(f"{SIZE_LABEL[sA]} vs {SIZE_LABEL[sB]}", fontsize=14,
                             fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(cond, fontsize=13, fontweight="bold", labelpad=8)
            ax.set_xlabel(f"Model ({SIZE_LABEL[sB]})", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("CKA (last layer)", fontsize=12); cbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Cross-Size Last-Layer Pairwise CKA — {SETTING_TITLE[setting]}",
                 fontsize=16, fontweight="bold", y=0.96)
    for fmt in ("pdf", "png"):
        out = os.path.join(out_dir, f"cka_cross_size_{FILE_TAG[setting]}.{fmt}")
        fig.savefig(out, dpi=dpi, format=fmt)
        print(f"Saved {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/size_scaling")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    publication_style()
    for setting in SETTINGS:
        plot_setting(args.results_dir, setting, args.out_dir, args.dpi)


if __name__ == "__main__":
    main()
