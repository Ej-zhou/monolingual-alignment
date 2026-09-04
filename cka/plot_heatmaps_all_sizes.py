#!/usr/bin/env python3
"""Appendix Figures 14-16: one figure per representation with 9x9 last-layer
CKA heatmaps at all four training sizes (columns) for matched (top row) and
shuffled (bottom row), FLORES.

Usage:
  python plot_heatmaps_all_sizes.py --results-dir results/size_scaling
"""
import argparse
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cka_results import (LANG_NAMES, SETTINGS, SIZE_LABEL, build_matrix, lang_of,
                         models_at)

SETTING_TITLE = {"mean_pool": "Sentence-Level Mean Pooling",
                 "token_aligned": "Token-Level Word-Aligned",
                 "sgpt": "SGPT Position-Weighted"}
FILE_TAG = {"mean_pool": "setting1_mean", "token_aligned": "setting2_token",
            "sgpt": "setting3_sgpt"}
ORDER = ["1000mb", "100mb", "10mb", "5mb"]


def publication_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "figure.titlesize": 16, "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def plot_setting(results_dir, setting, out_dir, dpi):
    n_sizes = len(ORDER)
    fig = plt.figure(figsize=(4.8 * n_sizes + 0.5, 2 * 4.8 + 0.8))
    gs = gridspec.GridSpec(2, n_sizes + 1, width_ratios=[1] * n_sizes + [0.03],
                           wspace=0.08, hspace=0.18, left=0.06, right=0.95, top=0.92, bottom=0.04)
    cmap, vmin, vmax = "RdYlBu_r", 0.0, 1.0
    for col, size in enumerate(ORDER):
        models = models_at(size)
        labels = [LANG_NAMES[lang_of(m)] for m in models]
        mat_m, mat_s = build_matrix(results_dir, models, models, setting, ["flores"], self_value=1.0)
        for row, (mat, cond) in enumerate([(mat_m, "Matched"), (mat_s, "Shuffled")]):
            ax = fig.add_subplot(gs[row, col])
            sns.heatmap(mat, ax=ax, xticklabels=labels, yticklabels=labels if col == 0 else False,
                        vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", square=True,
                        cbar=False, annot_kws={"size": 8, "weight": "medium"},
                        linewidths=0.5, linecolor="white")
            if row == 0:
                ax.set_title(SIZE_LABEL[size], fontsize=14, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(cond, fontsize=13, fontweight="bold", labelpad=8)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("CKA (last layer)", fontsize=12); cbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Last-Layer Pairwise CKA — {SETTING_TITLE[setting]}", fontsize=16,
                 fontweight="bold", y=0.98)
    for fmt in ("pdf", "png"):
        out = os.path.join(out_dir, f"cka_heatmaps_{FILE_TAG[setting]}_all_sizes.{fmt}")
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
