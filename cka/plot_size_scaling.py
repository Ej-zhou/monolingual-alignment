#!/usr/bin/env python3
"""Figure 4: mean off-diagonal matched last-layer CKA vs. training-data size
(5/10/100/1000 MB, FLORES) for the three representations, with the shuffled
baseline (averaged over representations) dashed.

Usage:
  python plot_size_scaling.py --results-dir results/size_scaling
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cka_results import SETTINGS, SIZES, build_matrix, models_at

SIZE_MB = {"5mb": 5, "10mb": 10, "100mb": 100, "1000mb": 1000}
LABEL = {"mean_pool": "Mean", "token_aligned": "Token", "sgpt": "SGPT"}
COLOR = {"mean_pool": "#1f77b4", "token_aligned": "#d62728", "sgpt": "#2ca02c"}
MARKER = {"mean_pool": "o", "token_aligned": "s", "sgpt": "^"}


def off_diagonal_mean(mat):
    vals = mat[~np.eye(mat.shape[0], dtype=bool)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/size_scaling")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    matched = {k: {} for k in SETTINGS}
    shuffled = {k: {} for k in SETTINGS}
    for setting in SETTINGS:
        for size in SIZES:
            models = models_at(size)
            m, s = build_matrix(args.results_dir, models, models, setting, ["flores"])
            matched[setting][size] = off_diagonal_mean(m)
            shuffled[setting][size] = off_diagonal_mean(s)

    print("Off-diagonal mean matched CKA (shuffled avg over settings):")
    for size in SIZES:
        print(f"  {size:>7}  " + "  ".join(f"{LABEL[k]}={matched[k][size]:.3f}" for k in SETTINGS)
              + f"  shuffled={np.nanmean([shuffled[k][size] for k in SETTINGS]):.3f}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 4.0, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    x = np.array([SIZE_MB[s] for s in SIZES], dtype=float)
    fig, ax = plt.subplots(figsize=(3.15, 1.75))
    for k in SETTINGS:
        ax.plot(x, [matched[k][s] for s in SIZES], color=COLOR[k], marker=MARKER[k],
                linestyle="-", label=LABEL[k], markeredgecolor="white", markeredgewidth=0.4,
                zorder=3)
    shuf = np.nanmean([[shuffled[k][s] for s in SIZES] for k in SETTINGS], axis=0)
    ax.plot(x, shuf, color="0.35", marker="x", markersize=4.0, linestyle="--", linewidth=1.0,
            label="Shuffled", zorder=2)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(["5", "10", "100", "1000"])
    ax.set_xlabel("Training data per language (MB, log)", labelpad=2)
    ax.set_ylabel("Mean matched CKA", labelpad=2)
    ax.set_ylim(0.0, 1.0); ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.4, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False,
              handlelength=1.4, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0,
              labelspacing=0.25)
    fig.tight_layout(pad=0.15)
    os.makedirs(args.out_dir, exist_ok=True)
    for fmt in ("pdf", "png"):
        out = os.path.join(args.out_dir, f"cka_size_scaling.{fmt}")
        fig.savefig(out, dpi=args.dpi, format=fmt)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
