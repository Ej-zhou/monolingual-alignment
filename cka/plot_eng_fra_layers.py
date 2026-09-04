#!/usr/bin/env python3
"""Figure 2: layerwise English-French CKA (Goldfish 1000 MB) under the three
representations, averaged over the four parallel datasets.  Solid = matched,
dashed = shuffled control.

Usage:
  python plot_eng_fra_layers.py --results-dir results/main
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from cka_results import load_pair

MODEL_A = "goldfish-models/eng_latn_1000mb"
MODEL_B = "goldfish-models/fra_latn_1000mb"

SETTINGS = {"mean_pool": "Mean pooling", "token_aligned": "Token-aligned",
            "sgpt": "SGPT pooling"}
SETTING_COLOURS = {"mean_pool": "#2D6A9F", "sgpt": "#E8A838", "token_aligned": "#6BAA75"}
SETTING_MARKERS = {"mean_pool": "o", "sgpt": "s", "token_aligned": "D"}
# Layer at which each setting's matched-shuffled gap is annotated.
ANNO_LAYERS = {"sgpt": 2, "token_aligned": 6, "mean_pool": 10}
ANNO_DX = {"sgpt": -1.3, "token_aligned": 1.3, "mean_pool": 1.3}


def make_figure(avg):
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    layers = sorted(avg["layer"].unique())
    y_vals = avg["cka_matched"].tolist() + avg["cka_shuffled"].tolist()
    y_lo, y_hi = max(0.0, min(y_vals) - 0.04), min(1.0, max(y_vals) + 0.06)

    shuffled_label_done = False
    for setting, label in SETTINGS.items():
        sub = avg[avg["setting"] == setting].sort_values("layer")
        if sub.empty:
            continue
        clr, mkr = SETTING_COLOURS[setting], SETTING_MARKERS[setting]
        l, m, s = sub["layer"].values, sub["cka_matched"].values, sub["cka_shuffled"].values
        ax.fill_between(l, m, s, color=clr, alpha=0.06, zorder=1)
        ax.plot(l, s, color=clr, linewidth=0.9, linestyle="--", dash_capstyle="round",
                alpha=0.55, label=None if shuffled_label_done else "Shuffled (control)",
                zorder=2)
        shuffled_label_done = True
        ax.plot(l, m, color=clr, linewidth=1.4, marker=mkr, markersize=3.0,
                markeredgecolor="white", markeredgewidth=0.3, label=label, zorder=3)

        # Gap annotation
        idx = int(np.where(l == ANNO_LAYERS[setting])[0][0])
        mv, sv = m[idx], s[idx]
        ax.annotate("", xy=(l[idx], mv - 0.005), xytext=(l[idx], sv + 0.005),
                    arrowprops=dict(arrowstyle="<->", color=clr, lw=0.9, shrinkA=0, shrinkB=0))
        ax.text(l[idx] + ANNO_DX[setting], (mv + sv) / 2, f"$\\Delta$={mv - sv:.2f}",
                fontsize=6, color=clr, fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_xticks(layers[::2])
    ax.set_xlabel("Layer", fontsize=8.5, labelpad=2)
    ax.set_ylabel("CKA similarity", fontsize=8.5, labelpad=2)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(0.5)
    ax.yaxis.grid(True, which="major", color="#EEEEEE", linewidth=0.4)
    ax.tick_params(axis="both", which="major", labelsize=7.5, colors="#333333",
                   width=0.5, length=2.5, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=0)
    leg = ax.legend(loc="upper center", fontsize=6, frameon=True, framealpha=0.92,
                    edgecolor="#DDDDDD", fancybox=False, handlelength=1.4, borderpad=0.3,
                    labelspacing=0.22, handletextpad=0.35, columnspacing=1.0, ncol=4,
                    bbox_to_anchor=(0.5, -0.22))
    leg.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/main")
    ap.add_argument("--out", default="figures/eng_fra_1000mb_combined.pdf")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300, "savefig.dpi": 300,
    })
    df = load_pair(args.results_dir, MODEL_A, MODEL_B)
    if df.empty:
        raise SystemExit(f"No eng-fra results in {args.results_dir}")
    avg = (df.groupby(["setting", "layer"])[["cka_matched", "cka_shuffled"]]
             .mean().reset_index())
    fig = make_figure(avg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
