#!/usr/bin/env python3
"""Appendix Figure 10: per-layer CKA, Goldfish vs. Pythia-160M (English) and
Goldfish vs. Zh-Pythia-160M (Chinese).  One figure per setting; the paper shows
the SGPT one (cka_pythia_layers_setting3_sgpt.pdf).

Usage:
  python plot_goldfish_vs_pythia.py --results-dir results --out-dir figures
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

PAIRS = {
    "eng_latn": ("goldfish-models/eng_latn_1000mb", "EleutherAI/pythia-160m", "English"),
    "zho_hans": ("goldfish-models/zho_hans_1000mb", "SJTU-CL/Zh-Pythia-160M", "Chinese"),
}
SETTINGS = {"mean_pool": ("Setting 1 — Mean Pooling", "setting1_mean"),
            "token_aligned": ("Setting 2 — Token-Level", "setting2_token"),
            "sgpt": ("Setting 3 — SGPT", "setting3_sgpt")}


def safe(s):
    return s.replace("/", "_").replace(":", "_")


def find_csv(results_dir, lang, a, b):
    for x, y in ((a, b), (b, a)):
        p = os.path.join(results_dir, f"cka_pythia_{lang}_{safe(x)}_{safe(y)}.csv")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "figure.titlesize": 16, "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })
    for setting, (title, tag) in SETTINGS.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        for ax, (lang, (model_a, model_b, name)) in zip(axes, PAIRS.items()):
            path = find_csv(args.results_dir, lang, model_a, model_b)
            if path is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(name)
                continue
            df = pd.read_csv(path)
            d = df[df["setting"] == setting].sort_values("layer")
            ax.plot(d["layer"], d["cka_matched"], "o-", color="#d62728", linewidth=2,
                    markersize=5, label="Matched")
            ax.plot(d["layer"], d["cka_shuffled"], "s--", color="#1f77b4", linewidth=2,
                    markersize=5, label="Shuffled")
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_title(f"Goldfish vs Pythia-160M ({name})", fontsize=12, fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("CKA", fontsize=11)
        fig.suptitle(f"Per-Layer CKA — {title}", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        for fmt in ("pdf", "png"):
            out = os.path.join(args.out_dir, f"cka_pythia_layers_{tag}.{fmt}")
            fig.savefig(out, dpi=args.dpi, format=fmt, bbox_inches="tight")
            print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
