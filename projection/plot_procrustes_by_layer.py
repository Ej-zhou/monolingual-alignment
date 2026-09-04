#!/usr/bin/env python3
"""
Per-layer Procrustes retrieval / MSE, one line per pair (Appendix F.1).

Writes to figures/:
  exp6_procrustes_p_at_1_pool_by_layer   P@1 against the train+test pool  (Figure 13)
  exp6_procrustes_p_at_1_by_layer        P@1 against the test pool
  exp6_procrustes_by_layer               P@1 + raw MSE, two panels
  exp6_procrustes_rel_mse_by_layer       MSE relative to the identity baseline

The four headline pairs (eng -> fra, zho, rus, arb) are bold; the other 32 are
drawn faint.  Dots mark each headline pair's best layer.

Usage:
  python plot_procrustes_by_layer.py --run-name all_pairs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HEADLINE_PAIRS = [
    "eng_latn-fra_latn",
    "eng_latn-zho_hans",
    "eng_latn-rus_cyrl",
    "eng_latn-arb_arab",
]
HEADLINE_COLORS = {
    "eng_latn-fra_latn": "#D62728",  # red
    "eng_latn-zho_hans": "#1F77B4",  # blue
    "eng_latn-rus_cyrl": "#2CA02C",  # green
    "eng_latn-arb_arab": "#9467BD",  # purple
}


def short_pair(p: str) -> str:
    return (p.replace("_latn", "").replace("_hans", "")
             .replace("_cyrl", "").replace("_arab", "")
             .replace("_deva", "").replace("_jpan", ""))


def plot_panel(ax, df: pd.DataFrame, metric: str, ylabel: str,
               higher_is_better: bool, mark_peaks: bool) -> None:
    pairs = sorted(df["pair"].unique())
    other = [p for p in pairs if p not in HEADLINE_PAIRS]

    # Background lines: the non-headline pairs, faint, single colormap.
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(other):
        d = df[df["pair"] == p].sort_values("layer")
        ax.plot(d["layer"], d[metric],
                color=cmap(i / max(len(other) - 1, 1)),
                alpha=0.45, linewidth=1.0)

    # Headline pairs: bold, labeled.
    for p in HEADLINE_PAIRS:
        if p not in df["pair"].unique():
            continue
        d = df[df["pair"] == p].sort_values("layer")
        ax.plot(d["layer"], d[metric],
                color=HEADLINE_COLORS[p],
                linewidth=2.4, label=short_pair(p),
                marker="o", markersize=4, zorder=3)
        if mark_peaks:
            best_idx = d[metric].idxmax() if higher_is_better else d[metric].idxmin()
            best_row = d.loc[best_idx]
            ax.scatter([best_row["layer"]], [best_row[metric]],
                       s=80, edgecolors=HEADLINE_COLORS[p],
                       facecolors="white", linewidths=2, zorder=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="all_pairs")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()

    results = args.results or (ROOT / "results" / args.run_name / "projection_results.csv")
    figures = args.figures or (ROOT / "figures")
    figures.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(results)
    df = raw[raw["method"] == "procrustes"].copy()
    if df.empty:
        raise RuntimeError(f"No Procrustes rows in {results}")

    # Compute relative-MSE: procrustes_mse / identity_mse, joined on (pair, layer).
    ident = (raw[raw["method"] == "identity"]
             .rename(columns={"mse": "identity_mse"})
             [["pair", "layer", "identity_mse"]])
    df = df.merge(ident, on=["pair", "layer"], how="left")
    df["rel_mse"] = df["mse"] / df["identity_mse"].replace(0, float("nan"))

    print(f"Read {len(df)} rows from {results}")
    print(f"Pairs covered: {df.pair.nunique()}")
    print(f"Layers per pair (max): {df.groupby('pair')['layer'].max().describe()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_panel(axes[0], df, "p_at_1", "Retrieval P@1",
               higher_is_better=True, mark_peaks=True)
    axes[0].set_title("Procrustes P@1 by layer")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="lower right", title="Headline pairs",
                   fontsize=9)

    plot_panel(axes[1], df, "mse", "Test MSE",
               higher_is_better=False, mark_peaks=True)
    axes[1].set_title("Procrustes test MSE by layer")
    axes[1].legend(frameon=False, loc="upper right", title="Headline pairs",
                   fontsize=9)

    n_other = df["pair"].nunique() - sum(1 for p in HEADLINE_PAIRS if p in df["pair"].unique())
    fig.suptitle(f"Procrustes per-layer reconstruction "
                 f"({df.pair.nunique()} pairs, {n_other} non-headline shown faint)",
                 y=1.02)
    fig.tight_layout()
    out = figures / "exp6_procrustes_by_layer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} and .pdf")

    # Standalone P@1-only figure for paper inclusion.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_panel(ax2, df, "p_at_1", "Retrieval P@1",
               higher_is_better=True, mark_peaks=True)
    ax2.set_title("Procrustes P@1 by layer")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, loc="lower right", title="Headline pairs",
               fontsize=9)
    fig2.tight_layout()
    out2 = figures / "exp6_procrustes_p_at_1_by_layer.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    fig2.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out2} and .pdf")

    # P@1 against the full target-language pool (train + test) — harder retrieval.
    if "p_at_1_pool" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        plot_panel(ax4, df, "p_at_1_pool",
                   "P@1 (against full target-lang pool: train + test)",
                   higher_is_better=True, mark_peaks=True)
        ax4.set_title("Procrustes P@1 against larger candidate pool")
        ax4.set_ylim(0, 1)
        ax4.legend(frameon=False, loc="lower right", title="Headline pairs",
                   fontsize=9)
        fig4.tight_layout()
        out4 = figures / "exp6_procrustes_p_at_1_pool_by_layer.png"
        fig4.savefig(out4, dpi=200, bbox_inches="tight")
        fig4.savefig(out4.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig4)
        print(f"Wrote {out4} and .pdf")

    # Standalone relative-MSE figure: procrustes_mse / identity_mse.
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    plot_panel(ax3, df, "rel_mse",
               "Relative MSE  (Procrustes / Identity)",
               higher_is_better=False, mark_peaks=True)
    ax3.set_title("Procrustes MSE relative to no-projection baseline")
    ax3.axhline(1.0, color="grey", linestyle=":", linewidth=1.0)
    ax3.legend(frameon=False, loc="upper left", title="Headline pairs",
               fontsize=9)
    fig3.tight_layout()
    out3 = figures / "exp6_procrustes_rel_mse_by_layer.png"
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    fig3.savefig(out3.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig3)
    print(f"Wrote {out3} and .pdf")

    # Console summary: best (= lowest) relative-MSE layer per headline pair.
    print("\nBest relative-MSE layers (Procrustes / Identity):")
    for p in HEADLINE_PAIRS:
        if p not in df["pair"].unique():
            continue
        sub = df[df["pair"] == p].dropna(subset=["rel_mse"])
        if sub.empty:
            continue
        i = sub["rel_mse"].idxmin()
        print(f"  {short_pair(p):<10} min rel_mse at layer "
              f"{int(sub.loc[i, 'layer']):>2} "
              f"(rel={sub.loc[i, 'rel_mse']:.3f}, "
              f"abs={sub.loc[i, 'mse']:.3f}, identity={sub.loc[i, 'identity_mse']:.3f})")

    # Console summary: peak layer per headline pair (P@1 and MSE).
    print("\nPeak layers (headline pairs):")
    for p in HEADLINE_PAIRS:
        if p not in df["pair"].unique():
            continue
        sub = df[df["pair"] == p]
        i_p1 = sub["p_at_1"].idxmax()
        i_mse = sub["mse"].idxmin()
        print(f"  {short_pair(p):<10} P@1 peaks at layer {sub.loc[i_p1, 'layer']:>2} "
              f"(P@1={sub.loc[i_p1, 'p_at_1']:.3f}); "
              f"MSE min at layer {sub.loc[i_mse, 'layer']:>2} "
              f"(MSE={sub.loc[i_mse, 'mse']:.3f})")


if __name__ == "__main__":
    main()
