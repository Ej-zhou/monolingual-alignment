#!/usr/bin/env python3
"""
Appendix Figure 18: before/after 2-D PCA panels at every layer for one pair
and projection method (default: eng_latn -> fra_latn, Procrustes).

  Before: source X_test (blue) vs target Y_test (orange).
  After:  predicted Y_hat (blue) vs target Y_test (orange).

Both panels for a given (layer, method) share a single 2-D PCA basis
fit on the union {X_test, Y_test, Y_hat}, so distance shrinkage from
the projection is visible directly. Saves PNG (300 dpi) and PDF
(vector) for inclusion in papers / appendices.

Reads predictions saved by fit_projections.py when --save-predictions-for
is set:
    results/<run-name>/predictions/<pair>/layer_<NN>.npz
        X_test, Y_test, pred_<method> arrays.

Usage:
  python plot_pca_layers.py --run-name eng_fra_pca --pair eng_latn-fra_latn \\
      --methods procrustes
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ── Publication style ──────────────────────────────────────────────────────
PUB_RC = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.weight":      "regular",
    "font.size":        9.5,
    "axes.labelsize":   9.5,
    "axes.titlesize":   10.5,
    "axes.titleweight": "regular",
    "legend.fontsize":  9.5,
    "figure.titlesize": 11.5,
    "axes.linewidth":   0.6,
    "axes.edgecolor":   "#888888",
    "axes.labelcolor":  "#222222",
    "text.color":       "#222222",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "pdf.fonttype":     42,   # editable in Illustrator / submitted PDFs
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}
COLOR_SOURCE = "#3A7DB5"   # muted blue — source / projected source
COLOR_TARGET = "#E89B53"   # warm orange — target

METHOD_TITLE = {
    "procrustes":          "Procrustes (rotation only)",
    "procrustes_centered": "Procrustes (centered, rotation + bias)",
    "affine":              "Affine least-squares",
    "ridge":               "Ridge regression",
    "mlp":                 "Two-layer MLP",
}


def _shared_pca(X: np.ndarray, Y: np.ndarray, Yhat: np.ndarray):
    U = np.vstack([X, Y, Yhat])
    Uc = U - U.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    proj = Uc @ Vt[:2].T
    var_share = (S ** 2) / (S ** 2).sum()
    return proj, var_share[:2]


def _plot_panel(ax, p_blue, p_orange, lim) -> None:
    # Smaller, softer dots so dense clouds reveal structure rather than blob.
    ax.scatter(p_blue[:, 0], p_blue[:, 1],
                s=2.2, alpha=0.42, color=COLOR_SOURCE,
                linewidths=0, rasterized=True, marker="o")
    ax.scatter(p_orange[:, 0], p_orange[:, 1],
                s=2.2, alpha=0.42, color=COLOR_TARGET,
                linewidths=0, rasterized=True, marker="o")
    ax.set_xlim(lim["x"])
    ax.set_ylim(lim["y"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#FBFBFD")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#B8B8C2")


def plot_method(pair_dir: Path, pair: str, method: str,
                 layers: List[int], out_dir: Path) -> None:
    """A4-portrait layout: two side-by-side blocks of (before, after).

    Left block holds the first half of layers, right block holds the
    second half. With 13 Goldfish layers the split is 7 / 6, leaving
    one empty cell in the bottom-right pair.
    """
    src, tgt = pair.split("-")

    half = (len(layers) + 1) // 2   # 7 for 13 layers
    n_rows = half
    layer_grid = [(layers[i] if i < len(layers) else None,
                    layers[i + half] if i + half < len(layers) else None)
                   for i in range(n_rows)]

    # A4 portrait: 8.27 x 11.69 in. Reserve a bit for title + legend at top.
    A4_W, A4_H = 8.27, 11.69

    with mpl.rc_context(PUB_RC):
        # Five columns: [before_L, after_L, spacer, before_R, after_R].
        # Column 2 is an invisible gutter so the two layer blocks read
        # as separate units without crowding L7's "Before" against the
        # left block's "After".
        fig = plt.figure(figsize=(A4_W, A4_H), facecolor="white")
        gs = fig.add_gridspec(
            n_rows, 5,
            wspace=0.10, hspace=0.18,
            width_ratios=[1.0, 1.0, 0.30, 1.0, 1.0],
        )
        axes = np.empty((n_rows, 4), dtype=object)
        for r in range(n_rows):
            axes[r, 0] = fig.add_subplot(gs[r, 0])
            axes[r, 1] = fig.add_subplot(gs[r, 1])
            axes[r, 2] = fig.add_subplot(gs[r, 3])
            axes[r, 3] = fig.add_subplot(gs[r, 4])

        def _draw_layer(ax_b, ax_a, layer):
            f = pair_dir / f"layer_{layer:02d}.npz"
            if not f.exists() or f"pred_{method}" not in np.load(f).files:
                for ax in (ax_b, ax_a):
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                             transform=ax.transAxes, color="0.5", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_color("#B8B8C2"); s.set_linewidth(0.5)
                return
            data = np.load(f)
            X = data["X_test"]; Y = data["Y_test"]
            Yhat = data[f"pred_{method}"]
            proj, var2 = _shared_pca(X, Y, Yhat)
            n_x, n_y = X.shape[0], Y.shape[0]
            pX  = proj[:n_x]
            pY  = proj[n_x:n_x + n_y]
            pYh = proj[n_x + n_y:]
            xs, ys = proj[:, 0], proj[:, 1]
            pad_x = 0.04 * (xs.max() - xs.min() + 1e-9)
            pad_y = 0.04 * (ys.max() - ys.min() + 1e-9)
            lim = dict(
                x=(xs.min() - pad_x, xs.max() + pad_x),
                y=(ys.min() - pad_y, ys.max() + pad_y),
            )
            _plot_panel(ax_b, pX,  pY, lim)
            _plot_panel(ax_a, pYh, pY, lim)
            ax_b.set_ylabel(f"L{layer}", fontsize=9, rotation=0,
                             labelpad=10, color="#222222",
                             ha="right", va="center")
            ax_b.text(
                0.03, 0.04,
                f"PC1 {var2[0]*100:.0f}%  PC2 {var2[1]*100:.0f}%",
                transform=ax_b.transAxes, fontsize=6.5, color="#555555",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.85, pad=1.2),
            )

        for row, (lL, lR) in enumerate(layer_grid):
            ax_bL, ax_aL, ax_bR, ax_aR = axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3]
            if lL is not None:
                _draw_layer(ax_bL, ax_aL, lL)
            else:
                for ax in (ax_bL, ax_aL):
                    ax.set_visible(False)
            if lR is not None:
                _draw_layer(ax_bR, ax_aR, lR)
            else:
                for ax in (ax_bR, ax_aR):
                    ax.set_visible(False)

        # Block-level column titles (top row only).
        for col, label in [(0, "Before"), (1, "After"),
                            (2, "Before"), (3, "After")]:
            axes[0, col].set_title(label, fontsize=10, pad=6,
                                    fontweight="bold", color="#1A1A1A")
        # Bottom-row PC label hints on the bottom-most visible row of each block.
        for col in range(4):
            for r in range(n_rows - 1, -1, -1):
                if axes[r, col].get_visible():
                    axes[r, col].set_xlabel("PC 1", fontsize=7,
                                             color="#777777", labelpad=2)
                    break

        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                    color=COLOR_SOURCE, markersize=6,
                    label=f"{src} (source / projected source)"),
            Line2D([0], [0], marker="o", linestyle="",
                    color=COLOR_TARGET, markersize=6,
                    label=f"{tgt} (target)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center", ncol=2, frameon=False, fontsize=9.5,
            bbox_to_anchor=(0.5, 0.965),
            handletextpad=0.5, columnspacing=2.4,
        )
        fig.suptitle(
            f"Sentence reps before / after {METHOD_TITLE.get(method, method)} "
            f"mapping  ({src} → {tgt})",
            y=0.995, fontsize=11.5, fontweight="semibold", color="#111111",
        )

        fig.subplots_adjust(top=0.92, bottom=0.04, left=0.05, right=0.985)

        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"pca_projection_{pair}_{method}.png"
        pdf_path = out_dir / f"pca_projection_{pair}_{method}.pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"  wrote {png_path}")
        print(f"  wrote {pdf_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--pair", default="eng_latn-fra_latn",
                     help="Which pair's predictions to plot (must have been "
                          "passed via --save-predictions-for during the run).")
    ap.add_argument("--methods", nargs="*",
                     default=["procrustes", "procrustes_centered",
                              "affine", "ridge", "mlp"])
    args = ap.parse_args()

    pair_dir = RESULTS_DIR / args.run_name / "predictions" / args.pair
    if not pair_dir.exists():
        raise SystemExit(
            f"No predictions at {pair_dir}. "
            f"Re-run fit_projections.py with --save-predictions-for {args.pair}."
        )

    layer_files = sorted(pair_dir.glob("layer_*.npz"))
    layers = [int(f.stem.split("_")[1]) for f in layer_files]
    if not layers:
        raise SystemExit(f"No layer_*.npz files in {pair_dir}")
    print(f"Found {len(layers)} layers: {layers}")

    fig_dir = FIGURES_DIR
    for m in args.methods:
        plot_method(pair_dir, args.pair, m, layers, fig_dir)


if __name__ == "__main__":
    main()
