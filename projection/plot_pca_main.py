#!/usr/bin/env python3
"""
Figure 5: single-layer before/after PCA figure (eng_latn -> fra_latn,
Procrustes, layer 12).  Reads results/<run>/predictions/<pair>/layer_NN.npz
written by fit_projections.py with --save-predictions-for, like
plot_pca_layers.py.

Usage:
  python plot_pca_main.py --run-name eng_fra_pca --pair eng_latn-fra_latn \\
      --method procrustes --layer 12
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ── EMNLP-leaning publication style ────────────────────────────────────────
# Sans-serif inside figure for crisp on-screen / projector reading; PDF
# fonts kept as type-3-free TrueType so the camera-ready remains editable.
PUB_RC = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "TeX Gyre Heros",
                          "DejaVu Sans", "sans-serif"],
    "font.weight":      "regular",
    "font.size":        9.0,
    "axes.labelsize":   8.5,
    "axes.titlesize":   10.5,
    "axes.titleweight": "semibold",
    "legend.fontsize":  9.0,
    "figure.titlesize": 11.0,
    "axes.linewidth":   0.6,
    "axes.edgecolor":   "#9A9AA4",
    "axes.labelcolor":  "#222226",
    "text.color":       "#1F1F24",
    "xtick.color":      "#5A5A66",
    "ytick.color":      "#5A5A66",
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
}

# Slightly desaturated palette — easier on the eye in printed papers, and
# still passes Coblis deuteranopia simulation cleanly.
COLOR_SOURCE = "#2F6FB0"   # source / projected source
COLOR_TARGET = "#E08A3C"   # target

METHOD_TITLE = {
    "procrustes":          "Procrustes",
    "procrustes_centered": "Procrustes (centered)",
    "affine":              "affine least-squares",
    "ridge":               "ridge regression",
    "mlp":                 "two-layer MLP",
}

LANG_DISPLAY = {
    "eng_latn": "English",
    "fra_latn": "French",
    "zho_hans": "Chinese",
    "arb_arab": "Arabic",
    "rus_cyrl": "Russian",
    "spa_latn": "Spanish",
    "hin_deva": "Hindi",
    "deu_latn": "German",
    "jpn_jpan": "Japanese",
}


def _shared_pca(*arrays):
    U = np.vstack(arrays)
    Uc = U - U.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    proj = Uc @ Vt[:2].T
    var_share = (S ** 2) / (S ** 2).sum()
    splits = np.cumsum([a.shape[0] for a in arrays])[:-1]
    return np.split(proj, splits), var_share[:2]


def _kde_contour(ax, pts, color, *, levels=(0.5, 0.8), xlim=None, ylim=None):
    """Gaussian-KDE contour outline for a 2-D point cloud.

    Drawn underneath the scatter to suggest distribution shape without
    overwhelming the dots. Falls back silently if SciPy isn't available.
    """
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return
    if pts.shape[0] < 5:
        return
    kde = gaussian_kde(pts.T, bw_method=0.35)
    gx = np.linspace(xlim[0], xlim[1], 160)
    gy = np.linspace(ylim[0], ylim[1], 160)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Z_sorted = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(Z_sorted) / Z_sorted.sum()
    thresholds = [Z_sorted[np.searchsorted(cum, lv)] for lv in levels]
    thresholds = sorted(set(thresholds))
    if not thresholds:
        return
    # Faint translucent fills + single soft outline.
    ax.contourf(GX, GY, Z, levels=[thresholds[0], Z.max()],
                colors=[color], alpha=0.10, zorder=1)
    ax.contour(GX, GY, Z, levels=thresholds, colors=[color],
               linewidths=0.7, alpha=0.55, zorder=1.5)


def _scatter_panel(ax, p_src, p_tgt, xlim, ylim):
    _kde_contour(ax, p_tgt, COLOR_TARGET, xlim=xlim, ylim=ylim)
    _kde_contour(ax, p_src, COLOR_SOURCE, xlim=xlim, ylim=ylim)
    ax.scatter(p_tgt[:, 0], p_tgt[:, 1],
               s=4.6, alpha=0.50, color=COLOR_TARGET,
               linewidths=0, rasterized=True, marker="o", zorder=2)
    ax.scatter(p_src[:, 0], p_src[:, 1],
               s=4.6, alpha=0.50, color=COLOR_SOURCE,
               linewidths=0, rasterized=True, marker="o", zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_facecolor("#FAFAFC")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#B5B5BE")
    ax.set_xticks([]); ax.set_yticks([])


def _metric_chip(ax, text, *, loc="lower right", emphasis=False):
    x, y, ha, va = {
        "lower right": (0.965, 0.05, "right", "bottom"),
        "lower left":  (0.035, 0.05, "left",  "bottom"),
        "upper right": (0.965, 0.95, "right", "top"),
        "upper left":  (0.035, 0.95, "left",  "top"),
    }[loc]
    bbox = dict(
        facecolor="white",
        edgecolor="#3F6FA0" if emphasis else "#C2C2CC",
        linewidth=0.7 if emphasis else 0.5,
        alpha=0.96, pad=2.6, boxstyle="round,pad=0.28",
    )
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8.2, color="#1A1A1F",
            ha=ha, va=va, fontweight="semibold" if emphasis else "regular",
            bbox=bbox, zorder=10)


def _read_metric(run_name, pair, method, layer, field="p_at_1"):
    csv = RESULTS_DIR / run_name / "projection_results.csv"
    if not csv.exists():
        return None
    import csv as _csv
    with csv.open() as f:
        r = _csv.DictReader(f)
        for row in r:
            if (row.get("pair") == pair
                and row.get("method") == method
                and int(row.get("layer", -1)) == layer):
                try:
                    return float(row[field])
                except (KeyError, ValueError):
                    return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="eng_fra_pca")
    ap.add_argument("--pair", default="eng_latn-fra_latn")
    ap.add_argument("--method", default="procrustes")
    ap.add_argument("--layer", type=int, default=12,
                    help="Layer index (0 = embedding output, 12 = final block).")
    ap.add_argument("--width", type=float, default=6.3,
                    help="Figure width in inches (EMNLP text width ≈ 6.3in).")
    ap.add_argument("--height", type=float, default=3.05,
                    help="Figure height in inches.")
    ap.add_argument("--outname", default=None,
                    help="Override output stem; defaults to a descriptive name.")
    args = ap.parse_args()

    src, tgt = args.pair.split("-")
    pair_dir = RESULTS_DIR / args.run_name / "predictions" / args.pair
    npz = pair_dir / f"layer_{args.layer:02d}.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}")
    data = np.load(npz)
    key = f"pred_{args.method}"
    if key not in data.files:
        raise SystemExit(f"{npz} has no {key}; available: {data.files}")

    X = data["X_test"]
    Y = data["Y_test"]
    Yhat = data[key]
    n_test = X.shape[0]
    (pX, pY, pYh), var2 = _shared_pca(X, Y, Yhat)

    # Shared limits with a small pad so before/after share scale.
    all_xy = np.vstack([pX, pY, pYh])
    pad_x = 0.05 * (all_xy[:, 0].max() - all_xy[:, 0].min() + 1e-9)
    pad_y = 0.05 * (all_xy[:, 1].max() - all_xy[:, 1].min() + 1e-9)
    xlim = (all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ylim = (all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)

    p1_proc = _read_metric(args.run_name, args.pair, args.method, args.layer)
    p1_id   = _read_metric(args.run_name, args.pair, "identity", args.layer)

    src_disp = LANG_DISPLAY.get(src, src)
    tgt_disp = LANG_DISPLAY.get(tgt, tgt)
    method_disp = METHOD_TITLE.get(args.method, args.method)

    with mpl.rc_context(PUB_RC):
        fig = plt.figure(figsize=(args.width, args.height), facecolor="white")
        # Three columns: [Before] [arrow gutter] [After]. The gutter is a
        # narrow axes that hosts only the transformation arrow.
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1.0, 0.26, 1.0],
            wspace=0.04,
        )
        ax_b   = fig.add_subplot(gs[0, 0])
        ax_arr = fig.add_subplot(gs[0, 1])
        ax_a   = fig.add_subplot(gs[0, 2])

        # ── Before ────────────────────────────────────────────────────
        _scatter_panel(ax_b, pX, pY, xlim, ylim)
        ax_b.set_title("Before", pad=6, color="#1A1A1F")
        ax_b.set_xlabel("PC 1", fontsize=8, color="#6A6A75", labelpad=2)
        ax_b.set_ylabel("PC 2", fontsize=8, color="#6A6A75", labelpad=2)
        if p1_id is not None:
            _metric_chip(ax_b, f"P@1  {p1_id*100:.1f}%", loc="lower right")
        ax_b.text(0.035, 0.05,
                  f"PC1 {var2[0]*100:.0f}%   PC2 {var2[1]*100:.0f}%",
                  transform=ax_b.transAxes, fontsize=7.0, color="#6A6A75",
                  ha="left", va="bottom",
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.85, pad=1.4))

        # ── Arrow gutter ──────────────────────────────────────────────
        ax_arr.set_xlim(0, 1); ax_arr.set_ylim(0, 1)
        ax_arr.axis("off")
        arrow = mpatches.FancyArrowPatch(
            (0.10, 0.5), (0.90, 0.5),
            arrowstyle="-|>", mutation_scale=13,
            linewidth=1.1, color="#3F6FA0",
        )
        ax_arr.add_patch(arrow)
        ax_arr.text(0.5, 0.60, method_disp, ha="center", va="bottom",
                    fontsize=8.5, color="#2F4F7A", fontweight="semibold",
                    clip_on=False)
        ax_arr.text(0.5, 0.40, r"$\hat{Y} = X W$", ha="center", va="top",
                    fontsize=8.5, color="#5A6E88",
                    clip_on=False)

        # ── After ─────────────────────────────────────────────────────
        _scatter_panel(ax_a, pYh, pY, xlim, ylim)
        ax_a.set_title("After alignment", pad=6, color="#1A1A1F")
        ax_a.set_xlabel("PC 1", fontsize=8, color="#6A6A75", labelpad=2)
        if p1_proc is not None:
            _metric_chip(ax_a, f"P@1  {p1_proc*100:.1f}%",
                         loc="lower right", emphasis=True)

        # Legend along the top, tight to the title row.
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   color=COLOR_SOURCE, markersize=6.5,
                   label=f"{src_disp} (source / projected source)"),
            Line2D([0], [0], marker="o", linestyle="",
                   color=COLOR_TARGET, markersize=6.5,
                   label=f"{tgt_disp} (target)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center", ncol=2, frameon=False,
            bbox_to_anchor=(0.5, 1.025),
            handletextpad=0.5, columnspacing=2.6,
        )

        # Bottom caption: layer + sample-size context.
        fig.text(
            0.5, -0.025,
            f"Layer {args.layer} sentence representations "
            f"({src_disp} → {tgt_disp}, n = {n_test} test sentences)",
            ha="center", va="top", fontsize=8.6, color="#3A3A42",
            fontstyle="italic",
        )

        fig.subplots_adjust(top=0.84, bottom=0.13, left=0.04, right=0.985)

        out_dir = FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.outname or (
            f"pca_projection_main_{args.pair}_{args.method}_L{args.layer:02d}"
        )
        png_path = out_dir / f"{stem}.png"
        pdf_path = out_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=400)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"wrote {png_path}")
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()