#!/usr/bin/env python3
"""Appendix Figure 17: 3x3 grid of 9x9 source x target heatmaps at the final
layer — rows = method (Procrustes, Affine, MLP), columns = metric (P@1 against
the train+test pool, test MSE, post-projection CKA).  Upper triangle only
(pairs are fit A -> B with A before B in the language ordering).

Usage:
  python plot_methods_x_metrics.py --run-name all_pairs
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent

# Same ordering as exp2 / the SLURM submission. Maps the FLORES code we use
# in pair filenames to the ISO-639-3 short label shown on heatmap axes.
LANG_ORDER = [
    "eng_latn", "zho_hans", "spa_latn", "arb_arab", "hin_deva",
    "fra_latn", "rus_cyrl", "deu_latn", "jpn_jpan",
]
LANG_SHORT = {
    "eng_latn": "eng",
    "zho_hans": "zho",
    "spa_latn": "spa",
    "arb_arab": "arb",
    "hin_deva": "hin",
    "fra_latn": "fra",
    "rus_cyrl": "rus",
    "deu_latn": "deu",
    "jpn_jpan": "jpn",
}

METRIC_TITLE = {
    "p_at_1_pool": "Retrieval P@1",
    "mse": "Test MSE",
    "cka": "Post-projection CKA",
}
METHOD_TITLE = {
    "procrustes": "Procrustes",
    "ridge":      "Affine",
    "mlp":        "MLP",
}


def to_matrix(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Build a 9x9 matrix indexed by LANG_ORDER. Cells without data become NaN."""
    n = len(LANG_ORDER)
    M = np.full((n, n), np.nan, dtype=float)
    idx_of = {l: i for i, l in enumerate(LANG_ORDER)}
    for _, r in df.iterrows():
        pair = r["pair"]
        try:
            a, b = pair.split("-")
        except ValueError:
            continue
        if a not in idx_of or b not in idx_of:
            continue
        M[idx_of[a], idx_of[b]] = r[metric]
    return M


def draw_heatmap(ax, M: np.ndarray, *, cmap, vmin, vmax,
                 title: str, annotate: bool = True,
                 cbar_label: str = "") -> mpl.image.AxesImage:
    n = M.shape[0]
    masked = np.ma.array(M, mask=np.isnan(M))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#E5E5E5")  # grey for missing (lower triangle + diagonal)

    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([LANG_SHORT[l] for l in LANG_ORDER], rotation=0, fontsize=9)
    ax.set_yticklabels([LANG_SHORT[l] for l in LANG_ORDER], fontsize=9)
    ax.set_xlabel("target", fontsize=10)
    ax.set_ylabel("source", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(top=False, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for i in range(n):
            for j in range(n):
                if np.isnan(M[i, j]):
                    continue
                # Choose text color from luminance for readability.
                norm = (M[i, j] - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if norm > 0.55 else "black"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)
    return im


def figure_combined(df: pd.DataFrame, out_path: Path) -> None:
    """3x3 grid: rows = method (Procrustes, Affine, MLP),
    columns = metric (P@1, MSE, CKA). Each cell is a 9x9 source x target heatmap."""
    methods = [m for m in ["procrustes", "ridge", "mlp"] if m in df["method"].unique()]
    metrics = [
        ("p_at_1_pool", "Blues",  "P@1",  0.0,  1.0),
        ("mse",         "Reds_r", "MSE",  None, None),
        ("cka",         "Greens", "CKA",  0.0,  1.0),
    ]

    # Shared vmin/vmax for MSE across methods so cells are visually comparable.
    mse_vals = []
    for m in methods:
        sub = df[df["method"] == m]
        if "mse" in sub.columns:
            mse_vals.append(sub["mse"].dropna().values)
    if mse_vals:
        all_mse = np.concatenate(mse_vals)
        mse_vmin = float(np.percentile(all_mse, 1))
        mse_vmax = float(np.percentile(all_mse, 99))
    else:
        mse_vmin, mse_vmax = 0.0, 1.0

    fig, axes = plt.subplots(len(methods), len(metrics),
                             figsize=(5 * len(metrics), 4.6 * len(methods)))
    if len(methods) == 1:
        axes = np.array([axes])
    if len(metrics) == 1:
        axes = axes.reshape(-1, 1)

    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        for j, (metric, cmap, title, vmin, vmax) in enumerate(metrics):
            ax = axes[i, j]
            if metric not in sub.columns:
                ax.set_axis_off()
                ax.set_title(f"{METHOD_TITLE.get(m, m)} — {title} (missing)")
                continue
            M = to_matrix(sub, metric)
            if metric == "mse":
                vmin_use, vmax_use = mse_vmin, mse_vmax
            elif vmin is None or vmax is None:
                finite = M[np.isfinite(M)]
                vmin_use, vmax_use = float(finite.min()), float(finite.max())
            else:
                vmin_use, vmax_use = vmin, vmax
            im = draw_heatmap(ax, M, cmap=cmap, vmin=vmin_use, vmax=vmax_use,
                              title=f"{METHOD_TITLE.get(m, m)} — {title}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Cross-lingual reconstruction: methods × metrics "
                 "(36 pairs, final layer)", fontsize=13, y=1.01)
    fig.tight_layout()

    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="all_pairs")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the PDF/PNG. Defaults to figures/.")
    args = parser.parse_args()

    results = args.results or (ROOT / "results" / args.run_name / "projection_results.csv")
    if not results.exists():
        raise FileNotFoundError(f"No results CSV at {results}. "
                                f"Has the SLURM job for {args.run_name} finished?")
    out_dir = args.output_dir or (ROOT / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results)
    last = df.groupby("pair")["layer"].transform("max")
    df = df[df["layer"] == last].copy()
    print(f"Read {results}")
    print(f"Pairs in CSV: {df.pair.nunique()} / 36 expected")
    print(f"Methods: {sorted(df.method.unique())}")
    if "p_at_1_pool" not in df.columns:
        raise RuntimeError(
            "p_at_1_pool column missing from the results CSV."
        )

    figure_combined(df, out_dir / "figure_a_methods_x_metrics.pdf")


if __name__ == "__main__":
    main()
