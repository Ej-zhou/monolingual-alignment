#!/usr/bin/env python3
"""Method comparison at the final layer over all 36 pairs (Section 4.2).

Reads results/<run-name>/projection_results.csv and writes to figures/:

  figure_4_procrustes_vs_affine.{pdf,png}   Appendix Figure 12: per-pair scatter of
      Procrustes vs Affine on P@1 (train+test pool) and on test MSE, with the
      paired mean difference and its bootstrap 95% CI.
  table_method_comparison.{csv,md,tex}       Table 4 / Table 23: mean over pairs of
      P@1 (test pool), P@1 (train+test pool), MSE and CKA per method with
      percentile-bootstrap 95% CIs (10,000 resamples), the identity baseline,
      and how many pairs each learned method wins per metric.

"Affine" in the paper is the ridge-regularised linear map (``ridge`` rows).

Usage:
  python plot_method_comparison.py --run-name all_pairs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LANG_SHORT = {
    "eng_latn": "eng", "zho_hans": "zho", "spa_latn": "spa", "arb_arab": "arb",
    "hin_deva": "hin", "fra_latn": "fra", "rus_cyrl": "rus", "deu_latn": "deu",
    "jpn_jpan": "jpn",
}
METHODS = ["procrustes", "ridge", "mlp", "identity"]
METHOD_LABEL = {"procrustes": "Procrustes", "ridge": "Affine", "mlp": "MLP",
                "identity": "Identity"}
METRICS = [("p_at_1", "P@1 (std)", True), ("p_at_1_pool", "P@1 (hard)", True),
           ("mse", "MSE", False), ("cka", "CKA", True)]
N_BOOT = 10000


def short_pair(p):
    a, b = p.split("-")
    return f"{LANG_SHORT.get(a, a)}→{LANG_SHORT.get(b, b)}"


def load_final_layer(results_csv):
    df = pd.read_csv(results_csv)
    last = df.groupby("pair")["layer"].transform("max")
    return df[df["layer"] == last].copy()


def bootstrap_ci(values, n_boot=N_BOOT, ci=95.0, seed=0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    half = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [half, 100.0 - half])
    return float(lo), float(hi)


def paired_diff_ci(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return float(np.mean(d)), *bootstrap_ci(d)


def label_outliers(ax, x, y, labels, k=5):
    diff = np.abs(np.asarray(x) - np.asarray(y))
    for i in np.argsort(-diff)[:k]:
        ax.annotate(labels[i], (x[i], y[i]), xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color="#444444")


def make_figure(df, out_path):
    pv = (df[df["method"].isin(["procrustes", "ridge"])]
          .pivot(index="pair", columns="method", values=["p_at_1_pool", "mse"]))
    pv = pv.dropna()
    pairs = pv.index.tolist()
    short = [short_pair(p) for p in pairs]
    proc_p1, ridge_p1 = pv[("p_at_1_pool", "procrustes")].values, pv[("p_at_1_pool", "ridge")].values
    proc_mse, ridge_mse = pv[("mse", "procrustes")].values, pv[("mse", "ridge")].values

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    ax = axes[0]
    ax.scatter(ridge_p1, proc_p1, s=22, color="#3B7BBF", edgecolors="white", linewidths=0.5, zorder=3)
    lo, hi = min(ridge_p1.min(), proc_p1.min()) * 0.97, max(ridge_p1.max(), proc_p1.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Affine  P@1"); ax.set_ylabel("Procrustes  P@1")
    ax.set_title("(a) Retrieval P@1", fontsize=10); ax.set_aspect("equal", adjustable="box")
    label_outliers(ax, ridge_p1, proc_p1, short)
    ax.grid(alpha=0.25, linestyle=":")
    md, clo, chi = paired_diff_ci(proc_p1, ridge_p1)
    ax.text(0.02, 0.98, f"Procrustes higher in {int((proc_p1 > ridge_p1).sum())}/{len(pairs)} pairs\n"
            f"$\\Delta$ = {md:+.3f} [95% CI {clo:+.3f}, {chi:+.3f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#333333")

    ax = axes[1]
    ax.scatter(ridge_mse, proc_mse, s=22, color="#C25E5E", edgecolors="white", linewidths=0.5, zorder=3)
    lo, hi = min(ridge_mse.min(), proc_mse.min()) * 0.97, max(ridge_mse.max(), proc_mse.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Affine  Test MSE"); ax.set_ylabel("Procrustes  Test MSE")
    ax.set_title("(b) Reconstruction MSE", fontsize=10); ax.set_aspect("equal", adjustable="box")
    label_outliers(ax, ridge_mse, proc_mse, short)
    ax.grid(alpha=0.25, linestyle=":")
    md, clo, chi = paired_diff_ci(proc_mse, ridge_mse)
    ax.text(0.02, 0.98, f"Affine lower in {int((proc_mse > ridge_mse).sum())}/{len(pairs)} pairs\n"
            f"$\\Delta$ = {md:+.3f} [95% CI {clo:+.3f}, {chi:+.3f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#333333")

    fig.suptitle("Procrustes vs Affine: retrieval-reconstruction tradeoff across "
                 f"{len(pairs)} language pairs", fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def make_table(df, out_stem):
    learned = ["procrustes", "ridge", "mlp"]
    pivot = df[df["method"].isin(METHODS)].pivot(index="pair", columns="method")
    n_pairs = pivot.shape[0]
    rows = []
    for m in METHODS:
        sub = df[df["method"] == m]
        row = {"Method": METHOD_LABEL[m]}
        for col, label, higher in METRICS:
            vals = sub[col].values
            lo, hi = bootstrap_ci(vals)
            row[label] = float(np.mean(vals))
            row[f"{label} CI lo"], row[f"{label} CI hi"] = lo, hi
        for col, label, higher in METRICS:
            table = pivot[col][learned]
            winners = table.idxmax(axis=1) if higher else table.idxmin(axis=1)
            row[f"Best {label}"] = f"{int((winners == m).sum())}/{n_pairs}" if m in learned else "—"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_stem.with_suffix(".csv"), index=False, float_format="%.4f")

    def cell(r, label):
        return f"{r[label]:.3f} [{r[label + ' CI lo']:.3f}, {r[label + ' CI hi']:.3f}]"
    cols = ["Method"] + [lab for _, lab, _ in METRICS] + [f"Best {lab}" for _, lab, _ in METRICS]
    fmt = pd.DataFrame([{"Method": r["Method"], **{lab: cell(r, lab) for _, lab, _ in METRICS},
                         **{f"Best {lab}": r[f"Best {lab}"] for _, lab, _ in METRICS}}
                        for _, r in table.iterrows()])[cols]
    note = (f"Mean over {n_pairs} pairs at the final layer with percentile-bootstrap 95% CI "
            f"({N_BOOT:,} resamples). P@1 (std) retrieves against the test pool, P@1 (hard) "
            "against the train+test pool. Affine = ridge-regularised linear map.")
    out_stem.with_suffix(".md").write_text(fmt.to_markdown(index=False) + f"\n\n_{note}_\n")
    tex = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(c.replace("@", "@") for c in cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(str(v).replace("—", "---") for v in r) + " \\\\" for r in fmt.values.tolist()]
    tex += ["\\bottomrule", "\\end{tabular}", f"% {note}"]
    out_stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print(fmt.to_markdown(index=False))
    print(f"Wrote {out_stem.with_suffix('.csv')} / .md / .tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="all_pairs")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: figures/")
    args = ap.parse_args()
    results = args.results or (ROOT / "results" / args.run_name / "projection_results.csv")
    out_dir = args.output_dir or (ROOT / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_final_layer(results)
    print(f"Read {results}: {df.pair.nunique()} pairs, methods {sorted(df.method.unique())}")
    make_figure(df, out_dir / "figure_4_procrustes_vs_affine.pdf")
    make_table(df, out_dir / "table_method_comparison.csv")


if __name__ == "__main__":
    main()
