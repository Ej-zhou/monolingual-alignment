#!/usr/bin/env python3
"""Tables from the 1000 MB pairwise CKA results.

  Table 1        mean last-layer matched / shuffled CKA over all language pairs,
                 per representation x dataset      -> tables/table1_summary.{csv,md}
  Tables 11-22   9x9 last-layer matched/shuffled matrices, one LaTeX table per
                 representation x dataset          -> tables/table_<setting>_<dataset>.tex
                 (+ tables/appendix_a.tex wrapper and CSV copies)

Usage:
  python make_tables.py --results-dir results/main --out-dir tables
"""
import argparse
import os

import numpy as np
import pandas as pd

from cka_results import (DATASETS, DATASET_LABEL, LANG_SHORT, LANGUAGES, SETTINGS,
                         build_matrix, lang_of, models_at)

SETTING_TITLE = {"mean_pool": "Sentence-Level Mean Pooling",
                 "token_aligned": "Token-Level Word-Aligned",
                 "sgpt": "SGPT Position-Weighted Pooling"}
SETTING_SHORT = {"mean_pool": "Mean Pooling", "token_aligned": "Token-Aligned", "sgpt": "SGPT"}


def upper_triangle_mean(mat):
    n = mat.shape[0]
    vals = mat[np.triu_indices(n, k=1)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan"), int(vals.size)


def latex_matrix(mat_m, mat_s, setting, dataset):
    labels = [LANG_SHORT[l] for l in LANGUAGES]
    n = len(labels)
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Last-layer pairwise CKA (\textit{matched\,/\,shuffled}) "
         f"--- {SETTING_TITLE[setting]}, {DATASET_LABEL[dataset]} dataset. "
         r"All values computed on 1000\,MB Goldfish models.}",
         r"\label{tab:cka_" + f"{setting}_{dataset}" + "}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * n + "}", r"\toprule",
         " & ".join([""] + [f"\\textbf{{{l}}}" for l in labels]) + r" \\", r"\midrule"]
    for i in range(n):
        cells = [f"\\textbf{{{labels[i]}}}"]
        for j in range(n):
            if i == j or np.isnan(mat_m[i, j]):
                cells.append("---")
            else:
                cells.append(f"{mat_m[i, j]:.2f}\\,/\\,{mat_s[i, j]:.2f}")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/main")
    ap.add_argument("--out-dir", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = models_at("1000mb")
    labels = [LANG_SHORT[lang_of(m)] for m in models]

    summary = []
    wrapper = [r"\section{Full Pairwise CKA Results}", r"\label{sec:appendix_cka}", ""]
    for setting in SETTINGS:
        wrapper += [r"\subsection{" + SETTING_TITLE[setting] + "}", ""]
        for dataset in DATASETS:
            mat_m, mat_s = build_matrix(args.results_dir, models, models, setting, [dataset])
            m_mean, n_pairs = upper_triangle_mean(mat_m)
            s_mean, _ = upper_triangle_mean(mat_s)
            summary.append({"setting": SETTING_SHORT[setting], "dataset": DATASET_LABEL[dataset],
                            "n_pairs": n_pairs, "matched": m_mean, "shuffled": s_mean,
                            "delta": m_mean - s_mean})
            tex_name = f"table_{setting}_{dataset}.tex"
            with open(os.path.join(args.out_dir, tex_name), "w") as f:
                f.write(latex_matrix(mat_m, mat_s, setting, dataset))
            wrapper += [r"\input{appendix_tables/" + tex_name + "}", ""]
            csv = pd.DataFrame([[labels[i]] + [
                "---" if i == j or np.isnan(mat_m[i, j]) else f"{mat_m[i, j]:.2f} / {mat_s[i, j]:.2f}"
                for j in range(len(labels))] for i in range(len(labels))],
                columns=[""] + labels)
            csv.to_csv(os.path.join(args.out_dir, f"matrix_{setting}_{dataset}.csv"), index=False)
    with open(os.path.join(args.out_dir, "appendix_a.tex"), "w") as f:
        f.write("\n".join(wrapper))

    df = pd.DataFrame(summary)
    avg = (df.groupby("setting", sort=False)[["matched", "shuffled", "delta"]].mean()
             .reset_index().assign(dataset="Avg.", n_pairs=""))
    df = pd.concat([df, avg], ignore_index=True)
    df.to_csv(os.path.join(args.out_dir, "table1_summary.csv"), index=False, float_format="%.3f")
    wide = df.pivot(index="dataset", columns="setting", values=["matched", "shuffled", "delta"])
    wide = wide.reindex([DATASET_LABEL[d] for d in DATASETS] + ["Avg."])
    md = wide.round(2).to_markdown()
    with open(os.path.join(args.out_dir, "table1_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nWrote tables to {args.out_dir}/")


if __name__ == "__main__":
    main()
