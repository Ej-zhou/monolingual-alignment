#!/usr/bin/env python3
"""Table 3: last-layer SGPT CKA between the five ~1B monolingual models.
Lower triangle = matched, upper triangle = shuffled.

Usage:
  python table_diverse_models.py --results-dir results
"""
import argparse
import os

import numpy as np
import pandas as pd

MODELS = [
    ("EleutherAI/pythia-1.4b", "Pyth. EN"),
    ("SJTU-CL/Zh-Pythia-1.4B", "Pyth. ZH"),
    ("TucanoBR/Tucano-1b1", "Tuc. PT"),
    ("speakleash/Bielik-1.5B-v3", "Biel. PL"),
    ("sapienzanlp/Minerva-1B-base-v1.0", "Min. IT"),
]


def safe(s):
    return s.replace("/", "_").replace(":", "_")


def load(results_dir, a, b, layer):
    for x, y in ((a, b), (b, a)):
        p = os.path.join(results_dir, f"cka_diverse_{safe(x)}_{safe(y)}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            row = df[(df["setting"] == "sgpt") & (df["layer"] == layer)]
            if not row.empty:
                return float(row["cka_matched"].iloc[0]), float(row["cka_shuffled"].iloc[0])
    return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--layer", default="last", choices=["first", "last"])
    ap.add_argument("--out-dir", default="tables")
    args = ap.parse_args()

    n = len(MODELS)
    labels = [m[1] for m in MODELS]
    table = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m, s = load(args.results_dir, MODELS[i][0], MODELS[j][0], args.layer)
            table[j, i] = m   # lower triangle: matched
            table[i, j] = s   # upper triangle: shuffled
    df = pd.DataFrame(table, index=labels, columns=labels)
    print(f"{args.layer} layer SGPT CKA (lower = matched, upper = shuffled):")
    print(df.round(2).to_string(na_rep="---"))
    lower = table[np.tril_indices(n, k=-1)]
    upper = table[np.triu_indices(n, k=1)]
    print(f"\nmean matched {np.nanmean(lower):.2f}   mean shuffled {np.nanmean(upper):.2f}")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"diverse_models_{args.layer}.csv")
    df.round(3).to_csv(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
