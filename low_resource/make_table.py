#!/usr/bin/env python3
"""Table 2: matched (M) / shuffled (S) SGPT CKA and their gap at the last and
middle (L/2) layer for English paired with four low-resource languages.

Usage:
  python make_table.py --results-dir results
"""
import argparse
import os

import numpy as np
import pandas as pd

ROWS = [  # (display pair, goldfish code a, goldfish code b, size, family/script)
    ("eng--tgl", "eng_latn", "tgl_latn", "1000mb", "Austronesian (Latn)"),
    ("eng--swh", "eng_latn", "swa_latn", "1000mb", "Atlantic--Congo (Latn)"),
    ("eng--uzn", "eng_latn", "uzb_latn", "1000mb", "Turkic (Latn)"),
    ("eng--amh", "eng_latn", "amh_ethi", "1000mb", "Afro-Asiatic (Ethi)"),
    ("eng--fra", "eng_latn", "fra_latn", "1000mb", "Indo-European (Latn)"),
    ("eng--zho", "eng_latn", "zho_hans", "1000mb", "Sino-Tibetan (Hans)"),
]


def load(results_dir, a, b, size):
    for x, y in ((a, b), (b, a)):
        p = os.path.join(results_dir,
                         f"cka_flores_goldfish-models_{x}_{size}_goldfish-models_{y}_{size}.csv")
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def cell(df, layer):
    r = df[(df["setting"] == "sgpt") & (df["layer"] == layer)].iloc[0]
    m, s = float(r["cka_matched"]), float(r["cka_shuffled"])
    return f"{m:.2f} / {s:.2f} (+{m - s:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="tables/lowres_cka.md")
    args = ap.parse_args()

    rows = []
    for label, a, b, size, family in ROWS:
        df = load(args.results_dir, a, b, size)
        if df is None:
            rows.append([label, family, "---", "---"]); continue
        last = int(df["layer"].max())
        rows.append([label, family, cell(df, last), cell(df, last // 2)])
    table = pd.DataFrame(rows, columns=["Pair", "Family (script)",
                                        "Last layer M / S (Δ)", "Middle layer M / S (Δ)"])
    md = table.to_markdown(index=False)
    print(md)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
