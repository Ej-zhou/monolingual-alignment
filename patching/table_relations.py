#!/usr/bin/env python3
"""Table 10: eng -> deu directional success rate (%) with Wilson 95% CIs across
factual relations, patch start j = 2, for within-model, Procrustes and shuffle.

Expects, per relation, results/w_deu_latn[_<relation>]/within_results.csv and
results/x_eng_latn-deu_latn[_<relation>]/cross_results.csv (no suffix for the
capital relation) as written by run_within.sh / run_cross.sh.  Rows whose donor
and target answers share the scored first subword are dropped (they cannot
change under patching), which is why n < facts^2 - facts for some relations.

Usage:
  python table_relations.py --j 2
"""
import argparse
from pathlib import Path

import pandas as pd

from patching_results import drop_degenerate, rate_ci, read_results

ROOT = Path(__file__).resolve().parent
RELATIONS = [
    ("capital", "country→capital"), ("city_to_country", "city→country"),
    ("continent", "country→continent"), ("landmark", "landmark→city"),
    ("author", "work→author"), ("composer", "work→composer"),
    ("inventor", "invention→inventor"),
]


def fmt(r):
    return "---" if r is None else f"{r[0]:.1f} [{r[1]:.1f}, {r[2]:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eng_latn")
    ap.add_argument("--tgt", default="deu_latn")
    ap.add_argument("--j", type=int, default=2, help="Span start layer")
    ap.add_argument("--out", type=Path, default=ROOT / "tables" / "relations.md")
    args = ap.parse_args()

    rows = []
    for rel, label in RELATIONS:
        suffix = "" if rel == "capital" else f"_{rel}"
        try:
            w = read_results(ROOT / "results" / f"w_{args.tgt}{suffix}", "within_results.csv")
            x = read_results(ROOT / "results" / f"x_{args.src}-{args.tgt}{suffix}", "cross_results.csv")
        except FileNotFoundError as e:
            print(f"[skip] {rel}: {e}")
            continue
        w = drop_degenerate(w[w["j_start"] == args.j])
        x = drop_degenerate(x[x["j_start"] == args.j])
        rows.append({
            "Relation": label, "n": len(w),
            "Within-lang": fmt(rate_ci(w["delta_logp_donor"])),
            "Procrustes": fmt(rate_ci(x[x["condition"] == "procrustes"]["delta_logp_donor"])),
            "Shuffled": fmt(rate_ci(x[x["condition"] == "shuffle"]["delta_logp_donor"])),
        })
    table = pd.DataFrame(rows)
    md = table.to_markdown(index=False)
    print(f"{args.src} -> {args.tgt}, j = {args.j}\n{md}")
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(md + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
