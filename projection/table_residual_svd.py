#!/usr/bin/env python3
"""Table 9: spectra of the target Y and the post-Procrustes residual R at one
layer (default 8) for the four headline pairs, from residual_svd.py output.

Usage:
  python table_residual_svd.py --run-name residual_svd --layer 8
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PAIRS = ["eng_latn-fra_latn", "eng_latn-rus_cyrl", "eng_latn-zho_hans", "eng_latn-arb_arab"]
COLUMNS = {
    "target_eff_rank": "eff_rank Y", "residual_eff_rank": "eff_rank R",
    "residual_k50": "R k50", "residual_k90": "R k90",
    "residual_frac_outside_Ytop10": "frac_out k=10",
    "residual_frac_outside_Ytop50": "frac_out k=50",
    "residual_to_target_frob_ratio": "||R||/||Y||",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="residual_svd")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tables")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "results" / args.run_name / "residual_summary.csv")
    sub = df[df["layer"] == args.layer].set_index("pair").reindex(PAIRS)
    table = sub[list(COLUMNS)].rename(columns=COLUMNS)
    table.index = [p.replace("_latn", "").replace("_cyrl", "").replace("_hans", "")
                    .replace("_arab", "").replace("-", "–") for p in table.index]
    print(f"Layer {args.layer} (d = {int(sub['dim'].iloc[0])})")
    print(table.round(2).to_string())
    args.out_dir.mkdir(exist_ok=True)
    out = args.out_dir / f"residual_svd_layer{args.layer}.csv"
    table.round(3).to_csv(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
