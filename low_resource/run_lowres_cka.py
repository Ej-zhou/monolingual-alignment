#!/usr/bin/env python3
"""Low-resource CKA (Section 3.2, Table 2): English paired with Tagalog,
Swahili, Northern Uzbek and Amharic, plus the eng-fra / eng-zho reference pairs.

Runs ../cka/cka_pairwise.py on FLORES for each pair at the given Goldfish size.
Language codes are Goldfish's (swa_latn = Swahili, tgl_latn = Tagalog,
uzb_latn = Northern Uzbek, amh_ethi = Amharic).  Parallel data comes from

  python ../data/download_parallel.py --out-dir ../data/lowres --datasets flores \
      --flores-splits dev devtest --languages eng_latn fra_latn zho_hans \
      swa_latn tgl_latn uzb_latn amh_ethi

Usage:
  python run_lowres_cka.py --device cuda
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKA = HERE.parent / "cka" / "cka_pairwise.py"

PAIRS = [
    ("eng_latn", "tgl_latn"), ("eng_latn", "swa_latn"),
    ("eng_latn", "uzb_latn"), ("eng_latn", "amh_ethi"),
    ("eng_latn", "fra_latn"), ("eng_latn", "zho_hans"),   # high-resource references
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1000mb")
    ap.add_argument("--data-dir", default=str(HERE.parent / "data" / "lowres"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for a, b in PAIRS:
        cmd = [sys.executable, str(CKA),
               "--model-a", f"goldfish-models/{a}_{args.size}",
               "--model-b", f"goldfish-models/{b}_{args.size}",
               "--dataset", "flores", "--data-dir", args.data_dir,
               "--results-dir", args.results_dir,
               "--num-samples", str(args.num_samples), "--aligner", "fallback",
               "--device", args.device]
        print(f"### {a} x {b} @ {args.size} ###")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
