#!/usr/bin/env python3
"""Download parallel sentences and write one CSV per language pair.

Every experiment in this repository reads parallel data from

    <out-dir>/<dataset>/<lang_a>-<lang_b>.csv

where the CSV has two columns named by the Goldfish language codes
(e.g. ``eng_latn``, ``fra_latn``).  Datasets:

  flores    FLORES-200 dev/devtest via ``openlanguagedata/flores_plus`` (open),
            falling back to the gated ``facebook/flores``.  Multi-parallel, so
            every pair of --languages is written.
  opus      OPUS-100 (``Helsinki-NLP/opus-100``).  Only pairs that exist as an
            OPUS-100 config (mostly English-centric).
  tatoeba   Tatoeba Challenge (``Helsinki-NLP/tatoeba_mt``).
  bouquet   BouQUET (``facebook/bouquet``, sentence level).  Gated: run
            ``huggingface-cli login`` and accept the dataset terms first.

The exact invocations used for the paper are listed in README.md.
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

# ── Languages ────────────────────────────────────────────────────────────────

# The nine Goldfish languages of the main analysis, in the paper's ordering.
LANGUAGES = [
    "eng_latn", "zho_hans", "spa_latn", "arb_arab", "hin_deva",
    "fra_latn", "rus_cyrl", "deu_latn", "jpn_jpan",
]

# Goldfish code -> FLORES+ config name.  Covers the nine main languages, the
# languages of the ~1B diverse-model comparison, and the low-resource set.
FLORES_PLUS_CFG = {
    "eng_latn": "eng_Latn", "zho_hans": "cmn_Hans", "spa_latn": "spa_Latn",
    "arb_arab": "arb_Arab", "hin_deva": "hin_Deva", "fra_latn": "fra_Latn",
    "rus_cyrl": "rus_Cyrl", "deu_latn": "deu_Latn", "jpn_jpan": "jpn_Jpan",
    "por_latn": "por_Latn", "pol_latn": "pol_Latn", "ita_latn": "ita_Latn",
    "swa_latn": "swh_Latn", "tgl_latn": "fil_Latn", "uzb_latn": "uzn_Latn",
    "amh_ethi": "amh_Ethi", "yor_latn": "yor_Latn",
}

# ISO 639-1 codes used by OPUS-100.
ISO2_MAP = {
    "eng_latn": "en", "zho_hans": "zh", "spa_latn": "es", "arb_arab": "ar",
    "hin_deva": "hi", "fra_latn": "fr", "rus_cyrl": "ru", "deu_latn": "de",
    "jpn_jpan": "ja", "por_latn": "pt", "pol_latn": "pl", "ita_latn": "it",
}

# tatoeba_mt uses ISO 639-3 (script-qualified for Mandarin); primary + fallbacks.
TATOEBA_MT_CODES = {
    "eng_latn": ["eng"], "zho_hans": ["cmn_Hans", "cmn", "zho"],
    "spa_latn": ["spa"], "arb_arab": ["arb", "ara"], "hin_deva": ["hin"],
    "fra_latn": ["fra"], "rus_cyrl": ["rus"], "deu_latn": ["deu"],
    "jpn_jpan": ["jpn"],
}

# NLLB-style codes used by facebook/bouquet (case-sensitive).
BOUQUET_LANG_MAP = {
    "eng_latn": "eng_Latn", "zho_hans": "cmn_Hans", "spa_latn": "spa_Latn",
    "arb_arab": "arz_Arab", "hin_deva": "hin_Deva", "fra_latn": "fra_Latn",
    "rus_cyrl": "rus_Cyrl", "deu_latn": "deu_Latn", "jpn_jpan": "jpn_Jpan",
}


# ── Shared helpers ───────────────────────────────────────────────────────────

def _save_pair_csv(lang_a: str, lang_b: str, sents_a: List[str], sents_b: List[str],
                   out_dir: Path, max_n: int) -> int:
    n = min(len(sents_a), len(sents_b))
    good = [(a, b) for a, b in zip(sents_a[:n], sents_b[:n])
            if a and a.strip() and b and b.strip()][:max_n]
    if not good:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{lang_a}-{lang_b}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([lang_a, lang_b])
        w.writerows(good)
    return len(good)


# ── FLORES ───────────────────────────────────────────────────────────────────

def _flores_plus(langs: List[str], splits: List[str]) -> Dict[str, Dict[str, str]]:
    """{lang: {'<split>:<id>': text}} from openlanguagedata/flores_plus."""
    text: Dict[str, Dict[str, str]] = {}
    for lang in langs:
        cfg = FLORES_PLUS_CFG[lang]
        rows = {}
        for split in splits:
            ds = load_dataset("openlanguagedata/flores_plus", cfg, split=split)
            for r in ds:
                rows[f"{split}:{r['id']}"] = r["text"]
        text[lang] = rows
        print(f"  {lang} ({cfg}): {len(rows)} sentences")
    return text


def _flores_gated(langs: List[str], splits: List[str]) -> Dict[str, Dict[str, str]]:
    """Fallback: facebook/flores 'all' config (sentence_<cfg> columns)."""
    ds = load_dataset("facebook/flores", "all")
    text: Dict[str, Dict[str, str]] = {}
    for lang in langs:
        col = f"sentence_{FLORES_PLUS_CFG[lang]}"
        rows = {}
        for split in splits:
            for i, s in enumerate(ds[split][col]):
                rows[f"{split}:{i}"] = s
        text[lang] = rows
        print(f"  {lang} ({col}): {len(rows)} sentences")
    return text


def download_flores(out_dir: Path, langs: List[str], max_n: int, splits: List[str]) -> None:
    print(f"=== FLORES ({'+'.join(splits)}) ===")
    try:
        text = _flores_plus(langs, splits)
    except Exception as e:
        print(f"  flores_plus failed ({type(e).__name__}: {e}); trying facebook/flores")
        text = _flores_gated(langs, splits)

    def key(k):  # keep dev before devtest, ids numeric
        sp, i = k.split(":")
        return (splits.index(sp), int(i))

    pair_dir = out_dir / "flores"
    for a, b in combinations(langs, 2):
        ids = sorted(set(text[a]) & set(text[b]), key=key)
        n = _save_pair_csv(a, b, [text[a][i] for i in ids], [text[b][i] for i in ids],
                           pair_dir, max_n)
        print(f"  {a}-{b}: {n} rows")


# ── OPUS-100 ─────────────────────────────────────────────────────────────────

def download_opus(out_dir: Path, langs: List[str], max_n: int, split_first: str) -> None:
    print(f"=== OPUS-100 (split priority: {split_first}) ===")
    order = [split_first] + [s for s in ("train", "validation", "test") if s != split_first]
    pair_dir = out_dir / "opus"
    for a, b in combinations(langs, 2):
        if a not in ISO2_MAP or b not in ISO2_MAP:
            continue
        ia, ib = ISO2_MAP[a], ISO2_MAP[b]
        config = f"{min(ia, ib)}-{max(ia, ib)}"
        for split in order:
            try:
                ds = load_dataset("Helsinki-NLP/opus-100", config, split=split)
            except Exception:
                continue
            sa, sb = [], []
            for i, row in enumerate(ds):
                if i >= max_n:
                    break
                sa.append(row["translation"][ia])
                sb.append(row["translation"][ib])
            n = _save_pair_csv(a, b, sa, sb, pair_dir, max_n)
            print(f"  {a}-{b}: {n} rows (opus-100 {config}, {split})")
            break
        else:
            print(f"  {a}-{b}: not available in opus-100")


# ── Tatoeba (tatoeba_mt) ─────────────────────────────────────────────────────

def _tatoeba_try_load(code_a: str, code_b: str):
    for src, tgt in ((code_a, code_b), (code_b, code_a)):
        try:
            return load_dataset("Helsinki-NLP/tatoeba_mt", f"{src}-{tgt}",
                                trust_remote_code=True)
        except Exception:
            continue
    return None


def download_tatoeba(out_dir: Path, langs: List[str], max_n: int) -> None:
    print("=== Tatoeba (tatoeba_mt) ===")
    pair_dir = out_dir / "tatoeba"
    for a, b in combinations(langs, 2):
        if a not in TATOEBA_MT_CODES or b not in TATOEBA_MT_CODES:
            continue
        ds_dict = None
        for ca in TATOEBA_MT_CODES[a]:
            for cb in TATOEBA_MT_CODES[b]:
                ds_dict = _tatoeba_try_load(ca, cb)
                if ds_dict:
                    break
            if ds_dict:
                break
        if ds_dict is None:
            print(f"  {a}-{b}: not available in tatoeba_mt")
            continue
        codes_a, codes_b = set(TATOEBA_MT_CODES[a]), set(TATOEBA_MT_CODES[b])
        sa, sb, seen = [], [], set()
        for split in ds_dict.values():
            for row in split:
                sl, tl = row["sourceLang"], row["targetlang"]
                if sl in codes_a and tl in codes_b:
                    x, y = row["sourceString"], row["targetString"]
                elif sl in codes_b and tl in codes_a:
                    x, y = row["targetString"], row["sourceString"]
                else:
                    continue
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                sa.append(x); sb.append(y)
                if len(sa) >= max_n:
                    break
            if len(sa) >= max_n:
                break
        n = _save_pair_csv(a, b, sa, sb, pair_dir, max_n)
        print(f"  {a}-{b}: {n} rows")


# ── BouQUET ──────────────────────────────────────────────────────────────────

def download_bouquet(out_dir: Path, langs: List[str], max_n: int) -> None:
    print("=== BouQUET ===")
    rows = []
    for split in ("dev", "test"):
        try:
            ds = load_dataset("facebook/bouquet", "sentence_level", split=split)
            rows.extend(ds)
            print(f"  loaded sentence_level/{split} ({len(ds)} rows)")
        except Exception as e:
            print(f"  could not load split {split}: {e}")
    if not rows:
        raise RuntimeError("Could not load facebook/bouquet (gated: accept the terms "
                           "on HuggingFace and run `huggingface-cli login`).")
    reverse = {v: k for k, v in BOUQUET_LANG_MAP.items()}
    texts: Dict[str, Dict[str, str]] = {l: {} for l in langs}
    for row in rows:
        uid = row["uniq_id"]
        if row.get("tgt_lang") == "eng_Latn" and "eng_latn" in texts:
            texts["eng_latn"][uid] = row["tgt_text"]
        src = reverse.get(row["src_lang"])
        if src in texts:
            texts[src][uid] = row["src_text"]
    pair_dir = out_dir / "bouquet"
    for a, b in combinations(langs, 2):
        shared = sorted(set(texts[a]) & set(texts[b]))
        if not shared:
            continue
        n = _save_pair_csv(a, b, [texts[a][u] for u in shared], [texts[b][u] for u in shared],
                           pair_dir, max_n)
        print(f"  {a}-{b}: {n} rows")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["flores", "opus", "tatoeba", "bouquet"],
                    choices=["flores", "opus", "tatoeba", "bouquet"])
    ap.add_argument("--languages", nargs="+", default=LANGUAGES,
                    help=f"Goldfish language codes (known: {sorted(FLORES_PLUS_CFG)})")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-instances", type=int, default=1000,
                    help="Cap on parallel sentences per (dataset, pair)")
    ap.add_argument("--flores-splits", nargs="+", default=["devtest"],
                    choices=["dev", "devtest"])
    ap.add_argument("--opus-split", default="train", choices=["train", "validation", "test"],
                    help="OPUS-100 split to try first")
    args = ap.parse_args()

    unknown = [l for l in args.languages if l not in FLORES_PLUS_CFG]
    if unknown:
        raise SystemExit(f"Unknown language codes: {unknown}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        try:
            if name == "flores":
                download_flores(args.out_dir, args.languages, args.max_instances, args.flores_splits)
            elif name == "opus":
                download_opus(args.out_dir, args.languages, args.max_instances, args.opus_split)
            elif name == "tatoeba":
                download_tatoeba(args.out_dir, args.languages, args.max_instances)
            elif name == "bouquet":
                download_bouquet(args.out_dir, args.languages, args.max_instances)
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}\n")

    print("Done. Pair files:")
    for name in args.datasets:
        d = args.out_dir / name
        print(f"  {d}/  [{len(list(d.glob('*.csv'))) if d.is_dir() else 0} pairs]")


if __name__ == "__main__":
    main()
