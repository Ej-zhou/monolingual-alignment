"""Shared constants and readers for the CKA result CSVs written by cka_pairwise.py.

A result file ``<results-dir>/cka_<dataset>_<modelA>_<modelB>.csv`` has columns
``model_a, model_b, dataset, setting, layer, cka_matched, cka_shuffled`` with one
row per (setting, layer).  Settings: ``mean_pool``, ``token_aligned``, ``sgpt``.
"""
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# The nine Goldfish languages, in the ordering used for every matrix in the paper.
LANGUAGES = [
    "eng_latn", "zho_hans", "spa_latn", "arb_arab", "hin_deva",
    "fra_latn", "rus_cyrl", "deu_latn", "jpn_jpan",
]
LANG_NAMES = {
    "eng_latn": "English", "zho_hans": "Chinese", "spa_latn": "Spanish",
    "arb_arab": "Arabic", "hin_deva": "Hindi", "fra_latn": "French",
    "rus_cyrl": "Russian", "deu_latn": "German", "jpn_jpan": "Japanese",
}
LANG_SHORT = {l: l[:3] for l in LANGUAGES}
BASE_MODELS = [f"goldfish-models/{l}" for l in LANGUAGES]

SETTINGS = ["mean_pool", "token_aligned", "sgpt"]
SETTING_LABEL = {"mean_pool": "Mean pooling", "token_aligned": "Token-aligned",
                 "sgpt": "SGPT"}
DATASETS = ["flores", "tatoeba", "opus", "bouquet"]
DATASET_LABEL = {"flores": "FLORES", "tatoeba": "Tatoeba", "opus": "OPUS",
                 "bouquet": "BouQUET"}
SIZES = ["5mb", "10mb", "100mb", "1000mb"]
SIZE_LABEL = {"5mb": "5 MB", "10mb": "10 MB", "100mb": "100 MB", "1000mb": "1 GB"}


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def models_at(size: str) -> List[str]:
    return [f"{m}_{size}" for m in BASE_MODELS]


def lang_of(model: str) -> str:
    """'goldfish-models/eng_latn_1000mb' -> 'eng_latn'."""
    return re.sub(r"_\d+mb$", "", model.split("/")[-1])


def find_result(results_dir: str, dataset: str, model_a: str, model_b: str) -> Optional[str]:
    for a, b in ((model_a, model_b), (model_b, model_a)):
        p = os.path.join(results_dir, f"cka_{dataset}_{safe_name(a)}_{safe_name(b)}.csv")
        if os.path.exists(p):
            return p
    return None


def load_pair(results_dir: str, model_a: str, model_b: str,
              datasets: List[str] = DATASETS) -> pd.DataFrame:
    """Concatenate the result rows of one model pair over the given datasets."""
    frames = []
    for ds in datasets:
        p = find_result(results_dir, ds, model_a, model_b)
        if p:
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def last_layer(df: pd.DataFrame, setting: str) -> Tuple[float, float]:
    """(matched, shuffled) at the last layer, averaged over the datasets in df."""
    if df.empty:
        return np.nan, np.nan
    sub = df[df["setting"] == setting]
    if sub.empty:
        return np.nan, np.nan
    avg = sub.groupby("layer")[["cka_matched", "cka_shuffled"]].mean()
    last = avg.loc[avg.index.max()]
    return float(last["cka_matched"]), float(last["cka_shuffled"])


def build_matrix(results_dir: str, models_row: List[str], models_col: List[str],
                 setting: str, datasets: List[str] = DATASETS,
                 self_value: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    """Last-layer (matched, shuffled) matrices; rows/cols are model ids.

    Identical row/column models (self-pairs) get ``self_value`` instead of a
    lookup.  Missing pairs are NaN.
    """
    nr, nc = len(models_row), len(models_col)
    mat_m = np.full((nr, nc), np.nan)
    mat_s = np.full((nr, nc), np.nan)
    for i, a in enumerate(models_row):
        for j, b in enumerate(models_col):
            if a == b:
                mat_m[i, j] = mat_s[i, j] = self_value
                continue
            mat_m[i, j], mat_s[i, j] = last_layer(load_pair(results_dir, a, b, datasets), setting)
    return mat_m, mat_s
