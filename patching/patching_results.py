"""Shared readers/statistics for the span-patching result CSVs."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def read_results(run_dir: Path, name: str) -> pd.DataFrame:
    """Read results/<run>/<name> (plain or gzipped; the shipped results are .gz)."""
    for cand in (run_dir / name, run_dir / f"{name}.gz"):
        if cand.exists():
            return pd.read_csv(cand)
    raise FileNotFoundError(f"{run_dir / name}[.gz] not found")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score 95% interval for a binomial proportion, as (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_ci(delta):
    """Directional success rate (% of rows with delta > 0) and Wilson CI, in %."""
    n = len(delta)
    if n == 0:
        return None
    k = int((delta > 0).sum())
    lo, hi = wilson(k, n)
    return 100 * k / n, 100 * lo, 100 * hi


def drop_degenerate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop (donor, target) rows whose answers share the scored first subword:
    their donor/target log-probabilities are identical by construction."""
    deg = ((df["logp_donor_base"] == df["logp_target_base"])
           & (df["logp_donor_patch"] == df["logp_target_patch"]))
    return df[~deg]
