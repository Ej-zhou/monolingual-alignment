#!/usr/bin/env python3
"""Dimensional analysis of the Procrustes residual (Appendix F.2, Table 9).

For each (pair, layer): fit Procrustes on the train split (same data pipeline
and embedding cache as fit_projections.py), take the held-out residual
R = Y_test - X_test W, and compare its singular-value spectrum with that of the
centred target Y_test:

  eff_rank        participation ratio (sum s^2)^2 / sum s^4 of the spectrum
  k_50/k_90/...   smallest k whose top singular values carry that energy fraction
  frac_out_k      fraction of residual energy outside the top-k right-singular
                  subspace of Y (residual_frac_outside_Ytop10 / Ytop50)

Outputs under results/<run-name>/: residual_summary.csv (one row per pair and
layer), residual_spectra.npz (full spectra + top-k residual directions),
args.json.  Summarise with table_residual_svd.py.

Usage:
  python residual_svd.py --datasets flores opus tatoeba bouquet \
      --quotas-file quotas.json --run-name residual_svd --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reuse the fit_projections helpers so the embedding pipeline (and cache) is identical.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fit_projections import (  # noqa: E402
    GOLDFISH_TEMPLATE,
    extract_layer_embeddings,
    fit_procrustes,
    load_pair_csv,
)

DEFAULT_DATA_DIR = ROOT.parent / "data" / "projection"
DEFAULT_CACHE_DIR = ROOT / "cache"       # shared with fit_projections.py
DEFAULT_RESULTS_DIR = ROOT / "results"

DEFAULT_PAIRS = [
    "eng_latn-fra_latn",
    "eng_latn-zho_hans",
    "eng_latn-rus_cyrl",
    "eng_latn-arb_arab",
]


# ── Residual SVD analysis ───────────────────────────────────────────────────

def _spectrum_stats(sigma: np.ndarray, thresholds=(0.5, 0.9, 0.95, 0.99)) -> Dict:
    """Given singular values sigma (descending), return summary stats.

    Energy is sigma^2 (Frobenius-norm contribution). k_p = smallest number of
    components whose cumulative energy >= p * total. Participation ratio
    (effective rank) = (sum sigma^2)^2 / sum sigma^4.
    """
    s2 = sigma.astype(np.float64) ** 2
    total = s2.sum()
    out: Dict[str, float] = {"frob2": float(total)}
    if total <= 0 or not np.isfinite(total):
        for p in thresholds:
            out[f"k_{int(p*100)}"] = 0
        out["eff_rank"] = 0.0
        return out
    cum = np.cumsum(s2) / total
    for p in thresholds:
        # smallest k such that cum[k-1] >= p (1-indexed count)
        k = int(np.searchsorted(cum, p) + 1)
        out[f"k_{int(p*100)}"] = k
    out["eff_rank"] = float(total ** 2 / (s2 ** 2).sum())
    return out


def residual_svd(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    top_k_vectors: int,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Fit Procrustes on train; analyze the test residual via SVD.

    Returns (scalar_summary, arrays) where arrays contains the residual
    singular values, cumulative-energy curve, the same for Y_test and the
    rotated source X_test @ W (centered), and the top-k right singular
    vectors of the residual.
    """
    W = fit_procrustes(X_train, Y_train)
    Yhat = X_test @ W                       # rotated source
    R = Y_test - Yhat                       # held-out residual (n, d)
    R_c = R - R.mean(0, keepdims=True)      # center so energy = variance-like

    # Use SVD of the centered residual.
    # full_matrices=False yields min(n, d) singular values.
    _, sR, VtR = np.linalg.svd(R_c, full_matrices=False)
    statsR = _spectrum_stats(sR)

    # Reference spectra: centered target and rotated-source on the same test.
    Y_c = Y_test - Y_test.mean(0, keepdims=True)
    Yhat_c = Yhat - Yhat.mean(0, keepdims=True)
    sY = np.linalg.svd(Y_c,    compute_uv=False)
    sX = np.linalg.svd(Yhat_c, compute_uv=False)
    statsY = _spectrum_stats(sY)
    statsX = _spectrum_stats(sX)

    # Per-dimension residual fraction along target's own principal axes:
    # how much of the residual sits along Y's own top directions vs. orthogonal
    # to them? Project R_c onto Y's right singular vectors.
    _, _, VtY = np.linalg.svd(Y_c, full_matrices=False)
    R_in_Ybasis = R_c @ VtY.T                       # (n, d)
    energy_per_Yaxis = (R_in_Ybasis ** 2).sum(0)    # length d, descending Y axes

    summary = {
        "residual_frob2":   statsR["frob2"],
        "target_frob2":     statsY["frob2"],
        "residual_eff_rank":  statsR["eff_rank"],
        "target_eff_rank":    statsY["eff_rank"],
        "rotated_eff_rank":   statsX["eff_rank"],
        "residual_k50":  statsR["k_50"],
        "residual_k90":  statsR["k_90"],
        "residual_k95":  statsR["k_95"],
        "residual_k99":  statsR["k_99"],
        "target_k90":    statsY["k_90"],
        "target_k95":    statsY["k_95"],
        # fraction of residual energy concentrated outside target's top-k axes
        "residual_frac_outside_Ytop10":
            float(energy_per_Yaxis[10:].sum() / max(energy_per_Yaxis.sum(), 1e-12)),
        "residual_frac_outside_Ytop50":
            float(energy_per_Yaxis[50:].sum() / max(energy_per_Yaxis.sum(), 1e-12)),
        "residual_to_target_frob_ratio":
            float(np.sqrt(statsR["frob2"] / max(statsY["frob2"], 1e-30))),
        "n_test": int(R.shape[0]),
        "dim":    int(R.shape[1]),
    }

    k = min(top_k_vectors, VtR.shape[0])
    arrays = {
        "sigma_residual": sR.astype(np.float32),
        "sigma_target":   sY.astype(np.float32),
        "sigma_rotated":  sX.astype(np.float32),
        "Vr_top":         VtR[:k].astype(np.float32),       # (k, d)
        "energy_per_Yaxis": energy_per_Yaxis.astype(np.float32),
    }
    return summary, arrays


# ── Data composition (mirrors fit_projections.py) ──────────────────────────

def compose_pair(
    args, pair: str, rng: np.random.Generator,
) -> Tuple[List[str], List[str], List[Tuple[str, int, int]], int]:
    la, lb = pair.split("-")
    sents_a: List[str] = []
    sents_b: List[str] = []
    composition: List[Tuple[str, int, int]] = []
    seen: set = set()

    quotas = args._quotas
    for ds in args.datasets:
        try:
            a, b = load_pair_csv(args.data_dir, ds, la, lb)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")
            continue
        n_raw = len(a)
        pairs = [(x, y) for x, y in zip(a, b) if (x, y) not in seen]
        if ds not in args.always_include and args.min_len_chars > 0:
            pairs = [(x, y) for x, y in pairs
                     if len(x) >= args.min_len_chars and len(y) >= args.min_len_chars]
        quota = None
        if quotas:
            qd = quotas.get(ds, {})
            quota = qd.get(pair, qd.get("_default"))
        if quota is not None and quota < len(pairs):
            idx = rng.choice(len(pairs), size=quota, replace=False)
            pairs = [pairs[i] for i in idx]
        for x, y in pairs:
            seen.add((x, y))
            sents_a.append(x)
            sents_b.append(y)
        composition.append((ds, n_raw, len(pairs)))

    if not quotas and args.num_samples > 0 and args.num_samples < len(sents_a):
        clean_ds = set(args.always_include)
        clean_count = sum(kept for ds, _, kept in composition if ds in clean_ds)
        n_target = args.num_samples
        if n_target <= clean_count:
            idx = rng.choice(clean_count, size=n_target, replace=False)
            sents_a = [sents_a[i] for i in idx]
            sents_b = [sents_b[i] for i in idx]
        else:
            noisy_total = len(sents_a) - clean_count
            need = min(n_target - clean_count, noisy_total)
            noisy_idx = clean_count + rng.choice(noisy_total, size=need, replace=False)
            keep_idx = list(range(clean_count)) + sorted(noisy_idx.tolist())
            sents_a = [sents_a[i] for i in keep_idx]
            sents_b = [sents_b[i] for i in keep_idx]

    n = len(sents_a)
    if n == 0:
        raise RuntimeError(f"No sentences left for {pair}.")
    return sents_a, sents_b, composition, n


def build_cache_paths(args, pair: str, n: int) -> Tuple[Path, Path]:
    """Reproduce fit_projections.py's cache-key string so its .npy caches are reused."""
    la, lb = pair.split("-")
    ds_tag = "+".join(sorted(args.datasets))
    filt_tag = f"_minlen{args.min_len_chars}" if args.min_len_chars > 0 else ""
    clean_tag = ("_clean-" + "+".join(sorted(args.always_include))
                 if args.always_include else "")
    if args._quotas:
        import hashlib
        spec = "|".join(f"{ds}:{args._quotas.get(ds, {}).get(pair, '')}"
                        for ds in sorted(args.datasets))
        quota_tag = "_q" + hashlib.md5(spec.encode()).hexdigest()[:8]
    else:
        quota_tag = ""
    base = f"{ds_tag}_{{lang}}_{args.size}_n{n}{filt_tag}{clean_tag}{quota_tag}_seed{args.seed}.npy"
    return (args.cache_dir / base.format(lang=la),
            args.cache_dir / base.format(lang=lb))


# ── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Dimensional analysis of the Procrustes residual")
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--datasets", nargs="+", default=["flores", "opus", "tatoeba", "bouquet"])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--size", default="1000mb")
    p.add_argument("--num-samples", type=int, default=0,
                   help="0 = use all available pairs (after quotas/filters).")
    p.add_argument("--min-len-chars", type=int, default=50)
    p.add_argument("--always-include", nargs="+", default=["flores", "bouquet"])
    p.add_argument("--quotas-file", type=Path, default=ROOT / "quotas.json")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--top-k-vectors", type=int, default=32,
                   help="How many leading right singular vectors of the "
                        "residual to save per (pair, layer).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--run-name", default="residual_svd")
    args = p.parse_args()
    args._quotas: Dict[str, Dict[str, int]] = {}
    if args.quotas_file is not None and Path(args.quotas_file).exists():
        with open(args.quotas_file) as f:
            args._quotas = json.load(f)
        print(f"Loaded quotas from {args.quotas_file}")

    out_dir = args.results_dir / args.run_name
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {args.run_name}\nWriting results to: {out_dir}\n")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_rows: List[Dict] = []
    spectra_store: Dict[str, np.ndarray] = {}

    for pair in args.pairs:
        la, lb = pair.split("-")
        print(f"\n{'='*64}\nPair: {la} -> {lb}\n{'='*64}")
        rng = np.random.default_rng(args.seed)

        sents_a, sents_b, composition, n = compose_pair(args, pair, rng)
        print("  Composition:")
        for ds, n_raw, kept in composition:
            print(f"    {ds:<8} raw={n_raw:>6} kept={kept:>6}")
        print(f"  -> total per-pair: {n}")

        cache_a, cache_b = build_cache_paths(args, pair, n)
        if args.no_cache:
            for cp in (cache_a, cache_b):
                if cp.exists():
                    cp.unlink()

        model_a = GOLDFISH_TEMPLATE.format(lang=la, size=args.size)
        model_b = GOLDFISH_TEMPLATE.format(lang=lb, size=args.size)
        emb_a = extract_layer_embeddings(model_a, sents_a, cache_a,
                                         args.device, args.batch_size, args.max_length)
        emb_b = extract_layer_embeddings(model_b, sents_b, cache_b,
                                         args.device, args.batch_size, args.max_length)

        n_layers = min(emb_a.shape[0], emb_b.shape[0])
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(args.train_frac * n)
        train_idx, test_idx = idx[:cut], idx[cut:]
        print(f"Train/test: {len(train_idx)} / {len(test_idx)}; "
              f"layers={n_layers}, dim={emb_a.shape[2]}")

        for li in tqdm(range(n_layers), desc="layers"):
            t0 = time.time()
            X_tr, Y_tr = emb_a[li, train_idx], emb_b[li, train_idx]
            X_te, Y_te = emb_a[li, test_idx],  emb_b[li, test_idx]
            summary, arrays = residual_svd(
                X_tr, Y_tr, X_te, Y_te, top_k_vectors=args.top_k_vectors,
            )
            summary.update({
                "pair": pair, "src": la, "tgt": lb, "layer": li,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "size": args.size,
                "elapsed_s": round(time.time() - t0, 3),
            })
            all_rows.append(summary)
            key = f"{pair}__layer{li}"
            for name, arr in arrays.items():
                spectra_store[f"{key}__{name}"] = arr

        # Per-pair partial dump.
        pd.DataFrame(all_rows).to_csv(out_dir / "residual_summary.csv", index=False)
        np.savez_compressed(out_dir / "residual_spectra.npz", **spectra_store)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "residual_summary.csv", index=False)
    np.savez_compressed(out_dir / "residual_spectra.npz", **spectra_store)
    with open(out_dir / "args.json", "w") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if not k.startswith("_")},
                  f, indent=2)
    print(f"\nWrote {out_dir / 'residual_summary.csv'}")
    print(f"Wrote {out_dir / 'residual_spectra.npz'}")

    # Console summary: best-layer (lowest k_90) per pair.
    if not df.empty:
        best = df.loc[df.groupby("pair")["residual_k90"].idxmin(),
                      ["pair", "layer", "dim", "residual_k90", "residual_k95",
                       "residual_eff_rank", "residual_to_target_frob_ratio"]]
        print("\nLayer with smallest residual k_90 per pair:")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
