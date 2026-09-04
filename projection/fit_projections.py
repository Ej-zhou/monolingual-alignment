#!/usr/bin/env python3
"""Cross-lingual representation reconstruction (Section 4).

For each language pair and each of the 13 hidden-state layers, learn a map from
the source Goldfish model's SGPT-pooled sentence embeddings to the target
model's, on an 80/20 train/test split of the pooled parallel data, and evaluate
it on the held-out split.  Methods (all minimise squared reconstruction error):

  procrustes  orthogonal rotation W = U V^T from the SVD of X^T Y   ("Procrustes")
  ridge       linear + bias with L2 penalty, lambda chosen on a
              validation slice of the train split                  ("Affine" in the paper)
  mlp         one hidden layer (width d), ReLU, Adam, early stopping ("MLP")
  affine      unregularised least squares (reported nowhere; overfits)
  procrustes_centered  rotation on centred data + bias (diagnostic)
  identity    no projection (baseline)

Metrics on the test split: retrieval P@1/5/10 by cosine against the test pool
(p_at_k) and against the train+test pool (p_at_k_pool, "P@1 hard"), MSE, and
post-projection linear CKA.

Parallel data: <data-dir>/<dataset>/<a>-<b>.csv (see data/download_parallel.py).
Clean datasets (--always-include) are used in full; noisy ones are filtered to
>= --min-len-chars characters on both sides and capped per pair by
--quotas-file.  SGPT embeddings are cached under cache/ so re-runs only redo
the projection step.

Usage:
  python fit_projections.py --datasets flores opus tatoeba bouquet \
      --quotas-file quotas.json --run-name all_pairs --device cuda
  python fit_projections.py --pairs eng_latn-fra_latn --save-predictions-for eng_latn-fra_latn \
      --datasets flores opus tatoeba bouquet --quotas-file quotas.json --run-name eng_fra_pca
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Defaults ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent / "data" / "projection"
DEFAULT_CACHE_DIR = ROOT / "cache"
DEFAULT_RESULTS_DIR = ROOT / "results"

LANGUAGES = [
    "eng_latn", "zho_hans", "spa_latn", "arb_arab", "hin_deva",
    "fra_latn", "rus_cyrl", "deu_latn", "jpn_jpan",
]
# All 36 pairs A -> B with A before B in the ordering above.
DEFAULT_PAIRS = [f"{a}-{b}" for i, a in enumerate(LANGUAGES) for b in LANGUAGES[i + 1:]]

GOLDFISH_TEMPLATE = "goldfish-models/{lang}_{size}"


# ── SGPT pooling ────────────────────────────────────────────────────────────

def sgpt_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Position-weighted mean pool (SGPT-BE). (B, T, D) -> (B, D)."""
    seq_len = hidden.size(1)
    positions = torch.arange(1, seq_len + 1, device=hidden.device, dtype=hidden.dtype)
    weights = positions.unsqueeze(0) * attention_mask.to(hidden.dtype)
    weighted = (hidden * weights.unsqueeze(-1)).sum(dim=1)
    return weighted / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)


# ── Embedding extraction (with caching) ─────────────────────────────────────

def extract_layer_embeddings(
    model_name: str,
    sentences: List[str],
    cache_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Return per-layer SGPT embeddings of shape (n_layers, n_sents, dim).

    The first layer index (0) is the embedding layer's hidden state; subsequent
    indices are post-block hidden states, matching huggingface output_hidden_states.
    """
    if cache_path.exists():
        arr = np.load(cache_path)
        return arr

    print(f"  Extracting embeddings for {model_name} -> {cache_path.name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device).eval()

    layer_buffers: List[List[np.ndarray]] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), batch_size), leave=False):
            batch = sentences[start:start + batch_size]
            enc = tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            out = model(**enc)
            mask = enc["attention_mask"]
            for li, h in enumerate(out.hidden_states):
                pooled = sgpt_pool(h, mask).cpu().numpy().astype(np.float32)
                if len(layer_buffers) <= li:
                    layer_buffers.append([])
                layer_buffers[li].append(pooled)

    arr = np.stack([np.concatenate(b, axis=0) for b in layer_buffers], axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return arr


# ── Projection methods ──────────────────────────────────────────────────────

def fit_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: W = U V^T, with U, _, V^T = SVD(X^T Y).

    Operates on raw (un-centered) X, Y — rotation only, no translation.
    """
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def apply_procrustes(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return X @ W


def fit_procrustes_centered(
    X: np.ndarray, Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centered orthogonal Procrustes: rotation on centered features + bias.

    Subtracts the per-language mean before solving for the rotation, then
    folds the centroid offset into a bias term so that

        x_new = (x - X.mean) @ W + Y.mean = x @ W + b
        b     = Y.mean - X.mean @ W

    This is the textbook Procrustes recipe; the un-centered variant above
    cannot move one centroid to the other and absorbs any mean offset
    into the rotation, which under-fits when the two clouds have
    different means (which Goldfish reps do — see geometry_exp1).
    """
    Xm = X.mean(0, keepdims=True)
    Ym = Y.mean(0, keepdims=True)
    Xc = X - Xm
    Yc = Y - Ym
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    W = (U @ Vt).astype(np.float32)
    b = (Ym - Xm @ W).reshape(-1).astype(np.float32)
    return W, b


def fit_affine(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form least squares: solve [X | 1] [W; b] = Y."""
    n = X.shape[0]
    Xa = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    Wb, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    W = Wb[:-1].astype(np.float32)
    b = Wb[-1].astype(np.float32)
    return W, b


def apply_affine(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def _ridge_closed_form(X: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge regression with bias (bias unregularized).

    min_W,b ||XW + b - Y||_F^2 + lam * ||W||_F^2.
    Centering X, Y absorbs the bias term.
    """
    Xm = X.mean(0, keepdims=True)
    Ym = Y.mean(0, keepdims=True)
    Xc = X - Xm
    Yc = Y - Ym
    d = X.shape[1]
    A = Xc.T @ Xc + lam * np.eye(d, dtype=X.dtype)
    W = np.linalg.solve(A, Xc.T @ Yc).astype(np.float32)
    b = (Ym - Xm @ W).reshape(-1).astype(np.float32)
    return W, b


def fit_ridge(
    X: np.ndarray, Y: np.ndarray,
    lambdas: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4),
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Ridge with lambda chosen by validation MSE on a held-out slice of train.

    Returns (W, b, best_lambda). The final fit uses *all* of (X, Y) at the
    chosen lambda, so the resulting W has access to the full training set.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = max(int((1 - val_frac) * n), 1)
    tr, va = idx[:cut], idx[cut:]
    if va.size == 0:
        # Tiny train: fall back to a sensible default lambda.
        W, b = _ridge_closed_form(X, Y, 1.0)
        return W, b, 1.0

    best_lam = lambdas[0]
    best_mse = float("inf")
    for lam in lambdas:
        W, b = _ridge_closed_form(X[tr], Y[tr], lam)
        mse = float(np.mean((X[va] @ W + b - Y[va]) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    W, b = _ridge_closed_form(X, Y, best_lam)
    return W, b, best_lam


class MLPProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(
    X: np.ndarray, Y: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    epochs: int, lr: float, batch_size: int, device: str, seed: int,
    weight_decay: float = 1e-4,
    patience: int = 20,
    val_frac: float = 0.1,
) -> Tuple[MLPProjector, Dict[str, List[float]]]:
    """Train MLP with weight decay and early stopping on an internal val slice.

    A held-out fraction of (X, Y) is reserved as a validation set used solely
    to drive early stopping — the test set is never seen during training.
    The model with lowest val loss seen during training is restored at the end.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    perm = rng.permutation(n_total)
    n_val = max(int(val_frac * n_total), 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    dim = X.shape[1]
    model = MLPProjector(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X[tr_idx]).to(device)
    Yt = torch.from_numpy(Y[tr_idx]).to(device)
    Xv = torch.from_numpy(X[val_idx]).to(device)
    Yv = torch.from_numpy(Y[val_idx]).to(device)
    Xtt = torch.from_numpy(X_test).to(device)
    Ytt = torch.from_numpy(Y_test).to(device)
    n_tr = Xt.size(0)

    history = {"train_loss": [], "val_loss": [], "test_loss": []}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(epochs):
        model.train()
        perm_tr = torch.randperm(n_tr, device=device)
        running = 0.0
        nb = 0
        for s in range(0, n_tr, batch_size):
            idx = perm_tr[s:s + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, Yt[idx])
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(nb, 1)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
            test_loss = loss_fn(model(Xtt), Ytt).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if patience > 0 and epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return model, history


def apply_mlp(model: MLPProjector, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


# ── Evaluation ──────────────────────────────────────────────────────────────

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def retrieval_at_k(P: np.ndarray, Y: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    """For each row i in P (projected source), check if i is in top-k by cosine
    similarity over rows of Y (target)."""
    sim = cosine_similarity_matrix(P, Y)
    n = sim.shape[0]
    out = {}
    max_k = max(ks)
    topk = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    for k in ks:
        top = topk[:, :k]
        # Refine ordering inside top-max_k for k-th rank correctness.
        rows = np.arange(n)[:, None]
        ordered = np.take_along_axis(top, np.argsort(-sim[rows, top], axis=1), axis=1)
        hits = (ordered[:, :k] == np.arange(n)[:, None]).any(axis=1).mean()
        out[f"p_at_{k}"] = float(hits)
    return out


def retrieval_at_k_pool(
    P: np.ndarray, Y_pool: np.ndarray, correct_idx: np.ndarray,
    ks=(1, 5, 10), suffix: str = "_pool",
) -> Dict[str, float]:
    """Retrieval against a fixed candidate pool with explicit correct indices.

    P:           (n_q, d) projected source queries.
    Y_pool:      (m,   d) candidate vectors.
    correct_idx: (n_q,) integer index in Y_pool for each query's correct match.
    """
    sim = cosine_similarity_matrix(P, Y_pool)
    n_q = sim.shape[0]
    out: Dict[str, float] = {}
    max_k = max(ks)
    topk = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    rows = np.arange(n_q)[:, None]
    ordered = np.take_along_axis(topk, np.argsort(-sim[rows, topk], axis=1), axis=1)
    correct = correct_idx[:, None]
    for k in ks:
        hits = (ordered[:, :k] == correct).any(axis=1).mean()
        out[f"p_at_{k}{suffix}"] = float(hits)
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = np.trace(Kc @ Lc)
    denom = math.sqrt(np.trace(Kc @ Kc) * np.trace(Lc @ Lc))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(hsic / denom)


# ── Data loading ────────────────────────────────────────────────────────────

def load_pair_csv(data_dir: Path, dataset: str, lang_a: str, lang_b: str) -> Tuple[List[str], List[str]]:
    p = data_dir / dataset / f"{lang_a}-{lang_b}.csv"
    if not p.exists():
        p = data_dir / dataset / f"{lang_b}-{lang_a}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {dataset} CSV for {lang_a}/{lang_b} in {data_dir / dataset}"
        )
    df = pd.read_csv(p).dropna(subset=[lang_a, lang_b])
    return df[lang_a].astype(str).tolist(), df[lang_b].astype(str).tolist()


def load_pooled_pair(
    data_dir: Path, datasets: List[str], lang_a: str, lang_b: str,
) -> Tuple[List[str], List[str]]:
    """Concatenate parallel sentences from multiple datasets, dedup on (a, b)."""
    seen = set()
    out_a: List[str] = []
    out_b: List[str] = []
    for ds in datasets:
        try:
            sa, sb = load_pair_csv(data_dir, ds, lang_a, lang_b)
        except FileNotFoundError as e:
            print(f"  SKIP dataset '{ds}': {e}")
            continue
        kept = 0
        for a, b in zip(sa, sb):
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out_a.append(a)
            out_b.append(b)
            kept += 1
        print(f"  {ds}: +{kept} unique pairs (running total {len(out_a)})")
    if not out_a:
        raise FileNotFoundError(
            f"No data for {lang_a}/{lang_b} across datasets {datasets}"
        )
    return out_a, out_b


# ── Driver ──────────────────────────────────────────────────────────────────

def evaluate_layer(
    src_layer: np.ndarray,
    tgt_layer: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int, lr: float, mlp_batch: int, device: str, seed: int,
    weight_decay: float = 1e-4,
    patience: int = 20,
    collect_predictions: bool = False,
) -> Tuple[List[Dict], Dict[str, List[float]], Dict[str, np.ndarray]]:
    Xtr, Ytr = src_layer[train_idx], tgt_layer[train_idx]
    Xte, Yte = src_layer[test_idx],  tgt_layer[test_idx]
    rows: List[Dict] = []
    preds: Dict[str, np.ndarray] = {}

    # Full target-language pool: train + test target embeddings stacked.
    # Correct match for test query i is at index n_train + i.
    Y_pool = np.concatenate([Ytr, Yte], axis=0)
    correct_idx = np.arange(len(Ytr), len(Ytr) + len(Yte))

    def metrics(P_train: np.ndarray, P_test: np.ndarray, name: str) -> Dict:
        return {
            "method": name,
            "train_mse": float(np.mean((P_train - Ytr) ** 2)),
            "mse": float(np.mean((P_test - Yte) ** 2)),
            "cka": linear_cka(P_test, Yte),
            **retrieval_at_k(P_test, Yte),
            **retrieval_at_k_pool(P_test, Y_pool, correct_idx, suffix="_pool"),
        }

    # baseline (no projection): "predictions" are the source representations themselves
    rows.append(metrics(Xtr, Xte, "identity"))
    if collect_predictions:
        preds["identity"] = Xte

    W = fit_procrustes(Xtr, Ytr)
    P_te = apply_procrustes(Xte, W)
    rows.append(metrics(apply_procrustes(Xtr, W), P_te, "procrustes"))
    if collect_predictions:
        preds["procrustes"] = P_te

    Wc, bc = fit_procrustes_centered(Xtr, Ytr)
    P_te = apply_affine(Xte, Wc, bc)
    rows.append(metrics(apply_affine(Xtr, Wc, bc), P_te, "procrustes_centered"))
    if collect_predictions:
        preds["procrustes_centered"] = P_te

    W, b = fit_affine(Xtr, Ytr)
    P_te = apply_affine(Xte, W, b)
    rows.append(metrics(apply_affine(Xtr, W, b), P_te, "affine"))
    if collect_predictions:
        preds["affine"] = P_te

    W, b, best_lam = fit_ridge(Xtr, Ytr, seed=seed)
    P_te = apply_affine(Xte, W, b)
    ridge_row = metrics(apply_affine(Xtr, W, b), P_te, "ridge")
    ridge_row["ridge_lambda"] = float(best_lam)
    rows.append(ridge_row)
    if collect_predictions:
        preds["ridge"] = P_te

    mlp, hist = fit_mlp(
        Xtr, Ytr, Xte, Yte, epochs, lr, mlp_batch, device, seed,
        weight_decay=weight_decay, patience=patience,
    )
    P_train = apply_mlp(mlp, Xtr, device)
    P_test = apply_mlp(mlp, Xte, device)
    mlp_row = metrics(P_train, P_test, "mlp")
    if collect_predictions:
        preds["mlp"] = P_test
    mlp_row.update({
        "mlp_train_loss_final": hist["train_loss"][-1],
        "mlp_val_loss_final":   hist["val_loss"][-1],
        "mlp_test_loss_final":  hist["test_loss"][-1],
        "mlp_best_val_loss":    hist["best_val_loss"],
        "mlp_best_epoch":       hist["best_epoch"],
        "mlp_epochs_run":       len(hist["train_loss"]),
    })
    rows.append(mlp_row)
    return rows, hist, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-lingual representation reconstruction")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="Language-pair directories of the form <a>-<b>")
    parser.add_argument("--datasets", nargs="+", default=["flores"],
                        help="Parallel-sentence datasets to pool "
                             "(e.g. flores tatoeba opus bouquet)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--size", default="1000mb",
                        help="Goldfish model size suffix")
    parser.add_argument("--num-samples", type=int, default=800,
                        help="How many parallel sentences to use (0 = all)")
    parser.add_argument("--min-len-chars", type=int, default=50,
                        help="Drop sentences shorter than this (in characters) "
                             "on EITHER side. Filters out OPUS subtitle / Tatoeba "
                             "interjections that cause SGPT representation collapse.")
    parser.add_argument("--always-include", nargs="+",
                        default=["flores", "bouquet"],
                        help="Datasets that bypass the length filter and are "
                             "always fully included (clean curated parallel data). "
                             "Datasets NOT listed here get length-filtered then "
                             "randomly sampled to fill the remaining budget.")
    parser.add_argument("--quotas-file", type=Path, default=None,
                        help="Path to a JSON file mapping {dataset: {pair: n_max}} "
                             "for fine-grained per-(dataset, pair) caps. When set, "
                             "--num-samples is ignored and total per-pair n is "
                             "determined by the quota spec plus all clean data.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Forward-pass batch size during extraction")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200,
                        help="MLP training epochs per (layer, pair)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mlp-batch", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization on MLP weights (Adam weight_decay).")
    parser.add_argument("--early-stop-patience", type=int, default=20,
                        help="Stop MLP training after this many epochs of no "
                             "improvement on internal val slice. 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-extraction even if cache exists")
    parser.add_argument("--run-name", default=None,
                        help="Subfolder under results/ and figures/ for this run. "
                             "Default: '<size>_n<num-samples>_minlen<min-len-chars>'.")
    parser.add_argument("--save-predictions-for", nargs="*", default=[],
                        help="Pairs (e.g. 'eng_latn-fra_latn') to dump per-layer "
                             "test predictions for (used by plot_pca_layers.py and "
                             "plot_pca_main.py).")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"{args.size}_n{args.num_samples}_minlen{args.min_len_chars}"

    quotas: Dict[str, Dict[str, int]] = {}
    if args.quotas_file is not None:
        with open(args.quotas_file) as f:
            quotas = json.load(f)
        print(f"Loaded quotas from {args.quotas_file}:")
        for ds, m in quotas.items():
            print(f"  {ds}: {m}")

    out_dir = args.results_dir / args.run_name
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {args.run_name}\nWriting results to: {out_dir}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_rows: List[Dict] = []
    mlp_curves: Dict[str, Dict[str, List[float]]] = {}

    for pair in args.pairs:
        try:
            la, lb = pair.split("-")
        except ValueError:
            raise ValueError(f"Bad pair format '{pair}', expected '<a>-<b>'")

        print(f"\n{'='*64}\nPair: {la} -> {lb}\n{'='*64}")

        # Per-dataset loading.
        # - Datasets in --always-include: load all, no length filter.
        # - Other datasets: length-filter, then cap by --quotas-file entry
        #   (if set) or by --num-samples global behavior (legacy).
        rng = np.random.default_rng(args.seed)
        sents_a: List[str] = []
        sents_b: List[str] = []
        composition: List[Tuple[str, int, int]] = []  # (dataset, raw, kept)
        seen: set = set()

        for ds in args.datasets:
            try:
                a, b = load_pair_csv(args.data_dir, ds, la, lb)
            except FileNotFoundError as e:
                print(f"  SKIP {ds}: {e}")
                continue

            n_raw = len(a)
            # Drop dups across datasets (FLORES vs OPUS overlap, etc.).
            pairs = [(x, y) for x, y in zip(a, b) if (x, y) not in seen]

            # Length filter on noisy datasets only.
            if ds not in args.always_include and args.min_len_chars > 0:
                pairs = [(x, y) for x, y in pairs
                         if len(x) >= args.min_len_chars
                         and len(y) >= args.min_len_chars]

            # Per-(dataset, pair) quota cap, if specified. Falls back to the
            # `_default` entry under the same dataset if a pair is not listed.
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

        # If no quotas file is set, fall back to the legacy global cap on n.
        if not quotas and args.num_samples > 0 and args.num_samples < len(sents_a):
            # Preserve always-include first, then sample noisy to fill budget.
            clean_ds = set(args.always_include)
            clean_count = sum(kept for ds, _, kept in composition if ds in clean_ds)
            n_target = args.num_samples
            if n_target <= clean_count:
                # Sample only from clean (rare edge case for very small n).
                idx = rng.choice(clean_count, size=n_target, replace=False)
                sents_a = [sents_a[i] for i in idx]
                sents_b = [sents_b[i] for i in idx]
            else:
                # Keep all clean, sample noisy to fill the rest.
                noisy_total = len(sents_a) - clean_count
                need = min(n_target - clean_count, noisy_total)
                noisy_idx = clean_count + rng.choice(noisy_total, size=need, replace=False)
                keep_idx = list(range(clean_count)) + sorted(noisy_idx.tolist())
                sents_a = [sents_a[i] for i in keep_idx]
                sents_b = [sents_b[i] for i in keep_idx]

        n = len(sents_a)
        if n == 0:
            raise RuntimeError(f"No sentences left for {pair}.")
        print("  Composition:")
        for ds, n_raw, kept in composition:
            if ds in args.always_include:
                mode = "all"
            elif quotas:
                qd = quotas.get(ds, {})
                q = qd.get(pair, qd.get("_default"))
                mode = f"quota={q}" if q is not None else "filtered"
            else:
                mode = "filtered"
            print(f"    {ds:<8} raw={n_raw:>6} kept={kept:>6}  ({mode})")
        print(f"  -> total per-pair: {n}")
        n_avail = n
        n_pooled = n

        model_a = GOLDFISH_TEMPLATE.format(lang=la, size=args.size)
        model_b = GOLDFISH_TEMPLATE.format(lang=lb, size=args.size)

        ds_tag = "+".join(sorted(args.datasets))
        filt_tag = f"_minlen{args.min_len_chars}" if args.min_len_chars > 0 else ""
        clean_tag = "_clean-" + "+".join(sorted(args.always_include)) if args.always_include else ""
        if quotas:
            import hashlib
            spec = "|".join(f"{ds}:{quotas.get(ds, {}).get(pair, '')}"
                            for ds in sorted(args.datasets))
            quota_tag = "_q" + hashlib.md5(spec.encode()).hexdigest()[:8]
        else:
            quota_tag = ""
        cache_a = args.cache_dir / f"{ds_tag}_{la}_{args.size}_n{n}{filt_tag}{clean_tag}{quota_tag}_seed{args.seed}.npy"
        cache_b = args.cache_dir / f"{ds_tag}_{lb}_{args.size}_n{n}{filt_tag}{clean_tag}{quota_tag}_seed{args.seed}.npy"
        if args.no_cache:
            for p in (cache_a, cache_b):
                if p.exists():
                    p.unlink()

        emb_a = extract_layer_embeddings(model_a, sents_a, cache_a,
                                         args.device, args.batch_size, args.max_length)
        emb_b = extract_layer_embeddings(model_b, sents_b, cache_b,
                                         args.device, args.batch_size, args.max_length)

        n_layers = min(emb_a.shape[0], emb_b.shape[0])
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(args.train_frac * n)
        train_idx, test_idx = idx[:cut], idx[cut:]
        print(f"Train/test: {len(train_idx)} / {len(test_idx)}")
        print(f"Layers: {n_layers}, embedding dim a={emb_a.shape[2]} b={emb_b.shape[2]}")

        if emb_a.shape[2] != emb_b.shape[2]:
            print("WARNING: embedding dims differ; affine/MLP will still work, "
                  "Procrustes requires square X^T Y so will use min-dim PCA fallback")

        save_preds_here = pair in args.save_predictions_for
        if save_preds_here:
            preds_dir = out_dir / "predictions" / pair
            preds_dir.mkdir(parents=True, exist_ok=True)

        for li in tqdm(range(n_layers), desc="layers"):
            t0 = time.time()
            rows, hist, preds = evaluate_layer(
                emb_a[li], emb_b[li],
                train_idx, test_idx,
                epochs=args.epochs, lr=args.lr, mlp_batch=args.mlp_batch,
                device=args.device, seed=args.seed + li,
                weight_decay=args.weight_decay,
                patience=args.early_stop_patience,
                collect_predictions=save_preds_here,
            )
            mlp_curves[f"{pair}__layer{li}"] = hist
            if save_preds_here:
                np.savez(
                    preds_dir / f"layer_{li:02d}.npz",
                    X_test=emb_a[li][test_idx],
                    Y_test=emb_b[li][test_idx],
                    **{f"pred_{m}": p for m, p in preds.items()},
                )
            for r in rows:
                r.update({
                    "pair": pair,
                    "src": la,
                    "tgt": lb,
                    "datasets": "+".join(args.datasets),
                    "size": args.size,
                    "layer": li,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "elapsed_s": round(time.time() - t0, 2),
                })
                all_rows.append(r)

        # Per-pair partial dump (in case of long jobs)
        df = pd.DataFrame(all_rows)
        df.to_csv(out_dir / "projection_results.csv", index=False)
        with open(out_dir / "mlp_curves.json", "w") as f:
            json.dump(mlp_curves, f)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "projection_results.csv"
    json_path = out_dir / "projection_results.json"
    curves_path = out_dir / "mlp_curves.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(curves_path, "w") as f:
        json.dump(mlp_curves, f)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {curves_path}")

    # Console summary: best-layer P@1 per (pair, method)
    if not df.empty:
        summary = (df[df["method"] != "identity"]
                   .groupby(["pair", "method"])["p_at_1"].max()
                   .unstack("method"))
        print("\nBest-layer P@1 per pair x method:")
        print(summary.round(3))


if __name__ == "__main__":
    main()
