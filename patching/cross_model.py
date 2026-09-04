#!/usr/bin/env python3
"""Cross-model span patching with a learned projection (Section 5, Figure 6).

For a (source, target) language pair:
  1. run the source model on the source-language donor prompt and extract the
     residual at the donor concept position at every layer;
  2. fit per-layer Procrustes and Affine maps source -> target on last-token
     activations of parallel sentences (FLORES + OPUS + BouQUET, capped by
     ../projection/quotas.json, 2000 sentences, 80/20 split);
  3. span-patch the projected residual into the target model at the target
     concept position, layers j..L, and score the donor answer.

Conditions: procrustes, affine, unprojected (raw source residual; basis-mismatch
control) and shuffle (Gaussian sample matching the source activation moments;
distributional control).  The within-model ceiling is within_model.py.

Writes results/<run-name>/cross_results.csv (one row per donor, target,
condition, j) and cross_recon.csv (per-layer held-out fit quality).

Usage:
  python cross_model.py --source-lang eng_latn --target-lang deu_latn \
      --n-test 30 --layers 1 2 3 4 5 6 7 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECTION_DIR = ROOT.parent / "projection"
FACTS_DIR = ROOT / "facts"
DATA_DIR = ROOT.parent / "data" / "projection"
sys.path.insert(0, str(PROJECTION_DIR))
sys.path.insert(0, str(ROOT))
from fit_projections import (apply_affine, apply_procrustes, fit_affine,  # noqa: E402
                             fit_procrustes, load_pair_csv)
from within_model import (LANG_VERB, build_naked_prompt,  # noqa: E402,F401
                          country_last_token_position, first_subword_id,
                          forward_collect, forward_patched, get_blocks,
                          get_capital, logp_at, rank_of)


# ── Parallel-sentence pool for fitting the projections ──────────────────────

def load_pooled_pair(pair, datasets, quotas, always_include, min_len_chars, seed, data_dir):
    """Same composition rule as fit_projections.py: clean datasets in full,
    noisy ones length-filtered and capped by the per-pair quota."""
    la, lb = pair.split("-")
    rng = np.random.default_rng(seed)
    seen: set = set()
    out_a: List[str] = []
    out_b: List[str] = []
    comp: List[Tuple[str, int, int]] = []
    for ds in datasets:
        try:
            a, b = load_pair_csv(data_dir, ds, la, lb)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")
            continue
        n_raw = len(a)
        pairs_list = [(x, y) for x, y in zip(a, b) if (x, y) not in seen]
        if ds not in always_include and min_len_chars > 0:
            pairs_list = [(x, y) for x, y in pairs_list
                          if len(x) >= min_len_chars and len(y) >= min_len_chars]
        if quotas:
            qd = quotas.get(ds, {})
            quota = qd.get(pair, qd.get("_default"))
            if quota is not None and quota < len(pairs_list):
                idx = rng.choice(len(pairs_list), size=quota, replace=False)
                pairs_list = [pairs_list[i] for i in idx]
        for x, y in pairs_list:
            seen.add((x, y))
            out_a.append(x)
            out_b.append(y)
        comp.append((ds, n_raw, len(pairs_list)))
    return out_a, out_b, comp


# ── Pair-aligned cache ──────────────────────────────────────────────────────

def build_or_load_pair_cache(
    src_lang: str, tgt_lang: str, size: str,
    model_s, tok_s, model_t, tok_t,
    device: str, max_length: int,
    fit_datasets: Tuple[str, ...] = ("flores", "opus", "bouquet"),
    always_include: Tuple[str, ...] = ("flores", "bouquet"),
    min_len_chars: int = 50, n_max: int = 2000, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or build the aligned (src, tgt) last-token activation caches for one
    pair: shape (n_layers, n_sents, dim) each, same sentence order on both sides.
    The cache key includes the pair so eng->fra and eng->deu use separate eng caches."""
    tag = f"pair_{src_lang}-{tgt_lang}_{size}_n{n_max}_minlen{min_len_chars}_seed{seed}"
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cs = cache_dir / f"{tag}_{src_lang}_lasttok.npy"
    ct = cache_dir / f"{tag}_{tgt_lang}_lasttok.npy"
    if cs.exists() and ct.exists():
        return np.load(cs), np.load(ct)

    quotas_path = PROJECTION_DIR / "quotas.json"
    quotas = json.loads(quotas_path.read_text()) if quotas_path.exists() else {}
    pair_str = f"{src_lang}-{tgt_lang}"
    sa, sb, comp = load_pooled_pair(
        pair_str, list(fit_datasets), quotas,
        always_include, min_len_chars, seed,
        DATA_DIR,
    )
    # Cap to n_max with reproducible shuffle.
    if len(sa) > n_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sa), size=n_max, replace=False)
        sa = [sa[i] for i in idx]; sb = [sb[i] for i in idx]
    print(f"  building pair cache for {pair_str}: {len(sa)} sentences")
    for ds, raw, kept in comp:
        print(f"    {ds:<10} raw={raw:>6}  kept={kept:>6}")
    fit_s = _extract_lasttok(model_s, tok_s, sa, device, max_length)
    fit_t = _extract_lasttok(model_t, tok_t, sb, device, max_length)
    np.save(cs, fit_s); np.save(ct, fit_t)
    print(f"  wrote {cs.name} and {ct.name}")
    return fit_s, fit_t


def _extract_lasttok(model, tok, sentences, device, max_length) -> np.ndarray:
    """Returns (n_layers, n_sents, dim) of last-token activations."""
    layer_buffers: List[List[np.ndarray]] = []
    with torch.no_grad():
        for s in sentences:
            enc = tok(s, return_tensors="pt", truncation=True,
                      max_length=max_length, add_special_tokens=True).to(device)
            out = model(**enc, output_hidden_states=True)
            for li, h in enumerate(out.hidden_states):
                last = h[0, -1, :].cpu().numpy().astype(np.float32)
                if len(layer_buffers) <= li:
                    layer_buffers.append([])
                layer_buffers[li].append(last)
    return np.stack([np.stack(b, axis=0) for b in layer_buffers], axis=0)


# ── Fitting + reconstruction quality ────────────────────────────────────────

def fit_per_layer(
    fit_src: np.ndarray, fit_tgt: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    """Returns (proc_per_layer, affine_per_layer). Each entry is a dict with
    fit params + test-set reconstruction stats: cosine, mse, r2."""
    n_layers = fit_src.shape[0]
    proc_layers, aff_layers = [], []
    for k in range(n_layers):
        Xtr, Ytr = fit_src[k, train_idx], fit_tgt[k, train_idx]
        Xte, Yte = fit_src[k, test_idx], fit_tgt[k, test_idx]
        # Procrustes
        Wp = fit_procrustes(Xtr, Ytr)
        Yhat_p = apply_procrustes(Xte, Wp)
        proc_layers.append({"W": Wp, **recon_stats(Yhat_p, Yte)})
        # Affine
        Wa, ba = fit_affine(Xtr, Ytr)
        Yhat_a = apply_affine(Xte, Wa, ba)
        aff_layers.append({"W": Wa, "b": ba, **recon_stats(Yhat_a, Yte)})
    return proc_layers, aff_layers


def recon_stats(Yhat: np.ndarray, Y: np.ndarray) -> Dict:
    """Test-set reconstruction quality — mean cosine sim, per-dim MSE,
    per-vector R² (1 − ‖Yhat−Y‖² / ‖Y − Y_mean‖²)."""
    diff = Yhat - Y
    mse = float((diff ** 2).mean())
    nh = np.linalg.norm(Yhat, axis=1) + 1e-12
    ny = np.linalg.norm(Y, axis=1) + 1e-12
    cos = float(((Yhat * Y).sum(axis=1) / (nh * ny)).mean())
    Y_mean = Y.mean(axis=0, keepdims=True)
    ss_res = float(((Yhat - Y) ** 2).sum())
    ss_tot = float(((Y - Y_mean) ** 2).sum() + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"cos": cos, "mse": mse, "r2": r2}


def fit_shuffle_per_layer(fit_src: np.ndarray, train_idx: np.ndarray, seed: int):
    n_layers = fit_src.shape[0]
    rngs = [np.random.default_rng(seed + 7919 + k) for k in range(n_layers)]
    means = [fit_src[k, train_idx].mean(axis=0) for k in range(n_layers)]
    chols = []
    for k in range(n_layers):
        cov = np.cov(fit_src[k, train_idx], rowvar=False)
        try:
            chols.append(np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0])))
        except np.linalg.LinAlgError:
            chols.append(None)
    def sample(k):
        d = means[k].shape[0]
        z = rngs[k].standard_normal(d)
        if chols[k] is None:
            return means[k] + z * 0.1
        return means[k] + chols[k] @ z
    return sample


def project_source_states(
    cond: str, source_states: List[torch.Tensor],
    proc_layers: List[Dict], aff_layers: List[Dict], sample_shuffle, device: str,
) -> List[torch.Tensor]:
    out = []
    for k, h in enumerate(source_states):
        h_np = h.detach().cpu().numpy().astype(np.float32)
        if cond == "unprojected":
            v = h_np
        elif cond == "procrustes":
            v = (h_np[None, :] @ proc_layers[k]["W"])[0]
        elif cond == "affine":
            v = apply_affine(h_np[None, :], aff_layers[k]["W"], aff_layers[k]["b"])[0]
        elif cond == "shuffle":
            v = sample_shuffle(k).astype(np.float32)
        else:
            raise KeyError(cond)
        out.append(torch.from_numpy(v.astype(np.float32)).to(device))
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def run(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    facts_all = json.loads(args.facts_file.read_text())
    facts = [f for f in facts_all
             if all(L in f["prompt"] and L in f["answer"]
                    for L in (args.source_lang, args.target_lang))]
    facts = facts[: args.n_test]
    print(f"facts: {len(facts)}; pair: {args.source_lang} -> {args.target_lang}")

    src_id = f"goldfish-models/{args.source_lang}_{args.size}"
    tgt_id = f"goldfish-models/{args.target_lang}_{args.size}"
    print(f"loading {src_id} ...")
    tok_s = AutoTokenizer.from_pretrained(src_id, use_fast=True)
    model_s = AutoModelForCausalLM.from_pretrained(src_id).to(args.device).eval()
    print(f"loading {tgt_id} ...")
    tok_t = AutoTokenizer.from_pretrained(tgt_id, use_fast=True)
    model_t = AutoModelForCausalLM.from_pretrained(tgt_id).to(args.device).eval()

    L = len(get_blocks(model_t))
    print(f"models loaded, L={L} blocks, hidden={model_t.config.n_embd}")

    fit_s, fit_t = build_or_load_pair_cache(
        args.source_lang, args.target_lang, args.size,
        model_s, tok_s, model_t, tok_t, args.device, args.max_length,
    )
    print(f"fit cache: src {fit_s.shape}, tgt {fit_t.shape}")

    rng = np.random.default_rng(args.seed)
    n_fit = fit_s.shape[1]
    idx = np.arange(n_fit); rng.shuffle(idx)
    cut = int(0.8 * n_fit)
    train_idx, test_idx = idx[:cut], idx[cut:]

    print("fitting per-layer Procrustes and Affine ...")
    proc_layers, aff_layers = fit_per_layer(fit_s, fit_t, train_idx, test_idx)
    sample_shuffle = fit_shuffle_per_layer(fit_s, train_idx, seed=args.seed)

    # Reconstruction-quality table (test set, per layer).
    recon_rows = []
    for k in range(len(proc_layers)):
        recon_rows.append({"layer": k, "method": "procrustes",
                           "cos": proc_layers[k]["cos"],
                           "mse": proc_layers[k]["mse"],
                           "r2":  proc_layers[k]["r2"]})
        recon_rows.append({"layer": k, "method": "affine",
                           "cos": aff_layers[k]["cos"],
                           "mse": aff_layers[k]["mse"],
                           "r2":  aff_layers[k]["r2"]})
    recon_df = pd.DataFrame(recon_rows)
    print("\n=== Per-layer reconstruction quality (test set) ===")
    pivot_cos = recon_df.pivot(index="layer", columns="method", values="cos").round(3)
    pivot_r2  = recon_df.pivot(index="layer", columns="method", values="r2").round(3)
    print("cosine similarity:")
    print(pivot_cos.to_string())
    print("\nR²:")
    print(pivot_r2.to_string())

    # Per-fact baselines.
    print("\n=== per-fact target baseline ===")
    print(f"{'id':<14} {'cap':<14} {'pos_s':>5} {'pos_t':>5} {'rank':>5} {'logp':>8}")
    src_states: Dict[str, List[torch.Tensor]] = {}
    tgt_prompt: Dict[str, str] = {}
    tgt_pos: Dict[str, int] = {}
    tgt_base: Dict[str, np.ndarray] = {}
    tgt_cap_tid: Dict[str, int] = {}
    for f in facts:
        prompt_s = build_naked_prompt(f, args.source_lang)
        pos_s = country_last_token_position(tok_s, prompt_s, args.source_lang)
        s_states, _ = forward_collect(model_s, tok_s, prompt_s, pos_s,
                                       args.device, args.max_length)
        src_states[f["id"]] = s_states
        prompt_t = build_naked_prompt(f, args.target_lang)
        pos_t = country_last_token_position(tok_t, prompt_t, args.target_lang)
        _, base_logits = forward_collect(model_t, tok_t, prompt_t, pos_t,
                                         args.device, args.max_length)
        capital_t = get_capital(f, args.target_lang)
        cap_tid = first_subword_id(tok_t, capital_t, prompt_t)
        tgt_prompt[f["id"]] = prompt_t
        tgt_pos[f["id"]] = pos_t
        tgt_base[f["id"]] = base_logits
        tgt_cap_tid[f["id"]] = cap_tid
        print(f"{f['id']:<14} {capital_t:<14} {pos_s:>5d} {pos_t:>5d} "
              f"{rank_of(base_logits, cap_tid):>5d} "
              f"{logp_at(base_logits, cap_tid):>8.3f}")

    if args.baseline_only:
        return pd.DataFrame(), recon_df

    layers = [int(x) for x in args.layers] if args.layers != ["all"] else list(range(0, L + 1))
    conditions = args.conditions
    print(f"\nj_start ∈ {layers}, conditions = {conditions}")

    rows = []
    for ti, target in enumerate(facts):
        t_id = target["id"]
        t_pos = tgt_pos[t_id]
        z_base = tgt_base[t_id]
        t_cap_tid = tgt_cap_tid[t_id]
        for di, donor in enumerate(facts):
            if di == ti:
                continue
            d_id = donor["id"]
            d_cap_in_target_tid = first_subword_id(
                tok_t, get_capital(donor, args.target_lang), tgt_prompt[t_id]
            )
            for cond in conditions:
                projected = project_source_states(
                    cond, src_states[d_id], proc_layers, aff_layers,
                    sample_shuffle, args.device,
                )
                for j in layers:
                    z_p = forward_patched(
                        model_t, tok_t, tgt_prompt[t_id], t_pos,
                        projected, j, args.device, args.max_length,
                    )
                    rows.append({
                        "donor_id": d_id, "target_id": t_id,
                        "j_start": j, "condition": cond,
                        "logp_donor_base":  logp_at(z_base, d_cap_in_target_tid),
                        "logp_donor_patch": logp_at(z_p,    d_cap_in_target_tid),
                        "logp_target_base":  logp_at(z_base, t_cap_tid),
                        "logp_target_patch": logp_at(z_p,    t_cap_tid),
                        "delta_logp_donor":
                            logp_at(z_p, d_cap_in_target_tid)
                            - logp_at(z_base, d_cap_in_target_tid),
                        "delta_logp_target":
                            logp_at(z_p, t_cap_tid) - logp_at(z_base, t_cap_tid),
                        "argmax_base":  int(z_base.argmax()),
                        "argmax_patch": int(z_p.argmax()),
                        "flip_to_donor": int(int(z_p.argmax()) == d_cap_in_target_tid),
                    })

    return pd.DataFrame(rows), recon_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-lang", default="eng_latn")
    p.add_argument("--target-lang", default="fra_latn")
    p.add_argument("--size", default="1000mb")
    p.add_argument("--facts-file", type=Path, default=FACTS_DIR / "capital.json")
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--layers", nargs="+",
                   default=["1", "2", "3", "4", "5", "6", "7", "8"])
    p.add_argument("--conditions", nargs="+",
                   default=["procrustes", "affine", "unprojected", "shuffle"])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--run-name", default=None, help="Default: x_<src>-<tgt>")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    if args.run_name is None:
        args.run_name = f"x_{args.source_lang}-{args.target_lang}"
    out_dir = args.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {args.run_name} -> {out_dir}\n")

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    df, recon_df = run(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_path = out_dir / "cross_recon.csv"
    recon_df.to_csv(recon_path, index=False)
    print(f"Wrote {recon_path}")
    if df.empty:
        return
    csv_path = out_dir / "cross_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    print("\n=== Summary by (condition, j_start) ===")
    g = df.groupby(["condition", "j_start"]).agg(
        mean_dl_donor=("delta_logp_donor", "mean"),
        mean_dl_target=("delta_logp_target", "mean"),
        flip_rate=("flip_to_donor", "mean"),
    ).round(3)
    g["rate_donor_pos"] = (
        df.groupby(["condition", "j_start"])["delta_logp_donor"]
          .apply(lambda s: (s > 0).mean()).round(3)
    )
    print(g)


if __name__ == "__main__":
    main()
