#!/usr/bin/env python3
"""Same language, different architecture: Goldfish vs. Pythia (Appendix C.1).

Per-layer linear CKA between a Goldfish model and a Pythia model trained on the
same language, on the SAME monolingual FLORES sentences (matched) and with one
side permuted (shuffled).  Settings as in cka/cka_pairwise.py; the token-level
setting uses identity word alignment since both sides see the same text.

Usage:
  python goldfish_vs_pythia.py --model-a goldfish-models/eng_latn_1000mb \
      --model-b EleutherAI/pythia-160m --lang eng_latn --device cuda
  python goldfish_vs_pythia.py --model-a goldfish-models/zho_hans_1000mb \
      --model-b SJTU-CL/Zh-Pythia-160M --lang zho_hans --device cuda
"""
import argparse
import csv
import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── CKA ──────────────────────────────────────────────────────────────────────

def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    assert X.shape[0] == Y.shape[0]
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
        return float('nan')
    return float(hsic / denom)


# ── Pooling helpers ───────────────────────────────────────────────────────────

def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = hidden.float()
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def sgpt_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = hidden.float()
    seq_len = hidden.size(1)
    positions = torch.arange(1, seq_len + 1, device=hidden.device, dtype=torch.float32)
    weights = positions.unsqueeze(0) * attention_mask.float()
    weighted_sum = (hidden * weights.unsqueeze(-1)).sum(dim=1)
    return weighted_sum / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)


# ── Token-level alignment helpers (identity alignment for same text) ────────

def word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), text[m.start():m.end()])
            for m in re.finditer(r'\S+', text)]


def map_word_to_token_indices(
    offsets: List[Tuple[int, int]],
    wspans: List[Tuple[int, int, str]],
) -> List[List[int]]:
    w2t: List[List[int]] = [[] for _ in wspans]
    for ti, (ts, te) in enumerate(offsets):
        if ts == te == 0:
            continue
        for wi, (ws, we, _) in enumerate(wspans):
            if not (te <= ws or ts >= we):
                w2t[wi].append(ti)
    return w2t


def identity_alignments(sentences: List[str]) -> List[List[Tuple[int, int]]]:
    """Identity word alignment: word i maps to word i (same text both sides)."""
    alignments = []
    for sent in sentences:
        n_words = len(word_spans(sent))
        alignments.append([(i, i) for i in range(n_words)])
    return alignments


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_monolingual_sentences(lang: str, data_dir: str) -> List[str]:
    """Monolingual FLORES sentences: the `lang` column of any FLORES pair CSV
    under <data_dir>/flores/ (written by data/download_parallel.py)."""
    import pandas as pd
    for csv_file in sorted((Path(data_dir) / 'flores').glob('*.csv')):
        df = pd.read_csv(csv_file)
        if lang in df.columns:
            sents = df[lang].dropna().astype(str).tolist()
            print(f'Loaded {len(sents)} sentences for {lang} from {csv_file}')
            return sents
    raise FileNotFoundError(
        f'No FLORES CSV with a {lang} column under {data_dir}/flores. '
        f'Run data/download_parallel.py first.'
    )


# ── Per-setting CKA runners ───────────────────────────────────────────────────

def run_setting1_mean(hiddens_a, hiddens_b, mask_a, mask_b, reps_a, reps_b):
    for li, h in enumerate(hiddens_a):
        reps_a[li].extend(mean_pool(h, mask_a).cpu().numpy())
    for li, h in enumerate(hiddens_b):
        reps_b[li].extend(mean_pool(h, mask_b).cpu().numpy())


def run_setting3_sgpt(hiddens_a, hiddens_b, mask_a, mask_b, reps_a, reps_b):
    for li, h in enumerate(hiddens_a):
        reps_a[li].extend(sgpt_pool(h, mask_a).cpu().numpy())
    for li, h in enumerate(hiddens_b):
        reps_b[li].extend(sgpt_pool(h, mask_b).cpu().numpy())


def run_setting2_token(
    hiddens_a, hiddens_b,
    off_map_a, off_map_b,
    batch_sents: List[str],
    alignments: List[List[Tuple[int, int]]],
    reps_a: List[List], reps_b: List[List],
) -> None:
    """Token-level CKA with identity word alignment (same text, different tokenizers)."""
    for bi in range(len(batch_sents)):
        wspan = word_spans(batch_sents[bi])
        off_a = off_map_a[bi].cpu().tolist()
        off_b = off_map_b[bi].cpu().tolist()
        w2t_a = map_word_to_token_indices(off_a, wspan)
        w2t_b = map_word_to_token_indices(off_b, wspan)

        for (wa_idx, wb_idx) in alignments[bi]:
            if wa_idx < 0 or wb_idx < 0:
                continue
            if wa_idx >= len(w2t_a) or wb_idx >= len(w2t_b):
                continue
            ta_inds = w2t_a[wa_idx]
            tb_inds = w2t_b[wb_idx]
            if not ta_inds or not tb_inds:
                continue
            for li in range(len(hiddens_a)):
                ha_seq_len = hiddens_a[li][bi].shape[0]
                hb_seq_len = hiddens_b[li][bi].shape[0]
                ta_valid = [t for t in ta_inds if t < ha_seq_len]
                tb_valid = [t for t in tb_inds if t < hb_seq_len]
                if not ta_valid or not tb_valid:
                    continue
                vec_a = hiddens_a[li][bi].cpu().float().numpy()[ta_valid].mean(axis=0)
                vec_b = hiddens_b[li][bi].cpu().float().numpy()[tb_valid].mean(axis=0)
                reps_a[li].append(vec_a)
                reps_b[li].append(vec_b)


def compute_and_print_cka(
    reps_a, reps_b, setting_name, col_matched, col_shuffled,
) -> List[Tuple[int, float, float]]:
    min_layers = min(len(reps_a), len(reps_b))
    n_samples = min(reps_a[0].shape[0], reps_b[0].shape[0])
    perm = np.random.permutation(n_samples)
    print(f'\n=== {setting_name} ({n_samples} samples, {min_layers} layers) ===')
    print(f'Layer, {col_matched}, {col_shuffled}')
    results = []
    for i in range(min_layers):
        X = reps_a[i][:n_samples]
        Y = reps_b[i][:n_samples]
        matched = linear_CKA(X, Y)
        shuffled = linear_CKA(X, Y[perm])
        print(f'{i}, {matched:.6f}, {shuffled:.6f}')
        results.append((i, matched, shuffled))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cross-architecture CKA (Goldfish vs Pythia), same language.'
    )
    parser.add_argument('--model-a', required=True,
                        help='First model (e.g. goldfish-models/eng_latn_1000mb)')
    parser.add_argument('--model-b', required=True,
                        help='Second model (e.g. EleutherAI/pythia-160m)')
    parser.add_argument('--lang', required=True,
                        help='FLORES language code (e.g. eng_latn, zho_hans)')
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'cka'),
                        help='Directory containing <data-dir>/flores/*.csv')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load monolingual sentences ──────────────────────────────────────────
    sentences = load_monolingual_sentences(args.lang, data_dir=args.data_dir)
    n_available = len(sentences)
    n = min(args.num_samples, n_available)
    print(f'Using {n} sentences (available: {n_available})')

    idx = np.random.choice(n_available, size=n, replace=False)
    sel = [sentences[i] for i in idx]

    # ── Pre-compute identity word alignments for Setting 2 ──────────────────
    alignments = identity_alignments(sel)

    # ── Load models ─────────────────────────────────────────────────────────
    print('Loading tokenizers and models...')
    tok_a = AutoTokenizer.from_pretrained(args.model_a, use_fast=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, use_fast=True)
    if tok_a.pad_token is None:
        tok_a.pad_token = tok_a.eos_token
    if tok_b.pad_token is None:
        tok_b.pad_token = tok_b.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, output_hidden_states=True)
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, output_hidden_states=True)
    model_a.to(args.device).eval()
    model_b.to(args.device).eval()
    print(f'Model A: {args.model_a} ({sum(p.numel() for p in model_a.parameters())/1e6:.1f}M params)')
    print(f'Model B: {args.model_b} ({sum(p.numel() for p in model_b.parameters())/1e6:.1f}M params)')

    # ── Storage ──────────────────────────────────────────────────────────────
    s1_ra, s1_rb = [], []
    s2_ra, s2_rb = [], []
    s3_ra, s3_rb = [], []

    # ── Inference loop ──────────────────────────────────────────────────────
    print('Running inference...')
    for start in tqdm(range(0, n, args.batch_size)):
        end = min(n, start + args.batch_size)
        batch = sel[start:end]

        with torch.no_grad():
            # Both models get the SAME sentences
            ta_full = tok_a(batch, return_tensors='pt', padding=True,
                            truncation=True, return_offsets_mapping=True).to(args.device)
            tb_full = tok_b(batch, return_tensors='pt', padding=True,
                            truncation=True, return_offsets_mapping=True).to(args.device)
            off_a = ta_full.pop('offset_mapping')
            off_b = tb_full.pop('offset_mapping')
            out_a = model_a(**ta_full)
            out_b = model_b(**tb_full)

        ha = out_a.hidden_states
        hb = out_b.hidden_states
        ma = ta_full['attention_mask']
        mb = tb_full['attention_mask']

        if not s1_ra:
            s1_ra = [[] for _ in ha]; s1_rb = [[] for _ in hb]
            s2_ra = [[] for _ in ha]; s2_rb = [[] for _ in hb]
            s3_ra = [[] for _ in ha]; s3_rb = [[] for _ in hb]

        run_setting1_mean(ha, hb, ma, mb, s1_ra, s1_rb)
        run_setting3_sgpt(ha, hb, ma, mb, s3_ra, s3_rb)
        run_setting2_token(
            ha, hb, off_a, off_b, batch,
            alignments[start:end], s2_ra, s2_rb,
        )

    # ── Stack to arrays ──────────────────────────────────────────────────────
    def stack(reps):
        return [np.vstack(layer) if len(layer) > 0 else np.zeros((0, 1))
                for layer in reps]

    s1_ra, s1_rb = stack(s1_ra), stack(s1_rb)
    s2_ra, s2_rb = stack(s2_ra), stack(s2_rb)
    s3_ra, s3_rb = stack(s3_ra), stack(s3_rb)

    # ── Compute & print CKA ─────────────────────────────────────────────────
    res1 = compute_and_print_cka(s1_ra, s1_rb,
                                  'Setting 1 — Mean pooling',
                                  'cka_mean_matched', 'cka_mean_shuffled')

    if s2_ra[0].shape[0] == 0:
        print('\nSetting 2 — Token level: no aligned pairs found, skipping.')
        res2 = []
    else:
        res2 = compute_and_print_cka(s2_ra, s2_rb,
                                      'Setting 2 — Token level (identity-aligned)',
                                      'cka_token_matched', 'cka_token_shuffled')

    res3 = compute_and_print_cka(s3_ra, s3_rb,
                                  'Setting 3 — SGPT position-weighted',
                                  'cka_sgpt_matched', 'cka_sgpt_shuffled')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    def safe(s):
        return s.replace('/', '_').replace(':', '_')

    out_path = Path(args.results_dir) / (
        f"cka_pythia_{args.lang}_{safe(args.model_a)}_{safe(args.model_b)}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model_a', 'model_b', 'lang', 'setting', 'layer',
                    'cka_matched', 'cka_shuffled'])
        for layer, m, s in res1:
            w.writerow([args.model_a, args.model_b, args.lang, 'mean_pool', layer, m, s])
        for layer, m, s in res2:
            w.writerow([args.model_a, args.model_b, args.lang, 'token_aligned', layer, m, s])
        for layer, m, s in res3:
            w.writerow([args.model_a, args.model_b, args.lang, 'sgpt', layer, m, s])
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
