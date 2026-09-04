#!/usr/bin/env python3
"""Diverse independently developed ~1B monolingual models (Section 3.4, Table 3).

Last-layer (and first-layer) SGPT-pooled linear CKA between two models from
different labs / languages / architectures on FLORES parallel sentences, with a
shuffled control.  Models used in the paper:

  EleutherAI/pythia-1.4b            eng_latn
  SJTU-CL/Zh-Pythia-1.4B            zho_hans
  TucanoBR/Tucano-1b1               por_latn
  speakleash/Bielik-1.5B-v3         pol_latn
  sapienzanlp/Minerva-1B-base-v1.0  ita_latn

Usage:
  python diverse_models.py --model-a EleutherAI/pythia-1.4b --lang-a eng_latn \
      --model-b TucanoBR/Tucano-1b1 --lang-b por_latn --device cuda
"""
import argparse
import csv
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
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


# ── Token-level alignment helpers ────────────────────────────────────────────

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


def _fallback_align(sent_a: str, sent_b: str) -> List[Tuple[int, int]]:
    wa = [w for _, _, w in word_spans(sent_a)]
    wb = [w for _, _, w in word_spans(sent_b)]
    b_index: dict = {}
    for j, w in enumerate(wb):
        b_index.setdefault(w.lower(), []).append(j)
    return [(i, b_index[w.lower()][0])
            for i, w in enumerate(wa) if w.lower() in b_index]


def get_alignments(
    sentences_a: List[str],
    sentences_b: List[str],
    method: str = 'simalign',
    device: str = 'cpu',
) -> List[List[Tuple[int, int]]]:
    if method == 'simalign':
        try:
            from simalign import SentenceAligner
            try:
                aligner = SentenceAligner(model='bert', token_type='bpe',
                                          matching_methods=['cosine'], device=device)
            except KeyError:
                aligner = SentenceAligner(model='bert', token_type='bpe', device=device)
            print(f"[SentenceAligner] running on: {device}")
            pairs = []
            for a, b in zip(sentences_a, sentences_b):
                try:
                    r = aligner.get_word_aligns(a, b)
                    if isinstance(r, dict) and 'inter' in r:
                        pairs.append([(int(x), int(y)) for x, y in r['inter']])
                    else:
                        pairs.append(list(r))
                except Exception:
                    pairs.append(_fallback_align(a, b))
            return pairs
        except Exception:
            pass

    if method == 'fast_align':
        try:
            with tempfile.TemporaryDirectory() as td:
                in_file = td + '/parallel.txt'
                with open(in_file, 'w', encoding='utf8') as f:
                    for a, b in zip(sentences_a, sentences_b):
                        f.write(a.replace('\n', ' ') + ' ||| ' +
                                b.replace('\n', ' ') + '\n')
                p = subprocess.run(
                    ['fast_align', '-i', in_file, '-d', '-o', '-v'],
                    capture_output=True, text=True,
                )
                if p.returncode != 0:
                    raise RuntimeError('fast_align failed')
                pairs = []
                for line in p.stdout.strip().splitlines():
                    pairs.append([
                        (int(a), int(b))
                        for item in line.strip().split()
                        if '-' in item
                        for a, b in [item.split('-')]
                    ])
                return pairs
        except Exception:
            pass

    return [_fallback_align(a, b) for a, b in zip(sentences_a, sentences_b)]


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_parallel_sentences(
    lang_a: str,
    lang_b: str,
    data_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Load parallel sentences from FLORES CSV at
    <data_dir>/flores/<lang_a>-<lang_b>.csv  (or reversed).
    """
    pair_dir = Path(data_dir) / 'flores'
    csv_path = pair_dir / f'{lang_a}-{lang_b}.csv'
    if not csv_path.exists():
        csv_path = pair_dir / f'{lang_b}-{lang_a}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'FLORES CSV not found: {pair_dir}/{lang_a}-{lang_b}.csv '
            f'(also tried reversed). Run data/download_parallel.py first.'
        )

    df = pd.read_csv(csv_path)
    print(f'Loaded FLORES from {csv_path} ({len(df)} rows)')

    if lang_a not in df.columns:
        raise RuntimeError(f"Column '{lang_a}' not in {csv_path}. Available: {list(df.columns)}")
    if lang_b not in df.columns:
        raise RuntimeError(f"Column '{lang_b}' not in {csv_path}. Available: {list(df.columns)}")

    sents_a = df[lang_a].dropna().astype(str).tolist()
    sents_b = df[lang_b].dropna().astype(str).tolist()
    m = min(len(sents_a), len(sents_b))
    return sents_a[:m], sents_b[:m]


# ── Per-setting CKA runners ───────────────────────────────────────────────────

def run_setting1_mean(hiddens_a, hiddens_b, mask_a, mask_b, reps_a, reps_b):
    for li, h in enumerate(hiddens_a):
        reps_a[li].extend(mean_pool(h, mask_a).cpu().numpy())
    for li, h in enumerate(hiddens_b):
        reps_b[li].extend(mean_pool(h, mask_b).cpu().numpy())


def run_setting3_sgpt(hiddens_a, hiddens_b, mask_a, mask_b, reps_a, reps_b):
    # Only first and last layer from each model
    for idx, h in enumerate([hiddens_a[0], hiddens_a[-1]]):
        reps_a[idx].extend(sgpt_pool(h, mask_a).cpu().numpy())
    for idx, h in enumerate([hiddens_b[0], hiddens_b[-1]]):
        reps_b[idx].extend(sgpt_pool(h, mask_b).cpu().numpy())


def run_setting2_token(
    hiddens_a, hiddens_b,
    off_map_a, off_map_b,
    batch_sents_a: List[str], batch_sents_b: List[str],
    alignments: List[List[Tuple[int, int]]],
    reps_a: List[List], reps_b: List[List],
) -> None:
    for bi in range(len(batch_sents_a)):
        wspan_a = word_spans(batch_sents_a[bi])
        wspan_b = word_spans(batch_sents_b[bi])
        off_a = off_map_a[bi].cpu().tolist()
        off_b = off_map_b[bi].cpu().tolist()
        w2t_a = map_word_to_token_indices(off_a, wspan_a)
        w2t_b = map_word_to_token_indices(off_b, wspan_b)

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
) -> List[Tuple[str, float, float]]:
    layer_names = ['first', 'last']
    n_samples = min(reps_a[0].shape[0], reps_b[0].shape[0])
    perm = np.random.permutation(n_samples)
    print(f'\n=== {setting_name} ({n_samples} samples) ===')
    print(f'Layer, {col_matched}, {col_shuffled}')
    results = []
    for i, name in enumerate(layer_names):
        X = reps_a[i][:n_samples]
        Y = reps_b[i][:n_samples]
        matched = linear_CKA(X, Y)
        shuffled = linear_CKA(X, Y[perm])
        print(f'{name}, {matched:.6f}, {shuffled:.6f}')
        results.append((name, matched, shuffled))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pairwise SGPT CKA between two independently developed monolingual models.'
    )
    parser.add_argument('--model-a', required=True,
                        help='First model HuggingFace ID (e.g. EleutherAI/pythia-1.4b)')
    parser.add_argument('--model-b', required=True,
                        help='Second model HuggingFace ID (e.g. TucanoBR/Tucano-1b1)')
    parser.add_argument('--lang-a', required=True,
                        help='FLORES language code for model A (e.g. eng_latn)')
    parser.add_argument('--lang-b', required=True,
                        help='FLORES language code for model B (e.g. por_latn)')
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'cka'),
                        help='Directory containing <data-dir>/flores/<a>-<b>.csv')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).resolve().parent / 'results'))
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load parallel sentences ──────────────────────────────────────────────
    sents_a, sents_b = load_parallel_sentences(
        args.lang_a, args.lang_b, data_dir=args.data_dir,
    )

    n_available = len(sents_a)
    n = min(args.num_samples, n_available)
    print(f'Using {n} sentence pairs (available: {n_available})')

    idx = np.random.choice(n_available, size=n, replace=False)
    sel_a = [sents_a[i] for i in idx]
    sel_b = [sents_b[i] for i in idx]

    # ── Load models ─────────────────────────────────────────────────────────
    print('Loading tokenizers and models...')
    tok_a = AutoTokenizer.from_pretrained(args.model_a, use_fast=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, use_fast=True)
    # Many causal LM tokenizers lack a pad token; use eos_token as fallback
    if tok_a.pad_token is None:
        tok_a.pad_token = tok_a.eos_token
    if tok_b.pad_token is None:
        tok_b.pad_token = tok_b.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, output_hidden_states=True)
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, output_hidden_states=True)
    model_a.to(args.device).eval()
    model_b.to(args.device).eval()
    print(f'Model A: {args.model_a} [{args.lang_a}] ({sum(p.numel() for p in model_a.parameters())/1e6:.1f}M params)')
    print(f'Model B: {args.model_b} [{args.lang_b}] ({sum(p.numel() for p in model_b.parameters())/1e6:.1f}M params)')

    # ── Storage ──────────────────────────────────────────────────────────────
    s3_ra, s3_rb = [], []

    # ── Inference loop ──────────────────────────────────────────────────────
    print('Running inference...')
    for start in tqdm(range(0, n, args.batch_size)):
        end = min(n, start + args.batch_size)
        batch_a = sel_a[start:end]
        batch_b = sel_b[start:end]

        with torch.no_grad():
            ta_full = tok_a(batch_a, return_tensors='pt', padding=True,
                            truncation=True).to(args.device)
            tb_full = tok_b(batch_b, return_tensors='pt', padding=True,
                            truncation=True).to(args.device)
            out_a = model_a(**ta_full)
            out_b = model_b(**tb_full)

        ha = out_a.hidden_states
        hb = out_b.hidden_states
        ma = ta_full['attention_mask']
        mb = tb_full['attention_mask']

        if not s3_ra:
            # Only 2 slots: first layer and last layer
            s3_ra = [[], []]; s3_rb = [[], []]

        run_setting3_sgpt(ha, hb, ma, mb, s3_ra, s3_rb)

    # ── Stack to arrays ──────────────────────────────────────────────────────
    def stack(reps):
        return [np.vstack(layer) if len(layer) > 0 else np.zeros((0, 1))
                for layer in reps]

    s3_ra, s3_rb = stack(s3_ra), stack(s3_rb)

    # ── Compute & print CKA ─────────────────────────────────────────────────
    res3 = compute_and_print_cka(s3_ra, s3_rb,
                                  'SGPT position-weighted',
                                  'cka_sgpt_matched', 'cka_sgpt_shuffled')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    def safe(s):
        return s.replace('/', '_').replace(':', '_')

    out_path = Path(args.results_dir) / f"cka_diverse_{safe(args.model_a)}_{safe(args.model_b)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model_a', 'model_b', 'lang_a', 'lang_b', 'setting', 'layer',
                    'cka_matched', 'cka_shuffled'])
        for layer, m, s in res3:
            w.writerow([args.model_a, args.model_b, args.lang_a, args.lang_b,
                        'sgpt', layer, m, s])
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
