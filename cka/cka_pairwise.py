#!/usr/bin/env python3
"""Pairwise linear CKA between two Goldfish models on parallel sentences.

Computes per-layer linear CKA between the hidden states of two monolingual
models under three sentence representations (Section 3.1 of the paper):

  mean_pool      attention-masked mean over token hidden states
  token_aligned  word-aligned token vectors (SimAlign / fast_align / exact-match
                 fallback), subwords averaged within each aligned word
  sgpt           SGPT position-weighted mean (Muennighoff, 2022)

For every setting and layer the script reports the *matched* CKA (parallel
sentences) and a *shuffled* control in which one side is permuted with a fixed
seed.  Parallel data is read from ``<data-dir>/<dataset>/<lang_a>-<lang_b>.csv``
(see ``data/download_parallel.py``).  One CSV per dataset is written to
``<results-dir>/cka_<dataset>_<modelA>_<modelB>.csv``.

Usage:
  python cka_pairwise.py --model-a goldfish-models/eng_latn_1000mb \
      --model-b goldfish-models/fra_latn_1000mb \
      --dataset flores tatoeba opus bouquet --num-samples 200 --device cuda
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
    """Uniform attention-masked mean pool.  (batch, seq, dim) -> (batch, dim)"""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def sgpt_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Position-weighted mean pool (SGPT-BE).  (batch, seq, dim) -> (batch, dim)"""
    seq_len = hidden.size(1)
    positions = torch.arange(1, seq_len + 1, device=hidden.device, dtype=hidden.dtype)
    weights = positions.unsqueeze(0) * attention_mask.to(hidden.dtype)
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

def _model_lang(model_name: str) -> str:
    """'goldfish-models/eng_latn_1000mb' -> 'eng_latn' (the CSV column name)."""
    return re.sub(r'_\d+mb$', '', model_name.split('/')[-1])


def load_parallel_sentences(
    dataset_name: str,
    lang_a: str,
    lang_b: str,
    data_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Load parallel sentences from a per-pair CSV at
    <data_dir>/<dataset_name>/<lang_a>-<lang_b>.csv  (or <lang_b>-<lang_a>.csv).
    Each CSV has two columns named by language code (e.g. eng_latn, fra_latn).
    """
    pair_dir = Path(data_dir) / dataset_name
    csv_path = pair_dir / f'{lang_a}-{lang_b}.csv'
    if not csv_path.exists():
        # Try reversed order
        csv_path = pair_dir / f'{lang_b}-{lang_a}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Dataset CSV not found: {pair_dir}/{lang_a}-{lang_b}.csv '
            f'(also tried {lang_b}-{lang_a}.csv)\n'
            f'Run data/download_parallel.py first to create it.'
        )

    df = pd.read_csv(csv_path)
    print(f'Loaded {dataset_name} from {csv_path} ({len(df)} rows)')

    if lang_a not in df.columns:
        raise RuntimeError(f"Column '{lang_a}' not found in {csv_path}. Available: {list(df.columns)}")
    if lang_b not in df.columns:
        raise RuntimeError(f"Column '{lang_b}' not found in {csv_path}. Available: {list(df.columns)}")

    sents_a = df[lang_a].dropna().astype(str).tolist()
    sents_b = df[lang_b].dropna().astype(str).tolist()
    m = min(len(sents_a), len(sents_b))
    return sents_a[:m], sents_b[:m]


# ── Per-setting CKA runners ───────────────────────────────────────────────────

def run_setting1_mean(
    hiddens_a, hiddens_b,
    mask_a: torch.Tensor, mask_b: torch.Tensor,
    reps_a: List[List], reps_b: List[List],
) -> None:
    for li, h in enumerate(hiddens_a):
        reps_a[li].extend(mean_pool(h, mask_a).cpu().numpy())
    for li, h in enumerate(hiddens_b):
        reps_b[li].extend(mean_pool(h, mask_b).cpu().numpy())


def run_setting3_sgpt(
    hiddens_a, hiddens_b,
    mask_a: torch.Tensor, mask_b: torch.Tensor,
    reps_a: List[List], reps_b: List[List],
) -> None:
    for li, h in enumerate(hiddens_a):
        reps_a[li].extend(sgpt_pool(h, mask_a).cpu().numpy())
    for li, h in enumerate(hiddens_b):
        reps_b[li].extend(sgpt_pool(h, mask_b).cpu().numpy())


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
                vec_a = hiddens_a[li][bi].cpu().numpy()[ta_inds].mean(axis=0)
                vec_b = hiddens_b[li][bi].cpu().numpy()[tb_inds].mean(axis=0)
                reps_a[li].append(vec_a)
                reps_b[li].append(vec_b)


def compute_and_print_cka(
    reps_a: List[np.ndarray],
    reps_b: List[np.ndarray],
    setting_name: str,
    col_matched: str,
    col_shuffled: str,
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
        description='Pairwise CKA between two Goldfish models under three pooling settings.'
    )
    parser.add_argument('--model-a', default='goldfish-models/eng_latn_1000mb')
    parser.add_argument('--model-b', default='goldfish-models/fra_latn_1000mb')
    parser.add_argument('--dataset', nargs='+',
                        choices=['flores', 'tatoeba', 'opus', 'bouquet'],
                        default=['flores', 'tatoeba', 'opus', 'bouquet'],
                        help='Which parallel-sentence dataset(s) to use')
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'cka'),
                        help='Directory containing per-pair dataset CSVs '
                             '(<data-dir>/flores/eng_latn-fra_latn.csv, ...)')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).resolve().parent / 'results' / 'main'),
                        help='Where to write cka_<dataset>_<modelA>_<modelB>.csv')
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aligner', choices=['simalign', 'fast_align', 'fallback'],
                        default='simalign',
                        help='Word-alignment backend for Setting 2')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Language codes from model names ────────────────────────────────────────
    token_a = _model_lang(args.model_a)
    token_b = _model_lang(args.model_b)

    # ── Load models once ─────────────────────────────────────────────────────
    print('Loading tokenizers and models...')
    tok_a = AutoTokenizer.from_pretrained(args.model_a, use_fast=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, use_fast=True)
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, output_hidden_states=True)
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, output_hidden_states=True)
    model_a.to(args.device).eval()
    model_b.to(args.device).eval()
    print(f'Model A: {args.model_a}')
    print(f'Model B: {args.model_b}')

    def safe(s: str) -> str:
        return s.replace('/', '_').replace(':', '_')

    # ── Loop over datasets ───────────────────────────────────────────────────
    for dataset_name in args.dataset:
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name.upper()}')
        print(f'{"="*60}')

        try:
            sents_a, sents_b = load_parallel_sentences(
                dataset_name, token_a, token_b, data_dir=args.data_dir,
            )
        except FileNotFoundError as e:
            print(f'SKIP: {e}')
            continue

        n_available = len(sents_a)
        n = min(args.num_samples, n_available)
        print(f'Using {n} sentence pairs (available: {n_available})')

        idx = np.random.choice(n_available, size=n, replace=False)
        sel_a = [sents_a[i] for i in idx]
        sel_b = [sents_b[i] for i in idx]

        # ── Pre-compute word alignments for Setting 2 ────────────────────────
        print(f'Computing word alignments (backend: {args.aligner})...')
        alignments = get_alignments(sel_a, sel_b, method=args.aligner, device=args.device)

        # ── Storage ──────────────────────────────────────────────────────────
        s1_ra: List[List] = []
        s1_rb: List[List] = []
        s2_ra: List[List] = []
        s2_rb: List[List] = []
        s3_ra: List[List] = []
        s3_rb: List[List] = []

        # ── Inference loop (single pass, all settings share hidden states) ───
        print('Running inference...')
        for start in tqdm(range(0, n, args.batch_size)):
            end = min(n, start + args.batch_size)
            batch_a = sel_a[start:end]
            batch_b = sel_b[start:end]

            with torch.no_grad():
                ta_full = tok_a(batch_a, return_tensors='pt', padding=True,
                                truncation=True, return_offsets_mapping=True).to(args.device)
                tb_full = tok_b(batch_b, return_tensors='pt', padding=True,
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
                s1_ra = [[] for _ in ha];  s1_rb = [[] for _ in hb]
                s2_ra = [[] for _ in ha];  s2_rb = [[] for _ in hb]
                s3_ra = [[] for _ in ha];  s3_rb = [[] for _ in hb]

            run_setting1_mean(ha, hb, ma, mb, s1_ra, s1_rb)
            run_setting3_sgpt(ha, hb, ma, mb, s3_ra, s3_rb)
            run_setting2_token(
                ha, hb, off_a, off_b, batch_a, batch_b,
                alignments[start:end], s2_ra, s2_rb,
            )

        # ── Stack to arrays ──────────────────────────────────────────────────
        def stack(reps):
            return [np.vstack(layer) if len(layer) > 0 else np.zeros((0, 1))
                    for layer in reps]

        s1_ra, s1_rb = stack(s1_ra), stack(s1_rb)
        s2_ra, s2_rb = stack(s2_ra), stack(s2_rb)
        s3_ra, s3_rb = stack(s3_ra), stack(s3_rb)

        # ── Compute & print CKA for each setting ────────────────────────────
        res1 = compute_and_print_cka(s1_ra, s1_rb,
                                      'Setting 1 — Mean pooling',
                                      'cka_mean_matched', 'cka_mean_shuffled')

        if s2_ra[0].shape[0] == 0:
            print('\nSetting 2 — Token level: no aligned pairs found, skipping.')
            res2 = []
        else:
            res2 = compute_and_print_cka(s2_ra, s2_rb,
                                          'Setting 2 — Token level (word-aligned)',
                                          'cka_token_matched', 'cka_token_shuffled')

        res3 = compute_and_print_cka(s3_ra, s3_rb,
                                      'Setting 3 — SGPT position-weighted',
                                      'cka_sgpt_matched', 'cka_sgpt_shuffled')

        # ── Save CSV ─────────────────────────────────────────────────────────
        out_path = Path(args.results_dir) / (
            f"cka_{dataset_name}_{safe(args.model_a)}_{safe(args.model_b)}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model_a', 'model_b', 'dataset', 'setting', 'layer',
                        'cka_matched', 'cka_shuffled'])
            for layer, m, s in res1:
                w.writerow([args.model_a, args.model_b, dataset_name, 'mean_pool', layer, m, s])
            for layer, m, s in res2:
                w.writerow([args.model_a, args.model_b, dataset_name, 'token_aligned', layer, m, s])
            for layer, m, s in res3:
                w.writerow([args.model_a, args.model_b, dataset_name, 'sgpt', layer, m, s])
        print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
