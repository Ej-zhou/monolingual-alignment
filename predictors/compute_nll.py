#!/usr/bin/env python3
"""
compute_nll.py

Mean per-sentence NLL of each Goldfish 1000 MB model on its own FLORES
sentences (the model-quality predictor of Appendix D).  Appends one row per
language to results/nll_per_lang.csv.

Usage:
  python compute_nll.py --lang eng_latn --device cuda
  python compute_nll.py --lang zho_hans --device cuda
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LANG_TO_MODEL = {
    'eng_latn': 'goldfish-models/eng_latn_1000mb',
    'arb_arab': 'goldfish-models/arb_arab_1000mb',
    'deu_latn': 'goldfish-models/deu_latn_1000mb',
    'fra_latn': 'goldfish-models/fra_latn_1000mb',
    'hin_deva': 'goldfish-models/hin_deva_1000mb',
    'jpn_jpan': 'goldfish-models/jpn_jpan_1000mb',
    'rus_cyrl': 'goldfish-models/rus_cyrl_1000mb',
    'spa_latn': 'goldfish-models/spa_latn_1000mb',
    'zho_hans': 'goldfish-models/zho_hans_1000mb',
}


def load_flores_sentences(lang: str, data_dir: str) -> list[str]:
    """Load FLORES sentences for a language from any available pair CSV."""
    flores_dir = Path(data_dir) / 'flores'
    for csv_path in flores_dir.glob('*.csv'):
        df = pd.read_csv(csv_path)
        if lang in df.columns:
            sents = df[lang].dropna().astype(str).tolist()
            if sents:
                print(f'Loaded {len(sents)} sentences for {lang} from {csv_path.name}')
                return sents
    raise FileNotFoundError(f'No FLORES CSV with column {lang} in {flores_dir}')


def compute_nll(model, tokenizer, sentences, device, batch_size=8):
    """Compute per-sentence NLL (mean over tokens)."""
    nlls = []
    for start in tqdm(range(0, len(sentences), batch_size), desc='NLL'):
        batch = sentences[start:start + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc['input_ids'])
            # out.loss is mean over all non-pad tokens in the batch
            # We want per-sentence NLL, so compute manually
            logits = out.logits[:, :-1, :]  # (B, T-1, V)
            labels = enc['input_ids'][:, 1:]  # (B, T-1)
            mask = enc['attention_mask'][:, 1:]  # (B, T-1)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            # mask out padding
            token_log_probs = token_log_probs * mask.float()

            # per-sentence mean NLL
            sent_nll = -(token_log_probs.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1))
            nlls.extend(sent_nll.cpu().tolist())
    return nlls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True, choices=list(LANG_TO_MODEL.keys()))
    parser.add_argument('--data-dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'cka'),
                        help='Directory containing <data-dir>/flores/*.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_id = LANG_TO_MODEL[args.lang]
    print(f'Loading {model_id} ...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(args.device).eval()

    sentences = load_flores_sentences(args.lang, args.data_dir)
    nlls = compute_nll(model, tokenizer, sentences, args.device, args.batch_size)

    mean_nll = float(np.mean(nlls))
    print(f'{args.lang}: mean NLL = {mean_nll:.4f} over {len(nlls)} sentences')

    # Save
    out_dir = Path(__file__).resolve().parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'nll_per_lang.csv'

    # Append or create
    rows = {}
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[row['lang']] = row

    rows[args.lang] = {
        'lang': args.lang,
        'model': model_id,
        'mean_nll': f'{mean_nll:.6f}',
        'n_sentences': str(len(nlls)),
    }

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lang', 'model', 'mean_nll', 'n_sentences'])
        writer.writeheader()
        for lang in sorted(rows):
            writer.writerow(rows[lang])

    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
