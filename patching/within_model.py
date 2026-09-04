#!/usr/bin/env python3
"""Within-model span patching: the ceiling condition of Section 5.

For every ordered pair of facts (donor, target) in one language, run the model
on the donor prompt, cache the residual stream at the donor concept's last
subword at every layer, then run the target prompt with the residual at the
target concept position replaced by the donor's at layers j..L (span patch).
Score the log-probability change of the donor answer at the prompt's last
position.  Following Dumas et al. (2025):

  * the patch lands on the concept word (e.g. the country), not the last token,
    so the injected information has to propagate through attention;
  * the patch spans layers j..L so downstream layers cannot undo it.

Prompts/answers come from facts/<relation>.json (default facts/capital.json,
30 country -> capital facts in 7 languages).  The concept position is found
per language from the verb marker that follows it (LANG_VERB).

Writes results/<run-name>/within_results.csv, one row per (donor, target, j).

Usage:
  python within_model.py --lang deu_latn --n-test 30 --layers 1 2 3 4 5 6 7 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
FACTS_DIR = ROOT / "facts"


# ── Per-language verb markers for finding the country position. The country
# is everything between the start of the prompt and the marker; its last token
# is just before the marker.

LANG_VERB = {
    # Per-language verb markers placed AFTER the country name. List form so a
    # single language can support multiple relations (e.g. capital vs language)
    # without code changes.
    "fra_latn": [" est"],
    "eng_latn": [" is"],
    "deu_latn": [" ist", " befindet", " liegt"],
    "spa_latn": [" es", " se encuentra"],
    "zho_hans": ["的首都是", "的官方语言是", "位于", "所在的国家是",
                 "的作者是", "的化学符号是"],
    "rus_cyrl": [" —"],   # em-dash; Russian copular pattern
    "jpn_jpan": ["の首都は", "の公用語は"],
}


# ── Tokenization helpers ────────────────────────────────────────────────────

def first_subword_id(tok, answer: str, prompt: str) -> int:
    """First BPE id of `answer` as it would extend `prompt`."""
    has_ascii_tail = any(ord(c) < 128 for c in answer[:1])
    joined = prompt + (" " if has_ascii_tail and not prompt.endswith(" ")
                       and not prompt.endswith('"') else "") + answer
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_joined = tok(joined, add_special_tokens=False)["input_ids"]
    n = min(len(ids_prompt), len(ids_joined))
    div = n
    for i in range(n):
        if ids_prompt[i] != ids_joined[i]:
            div = i
            break
    if div >= len(ids_joined):
        return tok(answer, add_special_tokens=False)["input_ids"][0]
    return ids_joined[div]


def logp_at(z: np.ndarray, tid: int) -> float:
    z = z.astype(np.float64); z = z - z.max()
    return float(z[tid] - np.log(np.exp(z).sum()))


def rank_of(z: np.ndarray, tid: int) -> int:
    return int((z > z[tid]).sum())


def get_blocks(model):
    return model.transformer.h


# ── Prompt + position ───────────────────────────────────────────────────────

def build_naked_prompt(fact: Dict, lang: str) -> str:
    """Use facts.json's existing prompt verbatim. Goldfish 1000mb's factual
    knowledge is best surfaced by the plain cloze prompt — few-shot demos
    actually hurt baseline rank in our probes."""
    return fact["prompt"][lang]


def country_last_token_position(tok, prompt: str, lang: str) -> int:
    """0-indexed position of the LAST country-token: tokenize prompt up to (but
    not including) the language-specific verb marker, then take the last index.
    Tries all candidate verbs in LANG_VERB[lang] (longest first to avoid
    matching a prefix when a longer verb is also present)."""
    for verb in sorted(LANG_VERB[lang], key=len, reverse=True):
        idx = prompt.rfind(verb)
        if idx >= 0:
            upto_country_end = prompt[: idx]
            ids = tok(upto_country_end, add_special_tokens=True)["input_ids"]
            return len(ids) - 1
    raise ValueError(f"no verb in {LANG_VERB[lang]!r} found in prompt {prompt!r}")


# ── Hooks ───────────────────────────────────────────────────────────────────

def make_post_replace_hook(replacement_vec: torch.Tensor, position: int):
    """Replace the residual at `position` (single index) with replacement_vec."""
    def hook(_module, _inp, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        x_new = x.clone()
        x_new[:, position, :] = replacement_vec.to(x.dtype).to(x.device)
        if is_tuple:
            return (x_new,) + output[1:]
        return x_new
    return hook


def make_pre_replace_hook(replacement_vec: torch.Tensor, position: int):
    """Pre-hook: replace the residual at `position` of the block's INPUT."""
    def hook(_module, args, kwargs):
        if args:
            x = args[0]
            x_new = x.clone()
            x_new[:, position, :] = replacement_vec.to(x.dtype).to(x.device)
            return (x_new,) + args[1:], kwargs
        kwargs = dict(kwargs)
        x = kwargs["hidden_states"]
        x_new = x.clone()
        x_new[:, position, :] = replacement_vec.to(x.dtype).to(x.device)
        kwargs["hidden_states"] = x_new
        return args, kwargs
    return hook


def register_span_replace_at_position(
    model, source_layer_acts: List[torch.Tensor], j_start: int, position: int,
) -> List:
    """Replace residual at `position` at layer outputs h^(j_start)..h^(L).
    j_start = 0 ALSO replaces h^(0) (embedding output) via a pre-hook on block 0.
    source_layer_acts[k] for k in [0, L] is the source's vector at the source's
    own concept position at hidden_states index k."""
    blocks = get_blocks(model)
    L = len(blocks)
    handles = []
    if j_start == 0:
        handles.append(blocks[0].register_forward_pre_hook(
            make_pre_replace_hook(source_layer_acts[0], position),
            with_kwargs=True,
        ))
    for j in range(max(j_start, 1), L + 1):
        blk_idx = j - 1
        handles.append(blocks[blk_idx].register_forward_hook(
            make_post_replace_hook(source_layer_acts[j], position)
        ))
    return handles


# ── Forward passes ──────────────────────────────────────────────────────────

@torch.no_grad()
def forward_collect(
    model, tok, prompt: str, position_of_interest: int, device: str,
    max_length: int = 512,
) -> Tuple[List[torch.Tensor], np.ndarray]:
    """Returns (per-layer last-token-of-prompt logits, list of vecs at
    `position_of_interest` at layers 0..L).
    But we want last-token logits AND the vec at concept position. So return
    (concept_position_states[0..L], final_logits_at_last_token[-1])."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length,
              add_special_tokens=True).to(device)
    out = model(**enc, output_hidden_states=True)
    states = [h[0, position_of_interest, :].detach().clone()
              for h in out.hidden_states]
    last_logits = out.logits[0, -1].detach().cpu().numpy().astype(np.float32)
    return states, last_logits


@torch.no_grad()
def forward_patched(
    model, tok, target_prompt: str, target_position: int,
    source_layer_acts: List[torch.Tensor], j_start: int, device: str,
    max_length: int = 512,
) -> np.ndarray:
    handles = register_span_replace_at_position(
        model, source_layer_acts, j_start, target_position,
    )
    try:
        enc = tok(target_prompt, return_tensors="pt", truncation=True,
                  max_length=max_length, add_special_tokens=True).to(device)
        out = model(**enc)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, -1].detach().cpu().numpy().astype(np.float32)


# ── Main experiment ──────────────────────────────────────────────────────────

def get_capital(fact: Dict, lang: str) -> str:
    return fact["answer"][lang]


def run(args) -> pd.DataFrame:
    facts_all = json.loads(args.facts_file.read_text())
    facts = [f for f in facts_all if args.lang in f["prompt"]]
    by_id = {f["id"]: f for f in facts}
    print(f"loaded {len(facts)} facts; lang={args.lang}")

    test_facts = facts[: args.n_test]
    print(f"test ({len(test_facts)}): {[(f['id'], get_capital(f, args.lang)) for f in test_facts]}")

    model_id = f"goldfish-models/{args.lang}_{args.size}"
    print(f"\nloading {model_id} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(args.device).eval()
    L = len(get_blocks(model))
    print(f"model: {L} blocks, hidden={model.config.n_embd}")

    fact_prompt: Dict[str, str] = {}
    fact_pos:    Dict[str, int] = {}
    fact_states: Dict[str, List[torch.Tensor]] = {}
    fact_base:   Dict[str, np.ndarray] = {}
    fact_capital_tid: Dict[str, int] = {}

    print("\n=== Baseline (no patch) ===")
    print(f"{'id':<14} {'cap':<14} {'pos':>4} {'rank':>5} {'logp':>8} {'top1':>14}")
    for f in test_facts:
        capital = get_capital(f, args.lang)
        prompt = build_naked_prompt(f, args.lang)
        pos = country_last_token_position(tok, prompt, args.lang)
        states, base = forward_collect(model, tok, prompt, pos, args.device, args.max_length)
        cap_tid = first_subword_id(tok, capital, prompt)
        fact_prompt[f["id"]] = prompt
        fact_pos[f["id"]] = pos
        fact_states[f["id"]] = states
        fact_base[f["id"]] = base
        fact_capital_tid[f["id"]] = cap_tid
        argmax_id = int(base.argmax())
        print(f"{f['id']:<14} {capital:<14} {pos:>4d} {rank_of(base, cap_tid):>5d} "
              f"{logp_at(base, cap_tid):>8.3f} {tok.decode([argmax_id])!r:>14}")

    if args.baseline_only:
        return pd.DataFrame()

    layers = ([int(x) for x in args.layers]
              if args.layers != ["all"] else list(range(1, L + 1)))
    print(f"\nSpan-patch start layers j ∈ {layers} (each patches j..L "
          f"at the COUNTRY position)")

    rows: List[Dict] = []
    for ti, target in enumerate(test_facts):
        t_id = target["id"]
        t_pos = fact_pos[t_id]
        z_base = fact_base[t_id]
        t_cap_tid = fact_capital_tid[t_id]
        for di, donor in enumerate(test_facts):
            if di == ti:
                continue
            d_id = donor["id"]
            d_cap_in_target_tid = first_subword_id(
                tok, get_capital(donor, args.lang), fact_prompt[t_id],
            )
            for j in layers:
                z_p = forward_patched(
                    model, tok, fact_prompt[t_id], t_pos,
                    fact_states[d_id], j, args.device, args.max_length,
                )
                rows.append({
                    "donor_id": d_id, "target_id": t_id, "j_start": j,
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
                    "argmax_patch_str": tok.decode([int(z_p.argmax())]),
                    "flip_to_donor": int(int(z_p.argmax()) == d_cap_in_target_tid),
                })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="fra_latn")
    p.add_argument("--size", default="1000mb")
    p.add_argument("--facts-file", type=Path, default=FACTS_DIR / "capital.json")
    p.add_argument("--n-test", type=int, default=30, help="Number of facts to use")
    p.add_argument("--layers", nargs="+", default=["1", "2", "3", "4", "5", "6", "7", "8"],
                   help="Span start layers j (each patches j..L)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--baseline-only", action="store_true",
                   help="Just check the baseline forward passes; skip patching.")
    p.add_argument("--run-name", default=None, help="Default: w_<lang>")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    if args.run_name is None:
        args.run_name = f"w_{args.lang}"
    out_dir = args.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {args.run_name} -> {out_dir}\n")

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = run(args)
    if df.empty:
        return
    csv_path = out_dir / "within_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows)")

    print("\n=== Summary by j_start (cross-fact pairs) ===")
    g = df.groupby("j_start").agg(
        n=("donor_id", "size"),
        mean_dl_donor=("delta_logp_donor", "mean"),
        mean_dl_target=("delta_logp_target", "mean"),
        flip_rate=("flip_to_donor", "mean"),
    ).round(3)
    g["donor_beats_target"] = (
        df.assign(diff=df["delta_logp_donor"] - df["delta_logp_target"])
          .groupby("j_start")["diff"].apply(lambda s: (s > 0).mean()).round(3)
    )
    print(g)


if __name__ == "__main__":
    main()
