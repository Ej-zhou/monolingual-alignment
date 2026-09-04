# Cross-Lingual Alignment Without Joint Training

Code and data for **"Cross-Lingual Alignment Without Joint Training: Do
Monolingual Language Models Converge on Universal Representations?"**
(Zhou, Salhan, Arnett, Korhonen — EMNLP 2026).

The paper asks whether strictly monolingual language models — the
[Goldfish](https://huggingface.co/goldfish-models) family and five
independently developed ~1B models — learn alignable representations without
any joint training, in three steps: **correlation** (CKA between models on
parallel sentences), **construction** (a single Procrustes rotation maps one
model's hidden states onto another's) and **causation** (patching the rotated
English residual into a German model flips its factual prediction).

Every figure and table in the paper maps to one script below.  The result
CSVs that the paper's figures/tables were generated from are included, so all
plots and tables can be regenerated **without a GPU**; the experiments
themselves need one GPU and the HuggingFace models/datasets.

## Layout

| Directory | Paper | What it does |
|---|---|---|
| `data/` | §3.1, §4.1 | `download_parallel.py`: FLORES-200, OPUS-100, Tatoeba, BouQUET → one CSV per language pair |
| `cka/` | §3.1–3.3, App. A–B | pairwise CKA between Goldfish models under three pooling strategies; 9×9 matrices, size scaling, cross-size |
| `cross_architecture/` | §3.4, App. C | Goldfish vs Pythia; five ~1B monolingual models from different labs |
| `predictors/` | §3.2, App. D | URIEL typological distance and model NLL as predictors of CKA |
| `low_resource/` | §3.2 (Table 2) | English vs Tagalog / Swahili / Uzbek / Amharic |
| `projection/` | §4, App. F | Procrustes / Affine / MLP maps between all 36 pairs at every layer; residual SVD |
| `patching/` | §5, App. G | cross-model span patching of country→capital (and six other relations) |

Conventions shared by all scripts:

* Models are `goldfish-models/<lang>_<size>` with Goldfish language codes
  (`eng_latn`, `zho_hans`, `spa_latn`, `arb_arab`, `hin_deva`, `fra_latn`,
  `rus_cyrl`, `deu_latn`, `jpn_jpan`) and `size ∈ {5mb, 10mb, 100mb, 1000mb}`.
  The nine languages are always ordered as listed.
* Hidden states use HuggingFace `output_hidden_states`: index 0 is the
  embedding output, index k ≥ 1 the output of block k−1 (13 layers for the
  1000 MB models).
* Parallel data lives in `data/<corpus>/<dataset>/<lang_a>-<lang_b>.csv` with
  two columns named by language code.
* Seed 42 everywhere.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login      # BouQUET and facebook/flores are gated; accept their terms on the Hub
```

The experiments were run with Python 3.10–3.11, PyTorch 2.x and `transformers` 4.35–4.44 (Goldfish
checkpoints do not load with `transformers` 5).  The word-aligned CKA setting
needs `simalign` (which downloads `bert-base-multilingual-cased`).

## Data

Three corpora are used, produced by the same script with different settings:

```bash
# Sections 3.1-3.4 and Appendices A-D (CKA): 9 languages, up to 1000 sentence pairs per dataset,
# FLORES devtest only.  Add the extra languages of §3.4 with a FLORES-only call.
python data/download_parallel.py --out-dir data/cka --flores-splits devtest --max-instances 1000 --opus-split test
python data/download_parallel.py --out-dir data/cka --datasets flores --flores-splits devtest \
    --languages eng_latn zho_hans por_latn pol_latn ita_latn

# Table 2 (low-resource languages), FLORES dev+devtest
python data/download_parallel.py --out-dir data/lowres --datasets flores --flores-splits dev devtest \
    --languages eng_latn fra_latn zho_hans swa_latn tgl_latn uzb_latn amh_ethi

# Sections 4-5 (projection fitting, patching): FLORES dev+devtest, OPUS/Tatoeba capped at 50k
python data/download_parallel.py --out-dir data/projection --flores-splits dev devtest --max-instances 50000
```

FLORES is read from the open `openlanguagedata/flores_plus` (falls back to the
gated `facebook/flores`).  For Sections 4–5 the per-pair OPUS/Tatoeba caps
actually used for fitting are in `projection/quotas.json`; clean datasets
(FLORES, BouQUET) are always used in full, noisy ones are filtered to ≥ 50
characters on both sides.

## Reproducing the paper

All commands run from inside the directory named in the first column.  "GPU"
marks the experiment step; the plot/table step reads `results/` and works on
the shipped files as-is.

### Section 3 — Representational alignment (`cka/`)

| Result | GPU step | Plot / table step | Output |
|---|---|---|---|
| Fig. 2 (eng–fra by layer) | `bash run_all_pairs.sh 1000mb results/main` | `python plot_eng_fra_layers.py` | `figures/eng_fra_1000mb_combined.pdf` |
| Fig. 3 (9×9 heatmap) | same | `python plot_heatmap_singlecol.py` | `figures/cka_singlecol_meanpool_1000mb.pdf` |
| Table 1, Tables 11–22 | same | `python make_tables.py` | `tables/table1_summary.md`, `tables/table_<setting>_<dataset>.tex` |
| Fig. 4 (size scaling) | `bash run_all_pairs.sh <size> results/size_scaling flores` for each size | `python plot_size_scaling.py` | `figures/cka_size_scaling.pdf` |
| Figs. 14–16 (9×9 at all sizes) | same | `python plot_heatmaps_all_sizes.py` | `figures/cka_heatmaps_<setting>_all_sizes.pdf` |
| Figs. 7–9 (cross-size) | `bash run_cross_size.sh 1000mb 100mb` and `10mb 5mb` | `python plot_cross_size_heatmaps.py` | `figures/cka_cross_size_<setting>.pdf` |

`cka_pairwise.py` computes one model pair (all three pooling settings, every
layer, matched + shuffled) and writes `results/<dir>/cka_<dataset>_<A>_<B>.csv`;
the shell scripts loop over pairs and accept `PAIR_INDEX` for array jobs.

### Section 3.4 — Other architectures (`cross_architecture/`)

| Result | GPU step | Plot / table step | Output |
|---|---|---|---|
| Fig. 10, Table 5 (Goldfish vs Pythia) | `bash run_goldfish_vs_pythia.sh` | `python plot_goldfish_vs_pythia.py` | `figures/cka_pythia_layers_setting3_sgpt.pdf` |
| Table 3, Table 6 (five ~1B models) | `bash run_diverse_models.sh` | `python table_diverse_models.py` | `tables/diverse_models_last.csv` |

### Appendix D — Predictors of alignment (`predictors/`)

| Result | GPU step | Plot / table step | Output |
|---|---|---|---|
| Table 7 | `bash run_nll.sh` (per-language FLORES NLL) | `python analyze_predictors.py` | `results/correlation_summary.csv` |
| Fig. 11 | — | `python plot_syntactic_distance.py` | `figures/cka_vs_syntactic_distance.pdf` |
| joint regression (§3.2) | — | `python regression_uriel_nll.py` | stdout |

URIEL distances (lang2vec cosine distances: syntactic, genetic, geographic,
inventory) are hard-coded in `analyze_predictors.py`.  CKA values are read from
`../cka/results/main`.

### Table 2 — Low-resource languages (`low_resource/`)

| GPU step | Table step | Output |
|---|---|---|
| `python run_lowres_cka.py --device cuda` | `python make_table.py` | `tables/lowres_cka.md` |

### Section 4 — Reconstruction (`projection/`)

| Result | GPU step | Plot / table step | Output |
|---|---|---|---|
| Table 4 / Table 23, Fig. 12 | `bash run_all_pairs.sh` (36 pairs × 13 layers × 3 methods) | `python plot_method_comparison.py` | `figures/table_method_comparison.{md,tex}`, `figures/figure_4_procrustes_vs_affine.pdf` |
| Fig. 13 (P@1 by layer) | same | `python plot_procrustes_by_layer.py` | `figures/exp6_procrustes_p_at_1_pool_by_layer.pdf` |
| Fig. 17 (methods × metrics) | same | `python plot_methods_x_metrics.py` | `figures/figure_a_methods_x_metrics.pdf` |
| Fig. 5, Fig. 18 (PCA before/after) | `bash run_pca_figures.sh` (refits eng→fra, saves test predictions) | included in the script | `figures/pca_projection_main_eng_latn-fra_latn_procrustes_L12.pdf`, `figures/pca_projection_eng_latn-fra_latn_procrustes.pdf` |
| Table 9 (residual SVD) | `bash run_residual_svd.sh` | `python table_residual_svd.py --layer 8` | `tables/residual_svd_layer8.csv` |

`fit_projections.py` writes `results/<run>/projection_results.csv` with one row
per (pair, layer, method): test MSE, post-projection CKA, P@1/5/10 against the
test pool (`p_at_k`) and against the train+test pool (`p_at_k_pool`, "hard").
"Affine" in the paper is the `ridge` method (λ chosen on a validation slice).
The predictions needed for the PCA figures are ~350 MB and are not included;
`run_pca_figures.sh` regenerates them.

### Section 5 — Cross-model activation patching (`patching/`)

| Result | GPU step | Plot / table step | Output |
|---|---|---|---|
| Fig. 6 | `bash run_within.sh <tgt>` and `bash run_cross.sh eng_latn <tgt>` for tgt ∈ fra, deu, spa, jpn, zho | `python plot_summary.py` | `figures/paper_summary_eng_to_x_capital.pdf` |
| Table 10 (relation sweep, eng→deu) | `bash run_within.sh deu_latn <rel>` and `bash run_cross.sh eng_latn deu_latn <rel>` for rel ∈ city_to_country, continent, landmark, author, composer, inventor | `python table_relations.py --j 2` | `tables/relations.md` |

`facts/<relation>.json` holds the prompts and answers (30 country→capital facts
in seven languages; the other relations in English + German and, for some,
French/Spanish/Chinese).  `within_model.py` is the within-model ceiling,
`cross_model.py` fits per-layer Procrustes/Affine maps on last-token activations
of parallel sentences and patches the projected English residual into the
target model under four conditions (`procrustes`, `affine`, `unprojected`,
`shuffle`).  Both write one row per (donor, target, condition, span start j).

## Shipped results

| Path | Contents |
|---|---|
| `cka/results/main/` | 1000 MB models, 36 pairs × {FLORES, Tatoeba, OPUS, BouQUET} (OPUS/Tatoeba only where the pair exists) |
| `cka/results/size_scaling/` | FLORES only: 36 pairs at 5/10/100/1000 MB and 81 cross-size pairs for 1000 vs 100 MB and 10 vs 5 MB |
| `cross_architecture/results/` | Goldfish vs Pythia (160M–1.4B) for English and Chinese; the 10 pairs of the ~1B models |
| `predictors/results/` | per-language NLL, per-pair predictor table, correlation summary |
| `low_resource/results/` | eng–{tgl, swh, uzn, amh, fra, zho} at 1000 MB (+ Yoruba at 100 MB) |
| `projection/results/all_pairs/` | 36 pairs × 13 layers × all methods; `residual_svd/` the four headline pairs |
| `patching/results/` | within-model ceilings for 7 languages and 11 cross-model pairs (gzipped CSVs) |

## Notes on replication

* **Sampling.** CKA experiments draw 200 sentence pairs per dataset with seed 42
  from the downloaded CSVs, so re-downloading data (different FLORES mirror or
  OPUS split order) changes the sample and moves values by a few hundredths.
  The size-scaling runs (`cka/results/size_scaling`) were produced by an earlier
  version of the driver that read FLORES devtest directly from the Hub; the
  1000 MB FLORES entries there therefore differ slightly from `results/main`.
* **Table 10.** The relation-sweep runs (`w_deu_latn_<rel>`, `x_eng_latn-deu_latn_<rel>`)
  were run on a separate cluster and are not included; `table_relations.py`
  reports the capital relation from the shipped results and skips missing ones.
* **Compute.** One pair of Goldfish 1000 MB models fits on a single 16 GB GPU;
  a CKA pair takes ~10–30 min, the full 36-pair projection run a few hours,
  one cross-model patching pair ~30 min.  The shell scripts are plain bash so
  they can be wrapped in any scheduler; each pair is independent.
* **Not included.** Figure 1 (hand-drawn schematic) and the GlotLID-v3 training
  corpus audit of Appendix E (Table 8), which was run outside this codebase.
