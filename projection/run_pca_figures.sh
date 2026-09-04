#!/bin/bash
# Figures 5 and 18: refit eng->fra with per-layer test predictions saved, then
# draw the before/after PCA panels.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
$PYTHON fit_projections.py --pairs eng_latn-fra_latn --save-predictions-for eng_latn-fra_latn \
    --size 1000mb --datasets flores opus tatoeba bouquet --num-samples 0 --min-len-chars 50 \
    --epochs 200 --quotas-file quotas.json --run-name eng_fra_pca --device cuda
$PYTHON plot_pca_main.py --run-name eng_fra_pca --pair eng_latn-fra_latn --method procrustes --layer 12
$PYTHON plot_pca_layers.py --run-name eng_fra_pca --pair eng_latn-fra_latn --methods procrustes
