#!/bin/bash
# Section 4: fit Procrustes / Affine (ridge) / MLP for all 36 pairs and 13
# layers, then regenerate Tables 4/23 and Figures 12, 13, 17.
# Requires ../data/projection (see README).  Set PAIRS="eng_latn-fra_latn ..."
# to run a subset (pairs are independent).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
RUN_NAME=${RUN_NAME:-all_pairs}
PAIRS_ARG=""; [ -n "${PAIRS:-}" ] && PAIRS_ARG="--pairs $PAIRS"

$PYTHON fit_projections.py --size 1000mb --datasets flores opus tatoeba bouquet \
    --num-samples 0 --min-len-chars 50 --epochs 200 --quotas-file quotas.json \
    --run-name "$RUN_NAME" $PAIRS_ARG --device cuda

$PYTHON plot_method_comparison.py --run-name "$RUN_NAME"
$PYTHON plot_procrustes_by_layer.py --run-name "$RUN_NAME"
$PYTHON plot_methods_x_metrics.py --run-name "$RUN_NAME"
