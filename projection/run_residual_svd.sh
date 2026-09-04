#!/bin/bash
# Appendix F.2 / Table 9: SVD of the Procrustes residual for the four headline
# pairs (reuses the embedding cache of fit_projections.py).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
$PYTHON residual_svd.py --size 1000mb --datasets flores opus tatoeba bouquet \
    --num-samples 0 --min-len-chars 50 --quotas-file quotas.json \
    --run-name residual_svd --device cuda
$PYTHON table_residual_svd.py --run-name residual_svd --layer 8
