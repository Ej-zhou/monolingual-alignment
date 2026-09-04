#!/bin/bash
# Cross-model span patching with Procrustes / Affine / unprojected / shuffle
# (Section 5).  Needs ../data/projection (fit corpus) and the Goldfish models.
#   bash run_cross.sh eng_latn deu_latn             # results/x_eng_latn-deu_latn
#   bash run_cross.sh eng_latn deu_latn landmark    # results/x_eng_latn-deu_latn_landmark
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
SRC=${1:-eng_latn}
TGT=${2:-fra_latn}
REL=${3:-capital}
SUFFIX=""; [ "$REL" != "capital" ] && SUFFIX="_$REL"
$PYTHON cross_model.py --source-lang "$SRC" --target-lang "$TGT" --facts-file "facts/$REL.json" \
    --n-test 30 --layers 1 2 3 4 5 6 7 8 --conditions procrustes affine unprojected shuffle \
    --run-name "x_${SRC}-${TGT}${SUFFIX}" --device cuda
