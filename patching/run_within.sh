#!/bin/bash
# Within-model ceiling (Section 5): the target model patches its own donor
# representation.
#   bash run_within.sh deu_latn             # country->capital, results/w_deu_latn
#   bash run_within.sh deu_latn landmark    # facts/landmark.json, results/w_deu_latn_landmark
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
LANG_CODE=${1:-fra_latn}
REL=${2:-capital}
SUFFIX=""; [ "$REL" != "capital" ] && SUFFIX="_$REL"
$PYTHON within_model.py --lang "$LANG_CODE" --facts-file "facts/$REL.json" \
    --n-test 30 --layers 1 2 3 4 5 6 7 8 --run-name "w_${LANG_CODE}${SUFFIX}" --device cuda
