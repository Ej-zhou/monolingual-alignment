#!/bin/bash
# Cross-size CKA (App. B.2): every language at size A against every language at
# size B on FLORES (9 x 9 = 81 pairs).
#
#   bash run_cross_size.sh 1000mb 100mb
#   bash run_cross_size.sh 10mb 5mb
#
# Set PAIR_INDEX=<0..80> to run a single pair (e.g. from a SLURM array job).
set -e
cd "$(dirname "$0")"
SIZE_A=${1:-1000mb}
SIZE_B=${2:-100mb}
RESULTS_DIR=${3:-results/size_scaling}
PYTHON=${PYTHON:-python}

LANGS=(eng_latn zho_hans spa_latn arb_arab hin_deva fra_latn rus_cyrl deu_latn jpn_jpan)
n=${#LANGS[@]}
for ((idx=0; idx<n*n; idx++)); do
  if [ -n "${PAIR_INDEX:-}" ] && [ "$PAIR_INDEX" -ne "$idx" ]; then continue; fi
  i=$((idx / n)); j=$((idx % n))
  echo "=== pair $idx: ${LANGS[$i]}_${SIZE_A} vs ${LANGS[$j]}_${SIZE_B} ==="
  $PYTHON cka_pairwise.py \
    --model-a "goldfish-models/${LANGS[$i]}_${SIZE_A}" \
    --model-b "goldfish-models/${LANGS[$j]}_${SIZE_B}" \
    --dataset flores --results-dir "$RESULTS_DIR" \
    --num-samples 200 --batch-size 8 --aligner simalign --device cuda
done
