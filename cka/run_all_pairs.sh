#!/bin/bash
# Pairwise CKA for all C(9,2)=36 Goldfish model pairs of one size.
#
#   bash run_all_pairs.sh 1000mb results/main flores tatoeba opus bouquet   # Sec. 3, App. A
#   bash run_all_pairs.sh 100mb  results/size_scaling flores                # App. B (each size)
#
# Every pair is independent: set PAIR_INDEX=<0..35> to run just one pair,
# e.g. from a SLURM array job with PAIR_INDEX=$SLURM_ARRAY_TASK_ID.
set -e
cd "$(dirname "$0")"
SIZE=${1:-1000mb}
RESULTS_DIR=${2:-results/main}
if [ $# -gt 2 ]; then shift 2; DATASETS="$*"; else DATASETS="flores tatoeba opus bouquet"; fi
PYTHON=${PYTHON:-python}

LANGS=(eng_latn zho_hans spa_latn arb_arab hin_deva fra_latn rus_cyrl deu_latn jpn_jpan)
n=${#LANGS[@]}
idx=0
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    if [ -z "${PAIR_INDEX:-}" ] || [ "$PAIR_INDEX" -eq "$idx" ]; then
      echo "=== pair $idx: ${LANGS[$i]} vs ${LANGS[$j]} ($SIZE) ==="
      $PYTHON cka_pairwise.py \
        --model-a "goldfish-models/${LANGS[$i]}_${SIZE}" \
        --model-b "goldfish-models/${LANGS[$j]}_${SIZE}" \
        --dataset $DATASETS --results-dir "$RESULTS_DIR" \
        --num-samples 200 --batch-size 8 --aligner simalign --device cuda
    fi
    idx=$((idx + 1))
  done
done
