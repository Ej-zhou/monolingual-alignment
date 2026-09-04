#!/bin/bash
# Five ~1B monolingual models, all C(5,2)=10 pairs (Section 3.4, Table 3).
# Set PAIR_INDEX=<0..9> to run a single pair.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
MODELS=("EleutherAI/pythia-1.4b|eng_latn"
        "SJTU-CL/Zh-Pythia-1.4B|zho_hans"
        "TucanoBR/Tucano-1b1|por_latn"
        "speakleash/Bielik-1.5B-v3|pol_latn"
        "sapienzanlp/Minerva-1B-base-v1.0|ita_latn")
n=${#MODELS[@]}
idx=0
for ((i=0; i<n; i++)); do
  for ((j=i+1; j<n; j++)); do
    if [ -z "${PAIR_INDEX:-}" ] || [ "$PAIR_INDEX" -eq "$idx" ]; then
      a="${MODELS[$i]}"; b="${MODELS[$j]}"
      echo "=== pair $idx: ${a%%|*} vs ${b%%|*} ==="
      $PYTHON diverse_models.py --model-a "${a%%|*}" --lang-a "${a##*|}" \
          --model-b "${b%%|*}" --lang-b "${b##*|}" --num-samples 200 --device cuda
    fi
    idx=$((idx + 1))
  done
done
$PYTHON table_diverse_models.py
