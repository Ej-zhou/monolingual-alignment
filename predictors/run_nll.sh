#!/bin/bash
# Mean FLORES NLL per Goldfish 1000 MB model (Appendix D), then the correlation
# analysis, Figure 11 and the joint regression.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
for lang in eng_latn zho_hans spa_latn arb_arab hin_deva fra_latn rus_cyrl deu_latn jpn_jpan; do
  $PYTHON compute_nll.py --lang "$lang" --batch-size 8 --device cuda
done
$PYTHON analyze_predictors.py
$PYTHON plot_syntactic_distance.py
$PYTHON regression_uriel_nll.py
