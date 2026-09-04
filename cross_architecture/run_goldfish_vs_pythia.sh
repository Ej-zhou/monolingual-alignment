#!/bin/bash
# Goldfish vs. Pythia (Appendix C.1, Figure 10 / Table 5).
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
$PYTHON goldfish_vs_pythia.py --model-a goldfish-models/eng_latn_1000mb \
    --model-b EleutherAI/pythia-160m --lang eng_latn --num-samples 200 --device cuda
$PYTHON goldfish_vs_pythia.py --model-a goldfish-models/zho_hans_1000mb \
    --model-b SJTU-CL/Zh-Pythia-160M --lang zho_hans --num-samples 200 --device cuda
$PYTHON plot_goldfish_vs_pythia.py
