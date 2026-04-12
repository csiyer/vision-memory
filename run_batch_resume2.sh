#!/bin/bash
# Resume exemplar batch from GPT-4o 400 images

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Resuming exemplar foil experiments from GPT-4o 400..."

# =============================================================================
# EXEMPLAR FOIL EXPERIMENTS - Resume from GPT-4o 400
# =============================================================================

echo ""
echo "=== Exemplar Foil Experiments (resumed) ==="

# GPT-4o 400 (skipping if already done, but including for completeness)
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[4/9] Completed: gpt-4o 400 images exemplar"

# Gemini exemplar scaling
python3 -m eval_scripts.eval_2afc --models gemini --n-images 100 --n-trials 50 --foil-type exemplar --dataset things
echo "[5/9] Completed: gemini 100 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 200 --n-trials 50 --foil-type exemplar --dataset things
echo "[6/9] Completed: gemini 200 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 300 --n-trials 50 --foil-type exemplar --dataset things
echo "[7/9] Completed: gemini 300 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[8/9] Completed: gemini 400 images exemplar"

# =============================================================================
# BRADY DATASET EXPERIMENTS - Local dataset, no streaming issues
# =============================================================================

echo ""
echo "=== Brady2008 Dataset Experiments ==="

# Exemplar foils - Brady (most interesting, similar objects)
python3 -m eval_scripts.eval_2afc --models gpt-4o gemini --n-images 50 --n-trials 50 --foil-type exemplar --dataset Brady2008
echo "[9/9] Completed: gpt-4o & gemini 50 images exemplar (Brady2008)"

echo ""
echo "All experiments complete!"
