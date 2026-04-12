#!/bin/bash
# Resume batch evaluation from Gemini 400 onwards

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Resuming scaling experiments from Gemini 400..."

# =============================================================================
# NOVEL FOILS - Resume from Gemini 400
# =============================================================================

echo ""
echo "=== Gemini Novel Foil Scaling (resumed) ==="

python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type novel --dataset things
echo "[7/18] Completed: gemini 400 images novel"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 600 --n-trials 50 --foil-type novel --dataset things
echo "[8/18] Completed: gemini 600 images novel"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 800 --n-trials 50 --foil-type novel --dataset things
echo "[9/18] Completed: gemini 800 images novel"

# Note: THINGS has ~1854 categories, novel foils need 2x categories, so max ~900 images
python3 -m eval_scripts.eval_2afc --models gemini --n-images 900 --n-trials 50 --foil-type novel --dataset things
echo "[10/18] Completed: gemini 900 images novel"

echo ""
echo "Novel foil experiments complete!"

# =============================================================================
# EXEMPLAR FOIL EXPERIMENTS - Same category, different instance
# This is harder: both images are plausible (e.g., two different dogs)
# Tests fine-grained visual discrimination rather than just category matching
# =============================================================================

echo ""
echo "=== Exemplar Foil Experiments ==="

# GPT-4o exemplar scaling
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 50 --foil-type exemplar --dataset things
echo "[11/18] Completed: gpt-4o 100 images exemplar"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 200 --n-trials 50 --foil-type exemplar --dataset things
echo "[12/18] Completed: gpt-4o 200 images exemplar"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 300 --n-trials 50 --foil-type exemplar --dataset things
echo "[13/18] Completed: gpt-4o 300 images exemplar"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[14/18] Completed: gpt-4o 400 images exemplar"

# Gemini exemplar scaling
python3 -m eval_scripts.eval_2afc --models gemini --n-images 100 --n-trials 50 --foil-type exemplar --dataset things
echo "[15/18] Completed: gemini 100 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 200 --n-trials 50 --foil-type exemplar --dataset things
echo "[16/18] Completed: gemini 200 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[17/18] Completed: gemini 400 images exemplar"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 600 --n-trials 50 --foil-type exemplar --dataset things
echo "[18/18] Completed: gemini 600 images exemplar"

echo ""
echo "All scaling experiments complete!"
