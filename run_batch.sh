#!/bin/bash
# Batch evaluation runner - Scaling experiments
# Test where models break with increasing image counts

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Starting scaling experiments..."

# =============================================================================
# SCALING EXPERIMENTS - Novel foils with increasing image counts
# Goal: Find where models break (accuracy drops below ceiling)
# =============================================================================

# GPT-4o scaling (can handle up to 500 images)
echo ""
echo "=== GPT-4o Scaling Experiments ==="
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 50 --foil-type novel --dataset things
echo "[1/10] Completed: gpt-4o 100 images"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 200 --n-trials 50 --foil-type novel --dataset things
echo "[2/10] Completed: gpt-4o 200 images"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 300 --n-trials 50 --foil-type novel --dataset things
echo "[3/10] Completed: gpt-4o 300 images"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 400 --n-trials 50 --foil-type novel --dataset things
echo "[4/10] Completed: gpt-4o 400 images"

# Gemini scaling (can handle up to 3600 images)
echo ""
echo "=== Gemini Scaling Experiments ==="
python3 -m eval_scripts.eval_2afc --models gemini --n-images 100 --n-trials 50 --foil-type novel --dataset things
echo "[5/10] Completed: gemini 100 images"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 200 --n-trials 50 --foil-type novel --dataset things
echo "[6/10] Completed: gemini 200 images"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type novel --dataset things
echo "[7/10] Completed: gemini 400 images"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 600 --n-trials 50 --foil-type novel --dataset things
echo "[8/10] Completed: gemini 600 images"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 800 --n-trials 50 --foil-type novel --dataset things
echo "[9/10] Completed: gemini 800 images"

# Note: THINGS has ~1854 categories, novel foils need 2x categories, so max ~900 images
python3 -m eval_scripts.eval_2afc --models gemini --n-images 900 --n-trials 50 --foil-type novel --dataset things
echo "[10/10] Completed: gemini 900 images"

# Claude baseline (capped at 98 study images due to 100 image limit)
# echo ""
# echo "=== Claude Baseline (100 image limit) ==="
# python3 -m eval_scripts.eval_2afc --models claude --n-images 98 --n-trials 50 --foil-type novel --dataset things
# echo "Completed: claude 98 images (max possible)"

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
