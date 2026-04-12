#!/bin/bash
# Exemplar foil experiments + Brady dataset for larger scales

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Starting exemplar foil and Brady dataset experiments..."

# =============================================================================
# EXEMPLAR FOIL EXPERIMENTS - THINGS dataset
# Same category, different instance (e.g., two different dogs)
# Tests fine-grained visual discrimination rather than just category matching
# Exemplar foils only need 1 category per image (not 2), so loading is faster
# =============================================================================

echo ""
echo "=== Exemplar Foil Experiments - THINGS Dataset ==="

# GPT-4o exemplar scaling
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 50 --foil-type exemplar --dataset things
echo "[1/12] Completed: gpt-4o 100 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 200 --n-trials 50 --foil-type exemplar --dataset things
echo "[2/12] Completed: gpt-4o 200 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 300 --n-trials 50 --foil-type exemplar --dataset things
echo "[3/12] Completed: gpt-4o 300 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[4/12] Completed: gpt-4o 400 images exemplar (THINGS)"

# Gemini exemplar scaling
python3 -m eval_scripts.eval_2afc --models gemini --n-images 100 --n-trials 50 --foil-type exemplar --dataset things
echo "[5/12] Completed: gemini 100 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 200 --n-trials 50 --foil-type exemplar --dataset things
echo "[6/12] Completed: gemini 200 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 300 --n-trials 50 --foil-type exemplar --dataset things
echo "[7/12] Completed: gemini 300 images exemplar (THINGS)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type exemplar --dataset things
echo "[8/12] Completed: gemini 400 images exemplar (THINGS)"

echo ""
echo "THINGS exemplar experiments complete!"

# =============================================================================
# BRADY DATASET EXPERIMENTS - Local dataset, no streaming issues
# Run both novel and exemplar foils with Brady2008 dataset
# =============================================================================

echo ""
echo "=== Brady2008 Dataset Experiments ==="

# Novel foils - Brady
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 50 --n-trials 50 --foil-type novel --dataset Brady2008
echo "[9/12] Completed: gpt-4o 50 images novel (Brady2008)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 50 --n-trials 50 --foil-type novel --dataset Brady2008
echo "[10/12] Completed: gemini 50 images novel (Brady2008)"

# Exemplar foils - Brady
python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 50 --n-trials 50 --foil-type exemplar --dataset Brady2008
echo "[11/12] Completed: gpt-4o 50 images exemplar (Brady2008)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 50 --n-trials 50 --foil-type exemplar --dataset Brady2008
echo "[12/12] Completed: gemini 50 images exemplar (Brady2008)"

echo ""
echo "All experiments complete!"
