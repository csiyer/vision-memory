#!/bin/bash
# Brady dataset tests only - local files, no streaming issues

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Running Brady2008 dataset experiments..."

# =============================================================================
# BRADY DATASET EXPERIMENTS - Local dataset, fast and reliable
# =============================================================================

echo ""
echo "=== Brady2008 Dataset - Novel Foils ==="

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 50 --n-trials 50 --foil-type novel --dataset Brady2008
echo "[1/4] Completed: gpt-4o 50 images novel (Brady2008)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 50 --n-trials 50 --foil-type novel --dataset Brady2008
echo "[2/4] Completed: gemini 50 images novel (Brady2008)"

echo ""
echo "=== Brady2008 Dataset - Exemplar Foils ==="

python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 50 --n-trials 50 --foil-type exemplar --dataset Brady2008
echo "[3/4] Completed: gpt-4o 50 images exemplar (Brady2008)"

python3 -m eval_scripts.eval_2afc --models gemini --n-images 50 --n-trials 50 --foil-type exemplar --dataset Brady2008
echo "[4/4] Completed: gemini 50 images exemplar (Brady2008)"

echo ""
echo "Brady2008 experiments complete!"
