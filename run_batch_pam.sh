#!/bin/bash
# Paired Associate Memory tests - actually shows VLM limitations

set -e  # Exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Running Paired Associate Memory experiments..."

# =============================================================================
# PAIRED ASSOCIATE MEMORY - Tests associative binding
# This task showed 58% accuracy for GPT-4o - actually revealing limitations!
# =============================================================================

echo ""
echo "=== PAM Experiments ==="

python3 -m eval_scripts.eval_pam --models gpt-4o --n-images 100 --dataset things
echo "[1/4] Completed: gpt-4o 100 images PAM"

python3 -m eval_scripts.eval_pam --models gemini --n-images 100 --dataset things
echo "[2/4] Completed: gemini 100 images PAM"

python3 -m eval_scripts.eval_pam --models gpt-4o --n-images 200 --dataset things
echo "[3/4] Completed: gpt-4o 200 images PAM"

python3 -m eval_scripts.eval_pam --models gemini --n-images 200 --dataset things
echo "[4/4] Completed: gemini 200 images PAM"

echo ""
echo "PAM experiments complete!"
