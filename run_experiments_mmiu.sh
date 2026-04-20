#!/bin/bash
#SBATCH --job-name=mmiu
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Run MMIU evaluation for GPT-4o, Gemini, and Qwen
# Full dataset: 11,698 samples across 52 tasks (~25 GB download on first run).
#
# Usage:
#   sbatch run_experiments_mmiu.sh            # full run
#   bash   run_experiments_mmiu.sh            # local run
#
# For a quick pilot, add --max-samples 200 to the python call below.
# If you've downloaded the dataset locally, add --image-root dataset/mmiu

set -e
cd /insomnia001/home/pm3361/vision-memory
source venv/bin/activate

MODELS="gpt-4o gemini qwen"

echo "=============================="
echo "MMIU — all tasks"
echo "Models: $MODELS"
echo "=============================="

python -m eval_scripts.eval_mmiu \
    --models $MODELS

echo "Done — results saved to results/results_mmiu_*.json"
