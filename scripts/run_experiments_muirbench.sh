#!/bin/bash
#SBATCH --job-name=muirbench
#SBATCH --partition=zgroup1
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Run MuirBench evaluation for GPT-4o, Gemini, and Qwen
# across all 12 tasks (2,600 samples total, avg 4.3 images each).
#
# Usage:
#   sbatch run_experiments_muirbench.sh            # full run
#   bash   run_experiments_muirbench.sh            # local run
#
# For a quick pilot, add --max-samples 100 to the python call below.

set -e
cd /insomnia001/home/pm3361/vision-memory
# source venv/bin/activate  # venv incomplete; use system python3 (~/.local has all packages)

MODELS="${MODELS:-gpt-4o gemini qwen}"

echo "=============================="
echo "MuirBench — all tasks"
echo "Models: $MODELS"
echo "=============================="

python -m eval_scripts.eval_muirbench \
    --models $MODELS

echo "Done — results saved to results/results_muirbench_*.json"
