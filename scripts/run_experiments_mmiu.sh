#!/bin/bash
#SBATCH --job-name=mmiu
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Run MMIU evaluation (Qwen only).
# Dataset images are expected at IMAGE_ROOT (run download_mmiu.sh first if needed).
#
# Usage:
#   sbatch run_experiments_mmiu.sh            # full run
#   bash   run_experiments_mmiu.sh            # local run
#
# For a quick pilot, add --max-samples 200 to the python call below.

set -e
cd /insomnia001/home/pm3361/vision-memory
source /insomnia001/home/pm3361/vision-memory/venv/bin/activate

MODELS="qwen"
IMAGE_ROOT=/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/mmiu

echo "=============================="
echo "MMIU — all tasks"
echo "Models: $MODELS"
echo "=============================="

python -m eval_scripts.eval_mmiu \
    --models $MODELS \
    --image-root $IMAGE_ROOT

echo "Done — results saved to results/results_mmiu_*.json"
