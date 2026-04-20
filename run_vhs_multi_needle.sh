#!/bin/bash
#SBATCH --job-name=vismem_multi_vhs
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Run Visual Haystacks multi-needle experiments for all 3 models
# across increasing haystack sizes (2, 5, 10, 50, 100 images).
#
# Multi-needle asks two types of questions:
#   - "do ALL images with [anchor] contain [target]?"
#   - "do ANY images with [anchor] contain [target]?"
# Both are covered by --mode multi_needle.
#
# Usage: bash eval_scripts/run_vhs_multi_needle.sh

set -e
cd /insomnia001/home/pm3361/vision-memory
source venv/bin/activate

MODELS="gpt-4o gemini qwen"
MAX_SAMPLES=100

for IMAGE_COUNT in 5 10 50 100; do
    echo "=============================="
    echo "Multi-needle | haystack=$IMAGE_COUNT"
    echo "=============================="
    python -m eval_scripts.eval_vhs \
        --models $MODELS \
        --mode multi_needle \
        --image-count $IMAGE_COUNT \
        --max-samples $MAX_SAMPLES \
        --fetch-missing-coco
done

echo "Done — results saved to results/results_vhs_*.json"
