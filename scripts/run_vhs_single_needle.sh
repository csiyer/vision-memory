#!/bin/bash
#SBATCH --job-name=vismem_single_vhs
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Run Visual Haystacks single-needle experiments for all 3 models
# across increasing haystack sizes (2, 5, 10, 50, 100 images).
#
# Usage: bash eval_scripts/run_vhs_single_needle.sh
#
# Requires: datasets/VHs_qa and datasets/coco to be present,
#           or pass --fetch-missing-coco to download COCO images on demand.

set -e
cd /insomnia001/home/pm3361/vision-memory
# source venv/bin/activate  # venv incomplete; use system python3 (~/.local has all packages)

MODELS="gpt-4o gemini qwen"
SPLIT="VHs_large"
MAX_SAMPLES=100

for IMAGE_COUNT in 2 5 10 50 100; do
    echo "=============================="
    echo "Single-needle | haystack=$IMAGE_COUNT"
    echo "=============================="
    python -m eval_scripts.eval_vhs \
        --models $MODELS \
        --mode single_needle \
        --split $SPLIT \
        --image-count $IMAGE_COUNT \
        --max-samples $MAX_SAMPLES \
        --fetch-missing-coco
done

echo "Done — results saved to results/results_vhs_*.json"
