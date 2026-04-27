#!/bin/bash
#SBATCH --job-name=vhs_single_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# VHS single-needle: gpt-4o
# Sizes fixed by benchmark: 2 5 10 50 100
# GPT-4o context limit => skip image_count>=500 (all valid VHS sizes are fine)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(2 5 10 50 100 250 500)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local image_count="$1"
    [ -f "$RESULTS_DIR/results_vhs_gpt-4o_n${image_count}_VHs_large_single_needle.json" ]
}

echo "========== VHS single_needle: $MODEL =========="

for size in "${SIZES[@]}"; do
    if check_existing_result "$size"; then
        echo "  [EXISTS] image_count=$size"
        continue
    fi
    echo "  [RUN] image_count=$size"
    python3 -m eval_scripts.eval_vhs \
        --models "$MODEL" \
        --mode single_needle \
        --split VHs_large \
        --image-count "$size" \
        --max-samples 100 \
        --fetch-missing-coco || echo "  [ERROR] image_count=$size"
done

echo "Done."
