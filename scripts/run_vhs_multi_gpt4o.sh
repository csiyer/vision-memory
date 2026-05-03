#!/bin/bash
#SBATCH --job-name=vhs_multi_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# VHS multi-needle: gpt-4o

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"
export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

# Stagger start to avoid concurrent API hammering
sleep 360

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(5 10 20 50 100 500)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local image_count="$1"
    [ -f "$RESULTS_DIR/results_vhs_gpt-4o_n${image_count}_VHs_large_multi_needle.json" ]
}

echo "========== VHS multi_needle: $MODEL =========="

for size in "${SIZES[@]}"; do
    if check_existing_result "$size"; then
        echo "  [EXISTS] image_count=$size"
        continue
    fi
    echo "  [RUN] image_count=$size"
    python3 -m eval_scripts.eval_vhs \
        --models "$MODEL" \
        --mode multi_needle \
        --image-count "$size" \
        --max-samples 100 \
        --fetch-missing-coco || echo "  [ERROR] image_count=$size"
done

echo "Done."
