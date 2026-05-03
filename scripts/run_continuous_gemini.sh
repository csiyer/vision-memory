#!/bin/bash
#SBATCH --job-name=continuous_gemini
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Continuous Recognition: gemini-3.1-flash-image-preview
# 1M token context => all sizes supported; skip n<5 (degenerate trial count)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"
export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

# Stagger start to avoid concurrent API hammering
sleep 180

MODEL="gemini"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 50 100 250)
DATASETS=("things" "Brady2008")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    [ -f "$RESULTS_DIR/results_continuous_gemini-3.1-flash-image-preview_n${n_images}_${dataset}.json" ]
}

echo "========== Continuous Recognition: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if check_existing_result "$dataset" "$size"; then
            echo "  [EXISTS] $dataset | n=$size"
            continue
        fi
        echo "  [RUN] $dataset | n=$size"
        python3 -m eval_scripts.eval_continuous \
            --models "$MODEL" \
            --n-images "$size" \
            --dataset "$dataset" \
            --n-trials 100 || echo "  [ERROR] $dataset | n=$size"
    done
done

echo "Done."
