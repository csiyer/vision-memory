#!/bin/bash
#SBATCH --job-name=continuous_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Continuous Recognition: gpt-4o
# GPT-4o 128K context => skip n>=500; skip n<10 (too few for continuous task)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 100 250 500 1000)
DATASETS=("things" "Brady2008")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    [ -f "$RESULTS_DIR/results_continuous_gpt-4o_n${n_images}_${dataset}.json" ]
}

echo "========== Continuous Recognition: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if [ "$size" -lt 10 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$size (too small for continuous task)"
            continue
        fi
        if [ "$size" -ge 500 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$size (GPT-4o context limit)"
            continue
        fi
        if check_existing_result "$dataset" "$size"; then
            echo "  [EXISTS] $dataset | n=$size"
            continue
        fi
        echo "  [RUN] $dataset | n=$size"
        python3 -m eval_scripts.eval_continuous \
            --models "$MODEL" \
            --n-images "$size" \
            --dataset "$dataset" || echo "  [ERROR] $dataset | n=$size"
    done
done

echo "Done."
