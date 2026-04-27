#!/bin/bash
#SBATCH --job-name=assoc_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Associative Inference: gpt-4o
# GPT-4o has 128K token context => skip n>=500

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
    [ -f "$RESULTS_DIR/results_assoc_gpt-4o_n${n_images}_${dataset}.json" ]
}

echo "========== Associative Inference: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        n=$(( size % 2 == 0 ? size : size + 1 ))
        if [ "$n" -lt 4 ]; then
            echo "  [SKIP] $dataset | n=$n (requires at least 2 chains)"
            continue
        fi
        if [ "$n" -ge 500 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$n (GPT-4o context limit)"
            continue
        fi
        if check_existing_result "$dataset" "$n"; then
            echo "  [EXISTS] $dataset | n=$n"
            continue
        fi
        echo "  [RUN] $dataset | n=$n"
        python3 -m eval_scripts.eval_associative_inference \
            --models "$MODEL" \
            --n-images "$n" \
            --dataset "$dataset" || echo "  [ERROR] $dataset | n=$n"
    done
done

echo "Done."
