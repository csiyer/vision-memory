#!/bin/bash
#SBATCH --job-name=continuous_qwen
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Continuous Recognition: qwen3-vl-8b (local inference, requires GPU)
# A6000 48GB VRAM => skip n>=500; skip n<5 (degenerate trial count)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

export HF_HOME="/insomnia001/home/pm3361/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="qwen"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 100 250 500 1000)
DATASETS=("things" "Brady2008")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    [ -f "$RESULTS_DIR/results_continuous_qwen3-vl-8b_n${n_images}_${dataset}.json" ]
}

echo "========== Continuous Recognition: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if [ "$size" -lt 5 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$size (too few unique images for a meaningful continuous task)"
            continue
        fi
        if [ "$size" -ge 500 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$size (VRAM limit)"
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
