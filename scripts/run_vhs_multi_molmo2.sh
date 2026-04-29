#!/bin/bash
#SBATCH --job-name=vhs_multi_molmo2
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# VHS multi-needle: molmo2-8b

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="molmo2"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(5 10 20 50 100 500)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local image_count="$1"
    [ -f "$RESULTS_DIR/results_vhs_molmo2-8b_n${image_count}_VHs_large_multi_needle.json" ]
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
