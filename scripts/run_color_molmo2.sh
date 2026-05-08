#!/bin/bash
#SBATCH --job-name=color_molmo2
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Color Memory: molmo2-8b (both continuous and named variants)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="molmo2"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 100 250)
VARIANTS=("continuous" "named")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local n_images="$1"
    local variant="$2"
    [ -f "$RESULTS_DIR/results_color_${variant}_molmo2-8b_n${n_images}.json" ]
}

echo "========== Color Memory: $MODEL =========="

for variant in "${VARIANTS[@]}"; do
    echo "  -- Variant: $variant --"
    for size in "${SIZES[@]}"; do
        if check_existing_result "$size" "$variant"; then
            echo "  [EXISTS] $variant | n=$size"
            continue
        fi
        echo "  [RUN] $variant | n=$size"
        python3 -m eval_scripts.eval_color_memory \
            --models "$MODEL" \
            --n-images "$size" \
            --variant "$variant" \
            --n-trials 10 || echo "  [ERROR] $variant | n=$size"
    done
done

echo "Done."
