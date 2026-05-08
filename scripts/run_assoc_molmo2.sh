#!/bin/bash
#SBATCH --job-name=assoc_molmo2
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Associative Inference: molmo2-8b (local inference, requires GPU)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="molmo2"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(2 4 6 10 100 250)
DATASETS=("things" "Brady2008")
VARIANTS=("word" "image")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local variant="$3"
    [ -f "$RESULTS_DIR/results_assoc_${variant}_molmo2-8b_n${n_images}_${dataset}.json" ]
}

echo "========== Associative Inference: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for variant in "${VARIANTS[@]}"; do
        echo "  -- Pair type: $variant --"
        for n in "${SIZES[@]}"; do
            if check_existing_result "$dataset" "$n" "$variant"; then
                echo "  [EXISTS] $dataset | $variant | n=$n"
                continue
            fi
            echo "  [RUN] $dataset | $variant | n=$n"
            python3 -m eval_scripts.eval_associative_inference \
                --models "$MODEL" \
                --n-images "$n" \
                --dataset "$dataset" \
                --pair-type "$variant" \
                --n-trials 10 || echo "  [ERROR] $dataset | $variant | n=$n"
        done
    done
done

echo "Done."
