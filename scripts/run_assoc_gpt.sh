#!/bin/bash
#SBATCH --job-name=assoc_gpt
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Associative Inference: gpt-4o

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"
export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

# Stagger start to avoid concurrent API hammering
sleep 60

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(2 4 6 10 50 100 250)
DATASETS=("things" "Brady2008")
VARIANTS=("word" "image")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local variant="$3"
    [ -f "$RESULTS_DIR/results_assoc_${variant}_gpt-4o_n${n_images}_${dataset}.json" ]
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
