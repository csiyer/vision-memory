#!/bin/bash
#SBATCH --job-name=mst_molmo2
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Mnemonic Similarity Task: molmo2-8b

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"

export HF_HOME="/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="molmo2"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 100 250)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local n_study="$1"
    [ -f "$RESULTS_DIR/results_mst_molmo2-8b_n${n_study}.json" ]
}

echo "========== Mnemonic Similarity Task: $MODEL =========="

for size in "${SIZES[@]}"; do
    if check_existing_result "$size"; then
        echo "  [EXISTS] n=$size"
        continue
    fi
    echo "  [RUN] n=$size"
    python3 -m eval_scripts.eval_mst \
        --models "$MODEL" \
        --n-study "$size" \
        --n-trials 10 || echo "  [ERROR] n=$size"
done

echo "Done."
