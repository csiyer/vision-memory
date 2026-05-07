#!/bin/bash
#SBATCH --job-name=mst_gemini
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Mnemonic Similarity Task: gemini-3-flash-preview

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/venv_vm/bin/activate"
export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

# Stagger start to avoid concurrent API hammering
sleep 240

MODEL="gemini"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 50 100 250)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local n_study="$1"
    [ -f "$RESULTS_DIR/results_mst_gemini-3-flash-preview_n${n_study}.json" ]
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
        --n-trials 100 || echo "  [ERROR] n=$size"
done

echo "Done."
