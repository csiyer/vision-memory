#!/bin/bash
#SBATCH --job-name=2afc_gemini
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 2-AFC Recognition: gemini-2.5-flash
# 1M token context => all sizes supported

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gemini"
N_TRIALS=50
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 100 250 500 1000)
DATASETS=("things" "Brady2008")
FOIL_TYPES=("novel" "exemplar" "state" "all")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local foil_type="$3"
    [ -f "$RESULTS_DIR/results_2afc_gemini-2.5-flash_n${n_images}_${dataset}_${foil_type}.json" ]
}

echo "========== 2-AFC Recognition: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for foil in "${FOIL_TYPES[@]}"; do
        if [ "$foil" = "state" ] && [ "$dataset" = "things" ]; then
            echo "  [SKIP] things | state (not supported)"
            continue
        fi
        for size in "${SIZES[@]}"; do
            if check_existing_result "$dataset" "$size" "$foil"; then
                echo "  [EXISTS] $dataset | $foil | n=$size"
                continue
            fi
            echo "  [RUN] $dataset | $foil | n=$size"
            trials=$N_TRIALS
            [ "$size" -lt "$N_TRIALS" ] && trials=$size
            python3 -m eval_scripts.eval_2afc \
                --models "$MODEL" \
                --n-images "$size" \
                --n-trials "$trials" \
                --foil-type "$foil" \
                --dataset "$dataset" || echo "  [ERROR] $dataset | $foil | n=$size"
        done
    done
done

echo "Done."
