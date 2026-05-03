#!/bin/bash
#SBATCH --job-name=2afc_gemini
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 2-AFC Recognition: gemini-3.1-flash-image-preview
# 1M token context => all sizes supported

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"
export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)

# Stagger start to avoid concurrent API hammering
sleep 0

MODEL="gemini"
N_TRIALS=100
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 2 5 10 50 100 250)
DATASETS=("things" "Brady2008")
FOIL_TYPES=("novel" "exemplar" "state" "all")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local foil_type="$3"
    [ -f "$RESULTS_DIR/results_2afc_gemini-3.1-flash-image-preview_n${n_images}_${dataset}_${foil_type}.json" ]
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
            if [ "$dataset" = "things" ] && [ "$size" -ge 250 ] && { [ "$foil" = "novel" ] || [ "$foil" = "all" ]; }; then
                echo "  [SKIP] things | $foil | n=$size (needs 2x categories, THINGS only has 225)"
                continue
            fi
            if check_existing_result "$dataset" "$size" "$foil"; then
                echo "  [EXISTS] $dataset | $foil | n=$size"
                continue
            fi
            echo "  [RUN] $dataset | $foil | n=$size"
            trials=$N_TRIALS
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
