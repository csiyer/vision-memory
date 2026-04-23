#!/bin/bash
#SBATCH --job-name=2afc_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 2-AFC Recognition: gpt-4o
# GPT-4o has 128K token context => skip n>=500

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gpt-4o"
N_TRIALS=50
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 5 10 100 500 1000)
DATASETS=("things" "Brady2008")
FOIL_TYPES=("novel" "exemplar" "state" "all")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local foil_type="$3"
    for f in "$RESULTS_DIR"/results_2afc_*.json; do
        [ -f "$f" ] || continue
        if python3 -c "
import json, sys
d = json.load(open('$f'))
m = d.get('_metadata', {})
if not m: sys.exit(1)
if '2-AFC' not in m.get('task', ''): sys.exit(1)
if not any('$MODEL' in x for x in m.get('models', [])): sys.exit(1)
if m.get('dataset') != '$dataset': sys.exit(1)
if m.get('n_images') != $n_images: sys.exit(1)
if m.get('foil_type') != '$foil_type': sys.exit(1)
if not any('$MODEL' in k for k in m.get('summary', {})): sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
    done
    return 1
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
            if [ "$size" -ge 500 ]; then
                echo "  [SKIP-LIMIT] $dataset | $foil | n=$size (GPT-4o context limit)"
                continue
            fi
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
