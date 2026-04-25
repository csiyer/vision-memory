#!/bin/bash
#SBATCH --job-name=serial_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Serial Order Memory: gpt-4o (both free-report and AFC variants)
# GPT-4o has 128K token context => skip n>=500

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 5 10 100 500 1000)
DATASETS=("things" "Brady2008")
VARIANTS=("free" "afc")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    local variant="$3"
    for f in "$RESULTS_DIR"/results_serial_${variant}_*.json; do
        [ -f "$f" ] || continue
        if python3 -c "
import json, sys
d = json.load(open('$f'))
m = d.get('_metadata', {})
if not m: sys.exit(1)
if 'Serial Order' not in m.get('task', ''): sys.exit(1)
if m.get('variant') != '$variant': sys.exit(1)
if not any('$MODEL' in x for x in m.get('models', [])): sys.exit(1)
if m.get('dataset') != '$dataset': sys.exit(1)
if m.get('n_images') != $n_images: sys.exit(1)
if not any('$MODEL' in k for k in m.get('summary', {})): sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

echo "========== Serial Order Memory: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for variant in "${VARIANTS[@]}"; do
        echo "  -- Variant: $variant --"
        for size in "${SIZES[@]}"; do
            if [ "$size" -ge 500 ]; then
                echo "  [SKIP-LIMIT] $dataset | $variant | n=$size (GPT-4o context limit)"
                continue
            fi
            if check_existing_result "$dataset" "$size" "$variant"; then
                echo "  [EXISTS] $dataset | $variant | n=$size"
                continue
            fi
            echo "  [RUN] $dataset | $variant | n=$size"
            python3 -m eval_scripts.eval_serial_order \
                --models "$MODEL" \
                --n-images "$size" \
                --variant "$variant" \
                --dataset "$dataset" || echo "  [ERROR] $dataset | $variant | n=$size"
        done
    done
done

echo "Done."
