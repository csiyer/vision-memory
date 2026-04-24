#!/bin/bash
#SBATCH --job-name=vhs_multi_gpt4o
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# VHS multi-needle: gpt-4o
# Sizes fixed by benchmark: 2 5 10 50 100
# GPT-4o context limit => skip image_count>=500 (all valid VHS sizes are fine)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

MODEL="gpt-4o"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(5 10 50 100)

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local image_count="$1"
    for f in "$RESULTS_DIR"/results_vhs_*.json; do
        [ -f "$f" ] || continue
        if python3 -c "
import json, sys
d = json.load(open('$f'))
m = d.get('_metadata', {})
if not m: sys.exit(1)
if m.get('mode') != 'multi_needle': sys.exit(1)
if str(m.get('image_count')) != '$image_count': sys.exit(1)
if not any('$MODEL' in k for k in m.get('summary', {})): sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
    done
    return 1
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
