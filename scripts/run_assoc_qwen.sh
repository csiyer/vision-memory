#!/bin/bash
#SBATCH --job-name=assoc_qwen
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Associative Inference: qwen3-vl-8b (local inference, requires GPU)
# A6000 48GB VRAM => skip n>=500

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

export HF_HOME="/insomnia001/home/pm3361/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="qwen"
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 5 10 100 500 1000)
DATASETS=("things" "Brady2008")

mkdir -p "$RESULTS_DIR" logs

check_existing_result() {
    local dataset="$1"
    local n_images="$2"
    for f in "$RESULTS_DIR"/results_assoc_*.json; do
        [ -f "$f" ] || continue
        if python3 -c "
import json, sys
d = json.load(open('$f'))
m = d.get('_metadata', {})
if not m: sys.exit(1)
if 'Associative' not in m.get('task', ''): sys.exit(1)
if not any('qwen' in x for x in m.get('models', [])): sys.exit(1)
if m.get('dataset') != '$dataset': sys.exit(1)
if m.get('n_images') != $n_images: sys.exit(1)
if not any('qwen' in k for k in m.get('summary', {})): sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

echo "========== Associative Inference: $MODEL =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        n=$(( size % 2 == 0 ? size : size + 1 ))
        if [ "$n" -lt 4 ]; then
            echo "  [SKIP] $dataset | n=$n (requires at least 2 chains)"
            continue
        fi
        if [ "$n" -ge 500 ]; then
            echo "  [SKIP-LIMIT] $dataset | n=$n (VRAM limit)"
            continue
        fi
        if check_existing_result "$dataset" "$n"; then
            echo "  [EXISTS] $dataset | n=$n"
            continue
        fi
        echo "  [RUN] $dataset | n=$n"
        python3 -m eval_scripts.eval_associative_inference \
            --models "$MODEL" \
            --n-images "$n" \
            --dataset "$dataset" || echo "  [ERROR] $dataset | n=$n"
    done
done

echo "Done."
