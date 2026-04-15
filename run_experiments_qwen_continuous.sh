#!/bin/bash
#SBATCH --job-name=vismem_qwen_cont
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Qwen continuous N=500 only (long-running, needs 12h)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

export HF_HOME="/insomnia001/home/pm3361/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="qwen"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR" logs

echo "=============================================="
echo "Vision Memory Experiments: $MODEL (Continuous N=500)"
echo "=============================================="

check_existing_result() {
    local task="$1"
    local model="$2"
    local dataset="$3"
    local n_images="$4"
    local foil_type="$5"

    for f in "$RESULTS_DIR"/results_*.json; do
        [ -f "$f" ] || continue
        if python3 -c "
import json, sys
with open('$f') as fp:
    data = json.load(fp)
meta = data.get('_metadata', {})
if not meta: sys.exit(1)
task_match = False
if '$task' == '2afc' and '2-AFC' in meta.get('task', ''): task_match = True
elif '$task' == 'continuous' and 'Continuous' in meta.get('task', ''): task_match = True
elif '$task' == 'pam' and 'Paired Associate' in meta.get('task', ''): task_match = True
elif '$task' == 'haystacks' and 'Haystacks' in meta.get('task', ''): task_match = True
if not task_match: sys.exit(1)
if not any('$model' in m for m in meta.get('models', [])): sys.exit(1)
if meta.get('dataset', '') != '$dataset': sys.exit(1)
if meta.get('n_images', 0) != $n_images: sys.exit(1)
if '$task' == '2afc' and '$foil_type':
    if meta.get('foil_type', '') != '$foil_type': sys.exit(1)
if not any('$model' in k for k in meta.get('summary', {})): sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

echo ""
echo "========== Continuous Recognition (N=500 only) =========="

for dataset in "things" "Brady2008"; do
    echo "--- Dataset: $dataset ---"
    if check_existing_result "continuous" "$MODEL" "$dataset" 500 ""; then
        echo "  [EXISTS] $dataset | 500 images"
    else
        echo "  [RUN] $dataset | 500 images"
        python3 -m eval_scripts.eval_continuous \
            --models "$MODEL" \
            --n-images 500 \
            --dataset "$dataset" || echo "  [ERROR] $dataset | 500"
    fi
done

echo ""
echo "=============================================="
echo "$MODEL continuous N=500 completed!"
echo "=============================================="
