#!/bin/bash
#SBATCH --job-name=vismem_qwen
#SBATCH --partition=short
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Qwen experiments (local inference, requires GPU)

set -e

SCRIPT_DIR="/insomnia001/home/pm3361/vision-memory"
source "$SCRIPT_DIR/venv/bin/activate"

# Use local HF cache to avoid rate limiting / download attempts
export HF_HOME="/insomnia001/home/pm3361/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="qwen"
N_TRIALS=50
RESULTS_DIR="$SCRIPT_DIR/results"
SIZES=(1 10 100 500)
DATASETS=("things" "Brady2008")
FOIL_TYPES=("novel" "exemplar" "state")

mkdir -p "$RESULTS_DIR" logs

echo "=============================================="
echo "Vision Memory Experiments: $MODEL"
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

# ============================================================================
# 2-AFC RECOGNITION
# ============================================================================
echo ""
echo "========== 2-AFC Recognition =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for foil in "${FOIL_TYPES[@]}"; do
        if [ "$dataset" = "things" ] && [ "$foil" = "state" ]; then continue; fi
        for size in "${SIZES[@]}"; do
            skip=false
            if [ "$dataset" = "Brady2008" ]; then
                [ "$foil" = "state" ]    && [ "$size" -gt 100 ] && skip=true
                [ "$foil" = "exemplar" ] && [ "$size" -gt 100 ] && skip=true
            fi
            if [ "$dataset" = "things" ]; then
                :
            fi
            [ "$skip" = true ] && echo "  [SKIP-LIMIT] $dataset | $foil | $size" && continue
            if check_existing_result "2afc" "$MODEL" "$dataset" "$size" "$foil"; then
                echo "  [EXISTS] $dataset | $foil | $size images"
                continue
            fi
            echo "  [RUN] $dataset | $foil | $size images"
            trials=$N_TRIALS
            [ "$size" -lt "$N_TRIALS" ] && trials=$size
            python3 -m eval_scripts.eval_2afc \
                --models "$MODEL" \
                --n-images "$size" \
                --n-trials "$trials" \
                --foil-type "$foil" \
                --dataset "$dataset" || echo "  [ERROR] $dataset | $foil | $size"
        done
    done
done

# ============================================================================
# CONTINUOUS RECOGNITION
# ============================================================================
echo ""
echo "========== Continuous Recognition =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if [ "$size" -lt 10 ]; then echo "  [SKIP-LIMIT] $dataset | $size (too small)"; continue; fi
        if check_existing_result "continuous" "$MODEL" "$dataset" "$size" ""; then
            echo "  [EXISTS] $dataset | $size images"
            continue
        fi
        echo "  [RUN] $dataset | $size images"
        python3 -m eval_scripts.eval_continuous \
            --models "$MODEL" \
            --n-images "$size" \
            --dataset "$dataset" || echo "  [ERROR] $dataset | $size"
    done
done

# ============================================================================
# PAIRED ASSOCIATE MEMORY
# ============================================================================
echo ""
echo "========== Paired Associate Memory =========="

for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    for size in "${SIZES[@]}"; do
        if check_existing_result "pam" "$MODEL" "$dataset" "$size" ""; then
            echo "  [EXISTS] $dataset | $size pairs"
            continue
        fi
        echo "  [RUN] $dataset | $size pairs"
        python3 -m eval_scripts.eval_pam \
            --models "$MODEL" \
            --n-images "$size" \
            --dataset "$dataset" || echo "  [ERROR] $dataset | $size"
    done
done

# ============================================================================
# VISUAL HAYSTACKS
# ============================================================================
echo ""
echo "========== Visual Haystacks =========="

for size in "${SIZES[@]}"; do
    if check_existing_result "haystacks" "$MODEL" "things" "$size" ""; then
        echo "  [EXISTS] $size images"
        continue
    fi
    echo "  [RUN] $size images"
    trials=$N_TRIALS
    [ "$size" -lt 10 ] && trials=20
    python3 -m eval_scripts.eval_visual_haystacks \
        --models "$MODEL" \
        --n-images "$size" \
        --n-trials "$trials" || echo "  [ERROR] $size"
done

echo ""
echo "=============================================="
echo "$MODEL experiments completed!"
echo "=============================================="
