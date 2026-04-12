#!/bin/bash
# Comprehensive experiment runner
# Runs all combinations: tasks x models x foil types x sizes x datasets
# Skips experiments that already have complete results
#
# Tasks:
# - 2-AFC Recognition (novel, exemplar, state foils)
# - Continuous Recognition
# - Paired Associate Memory
# - Visual Haystacks

set -e

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# Configuration
N_TRIALS=50  # Number of test trials per experiment
RESULTS_DIR="$SCRIPT_DIR/results"

# Models to test
MODELS=("gpt-4o" "gemini" "qwen")

# Study sequence lengths (logarithmic scale)
SIZES=(1 10 100 1000)

# Datasets
DATASETS=("things" "Brady2008")

# Foil types for 2-AFC
FOIL_TYPES=("novel" "exemplar" "state")

echo "=============================================="
echo "Comprehensive Vision Memory Experiments"
echo "=============================================="
echo "Models: ${MODELS[*]}"
echo "Sizes: ${SIZES[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Trials per experiment: $N_TRIALS"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="
echo ""

# Function to check if an experiment already exists
check_existing_result() {
    local task="$1"
    local model="$2"
    local dataset="$3"
    local n_images="$4"
    local foil_type="$5"  # optional, for 2-AFC only

    # Search through existing results
    for f in "$RESULTS_DIR"/results_*.json; do
        [ -f "$f" ] || continue

        # Use python to check metadata
        if python3 -c "
import json
import sys

with open('$f') as fp:
    data = json.load(fp)

meta = data.get('_metadata', {})
if not meta:
    sys.exit(1)

# Check task
task_match = False
if '$task' == '2afc' and '2-AFC' in meta.get('task', ''):
    task_match = True
elif '$task' == 'continuous' and 'Continuous' in meta.get('task', ''):
    task_match = True
elif '$task' == 'pam' and 'Paired Associate' in meta.get('task', ''):
    task_match = True
elif '$task' == 'haystacks' and 'Haystacks' in meta.get('task', ''):
    task_match = True

if not task_match:
    sys.exit(1)

# Check model
if '$model' not in meta.get('models', []):
    sys.exit(1)

# Check dataset
if meta.get('dataset', '') != '$dataset':
    sys.exit(1)

# Check n_images
if meta.get('n_images', 0) != $n_images:
    sys.exit(1)

# Check foil_type for 2-AFC
if '$task' == '2afc' and '$foil_type':
    if meta.get('foil_type', '') != '$foil_type':
        sys.exit(1)

# Check that it has valid results (not empty/errored)
summary = meta.get('summary', {})
if '$model' not in summary:
    sys.exit(1)

# All checks passed - experiment exists
sys.exit(0)
" 2>/dev/null; then
            return 0  # Exists
        fi
    done
    return 1  # Does not exist
}

# ============================================================================
# 2-AFC RECOGNITION EXPERIMENTS
# ============================================================================
echo ""
echo "========== 2-AFC Recognition Experiments =========="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=== Model: $model ==="

    for dataset in "${DATASETS[@]}"; do
        echo "--- Dataset: $dataset ---"

        for foil in "${FOIL_TYPES[@]}"; do
            # Skip state for THINGS
            if [ "$dataset" = "things" ] && [ "$foil" = "state" ]; then
                continue
            fi

            for size in "${SIZES[@]}"; do
                # Check dataset size limits
                skip=false

                if [ "$dataset" = "Brady2008" ]; then
                    if [ "$foil" = "state" ] && [ "$size" -gt 70 ]; then
                        skip=true
                    fi
                    if [ "$foil" = "exemplar" ] && [ "$size" -gt 90 ]; then
                        skip=true
                    fi
                    if [ "$foil" = "novel" ] && [ "$size" -gt 1000 ]; then
                        skip=true
                    fi
                fi

                if [ "$dataset" = "things" ]; then
                    if [ "$foil" = "novel" ] && [ "$size" -gt 900 ]; then
                        skip=true
                    fi
                fi

                if [ "$skip" = true ]; then
                    echo "  [SKIP-LIMIT] $model | $dataset | $foil | $size images"
                    continue
                fi

                # Check if result already exists
                if check_existing_result "2afc" "$model" "$dataset" "$size" "$foil"; then
                    echo "  [EXISTS] $model | $dataset | $foil | $size images"
                    continue
                fi

                echo "  [RUN] $model | $dataset | $foil | $size images"

                # Adjust n_trials if size is very small
                trials=$N_TRIALS
                if [ "$size" -lt "$N_TRIALS" ]; then
                    trials=$size
                fi

                python3 -m eval_scripts.eval_2afc \
                    --models "$model" \
                    --n-images "$size" \
                    --n-trials "$trials" \
                    --foil-type "$foil" \
                    --dataset "$dataset" || echo "  [ERROR] Failed: $model | $dataset | $foil | $size"
            done
        done
    done
done

# ============================================================================
# CONTINUOUS RECOGNITION EXPERIMENTS
# ============================================================================
echo ""
echo "========== Continuous Recognition Experiments =========="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=== Model: $model ==="

    for dataset in "${DATASETS[@]}"; do
        echo "--- Dataset: $dataset ---"

        for size in "${SIZES[@]}"; do
            # Skip very small sizes for continuous (need enough for repetitions)
            if [ "$size" -lt 10 ]; then
                echo "  [SKIP-LIMIT] $model | $dataset | $size images (too small for continuous)"
                continue
            fi

            # Check if result already exists
            if check_existing_result "continuous" "$model" "$dataset" "$size" ""; then
                echo "  [EXISTS] $model | $dataset | $size images"
                continue
            fi

            echo "  [RUN] $model | $dataset | $size images"

            python3 -m eval_scripts.eval_continuous \
                --models "$model" \
                --n-images "$size" \
                --dataset "$dataset" || echo "  [ERROR] Failed: $model | $dataset | $size"
        done
    done
done

# ============================================================================
# PAIRED ASSOCIATE MEMORY EXPERIMENTS
# ============================================================================
echo ""
echo "========== Paired Associate Memory Experiments =========="

for model in "${MODELS[@]}"; do
    echo ""
    echo "=== Model: $model ==="

    for dataset in "${DATASETS[@]}"; do
        echo "--- Dataset: $dataset ---"

        for size in "${SIZES[@]}"; do
            # Check if result already exists
            if check_existing_result "pam" "$model" "$dataset" "$size" ""; then
                echo "  [EXISTS] $model | $dataset | $size pairs"
                continue
            fi

            echo "  [RUN] $model | $dataset | $size pairs"

            python3 -m eval_scripts.eval_pam \
                --models "$model" \
                --n-images "$size" \
                --dataset "$dataset" || echo "  [ERROR] Failed: $model | $dataset | $size"
        done
    done
done

# ============================================================================
# VISUAL HAYSTACKS EXPERIMENTS
# ============================================================================
echo ""
echo "========== Visual Haystacks Experiments =========="

# Haystacks only uses THINGS (simplified version)
for model in "${MODELS[@]}"; do
    echo ""
    echo "=== Model: $model ==="

    for size in "${SIZES[@]}"; do
        # Check if result already exists
        if check_existing_result "haystacks" "$model" "things" "$size" ""; then
            echo "  [EXISTS] $model | $size images"
            continue
        fi

        echo "  [RUN] $model | $size images"

        # Adjust n_trials based on size
        trials=$N_TRIALS
        if [ "$size" -lt 10 ]; then
            trials=20  # Fewer trials for very small haystacks
        fi

        python3 -m eval_scripts.eval_visual_haystacks \
            --models "$model" \
            --n-images "$size" \
            --n-trials "$trials" || echo "  [ERROR] Failed: $model | $size"
    done
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "To generate plots, run:"
echo "  python3 plot_scaling_curves.py"
echo ""
