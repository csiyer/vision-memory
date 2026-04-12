#!/bin/bash
# Comprehensive batch runner - PAM, 2-AFC, and Continuous Recognition
# Runs all remaining tests with error recovery

# Don't exit on error - we'll handle errors ourselves
set +e

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Starting comprehensive batch run..."
echo "Tests: PAM (4), 2-AFC remaining (4), Continuous Recognition (4)"
echo "Total: 12 tests"

# Function to run a test with retry logic
run_test() {
    local test_name="$1"
    local command="$2"
    local max_retries=2
    local retry=0

    while [ $retry -le $max_retries ]; do
        if [ $retry -gt 0 ]; then
            echo "Retry $retry/$max_retries for: $test_name"
            sleep 10
        fi

        echo "Running: $test_name"
        eval "$command"
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "✓ Completed: $test_name"
            return 0
        else
            echo "✗ Failed: $test_name (exit code: $exit_code)"
            retry=$((retry + 1))
        fi
    done

    echo "⚠ Skipping after $max_retries retries: $test_name"
    return 1
}

# =============================================================================
# PAIRED ASSOCIATE MEMORY - 4 tests
# =============================================================================

echo ""
echo "=== PAM Experiments (4 tests) ==="

run_test "[1/12] GPT-4o 100 imgs PAM" \
    "python3 -m eval_scripts.eval_pam --models gpt-4o --n-images 100 --dataset things"

run_test "[2/12] Gemini 100 imgs PAM" \
    "python3 -m eval_scripts.eval_pam --models gemini --n-images 100 --dataset things"

run_test "[3/12] GPT-4o 200 imgs PAM" \
    "python3 -m eval_scripts.eval_pam --models gpt-4o --n-images 200 --dataset things"

run_test "[4/12] Gemini 200 imgs PAM" \
    "python3 -m eval_scripts.eval_pam --models gemini --n-images 200 --dataset things"

# =============================================================================
# 2-AFC EXEMPLAR - Remaining tests (4 tests: 300 and 400 for both models)
# =============================================================================

echo ""
echo "=== 2-AFC Exemplar Remaining (4 tests) ==="

run_test "[5/12] GPT-4o 300 imgs exemplar" \
    "python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 300 --n-trials 50 --foil-type exemplar --dataset things"

run_test "[6/12] GPT-4o 400 imgs exemplar" \
    "python3 -m eval_scripts.eval_2afc --models gpt-4o --n-images 400 --n-trials 50 --foil-type exemplar --dataset things"

run_test "[7/12] Gemini 300 imgs exemplar" \
    "python3 -m eval_scripts.eval_2afc --models gemini --n-images 300 --n-trials 50 --foil-type exemplar --dataset things"

run_test "[8/12] Gemini 400 imgs exemplar" \
    "python3 -m eval_scripts.eval_2afc --models gemini --n-images 400 --n-trials 50 --foil-type exemplar --dataset things"

# =============================================================================
# CONTINUOUS RECOGNITION - 4 tests
# =============================================================================

echo ""
echo "=== Continuous Recognition (4 tests) ==="

run_test "[9/12] GPT-4o 100 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gpt-4o --n-images 100 --dataset things"

run_test "[10/12] Gemini 100 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gemini --n-images 100 --dataset things"

run_test "[11/12] GPT-4o 200 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gpt-4o --n-images 200 --dataset things"

run_test "[12/12] Gemini 200 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gemini --n-images 200 --dataset things"

echo ""
echo "======================================"
echo "All experiments complete!"
echo "Check results/ folder for output files"
echo "======================================"
