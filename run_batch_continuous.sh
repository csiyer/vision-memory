#!/bin/bash
# Continuous Recognition tests only - the 4 remaining tests

# Don't exit on error - handle errors ourselves
set +e

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Running Continuous Recognition experiments..."
echo "4 remaining tests"

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
# CONTINUOUS RECOGNITION - 4 tests
# =============================================================================

echo ""
echo "=== Continuous Recognition (4 tests) ==="

run_test "[1/4] GPT-4o 100 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gpt-4o --n-images 100 --dataset things"

run_test "[2/4] Gemini 100 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gemini --n-images 100 --dataset things"

run_test "[3/4] GPT-4o 200 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gpt-4o --n-images 200 --dataset things"

run_test "[4/4] Gemini 200 imgs continuous" \
    "python3 -m eval_scripts.eval_continuous --models gemini --n-images 200 --dataset things"

echo ""
echo "======================================"
echo "Continuous recognition tests complete!"
echo "======================================"
