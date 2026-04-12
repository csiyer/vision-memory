#!/bin/bash
# Single test: Gemini 200 images continuous recognition

set +e  # Don't exit on error

# Activate virtual environment
source /Users/pranmodu/Projects/columbia/vision-memory/venv/bin/activate

echo "Running final test: Gemini 200 images continuous recognition..."

python3 -m eval_scripts.eval_continuous --models gemini --n-images 200 --dataset things

if [ $? -eq 0 ]; then
    echo "✓ Completed: Gemini 200 imgs continuous"
else
    echo "✗ Failed: Gemini 200 imgs continuous"
fi

echo "Test complete!"
