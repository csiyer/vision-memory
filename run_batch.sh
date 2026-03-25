#!/bin/bash
# Batch evaluation runner
# Add your commands below, one per line

set -e  # Exit on error

echo "Starting batch run..."

# Add your evaluation commands here:
python -m eval_scripts.eval_2afc --models gpt-4o claude gemini --n-images 20 --foil-type novel
python -m eval_scripts.eval_2afc --models gpt-4o --n-images 5

echo "Batch complete!"
