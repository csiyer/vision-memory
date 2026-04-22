#!/bin/bash
#SBATCH --job-name=mmiu
#SBATCH --partition=zgroup1
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000

# Run MMIU evaluation for GPT-4o, Gemini, and Qwen
# Full dataset: 11,698 samples across 52 tasks (~25 GB download on first run).
#
# Usage:
#   sbatch run_experiments_mmiu.sh            # full run
#   bash   run_experiments_mmiu.sh            # local run
#
# For a quick pilot, add --max-samples 200 to the python call below.
# If you've downloaded the dataset locally, add --image-root datasets/mmiu

set -e
cd /insomnia001/home/pm3361/vision-memory
# source venv/bin/activate  # venv incomplete; use system python3 (~/.local has all packages)

MODELS="gpt-4o gemini qwen"
IMAGE_ROOT=/tmp/mmiu
HF_BASE="https://huggingface.co/datasets/FanqingM/MMIU-Benchmark/resolve/main"

ZIPS=(
    "2D-spatial.zip"
    "3D-spatial.zip"
    "Continuous-temporal.zip"
    "Discrete-temporal.zip"
    "High-level-obj-semantic.zip"
    "High-level-sub-semantic.zip"
    "Low-level-semantic.zip"
)

echo "=============================="
echo "Downloading MMIU images to $IMAGE_ROOT"
echo "=============================="
mkdir -p "$IMAGE_ROOT"
for ZIP in "${ZIPS[@]}"; do
    CATEGORY="${ZIP%.zip}"
    if [ -d "$IMAGE_ROOT/$CATEGORY" ]; then
        echo "  Already extracted: $CATEGORY — skipping"
        continue
    fi
    echo "  Downloading $ZIP..."
    wget -q --show-progress -O "$IMAGE_ROOT/$ZIP" "$HF_BASE/$ZIP"
    echo "  Extracting $ZIP..."
    unzip -q "$IMAGE_ROOT/$ZIP" -d "$IMAGE_ROOT"
    rm "$IMAGE_ROOT/$ZIP"
done
echo "Images ready."

echo "=============================="
echo "MMIU — all tasks"
echo "Models: $MODELS"
echo "=============================="

python -m eval_scripts.eval_mmiu \
    --models $MODELS \
    --image-root $IMAGE_ROOT

echo "Done — results saved to results/results_mmiu_*.json"
