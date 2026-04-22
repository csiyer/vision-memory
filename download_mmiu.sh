#!/bin/bash
# Download and extract MMIU dataset images to dataset/mmiu/
# Run this once before sbatch run_experiments_mmiu.sh
# Total size: ~25 GB

set -e
cd /insomnia001/home/pm3361/vision-memory

DEST=dataset/mmiu
HF_BASE="https://huggingface.co/datasets/FanqingM/MMIU-Benchmark/resolve/main"

mkdir -p "$DEST"

ZIPS=(
    "2D-spatial.zip"
    "3D-spatial.zip"
    "Continuous-temporal.zip"
    "Discrete-temporal.zip"
    "High-level-obj-semantic.zip"
    "High-level-sub-semantic.zip"
    "Low-level-semantic.zip"
)

for ZIP in "${ZIPS[@]}"; do
    CATEGORY="${ZIP%.zip}"
    # Skip if already extracted (check for the directory)
    if [ -d "$DEST/$CATEGORY" ]; then
        echo "Already extracted: $CATEGORY — skipping"
        continue
    fi

    if [ ! -f "$DEST/$ZIP" ]; then
        echo "Downloading $ZIP..."
        wget -q --show-progress -O "$DEST/$ZIP" "$HF_BASE/$ZIP"
    else
        echo "Already downloaded: $ZIP — extracting..."
    fi

    unzip -q "$DEST/$ZIP" -d "$DEST"
    rm "$DEST/$ZIP"
    echo "Done: $CATEGORY"
done

echo "MMIU dataset ready at $DEST"
