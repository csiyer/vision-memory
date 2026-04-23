#!/bin/bash
#SBATCH --job-name=mmiu_download
#SBATCH --partition=zgroup1
#SBATCH --account=zgroup
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# Download and extract MMIU dataset images (~25 GB) to shared data dir.
# Usage:
#   sbatch download_mmiu.sh     # via slurm
#   bash   download_mmiu.sh     # interactively

set -e
cd /insomnia001/home/pm3361/vision-memory

DEST=/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/mmiu
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
