#!/usr/bin/env bash
# Batch Visual Haystacks evaluations.
#
# Prerequisites:
# 1) Download VHs QA files:
#    hf download --repo-type dataset tsunghanwu/visual_haystacks --local-dir dataset/VHs_qa
# 2) Set API keys for selected models.
#    COCO images are fetched on-demand by default in this script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

echo "Visual Haystacks batch - using: $PY"
echo "Repo root: $ROOT"
echo ""

# On this machine HTTPS cert validation can fail for images.cocodataset.org;
# use HTTP endpoint for on-demand COCO image fetches.
COCO_BASE_URL="${COCO_BASE_URL:-http://images.cocodataset.org}"

# Single-needle quick sweep.
for n in 5 10 20 50; do
  "$PY" -m eval_scripts.eval_vhs \
    --models gemini \
    --mode single_needle \
    --split VHs_large \
    --image-count "$n" \
    --fetch-missing-coco \
    --coco-base-url "$COCO_BASE_URL"
done

# Multi-needle pilot sweep (limit samples to control costs).
for n in 5 10 20; do
  "$PY" -m eval_scripts.eval_vhs \
    --models gemini \
    --mode multi_needle \
    --image-count "$n" \
    --max-samples 100 \
    --fetch-missing-coco \
    --coco-base-url "$COCO_BASE_URL"
done

echo ""
echo "Visual Haystacks batch complete."
