#!/usr/bin/env bash
# Set up 2-AFC recognition experiments for Qwen3VL-8B-instrct
# across foil conditions: novel, exemplar, state.
#
# Usage:
#   bash run_batch_2afc_qwen3vl.sh --dry-run
#   bash run_batch_2afc_qwen3vl.sh
#
# Optional env vars:
#   PYTHON=python3
#   MODEL_ID=Qwen3VL-8B-instrct
#   DATASET=Brady2008
#   N_IMAGES=20
#   N_TRIALS=20

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
MODEL_ID="${MODEL_ID:-Qwen3VL-8B-instrct}"
DATASET="${DATASET:-Brady2008}"
N_IMAGES="${N_IMAGES:-20}"
N_TRIALS="${N_TRIALS:-20}"
TS="$(date +%Y%m%d_%H%M%S)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

run_cmd() {
  echo "$*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

echo "2-AFC Qwen setup"
echo "  Python:   $PY"
echo "  Model:    $MODEL_ID"
echo "  Dataset:  $DATASET"
echo "  N images: $N_IMAGES"
echo "  N trials: $N_TRIALS"
echo "  Dry run:  $DRY_RUN"
echo ""

for foil in novel exemplar state; do
  out="results_2afc_qwen3vl8b_instrct_${foil}_${TS}.json"
  run_cmd "\"$PY\" -m eval_scripts.eval_2afc --models \"$MODEL_ID\" --dataset \"$DATASET\" --n-images \"$N_IMAGES\" --n-trials \"$N_TRIALS\" --foil-type \"$foil\" --output \"$out\""
done

echo ""
echo "Prepared 3 experiment commands."
echo "Results will be written under results/ when executed."
