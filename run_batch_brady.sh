#!/usr/bin/env bash
# Batch Brady2008 evaluations: 2-AFC with --foil-type all (novel + exemplar + state),
# plus continuous recognition and paired-associate memory grids.
#
# Prerequisites: unzip memory_datasets (Brady2008* folders), API keys for chosen models,
# and an activated venv if you use one.
#
# Usage: ./run_batch_brady.sh
# Dry run:  bash -n run_batch_brady.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"

echo "Brady2008 batch — using: $PY"
echo "Repo root: $ROOT"
echo ""

# --- 2-AFC: foil-type all only (novel + exemplar + state) ---
# Completed — uncomment to re-run:
# 2026-04-03: pilot 6×6, 100×100 gpt-4o, 100×50 gpt-4o, 100×50 gpt-4o+gemini
# 2026-04-04: 200×50 (gpt-4o, gpt-4o+gemini), 300×50 (gpt-4o+gemini, gpt-4o)

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 6 --n-trials 6 --foil-type all
# results_2afc_20260403_131146.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 100 --n-trials 100 --foil-type all
# results_2afc_20260403_134043.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 100 --n-trials 50 --foil-type all
# results_2afc_20260403_134622.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o gemini --n-images 100 --n-trials 50 --foil-type all
# results_2afc_20260403_135806.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 200 --n-trials 50 --foil-type all
# results_2afc_20260404_172214.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o gemini --n-images 200 --n-trials 50 --foil-type all
# results_2afc_20260404_174149.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o gemini --n-images 300 --n-trials 50 --foil-type all
# results_2afc_20260404_180737.json

# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 300 --n-trials 50 --foil-type all
# results_2afc_20260404_182407.json

# Remaining 2-AFC grid (n_trials=50)
# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o gemini --n-images 400 --n-trials 50 --foil-type all
# "$PY" -m eval_scripts.eval_2afc --dataset Brady2008 --models gpt-4o --n-images 400 --n-trials 50 --foil-type all

# --- Continuous recognition (no Brady2008 results in results/ yet) ---
# "$PY" -m eval_scripts.eval_continuous --dataset Brady2008 --models gpt-4o --n-images 100
# "$PY" -m eval_scripts.eval_continuous --dataset Brady2008 --models gemini --n-images 100
"$PY" -m eval_scripts.eval_continuous --dataset Brady2008 --models gpt-4o gemini --n-images 200

# --- Paired associate memory (no Brady2008 results in results/ yet) ---
"$PY" -m eval_scripts.eval_pam --dataset Brady2008 --models gpt-4o --n-images 20
"$PY" -m eval_scripts.eval_pam --dataset Brady2008 --models gpt-4o gemini --n-images 100
"$PY" -m eval_scripts.eval_pam --dataset Brady2008 --models gpt-4o gemini --n-images 200

echo ""
echo "Brady2008 batch complete."
