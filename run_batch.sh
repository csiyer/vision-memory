#!/bin/bash
# Batch evaluation runner — mirrors parameter grids seen under results/:
#   results_2afc_*     — n_images 5,100,200,300,400; foil exemplar+novel; gpt-4o + gemini;
#                        n_trials 50 (default sweep), plus n_images 100 & n_trials 100 (gpt-4o)
#   results_continuous_* — n_images 100 (200 trials) and 200 (400 trials); gpt-4o and gemini
#   results_pam_*      — n_images 20, 100, 200

set -e

echo "Starting batch run..."

# --- 2-AFC (results_2afc_*.json _metadata) ---
python -m eval_scripts.eval_2afc --models gpt-4o --n-images 5 --n-trials 2 --foil-type novel

python -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 100 --foil-type exemplar
python -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 100 --foil-type novel

for n in 100 200 300 400; do
  for foil in exemplar novel; do
    python -m eval_scripts.eval_2afc --models gpt-4o gemini --n-images "$n" --n-trials 50 --foil-type "$foil"
  done
done

# --- Continuous recognition (trial counts in files → n_images 100 → 200 trials, 200 → 400) ---
python -m eval_scripts.eval_continuous --models gpt-4o --n-images 100
python -m eval_scripts.eval_continuous --models gemini --n-images 100
python -m eval_scripts.eval_continuous --models gpt-4o --n-images 200

# --- Paired associate memory ---
for n in 20 100 200; do
  python -m eval_scripts.eval_pam --models gpt-4o gemini --n-images "$n"
done

echo "Batch complete!"
