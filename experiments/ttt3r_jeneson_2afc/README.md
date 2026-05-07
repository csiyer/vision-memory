This folder contains a Jeneson et al. (2010)-style `FC-C` recognition-memory comparison for `TTT3R`.

Main figure:

- `Controls` from Jeneson 2010 objects / FC-C: accuracy `0.92`, SEM `0.01`
- `Hippocampal lesions` from Jeneson 2010 objects / FC-C: accuracy `0.83`, SEM `0.01`
- `TTT3R` evaluated on a matched local-THINGS exemplar task

TTT3R task design:

- sample `12` THINGS categories from the first `500` local THINGS categories
- for each pair, choose one image as the studied target and the other as the corresponding foil
- present the `12` studied targets in random order
- repeat the same `12` targets in the same order
- total study sequence length: `24`
- test all `12` targets in `2-AFC` against the corresponding exemplar foil
- repeat the whole experiment `100` times with new random category draws to get error bars

Readouts:

- `mean beta_t`
- `accepted write norm ||ΔS_t||`
- `mean conf_self`

Outputs:

- `outputs/jeneson_ttt3r_trials.csv`
- `outputs/jeneson_ttt3r_run_summary.csv`
- `outputs/jeneson_ttt3r_summary.json`
- `outputs/jeneson_main_figure.png`
- `outputs/jeneson_ttt3r_readout_comparison.png`

Run:

```bash
python experiments/ttt3r_jeneson_2afc/run_ttt3r_jeneson_2afc.py
```

Options:

- `--num-runs 100`
- `--seed 13`
- `--device mps`
- `--unique-study-items 12`
- `--study-repeats 2`
