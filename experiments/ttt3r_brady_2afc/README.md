This folder contains the TTT3R Brady 2008 2-AFC recognition rerun using the four zero-shot memory readouts exposed by the local wrapper.

What this run does:
- Uses the repo's Brady 2008 stimuli.
- Loads the local `models/ttt3r/src/cut3r_512_dpt_4_64.pth` checkpoint.
- Encodes each study sequence once and then branches frozen memory state probes for old-vs-foil choices.
- Evaluates `novel`, `exemplar`, and `state` foil conditions at `1`, `10`, `100`, and `1000`.
- Reads out recognition from three metrics:
  - `beta_t` mean
  - `||delta_s_t||`
  - `mean(conf_self)`
- Overlays the Brady 2008 human reference point from `literature/brady_data.json` at `2500` study items.

Design choices:
- For `novel`, each studied target is paired with an unseen Brady object.
- For `exemplar` and `state`, each target is paired with its Brady foil mate.
- Accuracy is computed by comparing target vs foil scores within each `2-AFC` trial, with ties scored as `0.5`.
- We assume `beta_t` and `mean(conf_self)` should be higher for old targets than foils, while `||delta_s_t||` should be lower for old targets than foils.

Outputs:
- `outputs/summary.csv`
- `outputs/summary.json`
- `outputs/trials.csv`
- `outputs/ttt3r_brady_2afc_accuracy.png`

Run:

```bash
conda run -n dl python experiments/ttt3r_brady_2afc/run_ttt3r_brady_2afc.py
```
