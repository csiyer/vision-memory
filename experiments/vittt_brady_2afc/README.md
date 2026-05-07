This folder contains the ViT^3 Brady 2008 2-AFC recognition rerun using the three gradient-based readouts we kept.

What this run does:
- Uses the repo's Brady 2008 stimuli.
- Loads the local `models/vit3/vittt_base.pth` checkpoint.
- Encodes each study sequence once and probes that frozen memory state on old-vs-foil choices.
- Evaluates `novel`, `exemplar`, and `state` foil conditions at `1`, `10`, `100`, and `1000`.
- Reads out recognition from three metrics:
  - THINGS-trained `layerwise gradient linear probe`
  - `final layer raw gradient`
  - `final layer ratio gradient`
- Overlays the Brady 2008 human reference point from `literature/brady_data.json` at `2500` study items.

Design choices:
- For `novel`, each studied target is paired with an unseen Brady object.
- For `exemplar` and `state`, each target is paired with its Brady foil mate.
- Accuracy is computed by comparing target vs foil scores within each `2-AFC` trial, with ties scored as `0.5`.

Outputs:
- `outputs/summary.csv`
- `outputs/summary.json`
- `outputs/trials.csv`
- `outputs/vittt_brady_2afc_accuracy.png`

Run:

```bash
python experiments/vittt_brady_2afc/run_vittt_brady_2afc.py --device mps
```

Optional:

```bash
python experiments/vittt_brady_2afc/run_vittt_brady_2afc.py --device cpu --seed 7
```
