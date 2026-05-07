This folder contains a streamed-THINGS representation-probe experiment for ViT^3.

Workflow:

1. Train the linear probe on streamed THINGS categories that are excluded from the held-out local THINGS subset, and evaluate on held-out local THINGS as more training data is added:

```bash
python experiments/vittt_things_linear_probe/train_and_eval_linear_probe.py --device mps
```

2. Train linear probes on all layerwise THINGS losses / gradients, then evaluate them in fresh THINGS `2-AFC` episodes at list lengths `1`, `10`, and `100`:

```bash
python experiments/vittt_things_linear_probe/evaluate_layerwise_metric_probes.py
```

3. Optionally add the old Brady transfer evaluation for the final probe:

```bash
python experiments/vittt_things_linear_probe/train_and_eval_linear_probe.py --device mps --run-brady-eval
```

4. If you want extra sample-efficiency summaries on the cached training features, you can still run:

```bash
python experiments/vittt_things_linear_probe/assess_sample_efficiency.py
```

Key outputs:
- `outputs/things_probe_train_metadata.csv`
- `outputs/things_probe_train_representations.npz`
- `outputs/things_probe_train_layerwise_metrics.npz`
- `outputs/things_probe_heldout_metadata.csv`
- `outputs/things_probe_heldout_representations.npz`
- `outputs/things_probe_heldout_layerwise_metrics.npz`
- `outputs/linear_probe_on_zeroshots/trials.csv`
- `outputs/linear_probe_on_zeroshots/summary.csv`
- `outputs/linear_probe_on_zeroshots/summary.json`
- `outputs/linear_probe_on_zeroshots/vittt_things_linear_probe_eval.png`
- `outputs/things_probe_learning_curve.csv`
- `outputs/things_probe_learning_curve.png`
- `outputs/things_probe_summary.json`
- `outputs/things_probe_metadata.csv`
- `outputs/things_extraction_summary.json`
- `outputs/linear_probe.pkl`
- `outputs/things_train_summary.json`
- `outputs/brady_linear_probe_trials.csv`
- `outputs/brady_linear_probe_summary.csv`
- `outputs/brady_linear_probe_summary.json`
- `outputs/linear_probe_vs_zero_shot.png`
- `outputs/sample_efficiency_per_run.csv`
- `outputs/sample_efficiency_summary.csv`
- `outputs/sample_efficiency_summary.json`
- `outputs/sample_efficiency_learning_curve.png`
