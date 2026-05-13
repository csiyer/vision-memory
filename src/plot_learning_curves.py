#!/usr/bin/env python3
"""
Learning-curve diagnostic: how does the accuracy CI shrink with trial count?

For each (task, foil, dataset) in TASK_CONFIGS that lands on a Brady-named or
MST dataset: for each model, take the cell with the most trials and project
the Wilson CI that *would* result at subsample sizes k = 5, 10, 20, 40, 80,
... holding the observed accuracy fixed at p_hat. Plots accuracy +/- CI vs k.

Reading the plot: the smallest k where the CI half-width drops below the
effect size you care about is the trial count you actually need.

Skips color_continuous since its accuracy is derived from continuous circular
error, not a binomial proportion — Wilson is inappropriate there.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_scaling_curves import load_full_results
from plot_comprehensive import (
    MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER, TASK_CONFIGS,
)
from metrics import wilson_ci

OUTPUT_DIR = Path("plots/error_bars/learning_curves")
ALLOWED_DATASETS = {"Brady2008", "Brady2013ColorObjects", "MST"}
SKIPPED_TASKS = {"color_continuous"}
SUBSAMPLE_SIZES = [5, 10, 20, 40, 80, 160, 320]


def richest_cell(data, task, foil, dataset, model):
    """Return (k_correct, n_total, n_images_label) for the cell with the most
    valid trials, or None if the model has no proportion-style data here."""
    foil_data = data.get(task, {}).get(model, {}).get(dataset, {}).get(foil, {})
    best = None
    for n_images, cell in foil_data.items():
        trials = cell.get("trials", [])
        valid = [t for t in trials if t.get("correct") is not None]
        if not valid:
            continue
        k = sum(int(t["correct"]) for t in valid)
        n = len(valid)
        if best is None or n > best[1]:
            best = (k, n, n_images)
    return best


def make_learning_curve(data, task, foil, dataset, title):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    plotted = False

    print(f"\n{task}/{foil}/{dataset}:")
    for model in MODEL_ORDER:
        cell = richest_cell(data, task, foil, dataset, model)
        if cell is None:
            print(f"  {model:25s}  no data")
            continue
        k_full, n_full, n_images_label = cell
        p_hat = k_full / n_full
        print(f"  {model:25s}  p_hat={p_hat:.3f}  full_n={n_full}  (from n_images={n_images_label})")

        ks = sorted(set([k for k in SUBSAMPLE_SIZES if k <= n_full] + [n_full]))
        xs, ys, lo_err, hi_err = [], [], [], []
        for k in ks:
            k_hat = int(round(p_hat * k))
            point, lo, hi = wilson_ci(k_hat, k)
            xs.append(k)
            ys.append(100 * point)
            lo_err.append(max(0.0, 100 * (point - lo)))
            hi_err.append(max(0.0, 100 * (hi - point)))

        plotted = True
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err],
                    marker=MODEL_MARKERS.get(model, "o"),
                    markersize=8, linewidth=2.5,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    color=MODEL_COLORS.get(model, "gray"),
                    label=MODEL_LABELS.get(model, model))

    if not plotted:
        plt.close()
        return

    ax.set_xscale("log")
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Accuracy (%)  with 95% Wilson CI")
    base_title = title.split("\n")[0]
    ax.set_title(f"Learning curve — {base_title}\n({dataset}, foil={foil})")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"learning_curve_{task}_{foil}_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    data = load_full_results("results")

    for task, foil, datasets, title in TASK_CONFIGS:
        if task in SKIPPED_TASKS:
            continue
        for dataset in datasets:
            if dataset not in ALLOWED_DATASETS:
                continue
            make_learning_curve(data, task, foil, dataset, title)

    print(f"\nAll learning curves saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
