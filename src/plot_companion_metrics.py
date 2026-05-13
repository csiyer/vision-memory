#!/usr/bin/env python3
"""
Companion-metric plots: d', LDI, mean |error| — the task-appropriate metric
each task is "really" measuring, beyond raw accuracy. Bootstrap 95% CIs.

Restricted to Brady-named datasets (Brady2008, Brady2013ColorObjects). MST
is intentionally dropped here even though it has its own LDI plot — by
request, this folder only carries Brady-style results.

Saved to plots/companion_metrics/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from plot_scaling_curves import load_full_results, model_capability_boundary
from plot_comprehensive import (
    MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER,
    DATASET_LINESTYLES, DATASET_LABELS, TASK_CONFIGS,
    CAPABILITY_X_KWARGS, CAPABILITY_LEGEND_LABEL,
)
from metrics import bootstrap_ci

OUTPUT_DIR = Path("plots/companion_metrics")
ALLOWED_DATASETS = {"Brady2008", "Brady2013ColorObjects"}


def _dprime_2afc(trials):
    valid = [t for t in trials
             if t.get("response", -1) != -1 and t.get("correct") is not None]
    if not valid:
        return float("nan")
    k = sum(int(t["correct"]) for t in valid)
    n = len(valid)
    adj = (k + 0.5) / (n + 1)
    return float(np.sqrt(2) * norm.ppf(adj))


def _dprime_recognition(trials):
    olds = [t for t in trials if t.get("target") == 1]
    news = [t for t in trials if t.get("target") == 0]
    if not olds or not news:
        return float("nan")
    k_h = sum(1 for t in olds if t.get("response") == 1)
    k_f = sum(1 for t in news if t.get("response") == 1)
    adj_h = (k_h + 0.5) / (len(olds) + 1)
    adj_f = (k_f + 0.5) / (len(news) + 1)
    return float(norm.ppf(adj_h) - norm.ppf(adj_f))


def _color_avg_abs_error(trials):
    errs = []
    for t in trials:
        tgt, rep = t.get("target"), t.get("reported")
        if tgt is None or rep is None:
            continue
        diff = (rep - tgt + 180) % 360 - 180
        errs.append(abs(diff))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


COMPANION_METRICS = {
    "2afc":             {"fn": _dprime_2afc,         "label": "d'",                 "slug": "dprime", "ylim": (-0.5, 4.0), "chance": 0.0},
    "continuous":       {"fn": _dprime_recognition,  "label": "d'",                 "slug": "dprime", "ylim": (-0.5, 4.0), "chance": 0.0},
    "color_continuous": {"fn": _color_avg_abs_error, "label": "Mean |error| (deg)", "slug": "abserr", "ylim": (0.0, 95.0), "chance": 90.0},
}


def make_companion_plot(data, task, foil, datasets, title_accuracy, spec):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    has_data = False
    multi_dataset = len(datasets) > 1
    title = title_accuracy.split("\n")[0] + f"\n{spec['label']} vs Sequence Length"

    for dataset in datasets:
        ds_label = DATASET_LABELS.get(dataset, dataset)
        ds_linestyle = DATASET_LINESTYLES.get(dataset, "-")

        for model in MODEL_ORDER:
            foil_data = data.get(task, {}).get(model, {}).get(dataset, {}).get(foil, {})
            if not foil_data:
                continue

            sizes_full = sorted(foil_data.keys())
            sizes, points, lo_err, hi_err = [], [], [], []
            for s in sizes_full:
                trials = foil_data[s].get("trials", [])
                if not trials:
                    continue
                point, lo, hi = bootstrap_ci(trials, spec["fn"], n_boot=1500)
                if np.isnan(point):
                    continue
                sizes.append(s)
                points.append(point)
                lo_err.append(max(0.0, point - lo))
                hi_err.append(max(0.0, hi - point))

            if not sizes:
                continue
            has_data = True
            color = MODEL_COLORS.get(model, "gray")
            marker = MODEL_MARKERS.get(model, "o")
            model_label = MODEL_LABELS.get(model, model)
            label = f"{ds_label} — {model_label}" if multi_dataset else model_label
            ax.errorbar(sizes, points,
                        yerr=[lo_err, hi_err],
                        marker=marker, linestyle=ds_linestyle,
                        markersize=8, linewidth=2.5,
                        elinewidth=1.2, capsize=3, capthick=1.2,
                        color=color, label=label)

            boundary = model_capability_boundary(data, task, foil, dataset, model)
            if boundary is not None and points:
                ax.plot([sizes[-1]], [points[-1]], **CAPABILITY_X_KWARGS)

    if not has_data:
        plt.close()
        return

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 250])
    ax.set_xticklabels(["1", "10", "100", "250"])
    ax.set_xlim(0.8, 300)
    if spec.get("ylim"):
        ax.set_ylim(*spec["ylim"])
    if spec.get("chance") is not None:
        ax.axhline(spec["chance"], color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Study sequence length")
    ax.set_ylabel(spec["label"])
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], **CAPABILITY_X_KWARGS))
    labels.append(CAPABILITY_LEGEND_LABEL)
    ax.legend(handles, labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9, borderaxespad=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    ds_slug = "_".join(d.replace(" ", "_") for d in datasets)
    out_path = OUTPUT_DIR / f"companion_{task}_{foil}_{ds_slug}_{spec['slug']}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    data = load_full_results("results")

    for task, foil, datasets, title in TASK_CONFIGS:
        spec = COMPANION_METRICS.get(task)
        if spec is None:
            continue
        filtered = [d for d in datasets if d in ALLOWED_DATASETS]
        if not filtered:
            continue
        make_companion_plot(data, task, foil, filtered, title, spec)

    print(f"\nAll companion-metric plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
