#!/usr/bin/env python3
"""
Per-task accuracy plots with Wilson (or bootstrap) 95% CIs.

Same layout as plot_comprehensive.py but adds error bars and is restricted
to Brady-named datasets (Brady2008, Brady2013ColorObjects) plus MST. THINGS
results are dropped.

Saved to plots/error_bars/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_scaling_curves import load_full_results, model_capability_boundary
from plot_comprehensive import (
    MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER,
    DATASET_LINESTYLES, DATASET_LABELS,
    HUMAN_COLOR, HUMAN_MARKER, HUMAN_MARKERSIZE, HUMAN_DATA,
    TASK_CONFIGS, _apply_axes,
    CAPABILITY_X_KWARGS, CAPABILITY_LEGEND_LABEL,
)
from metrics import wilson_ci, bootstrap_ci

OUTPUT_DIR = Path("plots/error_bars")
ALLOWED_DATASETS = {"Brady2008", "Brady2013ColorObjects", "MST"}


def accuracy_ci(trials, task_key):
    """Return (point_pct, lo_pct, hi_pct) for accuracy on a cell's trials.

    Wilson for proportion-typed tasks (per-trial `correct` field); bootstrap
    for color_continuous where accuracy is derived from circular error.
    """
    if not trials:
        return None
    if task_key == "color_continuous":
        def _acc(ts):
            errs = []
            for t in ts:
                tgt, rep = t.get("target"), t.get("reported")
                if tgt is None or rep is None:
                    continue
                diff = (rep - tgt + 180) % 360 - 180
                errs.append(abs(diff))
            if not errs:
                return 0.0
            return max(0.0, 1.0 - float(np.mean(errs)) / 180.0)
        point, lo, hi = bootstrap_ci(trials, _acc, n_boot=2000)
        return 100 * point, 100 * lo, 100 * hi
    valid = [t for t in trials if t.get("correct") is not None]
    if not valid:
        return None
    k = sum(int(t["correct"]) for t in valid)
    n = len(valid)
    point, lo, hi = wilson_ci(k, n)
    return 100 * point, 100 * lo, 100 * hi


def make_error_bar_plot(data, task, foil, datasets, title):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    max_human_x = 0
    seen_human_citations = set()
    has_data = False
    multi_dataset = len(datasets) > 1

    for dataset in datasets:
        ds_label = DATASET_LABELS.get(dataset, dataset)
        ds_linestyle = DATASET_LINESTYLES.get(dataset, "-")

        for model in MODEL_ORDER:
            foil_data = data.get(task, {}).get(model, {}).get(dataset, {}).get(foil, {})
            if not foil_data:
                continue

            sizes = sorted(foil_data.keys())
            accs, lo_err, hi_err = [], [], []
            for s in sizes:
                cell = foil_data[s]
                ci = accuracy_ci(cell.get("trials", []), task)
                if ci is None:
                    fallback = cell.get("metrics", {}).get("accuracy", 0) * 100
                    accs.append(fallback)
                    lo_err.append(0.0)
                    hi_err.append(0.0)
                else:
                    point, lo, hi = ci
                    accs.append(point)
                    lo_err.append(max(0.0, point - lo))
                    hi_err.append(max(0.0, hi - point))

            has_data = True
            color = MODEL_COLORS.get(model, "gray")
            marker = MODEL_MARKERS.get(model, "o")
            model_label = MODEL_LABELS.get(model, model)
            label = f"{ds_label} — {model_label}" if multi_dataset else model_label

            ax.errorbar(sizes, accs,
                        yerr=[lo_err, hi_err],
                        marker=marker, linestyle=ds_linestyle,
                        markersize=8, linewidth=2.5,
                        elinewidth=1.2, capsize=3, capthick=1.2,
                        color=color, label=label)

            boundary = model_capability_boundary(data, task, foil, dataset, model)
            if boundary is not None and accs:
                ax.plot([sizes[-1]], [accs[-1]], **CAPABILITY_X_KWARGS)

        human_points = HUMAN_DATA.get((task, foil, dataset), [])
        for x, acc, citation in human_points:
            max_human_x = max(max_human_x, x)
            if citation not in seen_human_citations:
                ax.plot(x, acc, marker=HUMAN_MARKER, markersize=HUMAN_MARKERSIZE,
                        color=HUMAN_COLOR, linestyle="None",
                        label=f"Human ({citation})", zorder=5)
                seen_human_citations.add(citation)
            else:
                ax.plot(x, acc, marker=HUMAN_MARKER, markersize=HUMAN_MARKERSIZE,
                        color=HUMAN_COLOR, linestyle="None", zorder=5)

    if not has_data:
        print(f"  No data for {task}/{foil}/{datasets}, skipping.")
        plt.close()
        return

    _apply_axes(ax, max_human_x)
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
    out_path = OUTPUT_DIR / f"errorbars_{task}_{foil}_{ds_slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    data = load_full_results("results")

    for task, foil, datasets, title in TASK_CONFIGS:
        filtered = [d for d in datasets if d in ALLOWED_DATASETS]
        if not filtered:
            print(f"  Skipping {task}/{foil}: no Brady/MST datasets in {datasets}")
            continue
        make_error_bar_plot(data, task, foil, filtered, title)

    print(f"\nAll error-bar plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
