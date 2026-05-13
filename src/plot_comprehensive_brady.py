#!/usr/bin/env python3
"""
Brady-restricted comprehensive plots: no error bars, no companion metrics.

Same layout as plot_comprehensive.py but only Brady-named datasets
(Brady2008, Brady2013ColorObjects) and MST. THINGS is dropped.

Saved to plots/comprehensive_brady/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_scaling_curves import load_results, model_capability_boundary
from plot_comprehensive import (
    MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER,
    DATASET_LINESTYLES, DATASET_LABELS,
    HUMAN_COLOR, HUMAN_MARKER, HUMAN_MARKERSIZE, HUMAN_DATA,
    TASK_CONFIGS, _apply_axes,
    CAPABILITY_X_KWARGS, CAPABILITY_LEGEND_LABEL,
)

OUTPUT_DIR = Path("plots/comprehensive_brady")
ALLOWED_DATASETS = {"Brady2008", "Brady2013ColorObjects", "MST"}


def make_brady_plot(data, task, foil, datasets, title):
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
            has_data = True

            sizes = sorted(foil_data.keys())
            accs = [foil_data[s] for s in sizes]
            color = MODEL_COLORS.get(model, "gray")
            marker = MODEL_MARKERS.get(model, "o")
            model_label = MODEL_LABELS.get(model, model)
            label = f"{ds_label} — {model_label}" if multi_dataset else model_label

            ax.plot(sizes, accs,
                    marker=marker,
                    linestyle=ds_linestyle,
                    markersize=8, linewidth=2.5,
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
    out_path = OUTPUT_DIR / f"comprehensive_{task}_{foil}_{ds_slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    data = load_results("results")

    for task, foil, datasets, title in TASK_CONFIGS:
        filtered = [d for d in datasets if d in ALLOWED_DATASETS]
        if not filtered:
            continue
        make_brady_plot(data, task, foil, filtered, title)

    print(f"\nAll Brady-restricted comprehensive plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
