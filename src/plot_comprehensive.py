#!/usr/bin/env python3
"""
Comprehensive per-task plots: all models, all datasets on one axes.

One plot per task. Color encodes model, linestyle encodes dataset.
No forward projections. Saved to plots/comprehensive/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_scaling_curves import load_results

OUTPUT_DIR = Path("plots/comprehensive")

# Model colors and styles
MODEL_COLORS = {
    "qwen3-vl-8b":      "#E8853D",  # orange
    "molmo2-8b":        "#2980B9",  # blue
    "gpt-4o":           "#27AE60",  # green
    "gemini-2.5-flash": "#8E44AD",  # purple
}
MODEL_LABELS = {
    "qwen3-vl-8b":      "Qwen3-VL-8B",
    "molmo2-8b":        "Molmo2-8B",
    "gpt-4o":           "GPT-4o",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
}
MODEL_MARKERS = {
    "qwen3-vl-8b":      "o",
    "molmo2-8b":        "s",
    "gpt-4o":           "^",
    "gemini-2.5-flash": "D",
}
MODEL_ORDER = ["qwen3-vl-8b", "molmo2-8b", "gpt-4o", "gemini-2.5-flash"]

# Linestyle encodes dataset
DATASET_LINESTYLES = {
    "Brady2008":        "-",
    "things":           "--",
    "Visual Haystacks": "-",
}
DATASET_LABELS = {
    "Brady2008":        "Brady2008",
    "things":           "Things",
    "Visual Haystacks": "Visual Haystacks",
}

HUMAN_COLOR = "#222222"
HUMAN_MARKER = "*"
HUMAN_MARKERSIZE = 14

# Per-task config: (task, foil, datasets, title)
TASK_CONFIGS = [
    ("2afc",        "novel",  ["Brady2008", "things"],         "2AFC — Novel\nAccuracy vs Sequence Length"),
    ("2afc",        "exemplar",["Brady2008", "things"],        "2AFC — Exemplar\nAccuracy vs Sequence Length"),
    ("2afc",        "state",  ["Brady2008"],                   "2AFC — State\nAccuracy vs Sequence Length"),
    ("pam",         "all",    ["Brady2008", "things"],         "Paired Associate Memory\nAccuracy vs Sequence Length"),
    ("serial_free", "all",    ["Brady2008", "things"],         "Serial (Free Recall)\nAccuracy vs Sequence Length"),
    ("serial_afc",  "all",    ["Brady2008", "things"],         "Serial AFC\nAccuracy vs Sequence Length"),
    ("assoc",       "all",    ["Brady2008", "things"],         "Associative Memory\nAccuracy vs Sequence Length"),
    ("continuous",  "all",    ["Brady2008", "things"],         "Continuous Recognition\nAccuracy vs Sequence Length"),
    ("vhs_single",  "all",    ["Visual Haystacks"],            "Visual Haystacks — Single Needle\nAccuracy vs Sequence Length"),
    ("vhs_multi",   "all",    ["Visual Haystacks"],            "Visual Haystacks — Multi Needle\nAccuracy vs Sequence Length"),
]

# Human reference points: (task, foil, dataset) -> [(x, accuracy_pct, citation)]
HUMAN_DATA = {
    ("2afc", "novel", "Brady2008"):      [(2500, 93.0, "Brady et al. 2008")],
    ("serial_free", "all", "Brady2008"): [(25,   77.0, "Dubrow & Davachi 2014")],
    ("serial_free", "all", "things"):    [(25,   77.0, "Dubrow & Davachi 2014")],
}

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
})


def _apply_axes(ax, max_human_x=0):
    x_max = max(300, max_human_x * 1.3) if max_human_x > 250 else 300
    xticks = [1, 10, 100, 250]
    xticklabels = ["1", "10", "100", "250"]
    if max_human_x > 250:
        xticks.append(max_human_x)
        xticklabels.append(f"{max_human_x:,}")
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0.8, x_max)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Study sequence length")
    ax.set_ylabel("Accuracy (%)")


def make_comprehensive_plot(data, task, foil, datasets, title):
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

        # Human reference points
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
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

    # Hardcoded Qwen VHS results (from job 9050597)
    data["vhs_single"]["qwen3-vl-8b"]["Visual Haystacks"]["all"][50] = 50.0
    data["vhs_single"]["qwen3-vl-8b"]["Visual Haystacks"]["all"][100] = 50.0

    for task, foil, datasets, title in TASK_CONFIGS:
        make_comprehensive_plot(data, task, foil, datasets, title)

    print(f"\nAll comprehensive plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
