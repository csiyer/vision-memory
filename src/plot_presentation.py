#!/usr/bin/env python3
"""
Presentation-ready overlap plots for 4 tasks with simple forward projections.

Generates two sets:
  1. Per-dataset individual plots (Brady2008 / Things / Visual Haystacks)
  2. Combined plots with both datasets on the same axes (one per task)

Missing data points projected by linear extrapolation in log10(x) space.
Human reference points from literature/ overlaid as black stars.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_scaling_curves import load_results, MODEL_STYLES, MODEL_LABELS

OUTPUT_DIR = Path("plots/presentation")
PROJECT_TO = [100, 250]

PURPLE = "#6B2D8B"

MODEL_COLORS = {
    "qwen3-vl-8b": "#E8853D",  # orange
    "molmo2-8b":   "#2980B9",  # blue
}

# Linestyle per dataset for combined plots (color encodes model, linestyle encodes dataset)
DATASET_LINESTYLES = {
    "Brady2008":        "-",
    "things":           "--",
    "Visual Haystacks": "-",
}

DATASET_COLORS = {
    "Brady2008":        "#6B2D8B",  # purple (unused in combined plots, kept for reference)
    "things":           "#17A589",  # teal
    "Visual Haystacks": PURPLE,
}
DATASET_LABELS = {
    "Brady2008": "Brady2008",
    "things": "Things",
    "Visual Haystacks": "Visual Haystacks",
}

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})

# Individual per-dataset plots
TASK_CONFIGS = [
    # (task,          foil,    dataset,            title)
    ("2afc",          "novel", "Brady2008",        "Recognition — Novel (Brady2008)\nAccuracy vs Sequence Length"),
    ("pam",           "all",   "Brady2008",        "Paired Associate Memory (Brady2008)\nAccuracy vs Sequence Length"),
    ("serial_free",   "all",   "Brady2008",        "Serial (Free Recall) (Brady2008)\nAccuracy vs Sequence Length"),
    ("vhs_single",    "all",   "Visual Haystacks", "Visual Haystacks Single-Needle\nAccuracy vs Sequence Length"),
    ("2afc",          "novel", "things",           "Recognition — Novel (Things)\nAccuracy vs Sequence Length"),
    ("pam",           "all",   "things",           "Paired Associate Memory (Things)\nAccuracy vs Sequence Length"),
    ("serial_free",   "all",   "things",           "Serial (Free Recall) (Things)\nAccuracy vs Sequence Length"),
]

# Combined per-task plots (collapse datasets onto one axes)
COMBINED_TASK_CONFIGS = [
    # (task,          foil,    datasets,                        title)
    ("2afc",          "novel", ["Brady2008", "things"],         "2AFC — Novel\nAccuracy vs Sequence Length"),
    ("pam",           "all",   ["Brady2008", "things"],         "Paired Associate Memory\nAccuracy vs Sequence Length"),
    ("serial_free",   "all",   ["Brady2008", "things"],         "Serial (Free Recall)\nAccuracy vs Sequence Length"),
    ("vhs_single",    "all",   ["Visual Haystacks"],            "Visual Haystacks Single-Needle\nAccuracy vs Sequence Length"),
]

MODELS = ["qwen3-vl-8b", "molmo2-8b"]

# Human reference points: (task, foil, dataset) -> [(x, accuracy_pct, citation)]
HUMAN_DATA = {
    ("2afc", "novel", "Brady2008"):      [(2500, 93.0, "Brady et al. 2008")],
    ("serial_free", "all", "Brady2008"): [(25,   77.0, "Dubrow & Davachi 2014")],
    ("serial_free", "all", "things"):    [(25,   77.0, "Dubrow & Davachi 2014")],
}

HUMAN_COLOR = "#222222"
HUMAN_MARKER = "*"
HUMAN_MARKERSIZE = 14


def project_forward(sizes, accuracies, targets):
    """Linear extrapolation in log10(x) space from the last two known points."""
    if len(sizes) < 2:
        return {}
    log_x1, log_x2 = np.log10(sizes[-2]), np.log10(sizes[-1])
    y1, y2 = accuracies[-2], accuracies[-1]
    slope = (y2 - y1) / (log_x2 - log_x1) if log_x2 != log_x1 else 0
    result = {}
    for t in targets:
        if t not in sizes:
            y = y2 + slope * (np.log10(t) - log_x2)
            result[t] = float(np.clip(y, 1, 100))
    return result


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


def _plot_model_lines(ax, data, task, foil, dataset, color=None, label_prefix="", project=True, dataset_linestyle=None):
    """Plot both models for one task/foil/dataset combination onto ax."""
    for model in MODELS:
        foil_data = data.get(task, {}).get(model, {}).get(dataset, {}).get(foil, {})
        if not foil_data:
            continue

        sizes = sorted(foil_data.keys())
        accs = [foil_data[s] for s in sizes]
        style = MODEL_STYLES.get(model, {"linestyle": "-", "marker": "o"})
        model_label = MODEL_LABELS.get(model, model)
        label = f"{label_prefix}{model_label}" if label_prefix else model_label
        is_qwen = model == "qwen3-vl-8b"
        model_color = MODEL_COLORS.get(model, color)
        ls = dataset_linestyle if dataset_linestyle is not None else style["linestyle"]

        ax.plot(sizes, accs,
                marker=style["marker"],
                linestyle=ls,
                markersize=9, linewidth=2.5,
                color=model_color, label=label)

        if not project:
            continue

        proj = project_forward(sizes, accs, PROJECT_TO)
        if proj:
            proj_sizes = sorted(proj.keys())
            proj_accs = [proj[s] for s in proj_sizes]
            conn_x = [sizes[-1]] + proj_sizes
            conn_y = [accs[-1]] + proj_accs
            if is_qwen:
                ax.plot(conn_x, conn_y,
                        marker=style["marker"],
                        linestyle=ls,
                        markersize=9, linewidth=2.5,
                        color=model_color)
            else:
                proj_label = f"{label_prefix}{model_label} (projected)" if label_prefix else f"{model_label} (projected)"
                ax.plot(conn_x, conn_y,
                        marker=style["marker"],
                        linestyle=":",
                        markersize=9, linewidth=2,
                        color=model_color, alpha=0.45,
                        markerfacecolor="white",
                        markeredgewidth=2,
                        label=proj_label)


def make_plot(data, task, foil, dataset, title, output_dir, project=True):
    fig, ax = plt.subplots(figsize=(7, 6.2))

    _plot_model_lines(ax, data, task, foil, dataset, color=PURPLE, project=project)

    human_points = HUMAN_DATA.get((task, foil, dataset), [])
    max_human_x = max((x for x, _, _ in human_points), default=0)
    for x, acc, citation in human_points:
        ax.plot(x, acc, marker=HUMAN_MARKER, markersize=HUMAN_MARKERSIZE,
                color=HUMAN_COLOR, linestyle="None",
                label=f"Human ({citation})", zorder=5)

    _apply_axes(ax, max_human_x)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9, borderaxespad=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    suffix = "" if project else "_noproj"
    out_path = output_dir / f"presentation_{task}_{foil}_{dataset.replace(' ', '_')}{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def make_combined_plot(data, task, foil, datasets, title, output_dir, project=True):
    fig, ax = plt.subplots(figsize=(7, 6.2))

    max_human_x = 0
    seen_human_citations = set()

    for dataset in datasets:
        ds_label = DATASET_LABELS.get(dataset, dataset)
        label_prefix = f"{ds_label} — " if len(datasets) > 1 else ""
        ds_linestyle = DATASET_LINESTYLES.get(dataset, "-") if len(datasets) > 1 else None
        _plot_model_lines(ax, data, task, foil, dataset,
                          label_prefix=label_prefix, project=project,
                          dataset_linestyle=ds_linestyle)

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

    _apply_axes(ax, max_human_x)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9, borderaxespad=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    suffix = "" if project else "_noproj"
    out_path = output_dir / f"presentation_{task}_{foil}_combined{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading results...")
    data = load_results("results")

    # Hardcoded Qwen VHS results for n=50 and n=100 (from job 9050597)
    data["vhs_single"]["qwen3-vl-8b"]["Visual Haystacks"]["all"][50] = 50.0
    data["vhs_single"]["qwen3-vl-8b"]["Visual Haystacks"]["all"][100] = 50.0

    for project in [True, False]:
        tag = "with projections" if project else "no projections"

        print(f"\n--- Individual dataset plots ({tag}) ---")
        for task, foil, dataset, title in TASK_CONFIGS:
            make_plot(data, task, foil, dataset, title, OUTPUT_DIR, project=project)

        print(f"\n--- Combined dataset plots ({tag}) ---")
        for task, foil, datasets, title in COMBINED_TASK_CONFIGS:
            make_combined_plot(data, task, foil, datasets, title, OUTPUT_DIR, project=project)

    print(f"\nAll presentation plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()