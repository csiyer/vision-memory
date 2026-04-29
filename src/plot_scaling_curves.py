#!/usr/bin/env python3
"""
Generate scaling curve plots.

Creates plots showing:
- X-axis: Study sequence length (1, 10, 100, 1000) on log scale
- Y-axis: Accuracy (%)
- Separate lines for each foil type (Novel, Exemplar, State)
- Separate subplots for each model/readout method

"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


# Style settings to match reference
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

# Colors matching the reference plot
FOIL_COLORS = {
    'novel': '#E24A33',
    'exemplar': '#348ABD',
    'state': '#B8860B',
    'all': '#7A378B',
}

FOIL_LABELS = {
    'novel': 'Novel',
    'exemplar': 'Exemplar',
    'state': 'State',
    'all': 'All',
}

# Consistent foil ordering across all subplots
FOIL_ORDER = ['novel', 'exemplar', 'state', 'all']

# Only include these study sequence lengths (cap at 250)
VALID_N_IMAGES = {1, 2, 4, 5, 6, 10, 100, 250}


def _apply_axis_style(ax, sizes):
    """Apply standard axis styling; use log scale only when x values are positive."""
    if sizes and max(sizes) > 0:
        ax.set_xscale('log')
        ax.set_xticks([1, 10, 100, 250])
        ax.set_xticklabels(['1', '10', '100', '250'])
        ax.set_xlim(0.8, 300)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Sort legend entries by FOIL_ORDER
        order_map = {FOIL_LABELS[f]: i for i, f in enumerate(FOIL_ORDER) if f in FOIL_LABELS}
        paired = sorted(zip(labels, handles), key=lambda x: order_map.get(x[0], 99))
        labels_sorted, handles_sorted = zip(*paired) if paired else (labels, handles)
        ax.legend(handles_sorted, labels_sorted, loc='upper right')


TASK_GLOB_MAP = {
    "2afc":          "results_2afc_*.json",
    "continuous":    "results_continuous_*.json",
    "pam":           "results_pam_*.json",
    "vhs_single":    "results_vhs_*single_needle*.json",
    "vhs_multi":     "results_vhs_*multi_needle*.json",
    "serial_free":   "results_serial_free_*.json",
    "serial_afc":    "results_serial_afc_*.json",
    "assoc":         "results_assoc_*.json",
}


def load_results(results_dir="results"):
    """Load all task results and organize by task, model, dataset, foil, size."""
    results_path = Path(results_dir)

    # Structure: data[task][model][dataset][foil_type][n_images] = accuracy
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for task_key, glob_pattern in TASK_GLOB_MAP.items():
        for result_file in results_path.glob(glob_pattern):
            try:
                with open(result_file) as f:
                    result = json.load(f)

                metadata = result.get("_metadata", {})
                if not metadata:
                    continue

                models = metadata.get("models", [])
                foil_type = metadata.get("foil_type", "all")
                raw_count = metadata.get("n_images") or metadata.get("image_count", 0) or 0
                n_images = 1 if raw_count == "oracle" else int(raw_count)
                raw_dataset = metadata.get("dataset")
                task_name = metadata.get("task", task_key)
                dataset = raw_dataset if raw_dataset else task_name
                summary = metadata.get("summary", {})

                for model in models:
                    if model == "qwen3-vl":
                        continue
                    if n_images not in VALID_N_IMAGES:
                        continue
                    if foil_type == "accuracy":
                        continue
                    if model in summary:
                        accuracy = summary[model].get("accuracy", 0)
                        data[task_key][model][dataset][foil_type][n_images] = accuracy * 100

            except Exception as e:
                print(f"Error loading {result_file.name}: {e}")
                continue

    return data


def plot_model_scaling(data, task, model, dataset, output_dir="plots"):
    """Create a single scaling plot for a task/model/dataset combination."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get data for this task/model/dataset
    model_data = data.get(task, {}).get(model, {}).get(dataset, {})

    if not model_data:
        print(f"No data for {model} on {dataset}")
        return

    all_sizes = []
    for foil_type in FOIL_ORDER:
        foil_data = model_data.get(foil_type, {})
        if not foil_data:
            continue
        sizes = sorted(foil_data.keys())
        all_sizes.extend(sizes)
        accuracies = [foil_data[s] for s in sizes]
        ax.plot(sizes, accuracies,
                marker='o',
                markersize=8,
                linewidth=2,
                color=FOIL_COLORS.get(foil_type, 'gray'),
                label=FOIL_LABELS.get(foil_type, foil_type))

    ax.set_xlabel('Study sequence length')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model} on {dataset} ({task})')
    _apply_axis_style(ax, all_sizes)

    plt.tight_layout()

    # Save
    filename = f"{output_dir}/scaling_{task}_{model}_{dataset}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison_grid(data, task, output_dir="plots"):
    """Create a grid of plots comparing models/datasets for a given task."""
    Path(output_dir).mkdir(exist_ok=True)

    task_data = data.get(task, {})
    models = list(task_data.keys())
    datasets = set()
    for model_data in task_data.values():
        datasets.update(model_data.keys())
    datasets = sorted(datasets)

    if not models or not datasets:
        print(f"No data to plot for task '{task}'")
        return

    n_cols = len(datasets) if len(datasets) <= 2 else 2
    n_rows = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            if col >= n_cols:
                continue

            ax = axes[row, col]
            model_data = task_data.get(model, {}).get(dataset, {})

            all_sizes = []
            for foil_type in FOIL_ORDER:
                foil_data = model_data.get(foil_type, {})
                if not foil_data:
                    continue
                sizes = sorted(foil_data.keys())
                all_sizes.extend(sizes)
                ax.plot(sizes, [foil_data[s] for s in sizes],
                        marker='o', markersize=6, linewidth=2,
                        color=FOIL_COLORS.get(foil_type, 'gray'),
                        label=FOIL_LABELS.get(foil_type, foil_type))

            if not all_sizes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')

            if row == n_rows - 1:
                ax.set_xlabel('Study sequence length')
            if col == 0:
                ax.set_ylabel('Accuracy (%)')

            ax.set_title(f'{model} - {dataset}')
            _apply_axis_style(ax, all_sizes)

    plt.suptitle(f'{task}: Accuracy vs Study Sequence Length', fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"{output_dir}/scaling_{task}_comparison_grid.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_single_dataset_comparison(data, task, dataset, output_dir="plots"):
    """
    Create a side-by-side plot like the reference image.
    Multiple subplots, one per model, comparing foil types.
    """
    Path(output_dir).mkdir(exist_ok=True)

    task_data = data.get(task, {})
    models = [m for m in task_data.keys() if dataset in task_data[m] and task_data[m][dataset]]

    if not models:
        print(f"No models with data for task '{task}', dataset '{dataset}'")
        return

    n_cols = min(len(models), 3)
    n_rows = (len(models) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = task_data[model][dataset]

        all_sizes = []
        for foil_type in FOIL_ORDER:
            foil_data = model_data.get(foil_type, {})
            if not foil_data:
                continue
            sizes = sorted(foil_data.keys())
            all_sizes.extend(sizes)
            ax.plot(sizes, [foil_data[s] for s in sizes],
                    marker='o', markersize=8, linewidth=2,
                    color=FOIL_COLORS.get(foil_type, 'gray'),
                    label=FOIL_LABELS.get(foil_type, foil_type))

        ax.set_xlabel('Study sequence length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(model)
        _apply_axis_style(ax, all_sizes)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{task} - {dataset}', fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"{output_dir}/scaling_{task}_{dataset}_all_models.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


MODEL_STYLES = {
    'qwen3-vl-8b': {'linestyle': '-',  'marker': 'o'},
    'molmo2-8b':   {'linestyle': '--', 'marker': 's'},
}

MODEL_LABELS = {
    'qwen3-vl-8b': 'Qwen3-VL-8B',
    'molmo2-8b':   'Molmo2-8B',
}


def plot_model_overlap(data, task, dataset, models=None, output_dir="plots"):
    """
    Overlap two (or more) models on the same subplots.

    Layout: one subplot per foil type that has data.
    Each subplot shows both models as separate lines; models are distinguished
    by linestyle/marker, foil types by color.
    """
    Path(output_dir).mkdir(exist_ok=True)

    task_data = data.get(task, {})
    if models is None:
        models = list(task_data.keys())

    # Collect foil types present across all models
    foil_types_present = []
    for ft in FOIL_ORDER:
        for model in models:
            if task_data.get(model, {}).get(dataset, {}).get(ft):
                foil_types_present.append(ft)
                break

    if not foil_types_present:
        print(f"No data for task '{task}', dataset '{dataset}', models {models}")
        return

    n_cols = min(len(foil_types_present), 3)
    n_rows = (len(foil_types_present) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    for idx, foil_type in enumerate(foil_types_present):
        ax = axes_flat[idx]
        all_sizes = []

        for model in models:
            foil_data = task_data.get(model, {}).get(dataset, {}).get(foil_type, {})
            if not foil_data:
                continue
            sizes = sorted(foil_data.keys())
            all_sizes.extend(sizes)
            style = MODEL_STYLES.get(model, {'linestyle': '-', 'marker': 'o'})
            label = MODEL_LABELS.get(model, model)
            ax.plot(sizes, [foil_data[s] for s in sizes],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    markersize=8, linewidth=2,
                    color=FOIL_COLORS.get(foil_type, 'gray'),
                    label=label)

        ax.set_xlabel('Study sequence length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(FOIL_LABELS.get(foil_type, foil_type))
        if all_sizes and max(all_sizes) > 0:
            ax.set_xscale('log')
            ax.set_xticks([1, 10, 100, 250])
            ax.set_xticklabels(['1', '10', '100', '250'])
            ax.set_xlim(0.8, 300)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    for idx in range(len(foil_types_present), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    model_str = ' vs '.join(MODEL_LABELS.get(m, m) for m in models)
    plt.suptitle(f'{task} — {dataset}\n{model_str}', fontsize=14, y=1.02)
    plt.tight_layout()

    models_slug = '_vs_'.join(models)
    filename = f"{output_dir}/scaling_{task}_{models_slug}_{dataset}_overlap.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate scaling curve plots")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to load results from (default: final_results)")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory to save plots (default: plots)")
    parser.add_argument("--comparison-grid", action="store_true",
                        help="Also generate comparison grid plots")
    parser.add_argument("--all-models", action="store_true",
                        help="Also generate per-dataset all-models plots")
    parser.add_argument("--overlap", action="store_true",
                        help="Generate overlap plots comparing qwen3-vl-8b vs molmo2-8b")
    args = parser.parse_args()

    print("Loading results...")
    data = load_results(args.results_dir)

    if not data:
        print("No results found in results/ directory")
        return

    print(f"\nFound data for tasks: {list(data.keys())}")
    for task in data:
        for model in data[task]:
            print(f"  {task} / {model}: {list(data[task][model].keys())}")

    output_dir = args.output_dir

    for task in data:
        task_data = data[task]

        # Individual model/dataset plots (always generated)
        print(f"\nGenerating individual plots for task '{task}'...")
        for model in task_data:
            for dataset in task_data[model]:
                plot_model_scaling(data, task, model, dataset, output_dir)

        if args.comparison_grid:
            print(f"Generating comparison grid for task '{task}'...")
            plot_comparison_grid(data, task, output_dir)

        if args.all_models:
            print(f"Generating per-dataset all-model comparisons for task '{task}'...")
            datasets = set()
            for model_data in task_data.values():
                datasets.update(model_data.keys())
            for dataset in datasets:
                plot_single_dataset_comparison(data, task, dataset, output_dir)

        if args.overlap:
            overlap_tasks = {"2afc", "serial_free", "pam", "vhs_single", "vhs_multi"}
            if task in overlap_tasks:
                print(f"Generating overlap plots for task '{task}'...")
                datasets = set()
                for model_data in task_data.values():
                    datasets.update(model_data.keys())
                for dataset in datasets:
                    plot_model_overlap(data, task, dataset,
                                       models=['qwen3-vl-8b', 'molmo2-8b'],
                                       output_dir=output_dir)

    print(f"\nAll plots saved to '{output_dir}/' directory")


if __name__ == "__main__":
    main()
