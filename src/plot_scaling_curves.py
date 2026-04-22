#!/usr/bin/env python3
"""
Generate scaling curve plots matching the ViT^3 Base reference style.

Creates plots showing:
- X-axis: Study sequence length (1, 10, 100, 1000) on log scale
- Y-axis: Accuracy (%)
- Separate lines for each foil type (Novel, Exemplar, State)
- Separate subplots for each model/readout method

Reference: ViT^3 Base on Brady 2008 2-AFC Recognition plot
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
    'novel': '#E24A33',      # Red
    'exemplar': '#348ABD',   # Blue
    'state': '#B8860B',      # Dark goldenrod/olive
}

FOIL_LABELS = {
    'novel': 'Novel',
    'exemplar': 'Exemplar',
    'state': 'State',
}


def load_results(results_dir="results"):
    """Load all 2-AFC results and organize by model, dataset, foil, size."""
    results_path = Path(results_dir)

    # Structure: data[model][dataset][foil_type][n_images] = accuracy
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for result_file in results_path.glob("results_2afc_*.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)

            metadata = result.get("_metadata", {})
            if not metadata:
                continue

            models = metadata.get("models", [])
            foil_type = metadata.get("foil_type", "unknown")
            n_images = metadata.get("n_images", 0)
            dataset = metadata.get("dataset", "unknown")

            summary = metadata.get("summary", {})

            for model in models:
                if model in summary:
                    accuracy = summary[model].get("accuracy", 0)
                    # Convert to percentage
                    data[model][dataset][foil_type][n_images] = accuracy * 100

        except Exception as e:
            print(f"Error loading {result_file.name}: {e}")
            continue

    return data


def plot_model_scaling(data, model, dataset, output_dir="plots"):
    """Create a single scaling plot for a model/dataset combination."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get data for this model/dataset
    model_data = data.get(model, {}).get(dataset, {})

    if not model_data:
        print(f"No data for {model} on {dataset}")
        return

    # Plot each foil type
    for foil_type in ['novel', 'exemplar', 'state']:
        foil_data = model_data.get(foil_type, {})
        if not foil_data:
            continue

        sizes = sorted(foil_data.keys())
        accuracies = [foil_data[s] for s in sizes]

        ax.plot(sizes, accuracies,
                marker='o',
                markersize=8,
                linewidth=2,
                color=FOIL_COLORS.get(foil_type, 'gray'),
                label=FOIL_LABELS.get(foil_type, foil_type))

    # Styling
    ax.set_xscale('log')
    ax.set_xlabel('Study sequence length')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model} on {dataset} 2-AFC Recognition')

    # Set x-ticks to match reference (1, 10, 100, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1000'])
    ax.set_xlim(0.8, 1500)

    # Y-axis from 0 to 100
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Chance line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    ax.legend(loc='upper right')

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f"{output_dir}/scaling_{model}_{dataset}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison_grid(data, output_dir="plots"):
    """Create a grid of plots comparing models/datasets like the reference."""
    Path(output_dir).mkdir(exist_ok=True)

    # Get all models and datasets
    models = list(data.keys())
    datasets = set()
    for model_data in data.values():
        datasets.update(model_data.keys())
    datasets = sorted(datasets)

    if not models or not datasets:
        print("No data to plot")
        return

    # Create figure with subplots
    n_cols = len(datasets) if len(datasets) <= 2 else 2
    n_rows = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

    # Handle single row/col case
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
            model_data = data.get(model, {}).get(dataset, {})

            # Plot each foil type
            has_data = False
            for foil_type in ['novel', 'exemplar', 'state']:
                foil_data = model_data.get(foil_type, {})
                if not foil_data:
                    continue

                has_data = True
                sizes = sorted(foil_data.keys())
                accuracies = [foil_data[s] for s in sizes]

                ax.plot(sizes, accuracies,
                        marker='o',
                        markersize=6,
                        linewidth=2,
                        color=FOIL_COLORS.get(foil_type, 'gray'),
                        label=FOIL_LABELS.get(foil_type, foil_type))

            if not has_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')

            # Styling
            ax.set_xscale('log')
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels(['1', '10', '100', '1000'])
            ax.set_xlim(0.8, 1500)
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3)

            # Labels
            if row == n_rows - 1:
                ax.set_xlabel('Study sequence length')
            if col == 0:
                ax.set_ylabel('Accuracy (%)')

            # Title for each subplot
            ax.set_title(f'{model} - {dataset}')

            # Legend only on first subplot
            if row == 0 and col == n_cols - 1:
                ax.legend(loc='upper right')

    plt.suptitle('2-AFC Recognition: Accuracy vs Study Sequence Length', fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"{output_dir}/scaling_comparison_grid.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_single_dataset_comparison(data, dataset, output_dir="plots"):
    """
    Create a side-by-side plot like the reference image.
    Multiple subplots, one per model, comparing foil types.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get models that have data for this dataset
    models = [m for m in data.keys() if dataset in data[m] and data[m][dataset]]

    if not models:
        print(f"No models with data for {dataset}")
        return

    n_cols = min(len(models), 3)
    n_rows = (len(models) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))

    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = data[model][dataset]

        for foil_type in ['novel', 'exemplar', 'state']:
            foil_data = model_data.get(foil_type, {})
            if not foil_data:
                continue

            sizes = sorted(foil_data.keys())
            accuracies = [foil_data[s] for s in sizes]

            ax.plot(sizes, accuracies,
                    marker='o',
                    markersize=8,
                    linewidth=2,
                    color=FOIL_COLORS.get(foil_type, 'gray'),
                    label=FOIL_LABELS.get(foil_type, foil_type))

        # Styling
        ax.set_xscale('log')
        ax.set_xlabel('Study sequence length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(model)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['1', '10', '100', '1000'])
        ax.set_xlim(0.8, 1500)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{dataset} 2-AFC Recognition', fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"{output_dir}/scaling_{dataset}_all_models.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("Loading results...")
    data = load_results()

    if not data:
        print("No results found in results/ directory")
        return

    print(f"\nFound data for models: {list(data.keys())}")
    for model in data:
        print(f"  {model}: {list(data[model].keys())}")

    output_dir = "plots"

    # Generate individual model/dataset plots
    print("\nGenerating individual scaling plots...")
    for model in data:
        for dataset in data[model]:
            plot_model_scaling(data, model, dataset, output_dir)

    # Generate comparison grid
    print("\nGenerating comparison grid...")
    plot_comparison_grid(data, output_dir)

    # Generate per-dataset comparisons
    print("\nGenerating per-dataset comparisons...")
    datasets = set()
    for model_data in data.values():
        datasets.update(model_data.keys())

    for dataset in datasets:
        plot_single_dataset_comparison(data, dataset, output_dir)

    print(f"\nAll plots saved to '{output_dir}/' directory")


if __name__ == "__main__":
    main()
