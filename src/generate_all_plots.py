#!/usr/bin/env python3
"""
Generate plots from all test results in the results directory.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.plotting import plot_continuous_recognition, plot_2afc, plot_source_memory

def load_results(results_dir="results"):
    """Load all result files and organize by task type."""
    results_path = Path(results_dir)

    # Organize results by task type
    continuous_data = defaultdict(dict)
    afc_data = defaultdict(dict)
    pam_data = defaultdict(dict)

    for result_file in results_path.glob("results_*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Determine file type from filename
            filename = result_file.stem

            # Extract timestamp from filename
            parts = filename.split("_")
            if len(parts) >= 3:
                timestamp = "_".join(parts[2:])
            else:
                timestamp = filename

            # Check if has metadata (newer format)
            has_metadata = "_metadata" in data

            if has_metadata:
                metadata = data["_metadata"]
                task = metadata.get("task", "")

                # Get model data (skip metadata)
                for model_name, model_data in data.items():
                    if model_name == "_metadata":
                        continue

                    # Create unique key for this run
                    run_key = f"{model_name}_{timestamp}"

                    if "Continuous" in task:
                        summary = metadata.get("summary", {}).get(model_name, {})
                        continuous_data[run_key] = summary
                    elif "2-AFC" in task or "AFC" in task:
                        summary = metadata.get("summary", {}).get(model_name, {})
                        afc_data[run_key] = summary
                    elif "Paired Associate" in task or "PAM" in task:
                        summary = metadata.get("summary", {}).get(model_name, {})
                        pam_data[run_key] = summary
            else:
                # Older format - infer from filename
                if "continuous" in filename:
                    # Extract metrics from model data directly
                    for model_name, model_data in data.items():
                        if isinstance(model_data, dict) and "metrics" in model_data:
                            run_key = f"{model_name}_{timestamp}"
                            continuous_data[run_key] = model_data["metrics"]
                elif "2afc" in filename or "afc" in filename:
                    # Handle old 2AFC format
                    print(f"Skipping {result_file.name}: old format without metadata")
                elif "pam" in filename:
                    # Handle old PAM format
                    print(f"Skipping {result_file.name}: old format without metadata")

        except Exception as e:
            print(f"Error loading {result_file.name}: {e}")
            continue

    return continuous_data, afc_data, pam_data

def main():
    print("Loading results...")
    continuous_data, afc_data, pam_data = load_results()

    print(f"\nFound:")
    print(f"  Continuous Recognition: {len(continuous_data)} runs")
    print(f"  2-AFC Recognition: {len(afc_data)} runs")
    print(f"  PAM (Source Memory): {len(pam_data)} runs")

    output_dir = "plots"
    Path(output_dir).mkdir(exist_ok=True)

    # Generate plots
    if continuous_data:
        print(f"\nGenerating continuous recognition plots...")
        plot_continuous_recognition(continuous_data, output_dir)
        print(f"  ✓ Saved to {output_dir}/continuous_recognition.png")

    if afc_data:
        print(f"\nGenerating 2-AFC plots...")
        plot_2afc(afc_data, output_dir)
        print(f"  ✓ Saved to {output_dir}/afc_metrics.png")

    if pam_data:
        print(f"\nGenerating PAM plots...")
        plot_source_memory(pam_data, output_dir)
        print(f"  ✓ Saved to {output_dir}/source_memory.png")

    print(f"\nAll plots saved to '{output_dir}/' directory")

if __name__ == "__main__":
    main()
