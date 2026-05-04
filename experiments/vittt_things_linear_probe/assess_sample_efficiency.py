import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vittt_things_linear_probe.train_and_eval_linear_probe import (
    fit_probe,
    load_things_training_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess representation-probe sample efficiency on cached THINGS features.")
    parser.add_argument(
        "--things-metadata",
        type=Path,
        default=RUN_ROOT / "outputs" / "things_probe_metadata.csv",
        help="Cached THINGS metadata table.",
    )
    parser.add_argument(
        "--things-representations",
        type=Path,
        default=RUN_ROOT / "outputs" / "things_probe_representations.npz",
        help="Cached THINGS representation matrix.",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000],
        help="Training row counts to evaluate. Must be even because rows are sampled in old/foil pairs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of random pair subsets to fit per sample size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Base seed for subset sampling and probe fitting.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped_pairs(rows: List[Dict[str, str]]) -> List[List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(row["pair_id"], []).append(idx)
    pairs = list(grouped.values())
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Expected exactly 2 rows per pair, found {len(pair)} for pair indices={pair}")
    return pairs


def sample_indices_from_pairs(all_pairs: List[List[int]], sample_size: int, seed: int) -> List[int]:
    if sample_size % 2 != 0:
        raise ValueError(f"Sample size must be even because rows are sampled in pairs: {sample_size}")
    required_pairs = sample_size // 2
    if required_pairs > len(all_pairs):
        raise ValueError(f"Requested {sample_size} rows but only {len(all_pairs) * 2} rows are available.")
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(all_pairs), size=required_pairs, replace=False)
    sampled_indices = []
    for idx in chosen_indices:
        sampled_indices.extend(all_pairs[int(idx)])
    return sampled_indices


def aggregate_rows(rows: List[Dict], group_keys: List[str], value_key: str) -> List[Dict]:
    grouped: Dict[tuple, List[float]] = {}
    template: Dict[tuple, Dict] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(float(row[value_key]))
        template[key] = {group_key: row[group_key] for group_key in group_keys}
    aggregated = []
    for key in sorted(grouped):
        values = grouped[key]
        item = dict(template[key])
        item[f"{value_key}_mean"] = float(np.mean(values))
        item[f"{value_key}_std"] = float(np.std(values))
        item["num_repeats"] = len(values)
        aggregated.append(item)
    return aggregated


def plot_learning_curve(summary_rows: List[Dict], output_path: Path) -> None:
    subset = sorted(summary_rows, key=lambda row: int(row["sample_size"]))
    x = np.asarray([int(row["sample_size"]) for row in subset], dtype=np.float32)
    mean = np.asarray([float(row["train_auc_mean"]) for row in subset], dtype=np.float32)
    std = np.asarray([float(row["train_auc_std"]) for row in subset], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(x, mean, "o-", color="#111111", linewidth=2.5)
    ax.fill_between(x, mean - std, mean + std, color="#111111", alpha=0.18)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel("THINGS training rows")
    ax.set_ylabel("Train AUC")
    ax.set_title("ViT^3 Representation Probe Sample Efficiency")
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_dir = RUN_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    if not args.things_metadata.exists():
        raise FileNotFoundError(f"Missing THINGS metadata file: {args.things_metadata}")
    if not args.things_representations.exists():
        raise FileNotFoundError(f"Missing THINGS representations file: {args.things_representations}")

    all_rows, all_features, all_labels = load_things_training_arrays(args.things_metadata, args.things_representations)
    all_pairs = grouped_pairs(all_rows)
    requested_sizes = sorted(set(args.sample_sizes))
    per_run_rows: List[Dict] = []

    for sample_size in requested_sizes:
        for repeat_idx in range(args.repeats):
            subset_seed = args.seed + 1000 * repeat_idx + sample_size
            subset_indices = sample_indices_from_pairs(all_pairs, sample_size=sample_size, seed=subset_seed)
            subset_rows = [all_rows[idx] for idx in subset_indices]
            x_train = all_features[subset_indices]
            y_train = all_labels[subset_indices]
            model = fit_probe(x_train, y_train, seed=subset_seed)
            train_probs = model.predict_proba(x_train)[:, 1]
            train_auc = float(roc_auc_score(y_train, train_probs))

            per_run_rows.append(
                {
                    "sample_size": sample_size,
                    "repeat_index": repeat_idx,
                    "subset_seed": subset_seed,
                    "train_auc": train_auc,
                }
            )
            print(
                f"sample_size={sample_size}"
                f" repeat={repeat_idx + 1}/{args.repeats}"
                f" train_auc={train_auc:.4f}"
            )

    write_csv(outputs_dir / "sample_efficiency_per_run.csv", per_run_rows)
    merged_summary = aggregate_rows(per_run_rows, ["sample_size"], "train_auc")
    write_csv(outputs_dir / "sample_efficiency_summary.csv", merged_summary)

    with (outputs_dir / "sample_efficiency_summary.json").open("w") as handle:
        json.dump(
            {
                "sample_sizes": requested_sizes,
                "repeats": args.repeats,
                "summary_rows": merged_summary,
            },
            handle,
            indent=2,
        )

    plot_learning_curve(merged_summary, outputs_dir / "sample_efficiency_learning_curve.png")


if __name__ == "__main__":
    main()
