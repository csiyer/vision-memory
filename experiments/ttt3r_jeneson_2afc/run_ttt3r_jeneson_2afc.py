import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ttt3r import TTT3RMemoryWrapper
from stimuli import ThingsDataset


HUMAN_BENCHMARK = {
    "controls_accuracy": 0.92,
    "controls_sem": 0.01,
    "hipp_accuracy": 0.83,
    "hipp_sem": 0.01,
    "source": "Jeneson et al. 2010 objects FC-C",
}

READOUT_ORDER = [
    "beta_accuracy",
    "delta_s_accuracy",
    "conf_self_accuracy",
]

READOUT_LABELS = {
    "beta_accuracy": "Mean Beta_t",
    "delta_s_accuracy": "Accepted Write Norm ||ΔS_t||",
    "conf_self_accuracy": "Mean conf_self",
}

READOUT_COLORS = {
    "beta_accuracy": "#111111",
    "delta_s_accuracy": "#67a9cf",
    "conf_self_accuracy": "#1f78b4",
}


@dataclass
class TrialResult:
    run_index: int
    seed: int
    trial_index: int
    unique_study_items: int
    study_presentations: int
    target_name: str
    foil_name: str
    category: str
    target_choice_index: int
    foil_choice_index: int
    old_beta_mean: float
    foil_beta_mean: float
    old_delta_s_norm: float
    foil_delta_s_norm: float
    old_mean_conf_self: float
    foil_mean_conf_self: float
    beta_accuracy: float
    delta_s_accuracy: float
    conf_self_accuracy: float


@dataclass
class RunSummary:
    run_index: int
    seed: int
    unique_study_items: int
    study_presentations: int
    evaluated_trials: int
    beta_accuracy: float
    delta_s_accuracy: float
    conf_self_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Jeneson et al. (2010)-style FC-C comparison for TTT3R on local THINGS.")
    parser.add_argument("--seed", type=int, default=13, help="Base random seed.")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of independent TTT3R runs.")
    parser.add_argument("--unique-study-items", type=int, default=12, help="Number of unique studied targets per run.")
    parser.add_argument("--study-repeats", type=int, default=2, help="How many times to repeat the study sequence.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", None],
        help="Override device selection.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="TTT3R input resize passed through its image loader.",
    )
    parser.add_argument(
        "--things-categories",
        type=int,
        default=500,
        help="Number of local THINGS categories to load.",
    )
    return parser.parse_args()


def select_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sem(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def lower_is_better_accuracy(target_score: float, foil_score: float) -> float:
    if target_score < foil_score:
        return 1.0
    if target_score > foil_score:
        return 0.0
    return 0.5


def higher_is_better_accuracy(target_score: float, foil_score: float) -> float:
    if target_score > foil_score:
        return 1.0
    if target_score < foil_score:
        return 0.0
    return 0.5


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


class Jeneson2AFCRunner:
    def __init__(
        self,
        device: torch.device,
        seed: int,
        num_runs: int,
        unique_study_items: int,
        study_repeats: int,
        image_size: int,
        things_categories: int,
    ) -> None:
        self.device = device
        self.seed = seed
        self.num_runs = num_runs
        self.unique_study_items = unique_study_items
        self.study_repeats = study_repeats
        self.study_presentations = unique_study_items * study_repeats

        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

        self.wrapper = TTT3RMemoryWrapper(device=device, image_size=image_size, verbose=False)
        self.dataset = ThingsDataset(
            n_categories=things_categories,
            exemplars_per_category=2,
            source="local",
        )
        if len(self.dataset) < self.unique_study_items:
            raise ValueError(
                f"Requested {self.unique_study_items} study items, but only {len(self.dataset)} THINGS categories are available locally."
            )
        self._cached_views: Dict[str, dict] = {}

    def _image_key(self, category: str, exemplar_id: int) -> str:
        return f"{category}::exemplar_{exemplar_id}"

    def _local_path(self, category_index: int, exemplar_index: int) -> Path:
        metadata = self.dataset.get_metadata(category_index, exemplar_index)
        local_path = metadata.get("local_path")
        if not local_path:
            raise ValueError("TTT3R Jeneson runner requires local THINGS files with metadata['local_path'].")
        return Path(local_path)

    def _make_view(self, image_path: Path, idx: int) -> dict:
        key = str(image_path)
        if key not in self._cached_views:
            self._cached_views[key] = self.wrapper.prepare_views([image_path], update=True)[0]
        base_view = self._cached_views[key]
        return {
            "img": base_view["img"],
            "ray_map": base_view["ray_map"],
            "true_shape": base_view["true_shape"],
            "idx": idx,
            "instance": f"{image_path.name}:{idx}",
            "camera_pose": base_view["camera_pose"],
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }

    def _score_probe(self, image_path: Path, memory_state) -> Dict[str, float]:
        readout = self.wrapper.probe_view(
            view=self._make_view(image_path, idx=0),
            state=memory_state.clone(),
            keep_output=False,
        )
        return {
            "beta": float(readout.beta_mean.mean().item()),
            "delta_s": float(readout.delta_s_norm.mean().item()),
            "conf_self": float(readout.mean_conf_self.mean().item()),
        }

    def _encode_study_sequence(self, study_paths: Sequence[Path]):
        study_views = [self._make_view(path, idx=index) for index, path in enumerate(study_paths)]
        return self.wrapper.encode_views(study_views)

    def _sample_run_trials(self, rng: random.Random) -> tuple[List[Path], List[Dict[str, object]]]:
        category_indices = rng.sample(list(range(len(self.dataset))), self.unique_study_items)
        study_paths: List[Path] = []
        tests: List[Dict[str, object]] = []
        for category_index in category_indices:
            target_meta = self.dataset.get_metadata(category_index, 0)
            category = target_meta["category"]
            target_path = self._local_path(category_index, 0)
            foil_path = self._local_path(category_index, 1)
            study_paths.append(target_path)
            tests.append(
                {
                    "category": category,
                    "target_path": target_path,
                    "foil_path": foil_path,
                    "target_key": self._image_key(category, 0),
                    "foil_key": self._image_key(category, 1),
                }
            )
        order = list(range(len(study_paths)))
        rng.shuffle(order)
        ordered_study_paths = [study_paths[index] for index in order]
        ordered_tests = [tests[index] for index in order]
        return ordered_study_paths, ordered_tests

    def run_once(self, run_index: int, seed: int) -> tuple[RunSummary, List[TrialResult]]:
        rng = random.Random(seed)
        ordered_study_paths, ordered_tests = self._sample_run_trials(rng)
        study_sequence = ordered_study_paths * self.study_repeats
        memory_state = self._encode_study_sequence(study_sequence)

        trial_results: List[TrialResult] = []
        for trial_index, trial in enumerate(ordered_tests, start=1):
            target_scores = self._score_probe(trial["target_path"], memory_state)
            foil_scores = self._score_probe(trial["foil_path"], memory_state)

            if rng.random() < 0.5:
                target_choice_index = 0
                foil_choice_index = 1
            else:
                target_choice_index = 1
                foil_choice_index = 0

            trial_results.append(
                TrialResult(
                    run_index=run_index,
                    seed=seed,
                    trial_index=trial_index,
                    unique_study_items=self.unique_study_items,
                    study_presentations=self.study_presentations,
                    target_name=trial["target_key"],
                    foil_name=trial["foil_key"],
                    category=trial["category"],
                    target_choice_index=target_choice_index,
                    foil_choice_index=foil_choice_index,
                    old_beta_mean=target_scores["beta"],
                    foil_beta_mean=foil_scores["beta"],
                    old_delta_s_norm=target_scores["delta_s"],
                    foil_delta_s_norm=foil_scores["delta_s"],
                    old_mean_conf_self=target_scores["conf_self"],
                    foil_mean_conf_self=foil_scores["conf_self"],
                    beta_accuracy=higher_is_better_accuracy(target_scores["beta"], foil_scores["beta"]),
                    delta_s_accuracy=lower_is_better_accuracy(target_scores["delta_s"], foil_scores["delta_s"]),
                    conf_self_accuracy=higher_is_better_accuracy(target_scores["conf_self"], foil_scores["conf_self"]),
                )
            )

        summary = RunSummary(
            run_index=run_index,
            seed=seed,
            unique_study_items=self.unique_study_items,
            study_presentations=self.study_presentations,
            evaluated_trials=len(trial_results),
            beta_accuracy=mean([row.beta_accuracy for row in trial_results]),
            delta_s_accuracy=mean([row.delta_s_accuracy for row in trial_results]),
            conf_self_accuracy=mean([row.conf_self_accuracy for row in trial_results]),
        )
        return summary, trial_results

    def write_outputs(self, run_summaries: List[RunSummary], trial_rows: List[TrialResult], aggregate_summary: Dict[str, object]) -> None:
        run_summary_path = self.outputs_dir / "jeneson_ttt3r_run_summary.csv"
        with run_summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(run_summaries[0]).keys()))
            writer.writeheader()
            for row in run_summaries:
                writer.writerow(asdict(row))

        trial_path = self.outputs_dir / "jeneson_ttt3r_trials.csv"
        with trial_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(trial_rows[0]).keys()))
            writer.writeheader()
            for row in trial_rows:
                writer.writerow(asdict(row))

        with (self.outputs_dir / "jeneson_ttt3r_summary.json").open("w") as handle:
            json.dump(aggregate_summary, handle, indent=2)

    def plot_main_figure(self, aggregate_summary: Dict[str, object]) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
        for axis, readout_key in zip(axes, READOUT_ORDER):
            ttt3r = aggregate_summary["readouts"][readout_key]
            labels = ["Controls", "Hipp Lesions", "TTT3R"]
            values = [
                HUMAN_BENCHMARK["controls_accuracy"],
                HUMAN_BENCHMARK["hipp_accuracy"],
                ttt3r["mean_accuracy"],
            ]
            errors = [
                HUMAN_BENCHMARK["controls_sem"],
                HUMAN_BENCHMARK["hipp_sem"],
                ttt3r["sem_accuracy"],
            ]
            colors = ["#b8860b", "#c03d3d", READOUT_COLORS[readout_key]]
            x = np.arange(len(labels))
            axis.bar(x, values, yerr=errors, capsize=6, color=colors, edgecolor="#222222", linewidth=1.0)
            axis.set_xticks(x, labels)
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Accuracy")
            axis.set_title(READOUT_LABELS[readout_key])
        fig.suptitle("Jeneson 2010 FC-C Comparison on Local THINGS", fontsize=14)
        fig.savefig(self.outputs_dir / "jeneson_main_figure.png", dpi=250)
        plt.close(fig)

    def plot_readout_comparison(self, aggregate_summary: Dict[str, object]) -> None:
        labels = [READOUT_LABELS[key] for key in READOUT_ORDER]
        values = [aggregate_summary["readouts"][key]["mean_accuracy"] for key in READOUT_ORDER]
        errors = [aggregate_summary["readouts"][key]["sem_accuracy"] for key in READOUT_ORDER]
        colors = [READOUT_COLORS[key] for key in READOUT_ORDER]

        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=6, color=colors, edgecolor="#222222", linewidth=1.0)
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("TTT3R Readout Comparison on Jeneson-Style FC-C")
        fig.savefig(self.outputs_dir / "jeneson_ttt3r_readout_comparison.png", dpi=250)
        plt.close(fig)

    def run(self) -> Dict[str, object]:
        run_summaries: List[RunSummary] = []
        trial_rows: List[TrialResult] = []

        for run_index in range(self.num_runs):
            run_seed = self.seed + run_index
            summary, trials = self.run_once(run_index=run_index + 1, seed=run_seed)
            run_summaries.append(summary)
            trial_rows.extend(trials)
            print(
                f"run={summary.run_index}"
                f" seed={summary.seed}"
                f" beta={summary.beta_accuracy:.3f}"
                f" delta_s={summary.delta_s_accuracy:.3f}"
                f" conf={summary.conf_self_accuracy:.3f}"
            )

        aggregate_summary = {
            "seed": self.seed,
            "num_runs": self.num_runs,
            "device": str(self.device),
            "unique_study_items": self.unique_study_items,
            "study_repeats": self.study_repeats,
            "study_presentations": self.study_presentations,
            "things_categories_loaded": len(self.dataset),
            "human_benchmark": HUMAN_BENCHMARK,
            "readouts": {},
        }

        for readout_key in READOUT_ORDER:
            values = [getattr(summary, readout_key) for summary in run_summaries]
            aggregate_summary["readouts"][readout_key] = {
                "mean_accuracy": mean(values),
                "sem_accuracy": sem(values),
                "per_run_accuracy": values,
            }

        self.write_outputs(run_summaries, trial_rows, aggregate_summary)
        self.plot_main_figure(aggregate_summary)
        self.plot_readout_comparison(aggregate_summary)
        return aggregate_summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    runner = Jeneson2AFCRunner(
        device=select_device(args.device),
        seed=args.seed,
        num_runs=args.num_runs,
        unique_study_items=args.unique_study_items,
        study_repeats=args.study_repeats,
        image_size=args.image_size,
        things_categories=args.things_categories,
    )
    runner.run()


if __name__ == "__main__":
    main()
