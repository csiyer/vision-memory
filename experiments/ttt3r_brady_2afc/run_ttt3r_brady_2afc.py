import argparse
import ast
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ttt3r import TTT3RMemoryWrapper
from stimuli import BradyDataset


SEQUENCE_LENGTHS = [1, 10, 100, 1000]
FOIL_TYPES = ["novel", "exemplar", "state"]
TRIALS_PER_POINT = 100


@dataclass
class TrialResult:
    foil_type: str
    sequence_length: int
    episode_index: int
    trial_index_within_episode: int
    study_length: int
    target_name: str
    foil_name: str
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
class SummaryResult:
    foil_type: str
    sequence_length: int
    episode_count: int
    evaluated_trials: int
    beta_accuracy_pct: float
    delta_s_accuracy_pct: float
    conf_self_accuracy_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTT3R Brady 2008 2-AFC rerun with zero-shot readouts.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
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
        "--brady-data",
        type=Path,
        default=REPO_ROOT / "literature" / "brady_data.json",
        help="Path to Brady human reference data.",
    )
    parser.add_argument(
        "--foil-types",
        nargs="+",
        choices=FOIL_TYPES,
        default=FOIL_TYPES,
        help="Subset of foil conditions to run.",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=SEQUENCE_LENGTHS,
        help="Subset of sequence lengths to run.",
    )
    parser.add_argument(
        "--trials-per-point",
        type=int,
        default=TRIALS_PER_POINT,
        help="Number of 2-AFC trials to evaluate per condition.",
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


def lower_is_better_accuracy(old_score: float, foil_score: float) -> float:
    if old_score < foil_score:
        return 1.0
    if old_score > foil_score:
        return 0.0
    return 0.5


def higher_is_better_accuracy(old_score: float, foil_score: float) -> float:
    if old_score > foil_score:
        return 1.0
    if old_score < foil_score:
        return 0.0
    return 0.5


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def chunk_list(items: Sequence, chunk_size: int) -> List[List]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def distribute_counts(total: int, buckets: int, per_bucket_cap: int) -> List[int]:
    counts = [0] * buckets
    remaining = total
    for bucket_index in range(buckets):
        buckets_left = buckets - bucket_index
        target = min(per_bucket_cap, (remaining + buckets_left - 1) // buckets_left)
        counts[bucket_index] = target
        remaining -= target
    return counts


class Brady2AFCRunner:
    def __init__(
        self,
        device: torch.device,
        seed: int,
        foil_types: List[str],
        sequence_lengths: List[int],
        brady_data_path: Path,
        trials_per_point: int,
        image_size: int,
    ) -> None:
        self.device = device
        self.seed = seed
        self.rng = random.Random(seed)
        self.foil_types = foil_types
        self.sequence_lengths = sorted(sequence_lengths)
        self.brady_data_path = brady_data_path
        self.trials_per_point = trials_per_point

        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
        self.summary_csv_path = self.outputs_dir / "summary.csv"
        self.trials_csv_path = self.outputs_dir / "trials.csv"
        self.summary_json_path = self.outputs_dir / "summary.json"

        self.objects_dir = REPO_ROOT / "memory_datasets" / "Brady2008Objects"
        self.wrapper = TTT3RMemoryWrapper(device=device, image_size=image_size, verbose=False)
        self.human_reference = self._load_human_reference()

        self.object_paths = sorted(
            [
                path
                for path in self.objects_dir.glob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=lambda path: path.name.lower(),
        )
        self.object_name_to_path = {path.name: path for path in self.object_paths}

        self.exemplar_pairs = self._load_brady_pairs("Exemplar")
        self.state_pairs = self._load_brady_pairs("State")
        self.exemplar_map = {pair[0]: pair[1] for pair in self.exemplar_pairs}
        self.state_map = {pair[0]: pair[1] for pair in self.state_pairs}
        self.exemplar_targets = list(self.exemplar_map.keys())
        self.state_targets = list(self.state_map.keys())

        self._cached_views: Dict[str, dict] = {}
        self.summary_rows: List[SummaryResult] = []
        self.trial_rows: List[TrialResult] = []
        self._load_existing_outputs()

    def _load_human_reference(self) -> Dict[str, float]:
        if not self.brady_data_path.exists():
            raise FileNotFoundError(f"Missing Brady reference data: {self.brady_data_path}")
        raw_text = self.brady_data_path.read_text()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = ast.literal_eval(raw_text)
        return payload["Brady2008AFC"]["accuracy"]

    def _load_brady_pairs(self, dataset_type: str) -> List[Tuple[Path, Path]]:
        dataset = BradyDataset(type=dataset_type)
        return [(pair[0], pair[1]) for pair in dataset.pair_paths]

    def _load_existing_outputs(self) -> None:
        if self.summary_csv_path.exists():
            with self.summary_csv_path.open() as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                required = {
                    "foil_type",
                    "sequence_length",
                    "episode_count",
                    "evaluated_trials",
                    "beta_accuracy_pct",
                    "delta_s_accuracy_pct",
                    "conf_self_accuracy_pct",
                }
                if not required.issubset(fieldnames):
                    self.summary_rows = []
                else:
                    self.summary_rows = [
                        SummaryResult(
                            foil_type=row["foil_type"],
                            sequence_length=int(row["sequence_length"]),
                            episode_count=int(row["episode_count"]),
                            evaluated_trials=int(row["evaluated_trials"]),
                            beta_accuracy_pct=float(row["beta_accuracy_pct"]),
                            delta_s_accuracy_pct=float(row["delta_s_accuracy_pct"]),
                            conf_self_accuracy_pct=float(row["conf_self_accuracy_pct"]),
                        )
                        for row in reader
                    ]
        if self.trials_csv_path.exists():
            with self.trials_csv_path.open() as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                required = {
                    "foil_type",
                    "sequence_length",
                    "episode_index",
                    "trial_index_within_episode",
                    "study_length",
                    "target_name",
                    "foil_name",
                    "old_beta_mean",
                    "foil_beta_mean",
                    "old_delta_s_norm",
                    "foil_delta_s_norm",
                    "old_mean_conf_self",
                    "foil_mean_conf_self",
                    "beta_accuracy",
                    "delta_s_accuracy",
                    "conf_self_accuracy",
                }
                if not required.issubset(fieldnames):
                    self.trial_rows = []
                else:
                    self.trial_rows = [
                        TrialResult(
                            foil_type=row["foil_type"],
                            sequence_length=int(row["sequence_length"]),
                            episode_index=int(row["episode_index"]),
                            trial_index_within_episode=int(row["trial_index_within_episode"]),
                            study_length=int(row["study_length"]),
                            target_name=row["target_name"],
                            foil_name=row["foil_name"],
                            old_beta_mean=float(row["old_beta_mean"]),
                            foil_beta_mean=float(row["foil_beta_mean"]),
                            old_delta_s_norm=float(row["old_delta_s_norm"]),
                            foil_delta_s_norm=float(row["foil_delta_s_norm"]),
                            old_mean_conf_self=float(row["old_mean_conf_self"]),
                            foil_mean_conf_self=float(row["foil_mean_conf_self"]),
                            beta_accuracy=float(row["beta_accuracy"]),
                            delta_s_accuracy=float(row["delta_s_accuracy"]),
                            conf_self_accuracy=float(row["conf_self_accuracy"]),
                        )
                        for row in reader
                    ]

    def _make_view(self, image_path: Path, update: bool = True, reset: bool = False, idx: int = 0) -> dict:
        key = str(image_path)
        if key not in self._cached_views:
            base_view = self.wrapper.prepare_views([image_path], update=True)[0]
            self._cached_views[key] = base_view
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
            "update": torch.tensor(update).unsqueeze(0),
            "reset": torch.tensor(reset).unsqueeze(0),
        }

    def _prepare_study_views(self, study_paths: Sequence[Path]) -> List[dict]:
        return [self._make_view(path, update=True, reset=False, idx=index) for index, path in enumerate(study_paths)]

    def _score_probe(self, image_path: Path, memory_state) -> Dict[str, float]:
        readout = self.wrapper.probe_view(
            view=self._make_view(image_path, update=True, reset=False, idx=0),
            state=memory_state.clone(),
            keep_output=False,
        )
        return {
            "beta": float(readout.beta_mean.mean().item()),
            "delta_s": float(readout.delta_s_norm.mean().item()),
            "conf_self": float(readout.mean_conf_self.mean().item()),
        }

    def _encode_study_sequence(self, study_paths: Sequence[Path]):
        _, memory_state = self.wrapper.study_views(self._prepare_study_views(study_paths), keep_outputs=False)
        return memory_state

    def _sample_without_replacement(self, items: Sequence[Path], count: int) -> List[Path]:
        if count <= 0:
            return []
        return self.rng.sample(list(items), count)

    def _build_shared_episodes(self, sequence_length: int) -> List[dict]:
        requested_pair_types = [foil for foil in self.foil_types if foil in {"exemplar", "state"}]
        pair_items: List[Tuple[str, Path]] = []
        for foil_type in requested_pair_types:
            shuffled_targets = self.exemplar_targets[:] if foil_type == "exemplar" else self.state_targets[:]
            self.rng.shuffle(shuffled_targets)
            pair_items.extend((foil_type, target) for target in shuffled_targets[: self.trials_per_point])

        if pair_items:
            self.rng.shuffle(pair_items)
            episode_count = max(1, (len(pair_items) + sequence_length - 1) // sequence_length)
            pair_chunks = chunk_list(pair_items, sequence_length)
        else:
            episode_count = max(1, (self.trials_per_point + sequence_length - 1) // sequence_length)
            pair_chunks = [[] for _ in range(episode_count)]

        episodes: List[dict] = []
        novel_counts = distribute_counts(self.trials_per_point, episode_count, sequence_length) if "novel" in self.foil_types else [0] * episode_count

        for episode_index in range(episode_count):
            foil_targets: Dict[str, List[Path]] = {foil: [] for foil in requested_pair_types}
            for foil_type, target in pair_chunks[episode_index] if episode_index < len(pair_chunks) else []:
                foil_targets[foil_type].append(target)

            study_paths: List[Path] = []
            for foil in requested_pair_types:
                study_paths.extend(foil_targets[foil])

            fillers_needed = max(0, sequence_length - len(study_paths))
            excluded = set(study_paths)
            filler_pool = [path for path in self.object_paths if path not in excluded]
            if fillers_needed > len(filler_pool):
                raise ValueError(f"Not enough filler items for sequence length {sequence_length}.")
            study_paths.extend(self._sample_without_replacement(filler_pool, fillers_needed))
            self.rng.shuffle(study_paths)

            trials: Dict[str, List[Tuple[Path, Path]]] = {}
            if "novel" in self.foil_types:
                novel_target_count = novel_counts[episode_index]
                novel_targets = self._sample_without_replacement(study_paths, novel_target_count)
                unseen_pool = [path for path in self.object_paths if path not in set(study_paths)]
                novel_foils = self._sample_without_replacement(unseen_pool, novel_target_count)
                trials["novel"] = list(zip(novel_targets, novel_foils))
            if "exemplar" in self.foil_types:
                trials["exemplar"] = [(target, self.exemplar_map[target]) for target in foil_targets.get("exemplar", [])]
            if "state" in self.foil_types:
                trials["state"] = [(target, self.state_map[target]) for target in foil_targets.get("state", [])]

            episodes.append({"study_paths": study_paths, "trials": trials})

        return episodes

    def _drop_rows_for_sequence_length(self, sequence_length: int) -> None:
        self.summary_rows = [row for row in self.summary_rows if row.sequence_length != sequence_length]
        self.trial_rows = [row for row in self.trial_rows if row.sequence_length != sequence_length]

    def evaluate_sequence_length(self, sequence_length: int) -> Tuple[List[SummaryResult], List[TrialResult]]:
        episodes = self._build_shared_episodes(sequence_length)
        trial_results: List[TrialResult] = []

        for episode_index, episode in enumerate(episodes, start=1):
            if episode_index == 1 or episode_index % 10 == 0 or episode_index == len(episodes):
                print(f"    episode {episode_index}/{len(episodes)}")
            memory_state = self._encode_study_sequence(episode["study_paths"])
            for foil_type in self.foil_types:
                for trial_index, pair in enumerate(episode["trials"].get(foil_type, []), start=1):
                    target_path, foil_path = pair
                    target_scores = self._score_probe(target_path, memory_state)
                    foil_scores = self._score_probe(foil_path, memory_state)
                    trial_results.append(
                        TrialResult(
                            foil_type=foil_type,
                            sequence_length=sequence_length,
                            episode_index=episode_index,
                            trial_index_within_episode=trial_index,
                            study_length=len(episode["study_paths"]),
                            target_name=target_path.name,
                            foil_name=foil_path.name,
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

        summaries: List[SummaryResult] = []
        for foil_type in self.foil_types:
            foil_trials = [trial for trial in trial_results if trial.foil_type == foil_type]
            if not foil_trials:
                continue
            summaries.append(
                SummaryResult(
                    foil_type=foil_type,
                    sequence_length=sequence_length,
                    episode_count=len(episodes),
                    evaluated_trials=len(foil_trials),
                    beta_accuracy_pct=100.0 * mean([trial.beta_accuracy for trial in foil_trials]),
                    delta_s_accuracy_pct=100.0 * mean([trial.delta_s_accuracy for trial in foil_trials]),
                    conf_self_accuracy_pct=100.0 * mean([trial.conf_self_accuracy for trial in foil_trials]),
                )
            )
        return summaries, trial_results

    def write_outputs(self) -> None:
        if self.summary_rows:
            with self.summary_csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(self.summary_rows[0]).keys()))
                writer.writeheader()
                for row in sorted(self.summary_rows, key=lambda row: (row.sequence_length, row.foil_type)):
                    writer.writerow(asdict(row))

        if self.trial_rows:
            with self.trials_csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(self.trial_rows[0]).keys()))
                writer.writeheader()
                for row in sorted(self.trial_rows, key=lambda row: (row.sequence_length, row.foil_type, row.episode_index, row.trial_index_within_episode)):
                    writer.writerow(asdict(row))

        payload = {
            "seed": self.seed,
            "device": str(self.device),
            "trials_per_point": self.trials_per_point,
            "sequence_lengths": self.sequence_lengths,
            "foil_types": self.foil_types,
            "summaries": [asdict(row) for row in sorted(self.summary_rows, key=lambda row: (row.sequence_length, row.foil_type))],
            "human_reference": self.human_reference,
        }
        with self.summary_json_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    def plot_results(self) -> None:
        if not self.summary_rows:
            return
        metric_specs = [
            ("beta_accuracy_pct", "Mean Beta_t"),
            ("delta_s_accuracy_pct", "Accepted Write Norm ||ΔS_t||"),
            ("conf_self_accuracy_pct", "Mean conf_self"),
        ]
        colors = {"novel": "#c03d3d", "exemplar": "#1f78b4", "state": "#c48a1d"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        flat_axes = list(axes)
        for axis, (metric_key, title) in zip(flat_axes, metric_specs):
            for foil_type in self.foil_types:
                ordered_rows = sorted(
                    [row for row in self.summary_rows if row.foil_type == foil_type],
                    key=lambda row: row.sequence_length,
                )
                if not ordered_rows:
                    continue
                x_values = [row.sequence_length for row in ordered_rows]
                y_values = [getattr(row, metric_key) for row in ordered_rows]
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2.5,
                    color=colors[foil_type],
                    label=foil_type.title(),
                )
                if foil_type in self.human_reference:
                    axis.scatter(
                        [2500],
                        [100.0 * float(self.human_reference[foil_type])],
                        color=colors[foil_type],
                        marker="D",
                        s=48,
                        zorder=4,
                    )
            axis.axhline(50.0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xscale("log")
            axis.set_xticks(sorted(set(self.sequence_lengths + [2500])))
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axis.set_xlabel("Study sequence length")
            axis.set_ylabel("Accuracy (%)")
            axis.set_ylim(0, 100)
            axis.set_title(title)
        handles, labels = flat_axes[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="#444444", marker="D", linestyle="None"))
        labels.append("Brady 2008 Human (N=2500)")
        flat_axes[-1].legend(handles, labels, frameon=False, loc="best")
        fig.suptitle("TTT3R on Brady 2008 2-AFC Recognition", fontsize=14)
        fig.savefig(self.outputs_dir / "ttt3r_brady_2afc_accuracy.png", dpi=250)
        plt.close(fig)

    def run(self) -> Tuple[List[SummaryResult], List[TrialResult]]:
        existing_keys = {(row.foil_type, row.sequence_length) for row in self.summary_rows}
        for sequence_length in self.sequence_lengths:
            needed_foils = [foil for foil in self.foil_types if (foil, sequence_length) not in existing_keys]
            if not needed_foils:
                print(f"Skipping N={sequence_length}; existing checkpointed summaries found for all foil types.")
                continue
            print(f"Running shared N={sequence_length} on {self.device} for {', '.join(self.foil_types)}...")
            new_summaries, new_trials = self.evaluate_sequence_length(sequence_length)
            self._drop_rows_for_sequence_length(sequence_length)
            self.summary_rows.extend(new_summaries)
            self.trial_rows.extend(new_trials)
            self.write_outputs()
            self.plot_results()
            for summary in sorted(new_summaries, key=lambda row: row.foil_type):
                print(
                    f"  {summary.foil_type} N={summary.sequence_length}"
                    f" episodes={summary.episode_count}"
                    f" trials={summary.evaluated_trials}"
                    f" beta={summary.beta_accuracy_pct:.2f}%"
                    f" delta_s={summary.delta_s_accuracy_pct:.2f}%"
                    f" conf={summary.conf_self_accuracy_pct:.2f}%"
                )
        self.write_outputs()
        self.plot_results()
        return self.summary_rows, self.trial_rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    runner = Brady2AFCRunner(
        device=select_device(args.device),
        seed=args.seed,
        foil_types=args.foil_types,
        sequence_lengths=args.sequence_lengths,
        brady_data_path=args.brady_data,
        trials_per_point=args.trials_per_point,
        image_size=args.image_size,
    )
    runner.run()


if __name__ == "__main__":
    main()
