import argparse
import ast
import csv
import json
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.vit3.vittt import vittt_base
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
    old_grad_raw: float
    foil_grad_raw: float
    old_grad_ratio: float
    foil_grad_ratio: float
    old_linear_score: float
    foil_linear_score: float
    raw_grad_accuracy: float
    ratio_grad_accuracy: float
    linear_probe_accuracy: float


@dataclass
class SummaryResult:
    foil_type: str
    sequence_length: int
    episode_count: int
    evaluated_trials: int
    linear_probe_accuracy_pct: float
    raw_grad_accuracy_pct: float
    ratio_grad_accuracy_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT^3 Brady 2008 2-AFC rerun with gradient-based readouts.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--ttt-lr", type=float, default=1.0, help="TTT learning rate during study encoding.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", None],
        help="Override device selection.",
    )
    parser.add_argument(
        "--gradient-linear-probe",
        type=Path,
        default=REPO_ROOT / "experiments" / "vittt_things_linear_probe" / "outputs" / "linear_probe_on_zeroshots" / "grad_probe.pkl",
        help="Path to the saved THINGS-trained layerwise gradient linear probe.",
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
    return parser.parse_args()


def select_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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


class Brady2AFCRunner:
    def __init__(
        self,
        device: torch.device,
        ttt_lr: float,
        seed: int,
        foil_types: List[str],
        sequence_lengths: List[int],
        gradient_linear_probe_path: Path,
        brady_data_path: Path,
    ) -> None:
        self.device = device
        self.ttt_lr = ttt_lr
        self.seed = seed
        self.rng = random.Random(seed)
        self.foil_types = foil_types
        self.sequence_lengths = sorted(sequence_lengths)
        self.gradient_linear_probe_path = gradient_linear_probe_path
        self.brady_data_path = brady_data_path

        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

        self.objects_dir = REPO_ROOT / "memory_datasets" / "Brady2008Objects"
        self.transform = build_transform()
        self.model = self._load_model()
        self.gradient_linear_probe = self._load_gradient_linear_probe()
        self.human_reference = self._load_human_reference()

        self.object_paths = sorted(
            [
                path
                for path in self.objects_dir.glob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=lambda path: path.name.lower(),
        )
        self.exemplar_pairs = self._load_brady_pairs("Exemplar")
        self.state_pairs = self._load_brady_pairs("State")
        self.baseline_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _load_model(self) -> torch.nn.Module:
        model = vittt_base().to(self.device)
        checkpoint = torch.load(REPO_ROOT / "models" / "vit3" / "vittt_base.pth", map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _load_gradient_linear_probe(self):
        if not self.gradient_linear_probe_path.exists():
            raise FileNotFoundError(f"Missing gradient linear probe: {self.gradient_linear_probe_path}")
        with self.gradient_linear_probe_path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["model"] if isinstance(payload, dict) and "model" in payload else payload

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

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _score_probe_all_layers(self, image_path: Path, memory_state) -> Dict[str, np.ndarray]:
        image_tensor = self._load_image_tensor(image_path)
        with torch.no_grad():
            _, _, _, metrics = self.model(
                image_tensor,
                states=memory_state,
                learning_rate=0.0,
                return_grad_norm=True,
            )
        raw_losses = []
        raw_grads = []
        for block_metrics in metrics:
            if isinstance(block_metrics, list):
                block_metrics = block_metrics[0]
            raw_losses.append(float(block_metrics["ttt_loss"].item()))
            raw_grads.append(float(block_metrics["grad_norm"]))
        raw_losses_arr = np.asarray(raw_losses, dtype=np.float32)
        raw_grads_arr = np.asarray(raw_grads, dtype=np.float32)
        if memory_state is None:
            return {
                "raw_losses": raw_losses_arr,
                "raw_grads": raw_grads_arr,
                "ratio_losses": np.ones_like(raw_losses_arr),
                "ratio_grads": np.ones_like(raw_grads_arr),
            }
        baseline = self._baseline_metrics(image_path)
        return {
            "raw_losses": raw_losses_arr,
            "raw_grads": raw_grads_arr,
            "ratio_losses": raw_losses_arr / (baseline["raw_losses"] + 1e-8),
            "ratio_grads": raw_grads_arr / (baseline["raw_grads"] + 1e-8),
        }

    def _baseline_metrics(self, image_path: Path) -> Dict[str, np.ndarray]:
        key = str(image_path)
        if key not in self.baseline_cache:
            self.baseline_cache[key] = self._score_probe_all_layers(image_path, memory_state=None)
        return self.baseline_cache[key]

    def _score_both(self, image_path: Path, memory_state) -> Dict[str, float]:
        scores = self._score_probe_all_layers(image_path, memory_state)
        grad_vector = scores["raw_grads"].reshape(1, -1)
        return {
            "raw_grad": float(scores["raw_grads"][-1]),
            "ratio_grad": float(scores["ratio_grads"][-1]),
            "linear_probe": float(self.gradient_linear_probe.predict_proba(grad_vector)[0, 1]),
        }

    def _encode_study_sequence(self, study_paths: Sequence[Path]):
        current_state = None
        for image_path in study_paths:
            image_tensor = self._load_image_tensor(image_path)
            with torch.no_grad():
                _, _, current_state, _ = self.model(
                    image_tensor,
                    states=current_state,
                    learning_rate=self.ttt_lr,
                    return_grad_norm=False,
                )
        return current_state

    def _partition_trials(self, items: List, sequence_length: int, total_trials: int) -> List[List]:
        if sequence_length == 1000:
            return [self.rng.sample(items, total_trials)]
        if total_trials % sequence_length != 0:
            raise ValueError(f"Total trials {total_trials} must divide sequence length {sequence_length}.")
        shuffled = items[:]
        self.rng.shuffle(shuffled)
        episode_count = total_trials // sequence_length
        episodes = []
        for index in range(episode_count):
            start = index * sequence_length
            stop = start + sequence_length
            episodes.append(shuffled[start:stop])
        return episodes

    def _novel_episodes(self, sequence_length: int) -> List[Dict]:
        if sequence_length == 1000:
            study_paths = self.rng.sample(self.object_paths, sequence_length)
            probe_targets = self.rng.sample(study_paths, TRIALS_PER_POINT)
            unseen_pool = [path for path in self.object_paths if path not in set(study_paths)]
            foil_paths = self.rng.sample(unseen_pool, TRIALS_PER_POINT)
            trials = list(zip(probe_targets, foil_paths))
            return [{"study_paths": study_paths, "trials": trials}]

        total_studied = TRIALS_PER_POINT
        study_paths = self.rng.sample(self.object_paths, total_studied)
        unseen_pool = [path for path in self.object_paths if path not in set(study_paths)]
        foil_paths = self.rng.sample(unseen_pool, total_studied)
        study_episodes = self._partition_trials(study_paths, sequence_length, TRIALS_PER_POINT)
        foil_episodes = self._partition_trials(foil_paths, sequence_length, TRIALS_PER_POINT)
        episodes = []
        for study_episode, foil_episode in zip(study_episodes, foil_episodes):
            episodes.append({"study_paths": study_episode, "trials": list(zip(study_episode, foil_episode))})
        return episodes

    def _pair_episodes(self, sequence_length: int, pair_paths: List[Tuple[Path, Path]]) -> List[Dict]:
        if sequence_length == 1000:
            probe_pairs = self.rng.sample(pair_paths, TRIALS_PER_POINT)
            studied_targets = [pair[0] for pair in probe_pairs]
            filler_paths = self.rng.sample(self.object_paths, sequence_length - TRIALS_PER_POINT)
            study_paths = studied_targets + filler_paths
            self.rng.shuffle(study_paths)
            return [{"study_paths": study_paths, "trials": probe_pairs}]

        episode_pairs = self._partition_trials(pair_paths, sequence_length, TRIALS_PER_POINT)
        episodes = []
        for pair_chunk in episode_pairs:
            study_paths = [pair[0] for pair in pair_chunk]
            episodes.append({"study_paths": study_paths, "trials": pair_chunk})
        return episodes

    def build_episodes(self, foil_type: str, sequence_length: int) -> List[Dict]:
        if foil_type == "novel":
            return self._novel_episodes(sequence_length)
        if foil_type == "exemplar":
            return self._pair_episodes(sequence_length, self.exemplar_pairs)
        if foil_type == "state":
            return self._pair_episodes(sequence_length, self.state_pairs)
        raise ValueError(f"Unsupported foil type: {foil_type}")

    def evaluate_condition(self, foil_type: str, sequence_length: int) -> Tuple[SummaryResult, List[TrialResult]]:
        episodes = self.build_episodes(foil_type, sequence_length)
        trial_results: List[TrialResult] = []

        for episode_index, episode in enumerate(episodes, start=1):
            memory_state = self._encode_study_sequence(episode["study_paths"])
            for trial_index, pair in enumerate(episode["trials"], start=1):
                target_path, foil_path = pair
                target_scores = self._score_both(target_path, memory_state)
                foil_scores = self._score_both(foil_path, memory_state)

                trial_results.append(
                    TrialResult(
                        foil_type=foil_type,
                        sequence_length=sequence_length,
                        episode_index=episode_index,
                        trial_index_within_episode=trial_index,
                        study_length=len(episode["study_paths"]),
                        target_name=target_path.name,
                        foil_name=foil_path.name,
                        old_grad_raw=target_scores["raw_grad"],
                        foil_grad_raw=foil_scores["raw_grad"],
                        old_grad_ratio=target_scores["ratio_grad"],
                        foil_grad_ratio=foil_scores["ratio_grad"],
                        old_linear_score=target_scores["linear_probe"],
                        foil_linear_score=foil_scores["linear_probe"],
                        raw_grad_accuracy=lower_is_better_accuracy(target_scores["raw_grad"], foil_scores["raw_grad"]),
                        ratio_grad_accuracy=lower_is_better_accuracy(target_scores["ratio_grad"], foil_scores["ratio_grad"]),
                        linear_probe_accuracy=higher_is_better_accuracy(target_scores["linear_probe"], foil_scores["linear_probe"]),
                    )
                )

        summary = SummaryResult(
            foil_type=foil_type,
            sequence_length=sequence_length,
            episode_count=len(episodes),
            evaluated_trials=len(trial_results),
            linear_probe_accuracy_pct=100.0 * mean([trial.linear_probe_accuracy for trial in trial_results]),
            raw_grad_accuracy_pct=100.0 * mean([trial.raw_grad_accuracy for trial in trial_results]),
            ratio_grad_accuracy_pct=100.0 * mean([trial.ratio_grad_accuracy for trial in trial_results]),
        )
        return summary, trial_results

    def write_outputs(self, summaries: List[SummaryResult], trials: List[TrialResult]) -> None:
        with (self.outputs_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
            writer.writeheader()
            for row in summaries:
                writer.writerow(asdict(row))

        with (self.outputs_dir / "trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(trials[0]).keys()))
            writer.writeheader()
            for row in trials:
                writer.writerow(asdict(row))

        payload = {
            "seed": self.seed,
            "ttt_learning_rate": self.ttt_lr,
            "device": str(self.device),
            "trials_per_point": TRIALS_PER_POINT,
            "sequence_lengths": self.sequence_lengths,
            "foil_types": self.foil_types,
            "gradient_linear_probe": str(self.gradient_linear_probe_path),
            "summaries": [asdict(row) for row in summaries],
            "human_reference": self.human_reference,
        }
        with (self.outputs_dir / "summary.json").open("w") as handle:
            json.dump(payload, handle, indent=2)

    def plot_results(self, summaries: List[SummaryResult]) -> None:
        metric_specs = [
            ("linear_probe_accuracy_pct", "Layerwise Gradient Linear Probe"),
            ("raw_grad_accuracy_pct", "Final Layer Raw Gradient"),
            ("ratio_grad_accuracy_pct", "Final Layer Ratio Gradient"),
        ]
        colors = {"novel": "#c03d3d", "exemplar": "#1f78b4", "state": "#c48a1d"}
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
        for axis, (metric_key, title) in zip(axes, metric_specs):
            for foil_type in self.foil_types:
                ordered_rows = sorted(
                    [row for row in summaries if row.foil_type == foil_type],
                    key=lambda row: row.sequence_length,
                )
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
        axes[-1].plot([], [], color="#444444", marker="D", linestyle="None", label="Brady 2008 Human (N=2500)")
        axes[-1].legend(frameon=False, loc="best")
        fig.suptitle("ViT^3 Base on Brady 2008 2-AFC Recognition", fontsize=14)
        fig.savefig(self.outputs_dir / "vittt_brady_2afc_accuracy.png", dpi=250)
        plt.close(fig)

    def run(self) -> Tuple[List[SummaryResult], List[TrialResult]]:
        summaries: List[SummaryResult] = []
        trials: List[TrialResult] = []
        for foil_type in self.foil_types:
            print(f"Running {foil_type} on {self.device}...")
            for sequence_length in self.sequence_lengths:
                summary, trial_rows = self.evaluate_condition(foil_type, sequence_length)
                summaries.append(summary)
                trials.extend(trial_rows)
                print(
                    f"  N={summary.sequence_length}"
                    f" episodes={summary.episode_count}"
                    f" trials={summary.evaluated_trials}"
                    f" linear={summary.linear_probe_accuracy_pct:.2f}%"
                    f" raw_grad={summary.raw_grad_accuracy_pct:.2f}%"
                    f" ratio_grad={summary.ratio_grad_accuracy_pct:.2f}%"
                )
        self.write_outputs(summaries, trials)
        self.plot_results(summaries)
        return summaries, trials


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    runner = Brady2AFCRunner(
        device=select_device(args.device),
        ttt_lr=args.ttt_lr,
        seed=args.seed,
        foil_types=args.foil_types,
        sequence_lengths=args.sequence_lengths,
        gradient_linear_probe_path=args.gradient_linear_probe,
        brady_data_path=args.brady_data,
    )
    runner.run()


if __name__ == "__main__":
    main()
