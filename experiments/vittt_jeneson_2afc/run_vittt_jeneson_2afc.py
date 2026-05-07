import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

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
from stimuli import ThingsDataset


HUMAN_BENCHMARK = {
    "controls_accuracy": 0.92,
    "controls_sem": 0.01,
    "hipp_accuracy": 0.83,
    "hipp_sem": 0.01,
    "source": "Jeneson et al. 2010 objects FC-C",
}

READOUT_ORDER = [
    "linear_probe_accuracy",
    "raw_grad_accuracy",
    "ratio_grad_accuracy",
]

READOUT_LABELS = {
    "linear_probe_accuracy": "Layerwise Grad Probe",
    "raw_grad_accuracy": "Final Raw Grad",
    "ratio_grad_accuracy": "Final Ratio Grad",
}

READOUT_COLORS = {
    "linear_probe_accuracy": "#111111",
    "raw_grad_accuracy": "#67a9cf",
    "ratio_grad_accuracy": "#1f78b4",
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
class RunSummary:
    run_index: int
    seed: int
    unique_study_items: int
    study_presentations: int
    evaluated_trials: int
    linear_probe_accuracy: float
    raw_grad_accuracy: float
    ratio_grad_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Jeneson et al. (2010)-style FC-C comparison for ViT^3 on local THINGS.")
    parser.add_argument("--seed", type=int, default=13, help="Base random seed.")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of independent ViT^3 runs.")
    parser.add_argument("--unique-study-items", type=int, default=12, help="Number of unique studied targets per run.")
    parser.add_argument("--study-repeats", type=int, default=2, help="How many times to repeat the study sequence.")
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
        help="Path to the saved THINGS-trained layerwise gradient probe.",
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
        ttt_lr: float,
        gradient_linear_probe_path: Path,
        things_categories: int,
    ) -> None:
        self.device = device
        self.seed = seed
        self.num_runs = num_runs
        self.unique_study_items = unique_study_items
        self.study_repeats = study_repeats
        self.study_presentations = unique_study_items * study_repeats
        self.ttt_lr = ttt_lr
        self.gradient_linear_probe_path = gradient_linear_probe_path

        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

        self.transform = build_transform()
        self.model = self._load_model()
        self.dataset = ThingsDataset(
            n_categories=things_categories,
            exemplars_per_category=2,
            source="local",
        )
        if len(self.dataset) < self.unique_study_items:
            raise ValueError(
                f"Requested {self.unique_study_items} study items, but only {len(self.dataset)} THINGS categories are available locally."
            )
        self.baseline_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.gradient_linear_probe = self._load_gradient_linear_probe()

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

    def _image_key(self, category: str, exemplar_id: int) -> str:
        return f"{category}::exemplar_{exemplar_id}"

    def _image_from_dataset(self, category_index: int, exemplar_index: int) -> Image.Image:
        return self.dataset.get_image(category_index, exemplar_index)

    def _image_tensor_from_pil(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def _score_probe_all_layers(self, image: Image.Image, image_key: str, memory_state) -> Dict[str, np.ndarray]:
        image_tensor = self._image_tensor_from_pil(image)
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
        baseline = self._baseline_metrics(image, image_key)
        return {
            "raw_losses": raw_losses_arr,
            "raw_grads": raw_grads_arr,
            "ratio_losses": raw_losses_arr / (baseline["raw_losses"] + 1e-8),
            "ratio_grads": raw_grads_arr / (baseline["raw_grads"] + 1e-8),
        }

    def _baseline_metrics(self, image: Image.Image, image_key: str) -> Dict[str, np.ndarray]:
        if image_key not in self.baseline_cache:
            self.baseline_cache[image_key] = self._score_probe_all_layers(image, image_key, memory_state=None)
        return self.baseline_cache[image_key]

    def _score_both(self, image: Image.Image, image_key: str, memory_state) -> Dict[str, float]:
        scores = self._score_probe_all_layers(image, image_key, memory_state)
        grad_vector = scores["raw_grads"].reshape(1, -1)
        return {
            "raw_grad": float(scores["raw_grads"][-1]),
            "ratio_grad": float(scores["ratio_grads"][-1]),
            "linear_probe": float(self.gradient_linear_probe.predict_proba(grad_vector)[0, 1]),
        }

    def _encode_study_sequence(self, study_images: Sequence[Image.Image]):
        current_state = None
        for image in study_images:
            image_tensor = self._image_tensor_from_pil(image)
            with torch.no_grad():
                _, _, current_state, _ = self.model(
                    image_tensor,
                    states=current_state,
                    learning_rate=self.ttt_lr,
                    return_grad_norm=False,
                )
        return current_state

    def _sample_run_trials(self, rng: random.Random) -> tuple[List[Image.Image], List[Dict[str, object]]]:
        category_indices = rng.sample(list(range(len(self.dataset))), self.unique_study_items)
        study_images: List[Image.Image] = []
        tests: List[Dict[str, object]] = []
        for category_index in category_indices:
            metadata = self.dataset.get_metadata(category_index, 0)
            category = metadata["category"]
            target_image = self._image_from_dataset(category_index, 0)
            foil_image = self._image_from_dataset(category_index, 1)
            study_images.append(target_image)
            tests.append(
                {
                    "category": category,
                    "target_image": target_image,
                    "foil_image": foil_image,
                    "target_key": self._image_key(category, 0),
                    "foil_key": self._image_key(category, 1),
                }
            )
        order = list(range(len(study_images)))
        rng.shuffle(order)
        ordered_study_images = [study_images[index] for index in order]
        ordered_tests = [tests[index] for index in order]
        return ordered_study_images, ordered_tests

    def run_once(self, run_index: int, seed: int) -> tuple[RunSummary, List[TrialResult]]:
        rng = random.Random(seed)
        ordered_study_images, ordered_tests = self._sample_run_trials(rng)
        study_sequence = ordered_study_images * self.study_repeats
        memory_state = self._encode_study_sequence(study_sequence)

        trial_results: List[TrialResult] = []
        for trial_index, trial in enumerate(ordered_tests, start=1):
            target_scores = self._score_both(trial["target_image"], trial["target_key"], memory_state)
            foil_scores = self._score_both(trial["foil_image"], trial["foil_key"], memory_state)

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

        summary = RunSummary(
            run_index=run_index,
            seed=seed,
            unique_study_items=self.unique_study_items,
            study_presentations=self.study_presentations,
            evaluated_trials=len(trial_results),
            linear_probe_accuracy=mean([row.linear_probe_accuracy for row in trial_results]),
            raw_grad_accuracy=mean([row.raw_grad_accuracy for row in trial_results]),
            ratio_grad_accuracy=mean([row.ratio_grad_accuracy for row in trial_results]),
        )
        return summary, trial_results

    def write_outputs(self, run_summaries: List[RunSummary], trial_rows: List[TrialResult], aggregate_summary: Dict[str, object]) -> None:
        run_summary_path = self.outputs_dir / "jeneson_vittt_run_summary.csv"
        with run_summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(run_summaries[0]).keys()))
            writer.writeheader()
            for row in run_summaries:
                writer.writerow(asdict(row))

        trial_path = self.outputs_dir / "jeneson_vittt_trials.csv"
        with trial_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(trial_rows[0]).keys()))
            writer.writeheader()
            for row in trial_rows:
                writer.writerow(asdict(row))

        with (self.outputs_dir / "jeneson_vittt_summary.json").open("w") as handle:
            json.dump(aggregate_summary, handle, indent=2)

    def plot_main_figure(self, aggregate_summary: Dict[str, object]) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
        for axis, readout_key in zip(axes, READOUT_ORDER):
            vit = aggregate_summary["readouts"][readout_key]
            labels = ["Controls", "Hipp Lesions", "ViT^3"]
            values = [
                HUMAN_BENCHMARK["controls_accuracy"],
                HUMAN_BENCHMARK["hipp_accuracy"],
                vit["mean_accuracy"],
            ]
            errors = [
                HUMAN_BENCHMARK["controls_sem"],
                HUMAN_BENCHMARK["hipp_sem"],
                vit["sem_accuracy"],
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
        ax.set_title("ViT^3 Readout Comparison on Jeneson-Style FC-C")
        fig.savefig(self.outputs_dir / "jeneson_vittt_readout_comparison.png", dpi=250)
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
                f" linear={summary.linear_probe_accuracy:.3f}"
                f" raw_grad={summary.raw_grad_accuracy:.3f}"
                f" ratio_grad={summary.ratio_grad_accuracy:.3f}"
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
        ttt_lr=args.ttt_lr,
        gradient_linear_probe_path=args.gradient_linear_probe,
        things_categories=args.things_categories,
    )
    runner.run()


if __name__ == "__main__":
    main()
