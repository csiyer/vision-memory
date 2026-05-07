import argparse
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


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vittt_brady_2afc.run_vittt_brady_2afc import Brady2AFCRunner, select_device
from experiments.vittt_things_linear_probe.train_and_eval_linear_probe import fit_probe
from stimuli import ThingsDataset


FOIL_TYPES = ["novel", "exemplar"]
SEQUENCE_LENGTHS = [1, 10, 100]
TRIALS_PER_POINT = 100


@dataclass
class TrialResult:
    probe_type: str
    foil_type: str
    sequence_length: int
    episode_index: int
    trial_index_within_episode: int
    study_length: int
    target_key: str
    foil_key: str
    old_familiarity: float
    foil_familiarity: float
    linear_probe_accuracy: float


@dataclass
class SummaryResult:
    probe_type: str
    foil_type: str
    sequence_length: int
    episode_count: int
    evaluated_trials: int
    linear_probe_accuracy_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate layerwise loss/grad linear probes on THINGS 2-AFC episodes."
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=RUN_ROOT / "outputs" / "things_probe_train_metadata.csv",
        help="Train-split THINGS metadata table.",
    )
    parser.add_argument(
        "--train-layerwise-metrics",
        type=Path,
        default=RUN_ROOT / "outputs" / "things_probe_train_layerwise_metrics.npz",
        help="Train-split layerwise loss/gradient arrays.",
    )
    parser.add_argument(
        "--extraction-summary",
        type=Path,
        default=RUN_ROOT / "outputs" / "things_extraction_summary.json",
        help="Summary from the streamed-train/local-heldout probe run.",
    )
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
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=SEQUENCE_LENGTHS,
        help="Sequence lengths to evaluate.",
    )
    parser.add_argument(
        "--trials-per-point",
        type=int,
        default=TRIALS_PER_POINT,
        help="Total target-vs-foil trials per condition.",
    )
    return parser.parse_args()


def compare_scores(old_score: float, foil_score: float) -> float:
    if old_score > foil_score:
        return 1.0
    if old_score < foil_score:
        return 0.0
    return 0.5


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_train_probe_models(
    metadata_path: Path,
    metrics_path: Path,
    seed: int,
) -> Dict[str, object]:
    rows = load_csv_rows(metadata_path)
    labels = np.asarray([int(row["is_old"]) for row in rows], dtype=np.int64)
    payload = np.load(metrics_path)
    return {
        "loss_probe": fit_probe(np.asarray(payload["raw_losses"], dtype=np.float32), labels, seed=seed),
        "grad_probe": fit_probe(np.asarray(payload["raw_grads"], dtype=np.float32), labels, seed=seed),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class LayerwiseProbeEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = select_device(args.device)
        self.rng = random.Random(args.seed)
        self.sequence_lengths = sorted(args.sequence_lengths)
        self.trials_per_point = args.trials_per_point

        self.outputs_dir = RUN_ROOT / "outputs" / "linear_probe_on_zeroshots"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "outputs" / ".mplconfig").mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.engine = Brady2AFCRunner(
            device=self.device,
            ttt_lr=args.ttt_lr,
            seed=args.seed,
            foil_types=["novel"],
            sequence_lengths=[1],
        )

        with args.extraction_summary.open() as handle:
            extraction_summary = json.load(handle)
        heldout_categories = int(extraction_summary["heldout_categories_loaded"])
        self.dataset = ThingsDataset(
            n_categories=heldout_categories,
            exemplars_per_category=2,
            source="local",
        )
        if len(self.dataset) < heldout_categories:
            raise ValueError(
                f"Loaded only {len(self.dataset)} held-out local categories, fewer than expected {heldout_categories}."
            )
        if len(self.dataset) < 2 * max(self.sequence_lengths):
            raise ValueError(
                f"Held-out local dataset has {len(self.dataset)} categories, but evaluation up to N={max(self.sequence_lengths)} "
                "needs at least twice that many for novel foil episodes."
            )

        self.probe_models = load_train_probe_models(
            metadata_path=args.train_metadata,
            metrics_path=args.train_layerwise_metrics,
            seed=args.seed,
        )

    def _image_key(self, category: str, exemplar_id: int) -> str:
        return f"{category}::exemplar_{exemplar_id}"

    def _score_pil_all_layers(self, image, memory_state) -> Dict[str, np.ndarray]:
        image_tensor = self.engine.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, metrics = self.engine.model(
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
        return {
            "raw_losses": np.asarray(raw_losses, dtype=np.float32),
            "raw_grads": np.asarray(raw_grads, dtype=np.float32),
        }

    def _encode_study_sequence(self, study_images: Sequence) -> List[Dict[str, torch.Tensor]]:
        current_state = None
        for image in study_images:
            image_tensor = self.engine.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, current_state, _ = self.engine.model(
                    image_tensor,
                    states=current_state,
                    learning_rate=self.args.ttt_lr,
                    return_grad_norm=False,
                )
        return current_state

    def _sample_exemplar_episode(self, sequence_length: int, probe_trials: int | None = None) -> Dict:
        category_indices = self.rng.sample(list(range(len(self.dataset))), sequence_length)
        study_items = []
        probe_pairs = []
        for idx in category_indices:
            metadata = self.dataset.get_metadata(idx, 0)
            category = metadata["category"]
            study_image = self.dataset.get_image(idx, 0)
            foil_image = self.dataset.get_image(idx, 1)
            study_items.append((study_image, category, 0))
            probe_pairs.append(
                {
                    "category": category,
                    "target": (study_image, 0),
                    "foil": (foil_image, 1),
                }
            )
        if probe_trials is not None and probe_trials < len(probe_pairs):
            probe_pairs = self.rng.sample(probe_pairs, probe_trials)
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def _sample_novel_episode(self, sequence_length: int, probe_trials: int | None = None) -> Dict:
        probe_trials = sequence_length if probe_trials is None else probe_trials
        category_indices = self.rng.sample(list(range(len(self.dataset))), sequence_length + probe_trials)
        study_indices = category_indices[:sequence_length]
        foil_indices = category_indices[sequence_length : sequence_length + probe_trials]
        study_items = []
        probe_pairs = []
        if probe_trials < sequence_length:
            chosen_study_indices = self.rng.sample(study_indices, probe_trials)
        else:
            chosen_study_indices = study_indices
        for study_idx, foil_idx in zip(chosen_study_indices, foil_indices):
            study_meta = self.dataset.get_metadata(study_idx, 0)
            foil_meta = self.dataset.get_metadata(foil_idx, 0)
            study_image = self.dataset.get_image(study_idx, 0)
            foil_image = self.dataset.get_image(foil_idx, 0)
            study_items.append((study_image, study_meta["category"], 0))
            probe_pairs.append(
                {
                    "category": study_meta["category"],
                    "target": (study_image, 0),
                    "foil": (foil_image, 0),
                    "foil_category": foil_meta["category"],
                }
            )
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def build_episodes(self, foil_type: str, sequence_length: int) -> List[Dict]:
        episodes = []
        if sequence_length >= self.trials_per_point:
            num_episodes = 1
            probe_trials = self.trials_per_point
        else:
            if self.trials_per_point % sequence_length != 0:
                raise ValueError(
                    f"trials_per_point={self.trials_per_point} must divide sequence_length={sequence_length} when sequence_length < trials_per_point."
                )
            num_episodes = self.trials_per_point // sequence_length
            probe_trials = sequence_length
        for _ in range(num_episodes):
            if foil_type == "novel":
                episodes.append(self._sample_novel_episode(sequence_length, probe_trials=probe_trials))
            elif foil_type == "exemplar":
                episodes.append(self._sample_exemplar_episode(sequence_length, probe_trials=probe_trials))
            else:
                raise ValueError(f"Unsupported foil type: {foil_type}")
        return episodes

    def evaluate_condition(self, probe_type: str, foil_type: str, sequence_length: int) -> Tuple[SummaryResult, List[TrialResult]]:
        model = self.probe_models[probe_type]
        feature_key = "raw_losses" if probe_type == "loss_probe" else "raw_grads"
        episodes = self.build_episodes(foil_type, sequence_length)
        trial_results: List[TrialResult] = []

        for episode_index, episode in enumerate(episodes, start=1):
            study_images = [item[0] for item in episode["study_items"]]
            memory_state = self._encode_study_sequence(study_images)
            for trial_index, pair in enumerate(episode["probe_pairs"], start=1):
                target_image, target_exemplar = pair["target"]
                foil_image, foil_exemplar = pair["foil"]
                target_category = pair["category"]
                foil_category = target_category if foil_type == "exemplar" else pair["foil_category"]
                target_key = self._image_key(target_category, target_exemplar)
                foil_key = self._image_key(foil_category, foil_exemplar)

                target_metrics = self._score_pil_all_layers(target_image, memory_state)[feature_key].reshape(1, -1)
                foil_metrics = self._score_pil_all_layers(foil_image, memory_state)[feature_key].reshape(1, -1)
                old_prob = float(model.predict_proba(target_metrics)[0, 1])
                foil_prob = float(model.predict_proba(foil_metrics)[0, 1])

                trial_results.append(
                    TrialResult(
                        probe_type=probe_type,
                        foil_type=foil_type,
                        sequence_length=sequence_length,
                        episode_index=episode_index,
                        trial_index_within_episode=trial_index,
                        study_length=len(study_images),
                        target_key=target_key,
                        foil_key=foil_key,
                        old_familiarity=old_prob,
                        foil_familiarity=foil_prob,
                        linear_probe_accuracy=compare_scores(old_prob, foil_prob),
                    )
                )

        summary = SummaryResult(
            probe_type=probe_type,
            foil_type=foil_type,
            sequence_length=sequence_length,
            episode_count=len(episodes),
            evaluated_trials=len(trial_results),
            linear_probe_accuracy_pct=100.0 * mean([trial.linear_probe_accuracy for trial in trial_results]),
        )
        return summary, trial_results

    def write_outputs(self, summaries: List[SummaryResult], trials: List[TrialResult]) -> None:
        write_csv(self.outputs_dir / "summary.csv", [asdict(row) for row in summaries])
        write_csv(self.outputs_dir / "trials.csv", [asdict(row) for row in trials])
        with (self.outputs_dir / "summary.json").open("w") as handle:
            json.dump(
                {
                    "seed": self.args.seed,
                    "ttt_learning_rate": self.args.ttt_lr,
                    "device": str(self.device),
                    "trials_per_point": self.trials_per_point,
                    "sequence_lengths": self.sequence_lengths,
                    "foil_types": FOIL_TYPES,
                    "probe_types": list(self.probe_models.keys()),
                    "things_local_categories_loaded": len(self.dataset),
                    "summaries": [asdict(row) for row in summaries],
                },
                handle,
                indent=2,
            )
        with (self.outputs_dir / "loss_probe.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "probe_type": "loss_probe",
                    "model": self.probe_models["loss_probe"],
                    "seed": self.args.seed,
                    "ttt_learning_rate": self.args.ttt_lr,
                    "sequence_lengths": self.sequence_lengths,
                    "foil_types": FOIL_TYPES,
                },
                handle,
            )
        with (self.outputs_dir / "grad_probe.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "probe_type": "grad_probe",
                    "model": self.probe_models["grad_probe"],
                    "seed": self.args.seed,
                    "ttt_learning_rate": self.args.ttt_lr,
                    "sequence_lengths": self.sequence_lengths,
                    "foil_types": FOIL_TYPES,
                },
                handle,
            )

    def plot_results(self, summaries: List[SummaryResult]) -> None:
        palette = {
            "loss_probe": "#ef8a62",
            "grad_probe": "#67a9cf",
        }
        labels = {
            "loss_probe": "Layerwise Loss Probe",
            "grad_probe": "Layerwise Grad Probe",
        }
        fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
        for row_index, foil_type in enumerate(FOIL_TYPES):
            axis = axes[row_index]
            subset = [row for row in summaries if row.foil_type == foil_type]
            for probe_type in ["loss_probe", "grad_probe"]:
                ordered = sorted(
                    [row for row in subset if row.probe_type == probe_type],
                    key=lambda row: row.sequence_length,
                )
                x_values = [row.sequence_length for row in ordered]
                axis.plot(
                    x_values,
                    [row.linear_probe_accuracy_pct for row in ordered],
                    marker="o",
                    linewidth=2.2,
                    color=palette[probe_type],
                    label=labels[probe_type],
                )
            axis.axhline(50.0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xscale("log")
            axis.set_xticks(self.sequence_lengths)
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axis.set_ylim(0, 100)
            axis.set_xlabel("Study sequence length")
            axis.set_ylabel("Accuracy (%)")
            axis.set_title(f"{foil_type.title()} Foils")
        axes[0].legend(frameon=False, loc="best")
        fig.suptitle("ViT^3 THINGS Linear Probes on Layerwise Losses/Gradients", fontsize=14)
        fig.savefig(self.outputs_dir / "vittt_things_linear_probe_eval.png", dpi=250)
        plt.close(fig)

    def run(self) -> Tuple[List[SummaryResult], List[TrialResult]]:
        summaries: List[SummaryResult] = []
        trials: List[TrialResult] = []
        for probe_type in ["loss_probe", "grad_probe"]:
            print(f"Running {probe_type} on {self.device}...")
            for foil_type in FOIL_TYPES:
                for sequence_length in self.sequence_lengths:
                    summary, trial_rows = self.evaluate_condition(probe_type, foil_type, sequence_length)
                    summaries.append(summary)
                    trials.extend(trial_rows)
                    print(
                        f"  foil={foil_type}"
                        f" N={summary.sequence_length}"
                        f" episodes={summary.episode_count}"
                        f" trials={summary.evaluated_trials}"
                        f" accuracy={summary.linear_probe_accuracy_pct:.2f}%"
                    )
        self.write_outputs(summaries, trials)
        self.plot_results(summaries)
        return summaries, trials


def main() -> None:
    args = parse_args()
    runner = LayerwiseProbeEvaluator(args)
    runner.run()


if __name__ == "__main__":
    main()
