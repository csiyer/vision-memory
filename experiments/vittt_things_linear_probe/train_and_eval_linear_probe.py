import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vittt_brady_2afc.run_vittt_brady_2afc import Brady2AFCRunner, select_device
from stimuli import ThingsDataset


@dataclass
class ProbeRow:
    split: str
    pair_id: str
    episode_index: int
    sequence_length: int
    foil_type: str
    category: str
    exemplar_id: int
    is_old: int
    image_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ViT^3 THINGS linear probe on streamed categories and track held-out local THINGS performance."
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
        "--train-categories",
        type=int,
        default=1500,
        help="Number of streamed THINGS categories to load for probe training after excluding held-out local THINGS categories.",
    )
    parser.add_argument(
        "--heldout-categories",
        type=int,
        default=500,
        help="Number of local THINGS categories to reserve for held-out evaluation.",
    )
    parser.add_argument(
        "--train-probe-images",
        type=int,
        default=20000,
        help="Total streamed probe rows for fitting. Must be divisible by 4.",
    )
    parser.add_argument(
        "--heldout-probe-images",
        type=int,
        default=4000,
        help="Total local held-out probe rows for evaluation. Must be divisible by 4.",
    )
    parser.add_argument("--min-sequence-length", type=int, default=1, help="Minimum study list length.")
    parser.add_argument("--max-sequence-length", type=int, default=100, help="Maximum study list length.")
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 5000, 10000, 20000],
        help="Training row counts to evaluate as the probe sees more data. Rows are added in old/foil pairs.",
    )
    parser.add_argument(
        "--brady-summary",
        type=Path,
        default=REPO_ROOT / "experiments" / "vittt_brady_2afc" / "outputs" / "summary.csv",
        help="Cached Brady zero-shot summary table for comparison plotting when Brady eval is enabled.",
    )
    parser.add_argument(
        "--run-brady-eval",
        action="store_true",
        help="Also evaluate the final probe on Brady and write the old comparison outputs.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_things_training_arrays(metadata_path: Path, representations_path: Path) -> tuple[List[Dict[str, str]], np.ndarray, np.ndarray]:
    rows = load_csv_rows(metadata_path)
    payload = np.load(representations_path)
    features = np.asarray(payload["features"], dtype=np.float32)
    labels = np.asarray([int(row["is_old"]) for row in rows], dtype=np.int64)
    if features.shape[0] != len(rows):
        raise ValueError(f"Representation row count {features.shape[0]} does not match metadata rows {len(rows)}.")
    return rows, features, labels


def group_pairwise_accuracy(rows: List[Dict[str, str]], scores: np.ndarray) -> float:
    grouped: Dict[str, List[tuple[int, float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row["pair_id"], []).append((int(row["is_old"]), float(score)))
    correct = 0.0
    for entries in grouped.values():
        old_score = [value for is_old, value in entries if is_old == 1][0]
        foil_score = [value for is_old, value in entries if is_old == 0][0]
        if old_score > foil_score:
            correct += 1.0
        elif old_score == foil_score:
            correct += 0.5
    return correct / len(grouped)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    ).fit(x_train, y_train)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ordered_pair_indices(rows: List[Dict[str, str]]) -> List[List[int]]:
    grouped: "OrderedDict[str, List[int]]" = OrderedDict()
    for idx, row in enumerate(rows):
        grouped.setdefault(row["pair_id"], []).append(idx)
    ordered = list(grouped.values())
    for pair in ordered:
        if len(pair) != 2:
            raise ValueError(f"Expected exactly 2 rows per pair, found {len(pair)}.")
    return ordered


def summarize_probe_scores(
    rows: List[Dict[str, str]],
    labels: np.ndarray,
    probs: np.ndarray,
    split: str,
    num_training_rows: int,
) -> List[Dict[str, float]]:
    summary_rows = []
    conditions = ["overall", "novel", "exemplar"]
    for foil_type in conditions:
        subset_indices = [
            idx
            for idx, row in enumerate(rows)
            if foil_type == "overall" or row["foil_type"] == foil_type
        ]
        if not subset_indices:
            continue
        subset_rows = [rows[idx] for idx in subset_indices]
        subset_labels = labels[subset_indices]
        subset_probs = probs[subset_indices]
        subset_preds = (subset_probs >= 0.5).astype(np.int64)
        summary_rows.append(
            {
                "split": split,
                "foil_type": foil_type,
                "num_training_rows": int(num_training_rows),
                "num_rows": int(len(subset_indices)),
                "num_pairs": int(len({row["pair_id"] for row in subset_rows})),
                "accuracy": float(accuracy_score(subset_labels, subset_preds)),
                "auc": float(roc_auc_score(subset_labels, subset_probs)),
                "pairwise_accuracy": float(group_pairwise_accuracy(subset_rows, subset_probs)),
            }
        )
    return summary_rows


def plot_learning_curve(summary_rows: List[Dict[str, float]], output_path: Path) -> None:
    heldout_rows = [
        row for row in summary_rows if row["split"] == "heldout" and row["foil_type"] == "overall"
    ]
    heldout_rows = sorted(heldout_rows, key=lambda row: int(row["num_training_rows"]))
    x = np.asarray([int(row["num_training_rows"]) for row in heldout_rows], dtype=np.float32)
    auc = np.asarray([float(row["auc"]) for row in heldout_rows], dtype=np.float32)
    afc = np.asarray([float(row["pairwise_accuracy"]) for row in heldout_rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(x, auc, "o-", color="#111111", linewidth=2.5)
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("Streamed THINGS training rows")
    axes[0].set_ylabel("Held-out Y/N AUC")
    axes[0].set_title("Held-out THINGS AUC")

    axes[1].plot(x, afc, "o-", color="#1f78b4", linewidth=2.5)
    axes[1].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Streamed THINGS training rows")
    axes[1].set_ylabel("Held-out pairwise accuracy")
    axes[1].set_title("Held-out THINGS Pairwise Accuracy")

    fig.savefig(output_path, dpi=250)
    plt.close(fig)


class BradyRepresentationEvaluator:
    def __init__(self, device: torch.device, ttt_lr: float, seed: int) -> None:
        self.engine = Brady2AFCRunner(
            device=device,
            ttt_lr=ttt_lr,
            seed=seed,
            foil_types=["novel", "exemplar", "state"],
            sequence_lengths=[1, 10, 100, 1000],
        )

    def _extract_representation(self, image_path: Path, memory_state) -> np.ndarray:
        image_tensor = self.engine._load_image_tensor(image_path)
        with torch.no_grad():
            _, features, _, _ = self.engine.model(
                image_tensor,
                states=memory_state,
                learning_rate=0.0,
                return_grad_norm=False,
            )
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def evaluate(self, model: Pipeline) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        per_trial_rows = []
        for foil_type in ["novel", "exemplar", "state"]:
            for sequence_length in [1, 10, 100, 1000]:
                episodes = self.engine.build_episodes(foil_type, sequence_length)
                for episode_index, episode in enumerate(episodes, start=1):
                    memory_state = self.engine._encode_study_sequence(episode["study_paths"])
                    for trial_index, pair in enumerate(episode["trials"], start=1):
                        target_path, foil_path = pair
                        target_features = self._extract_representation(target_path, memory_state).reshape(1, -1)
                        foil_features = self._extract_representation(foil_path, memory_state).reshape(1, -1)
                        old_prob = float(model.predict_proba(target_features)[0, 1])
                        foil_prob = float(model.predict_proba(foil_features)[0, 1])
                        if old_prob > foil_prob:
                            acc = 1.0
                        elif old_prob < foil_prob:
                            acc = 0.0
                        else:
                            acc = 0.5
                        per_trial_rows.append(
                            {
                                "foil_type": foil_type,
                                "sequence_length": sequence_length,
                                "episode_index": episode_index,
                                "trial_index_within_episode": trial_index,
                                "target_name": target_path.name,
                                "foil_name": foil_path.name,
                                "old_familiarity": old_prob,
                                "foil_familiarity": foil_prob,
                                "linear_probe_accuracy": acc,
                            }
                        )

        summaries = []
        for foil_type in ["novel", "exemplar", "state"]:
            for sequence_length in [1, 10, 100, 1000]:
                subset = [row for row in per_trial_rows if row["foil_type"] == foil_type and row["sequence_length"] == sequence_length]
                summaries.append(
                    {
                        "foil_type": foil_type,
                        "sequence_length": sequence_length,
                        "evaluated_trials": len(subset),
                        "linear_probe_accuracy_pct": 100.0 * sum(row["linear_probe_accuracy"] for row in subset) / len(subset),
                    }
                )
        return per_trial_rows, summaries


def plot_comparison(linear_rows: List[Dict[str, float]], zero_shot_rows: List[Dict[str, str]], output_path: Path) -> None:
    colors = {
        "linear": "#111111",
        "ratio_loss": "#c03d3d",
        "ratio_grad": "#1f78b4",
        "raw_loss": "#ef8a62",
        "raw_grad": "#67a9cf",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, foil_type in zip(axes, ["novel", "exemplar", "state"]):
        ordered_linear = sorted(
            [row for row in linear_rows if row["foil_type"] == foil_type],
            key=lambda row: row["sequence_length"],
        )
        ordered_zero = sorted(
            [row for row in zero_shot_rows if row["foil_type"] == foil_type],
            key=lambda row: int(row["sequence_length"]),
        )
        x_values = [row["sequence_length"] for row in ordered_linear]
        axis.plot(
            x_values,
            [row["linear_probe_accuracy_pct"] for row in ordered_linear],
            "o-",
            color=colors["linear"],
            linewidth=2.5,
            label="Rep Probe",
        )
        axis.plot(
            x_values,
            [float(row["ratio_loss_accuracy_pct"]) for row in ordered_zero],
            "o--",
            color=colors["ratio_loss"],
            linewidth=1.8,
            label="Zero-shot Ratio Loss",
        )
        axis.plot(
            x_values,
            [float(row["ratio_grad_accuracy_pct"]) for row in ordered_zero],
            "s--",
            color=colors["ratio_grad"],
            linewidth=1.8,
            label="Zero-shot Ratio Grad",
        )
        axis.plot(
            x_values,
            [float(row["raw_loss_accuracy_pct"]) for row in ordered_zero],
            "o:",
            color=colors["raw_loss"],
            linewidth=1.8,
            label="Zero-shot Raw Loss",
        )
        axis.plot(
            x_values,
            [float(row["raw_grad_accuracy_pct"]) for row in ordered_zero],
            "s:",
            color=colors["raw_grad"],
            linewidth=1.8,
            label="Zero-shot Raw Grad",
        )
        axis.axhline(50.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_xticks([1, 10, 100, 1000])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_ylim(0, 100)
        axis.set_title(foil_type.title())
        axis.set_xlabel("Study sequence length")
        axis.set_ylabel("Accuracy (%)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("ViT^3 Representation Probe vs Zero-shot Readouts", fontsize=14)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


class ThingsProbeExperiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = select_device(args.device)
        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.rng = random.Random(args.seed)

        self.engine = Brady2AFCRunner(
            device=self.device,
            ttt_lr=args.ttt_lr,
            seed=args.seed,
            foil_types=["novel"],
            sequence_lengths=[1],
        )
        self.feature_dim = int(self.engine.model.embed_dim)
        self.num_layers = int(len(self.engine.model.blocks))
        self.baseline_cache: Dict[str, Dict[str, np.ndarray]] = {}

        self.heldout_dataset = ThingsDataset(
            n_categories=args.heldout_categories,
            exemplars_per_category=2,
            source="local",
        )
        if len(self.heldout_dataset) < args.heldout_categories:
            raise ValueError(
                f"Loaded only {len(self.heldout_dataset)} local THINGS categories, fewer than requested {args.heldout_categories}."
            )

        self.train_dataset = ThingsDataset(
            n_categories=args.train_categories,
            exemplars_per_category=2,
            source="streaming",
            excluded_categories=set(self.heldout_dataset.category_names),
        )
        if len(self.train_dataset) < args.train_categories:
            raise ValueError(
                f"Loaded only {len(self.train_dataset)} streamed THINGS categories after exclusions, fewer than requested {args.train_categories}."
            )

        overlap = set(self.train_dataset.category_names) & set(self.heldout_dataset.category_names)
        if overlap:
            raise ValueError(f"Expected train/heldout category split to be disjoint, but found overlap such as {sorted(overlap)[:5]}.")

        self._validate_dataset(self.train_dataset, split_name="train")
        self._validate_dataset(self.heldout_dataset, split_name="heldout")

    def _validate_dataset(self, dataset: ThingsDataset, split_name: str) -> None:
        if len(dataset) < 2 * self.args.max_sequence_length:
            raise ValueError(
                f"{split_name} dataset has {len(dataset)} categories, but max sequence length {self.args.max_sequence_length} "
                "needs at least twice that many categories for balanced novel episodes."
            )

    def _image_key(self, split: str, category: str, exemplar_id: int) -> str:
        return f"{split}::{category}::exemplar_{exemplar_id}"

    def _encode_study_sequence(self, study_images: Sequence[Image.Image]):
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

    def _extract_representation(self, image: Image.Image, memory_state) -> np.ndarray:
        image_tensor = self.engine.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, features, _, _ = self.engine.model(
                image_tensor,
                states=memory_state,
                learning_rate=0.0,
                return_grad_norm=False,
            )
        return features.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _baseline_metrics(self, image: Image.Image, image_key: str) -> Dict[str, np.ndarray]:
        if image_key not in self.baseline_cache:
            image_tensor = self.engine.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, _, metrics = self.engine.model(
                    image_tensor,
                    states=None,
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
            self.baseline_cache[image_key] = {
                "raw_losses": np.asarray(raw_losses, dtype=np.float32),
                "raw_grads": np.asarray(raw_grads, dtype=np.float32),
            }
        return self.baseline_cache[image_key]

    def _score_image_all_layers(self, image: Image.Image, image_key: str, memory_state) -> Dict[str, np.ndarray]:
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
        raw_losses_arr = np.asarray(raw_losses, dtype=np.float32)
        raw_grads_arr = np.asarray(raw_grads, dtype=np.float32)
        baseline = self._baseline_metrics(image, image_key)
        return {
            "raw_losses": raw_losses_arr,
            "raw_grads": raw_grads_arr,
            "ratio_losses": raw_losses_arr / (baseline["raw_losses"] + 1e-8),
            "ratio_grads": raw_grads_arr / (baseline["raw_grads"] + 1e-8),
        }

    def _sample_exemplar_episode(self, dataset: ThingsDataset, sequence_length: int) -> Dict:
        category_indices = self.rng.sample(list(range(len(dataset))), sequence_length)
        study_items = []
        probe_pairs = []
        for idx in category_indices:
            metadata = dataset.get_metadata(idx, 0)
            category = metadata["category"]
            study_image = dataset.get_image(idx, 0)
            foil_image = dataset.get_image(idx, 1)
            study_items.append((study_image, category, 0))
            probe_pairs.append(
                {
                    "category": category,
                    "target": (study_image, 0),
                    "foil": (foil_image, 1),
                    "foil_type": "exemplar",
                }
            )
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def _sample_novel_episode(self, dataset: ThingsDataset, sequence_length: int) -> Dict:
        category_indices = self.rng.sample(list(range(len(dataset))), 2 * sequence_length)
        study_indices = category_indices[:sequence_length]
        foil_indices = category_indices[sequence_length:]
        study_items = []
        probe_pairs = []
        for study_idx, foil_idx in zip(study_indices, foil_indices):
            study_meta = dataset.get_metadata(study_idx, 0)
            foil_meta = dataset.get_metadata(foil_idx, 0)
            study_image = dataset.get_image(study_idx, 0)
            foil_image = dataset.get_image(foil_idx, 0)
            study_items.append((study_image, study_meta["category"], 0))
            probe_pairs.append(
                {
                    "category": study_meta["category"],
                    "target": (study_image, 0),
                    "foil": (foil_image, 0),
                    "foil_type": "novel",
                    "foil_category": foil_meta["category"],
                }
            )
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def _collect_split(self, dataset: ThingsDataset, split: str, total_probe_images: int) -> tuple[List[ProbeRow], Dict[str, np.ndarray]]:
        if total_probe_images % 4 != 0:
            raise ValueError(f"{split} total probe images must be divisible by 4.")

        target_pairs_per_foil = total_probe_images // 4
        remaining_pairs = {"novel": target_pairs_per_foil, "exemplar": target_pairs_per_foil}
        rows: List[ProbeRow] = []
        representation_rows: List[np.ndarray] = []
        raw_loss_rows: List[np.ndarray] = []
        raw_grad_rows: List[np.ndarray] = []
        ratio_loss_rows: List[np.ndarray] = []
        ratio_grad_rows: List[np.ndarray] = []
        episode_index = 0
        foil_cycle = ["novel", "exemplar"]
        foil_cursor = 0

        while remaining_pairs["novel"] > 0 or remaining_pairs["exemplar"] > 0:
            foil_type = foil_cycle[foil_cursor % len(foil_cycle)]
            foil_cursor += 1
            if remaining_pairs[foil_type] <= 0:
                continue

            sequence_length = self.rng.randint(self.args.min_sequence_length, self.args.max_sequence_length)
            sequence_length = min(sequence_length, remaining_pairs[foil_type])
            episode_index += 1

            if foil_type == "novel":
                episode = self._sample_novel_episode(dataset, sequence_length)
            else:
                episode = self._sample_exemplar_episode(dataset, sequence_length)
            study_images = [item[0] for item in episode["study_items"]]
            memory_state = self._encode_study_sequence(study_images)

            for trial_index, pair in enumerate(episode["probe_pairs"], start=1):
                pair_id = f"{split}_{foil_type}_ep{episode_index:05d}_pair{trial_index:03d}"
                target_image, target_exemplar = pair["target"]
                foil_image, foil_exemplar = pair["foil"]
                category = pair["category"]
                target_key = self._image_key(split, category, target_exemplar)

                representation_rows.append(self._extract_representation(target_image, memory_state))
                target_metrics = self._score_image_all_layers(target_image, target_key, memory_state)
                raw_loss_rows.append(target_metrics["raw_losses"])
                raw_grad_rows.append(target_metrics["raw_grads"])
                ratio_loss_rows.append(target_metrics["ratio_losses"])
                ratio_grad_rows.append(target_metrics["ratio_grads"])
                rows.append(
                    ProbeRow(
                        split=split,
                        pair_id=pair_id,
                        episode_index=episode_index,
                        sequence_length=sequence_length,
                        foil_type=foil_type,
                        category=category,
                        exemplar_id=target_exemplar,
                        is_old=1,
                        image_key=target_key,
                    )
                )

                foil_category = category if foil_type == "exemplar" else pair["foil_category"]
                foil_key = self._image_key(split, foil_category, foil_exemplar)
                representation_rows.append(self._extract_representation(foil_image, memory_state))
                foil_metrics = self._score_image_all_layers(foil_image, foil_key, memory_state)
                raw_loss_rows.append(foil_metrics["raw_losses"])
                raw_grad_rows.append(foil_metrics["raw_grads"])
                ratio_loss_rows.append(foil_metrics["ratio_losses"])
                ratio_grad_rows.append(foil_metrics["ratio_grads"])
                rows.append(
                    ProbeRow(
                        split=split,
                        pair_id=pair_id,
                        episode_index=episode_index,
                        sequence_length=sequence_length,
                        foil_type=foil_type,
                        category=foil_category,
                        exemplar_id=foil_exemplar,
                        is_old=0,
                        image_key=foil_key,
                    )
                )

            remaining_pairs[foil_type] -= sequence_length
            print(
                f"split={split}"
                f" episode={episode_index}"
                f" foil_type={foil_type}"
                f" sequence_length={sequence_length}"
                f" remaining_pairs={remaining_pairs}"
            )

        return rows, {
            "representations": np.stack(representation_rows, axis=0),
            "raw_losses": np.stack(raw_loss_rows, axis=0),
            "raw_grads": np.stack(raw_grad_rows, axis=0),
            "ratio_losses": np.stack(ratio_loss_rows, axis=0),
            "ratio_grads": np.stack(ratio_grad_rows, axis=0),
        }

    def _save_split_outputs(self, split: str, rows: List[ProbeRow], features: Dict[str, np.ndarray]) -> None:
        metadata_path = self.outputs_dir / f"things_probe_{split}_metadata.csv"
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

        np.savez_compressed(
            self.outputs_dir / f"things_probe_{split}_representations.npz",
            features=features["representations"],
            feature_dim=np.asarray([self.feature_dim], dtype=np.int32),
        )
        np.savez_compressed(
            self.outputs_dir / f"things_probe_{split}_layerwise_metrics.npz",
            raw_losses=features["raw_losses"],
            raw_grads=features["raw_grads"],
            ratio_losses=features["ratio_losses"],
            ratio_grads=features["ratio_grads"],
            num_layers=np.asarray([self.num_layers], dtype=np.int32),
        )
        if split == "train":
            legacy_metadata_path = self.outputs_dir / "things_probe_metadata.csv"
            with legacy_metadata_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(asdict(row))
            np.savez_compressed(
                self.outputs_dir / "things_probe_representations.npz",
                features=features["representations"],
                feature_dim=np.asarray([self.feature_dim], dtype=np.int32),
            )

    def run(self) -> Dict[str, object]:
        train_rows, train_features = self._collect_split(
            dataset=self.train_dataset,
            split="train",
            total_probe_images=self.args.train_probe_images,
        )
        heldout_rows, heldout_features = self._collect_split(
            dataset=self.heldout_dataset,
            split="heldout",
            total_probe_images=self.args.heldout_probe_images,
        )
        self._save_split_outputs("train", train_rows, train_features)
        self._save_split_outputs("heldout", heldout_rows, heldout_features)

        return {
            "train_rows": [asdict(row) for row in train_rows],
            "train_features": train_features["representations"],
            "train_labels": np.asarray([row.is_old for row in train_rows], dtype=np.int64),
            "heldout_rows": [asdict(row) for row in heldout_rows],
            "heldout_features": heldout_features["representations"],
            "heldout_labels": np.asarray([row.is_old for row in heldout_rows], dtype=np.int64),
            "summary": {
                "seed": self.args.seed,
                "device": str(self.device),
                "ttt_learning_rate": self.args.ttt_lr,
                "representation": "memory-conditioned final pooled feature vector",
                "feature_dim": self.feature_dim,
                "num_layers": self.num_layers,
                "train_categories_loaded": len(self.train_dataset),
                "heldout_categories_loaded": len(self.heldout_dataset),
                "excluded_local_categories": len(self.heldout_dataset.category_names),
                "train_probe_rows": len(train_rows),
                "heldout_probe_rows": len(heldout_rows),
                "min_sequence_length": self.args.min_sequence_length,
                "max_sequence_length": self.args.max_sequence_length,
            },
        }


def main() -> None:
    args = parse_args()
    outputs_dir = RUN_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

    experiment = ThingsProbeExperiment(args)
    payload = experiment.run()
    train_rows = payload["train_rows"]
    train_features = payload["train_features"]
    train_labels = payload["train_labels"]
    heldout_rows = payload["heldout_rows"]
    heldout_features = payload["heldout_features"]
    heldout_labels = payload["heldout_labels"]

    with (outputs_dir / "things_extraction_summary.json").open("w") as handle:
        json.dump(payload["summary"], handle, indent=2)

    ordered_pairs = ordered_pair_indices(train_rows)
    requested_sizes = sorted(set(args.sample_sizes))
    learning_curve_rows: List[Dict[str, float]] = []
    final_model = None
    final_train_summary = None
    final_heldout_summary = None

    max_train_rows = len(train_rows)
    valid_sizes = [size for size in requested_sizes if size <= max_train_rows and size % 2 == 0]
    if max_train_rows % 2 != 0:
        raise ValueError("Expected an even number of training rows because old/foil rows are stored in pairs.")
    if max_train_rows not in valid_sizes:
        valid_sizes.append(max_train_rows)
    valid_sizes = sorted(set(valid_sizes))
    if not valid_sizes:
        raise ValueError("No valid sample sizes remain after clipping to the number of generated training rows.")

    for sample_size in valid_sizes:
        required_pairs = sample_size // 2
        subset_indices = [idx for pair in ordered_pairs[:required_pairs] for idx in pair]
        x_train = train_features[subset_indices]
        y_train = train_labels[subset_indices]
        subset_rows = [train_rows[idx] for idx in subset_indices]

        model = fit_probe(x_train, y_train, seed=args.seed)
        train_probs = model.predict_proba(x_train)[:, 1]
        heldout_probs = model.predict_proba(heldout_features)[:, 1]

        train_summary_rows = summarize_probe_scores(
            subset_rows,
            y_train,
            train_probs,
            split="train",
            num_training_rows=sample_size,
        )
        heldout_summary_rows = summarize_probe_scores(
            heldout_rows,
            heldout_labels,
            heldout_probs,
            split="heldout",
            num_training_rows=sample_size,
        )
        learning_curve_rows.extend(train_summary_rows)
        learning_curve_rows.extend(heldout_summary_rows)

        overall_heldout = next(row for row in heldout_summary_rows if row["foil_type"] == "overall")
        print(
            f"training_rows={sample_size}"
            f" heldout_auc={overall_heldout['auc']:.4f}"
            f" heldout_pairwise={overall_heldout['pairwise_accuracy']:.4f}"
        )

        if sample_size == max(valid_sizes):
            final_model = model
            final_train_summary = train_summary_rows
            final_heldout_summary = heldout_summary_rows

    if final_model is None or final_train_summary is None or final_heldout_summary is None:
        raise RuntimeError("Final probe fit did not complete.")

    write_csv(outputs_dir / "things_probe_learning_curve.csv", learning_curve_rows)
    plot_learning_curve(learning_curve_rows, outputs_dir / "things_probe_learning_curve.png")

    final_summary = {
        "extraction": payload["summary"],
        "sample_sizes": valid_sizes,
        "final_train_summary": final_train_summary,
        "final_heldout_summary": final_heldout_summary,
    }
    with (outputs_dir / "things_probe_summary.json").open("w") as handle:
        json.dump(final_summary, handle, indent=2)
    with (outputs_dir / "things_train_summary.json").open("w") as handle:
        json.dump(
            {
                "extraction": payload["summary"],
                "final_train_summary": final_train_summary,
            },
            handle,
            indent=2,
        )

    with (outputs_dir / "linear_probe.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": final_model,
                "summary": final_summary,
            },
            handle,
        )

    if args.run_brady_eval:
        if not args.brady_summary.exists():
            raise FileNotFoundError(f"Missing Brady summary file: {args.brady_summary}")
        brady_zero_rows = load_csv_rows(args.brady_summary)
        evaluator = BradyRepresentationEvaluator(device=select_device(args.device), ttt_lr=args.ttt_lr, seed=args.seed)
        brady_linear_trials, brady_linear_summary = evaluator.evaluate(final_model)
        write_csv(outputs_dir / "brady_linear_probe_trials.csv", brady_linear_trials)
        write_csv(outputs_dir / "brady_linear_probe_summary.csv", brady_linear_summary)
        with (outputs_dir / "brady_linear_probe_summary.json").open("w") as handle:
            json.dump(
                {
                    "probe_summary": final_summary,
                    "brady_summary": brady_linear_summary,
                },
                handle,
                indent=2,
            )
        plot_comparison(
            linear_rows=brady_linear_summary,
            zero_shot_rows=brady_zero_rows,
            output_path=outputs_dir / "linear_probe_vs_zero_shot.png",
        )


if __name__ == "__main__":
    main()
