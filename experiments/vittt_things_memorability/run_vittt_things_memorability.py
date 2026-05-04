import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass
class ImageRecord:
    percentile_rank: int
    global_rank: int
    concept: str
    image_name: str
    human_cr: float
    local_path: str


@dataclass
class ProbeTrial:
    run_index: int
    seed: int
    probe_index: int
    study_position: int
    lag: int
    image_name: str
    concept: str
    human_cr: float
    raw_grad: float
    memory_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a THINGS memorability assay with frozen-state ViT^3 probes.")
    parser.add_argument("--seed", type=int, default=13, help="Base random seed.")
    parser.add_argument(
        "--target-probes-per-image",
        type=int,
        default=100,
        help="Target number of probe scores per image under the balanced schedule.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RUN_ROOT / "selected_local_object_images.csv",
        help="Selected local-THINGS memorability image manifest.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=REPO_ROOT / "memory_datasets" / "THINGS" / "object_images",
        help="Local THINGS object_images root used for default manifest generation.",
    )
    parser.add_argument("--ttt-lr", type=float, default=1.0, help="TTT learning rate during study encoding.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", None],
        help="Override device selection.",
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


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


class ThingsMemorabilityRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = select_device(args.device)
        self.outputs_dir = RUN_ROOT / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        self.transform = build_transform()
        self.model = self._load_model()
        self._ensure_default_manifest()
        self.records = self._load_records()

    def _load_model(self) -> torch.nn.Module:
        model = vittt_base().to(self.device)
        checkpoint = torch.load(REPO_ROOT / "models" / "vit3" / "vittt_base.pth", map_location=self.device, weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _ensure_default_manifest(self) -> None:
        if self.args.manifest.exists():
            return
        score_path = REPO_ROOT / "memory_datasets" / "THINGS" / "THINGS_Memorability_Scores.csv"
        image_root = self.args.image_root
        category_dirs = sorted([path for path in image_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())[:500]
        if len(category_dirs) < 500:
            raise ValueError(f"Expected at least 500 THINGS categories under {image_root}, found {len(category_dirs)}.")

        scores_by_image: Dict[str, float] = {}
        with score_path.open() as handle:
            for row in csv.DictReader(handle):
                scores_by_image[row["image_name"]] = float(row["cr"])

        candidate_rows = []
        for category_dir in category_dirs:
            image_paths = sorted(
                [path for path in category_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
                key=lambda path: path.name.lower(),
            )
            if not image_paths:
                continue
            first_image = image_paths[0]
            if first_image.name not in scores_by_image:
                continue
            candidate_rows.append(
                {
                    "concept": category_dir.name,
                    "local_image_name": first_image.name,
                    "local_path": str(first_image),
                    "human_cr": scores_by_image[first_image.name],
                }
            )

        if len(candidate_rows) < 100:
            raise ValueError(
                f"Only found {len(candidate_rows)} scored local THINGS images among the first 500 categories; need at least 100."
            )

        candidate_rows.sort(key=lambda row: (float(row["human_cr"]), row["concept"]))
        selected_rows = []
        for percentile_rank in range(100):
            global_rank = percentile_rank * len(candidate_rows) // 100
            item = dict(candidate_rows[global_rank])
            item["percentile_rank"] = percentile_rank + 1
            item["global_rank_within_local_subset"] = global_rank + 1
            selected_rows.append(item)

        with self.args.manifest.open("w", newline="") as handle:
            fieldnames = [
                "percentile_rank",
                "global_rank_within_local_subset",
                "concept",
                "local_image_name",
                "local_path",
                "human_cr",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow(row)

    def _load_records(self) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        missing: List[Path] = []
        with self.args.manifest.open() as handle:
            for row in csv.DictReader(handle):
                local_path = Path(row["local_path"])
                if not local_path.exists():
                    missing.append(local_path)
                records.append(
                    ImageRecord(
                        percentile_rank=int(row["percentile_rank"]),
                        global_rank=int(row.get("global_rank_within_local_subset", row.get("global_rank_within_cache_concepts", row["percentile_rank"]))),
                        concept=row["concept"],
                        image_name=f"{row['concept']}/{row['local_image_name']}",
                        human_cr=float(row.get("human_cr", row.get("concept_cr_mean"))),
                        local_path=row["local_path"],
                    )
                )
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(
                "Selected local THINGS memorability images are missing.\n"
                f"Missing examples:\n{preview}"
            )
        return records

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _raw_final_grad(self, image_path: Path, memory_state) -> float:
        image_tensor = self._load_image_tensor(image_path)
        with torch.no_grad():
            _, _, _, metrics = self.model(
                image_tensor,
                states=memory_state,
                learning_rate=0.0,
                return_grad_norm=True,
            )
        final_metrics = metrics[-1]
        if isinstance(final_metrics, list):
            final_metrics = final_metrics[0]
        return float(final_metrics["grad_norm"])

    def _encode_study_sequence(self, ordered_records: List[ImageRecord]):
        current_state = None
        for record in ordered_records:
            image_tensor = self._load_image_tensor(Path(record.local_path))
            with torch.no_grad():
                _, _, current_state, _ = self.model(
                    image_tensor,
                    states=current_state,
                    learning_rate=self.args.ttt_lr,
                    return_grad_norm=False,
                )
        return current_state

    def _planned_run_count(self) -> int:
        num_images = len(self.records)
        probes_per_run = num_images // 2
        if num_images % 2 != 0:
            raise ValueError(f"Balanced probe scheduling requires an even number of images, found {num_images}.")
        planned_runs = self.args.target_probes_per_image * 2
        if planned_runs % num_images != 0:
            raise ValueError(
                "Balanced probe scheduling requires 2 * target_probes_per_image to be divisible by the number "
                f"of images. Got target={self.args.target_probes_per_image}, num_images={num_images}."
            )
        expected_per_image = planned_runs * probes_per_run // num_images
        if expected_per_image != self.args.target_probes_per_image:
            raise ValueError(
                "Balanced probe scheduling could not satisfy the requested target exactly: "
                f"expected {expected_per_image}, requested {self.args.target_probes_per_image}."
            )
        return planned_runs

    def run(self) -> Dict[str, object]:
        all_trials: List[ProbeTrial] = []
        probes_by_image: Dict[str, List[float]] = {record.image_name: [] for record in self.records}
        runs_completed = 0
        num_images = len(self.records)
        probes_per_run = num_images // 2
        planned_runs = self._planned_run_count()

        for run_index in range(planned_runs):
            run_seed = self.args.seed + run_index
            rng = random.Random(run_seed)
            probed_indices = [idx for idx in range(num_images) if ((idx + run_index) % num_images) < probes_per_run]
            unprobed_indices = [idx for idx in range(num_images) if idx not in probed_indices]

            probed_records = [self.records[idx] for idx in probed_indices]
            unprobed_records = [self.records[idx] for idx in unprobed_indices]
            rng.shuffle(probed_records)
            rng.shuffle(unprobed_records)
            ordered_records = probed_records + unprobed_records

            memory_state = self._encode_study_sequence(ordered_records)
            probe_records = ordered_records[:probes_per_run]
            for probe_index, record in enumerate(probe_records, start=1):
                study_position = probe_index
                lag = len(ordered_records) - study_position
                raw_grad = self._raw_final_grad(Path(record.local_path), memory_state)
                memory_score = -raw_grad
                all_trials.append(
                    ProbeTrial(
                        run_index=run_index + 1,
                        seed=run_seed,
                        probe_index=probe_index,
                        study_position=study_position,
                        lag=lag,
                        image_name=record.image_name,
                        concept=record.concept,
                        human_cr=record.human_cr,
                        raw_grad=raw_grad,
                        memory_score=memory_score,
                    )
                )
                probes_by_image[record.image_name].append(memory_score)

            runs_completed = run_index + 1
            min_count = min(len(scores) for scores in probes_by_image.values())
            print(
                f"run={runs_completed}"
                f" min_probes_per_image={min_count}"
                f" max_probes_per_image={max(len(scores) for scores in probes_by_image.values())}"
            )

        final_counts = {len(scores) for scores in probes_by_image.values()}
        if final_counts != {self.args.target_probes_per_image}:
            raise RuntimeError(
                "Balanced probe schedule failed to allocate the requested number of probes per image. "
                f"Observed counts: {sorted(final_counts)}"
            )

        image_rows = []
        for record in self.records:
            scores = probes_by_image[record.image_name]
            image_rows.append(
                {
                    "image_name": record.image_name,
                    "concept": record.concept,
                    "human_cr": record.human_cr,
                    "num_probes": len(scores),
                    "mean_memory_score": mean(scores),
                    "sem_memory_score": sem(scores),
                }
            )

        x = np.asarray([row["human_cr"] for row in image_rows], dtype=np.float64)
        y = np.asarray([row["mean_memory_score"] for row in image_rows], dtype=np.float64)
        summary = {
            "seed": self.args.seed,
            "device": str(self.device),
            "ttt_learning_rate": self.args.ttt_lr,
            "target_probes_per_image": self.args.target_probes_per_image,
            "runs_completed": runs_completed,
            "num_images": len(self.records),
            "probes_per_run": probes_per_run,
            "lag_range": [50, 99],
            "schedule": "balanced_fixed_runs",
            "pearson_r": pearson_corr(x, y),
            "spearman_r": spearman_corr(x, y),
        }

        self._write_outputs(all_trials, image_rows, summary)
        self._plot_results(image_rows, summary)
        return summary

    def _write_outputs(self, all_trials: List[ProbeTrial], image_rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
        with (self.outputs_dir / "memorability_probe_trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(all_trials[0]).keys()))
            writer.writeheader()
            for row in all_trials:
                writer.writerow(asdict(row))

        with (self.outputs_dir / "memorability_image_scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            for row in image_rows:
                writer.writerow(row)

        with (self.outputs_dir / "memorability_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)

    def _plot_results(self, image_rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
        human_scores = np.asarray([row["human_cr"] for row in image_rows], dtype=np.float64)
        model_scores = np.asarray([row["mean_memory_score"] for row in image_rows], dtype=np.float64)
        model_sems = np.asarray([row["sem_memory_score"] for row in image_rows], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        ax.errorbar(
            human_scores,
            model_scores,
            yerr=model_sems,
            fmt="o",
            color="#1f78b4",
            ecolor="#9ecae1",
            elinewidth=1,
            capsize=2,
            alpha=0.85,
        )
        fit = np.polyfit(human_scores, model_scores, deg=1)
        x_line = np.linspace(float(np.min(human_scores)), float(np.max(human_scores)), 100)
        y_line = fit[0] * x_line + fit[1]
        ax.plot(x_line, y_line, color="#111111", linewidth=2)
        ax.set_xlabel("THINGS human memorability (CR)")
        ax.set_ylabel("ViT^3 memory score (-final raw grad)")
        ax.set_title("ViT^3 THINGS Memorability")
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {summary['pearson_r']:.3f}\nSpearman r = {summary['spearman_r']:.3f}\nRuns = {summary['runs_completed']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        fig.savefig(self.outputs_dir / "vittt_things_memorability.png", dpi=250)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    runner = ThingsMemorabilityRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
