import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(RUN_ROOT / "outputs" / ".mplconfig"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vittt_brady_2afc.run_vittt_brady_2afc import Brady2AFCRunner, select_device
from stimuli import ThingsDataset


@dataclass
class ProbeRow:
    pair_id: str
    episode_index: int
    sequence_length: int
    foil_type: str
    category: str
    exemplar_id: int
    is_old: int
    image_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract THINGS pooled representations for a ViT^3 linear probe.")
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
        "--things-categories",
        type=int,
        default=500,
        help="Number of THINGS categories to stream with two exemplars each.",
    )
    parser.add_argument(
        "--total-probe-images",
        type=int,
        default=10000,
        help="Total number of single-image probe rows to cache. Must be divisible by 4.",
    )
    parser.add_argument("--min-sequence-length", type=int, default=1, help="Minimum study length.")
    parser.add_argument("--max-sequence-length", type=int, default=100, help="Maximum study length.")
    return parser.parse_args()


class ThingsProbeExtractor:
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
        self.dataset = ThingsDataset(n_categories=args.things_categories, exemplars_per_category=2)
        if len(self.dataset) < max(args.max_sequence_length, 2 * args.max_sequence_length):
            raise ValueError(
                f"Loaded only {len(self.dataset)} THINGS categories; need at least {2 * args.max_sequence_length} "
                "to support balanced novel episodes up to the requested max sequence length."
            )
        self.feature_dim = int(self.engine.model.embed_dim)

    def _image_key(self, category: str, exemplar_id: int) -> str:
        return f"{category}::exemplar_{exemplar_id}"

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

    def _sample_exemplar_episode(self, sequence_length: int) -> Dict:
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
                    "foil_type": "exemplar",
                }
            )
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def _sample_novel_episode(self, sequence_length: int) -> Dict:
        category_indices = self.rng.sample(list(range(len(self.dataset))), 2 * sequence_length)
        study_indices = category_indices[:sequence_length]
        foil_indices = category_indices[sequence_length:]
        study_items = []
        probe_pairs = []
        for study_idx, foil_idx in zip(study_indices, foil_indices):
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
                    "foil_type": "novel",
                    "foil_category": foil_meta["category"],
                }
            )
        return {"study_items": study_items, "probe_pairs": probe_pairs}

    def run(self) -> None:
        total_probe_images = self.args.total_probe_images
        if total_probe_images % 4 != 0:
            raise ValueError("--total-probe-images must be divisible by 4 to balance old/foil and novel/exemplar.")

        target_pairs_per_foil = total_probe_images // 4
        remaining_pairs = {"novel": target_pairs_per_foil, "exemplar": target_pairs_per_foil}
        rows: List[ProbeRow] = []
        feature_rows: List[np.ndarray] = []
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

            episode = self._sample_novel_episode(sequence_length) if foil_type == "novel" else self._sample_exemplar_episode(sequence_length)
            study_images = [item[0] for item in episode["study_items"]]
            memory_state = self._encode_study_sequence(study_images)

            for trial_index, pair in enumerate(episode["probe_pairs"], start=1):
                pair_id = f"{foil_type}_ep{episode_index:05d}_pair{trial_index:03d}"
                target_image, target_exemplar = pair["target"]
                foil_image, foil_exemplar = pair["foil"]
                category = pair["category"]

                old_key = self._image_key(category, target_exemplar)
                old_features = self._extract_representation(target_image, memory_state)
                feature_rows.append(old_features)
                rows.append(
                    ProbeRow(
                        pair_id=pair_id,
                        episode_index=episode_index,
                        sequence_length=sequence_length,
                        foil_type=foil_type,
                        category=category,
                        exemplar_id=target_exemplar,
                        is_old=1,
                        image_key=old_key,
                    )
                )

                foil_category = category if foil_type == "exemplar" else pair["foil_category"]
                foil_key = self._image_key(foil_category, foil_exemplar)
                foil_features = self._extract_representation(foil_image, memory_state)
                feature_rows.append(foil_features)
                rows.append(
                    ProbeRow(
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
                f"episode={episode_index}"
                f" foil_type={foil_type}"
                f" sequence_length={sequence_length}"
                f" remaining_pairs={remaining_pairs}"
            )

        metadata_path = self.outputs_dir / "things_probe_metadata.csv"
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

        features = np.stack(feature_rows, axis=0)
        np.savez_compressed(
            self.outputs_dir / "things_probe_representations.npz",
            features=features,
            feature_dim=np.asarray([self.feature_dim], dtype=np.int32),
        )

        summary = {
            "seed": self.args.seed,
            "device": str(self.device),
            "ttt_learning_rate": self.args.ttt_lr,
            "representation": "memory-conditioned final pooled feature vector",
            "feature_dim": self.feature_dim,
            "things_categories_loaded": len(self.dataset),
            "total_probe_images": len(rows),
            "total_pairs": len(rows) // 2,
            "foil_type_counts": {
                foil_type: sum(1 for row in rows if row.foil_type == foil_type) for foil_type in ["novel", "exemplar"]
            },
            "label_counts": {
                "old": sum(1 for row in rows if row.is_old == 1),
                "foil": sum(1 for row in rows if row.is_old == 0),
            },
        }
        with (self.outputs_dir / "things_extraction_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    extractor = ThingsProbeExtractor(args)
    extractor.run()


if __name__ == "__main__":
    main()
