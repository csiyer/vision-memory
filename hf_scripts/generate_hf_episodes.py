#!/usr/bin/env python3
"""Generate standardized Hugging Face episode manifests for vision-memory tasks.

The default output creates two runnable suites:
  - lite: 1 episode per task x length
  - full: 10 episodes per task x length

Episodes reference stable asset IDs exported by hf_scripts/export_hf_assets.py. They do
not embed image bytes; loaders hydrate image IDs into PIL images at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "memory_datasets"
DEFAULT_OUT = REPO_ROOT.parent / "vision-memory-tasks" / "episodes"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SCHEMA_VERSION = "0.1.0"

SUITES = {
    "lite": 1,
    "full": 10,
}

COLOR_NAMES = [
    ("red", 0),
    ("orange", 30),
    ("yellow", 60),
    ("green", 120),
    ("blue", 240),
    ("purple", 300),
]

LENGTHS = {
    "recognition_continuous": [1, 10, 100, 1000],
    "recognition_2afc_all": [3, 12, 99, 300],
    "mnemonic_similarity": [1, 10, 100, 500],
    "serial_order_free": [2, 10, 100, 1000],
    "serial_order_2afc": [2, 10, 100, 1000],
    "color_memory_continuous": [1, 10, 100, 500],
    "color_memory_named": [6, 12, 60, 300],
    "paired_associate_image_word": [1, 10, 100, 1000],
    "paired_associate_image_image": [1, 10, 100, 1000],
    "associative_inference_image_word": [1, 10, 100, 1000],
    "associative_inference_image_image": [1, 10, 100, 800],
}

TASK_ORDER = list(LENGTHS)
TASK_OFFSETS = {task_name: index * 100_000 for index, task_name in enumerate(TASK_ORDER)}
SUITE_OFFSETS = {"lite": 0, "full": 10_000_000}


class AssetIndex:
    def __init__(self, source_root: Path):
        self.source_root = source_root
        self.brady_objects = self._image_ids("Brady2008Objects", "brady_objects")
        self.brady_exemplar_pairs = self._brady_pair_ids("Brady2008Exemplar", "brady_exemplar")
        self.brady_state_pairs = self._brady_pair_ids("Brady2008State", "brady_state")
        self.brady_color_objects = self._image_ids("Brady2013ColorObjects", "brady_color_objects")
        self.mst_pairs = self._mst_pairs()
        self.wordpool = self._wordpool()

    def _image_paths(self, directory_name: str) -> list[Path]:
        root = self.source_root / directory_name
        if not root.exists():
            raise FileNotFoundError(f"Missing asset directory: {root}")
        return sorted(
            [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: path.name.lower(),
        )

    def _image_ids(self, directory_name: str, collection_id: str) -> list[str]:
        return [f"{collection_id}/{path.name}" for path in self._image_paths(directory_name)]

    def _brady_pair_ids(self, directory_name: str, collection_id: str) -> list[dict]:
        groups: dict[str, list[Path]] = {}
        for path in self._image_paths(directory_name):
            match = re.match(r"^(.*?)(\d+)?$", path.stem)
            base_name = match.group(1).lower() if match else path.stem.lower()
            groups.setdefault(base_name, []).append(path)

        pairs = []
        for pair_index, base_name in enumerate(sorted(groups)):
            members = sorted(groups[base_name], key=lambda path: path.name.lower())
            if len(members) != 2:
                continue
            pairs.append(
                {
                    "pair_id": f"{collection_id}/{base_name}",
                    "group_name": base_name,
                    "pair_index": pair_index,
                    "image_ids": [f"{collection_id}/{members[0].name}", f"{collection_id}/{members[1].name}"],
                }
            )
        return pairs

    def _parse_mst_bins(self, path: Path) -> dict[int, int]:
        if not path.exists():
            return {}
        bin_map: dict[int, int] = {}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                try:
                    if len(row) >= 2:
                        item_id = int(row[0])
                        bin_num = int(row[1])
                    else:
                        item_id = row_index + 1
                        bin_num = int(row[0])
                except ValueError:
                    continue
                bin_map[item_id] = bin_num
        return bin_map

    def _mst_pairs(self) -> list[dict]:
        root = self.source_root / "MST"
        if not root.exists():
            raise FileNotFoundError(f"Missing MST directory: {root}")

        pairs = []
        for set_number in range(1, 7):
            set_dir = root / f"Set {set_number}"
            if not set_dir.exists():
                continue
            bin_map = self._parse_mst_bins(root / f"Set{set_number} bins.txt")
            targets = sorted(
                [
                    path
                    for path in set_dir.iterdir()
                    if path.is_file() and path.stem.endswith("a") and path.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name.lower(),
            )
            for target_path in targets:
                item_stem = target_path.stem[:-1]
                try:
                    item_id = int(item_stem)
                except ValueError:
                    continue
                lure_path = target_path.with_name(f"{item_stem}b{target_path.suffix}")
                if not lure_path.exists():
                    continue
                pairs.append(
                    {
                        "pair_id": f"mst/set_{set_number}/{item_id:03d}",
                        "set_number": set_number,
                        "item_id": item_id,
                        "bin": bin_map.get(item_id),
                        "target_image_id": f"mst/set_{set_number}/{target_path.name}",
                        "lure_image_id": f"mst/set_{set_number}/{lure_path.name}",
                    }
                )
        return pairs

    def _wordpool(self) -> list[dict]:
        path = self.source_root / "wasnorm_wordpool.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing wordpool: {path}")
        words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [{"word_id": f"wordpool/{index:05d}", "word": word} for index, word in enumerate(words)]


def choice_slots(rng: random.Random, correct, foil) -> tuple[list, int]:
    if rng.random() < 0.5:
        return [correct, foil], 1
    return [foil, correct], 2


def sample_without_replacement(rng: random.Random, items: list, n: int, label: str) -> list:
    if n > len(items):
        raise ValueError(f"Need {n} {label}; found {len(items)}.")
    selected = items[:]
    rng.shuffle(selected)
    return selected[:n]


def balanced_sample_by_key(rng: random.Random, items: list, n: int, key_fn: Callable) -> list:
    """Sample as evenly as possible across key strata without replacement."""
    groups: dict[object, list] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    for values in groups.values():
        rng.shuffle(values)

    selected = []
    keys = sorted(groups, key=lambda value: (str(type(value)), str(value)))
    while len(selected) < n:
        progressed = False
        round_keys = keys[:]
        rng.shuffle(round_keys)
        for key in round_keys:
            if len(selected) >= n:
                break
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) < n:
        raise ValueError(f"Could only sample {len(selected)} balanced items; requested {n}.")
    rng.shuffle(selected)
    return selected


def lag_bin(serial_lag: int) -> str:
    if serial_lag <= 1:
        return "lag_1"
    if serial_lag <= 3:
        return "lag_2_3"
    if serial_lag <= 10:
        return "lag_4_10"
    if serial_lag <= 100:
        return "lag_11_100"
    return "lag_gt_100"


def base_episode(task_name: str, suite: str, length: int, episode_index: int, seed: int, params: dict, metrics: list[str]) -> dict:
    return {
        "episode_id": f"{task_name}_v1_{suite}_len{length:04d}_{episode_index:03d}",
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "task_name": task_name,
        "length": length,
        "episode_index": episode_index,
        "seed": seed,
        "params": params,
        "metric_names": metrics,
    }


def build_recognition_continuous(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    image_ids = sample_without_replacement(rng, assets.brady_objects, length, "Brady object images")

    if length == 1:
        repeat_delays = [1]
    else:
        # Sample repeat delays across the whole episode so old and new trials are interleaved.
        repeat_delays = [rng.randint(1, length) for _ in range(length)]

    events = []
    for image_index, image_id in enumerate(image_ids):
        first_time = image_index
        repeat_time = first_time + repeat_delays[image_index]
        events.append((first_time, 0, image_index, image_id))
        events.append((repeat_time, 1, image_index, image_id))

    # Sort by scheduled time, then place new trials before old trials when tied so an
    # item can never repeat before its first presentation in the final sequence.
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    first_seen = {}
    trials = []
    for trial_index, (_scheduled_time, is_repeat, image_index, image_id) in enumerate(events):
        if not is_repeat:
            first_seen[image_id] = trial_index
            trials.append(
                {
                    "trial_index": trial_index,
                    "image_id": image_id,
                    "prompt": "Has this image already appeared in the sequence? (yes/no)",
                    "target": 0,
                    "target_label": "new",
                    "target_type": "binary_old_new",
                    "delay": None,
                    "first_seen_trial_index": trial_index,
                    "metadata": {"image_index": image_index, "scheduled_time": _scheduled_time},
                }
            )
        else:
            first_trial = first_seen[image_id]
            trials.append(
                {
                    "trial_index": trial_index,
                    "image_id": image_id,
                    "prompt": "Has this image already appeared in the sequence? (yes/no)",
                    "target": 1,
                    "target_label": "old",
                    "target_type": "binary_old_new",
                    "delay": trial_index - first_trial - 1,
                    "first_seen_trial_index": first_trial,
                    "metadata": {"image_index": image_index, "scheduled_time": _scheduled_time},
                }
            )

    episode = base_episode(
        "recognition_continuous",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_unique": length, "p_old": 0.5, "total_trials": len(trials), "sequence_policy": "random_interleaved_old_new"},
        ["recognition_accuracy", "recognition_dprime", "hit_rate_by_delay"],
    )
    episode.update({"dataset_name": "Brady2008", "trials": trials})
    return episode

def build_recognition_2afc_all(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    if length % 3 != 0:
        raise ValueError("recognition_2afc_all lengths must be divisible by 3.")
    rng = random.Random(seed)
    n_each = length // 3

    object_ids = sample_without_replacement(rng, assets.brady_objects, n_each * 2, "Brady object images")
    pairs = []
    for index in range(n_each):
        pairs.append(
            {
                "pair_id": f"novel/{index:04d}",
                "pair_source": "brady_objects",
                "foil_type": "novel",
                "correct_image_id": object_ids[index * 2],
                "foil_image_id": object_ids[index * 2 + 1],
            }
        )

    for foil_type, source_pairs in (("exemplar", assets.brady_exemplar_pairs), ("state", assets.brady_state_pairs)):
        for source_pair in sample_without_replacement(rng, source_pairs, n_each, f"Brady {foil_type} pairs"):
            image_ids = source_pair["image_ids"][:]
            rng.shuffle(image_ids)
            pairs.append(
                {
                    "pair_id": source_pair["pair_id"],
                    "pair_source": f"brady_{foil_type}",
                    "foil_type": foil_type,
                    "correct_image_id": image_ids[0],
                    "foil_image_id": image_ids[1],
                    "source_pair_image_ids": source_pair["image_ids"],
                    "target_direction": "randomized_from_pair",
                }
            )

    rng.shuffle(pairs)
    study_sequence = []
    test_phase = []
    for trial_index, pair in enumerate(pairs):
        study_sequence.append(
            {
                "study_index": trial_index,
                "image_id": pair["correct_image_id"],
                "metadata": {"foil_type": pair["foil_type"], "pair_id": pair["pair_id"]},
            }
        )
        options, target = choice_slots(rng, pair["correct_image_id"], pair["foil_image_id"])
        metadata = {
            "pair_id": pair["pair_id"],
            "pair_source": pair["pair_source"],
            "correct_image_id": pair["correct_image_id"],
            "foil_image_id": pair["foil_image_id"],
        }
        if "source_pair_image_ids" in pair:
            metadata["source_pair_image_ids"] = pair["source_pair_image_ids"]
            metadata["target_direction"] = pair["target_direction"]
        test_phase.append(
            {
                "trial_index": trial_index,
                "image_ids": options,
                "prompt": "Which of these two images (1 or 2) was in the study sequence?",
                "target": target,
                "target_type": "choice_index_1_based",
                "foil_type": pair["foil_type"],
                "metadata": metadata,
            }
        )

    episode = base_episode(
        "recognition_2afc_all",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_images": length, "foil_type": "all", "n_novel": n_each, "n_exemplar": n_each, "n_state": n_each},
        ["2afc_accuracy", "2afc_accuracy_by_foil_type"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": "Here is a sequence of images to remember.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_mnemonic_similarity(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    n_study = length
    if n_study * 2 > len(assets.mst_pairs):
        raise ValueError(f"MST study length {n_study} requires {n_study * 2} pairs; found {len(assets.mst_pairs)}.")

    studied = balanced_sample_by_key(rng, assets.mst_pairs, n_study, lambda pair: pair["bin"])
    studied_ids = {pair["pair_id"] for pair in studied}
    foil_pool = [pair for pair in assets.mst_pairs if pair["pair_id"] not in studied_ids]
    foils = balanced_sample_by_key(rng, foil_pool, n_study, lambda pair: pair["bin"])

    study_sequence = [
        {
            "study_index": index,
            "image_id": pair["target_image_id"],
            "pair_id": pair["pair_id"],
            "metadata": {"set_number": pair["set_number"], "item_id": pair["item_id"], "bin": pair["bin"]},
        }
        for index, pair in enumerate(studied)
    ]

    test_items = []
    for pair in studied:
        test_items.append({"image_id": pair["target_image_id"], "target": "old", "type": "target", "pair": pair, "bin": None})
        test_items.append({"image_id": pair["lure_image_id"], "target": "similar", "type": "lure", "pair": pair, "bin": pair["bin"]})
    for pair in foils:
        test_items.append({"image_id": pair["target_image_id"], "target": "new", "type": "foil", "pair": pair, "bin": None})
    rng.shuffle(test_items)

    prompt = "Is this image old, similar, or new? Respond with one word: old, similar, or new."
    test_phase = []
    for trial_index, item in enumerate(test_items):
        pair = item["pair"]
        test_phase.append(
            {
                "trial_index": trial_index,
                "image_id": item["image_id"],
                "prompt": prompt,
                "target": item["target"],
                "target_type": "old_similar_new_label",
                "type": item["type"],
                "metadata": {"pair_id": pair["pair_id"], "set_number": pair["set_number"], "item_id": pair["item_id"], "bin": item["bin"]},
            }
        )

    episode = base_episode(
        "mnemonic_similarity",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "MST", "study_length": n_study, "n_targets": n_study, "n_lures": n_study, "n_foils": n_study, "test_trials": len(test_phase), "sampling_policy": "bin_balanced_study_and_foil_pairs"},
        ["mst_ldi", "mst_accuracy_by_type", "mst_lure_accuracy_by_bin"],
    )
    episode.update({"dataset_name": "MST", "study_prompt": f"Study these {n_study} images carefully. You will later be tested on your memory for them.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode

def build_serial_order_free(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    image_ids = sample_without_replacement(rng, assets.brady_objects, length, "Brady object images")
    study_sequence = [
        {"study_index": index, "image_id": image_id, "serial_position": index + 1}
        for index, image_id in enumerate(image_ids)
    ]
    test_items = study_sequence[:]
    rng.shuffle(test_items)
    test_phase = [
        {
            "trial_index": trial_index,
            "image_id": item["image_id"],
            "prompt": f"What position in the sequence did this image appear (1-{length})?",
            "target": item["serial_position"],
            "target_type": "serial_position_1_based",
            "metadata": {"serial_position": item["serial_position"]},
        }
        for trial_index, item in enumerate(test_items)
    ]
    episode = base_episode(
        "serial_order_free",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_images": length, "test_trials": len(test_phase)},
        ["serial_order_absolute_error", "serial_order_rank_correlation"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": f"Here is a sequence of {length} images to remember in order.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_serial_order_2afc(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    image_ids = sample_without_replacement(rng, assets.brady_objects, length, "Brady object images")
    study_sequence = [
        {"study_index": index, "image_id": image_id, "serial_position": index + 1}
        for index, image_id in enumerate(image_ids)
    ]

    all_pairs = []
    for first, second in combinations(study_sequence, 2):
        serial_lag = abs(first["serial_position"] - second["serial_position"])
        all_pairs.append({"first": first, "second": second, "serial_lag": serial_lag, "lag_bin": lag_bin(serial_lag)})
    n_tests = min(length, len(all_pairs))
    selected_pairs = balanced_sample_by_key(rng, all_pairs, n_tests, lambda pair: pair["lag_bin"])

    test_phase = []
    for trial_index, pair_info in enumerate(selected_pairs):
        first = pair_info["first"]
        second = pair_info["second"]
        options = [first, second]
        rng.shuffle(options)
        target = 1 if options[0]["serial_position"] < options[1]["serial_position"] else 2
        serial_lag = pair_info["serial_lag"]
        test_phase.append(
            {
                "trial_index": trial_index,
                "image_ids": [options[0]["image_id"], options[1]["image_id"]],
                "prompt": "Which of these two images (1 or 2) appeared first in the study sequence?",
                "target": target,
                "target_type": "choice_index_1_based",
                "metadata": {
                    "left_serial_position": options[0]["serial_position"],
                    "right_serial_position": options[1]["serial_position"],
                    "first_serial_position": min(first["serial_position"], second["serial_position"]),
                    "second_serial_position": max(first["serial_position"], second["serial_position"]),
                    "serial_lag": serial_lag,
                    "lag_bin": pair_info["lag_bin"],
                    "distance": serial_lag - 1,
                },
            }
        )

    episode = base_episode(
        "serial_order_2afc",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_images": length, "n_tests": n_tests, "test_sampling": "lag_bin_balanced_subset_capped_at_sequence_length"},
        ["2afc_serial_order_accuracy", "2afc_serial_order_accuracy_by_lag"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": f"Here is a sequence of {length} images to remember in order.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode

def balanced_angles(rng: random.Random, n: int) -> list[float]:
    offset = rng.random() * 360.0 / max(n, 1)
    angles = [((index * 360.0 / n) + offset) % 360.0 for index in range(n)] if n else []
    rng.shuffle(angles)
    return angles


def build_color_memory_continuous(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    image_ids = sample_without_replacement(rng, assets.brady_color_objects, length, "Brady color objects")
    angles = balanced_angles(rng, length)
    transform = {"name": "brady_cielab_hue_rotation", "version": "0.1.0"}

    study_sequence = [
        {"study_index": index, "source_image_id": image_id, "target_angle_degrees": angles[index], "render_transform": transform["name"] + "_v" + transform["version"]}
        for index, image_id in enumerate(image_ids)
    ]
    test_items = study_sequence[:]
    rng.shuffle(test_items)
    test_phase = [
        {
            "trial_index": trial_index,
            "source_image_id": item["source_image_id"],
            "probe_transform": "grayscale",
            "prompt": "What was the color of this item in the study sequence? Report the hue angle in degrees.",
            "target": item["target_angle_degrees"],
            "target_type": "hue_angle_degrees",
            "metadata": {"target_angle_degrees": item["target_angle_degrees"]},
        }
        for trial_index, item in enumerate(test_items)
    ]
    episode = base_episode(
        "color_memory_continuous",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2013Color", "n_images": length, "color_transform": transform, "target_sampling": "balanced_rotation_angles_with_random_offset"},
        ["continuous_color_circular_error", "continuous_color_mean_absolute_error", "continuous_color_precision"],
    )
    episode.update({"dataset_name": "Brady2013Color", "study_prompt": "Remember the colors of these items.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_color_memory_named(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    image_ids = sample_without_replacement(rng, assets.brady_color_objects, length, "Brady color objects")
    colors = [COLOR_NAMES[index % len(COLOR_NAMES)] for index in range(length)]
    rng.shuffle(colors)
    transform = {"name": "brady_cielab_hue_rotation", "version": "0.1.0"}

    study_sequence = []
    for index, image_id in enumerate(image_ids):
        color_name, angle = colors[index]
        study_sequence.append(
            {"study_index": index, "source_image_id": image_id, "target_color_name": color_name, "target_angle_degrees": angle, "render_transform": transform["name"] + "_v" + transform["version"]}
        )
    test_items = study_sequence[:]
    rng.shuffle(test_items)
    test_phase = [
        {
            "trial_index": trial_index,
            "source_image_id": item["source_image_id"],
            "probe_transform": "grayscale",
            "prompt": "What was the color of this item in the study sequence? Respond with a color name.",
            "target": item["target_color_name"],
            "target_type": "color_name",
            "metadata": {"target_color_name": item["target_color_name"], "target_angle_degrees": item["target_angle_degrees"]},
        }
        for trial_index, item in enumerate(test_items)
    ]
    episode = base_episode(
        "color_memory_named",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2013Color", "n_images": length, "color_names": [name for name, _ in COLOR_NAMES], "color_transform": transform, "target_sampling": "balanced_named_color_rotation_angles"},
        ["named_color_accuracy", "named_color_confusion_matrix"],
    )
    episode.update({"dataset_name": "Brady2013Color", "study_prompt": "Remember the colors of these items.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_paired_associate_image_word(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    images = sample_without_replacement(rng, assets.brady_objects, length, "Brady object images")
    words = sample_without_replacement(rng, assets.wordpool, length, "wordpool words")
    pairs = list(zip(images, words))
    study_sequence = [
        {"study_index": index, "image_id": image_id, "word_id": word["word_id"], "word": word["word"]}
        for index, (image_id, word) in enumerate(pairs)
    ]
    test_items = study_sequence[:]
    rng.shuffle(test_items)
    test_phase = [
        {"trial_index": trial_index, "image_id": item["image_id"], "prompt": "What was the word paired with this image?", "target": item["word"], "target_type": "word", "metadata": {"target_word_id": item["word_id"]}}
        for trial_index, item in enumerate(test_items)
    ]
    episode = base_episode(
        "paired_associate_image_word",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_pairs": length},
        ["paired_associate_exact_match_accuracy"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": "Remember the word paired with each image.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_paired_associate_image_image(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    images = sample_without_replacement(rng, assets.brady_objects, length * 2, "Brady object images")
    cues = images[:length]
    targets = images[length:]
    study_sequence = [
        {"study_index": index, "cue_image_id": cues[index], "target_image_id": targets[index], "pair_type": "image_image"}
        for index in range(length)
    ]
    test_order = list(range(length))
    rng.shuffle(test_order)
    test_phase = []
    for trial_index, pair_index in enumerate(test_order):
        foil_candidates = [index for index in range(length) if index != pair_index]
        if foil_candidates:
            foil_index = rng.choice(foil_candidates)
            foil = targets[foil_index]
        else:
            foil = sample_without_replacement(rng, [img for img in assets.brady_objects if img not in images], 1, "unused Brady object foils")[0]
        options, target = choice_slots(rng, targets[pair_index], foil)
        test_phase.append(
            {
                "trial_index": trial_index,
                "cue_image_id": cues[pair_index],
                "image_ids": options,
                "prompt": "Which of these two images (1 or 2) was paired with the cue image?",
                "target": target,
                "target_type": "choice_index_1_based",
                "metadata": {"correct_target_image_id": targets[pair_index], "foil_image_id": foil},
            }
        )
    episode = base_episode(
        "paired_associate_image_image",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_pairs": length},
        ["paired_associate_2afc_accuracy"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": "Remember which images were paired together.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_associative_inference_image_word(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    images = sample_without_replacement(rng, assets.brady_objects, length * 2, "Brady object images")
    words = sample_without_replacement(rng, assets.wordpool, length + 1, "wordpool words")
    a_images = images[:length]
    b_images = images[length:]
    c_words = words[:length]
    extra_words = words[length:]

    study_sequence = []
    for chain_index in range(length):
        study_sequence.append({"study_index": len(study_sequence), "pair_type": "AB", "image_ids": [a_images[chain_index], b_images[chain_index]], "chain_index": chain_index})
    for chain_index in range(length):
        study_sequence.append({"study_index": len(study_sequence), "pair_type": "BC", "image_id": b_images[chain_index], "word_id": c_words[chain_index]["word_id"], "word": c_words[chain_index]["word"], "chain_index": chain_index})

    test_order = list(range(length))
    rng.shuffle(test_order)
    test_phase = []
    for trial_index, chain_index in enumerate(test_order):
        foil_candidates = [index for index in range(length) if index != chain_index]
        foil_word = c_words[rng.choice(foil_candidates)] if foil_candidates else extra_words[0]
        options, target = choice_slots(rng, c_words[chain_index]["word"], foil_word["word"])
        test_phase.append(
            {
                "trial_index": trial_index,
                "cue_image_id": a_images[chain_index],
                "options": options,
                "prompt": "Which of these two options (1 or 2) is indirectly associated with the cue image?",
                "target": target,
                "target_type": "choice_index_1_based",
                "metadata": {"chain_index": chain_index, "correct_word_id": c_words[chain_index]["word_id"], "foil_word_id": foil_word["word_id"]},
            }
        )
    episode = base_episode(
        "associative_inference_image_word",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_chains": length, "pair_type": "image_word", "study_order": "all_ab_then_all_bc"},
        ["associative_inference_2afc_accuracy"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": "Remember these pairs and the hidden connections between them.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


def build_associative_inference_image_image(assets: AssetIndex, suite: str, length: int, episode_index: int, seed: int) -> dict:
    rng = random.Random(seed)
    images_needed = length * 3
    images = sample_without_replacement(rng, assets.brady_objects, images_needed, "Brady object images")
    a_images = images[:length]
    b_images = images[length : length * 2]
    c_images = images[length * 2 :]
    unused = [image_id for image_id in assets.brady_objects if image_id not in set(images)]

    study_sequence = []
    for chain_index in range(length):
        study_sequence.append({"study_index": len(study_sequence), "pair_type": "AB", "image_ids": [a_images[chain_index], b_images[chain_index]], "chain_index": chain_index})
    for chain_index in range(length):
        study_sequence.append({"study_index": len(study_sequence), "pair_type": "BC", "image_ids": [b_images[chain_index], c_images[chain_index]], "chain_index": chain_index})

    test_order = list(range(length))
    rng.shuffle(test_order)
    test_phase = []
    for trial_index, chain_index in enumerate(test_order):
        foil_candidates = [index for index in range(length) if index != chain_index]
        if foil_candidates:
            foil = c_images[rng.choice(foil_candidates)]
        else:
            foil = sample_without_replacement(rng, unused, 1, "unused Brady object foils")[0]
        options, target = choice_slots(rng, c_images[chain_index], foil)
        test_phase.append(
            {
                "trial_index": trial_index,
                "cue_image_id": a_images[chain_index],
                "image_ids": options,
                "prompt": "Which of these two images (1 or 2) is indirectly associated with the cue image?",
                "target": target,
                "target_type": "choice_index_1_based",
                "metadata": {"chain_index": chain_index, "correct_c_image_id": c_images[chain_index], "foil_c_image_id": foil},
            }
        )
    episode = base_episode(
        "associative_inference_image_image",
        suite,
        length,
        episode_index,
        seed,
        {"dataset_name": "Brady2008", "n_chains": length, "pair_type": "image_image", "study_order": "all_ab_then_all_bc"},
        ["associative_inference_2afc_accuracy"],
    )
    episode.update({"dataset_name": "Brady2008", "study_prompt": "Remember these pairs and the hidden connections between them.", "study_sequence": study_sequence, "test_phase": test_phase})
    return episode


BUILDERS: dict[str, Callable[[AssetIndex, str, int, int, int], dict]] = {
    "recognition_continuous": build_recognition_continuous,
    "recognition_2afc_all": build_recognition_2afc_all,
    "mnemonic_similarity": build_mnemonic_similarity,
    "serial_order_free": build_serial_order_free,
    "serial_order_2afc": build_serial_order_2afc,
    "color_memory_continuous": build_color_memory_continuous,
    "color_memory_named": build_color_memory_named,
    "paired_associate_image_word": build_paired_associate_image_word,
    "paired_associate_image_image": build_paired_associate_image_image,
    "associative_inference_image_word": build_associative_inference_image_word,
    "associative_inference_image_image": build_associative_inference_image_image,
}


def seed_for(base_seed: int, suite: str, task_name: str, length_index: int, episode_index: int) -> int:
    return base_seed + SUITE_OFFSETS[suite] + TASK_OFFSETS[task_name] + length_index * 1_000 + episode_index


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--suites", nargs="+", choices=sorted(SUITES), default=list(SUITES))
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=TASK_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = AssetIndex(args.source_root.resolve())
    out_root = args.out.resolve()
    manifest_rows = []

    for suite in args.suites:
        episodes_per_length = SUITES[suite]
        for task_name in args.tasks:
            builder = BUILDERS[task_name]
            rows = []
            for length_index, length in enumerate(LENGTHS[task_name]):
                for episode_index in range(episodes_per_length):
                    seed = seed_for(args.base_seed, suite, task_name, length_index, episode_index)
                    rows.append(builder(assets, suite, length, episode_index, seed))
            output_path = out_root / suite / f"{task_name}_v1.jsonl"
            write_jsonl(output_path, rows)
            manifest_rows.append(
                {
                    "suite": suite,
                    "task_name": task_name,
                    "lengths": LENGTHS[task_name],
                    "episodes_per_length": episodes_per_length,
                    "n_episodes": len(rows),
                    "path": str(output_path.relative_to(out_root)),
                }
            )
            print(f"Wrote {len(rows):4d} episodes: {output_path}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_seed": args.base_seed,
        "suites": {suite: SUITES[suite] for suite in args.suites},
        "tasks": manifest_rows,
    }
    manifest_path = out_root / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
