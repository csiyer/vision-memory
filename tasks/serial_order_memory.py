import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, ThingsDataset


class SerialOrderMemoryBase:
    def __init__(self, dataset_name="things", n_images=20):
        self.dataset_name = dataset_name
        self.n_images = n_images
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == "things":
            return ThingsDataset(n_categories=self.n_images)
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects")
        return ThingsDataset(n_categories=self.n_images)

    def _sample_study_items(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]

        study_sequence = []
        study_items = []
        for serial_position, dataset_index in enumerate(selected_indices, start=1):
            image = self.dataset.get_image(dataset_index)
            metadata = {
                **self.dataset.get_metadata(dataset_index),
                "serial_position": serial_position,
            }
            study_sequence.append(image)
            study_items.append(
                {
                    "dataset_index": dataset_index,
                    "serial_position": serial_position,
                    "image": image,
                    "metadata": metadata,
                }
            )
        return study_sequence, study_items


class SerialOrderMemoryTask(SerialOrderMemoryBase):
    def get_trials(self):
        study_sequence, study_items = self._sample_study_items()
        n = len(study_items)

        test_indices = list(range(n))
        random.shuffle(test_indices)

        test_phase = []
        for item_index in test_indices:
            item = study_items[item_index]
            test_phase.append(
                {
                    "image": item["image"],
                    "prompt": f"When in the sequence did this image occur (1-{n})?",
                    "target": item["serial_position"],
                    "metadata": item["metadata"],
                }
            )

        return {
            "study_prompt": f"Here is a sequence of {n} images to remember in order.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


class AFCSerialOrderMemoryTask(SerialOrderMemoryBase):
    def __init__(self, dataset_name="things", n_images=20, n_tests=None):
        super().__init__(dataset_name=dataset_name, n_images=n_images)
        self.n_tests = n_tests

    def get_trials(self):
        study_sequence, study_items = self._sample_study_items()
        n = len(study_items)

        pair_indices = list(combinations(range(n), 2))
        random.shuffle(pair_indices)

        n_tests = min(self.n_tests, len(pair_indices)) if self.n_tests is not None else min(n, len(pair_indices))
        selected_pairs = pair_indices[:n_tests]

        test_phase = []
        for left_idx, right_idx in selected_pairs:
            first_item = study_items[left_idx]
            second_item = study_items[right_idx]
            pair = [first_item, second_item]
            random.shuffle(pair)

            target = 1 if pair[0]["serial_position"] < pair[1]["serial_position"] else 2
            serial_lag = abs(first_item["serial_position"] - second_item["serial_position"])
            distance = serial_lag - 1

            test_phase.append(
                {
                    "images": [pair[0]["image"], pair[1]["image"]],
                    "prompt": "Which of these two images appeared first in the study sequence? (1 or 2)",
                    "target": target,
                    "metadata": {
                        "first_serial_position": min(first_item["serial_position"], second_item["serial_position"]),
                        "second_serial_position": max(first_item["serial_position"], second_item["serial_position"]),
                        "serial_lag": serial_lag,
                        "distance": distance,
                        "left_image_serial_position": pair[0]["serial_position"],
                        "right_image_serial_position": pair[1]["serial_position"],
                    },
                }
            )

        return {
            "study_prompt": f"Here is a sequence of {n} images to remember in order.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


__all__ = ["SerialOrderMemoryTask", "AFCSerialOrderMemoryTask"]
