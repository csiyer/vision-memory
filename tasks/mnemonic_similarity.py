import csv
import random
from pathlib import Path

from PIL import Image


class MSTDataset:
    """
    Loads MST stimulus sets from memory_datasets/MST/.
    Expected layout:
        memory_datasets/MST/Set 1/001a.jpg  (target)
        memory_datasets/MST/Set 1/001b.jpg  (lure)
        ...
        memory_datasets/MST/Set1 bins.txt   (tab-delimited: item_num TAB bin_num)
    Set numbers 1-6 are used by default.
    """

    def __init__(self, set_numbers=None, root="memory_datasets/MST"):
        if set_numbers is None:
            set_numbers = [1, 2, 3, 4, 5, 6]
        self.root = Path(root)
        self.pairs = []  # list of dicts: target_path, lure_path, bin, set_number, item_id
        self._load(set_numbers)

    def _load(self, set_numbers):
        for set_num in set_numbers:
            set_dir = self.root / f"Set {set_num}"
            bins_path = self.root / f"Set{set_num} bins.txt"

            if not set_dir.exists():
                print(f"Warning: MST Set {set_num} not found at {set_dir}")
                continue

            # Parse bin assignments: item_number -> bin (1-5)
            bin_map = self._parse_bins(bins_path)

            target_paths = sorted(set_dir.glob("*a.jpg")) + sorted(set_dir.glob("*a.png"))
            for target_path in target_paths:
                stem = target_path.stem  # e.g. "001a"
                item_id_str = stem[:-1]  # strip trailing 'a'
                try:
                    item_id = int(item_id_str)
                except ValueError:
                    continue

                lure_path = target_path.with_name(item_id_str + "b" + target_path.suffix)
                if not lure_path.exists():
                    continue

                self.pairs.append({
                    "target_path": target_path,
                    "lure_path": lure_path,
                    "bin": bin_map.get(item_id),
                    "set_number": set_num,
                    "item_id": item_id,
                })

        if not self.pairs:
            print(
                "Warning: No MST pairs loaded. Unzip memory_datasets.zip and ensure "
                "memory_datasets/MST/Set 1/ ... Set 6/ directories exist."
            )

    def _parse_bins(self, bins_path):
        bin_map = {}
        if not bins_path.exists():
            return bin_map
        with open(bins_path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if not row:
                    continue
                try:
                    if len(row) >= 2:
                        item_id = int(row[0])
                        bin_num = int(row[1])
                    else:
                        # Single-column format: row index + 1 is item_id
                        item_id = i + 1
                        bin_num = int(row[0])
                    bin_map[item_id] = bin_num
                except (ValueError, IndexError):
                    continue
        return bin_map

    def __len__(self):
        return len(self.pairs)


class MnemonicSimilarityTask:
    """
    Study-test MST. Study phase shows target images. Test phase mixes:
      - targets  (correct response: 'old')
      - lures    (correct response: 'similar') — similar to a studied item, binned 1-5
      - foils    (correct response: 'new')    — unstudied, unrelated images

    Default design follows Stark et al.: study 128 items, test equal thirds
    (n_study targets + n_study lures + n_study foils).
    """

    PROMPT = (
        "Is this image: (1) old — exactly as you saw it before, "
        "(2) similar — like something you saw but not identical, or "
        "(3) new — not seen before? "
        "Respond with one word: old, similar, or new."
    )

    def __init__(self, set_numbers=None, n_study=128, root="memory_datasets/MST"):
        if set_numbers is None:
            set_numbers = [1, 2, 3, 4, 5, 6]
        self.n_study = n_study
        self.dataset = MSTDataset(set_numbers=set_numbers, root=root)

    def get_trials(self):
        pairs = list(self.dataset.pairs)
        if not pairs:
            raise ValueError("No MST stimuli loaded. Check memory_datasets/MST/ directory.")

        n = min(self.n_study, len(pairs) // 2)  # need half for foils
        random.shuffle(pairs)
        studied = pairs[:n]
        unstudied = pairs[n:]

        n_foils = min(n, len(unstudied))
        foils = random.sample(unstudied, n_foils)

        study_sequence = [
            Image.open(p["target_path"]).convert("RGB") for p in studied
        ]

        test_items = []

        for pair in studied:
            test_items.append({
                "image": Image.open(pair["target_path"]).convert("RGB"),
                "prompt": self.PROMPT,
                "target": "old",
                "type": "target",
                "metadata": {
                    "set_number": pair["set_number"],
                    "item_id": pair["item_id"],
                    "bin": None,
                },
            })

        for pair in studied:
            test_items.append({
                "image": Image.open(pair["lure_path"]).convert("RGB"),
                "prompt": self.PROMPT,
                "target": "similar",
                "type": "lure",
                "metadata": {
                    "set_number": pair["set_number"],
                    "item_id": pair["item_id"],
                    "bin": pair["bin"],
                },
            })

        for pair in foils:
            test_items.append({
                "image": Image.open(pair["target_path"]).convert("RGB"),
                "prompt": self.PROMPT,
                "target": "new",
                "type": "foil",
                "metadata": {
                    "set_number": pair["set_number"],
                    "item_id": pair["item_id"],
                    "bin": None,
                },
            })

        random.shuffle(test_items)

        return {
            "study_prompt": (
                f"Study these {n} images carefully. "
                "You will later be tested on your memory for them."
            ),
            "study_sequence": study_sequence,
            "test_phase": test_items,
        }


if __name__ == "__main__":
    task = MnemonicSimilarityTask(n_study=10)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"Test phase length: {len(results['test_phase'])}")
    types = [t["type"] for t in results["test_phase"]]
    print(f"  targets: {types.count('target')}, lures: {types.count('lure')}, foils: {types.count('foil')}")
