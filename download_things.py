"""Download THINGS dataset images to memory_datasets/THINGS/images/.

Run once on the login node before submitting SLURM jobs:
    python3 download_things.py

Images are saved as:
    memory_datasets/THINGS/images/<category_name>/<exemplar_index>.jpg
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from datasets import load_dataset
from PIL import Image

DEST = Path(__file__).parent / "memory_datasets" / "THINGS" / "images"
DATASET_NAME = "Haitao999/things-eeg"
EXEMPLARS_PER_CATEGORY = 10  # Download up to 10 exemplars per category

def main():
    DEST.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} (streaming)...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    category_counts = {}
    saved = 0
    skipped = 0

    for item in ds:
        cat = item.get("category") or item.get("concept") or f"cat_{item.get('label', 0)}"
        cat_safe = cat.replace("/", "_").replace(" ", "_")
        cat_dir = DEST / cat_safe

        count = category_counts.get(cat_safe, 0)
        if count >= EXEMPLARS_PER_CATEGORY:
            skipped += 1
            continue

        cat_dir.mkdir(exist_ok=True)
        out_path = cat_dir / f"{count}.jpg"

        if not out_path.exists():
            img = item["image"]
            if not isinstance(img, Image.Image):
                import numpy as np
                img = Image.fromarray(np.array(img))
            img.convert("RGB").save(out_path, format="JPEG", quality=90)
            saved += 1
        else:
            saved += 1  # Already exists, count it

        category_counts[cat_safe] = count + 1

        if saved % 500 == 0:
            print(f"  {len(category_counts)} categories, {saved} images saved...")

    print(f"\nDone. {len(category_counts)} categories, {saved} images in {DEST}")

if __name__ == "__main__":
    main()
