#!/usr/bin/env python3
"""Render the fixed color palette used by the color memory tasks.

Reads memory_datasets/Brady2013ColorObjects and writes memory_datasets/Brady2013ColorRendered:
  images/h000/ ... images/h350/   each object hue-rotated in CIELAB space (10-degree steps)
  images/gray/                    grayscale probes
  images/color_wheel.png          response wheel for continuous report
  metadata.jsonl                  one row per image, keyed by HF-style image_id

Run once after unzipping memory_datasets_brady.zip. export_hf_assets.py copies the
result into the Hugging Face dataset, so local runs and HF episodes use identical files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.color_palette import (  # noqa: E402
    COLOR_WHEEL_FILE,
    COLOR_WHEEL_IMAGE_ID,
    JPEG_QUALITY,
    PALETTE_HUES,
    RENDERED_DIR_NAME,
    colored_file_name,
    colored_image_id,
    gray_file_name,
    gray_image_id,
)
from tasks.color_memory import generate_cielab_color_wheel, rotate_image_hue  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "memory_datasets")
    parser.add_argument("--force", action="store_true", help="Re-render images that already exist.")
    args = parser.parse_args()

    src_dir = args.source_root / "Brady2013ColorObjects"
    out_dir = args.source_root / RENDERED_DIR_NAME
    if not src_dir.exists():
        raise SystemExit(f"Missing {src_dir}. Unzip memory_datasets_brady.zip first.")

    sources = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    rows = []
    n_written = 0
    for obj_index, src in enumerate(sources):
        source_id = f"brady_color_objects/{src.name}"
        img = np.array(Image.open(src).convert("RGB"))

        gray_path = out_dir / gray_file_name(src.name)
        if args.force or not gray_path.exists():
            gray_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).convert("L").convert("RGB").save(gray_path, quality=JPEG_QUALITY)
            n_written += 1
        rows.append({"image_id": gray_image_id(src.name), "file_name": gray_file_name(src.name),
                     "source_image_id": source_id, "object_index": obj_index, "hue_degrees": None, "kind": "gray"})

        for hue in PALETTE_HUES:
            path = out_dir / colored_file_name(src.name, hue)
            if args.force or not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rotate_image_hue(img, hue)).save(path, quality=JPEG_QUALITY)
                n_written += 1
            rows.append({"image_id": colored_image_id(src.name, hue), "file_name": colored_file_name(src.name, hue),
                         "source_image_id": source_id, "object_index": obj_index, "hue_degrees": hue, "kind": "colored"})
        if (obj_index + 1) % 50 == 0:
            print(f"  {obj_index + 1}/{len(sources)} objects")

    wheel_path = out_dir / "images" / COLOR_WHEEL_FILE
    if args.force or not wheel_path.exists():
        wheel_path.parent.mkdir(parents=True, exist_ok=True)
        generate_cielab_color_wheel().save(wheel_path)
        n_written += 1
    rows.append({"image_id": COLOR_WHEEL_IMAGE_ID, "file_name": f"images/{COLOR_WHEEL_FILE}",
                 "source_image_id": None, "object_index": None, "hue_degrees": None, "kind": "color_wheel"})

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Rendered {n_written} new files; {len(rows)} images indexed in {out_dir}")


if __name__ == "__main__":
    main()
