#!/usr/bin/env python
"""
Prefetch every COCO image referenced by Visual Haystacks QA JSONs (no full zips).

Requires VHs QA files under --qa-root (e.g. from Hugging Face). Downloads from
https://images.cocodataset.org/ into --image-root. Can take a long time and use
significant disk — same images as a full local COCO mirror for *those* paths, but
you never store the unused remainder of each zip.

Usage:
  python -m eval_scripts.prefetch_vhs_coco --qa-root dataset/VHs_qa --image-root dataset/coco
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.visual_haystacks import (
    COCO_IMAGE_BASE_URL,
    iter_unique_image_paths_in_vhs_qa,
    prefetch_vhs_coco_images,
)


def main():
    parser = argparse.ArgumentParser(description="Prefetch COCO images for Visual Haystacks")
    parser.add_argument("--qa-root", type=str, default="dataset/VHs_qa", help="VHs QA directory")
    parser.add_argument(
        "--image-root",
        type=str,
        default="dataset/coco",
        help="Where to write train2017/val2017/test2017 JPEGs",
    )
    parser.add_argument(
        "--coco-base-url",
        type=str,
        default=COCO_IMAGE_BASE_URL,
        help="COCO image CDN base URL",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=120,
        help="Per-image HTTP timeout (seconds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many unique image paths were found; do not download",
    )
    args = parser.parse_args()

    qa_root = Path(args.qa_root)
    if not qa_root.is_dir():
        print(f"QA root not found: {qa_root}", file=sys.stderr)
        sys.exit(1)

    unique = iter_unique_image_paths_in_vhs_qa(qa_root)
    print(f"Unique image paths referenced across all visual_haystack_*.json: {len(unique)}")

    if args.dry_run:
        print("Dry run: no downloads.")
        sys.exit(0)

    stats = prefetch_vhs_coco_images(
        qa_root=args.qa_root,
        image_root=args.image_root,
        coco_base_url=args.coco_base_url,
        fetch_timeout_s=args.fetch_timeout,
        skip_existing=True,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
