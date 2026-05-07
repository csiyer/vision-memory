#!/usr/bin/env python3
"""Export local vision-memory assets into a Hugging Face dataset repo layout.

This script writes stable asset metadata and, by default, copies the underlying
image/text files into a sibling HF dataset checkout such as ../vision-memory-tasks.
It does not upload anything; commit/push from the dataset repo when ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "memory_datasets"
DEFAULT_OUT = REPO_ROOT.parent / "vision-memory-tasks"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BRADY_COLLECTIONS = {
    "brady_objects": "Brady2008Objects",
    "brady_exemplar": "Brady2008Exemplar",
    "brady_state": "Brady2008State",
    "brady_color_objects": "Brady2013ColorObjects",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_paths(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def brady_pair_rows(paths: list[Path], *, collection_id: str) -> list[dict]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        stem = path.stem
        base = stem[:-1] if stem[-1:].isdigit() else stem
        groups.setdefault(base.lower(), []).append(path)

    rows = []
    for group_index, base_name in enumerate(sorted(groups)):
        members = sorted(groups[base_name], key=lambda path: path.name.lower())
        if len(members) != 2:
            continue
        rows.append(
            {
                "pair_id": f"{collection_id}/{base_name}",
                "collection_id": collection_id,
                "group_name": base_name,
                "image_ids": [f"{collection_id}/{member.name}" for member in members],
                "role_labels": ["original", "foil"],
                "pair_index": group_index,
            }
        )
    return rows


def export_brady_collection(source_root: Path, out_root: Path, collection_id: str, directory_name: str, *, dry_run: bool) -> dict:
    source_dir = source_root / directory_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing Brady source directory: {source_dir}")

    rows = []
    paths = image_paths(source_dir)
    collection_out = out_root / "data" / collection_id

    for index, src in enumerate(paths):
        dst = collection_out / "images" / src.name
        copy_file(src, dst, dry_run=dry_run)
        rows.append(
            {
                "image_id": f"{collection_id}/{src.name}",
                "collection_id": collection_id,
                "source_dataset": directory_name,
                "file_name": f"images/{src.name}",
                "original_file_name": src.name,
                "asset_index": index,
            }
        )

    write_jsonl(collection_out / "metadata.jsonl", rows)

    pair_count = None
    if collection_id in {"brady_exemplar", "brady_state"}:
        pair_rows = brady_pair_rows(paths, collection_id=collection_id)
        write_jsonl(collection_out / "pairs.jsonl", pair_rows)
        pair_count = len(pair_rows)

    return {
        "collection_id": collection_id,
        "source_dataset": directory_name,
        "n_images": len(rows),
        "n_pairs": pair_count,
    }


def parse_mst_bins(path: Path) -> dict[int, int]:
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


def export_mst(source_root: Path, out_root: Path, *, dry_run: bool) -> dict:
    source_dir = source_root / "MST"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing MST source directory: {source_dir}")

    collection_out = out_root / "data" / "mst"
    image_rows = []
    pair_rows = []
    asset_index = 0

    for set_number in range(1, 7):
        set_dir = source_dir / f"Set {set_number}"
        if not set_dir.exists():
            continue
        bin_map = parse_mst_bins(source_dir / f"Set{set_number} bins.txt")
        target_paths = sorted(
            [path for path in set_dir.iterdir() if path.is_file() and path.stem.endswith("a") and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: path.name.lower(),
        )
        for target_path in target_paths:
            item_stem = target_path.stem[:-1]
            try:
                item_id = int(item_stem)
            except ValueError:
                continue
            lure_path = target_path.with_name(f"{item_stem}b{target_path.suffix}")
            if not lure_path.exists():
                continue

            image_ids = []
            for role, src in (("target", target_path), ("lure", lure_path)):
                dst = collection_out / "images" / f"set_{set_number}" / src.name
                copy_file(src, dst, dry_run=dry_run)
                image_id = f"mst/set_{set_number}/{src.name}"
                image_ids.append(image_id)
                image_rows.append(
                    {
                        "image_id": image_id,
                        "collection_id": "mst",
                        "source_dataset": "MST",
                        "file_name": f"images/set_{set_number}/{src.name}",
                        "original_file_name": src.name,
                        "set_number": set_number,
                        "item_id": item_id,
                        "mst_role": role,
                        "bin": bin_map.get(item_id),
                        "asset_index": asset_index,
                    }
                )
                asset_index += 1

            pair_rows.append(
                {
                    "pair_id": f"mst/set_{set_number}/{item_id:03d}",
                    "collection_id": "mst",
                    "set_number": set_number,
                    "item_id": item_id,
                    "target_image_id": image_ids[0],
                    "lure_image_id": image_ids[1],
                    "bin": bin_map.get(item_id),
                }
            )

    write_jsonl(collection_out / "metadata.jsonl", image_rows)
    write_jsonl(collection_out / "pairs.jsonl", pair_rows)

    readme = source_dir / "README.md"
    if readme.exists():
        copy_file(readme, collection_out / "SOURCE_README.md", dry_run=dry_run)

    return {"collection_id": "mst", "source_dataset": "MST", "n_images": len(image_rows), "n_pairs": len(pair_rows)}


def export_wordpool(source_root: Path, out_root: Path, *, dry_run: bool) -> dict | None:
    src = source_root / "wasnorm_wordpool.txt"
    if not src.exists():
        return None
    dst = out_root / "data" / "wordpool" / "wasnorm_wordpool.txt"
    copy_file(src, dst, dry_run=dry_run)
    words = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_jsonl(
        out_root / "data" / "wordpool" / "metadata.jsonl",
        [{"word_id": f"wordpool/{index:05d}", "word": word, "asset_index": index} for index, word in enumerate(words)],
    )
    return {"collection_id": "wordpool", "n_words": len(words)}


def export_benchmarks(out_root: Path, *, dry_run: bool) -> dict:
    literature_dir = REPO_ROOT / "literature"
    benchmark_out = out_root / "benchmarks"
    rows = []
    for src in sorted(literature_dir.glob("*.json")):
        copy_file(src, benchmark_out / src.name, dry_run=dry_run)
        rows.append({"benchmark_file": src.name, "source_path": f"benchmarks/{src.name}"})
    write_jsonl(benchmark_out / "metadata.jsonl", rows)
    return {"collection_id": "human_benchmarks", "n_files": len(rows)}


def write_dataset_card(out_root: Path, manifest: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    readme = out_root / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "\n".join(
            [
                "---",
                "pretty_name: Vision Memory Tasks",
                "tags:",
                "- vision",
                "- memory",
                "- psychology",
                "- benchmark",
                "---",
                "",
                "# Vision Memory Tasks",
                "",
                "This dataset hosts visual memory task assets and standardized episode manifests generated from the companion code repository.",
                "",
                "The dataset is currently an initial private export. Licensing, citations, and standardized task schemas should be reviewed before public release.",
                "",
                "## Export Manifest",
                "",
                "```json",
                json.dumps(manifest, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Write metadata manifests but skip copying files and README creation.")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["brady", "mst", "wordpool", "benchmarks"],
        choices=["brady", "mst", "wordpool", "benchmarks"],
        help="Asset groups to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out.resolve()
    source_root = args.source_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exported = []
    if "brady" in args.collections:
        for collection_id, directory_name in BRADY_COLLECTIONS.items():
            exported.append(export_brady_collection(source_root, out_root, collection_id, directory_name, dry_run=args.dry_run))
    if "mst" in args.collections:
        exported.append(export_mst(source_root, out_root, dry_run=args.dry_run))
    if "wordpool" in args.collections:
        wordpool_summary = export_wordpool(source_root, out_root, dry_run=args.dry_run)
        if wordpool_summary:
            exported.append(wordpool_summary)
    if "benchmarks" in args.collections:
        exported.append(export_benchmarks(out_root, dry_run=args.dry_run))

    manifest = {
        "schema_version": "0.1.0",
        "source_repo": str(REPO_ROOT),
        "source_root": str(source_root),
        "collections": exported,
    }
    manifest_path = out_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_dataset_card(out_root, manifest, dry_run=args.dry_run)

    print(f"Wrote export manifest: {manifest_path}")
    for item in exported:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
