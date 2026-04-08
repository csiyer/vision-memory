import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

# Official COCO 2017 image hosting (same paths as local train2017/val2017/test2017).
COCO_IMAGE_BASE_URL = "https://images.cocodataset.org"


def resolve_coco_image_path(image_root: Path, rel_path: str) -> Path:
    """Map a VHs/COCO-relative path to a file under ``image_root``."""
    candidate = image_root / rel_path
    if candidate.exists():
        return candidate

    parts = Path(rel_path).parts
    if "coco" in parts:
        coco_idx = parts.index("coco")
        candidate = image_root / Path(*parts[coco_idx + 1 :])
        if candidate.exists():
            return candidate

    return image_root / rel_path


def is_coco_split_relpath(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if len(parts) < 2:
        return False
    return parts[0] in ("train2017", "val2017", "test2017")


def download_coco_image(
    image_root: Path,
    rel_path: str,
    *,
    coco_base_url: str = COCO_IMAGE_BASE_URL,
    fetch_timeout_s: int = 120,
) -> Path:
    """Download one COCO image from the public CDN into ``image_root``."""
    base = coco_base_url.rstrip("/")
    url = f"{base}/{rel_path.replace('\\', '/')}"
    dest = resolve_coco_image_path(image_root, rel_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "vision-memory-vhs/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=fetch_timeout_s) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(
            f"Could not fetch COCO image from '{url}' (HTTP {e.code}). "
            "Check network or download the full COCO split locally."
        ) from e
    except OSError as e:
        raise FileNotFoundError(f"Could not fetch COCO image from '{url}': {e}") from e
    dest.write_bytes(data)
    return dest


def iter_unique_image_paths_in_vhs_qa(qa_root: Path) -> set[str]:
    """All unique ``pos_image`` / ``neg_image`` paths across every ``visual_haystack_*.json``."""
    paths: set[str] = set()
    for json_path in sorted(qa_root.rglob("visual_haystack_*.json")):
        with open(json_path, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            paths.update(entry.get("pos_image") or [])
            paths.update(entry.get("neg_image") or [])
    return paths


def prefetch_vhs_coco_images(
    qa_root: str | Path = "dataset/VHs_qa",
    image_root: str | Path = "dataset/coco",
    *,
    coco_base_url: str = COCO_IMAGE_BASE_URL,
    fetch_timeout_s: int = 120,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Download every COCO image referenced by Visual Haystacks QA JSONs (on-demand URLs).

    This avoids downloading full COCO zips; total disk use can still be large because VHs
    may reference many unique images across all haystack sizes and splits.

    Returns counts: ``{"total": n, "downloaded": d, "skipped": s, "failed": f}``.
    """
    qa_root = Path(qa_root)
    image_root = Path(image_root)
    rel_paths = iter_unique_image_paths_in_vhs_qa(qa_root)
    downloaded = 0
    skipped = 0
    failed = 0
    skipped_non_coco = 0
    for rel in sorted(rel_paths):
        if not is_coco_split_relpath(rel):
            skipped_non_coco += 1
            continue
        dest = resolve_coco_image_path(image_root, rel)
        if skip_existing and dest.exists():
            skipped += 1
            continue
        try:
            download_coco_image(
                image_root,
                rel,
                coco_base_url=coco_base_url,
                fetch_timeout_s=fetch_timeout_s,
            )
            downloaded += 1
        except (OSError, FileNotFoundError):
            failed += 1
    return {
        "total": len(rel_paths),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "skipped_non_coco_paths": skipped_non_coco,
        "failed": failed,
    }


class VisualHaystacksTask:
    """Loader for Visual Haystacks benchmark JSON and COCO image files."""

    def __init__(
        self,
        qa_root: str = "dataset/VHs_qa",
        image_root: str = "dataset/coco",
        mode: str = "single_needle",
        split: str = "VHs_large",
        image_count: str = "10",
        max_samples: int | None = None,
        shuffle_images: bool = True,
        seed: int = 0,
        fetch_missing_coco: bool = False,
        coco_base_url: str = COCO_IMAGE_BASE_URL,
        fetch_timeout_s: int = 120,
    ):
        self.qa_root = Path(qa_root)
        self.image_root = Path(image_root)
        self.mode = mode
        self.split = split
        self.image_count = str(image_count)
        self.max_samples = max_samples
        self.shuffle_images = shuffle_images
        self.seed = seed
        self.fetch_missing_coco = fetch_missing_coco
        self.coco_base_url = coco_base_url.rstrip("/")
        self.fetch_timeout_s = fetch_timeout_s

        self._rng = random.Random(seed)
        self.samples = self._load_samples()

    def _test_file(self) -> Path:
        if self.mode == "single_needle":
            base = self.qa_root / self.mode / self.split
        else:
            base = self.qa_root / self.mode
        return base / f"visual_haystack_{self.image_count}.json"

    def _load_samples(self) -> List[Dict[str, Any]]:
        test_file = self._test_file()
        if not test_file.exists():
            raise FileNotFoundError(
                f"Visual Haystacks QA file not found: '{test_file}'. "
                "On-demand download only fetches COCO images (--fetch-missing-coco); it does not "
                "install the benchmark JSONs. Download the VHs dataset first, e.g.\n"
                "  hf download --repo-type dataset tsunghanwu/visual_haystacks --local-dir dataset/VHs_qa\n"
                "Then pass --qa-root if you put it elsewhere."
            )

        with open(test_file, "r") as f:
            entries = json.load(f)

        if self.max_samples is not None and self.max_samples > 0:
            entries = entries[: self.max_samples]
        return entries

    def _open_image(self, rel_path: str) -> Image.Image:
        image_path = resolve_coco_image_path(self.image_root, rel_path)
        if not image_path.exists():
            if self.fetch_missing_coco and is_coco_split_relpath(rel_path):
                download_coco_image(
                    self.image_root,
                    rel_path,
                    coco_base_url=self.coco_base_url,
                    fetch_timeout_s=self.fetch_timeout_s,
                )
            else:
                hint = ""
                first = Path(rel_path).parts[0] if rel_path else ""
                if first in ("train2017", "val2017", "test2017"):
                    hint = (
                        f" COCO 2017 includes {first}; either download and unzip it under "
                        f"'{self.image_root}/{first}/', or run eval with "
                        f"'--fetch-missing-coco' to download individual images on demand, "
                        f"or `python -m eval_scripts.prefetch_vhs_coco` to prefetch all VHs images."
                    )
                raise FileNotFoundError(
                    f"Image not found for VHs sample: '{rel_path}' "
                    f"(looked for '{image_path}').{hint}"
                )
        return Image.open(image_path).convert("RGB")

    def get_trials(self) -> List[Dict[str, Any]]:
        trials: List[Dict[str, Any]] = []
        for idx, entry in enumerate(self.samples):
            question = entry["conversations"][0]["value"]
            answer = str(entry["conversations"][1]["value"]).strip().lower()

            image_lists = list(entry.get("pos_image", [])) + list(entry.get("neg_image", []))
            if self.shuffle_images:
                self._rng.shuffle(image_lists)

            images = [self._open_image(p) for p in image_lists]

            trials.append(
                {
                    "trial": idx,
                    "prompt": (
                        "You are given a set of images. "
                        "Answer the following question with only 'yes' or 'no': "
                        f"{question}"
                    ),
                    "images": images,
                    "target": answer,
                    "metadata": {
                        "mode": self.mode,
                        "split": self.split,
                        "image_count": self.image_count,
                        "n_images": len(image_lists),
                        "pos_images": entry.get("pos_image", []),
                        "neg_images": entry.get("neg_image", []),
                        "shuffled_image_paths": image_lists,
                    },
                }
            )
        return trials
