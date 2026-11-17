"""Fixed hue palette for the color memory tasks.

All color-task stimuli are pre-rendered at PALETTE_HUES (every 10 degrees of CIELAB hue
rotation) by hf_scripts/render_color_palette.py, so local runs and the Hugging Face
episodes draw from exactly the same image files.

Stdlib only, so the episode generator can import it without heavy dependencies.
"""

from __future__ import annotations

import random

PALETTE_STEP_DEGREES = 10
PALETTE_HUES = list(range(0, 360, PALETTE_STEP_DEGREES))

RENDERED_DIR_NAME = "Brady2013ColorRendered"   # under memory_datasets/
HF_COLLECTION_ID = "brady_color_rendered"      # under data/ in the HF dataset
COLOR_WHEEL_FILE = "color_wheel.png"
JPEG_QUALITY = 92


def hue_dir(hue: int) -> str:
    return f"h{int(hue):03d}"


def colored_file_name(source_name: str, hue: int) -> str:
    """Path of a rendered colored object, relative to the rendered collection root."""
    return f"images/{hue_dir(hue)}/{source_name}"


def gray_file_name(source_name: str) -> str:
    return f"images/gray/{source_name}"


def colored_image_id(source_name: str, hue: int) -> str:
    return f"{HF_COLLECTION_ID}/{hue_dir(hue)}/{source_name}"


def gray_image_id(source_name: str) -> str:
    return f"{HF_COLLECTION_ID}/gray/{source_name}"


COLOR_WHEEL_IMAGE_ID = f"{HF_COLLECTION_ID}/{COLOR_WHEEL_FILE}"


def snap_to_palette(angle: float) -> int:
    """Nearest palette hue to an arbitrary angle."""
    return int(round(angle / PALETTE_STEP_DEGREES) * PALETTE_STEP_DEGREES) % 360


def opposite_hue(hue: int) -> int:
    """Hue 180 degrees away (always in the palette, since 36 is even)."""
    return (int(hue) + 180) % 360


def sample_palette_hues(rng: random.Random, n: int) -> list[int]:
    """n palette hues spread evenly around the wheel with a random offset, then shuffled.

    For n <= 36 all hues are distinct; for larger n every hue is used ~n/36 times.
    """
    if n <= 0:
        return []
    k = len(PALETTE_HUES)
    offset = rng.random()
    hues = [PALETTE_HUES[int((index + offset) * k / n) % k] for index in range(n)]
    rng.shuffle(hues)
    return hues
