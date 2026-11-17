import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.color_palette import opposite_hue, sample_palette_hues
from src.stimuli import ColorPaletteDataset

# (name, hue_angle_degrees) — hue angles chosen as perceptual centers on the HSV wheel
NAMED_COLORS = [
    ("red", 0),
    ("orange", 30),
    ("yellow", 60),
    ("green", 120),
    ("blue", 240),
    ("purple", 300),
]


def rotate_image_hue(img, angle_deg):
    """Rotate image hue in CIELAB space, matching the Brady color helper."""
    import cv2

    img_float = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

    a = lab[:, :, 1]
    b = lab[:, :, 2]
    orig_shape = a.shape
    v = np.stack((a.flatten(), b.flatten()), axis=0)

    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    v_rotated = rotation_matrix @ v

    lab[:, :, 1] = v_rotated[0, :].reshape(orig_shape)
    lab[:, :, 2] = v_rotated[1, :].reshape(orig_shape)

    rgb_rotated = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(rgb_rotated * 255, 0, 255).astype(np.uint8)


def generate_cielab_color_wheel(size=360, lightness=70, chroma=55):
    """Generate a smooth CIELAB hue wheel for continuous degree reports."""
    import cv2

    center = (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    dx = x - center
    dy = center - y
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = center * 0.92
    mask = radius <= max_radius

    angles = np.arctan2(dy, dx)
    scaled_chroma = chroma * (radius / max_radius)

    lab = np.zeros((size, size, 3), dtype=np.float32)
    lab[:, :, 0] = lightness
    lab[:, :, 1] = np.cos(angles) * scaled_chroma
    lab[:, :, 2] = np.sin(angles) * scaled_chroma

    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rgb[~mask] = 255

    wheel = Image.fromarray(rgb)
    draw = ImageDraw.Draw(wheel)
    bbox = [
        center - max_radius,
        center - max_radius,
        center + max_radius,
        center + max_radius,
    ]
    draw.ellipse(bbox, outline=(0, 0, 0), width=2)

    tick_length = max(8, size // 35)
    label_offset = max(28, size // 11)
    label_positions = {
        "0/360": (center + max_radius - label_offset, center),
        "90": (center, center - max_radius + label_offset),
        "180": (center - max_radius + label_offset, center),
        "270": (center, center + max_radius - label_offset),
    }
    tick_positions = {
        "0/360": ((center + max_radius - tick_length, center), (center + max_radius, center)),
        "90": ((center, center - max_radius + tick_length), (center, center - max_radius)),
        "180": ((center - max_radius + tick_length, center), (center - max_radius, center)),
        "270": ((center, center + max_radius - tick_length), (center, center + max_radius)),
    }
    for label, (start, end) in tick_positions.items():
        draw.line([start, end], fill=(0, 0, 0), width=2)
    for label, (x_pos, y_pos) in label_positions.items():
        bbox = draw.textbbox((0, 0), label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_xy = (x_pos - text_width / 2, y_pos - text_height / 2)
        pad = 2
        draw.rectangle(
            [
                text_xy[0] - pad,
                text_xy[1] - pad,
                text_xy[0] + text_width + pad,
                text_xy[1] + text_height + pad,
            ],
            fill=(255, 255, 255),
        )
        draw.text(text_xy, label, fill=(0, 0, 0))
    return wheel


def colorize_image_to_cielab_hue(img, hue_angle_degrees, chroma=55):
    """Colorize object pixels to an absolute CIELAB hue angle while preserving lightness."""
    import cv2

    img_float = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

    theta = np.radians(hue_angle_degrees)
    target_a = np.cos(theta) * chroma
    target_b = np.sin(theta) * chroma

    # Brady object stimuli are on light backgrounds; preserve those backgrounds.
    object_mask = np.any(img < 245, axis=2)
    lab[:, :, 1][object_mask] = target_a
    lab[:, :, 2][object_mask] = target_b

    recolored = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(recolored * 255, 0, 255).astype(np.uint8)


class ColorMemoryTask:
    """Color memory on Brady et al. (2013) objects.

    All stimuli come from a fixed pre-rendered palette (12 hues, 30-degree steps; see
    src/color_palette.py), so local runs and the Hugging Face episodes use identical images.

    mode: 'continuous_color_report' — report the studied hue as an angle on a color wheel
          'named'                   — report one of 6 named colors
          'afc'                     — 2AFC: studied color vs. the same object rotated 180° in hue
                                      (Brady, Konkle, Alvarez & Oliva 2013, JEP:G)
    In continuous and named modes the test probe is shown in grayscale; in afc mode both
    options are shown in color.
    """

    CONTINUOUS_PROMPT = (
        "What was the color of this item in the study sequence? "
        "Report the hue angle in degrees on the color wheel. "
        "0/360 degrees is the rightmost point, 90 is the top, "
        "180 is the left, and 270 is the bottom. Use any number from 0 up to 360."
    )
    AFC_PROMPT = (
        "Both images show the same item in different colors. "
        "Which one (1 or 2) shows the color the item had in the study sequence?"
    )

    def __init__(self, n_images=10, n_colors=None, mode="continuous_color_report",
                 source="local", repo_id="chrisiyer/vision-memory-tasks", root_dir="memory_datasets"):
        if mode not in {"continuous_color_report", "named", "afc"}:
            raise ValueError(f"Unknown color memory mode {mode!r}.")
        self.n_images = n_images
        self.n_colors = n_colors
        self.mode = mode
        self.dataset = ColorPaletteDataset(root_dir=root_dir, source=source, repo_id=repo_id)
        if mode == "continuous_color_report":
            self.color_wheel = self.dataset.get_color_wheel()

    def get_trials(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected = indices[:n]

        if self.mode == "named":
            named = [NAMED_COLORS[i % len(NAMED_COLORS)] for i in range(n)]
            random.shuffle(named)
            names = [name for name, _ in named]
            hues = [hue for _, hue in named]
        else:
            names = None
            hues = sample_palette_hues(random, n)

        study_sequence = [self.dataset.get_image(idx, hue) for idx, hue in zip(selected, hues)]

        test_order = list(range(n))
        random.shuffle(test_order)
        if self.mode == "named":
            test_phase = [self._named_trial(selected[i], hues[i], names[i]) for i in test_order]
        elif self.mode == "afc":
            test_phase = [self._afc_trial(selected[i], hues[i], study_sequence[i]) for i in test_order]
        else:
            test_phase = [self._continuous_trial(selected[i], hues[i]) for i in test_order]

        result = {
            "study_prompt": "Remember the colors of these items.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }
        if self.mode == "continuous_color_report":
            result["color_wheel"] = self.color_wheel
        return result

    def _base_metadata(self, idx, hue):
        return {
            **self.dataset.get_metadata(idx, hue),
            "target_angle_degrees": hue,
            "rotation_angle_degrees": hue,
            "color_transform": "brady_cielab_hue_rotation",
        }

    def _continuous_trial(self, idx, hue):
        return {
            "image": self.dataset.get_gray(idx),
            "color_wheel": self.color_wheel,
            "prompt": self.CONTINUOUS_PROMPT,
            "target": hue,
            "metadata": {
                **self._base_metadata(idx, hue),
                "response_unit": "degrees",
                "wheel_space": "CIELAB a/b hue angle",
                "angle_reference": {"0/360": "rightmost point", "90": "top", "180": "left", "270": "bottom"},
            },
        }

    def _named_trial(self, idx, hue, name):
        options_str = ", ".join(n for n, _ in NAMED_COLORS)
        return {
            "image": self.dataset.get_gray(idx),
            "prompt": f"What color was this item in the study sequence? Choose one: {options_str}.",
            "target": name,
            "metadata": {**self._base_metadata(idx, hue), "target_color_name": name},
        }

    def _afc_trial(self, idx, hue, studied_img):
        foil_hue = opposite_hue(hue)
        foil_img = self.dataset.get_image(idx, foil_hue)
        if random.random() < 0.5:
            images, target, option_hues = [studied_img, foil_img], 1, [hue, foil_hue]
        else:
            images, target, option_hues = [foil_img, studied_img], 2, [foil_hue, hue]
        return {
            "images": images,
            "prompt": self.AFC_PROMPT,
            "target": target,
            "metadata": {
                **self._base_metadata(idx, hue),
                "foil_angle_degrees": foil_hue,
                "option_angles_degrees": option_hues,
                "foil_offset_degrees": 180,
            },
        }


if __name__ == "__main__":
    task = ColorMemoryTask(n_images=5)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
