import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stimuli import BradyDataset

sys.path.append(str(Path(__file__).parent.parent))
from memory_datasets.Brady2013ColorRotate import rotate_image_hue

# (name, hue_angle_degrees) — hue angles chosen as perceptual centers on the HSV wheel
NAMED_COLORS = [
    ("red", 0),
    ("orange", 30),
    ("yellow", 60),
    ("green", 120),
    ("blue", 240),
    ("purple", 300),
]


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
    def __init__(self, n_images=10, n_colors=None, mode="continuous_color_report"):
        """
        mode: 'continuous_color_report' — study/report as a continuous hue angle in degrees
              'named'                   — study/report as one of 6 named colors
        In both modes the test probe image is shown in grayscale.
        """
        self.n_images = n_images
        self.n_colors = n_colors
        self.mode = mode
        self.dataset = BradyDataset(type='Brady2013ColorObjects')
        if mode == "continuous_color_report":
            self.color_wheel = generate_cielab_color_wheel()

    def get_trials(self):
        if self.mode == "named":
            return self._get_named_trials()
        return self._get_continuous_color_report_trials()

    def _get_continuous_color_report_trials(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]

        target_angles = [random.random() * 360.0 for _ in range(n)]

        study_sequence = []
        for idx, angle in zip(selected_indices, target_angles):
            img = np.array(self.dataset.get_image(idx))
            recolored_img = Image.fromarray(colorize_image_to_cielab_hue(img, angle))
            study_sequence.append(recolored_img)

        test_phase = []
        test_indices = list(range(n))
        random.shuffle(test_indices)

        for i in test_indices:
            gray_probe = self.dataset.get_image(selected_indices[i]).convert("L").convert("RGB")
            test_phase.append({
                "image": gray_probe,
                "color_wheel": self.color_wheel,
                "prompt": (
                    f"What was the color of this item in the study sequence? "
                    "Report the hue angle in degrees on the color wheel. "
                    "0/360 degrees is the rightmost point, 90 is the top, "
                    "180 is the left, and 270 is the bottom. Use any number from 0 up to 360."
                ),
                "target": target_angles[i],
                "metadata": {
                    **self.dataset.get_metadata(selected_indices[i]),
                    "target_angle_degrees": target_angles[i],
                    "response_unit": "degrees",
                    "wheel_space": "CIELAB a/b hue angle",
                    "wheel_match": "Continuous-mode stimuli are recolored to this absolute CIELAB hue angle.",
                    "angle_reference": {
                        "0/360": "rightmost point",
                        "90": "top",
                        "180": "left",
                        "270": "bottom",
                    },
                }
            })

        return {
            "study_prompt": "Remember the colors of these items.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
            "color_wheel": self.color_wheel,
        }

    def _get_named_trials(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]

        target_colors = [random.choice(NAMED_COLORS) for _ in range(n)]

        study_sequence = []
        for idx, (color_name, hue_angle) in zip(selected_indices, target_colors):
            img = np.array(self.dataset.get_image(idx))
            rotated_img = Image.fromarray(rotate_image_hue(img, hue_angle))
            study_sequence.append(rotated_img)

        test_phase = []
        test_indices = list(range(n))
        random.shuffle(test_indices)

        options_str = ", ".join(name for name, _ in NAMED_COLORS)

        for i in test_indices:
            gray_probe = self.dataset.get_image(selected_indices[i]).convert("L").convert("RGB")
            test_phase.append({
                "image": gray_probe,
                "prompt": (
                    f"What color was this item in the study sequence? "
                    f"Choose one: {options_str}."
                ),
                "target": target_colors[i][0],
                "metadata": self.dataset.get_metadata(selected_indices[i])
            })

        return {
            "study_prompt": "Remember the colors of these items.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


if __name__ == "__main__":
    task = ColorMemoryTask(n_images=5)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
