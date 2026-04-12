"""
Visual Haystacks Task

A vision-centric Needle-In-A-Haystack benchmark for evaluating VLM long-context
visual retrieval and reasoning capabilities.

Based on: https://visual-haystacks.github.io/
Dataset: https://huggingface.co/datasets/tsunghanwu/visual_haystacks

Task types:
- Single-needle: "For the image with [anchor], is there [target]?"
- Multi-needle (all): "For all images with [anchor], do all of them contain [target]?"
- Multi-needle (any): "For all images with [anchor], do any of them contain [target]?"
"""
import random
from pathlib import Path
from PIL import Image


class VisualHaystacksTask:
    """Visual Haystacks task using HuggingFace dataset or local COCO images."""

    def __init__(
        self,
        n_images: int = 10,
        task_type: str = "single",  # "single", "multi_all", "multi_any"
        use_hf_dataset: bool = True,
        coco_root: str = None,
        n_trials: int = None,
    ):
        """
        Args:
            n_images: Total number of images in the haystack (1 to 10000)
            task_type: Type of question ("single", "multi_all", "multi_any")
            use_hf_dataset: If True, load from HuggingFace. If False, use local COCO.
            coco_root: Path to local COCO dataset (required if use_hf_dataset=False)
            n_trials: Number of trials to run (defaults to available in dataset)
        """
        self.n_images = n_images
        self.task_type = task_type
        self.n_trials = n_trials
        self.use_hf_dataset = use_hf_dataset
        self.coco_root = Path(coco_root) if coco_root else None

        self.dataset = None
        self.filtered_data = []

        self._load_data()

    def _load_data(self):
        """Load and filter the Visual Haystacks dataset."""
        if self.use_hf_dataset:
            from datasets import load_dataset

            print("Loading Visual Haystacks from HuggingFace...")
            self.dataset = load_dataset("tsunghanwu/visual_haystacks", split="train")

            # Filter by task type and haystack size
            for item in self.dataset:
                # Determine task type from question
                question = item["conversations"][0]["value"].lower()

                if self.task_type == "single" and "for the image with" in question:
                    pass  # Single needle
                elif self.task_type == "multi_all" and "do all of them" in question:
                    pass  # Multi-needle all
                elif self.task_type == "multi_any" and "do any of them" in question:
                    pass  # Multi-needle any
                else:
                    continue

                # Check haystack size (neg_image count + pos_image count)
                total_images = len(item["pos_image"]) + len(item["neg_image"])

                # Accept if we can construct a haystack of requested size
                if total_images >= self.n_images:
                    self.filtered_data.append(item)

            print(f"Found {len(self.filtered_data)} trials matching criteria")

            if self.n_trials:
                self.filtered_data = self.filtered_data[: self.n_trials]
        else:
            raise NotImplementedError(
                "Local COCO loading not implemented. Use use_hf_dataset=True"
            )

    def _load_coco_image(self, image_path: str) -> Image.Image:
        """Load a COCO image from path reference."""
        if self.coco_root:
            full_path = self.coco_root / image_path
            return Image.open(full_path).convert("RGB")
        else:
            # Try to load from HuggingFace cached images
            # The dataset stores paths like "train2017/000000402000.jpg"
            # We need COCO images downloaded separately
            raise ValueError(
                "COCO images must be downloaded separately. "
                "Set coco_root to the path containing train2017/ and val2017/ folders."
            )

    def get_trials(self):
        """Generate trials for the Visual Haystacks task.

        Returns:
            List of trial dicts with:
                - images: List of PIL Images (haystack)
                - prompt: Question to ask
                - target: Expected answer ("yes" or "no")
                - metadata: Additional info (needle, target object, etc.)
        """
        trials = []

        for item in self.filtered_data:
            try:
                # Get positive and negative image paths
                pos_paths = item["pos_image"]
                neg_paths = item["neg_image"]

                # Construct haystack of requested size
                # Include all positive images, fill rest with negatives
                n_neg_needed = self.n_images - len(pos_paths)

                if n_neg_needed > len(neg_paths):
                    # Not enough images, skip this trial
                    continue

                selected_neg = random.sample(neg_paths, n_neg_needed)
                all_paths = pos_paths + selected_neg
                random.shuffle(all_paths)

                # Load images
                images = []
                for path in all_paths:
                    try:
                        img = self._load_coco_image(path)
                        images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not load {path}: {e}")
                        continue

                if len(images) < self.n_images:
                    continue

                # Get question and answer
                question = item["conversations"][0]["value"]
                answer = item["conversations"][1]["value"].lower().strip()

                trials.append(
                    {
                        "images": images,
                        "prompt": question + " Answer with only 'yes' or 'no'.",
                        "target": answer,
                        "metadata": {
                            "needle": item["needle"],
                            "target_object": item["target"],
                            "n_positive": len(pos_paths),
                            "n_negative": n_neg_needed,
                            "id": item["id"],
                        },
                    }
                )

            except Exception as e:
                print(f"Error processing trial: {e}")
                continue

        return trials


class VisualHaystacksTaskSimple:
    """Simplified Visual Haystacks using THINGS dataset (no COCO dependency).

    Creates single-needle style questions:
    "For the image containing [category], is there also [other category]?"

    Since THINGS images are single objects, the answer is always based on
    whether the target category appears in the haystack.
    """

    def __init__(
        self, n_images: int = 10, n_trials: int = 20, n_needles: int = 1
    ):
        """
        Args:
            n_images: Total images in haystack
            n_trials: Number of trials
            n_needles: Number of needle images (default 1)
        """
        self.n_images = n_images
        self.n_trials = n_trials
        self.n_needles = n_needles

        from stimuli import ThingsDataset

        # Need enough categories for haystack + needle + target variations
        n_cats = max(n_images * 2, 100)
        self.dataset = ThingsDataset(n_categories=n_cats, exemplars_per_category=1)

    def get_trials(self):
        """Generate Visual Haystacks trials using THINGS."""
        trials = []
        n_cats = len(self.dataset)

        for _ in range(self.n_trials):
            # Select categories
            indices = list(range(n_cats))
            random.shuffle(indices)

            # Needle category (the "anchor" - what we're looking for)
            needle_idx = indices[0]
            needle_cat = self.dataset.category_names[needle_idx]

            # Target category (what we ask about)
            target_idx = indices[1]
            target_cat = self.dataset.category_names[target_idx]

            # Decide if target should be present (50/50)
            target_present = random.random() < 0.5

            # Build haystack
            images = []

            # Add needle image(s)
            for i in range(self.n_needles):
                images.append(self.dataset.get_image(needle_idx, 0))

            # Add target if present
            start_idx = 2
            if target_present:
                images.append(self.dataset.get_image(target_idx, 0))
                start_idx = 3

            # Fill rest with distractors (not needle or target)
            n_distractors = self.n_images - len(images)
            distractor_indices = indices[start_idx : start_idx + n_distractors]

            for idx in distractor_indices:
                images.append(self.dataset.get_image(idx, 0))

            random.shuffle(images)

            # Create question
            question = f"For the image with a {needle_cat}, is there also a {target_cat}?"
            answer = "yes" if target_present else "no"

            trials.append(
                {
                    "images": images,
                    "prompt": question + " Answer with only 'yes' or 'no'.",
                    "target": answer,
                    "metadata": {
                        "needle": needle_cat,
                        "target": target_cat,
                        "target_present": target_present,
                        "n_images": len(images),
                    },
                }
            )

        return trials


if __name__ == "__main__":
    # Test with THINGS-based simple version
    print("Testing VisualHaystacksTaskSimple...")
    task = VisualHaystacksTaskSimple(n_images=5, n_trials=3)
    trials = task.get_trials()

    print(f"Generated {len(trials)} trials")
    for i, t in enumerate(trials):
        print(f"\nTrial {i}:")
        print(f"  Images: {len(t['images'])}")
        print(f"  Question: {t['prompt'][:60]}...")
        print(f"  Answer: {t['target']}")
        print(f"  Metadata: {t['metadata']}")
