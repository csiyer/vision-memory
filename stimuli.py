import os
import random
from PIL import Image
from pathlib import Path

class DirectoryDataset():
    def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir() 
            if p.suffix.lower() in extensions
        ])
        if not self.image_paths:
            print(f"Warning: No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_metadata(self, index):
        return {"path": str(self.image_paths[index]), "name": self.image_paths[index].name}


class LaMemDataset():
    """
    Skeleton for LaMem dataset. 
    In a real scenario, this would handle downloading and metadata parsing.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.metadata_file = self.root_dir / "lamem_score.txt" # Common filename for LaMem
        
        self.image_paths = []
        self.scores = {} # image_name -> score
        
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                for line in f:
                    # Format: folder/image_name score
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_rel_path, score = parts
                        img_path = self.image_dir / img_rel_path
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.scores[img_path.name] = float(score)

    def __len__(self):
        return len(self.image_paths)

    def get_image(self, index):
        return Image.open(self.image_paths[index]).convert("RGB")

    def get_metadata(self, index):
        path = self.image_paths[index]
        return {
            "path": str(path),
            "name": path.name,
            "memorability": self.scores.get(path.name)
        }

    def get_ground_truth(self):
        """Returns a dict of {image_name: score} for all images in the dataset."""
        return self.scores


# Helper to create a dummy dataset for testing
# (Removed as requested)
