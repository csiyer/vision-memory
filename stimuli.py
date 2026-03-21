import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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


class ThingsDataset:
    """Gets images from the THINGS dataset via Hugging Face."""
    def __init__(self, n_categories=None, exemplars_per_category=1):
        try:
            from datasets import load_dataset
            # Prioritize the one the user mentioned
            names = ["Haitao999/things-eeg", "He-Bart/THINGS", "Hebart/THINGS", "THINGS-data/THINGS", "RAIlab/THINGS"]
            self.ds = None
            self.name = None
            for name in names:
                try:
                    # Use streaming=True to avoid downloading 30GB at once
                    self.ds = load_dataset(name, split="train", streaming=True)
                    self.name = name
                    print(f"Successfully connected to THINGS via {name}")
                    break
                except:
                    continue
            
            if self.ds is None:
                raise ValueError("Could not find THINGS dataset on Hugging Face. Please verify the dataset name.")
            
            # Map category name to list of images
            self.category_groups = {} 
            self.category_names = [] # Maintain order of arrival
            
            # For ImageFolder style datasets, we need the names from features
            meta_ds = load_dataset(self.name, split="train") # Small download for metadata
            if hasattr(meta_ds, 'features') and 'label' in meta_ds.features:
                id_to_name = meta_ds.features['label'].names
            else:
                id_to_name = None

            print("Fetching images from streaming dataset...")
            for item in self.ds:
                # Determine category name
                label = item.get('label')
                if label is not None and id_to_name:
                    cat = id_to_name[label]
                else:
                    cat = item.get('category') or item.get('concept') or f"cat_{label}"
                
                if cat not in self.category_groups:
                    if n_categories and len(self.category_groups) >= n_categories:
                        continue
                    self.category_groups[cat] = []
                    self.category_names.append(cat)
                
                if len(self.category_groups[cat]) < exemplars_per_category:
                    img = item['image']
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    self.category_groups[cat].append(img.convert("RGB"))
                
                # Check if we have enough
                if n_categories and len(self.category_groups) >= n_categories:
                    # Check if all have enough exemplars
                    if all(len(self.category_groups[c]) >= exemplars_per_category for c in self.category_names):
                        break
            
            print(f"Loaded {len(self.category_groups)} categories with up to {exemplars_per_category} exemplars each.")
            
        except Exception as e:
            print(f"Error loading THINGS dataset: {e}")
            self.category_groups = {}
            self.category_names = []

    def __len__(self):
        return len(self.category_names)

    def get_image(self, index, exemplar_index=0):
        cat = self.category_names[index]
        exemplars = self.category_groups[cat]
        return exemplars[exemplar_index % len(exemplars)]

    def get_metadata(self, index, exemplar_index=0):
        cat = self.category_names[index]
        return {"category": cat, "category_id": index, "exemplar_id": exemplar_index}


class BradyDataset:
    """Handles Brady2008 and Brady2013 datasets."""
    def __init__(self, type='Objects', root_dir='memory_datasets'):
        self.root = Path(root_dir)
        if 'Brady' in type:
             self.path = self.root / type
        else:
             # Default to Brady2008 prefix if just 'Objects', 'Exemplar', 'State'
             self.path = self.root / f"Brady2008{type}"
        
        self.image_paths = sorted([p for p in self.path.glob('*') if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])

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


def generate_color_palette(n_colors=36):
    """Generates a visual palette with numbered colors."""
    import colorsys
    
    tile_size = 100
    cols = 6
    rows = (n_colors + cols - 1) // cols
    palette = Image.new("RGB", (cols * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(palette)
    
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    hues = np.linspace(0, 1.0, n_colors, endpoint=False)
    
    for i, hue in enumerate(hues):
        r_idx = i // cols
        c_idx = i % cols
        
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = tuple(int(x * 255) for x in rgb)
        
        x0, y0 = c_idx * tile_size, r_idx * tile_size
        x1, y1 = x0 + tile_size, y0 + tile_size
        draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
        draw.text((x0 + 5, y0 + 5), str(i + 1), fill=(0, 0, 0), font=font)
        
    return palette


# Helper to create a dummy dataset for testing
# (Removed as requested)
