import random
import numpy as np
from PIL import Image
from stimuli import BradyDataset, generate_color_palette
import sys
from pathlib import Path

# Add parent directory to path to import rotate_image_hue
sys.path.append(str(Path(__file__).parent.parent))
from memory_datasets.Brady2013ColorRotate import rotate_image_hue

class ColorMemoryTask:
    def __init__(self, n_images=10, n_colors=36):
        self.n_images = n_images
        self.n_colors = n_colors
        self.dataset = BradyDataset(type='Brady2013Color')
        self.palette = generate_color_palette(n_colors=n_colors)

    def get_trials(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]
        
        # Assign random color to each image
        # Colors are 1-based indices into the palette (1 to n_colors)
        target_colors = [random.randint(1, self.n_colors) for _ in range(n)]
        
        study_sequence = []
        for idx, col_idx in zip(selected_indices, target_colors):
            img = np.array(self.dataset.get_image(idx))
            # Angle for rotation: based on 360 degrees divided by n_colors
            angle = (col_idx - 1) * (360.0 / self.n_colors)
            rotated_img = Image.fromarray(rotate_image_hue(img, angle))
            study_sequence.append(rotated_img)
            
        # Test phase: show grayscale or just the object and ask for color number
        test_phase = []
        test_indices = list(range(n))
        random.shuffle(test_indices)
        
        for i in test_indices:
            # We show the original (usually canonical color) or maybe just the item
            # The user said "report the color of the item". Usually shown the item 
            # and they have to remember which color it was.
            test_phase.append({
                "image": self.dataset.get_image(selected_indices[i]),
                "palette": self.palette,
                "prompt": f"What was the color of this item in the sequence? Report the number (1-{self.n_colors}) from the palette.",
                "target": target_colors[i],
                "metadata": self.dataset.get_metadata(selected_indices[i])
            })
            
        return {
            "study_prompt": "Remember the colors of these items.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
            "palette": self.palette
        }

if __name__ == "__main__":
    task = ColorMemoryTask(n_images=5)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
