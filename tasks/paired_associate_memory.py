import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import ThingsDataset, BradyDataset

class PairedAssociateMemoryTask:
    def __init__(self, dataset_name='things', n_images=20, wordpool_path='datasets/wasnorm_wordpool.txt'):
        self.n_images = n_images
        self.dataset_name = dataset_name
        
        # Load images
        if dataset_name == 'things':
            self.dataset = ThingsDataset(n_categories=n_images)
        else:
            self.dataset = BradyDataset(type='Objects')
            
        # Load words
        self.wordpool_path = Path(wordpool_path)
        if self.wordpool_path.exists():
            with open(self.wordpool_path, 'r') as f:
                self.wordpool = [line.strip() for line in f if line.strip()]
        else:
            print(f"Warning: Wordpool not found at {wordpool_path}, using dummy words.")
            self.wordpool = [f"WORD{i}" for i in range(1000)]

    def get_trials(self):
        n = min(self.n_images, len(self.dataset), len(self.wordpool))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        selected_indices = indices[:n]

        words = random.sample(self.wordpool, n)

        study_sequence = []
        pairs = []
        for i in range(n):
            img = self.dataset.get_image(selected_indices[i])
            word = words[i]
            study_sequence.append((img, word))
            pairs.append({
                "image": img,
                "word": word,
                "metadata": self.dataset.get_metadata(selected_indices[i])
            })

        test_indices = list(range(n))
        random.shuffle(test_indices)

        test_phase = []
        for i in test_indices:
            test_phase.append({
                "image": pairs[i]["image"],
                "prompt": "What was the word paired with this image? Respond with only the single word, nothing else.",
                "target": pairs[i]["word"],
                "metadata": pairs[i]["metadata"]
            })

        return {
            "study_prompt": "Remember the word paired with each image.",
            "study_sequence": study_sequence,
            "test_phase": test_phase
        }

if __name__ == "__main__":
    task = PairedAssociateMemoryTask(n_images=5)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
