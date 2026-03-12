import random
from stimuli import ThingsDataset, BradyDataset

class SourceMemoryTask:
    def __init__(self, dataset_name='things', n_images=20):
        self.n_images = n_images
        if dataset_name == 'things':
            self.dataset = ThingsDataset(n_images=n_images)
        elif dataset_name == 'Brady2008':
            self.dataset = BradyDataset(type='Objects')
        else:
            self.dataset = ThingsDataset(n_images=n_images)

    def get_trials(self):
        n = min(self.n_images, len(self.dataset))
        indices = list(range(n))
        random.shuffle(indices)
        
        study_sequence = [self.dataset.get_image(i) for i in indices]
        
        # Test phase: for each image, ask for its position (1 to n)
        test_phase = []
        # We test in a random order
        test_indices = list(range(n))
        random.shuffle(test_indices)
        
        for i in test_indices:
            test_phase.append({
                "image": study_sequence[i],
                "prompt": f"When in the sequence did this image occur (1-{n})?",
                "target": i + 1,
                "metadata": self.dataset.get_metadata(indices[i])
            })
            
        return {
            "study_prompt": f"Here is a sequence of {n} images to remember.",
            "study_sequence": study_sequence,
            "test_phase": test_phase
        }

if __name__ == "__main__":
    task = SourceMemoryTask(n_images=10)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"First test trial target: {results['test_phase'][0]['target']}")
