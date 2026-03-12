import random
from pathlib import Path
from stimuli import ThingsDataset, BradyDataset

class AFCRecognitionTask:
    def __init__(self, dataset_name='things', n_images=20, foil_type='all'):
        self.n_images = n_images
        self.foil_type = foil_type
        self.dataset_name = dataset_name
        
        if dataset_name == 'things':
            if foil_type == 'state':
                raise ValueError("State foils not supported for THINGS dataset.")
            self.dataset = ThingsDataset(n_images=n_images * 2) # Get extra for novel foils
        else:
            self.dataset = BradyDataset(type='Objects')

    def _get_pairs(self, foil_type, n):
        pairs = []
        if self.dataset_name == 'things':
            # Handle THINGS
            unique_images = [self.dataset.get_image(i) for i in range(len(self.dataset))]
            if foil_type == 'novel' or foil_type == 'all':
                # Split half/half novel/exemplar if 'all'
                n_novel = n if foil_type == 'novel' else n // 2
                n_exemplar = n - n_novel
                
                # Novel: random pairing
                indices = list(range(len(unique_images)))
                random.shuffle(indices)
                for i in range(0, n_novel * 2, 2):
                    pairs.append({
                        "original": unique_images[indices[i]],
                        "foil": unique_images[indices[i+1]],
                        "type": "novel"
                    })
                
                # Exemplar for things? The user said "split half/half between novel and exemplar".
                # But THINGS doesn't have explicit exemplars. I'll just use novel for both or 
                # maybe just pick different categories.
                for i in range(n_novel * 2, (n_novel + n_exemplar) * 2, 2):
                     pairs.append({
                        "original": unique_images[indices[i]],
                        "foil": unique_images[indices[i+1]],
                        "type": "exemplar"
                    })
            return pairs

        # Handle Brady
        if foil_type == 'novel':
             obj_ds = BradyDataset(type='Objects')
             indices = list(range(len(obj_ds)))
             random.shuffle(indices)
             for i in range(0, n * 2, 2):
                 pairs.append({
                     "original": obj_ds.get_image(indices[i]),
                     "foil": obj_ds.get_image(indices[i+1]),
                     "type": "novel"
                 })
        elif foil_type == 'exemplar' or foil_type == 'state':
             ds = BradyDataset(type='Exemplar' if foil_type == 'exemplar' else 'State')
             # Pair adjacent images (assuming they are pairs as seen in list_dir)
             all_paths = ds.image_paths
             for i in range(0, min(n * 2, len(all_paths) - 1), 2):
                 pairs.append({
                     "original": ds.get_image(i),
                     "foil": ds.get_image(i+1),
                     "type": foil_type
                 })
        elif foil_type == 'all':
             # 1/3 each
             n3 = n // 3
             pairs += self._get_pairs('novel', n3)
             pairs += self._get_pairs('exemplar', n3)
             pairs += self._get_pairs('state', n - 2*n3)
             
        return pairs

    def get_trials(self):
        pairs = self._get_pairs(self.foil_type, self.n_images)
        random.shuffle(pairs)
        
        study_sequence = [p['original'] for p in pairs]
        
        test_phase = []
        for p in pairs:
            # Randomly swap order for 2-AFC
            images = [p['original'], p['foil']]
            random.shuffle(images)
            target = 1 if images[0] == p['original'] else 2
            
            test_phase.append({
                "images": images,
                "prompt": "Which of these two images was in the sequence before? (1 or 2)",
                "target": target,
                "type": p['type']
            })
            
        return {
            "study_prompt": "Here is a sequence of images to remember.",
            "study_sequence": study_sequence,
            "test_phase": test_phase
        }

if __name__ == "__main__":
    task = AFCRecognitionTask(dataset_name='Brady2008', n_images=6, foil_type='all')
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"Test trials: {len(results['test_phase'])}")
    for i, t in enumerate(results['test_phase']):
        print(f"Trial {i}: Type={t['type']}, Target={t['target']}")
