import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import ThingsDataset, BradyDataset

# Same prompts for THINGS and Brady2008 so results are comparable.
AFC_STUDY_PROMPT = "Here is a sequence of images to remember."
AFC_TEST_PROMPT = (
    "Which of these two images was in the study sequence? "
    "The first image below is 1, the second is 2. "
    "Reply with only the digit 1 or 2 and nothing else."
)

class AFCRecognitionTask:
    def __init__(self, dataset_name='things', n_images=20, n_trials=None, foil_type='all'):
        self.n_images = n_images
        self.n_trials = n_trials if n_trials is not None else n_images
        self.foil_type = foil_type
        self.dataset_name = dataset_name

        if dataset_name == 'things':
            if foil_type == 'state':
                raise ValueError("State foils not supported for THINGS dataset.")
            # If we need exemplars, we need 2 per category
            # For novel foils, we need 2x categories (one for original, one for foil)
            exemplars = 2 if foil_type in ['exemplar', 'all'] else 1
            n_cats = n_images * 2 if foil_type == 'novel' else n_images
            self.dataset = ThingsDataset(n_categories=n_cats, exemplars_per_category=exemplars)
        else:
            self.dataset = BradyDataset(type='Objects')

    def _get_pairs(self, foil_type, n):
        pairs = []
        if self.dataset_name == 'things':
            # Handle THINGS
            if foil_type == 'novel' or foil_type == 'all':
                # For 'all', we split half/half novel/exemplar (since state isn't supported)
                n_novel = n if foil_type == 'novel' else n // 2
                n_exemplar = n - n_novel

                # Novel pairs need 2 distinct categories each; exemplar pairs reuse 1 category
                n_needed = n_novel * 2 + n_exemplar
                n_available = len(self.dataset)
                if n_available < n_needed:
                    raise ValueError(
                        f"THINGS dataset has only {n_available} categories but "
                        f"foil_type={foil_type!r} with n={n} requires {n_needed} "
                        f"({n_novel} novel pairs need {n_novel*2} categories, "
                        f"{n_exemplar} exemplar pairs need {n_exemplar} more). "
                        f"Reduce --n-images to at most "
                        f"{n_available // 2 if foil_type == 'novel' else (n_available * 2) // 3}."
                    )

                # Novel: random categories
                indices = list(range(n_available))
                random.shuffle(indices)

                # First n_novel pairs use two different categories
                for i in range(0, n_novel * 2, 2):
                    pairs.append({
                        "original": self.dataset.get_image(indices[i], 0),
                        "foil": self.dataset.get_image(indices[i+1], 0),
                        "type": "novel"
                    })

                # Remaining n_exemplar pairs use two exemplars of the same category
                start_idx = n_novel * 2
                for i in range(start_idx, start_idx + n_exemplar):
                    idx = indices[i]
                    pairs.append({
                        "original": self.dataset.get_image(idx, 0),
                        "foil": self.dataset.get_image(idx, 1),
                        "type": "exemplar"
                    })
            elif foil_type == 'exemplar':
                # All pairs are exemplars of the same category
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
                for i in range(min(n, len(indices))):
                    idx = indices[i]
                    pairs.append({
                        "original": self.dataset.get_image(idx, 0),
                        "foil": self.dataset.get_image(idx, 1),
                        "type": "exemplar"
                    })
            return pairs

        # Handle Brady (self.dataset is BradyDataset(type='Objects') for Brady2008)
        if foil_type == 'novel':
            obj_ds = self.dataset
            n_available = len(obj_ds)
            if n_available == 0:
                raise ValueError(
                    f"No Brady object images found under '{obj_ds.path}'. "
                    "Populate that folder with Brady et al. (2008) object images, or use --dataset things."
                )
            need = n * 2
            if n_available < need:
                max_study = n_available // 2
                raise ValueError(
                    f"Brady Objects at '{obj_ds.path}' has {n_available} images; "
                    f"novel foils need 2 distinct images per study item ({need} for n_images={n}). "
                    f"Add more images or set --n-images to at most {max_study}."
                )
            indices = list(range(n_available))
            random.shuffle(indices)
            for i in range(0, n * 2, 2):
                pairs.append({
                    "original": obj_ds.get_image(indices[i]),
                    "foil": obj_ds.get_image(indices[i + 1]),
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

        # Limit test trials if n_trials < n_images
        test_pairs = pairs[:self.n_trials]

        test_phase = []
        for p in test_pairs:
            # Randomly swap order for 2-AFC
            images = [p['original'], p['foil']]
            random.shuffle(images)
            target = 1 if images[0] == p['original'] else 2

            test_phase.append({
                "images": images,
                "prompt": AFC_TEST_PROMPT,
                "target": target,
                "type": p['type']
            })

        if self.n_trials < len(test_phase):
            test_phase = random.sample(test_phase, self.n_trials)

        return {
            "study_prompt": AFC_STUDY_PROMPT,
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
