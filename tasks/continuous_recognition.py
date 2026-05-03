import random
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import ThingsDataset, BradyDataset

class ContinuousRecognitionTask:
    def __init__(self, dataset_name='things', n_images=50, n_trials=None, min_delay=2, max_delay=15, p_old=0.5):
        self.n_images = n_images
        self.n_trials = n_trials  # If None, computed from n_images and p_old
        # For very small n, delay constraints can't be satisfied — drop them so
        # every image is immediately eligible to repeat after first presentation.
        if n_images < min_delay + 2:
            min_delay = 0
            max_delay = n_images
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.p_old = p_old

        if dataset_name == 'things':
            self.dataset = ThingsDataset(n_categories=n_images)
        elif dataset_name == 'Brady2008':
            self.dataset = BradyDataset(type='Objects')
        else:
            self.dataset = ThingsDataset(n_categories=n_images)

    def generate_sequence(self):
        """
        Generates a sequence of images satisfying delay and p_old constraints.
        """
        n_unique = min(self.n_images, len(self.dataset))
        if self.n_trials is not None:
            total_trials = self.n_trials
            n_old_needed = int(total_trials * self.p_old)
            # Adjust n_unique if needed (can't have more unique than total - old)
            n_unique = min(n_unique, total_trials - n_old_needed)
        else:
            total_trials = int(n_unique / (1 - self.p_old))
            n_old_needed = total_trials - n_unique

        sequence = []
        new_indices = list(range(n_unique))
        random.shuffle(new_indices)

        waiting_room = [] # (img_idx, introduction_time)

        current_time = 0
        while len(sequence) < total_trials:
            eligible_to_repeat = [
                (idx, img_idx, t_intro) for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.min_delay
            ]
            must_repeat = [
                (idx, img_idx, t_intro) for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.max_delay
            ]

            # Decide whether to show 'old' or 'new'
            show_old = False
            if must_repeat:
                show_old = True
            elif not new_indices:
                show_old = True
            elif eligible_to_repeat and n_old_needed > 0:
                # Weighted choice to maintain p_old
                if random.random() < self.p_old:
                    show_old = True

            if show_old and eligible_to_repeat:
                # Pick one from eligible
                if must_repeat:
                    # Pick the oldest must_repeat
                    idx_in_waiting, img_idx, t_intro = must_repeat[0]
                else:
                    idx_in_waiting, img_idx, t_intro = random.choice(eligible_to_repeat)

                waiting_room.pop(idx_in_waiting)
                sequence.append({
                    "image_idx": img_idx,
                    "target": 1,
                    "delay": current_time - t_intro - 1
                })
                n_old_needed -= 1
            elif new_indices:
                img_idx = new_indices.pop(0)
                waiting_room.append((img_idx, current_time))
                sequence.append({
                    "image_idx": img_idx,
                    "target": 0,
                    "delay": None
                })
            elif waiting_room:
                # All new images exhausted; wait until the oldest waiting image
                # becomes eligible (respects min_delay) before repeating.
                img_idx, t_intro = waiting_room[0]
                delay = current_time - t_intro - 1
                if delay >= self.min_delay:
                    waiting_room.pop(0)
                    sequence.append({
                        "image_idx": img_idx,
                        "target": 1,
                        "delay": delay
                    })
                    n_old_needed -= 1
                # else: not yet eligible — advance current_time and retry
            else:
                break  # Out of images

            current_time += 1

        return sequence

    def get_trials(self):
        sequence = self.generate_sequence()
        trials = []
        for s in sequence:
            trials.append({
                "image": self.dataset.get_image(s['image_idx']),
                "prompt": "Has this image already appeared in the sequence (yes/no)?",
                "target": s['target'],
                "metadata": {
                    **self.dataset.get_metadata(s['image_idx']),
                    "delay": s['delay']
                }
            })
        return trials

if __name__ == "__main__":
    task = ContinuousRecognitionTask(n_images=20, p_old=0.5)
    trials = task.get_trials()
    print(f"Generated {len(trials)} trials.")
    for i, t in enumerate(trials[:10]):
        print(f"Trial {i}: Target={t['target']}, Delay={t['metadata'].get('delay')}")
