import random
import numpy as np
from stimuli import DirectoryDataset
from metrics import calculate_metrics

class ContinuousRecognitionSequence:
    def __init__(self, dataset, min_delay: int, max_delay: int, n_images: int = None):
        self.dataset = dataset
        self.min_delay = min_delay
        self.max_delay = max_delay
        available = len(dataset)
        self.n_images = min(n_images, available) if n_images else available

    def generate_sequence(self):
        """
        Generates a sequence of images for continuous-recognition.
        Each image appears exactly twice, once as 'new' (target=0), once as 'old' (target=1)
        Returns:
            List of (img_id, is_old, delay)
        """
        n = self.n_images
        new_image_pool = list(range(n))
        random.shuffle(new_image_pool)
        sequence = []
        waiting_room = [] # waiting_room stores (img_id, time_introduced)
        
        for t in range(2 * n):
            # Check availability for repeats
            eligible_to_repeat = [
                idx for idx, (img, t_first) in enumerate(waiting_room)
                if (t - t_first - 1) >= self.min_delay
            ]
            must_repeat = [
                idx for idx, (img, t_first) in enumerate(waiting_room)
                if (t - t_first - 1) >= self.max_delay
            ]

            # if any images are overdue, show one
            if must_repeat:
                img_id, t_first = waiting_room.pop(must_repeat[0])
                sequence.append((img_id, True, t - t_first - 1))
            
            # if no new images are left, show an old image
            elif not new_image_pool:
                img_id, t_first = waiting_room.pop( random.choice(eligible_to_repeat) )
                sequence.append((img_id, True, t - t_first - 1))

            # no images ready to be repeated, show a new image and add to waiting room
            elif not eligible_to_repeat:
                img_id = new_image_pool.pop()
                waiting_room.append((img_id, t))
                sequence.append((img_id, False, None))

            # could do either; show old with 50% probability 
            else:
                if random.random() > 0.5:
                    img_id, t_first = waiting_room.pop( random.choice(eligible_to_repeat) )
                    sequence.append((img_id, True, t - t_first - 1))
                else:
                    img_id = new_image_pool.pop()
                    waiting_room.append((img_id, t))
                    sequence.append((img_id, False, None))
                    
        return sequence


def run_task(dataset, min_delay, max_delay, n_images=None):
    sequencer = ContinuousRecognitionSequence(dataset, min_delay, max_delay, n_images)
    sequence = sequencer.generate_sequence()
    trials = []
    for img_id, is_old, delay in sequence:
        trials.append({
            "image": dataset.get_image(img_id),
            "metadata": dataset.get_metadata(img_id),
            "prompt":  "Has this image already appeared in the sequence (yes/no)?",
            "target":  1 if is_old else 0,
            "delay": delay
        })
    return trials
    

if __name__ == "__main__":
    dummy = DummyDataset(n=50)
    trials = run_task(dummy, min_delay=2, max_delay=15, n_images=50)
    
    print(f"Generated {len(trials)} trials.")
    
    # Verify constraints and print sample
    violations = 0
    for i, t in enumerate(trials):
        if t['target'] == 1:
            if not (2 <= t['delay'] <= 15):
                violations += 1
        
        if i < 20:
            id_str = f"ID={t['metadata']['id']}"
            delay_str = f"Delay={t['delay']}" if t['target'] else "New"
            print(f"Trial {i:02}: {id_str:<8} Target={t['target']} {delay_str}")
    
    print(f"\nConstraint Violations: {violations}")
    
    # Mock Results
    targets = [t['target'] for t in trials]
    responses = [1 if (t == 1 and random.random() > 0.1) or (t == 0 and random.random() > 0.9) else 0 for t in targets]
    metrics = calculate_metrics(responses, targets)
    print(f"Accuracy: {metrics['accuracy']:.3f}, D-Prime: {metrics['d_prime']:.3f}")

