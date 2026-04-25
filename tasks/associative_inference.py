import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stimuli import BradyDataset, ThingsDataset


class AssociativeInferenceTask:
    def __init__(self, dataset_name="things", n_trials=20):
        if n_trials % 2 != 0:
            raise ValueError("n_trials must be even so the study phase can split into AB and BC pairs.")

        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.n_pairs = n_trials // 2
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        n_required_images = self.n_pairs * 3
        if self.dataset_name == "things":
            return ThingsDataset(n_categories=n_required_images)
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects")
        return ThingsDataset(n_categories=n_required_images)

    def get_trials(self):
        n_available = len(self.dataset)
        n_pairs = min(self.n_pairs, n_available // 3)
        if n_pairs < 2:
            raise ValueError("Associative inference requires at least 2 latent ABC chains.")

        indices = list(range(n_available))
        random.shuffle(indices)
        selected_indices = indices[: n_pairs * 3]

        a_indices = selected_indices[:n_pairs]
        b_indices = selected_indices[n_pairs : 2 * n_pairs]
        c_indices = selected_indices[2 * n_pairs : 3 * n_pairs]

        chain_items = []
        for chain_index in range(n_pairs):
            a_index = a_indices[chain_index]
            b_index = b_indices[chain_index]
            c_index = c_indices[chain_index]
            chain_items.append(
                {
                    "chain_index": chain_index,
                    "A": {
                        "image": self.dataset.get_image(a_index),
                        "metadata": {
                            **self.dataset.get_metadata(a_index),
                            "role": "A",
                            "chain_index": chain_index,
                        },
                    },
                    "B": {
                        "image": self.dataset.get_image(b_index),
                        "metadata": {
                            **self.dataset.get_metadata(b_index),
                            "role": "B",
                            "chain_index": chain_index,
                        },
                    },
                    "C": {
                        "image": self.dataset.get_image(c_index),
                        "metadata": {
                            **self.dataset.get_metadata(c_index),
                            "role": "C",
                            "chain_index": chain_index,
                        },
                    },
                }
            )

        study_sequence = []
        for chain in chain_items:
            study_sequence.append(
                {
                    "images": [chain["A"]["image"], chain["B"]["image"]],
                    "pair_type": "AB",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "A",
                        "right_role": "B",
                        "left_item": chain["A"]["metadata"],
                        "right_item": chain["B"]["metadata"],
                    },
                }
            )
        for chain in chain_items:
            study_sequence.append(
                {
                    "images": [chain["B"]["image"], chain["C"]["image"]],
                    "pair_type": "BC",
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "left_role": "B",
                        "right_role": "C",
                        "left_item": chain["B"]["metadata"],
                        "right_item": chain["C"]["metadata"],
                    },
                }
            )

        test_phase = []
        foil_candidates = list(range(n_pairs))
        random.shuffle(foil_candidates)

        for chain in chain_items:
            foil_options = [idx for idx in foil_candidates if idx != chain["chain_index"]]
            if not foil_options:
                foil_options = [idx for idx in range(n_pairs) if idx != chain["chain_index"]]
            foil_chain = chain_items[random.choice(foil_options)]

            options = [chain["C"], foil_chain["C"]]
            random.shuffle(options)
            target = 1 if options[0]["metadata"]["chain_index"] == chain["chain_index"] else 2

            test_phase.append(
                {
                    "cue_image": chain["A"]["image"],
                    "images": [options[0]["image"], options[1]["image"]],
                    "prompt": "Which of these two images is indirectly associated with the cue image? (1 or 2)",
                    "target": target,
                    "metadata": {
                        "chain_index": chain["chain_index"],
                        "bridge_item": chain["B"]["metadata"],
                        "cue_item": chain["A"]["metadata"],
                        "correct_option": chain["C"]["metadata"],
                        "foil_option": foil_chain["C"]["metadata"],
                    },
                }
            )

        return {
            "study_prompt": (
                "Remember these image pairs. First you will see all A-B pairs, then all B-C pairs. "
                "Later you will infer which C image goes with each A image."
            ),
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


if __name__ == "__main__":
    task = AssociativeInferenceTask(n_trials=6)
    results = task.get_trials()
    print(f"Study sequence length: {len(results['study_sequence'])}")
    print(f"Test trials: {len(results['test_phase'])}")
    print(f"First test target: {results['test_phase'][0]['target']}")
