import random

from stimuli import BradyDataset, ThingsDataset


class RecognitionTaskBase:
    def __init__(self, dataset_name="things", n_images=20):
        self.dataset_name = dataset_name
        self.n_images = n_images

    def _load_recognition_dataset(self, exemplars_per_category=1):
        if self.dataset_name == "things":
            return ThingsDataset(
                n_categories=self.n_images,
                exemplars_per_category=exemplars_per_category,
            )
        if self.dataset_name == "Brady2008":
            return BradyDataset(type="Objects")
        return ThingsDataset(
            n_categories=self.n_images,
            exemplars_per_category=exemplars_per_category,
        )


class ContinuousRecognitionTask(RecognitionTaskBase):
    def __init__(self, dataset_name="things", n_images=50, min_delay=2, max_delay=15, p_old=0.5):
        super().__init__(dataset_name=dataset_name, n_images=n_images)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.p_old = p_old
        self.dataset = self._load_recognition_dataset()

    def generate_sequence(self):
        """
        Generates a sequence of images satisfying delay and p_old constraints.
        """
        n_unique = min(self.n_images, len(self.dataset))
        total_trials = int(n_unique / (1 - self.p_old))
        n_old_needed = total_trials - n_unique

        sequence = []
        new_indices = list(range(n_unique))
        random.shuffle(new_indices)

        waiting_room = []  # (img_idx, introduction_time)

        current_time = 0
        while len(sequence) < total_trials:
            eligible_to_repeat = [
                (idx, img_idx, t_intro)
                for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.min_delay
            ]
            must_repeat = [
                (idx, img_idx, t_intro)
                for idx, (img_idx, t_intro) in enumerate(waiting_room)
                if (current_time - t_intro - 1) >= self.max_delay
            ]

            show_old = False
            if must_repeat:
                show_old = True
            elif not new_indices:
                show_old = True
            elif eligible_to_repeat and n_old_needed > 0 and random.random() < self.p_old:
                show_old = True

            if show_old and eligible_to_repeat:
                if must_repeat:
                    idx_in_waiting, img_idx, t_intro = must_repeat[0]
                else:
                    idx_in_waiting, img_idx, t_intro = random.choice(eligible_to_repeat)

                waiting_room.pop(idx_in_waiting)
                sequence.append(
                    {
                        "image_idx": img_idx,
                        "target": 1,
                        "delay": current_time - t_intro - 1,
                    }
                )
                n_old_needed -= 1
            elif new_indices:
                img_idx = new_indices.pop(0)
                waiting_room.append((img_idx, current_time))
                sequence.append(
                    {
                        "image_idx": img_idx,
                        "target": 0,
                        "delay": None,
                    }
                )
            else:
                if waiting_room:
                    idx_in_waiting, img_idx, t_intro = waiting_room.pop(0)
                    sequence.append(
                        {
                            "image_idx": img_idx,
                            "target": 1,
                            "delay": current_time - t_intro - 1,
                        }
                    )
                else:
                    break

            current_time += 1

        return sequence

    def get_trials(self):
        sequence = self.generate_sequence()
        trials = []
        for item in sequence:
            trials.append(
                {
                    "image": self.dataset.get_image(item["image_idx"]),
                    "prompt": "Has this image already appeared in the sequence (yes/no)?",
                    "target": item["target"],
                    "metadata": {
                        **self.dataset.get_metadata(item["image_idx"]),
                        "delay": item["delay"],
                    },
                }
            )
        return trials


class AFCRecognitionTask(RecognitionTaskBase):
    def __init__(self, dataset_name="things", n_images=20, foil_type="all"):
        super().__init__(dataset_name=dataset_name, n_images=n_images)
        self.foil_type = foil_type

        if dataset_name == "things":
            if foil_type == "state":
                raise ValueError("State foils not supported for THINGS dataset.")
            exemplars = 2 if foil_type in ["exemplar", "all"] else 1
            self.dataset = self._load_recognition_dataset(exemplars_per_category=exemplars)
        else:
            self.dataset = BradyDataset(type="Objects")

    def _get_pairs(self, foil_type, n):
        pairs = []
        if self.dataset_name == "things":
            if foil_type == "novel" or foil_type == "all":
                n_novel = n if foil_type == "novel" else n // 2
                n_exemplar = n - n_novel

                indices = list(range(len(self.dataset)))
                random.shuffle(indices)

                for i in range(0, n_novel * 2, 2):
                    pairs.append(
                        {
                            "original": self.dataset.get_image(indices[i], 0),
                            "foil": self.dataset.get_image(indices[i + 1], 0),
                            "type": "novel",
                        }
                    )

                start_idx = n_novel * 2
                for i in range(start_idx, start_idx + n_exemplar):
                    if i < len(indices):
                        idx = indices[i]
                        pairs.append(
                            {
                                "original": self.dataset.get_image(idx, 0),
                                "foil": self.dataset.get_image(idx, 1),
                                "type": "exemplar",
                            }
                        )
            elif foil_type == "exemplar":
                indices = list(range(len(self.dataset)))
                random.shuffle(indices)
                for i in range(min(n, len(indices))):
                    idx = indices[i]
                    pairs.append(
                        {
                            "original": self.dataset.get_image(idx, 0),
                            "foil": self.dataset.get_image(idx, 1),
                            "type": "exemplar",
                        }
                    )
            return pairs

        if foil_type == "novel":
            obj_ds = BradyDataset(type="Objects")
            indices = list(range(len(obj_ds)))
            random.shuffle(indices)
            for i in range(0, n * 2, 2):
                pairs.append(
                    {
                        "original": obj_ds.get_image(indices[i]),
                        "foil": obj_ds.get_image(indices[i + 1]),
                        "type": "novel",
                    }
                )
        elif foil_type == "exemplar" or foil_type == "state":
            ds = BradyDataset(type="Exemplar" if foil_type == "exemplar" else "State")
            for i in range(min(n, len(ds.pair_paths))):
                original, foil = ds.get_pair(i)
                pairs.append(
                    {
                        "original": original,
                        "foil": foil,
                        "type": foil_type,
                    }
                )
        elif foil_type == "all":
            n3 = n // 3
            pairs += self._get_pairs("novel", n3)
            pairs += self._get_pairs("exemplar", n3)
            pairs += self._get_pairs("state", n - 2 * n3)

        return pairs

    def get_trials(self):
        pairs = self._get_pairs(self.foil_type, self.n_images)
        random.shuffle(pairs)

        study_sequence = [pair["original"] for pair in pairs]

        test_phase = []
        for pair in pairs:
            images = [pair["original"], pair["foil"]]
            random.shuffle(images)
            target = 1 if images[0] == pair["original"] else 2

            test_phase.append(
                {
                    "images": images,
                    "prompt": "Which of these two images was in the sequence before? (1 or 2)",
                    "target": target,
                    "type": pair["type"],
                }
            )

        return {
            "study_prompt": "Here is a sequence of images to remember.",
            "study_sequence": study_sequence,
            "test_phase": test_phase,
        }


__all__ = ["RecognitionTaskBase", "ContinuousRecognitionTask", "AFCRecognitionTask"]
