# Vision Memory Tasks

**Data layout:** Large assets live under `dataset/`: COCO + Visual Haystacks QA (`dataset/coco`, `dataset/VHs_qa`), and Brady / local stimulus files (`dataset/memory_datasets`). `dataset/coco` and `dataset/VHs_qa` can stay empty on disk: re-fetch VHs JSON with `hf download … tsunghanwu/visual_haystacks`, then either `python -m eval_scripts.prefetch_vhs_coco` or `--fetch-missing-coco` on eval. For backwards compatibility, the repo root has a symlink `memory_datasets` → `dataset/memory_datasets`, so existing paths and imports keep working. Brady images are not on the COCO CDN—keep `dataset/memory_datasets` or your `memory_datasets.zip` if you need those tasks; THINGS can stream from Hugging Face without them.

### 1. Continuous Recognition
Show a sequence of images, for each image respond if the image has already appeared in the sequence or not.

By default, images are streamed from the [THINGS Dataset](https://things-initiative.org) via [HuggingFace](https://huggingface.co/datasets/Haitao999/things-eeg).

Each image is accompanied by an estimate of image memorability, from [Kramer et al. (2023)](https://www.science.org/doi/full/10.1126/sciadv.add2981). Memorability is computed as the proportion of correct identifications of the image minus the proportion of false alarms on that image, across their pool of 13,946 subjects performing a continuous recognition task (each subject saw 187 images).

Note that the continuous recognition task was also used in the [LaMem dataset](http://memorability.csail.mit.edu) and the [Natural Scenes Dataset](https://naturalscenesdataset.org)

Usage:
```python
from tasks.continuous_recognition import ContinuousRecognitionTask
task = ContinuousRecognitionTask(dataset_name='things', n_images=50, min_delay=2, max_delay=15, p_old=0.5)
# trials: list of {image, prompt, target, metadata}
```

### 2. 2-AFC Recognition ([Brady et al. 2008](https://www.pnas.org/doi/10.1073/pnas.0803390105))
Show a sequence of images, then show 1 old image and 1 new images and ask which is old

3 conditions of "new" images (foils): novel (unrelated image), exemplar (different instance of same category), and state (same instance, different state). The "novel" and "exemplar" conditions can be implemented with the THINGS images as well, although the "state" condition cannot.

Usage:
```python
from tasks.afc_recognition import AFCRecognitionTask
task = AFCRecognitionTask(dataset_name='Brady2008', n_images=20, foil_type='all')
results = task.get_trials()
# results: {study_sequence, test_phase[{images, prompt, target, type}]}
```


### 3. Source Memory
Study a sequence, then report the position (1 to N) of each image.

Usage:
```python
from tasks.source_memory import SourceMemoryTask
task = SourceMemoryTask(dataset_name='things', n_images=20)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target}]}
```


### 4. Color Memory ([Brady et al. 2013](https://konklab.fas.harvard.edu/Papers/Brady_2013_PsychSci.pdf))
Show a sequence of colored objects, then show one object in grayscale and ask for a continuous color report.

Usage:
```python
from tasks.color_memory import ColorMemoryTask
task = ColorMemoryTask(n_images=10, n_colors=36)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, palette, prompt, target}]}
```

### 5. Paired Associate Memory
Show a sequence of images, each paired to a word. Then, show an image and ask for the word it was paired with.
By default, images are drawn from the [THINGS dataset](https://things-initiative.org) and words from the normed wordpool of the [PEERS dataset](https://memory.psych.upenn.edu/PEERS)

Usage:
```python
from tasks.paired_associate_memory import PairedAssociateMemoryTask
task = PairedAssociateMemoryTask(dataset_name='things', n_images=20)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target}]}
```


## Metrics and Plotting
Standardized metrics including d-prime (z(hit_rate) - z(false_alarm_rate)), weighted F1, and hit rate by delay are calculated in `metrics.py`. Plotting tools in `plotting.py` include overlays of human performance data from Brady et al. (2008, 2013) stored in `memory_datasets/target_data.json`.


### 6. Visual Haystacks ([Wu et al. 2025](https://visual-haystacks.github.io/))
Vision-centric needle-in-a-haystack benchmark on COCO image sets with binary yes/no questions.

Preparation:
1. Download VHs QA files:
```bash
hf download --repo-type dataset tsunghanwu/visual_haystacks --local-dir dataset/VHs_qa
```
2. Download **COCO 2017** images and annotations. VHs references **train**, **val**, and **test** images, so you need all three splits (train is large, ~18 GB zipped):
```text
dataset/coco/{train2017,val2017,test2017,annotations}
```
Example (from repo root, into `dataset/coco`):
```bash
curl -L -o dataset/coco/train2017.zip   http://images.cocodataset.org/zips/train2017.zip
curl -L -o dataset/coco/val2017.zip     http://images.cocodataset.org/zips/val2017.zip
curl -L -o dataset/coco/test2017.zip    http://images.cocodataset.org/zips/test2017.zip
curl -L -o dataset/coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# then unzip each archive in dataset/coco/
```

**Low storage:** You can skip downloading full COCO zips.

- **During eval:** pass `--fetch-missing-coco` so each referenced image is downloaded once from `https://images.cocodataset.org/...` into `dataset/coco/` (only images touched by that run). Combine with `--max-samples` for small pilots. Requires network while eval runs.

- **Prefetch everything VHs needs:** walk all `visual_haystack_*.json` files and download every unique COCO path (still no zip; disk use can grow large because many trials reference many images):

```bash
python -m eval_scripts.prefetch_vhs_coco --qa-root dataset/VHs_qa --image-root dataset/coco
# see scale without downloading:
python -m eval_scripts.prefetch_vhs_coco --dry-run
```

Usage:
```bash
# Single-needle (VHs_large, 10 images)
python -m eval_scripts.eval_vhs --models gpt-4o gemini --mode single_needle --split VHs_large --image-count 10

# Same, but fetch COCO images on demand (no full train2017 zip)
python -m eval_scripts.eval_vhs --fetch-missing-coco --models gemini --mode single_needle --split VHs_large --image-count 5 --max-samples 3

# Multi-needle (10 images), run only first 100 samples
python -m eval_scripts.eval_vhs --models gpt-4o --mode multi_needle --image-count 10 --max-samples 100
```

The script expects VHs files named like `visual_haystack_{count}.json` and computes accuracy, valid accuracy (excluding parse failures), and yes/no compliance.

NOTE: Tasks not yet implemented: paired associate inference, graph learning, navigation & spatial memory

