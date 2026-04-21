# Vision Memory Tasks

This repository implements several classic psychological tasks that measure episodic memory. They progress in difficulty, and vary in the degree they rely strictly on hippocampal mechanisms. Note that many of these tasks are frequently implemented with word stimuli in the literature, but here we use images to evaluate memory when learning representations from raw sensory data (i.e., pixels). 

NOTE: make sure to unzip `memory_datasets.zip` into `memory_datasets/` before running tasks that rely on the Brady stimulus sets, local THINGS assets, or other bundled dataset files. Continuous recognition can still stream THINGS from HuggingFace when local files are unavailable.

### 1. Recognition Memory

#### 1a. Continuous Recognition
Show a sequence of images, for each image respond if the image has already appeared in the sequence or not. 

By default, images are streamed from the [THINGS Dataset](https://things-initiative.org) via [HuggingFace](https://huggingface.co/datasets/Haitao999/things-eeg).

Each image is accompanied by an estimate of image memorability, from [Kramer et al. (2023)](https://www.science.org/doi/full/10.1126/sciadv.add2981). Memorability is computed as the proportion of correct identifications of the image minus the proportion of false alarms on that image, across their pool of 13,946 subjects performing a continuous recognition task (each subject saw 187 images).

Note that the continuous recognition task was also used in the [LaMem dataset](http://memorability.csail.mit.edu) and the [Natural Scenes Dataset](https://naturalscenesdataset.org)

Usage:
```python
from tasks.recognition import ContinuousRecognitionTask
task = ContinuousRecognitionTask(dataset_name='things', n_images=50, min_delay=2, max_delay=15, p_old=0.5)
# trials: list of {image, prompt, target, metadata}
```

#### 1b. 2-AFC Recognition ([Brady et al. 2008](https://www.pnas.org/doi/10.1073/pnas.0803390105))
Show a sequence of images, then show 1 old image and 1 new images and ask which is old

3 conditions of "new" images (foils): novel (unrelated image), exemplar (different instance of same category), and state (same instance, different state). The "novel" and "exemplar" conditions can be implemented with the THINGS images as well, although the "state" condition cannot.

Usage:
```python
from tasks.recognition import AFCRecognitionTask
task = AFCRecognitionTask(dataset_name='Brady2008', n_images=20, foil_type='all')
results = task.get_trials()
# results: {study_sequence, test_phase[{images, prompt, target, type}]}
```


### 2. Serial Order Memory
Study a sequence, then test memory for the order in which items appeared. Similarly to above, there are two versions of the task:

- Serial Order: report the position (1 to N) of a studied image.
- AFC Serial Order: show two studied images and report which appeared first.

Usage:
```python
from tasks.serial_order_memory import SerialOrderMemoryTask, AFCSerialOrderMemoryTask

task = SerialOrderMemoryTask(dataset_name='things', n_images=20)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target}]}

afc_task = AFCSerialOrderMemoryTask(dataset_name='things', n_images=20)
afc_results = afc_task.get_trials()
# results: {study_sequence, test_phase[{images, prompt, target, metadata.distance}]}
```


### 3. Color Memory ([Brady et al. 2013](https://konklab.fas.harvard.edu/Papers/Brady_2013_PsychSci.pdf))
Show a sequence of colored objects, then show one object in grayscale and ask for a continuous color report.

Usage:
```python
from tasks.color_memory import ColorMemoryTask
task = ColorMemoryTask(n_images=10, n_colors=36)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, palette, prompt, target}]}
```

### 4. Paired Associate Memory
Show a sequence of images, each paired to a word. Then, show an image and ask for the word it was paired with.
By default, images are drawn from the [THINGS dataset](https://things-initiative.org) and words from the normed wordpool of the [PEERS dataset](https://memory.psych.upenn.edu/PEERS)

Usage:
```python
from tasks.paired_associate_memory import PairedAssociateMemoryTask
task = PairedAssociateMemoryTask(dataset_name='things', n_images=20)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target}]}
```

### 5. Associative Inference
Study latent `A-B-C` chains across two blocks of image pairs. First, all `A-B` pairs are shown. Second, all `B-C` pairs are shown. At test, the model sees an `A` image and must choose which of two `C` images is indirectly associated with it.

Usage:
```python
from tasks.associative_inference import AssociativeInferenceTask

task = AssociativeInferenceTask(dataset_name='things', n_trials=20)
results = task.get_trials()
# results: {study_sequence[{images, pair_type}], test_phase[{cue_image, images, prompt, target}]}
```


## Metrics and Plotting
Standardized metrics including d-prime (z(hit_rate) - z(false_alarm_rate)), weighted F1, serial order error, AFC serial order accuracy by distance, associative inference accuracy, and hit rate by delay are calculated in `metrics.py`. Plotting tools in `plotting.py` include overlays of human performance data from Brady et al. (2008, 2013) stored in `memory_datasets/target_data.json`.


NOTE: these tasks are designed to be evaluated alongside the [Visual Haystacks](https://visual-haystacks.github.io) benchmark. The "single-needle" challenge involves showing a sequence of images, and then asking "for the image with the anchor object, is there a target object?" (e.g., "for the image with the truck, is there a dog?"). The "multiple-needles" asks "For all images with the anchor object, do all \[any\] of them contain the target object?" They evaluate several long-context vision-language models and find that performance drops from perfect to 50-60% accuracy by the time the haystack is 100 images.
