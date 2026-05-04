# Vision Memory Tasks

This repository implements classic psychological tasks for evaluating visual episodic memory in models. Many of these paradigms are often run with words in the human literature; here, the emphasis is on image stimuli so models must learn from raw visual input.

Make sure to unzip the dataset archives from the repository root before running the tasks:

```bash
unzip memory_datasets_brady.zip
unzip memory_datasets_mst.zip
```

Both archives contain the top-level `memory_datasets/` directory, so unzipping both will merge the Brady, wordpool, color, and MST assets into the expected layout. The full THINGS image set is not bundled.

## 1. Recognition Memory

### 1a. Continuous Recognition

Show images one at a time. For each image, answer whether it has already appeared in the sequence.

```python
from tasks.recognition import ContinuousRecognitionTask

task = ContinuousRecognitionTask(dataset_name="Brady2008", n_images=50, min_delay=2, max_delay=15, p_old=0.5)
trials = task.get_trials()
# trials: [{image, prompt, target, metadata}]
# target: 1 = old / yes, 0 = new / no
```

### 1b. 2-AFC Recognition

Study a sequence of images, then choose which of two images was in the study sequence. Brady-style foil types are:

- `novel`: unrelated new image
- `exemplar`: different instance of the same category
- `state`: same object in a different state
- `all`: mix of the three foil types

```python
from tasks.recognition import AFCRecognitionTask

task = AFCRecognitionTask(dataset_name="Brady2008", n_images=20, foil_type="all")
results = task.get_trials()
# results: {study_sequence, test_phase[{images, prompt, target, type}]}
# target: 1 or 2
```

## 2. Mnemonic Similarity Task

Study target images, then classify test images as `old`, `similar`, or `new`. Lures are visually similar but non-identical images, making this a recognition task with an explicit "similar" response option.

Requires MST stimuli in `memory_datasets/MST/` with `Set 1` through `Set 6` folders and matching `Set1 bins.txt` through `Set6 bins.txt` files.

```python
from tasks.mnemonic_similarity import MnemonicSimilarityTask

task = MnemonicSimilarityTask(n_study=128, root="memory_datasets/MST")
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target, type, metadata}]}
# target: "old", "similar", or "new"
```

## 3. Serial Order Memory

Study a sequence, then report the order in which items appeared.

### 3a. Free Report

Report the serial position of a studied image.

```python
from tasks.serial_order_memory import SerialOrderMemoryTask

task = SerialOrderMemoryTask(dataset_name="Brady2008", n_images=20)
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target, metadata}]}
# target: serial position from 1 to N
```

### 3b. 2-AFC Serial Order

Choose which of two studied images appeared first.

```python
from tasks.serial_order_memory import AFCSerialOrderMemoryTask

task = AFCSerialOrderMemoryTask(dataset_name="Brady2008", n_images=20, n_tests=None)
results = task.get_trials()
# results: {study_sequence, test_phase[{images, prompt, target, metadata}]}
# target: 1 or 2
```

## 4. Color Memory

Study colored objects, then see a grayscale probe and report the remembered color.

### 4a. Continuous Color Report

Report a continuous hue angle on a CIELAB color wheel. The wheel convention is `0/360` = right, `90` = top, `180` = left, `270` = bottom.

```python
from tasks.color_memory import ColorMemoryTask

task = ColorMemoryTask(n_images=10, mode="continuous_color_report")
results = task.get_trials()
# results: {study_sequence, color_wheel, test_phase[{image, color_wheel, prompt, target, metadata}]}
# target: continuous hue angle in degrees
```

### 4b. Named Colors

Report one of six named colors: `red`, `orange`, `yellow`, `green`, `blue`, or `purple`.

```python
from tasks.color_memory import ColorMemoryTask

task = ColorMemoryTask(n_images=10, mode="named")
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target, metadata}]}
# target: color name
```

## 5. Paired Associate Memory

Study paired associates, then retrieve the item associated with a cue.

### 5a. Image-Word

Study image-word pairs, then report the word paired with a cue image.

```python
from tasks.paired_associate_memory import PairedAssociateMemoryTask

task = PairedAssociateMemoryTask(dataset_name="Brady2008", n_images=20, pair_type="word")
results = task.get_trials()
# results: {study_sequence, test_phase[{image, prompt, target, metadata}]}
# target: word
```

### 5b. Image-Image 2-AFC

Study image-image pairs, then choose which of two images was paired with a cue image.

```python
from tasks.paired_associate_memory import PairedAssociateMemoryTask

task = PairedAssociateMemoryTask(dataset_name="Brady2008", n_images=20, pair_type="image")
results = task.get_trials()
# results: {study_sequence, test_phase[{cue_image, images, prompt, target, metadata}]}
# target: 1 or 2
```

## 6. Associative Inference

Study latent `A-B-C` chains across `A-B` and `B-C` pairs. At test, infer which `C` item is indirectly associated with an `A` cue.

### 6a. Word 2-AFC

Study `A-B` image-image pairs and `B-C` image-word pairs, then choose which of two words is indirectly associated with the cue image.

```python
from tasks.associative_inference import AssociativeInferenceTask

task = AssociativeInferenceTask(dataset_name="Brady2008", n_trials=20, pair_type="word")
results = task.get_trials()
# results: {study_sequence, test_phase[{cue_image, options, prompt, target, metadata}]}
# target: 1 or 2
```

### 6b. Image 2-AFC

Study `A-B` and `B-C` image-image pairs, then choose which of two images is indirectly associated with the cue image.

```python
from tasks.associative_inference import AssociativeInferenceTask

task = AssociativeInferenceTask(dataset_name="Brady2008", n_trials=20, pair_type="image")
results = task.get_trials()
# results: {study_sequence, test_phase[{cue_image, images, prompt, target, metadata}]}
# target: 1 or 2
```

## Metrics and Plotting

Standardized metrics are implemented in `metrics.py`, including recognition accuracy/d-prime, hit rate by delay, serial order error, 2-AFC accuracy by distance, continuous color circular error, named-color accuracy, paired associate accuracy, MST LDI, and associative inference accuracy.

Plotting helpers in `plotting.py` create task-specific summaries and overlay available human benchmarks from files in `literature/`.

## Related Benchmarks

These tasks are designed to be evaluated alongside benchmarks such as [Visual Haystacks](https://visual-haystacks.github.io), where models search for target visual information across long image sequences.
