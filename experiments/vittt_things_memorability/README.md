This folder contains setup files for a `ViT^3` memorability run on `THINGS`.

Current contents:

- `selected_local_object_images.csv`: auto-generated `100` local THINGS images, one per category, chosen from the first `500` categories and spread across image-level memorability.
- `run_vittt_things_memorability.py`: runs the frozen-state memorability assay and correlates model scores with THINGS memorability.

Selection logic:

- source table: `memory_datasets/THINGS/THINGS_Memorability_Scores.csv`
- score column: `cr`
- image source: `memory_datasets/THINGS/object_images`
- local subset: first `500` categories in sorted order, with the first image from each category used as the candidate image
- selection: `100` percentile-spanning exact images with image-level memorability scores

Memorability experiment:

- present all `100` selected local THINGS images in a random order and update memory
- freeze the final memory state
- probe the first `50` images again without updating memory
- use `-final raw gradient` as the memory score
- repeat random runs until each image has at least `100` probe scores
- correlate the average model score per image with that exact image's THINGS memorability score

Run:

```bash
python experiments/vittt_things_memorability/run_vittt_things_memorability.py --device mps
```

Outputs:

- `outputs/memorability_probe_trials.csv`
- `outputs/memorability_image_scores.csv`
- `outputs/memorability_summary.json`
- `outputs/vittt_things_memorability.png`
