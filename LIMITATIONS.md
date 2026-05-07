# Experimental Limitations

## Dataset Limits

| Condition | Limit | Reason |
|-----------|-------|--------|
| Brady2008 \| exemplar \| N>100 | SKIP | Only ~200 images in dataset |
| Brady2008 \| state \| N>100 | SKIP | Only ~200 images in dataset |
| Continuous recognition \| N<4 | Runs with min_delay=0 | Too few images to satisfy min_delay=2 gap between study and test; delay constraint is dropped so every image is immediately eligible to repeat. N=1 → 2 total trials, N=2 → 4, N=3 → 6. |
| All tasks \| THINGS \| N>1854 | Runs with 1854 unique images | Full THINGS (1854 categories, ~14 exemplars each, ~5.2 GB) now lives at `/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/THINGS/object_images_full/object_images/` and is symlinked from `memory_datasets/THINGS/object_images`. Provisioning script: `scripts/download_things_osf.sh` (pulls from OSF project `jum2f`, password-protected zip). Older runs (pre-2026-05-07) used a 225-category subset — see git history of this file and any `n>=100` THINGS results from before that date for context. |
| Associative inference \| N<4 | Runs with novel foil | With only 1 ABC chain, no second chain is available for a foil. An unstudied novel image is used instead. Dataset loads one extra image beyond the 3 chain images to serve as foil. |

## Model/API Limits

### GPT-4o
- **N=500 skipped across all tasks** — OpenAI's API has a ~20MB JSON body limit. At 500 images (512px JPEG), the request exceeds this limit and returns a 400 error.
- Max tested: N=100.

### Anthropic (Claude)
- **Max 98 study images per 2-AFC trial** — API limit of 100 images per request, with 2 reserved for test images.
- **Max 98 study pairs per PAM trial** — same limit, 1 reserved for test image.

### Gemini (gemini-2.5-flash)
- No hard image count limit, but subject to **rate limiting (429)**. Retry logic handles this with exponential backoff (up to ~64 min total wait).

### Qwen (Qwen3-VL-8B-Instruct)
- Local inference, no API limits. Requires GPU (A6000) and all model weights present in `$HF_HOME/hub/`.
- **N>=500 skipped across all tasks** — VRAM exhaustion at large haystack sizes on A6000 (48GB).
- Model weights are split across two cache locations — shards 1–2 in `~/.cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/` and shards 3–4 in `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`. Symlinks resolve this.
- **Model name bug (fixed):** `get_name()` used case-sensitive `"8B" in self.model_id`, causing results to be labeled `qwen3-vl` instead of `qwen3-vl-8b` when the model ID contained lowercase `8b`. Fixed to use `.upper()`. Affected result files have been patched.

## Task Scope

### Visual Haystacks (VHS benchmark)
- Uses the published VHs benchmark (`eval_vhs.py`) with `single_needle` and `multi_needle` modes.
- Haystack sizes are **fixed by the benchmark's file structure**: valid `--image-count` values for single_needle are `oracle, 2, 5, 10, 20, 50, 100`. Sizes 1 and 250 have no QA file and are skipped by the run scripts.
- **`oracle` is the 1-image condition** — each trial shows only the positive (needle) image with no distractors. This is a perception baseline, not a memory/search test, and should be interpreted as a ceiling condition.
- **multi_needle does not have a size-2 or oracle file**. Multi-needle valid sizes: `5, 10, 50, 100`.
- The custom `eval_visual_haystacks.py` (results prefix `results_haystacks_*`) is a separate internal task and is **not included in final results**.
- **N=500 skipped** — only 224 THINGS images are available for this task; raises `AssertionError: Haystack size mismatch: expected 500, got 224`. Valid sizes: N=10, 100. Applies to all models.

### Foil Types (2-AFC)
- **THINGS dataset:** supports `novel`, `exemplar`, `all` (mixed novel+exemplar). `state` is **not supported** — THINGS images have no within-object state variation (e.g., no open vs. closed, empty vs. full). State foil trials are skipped entirely for this dataset.
- **THINGS N cap for novel and all foils:** `novel` foils require 2 distinct categories per trial (one for original, one for foil), so N images need 2N categories → max N = 927 with 1854 available. `all` foils are split half novel / half exemplar, requiring `N + N/2 = 1.5N` categories → max N = 1235. `exemplar` only needs 1 category per trial → max N = 1854. The pre-2026-05-07 SKIP guards in `scripts/run_2afc_*.sh` (which capped novel/all at N<250) were removed once the full set was provisioned, since current `SIZES=(1 2 5 10 50 100 250)` is well under every cap. If `SIZES` is later expanded past these limits, the underlying `AFCRecognitionTask` already raises a descriptive `ValueError` from `tasks/afc_recognition.py:_get_pairs`.
- **Brady2008 dataset:** supports `novel`, `exemplar`, `state`, and `all` (mixed novel+exemplar+state).
- Older runs used `foil_type: "all"` or `foil_type: "accuracy"` to indicate a mixed/undifferentiated foil condition. These have been renamed to `"all"` for consistency and are plotted as a separate series.

## Result File Conventions
- Standard sizes for all tasks: N = 1, 5, 10, 100, 500, 1000 (subject to per-model and per-task limits above).
- Non-standard sizes (3, 6, 20, 50, 200, 300, 400, etc.) have been moved to `results_archive/` and are excluded from analysis.
