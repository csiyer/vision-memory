# Experimental Limitations

## Dataset Limits

| Condition | Limit | Reason |
|-----------|-------|--------|
| Brady2008 \| exemplar \| N>100 | SKIP | Only ~200 images in dataset |
| Brady2008 \| state \| N>100 | SKIP | Only ~200 images in dataset |
| Continuous recognition \| N<5 | SKIP | Too few trials to be a meaningful experiment (N=1 → 2 total trials; N<5 excluded by convention) |

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
- Haystack sizes are **fixed by the benchmark's file structure**: valid `--image-count` values are `2, 3, 5, 10, 20, 50, 100`. Arbitrary sizes (e.g. 1, 500, 1000) are not supported.
- **multi_needle does not have a size-2 file** (`visual_haystack_2.json` only exists for single_needle). Multi-needle valid sizes: `5, 10, 50, 100`.
- The custom `eval_visual_haystacks.py` (results prefix `results_haystacks_*`) is a separate internal task and is **not included in final results**.
- **N=1 skipped** — requires at least `n_needles + 1 = 2` images; raises `ValueError` for N=1.
- **N=500 skipped** — only 224 THINGS images are available for this task; raises `AssertionError: Haystack size mismatch: expected 500, got 224`. Valid sizes: N=10, 100. Applies to all models.

### Foil Types (2-AFC)
- **THINGS dataset:** supports `novel`, `exemplar`, `all` (mixed novel+exemplar). `state` is **not supported** (no within-object state variation in THINGS).
- **Brady2008 dataset:** supports `novel`, `exemplar`, `state`, and `all` (mixed novel+exemplar+state).
- Older runs used `foil_type: "all"` or `foil_type: "accuracy"` to indicate a mixed/undifferentiated foil condition. These have been renamed to `"all"` for consistency and are plotted as a separate series.

## Result File Conventions
- Standard sizes for all tasks: N = 1, 5, 10, 100, 500, 1000 (subject to per-model and per-task limits above).
- Non-standard sizes (3, 6, 20, 50, 200, 300, 400, etc.) have been moved to `results_archive/` and are excluded from analysis.
