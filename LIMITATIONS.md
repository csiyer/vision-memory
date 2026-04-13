# Experimental Limitations

## Dataset Limits

| Condition | Limit | Reason |
|-----------|-------|--------|
| Brady2008 \| exemplar \| N>100 | SKIP | Only ~200 images in dataset |
| Brady2008 \| state \| N>100 | SKIP | Only ~200 images in dataset |
| Continuous recognition \| N<10 | SKIP | Too few images to construct valid trials |

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
- Model weights are split across two cache locations — shards 1–2 in `~/.cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/` and shards 3–4 in `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`. Symlinks resolve this.
