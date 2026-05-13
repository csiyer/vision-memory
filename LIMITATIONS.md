# Experimental Limitations

## Per-Trial Image Budget

Each task packs a different number of images into a single API request. The capacity probe (`check_image_capacity`) is called with this count before a run; if the model rejects it, the run is skipped (an empty stub result file may still be written).

| Task | Variant | Images per trial | Source |
|------|---------|------------------|--------|
| 2-AFC recognition | — | **N + 2** | `eval_2afc.py:90` |
| Continuous recognition | — | **N** (sequential, see note) | `eval_continuous.py:65` |
| Color memory | continuous | **N + 2** | `eval_color_memory.py:130` |
| Color memory | named | **N + 1** | `eval_color_memory.py:187` |
| Serial order | free recall | **N + 1** | `eval_serial_order.py:139` |
| Serial order | afc | **N + 2** | `eval_serial_order.py:197` |
| PAM | word | **N + 1** | `eval_pam.py:154` |
| PAM | image | **2N + 3** | `eval_pam.py:154` |
| Associative inference | word | **1.5·N + 1** (N must be even) | `eval_associative_inference.py:114` |
| Associative inference | image | **2N + 3** (N must be even) | `eval_associative_inference.py:114` |
| MST | — | **N + 1** | `eval_mst.py:103` |

This determines which `(task, N, model)` combos are reachable — see per-model image limits below.

## Per-Model Image Limits

### GPT-4o
- **Hard cap: 500 images per request** (`Too many images in request: 501, maximum allowed: 500`).
- Effective max N by task (capacity_n ≤ 500):
  - 2-AFC: N ≤ 498 · PAM word: N ≤ 499 · **PAM image: N ≤ 248**
  - Assoc image: N ≤ 248 (even) · Color continuous: N ≤ 498 · MST: N ≤ 499
- Confirmed failure: PAM image at N=250 needs 503 → probe rejects → result file written with empty `summary: {}` and no trials (both `things` and `Brady2008`).
- Also: ~20 MB JSON body limit at 512-px JPEGs — would compound at N≈500.

### Anthropic (Claude, claude-sonnet-4-0)
- Hard cap: 100 images per request.
- Effective max N: 2-AFC ≤ 98 · PAM word ≤ 99 · PAM image ≤ 48 · Assoc image ≤ 48 · Color continuous ≤ 98 · MST ≤ 99.
- Note: Claude scripts (`run_*_claude.sh`) currently have no VHS variants.

### Gemini (gemini-2.5-flash)
- No hard image-count cap, but **rate-limited (429 `RESOURCE_EXHAUSTED`)** against the 1M input-tokens/min quota.
- Per-trial calls retry on 429 with exponential backoff (up to ~64 min total wait).
- **Capacity probe does NOT retry on 429.** A single 429 during `check_image_capacity` (`evaluators/google_evaluator.py:76`) aborts the whole `(task, N)` cell. Symptom: sparse, non-adjacent missing cells for a single task (e.g., `color_named` n=1, 2, 250 in `logs/9443347.err`). Fix: resubmit when quota refreshes; or add 429 retry to the probe.
- VHS uses `gemini-3-flash-preview` (different model) per `run_vhs_*_gemini.sh`.

### Qwen (Qwen3-VL-8B-Instruct)
- Local inference on A6000 (48 GB VRAM). Images are downscaled to max 512×512 in `qwen_evaluator.py:42`.
- **Empirical ceiling: N ≈ 250** with 512-px images on A6000.
  - Capacity probe (with 512-px images, `qwen_evaluator.py:49`) reports OK up to ~376 images, but real eval runs at N=250 have OOM'd (47.4 GB / 48 GB allocated — see `logs/9369819.err`). Probe optimistic vs. trial-time KV-cache + activations.
- Standard scripts cap at N=250 and skip N=50 (`SIZES=(1 2 5 10 100 250)`).
- VHS up to N=500 is **untested** for qwen (no result files); likely OOMs without reducing `max_size` (line 42), enabling `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or moving to a larger GPU.
- Model weights are split across two cache locations — shards 1–2 in `~/.cache/huggingface/models--Qwen--Qwen3-VL-8B-Instruct/`, shards 3–4 in `~/.cache/huggingface/hub/...`. Symlinks resolve this.

### Molmo2
- Local inference on A6000 (48 GB VRAM).
- **Empirical ceiling: N < 50.** Capacity probe (1×1-px images) reports OK at n=50, but real 2-AFC eval OOMs at n=50, 100, and 250 across all foil types (`logs/9369816.out` shows `Probing capacity for 50 images... OK` followed by `[ERROR] things | novel | n=50`; `logs/9369816.err` shows `CUDA out of memory ... 47.20 GiB memory in use` of 47.40 GiB total). Probe optimistic vs. trial-time KV-cache + activations, same pattern as Qwen but at a lower threshold.
- Standard scripts use `SIZES=(1 2 5 10 100 250)`; cells N≥50 currently fail. Effective max N ≈ 10 on A6000 until `max_size` is reduced, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set, or a larger GPU is used.

## Dataset Limits

| Condition | Limit | Reason |
|-----------|-------|--------|
| Brady2008 \| exemplar \| N>100 | SKIP | Only ~200 images in dataset (`exemplar`/`state` foils consume 2 per trial) |
| Brady2008 \| state \| N>100 | SKIP | Only ~200 images in dataset |
| All tasks \| THINGS \| N>1854 | Auto-clamp to 1854 | Full THINGS (1854 categories) symlinked from `/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/data/THINGS/object_images_full/object_images/` via `memory_datasets/THINGS/object_images`. Provisioning: `scripts/download_things_osf.sh` (OSF `jum2f`, password-protected). Pre-2026-05-07 runs used a 225-category subset. |
| Continuous recognition \| N<4 | Runs with `min_delay=0` | Too few images for `min_delay=2`; delay constraint dropped. N=1 → 2 trials, N=2 → 4, N=3 → 6. |
| Associative inference \| image variant \| THINGS \| N=2 | SKIP (all models) | n=2 → n_pairs=1; foil must come from outside the ABC triplet but `ThingsDataset` is loaded with exactly `n_pairs*3=3` items → `ValueError` at `tasks/associative_inference.py:140`. Brady2008 OK (~200 images). To enable, load `n_pairs*3 + 1`. |
| Paired associate memory \| image variant \| THINGS \| N=1 | SKIP (all models) | n=1 → one pair; 2-AFC foil must come from outside but only 2 categories loaded → `ValueError` at `tasks/paired_associate_memory.py:132`. Brady2008 OK. To enable, load `n_images*2 + 1`. |
| Associative inference \| any \| odd N | Rejected | `n_trials` must be even so study splits into AB/BC pairs (`tasks/associative_inference.py:13-14`). SIZES start at N=2. |

## Task Scope

### Visual Haystacks (VHS benchmark)
- Uses the published VHs benchmark (`eval_vhs.py`) with `single_needle` and `multi_needle` modes.
- Haystack sizes are **fixed by the benchmark's file structure**:
  - `single_needle`: valid `--image-count` ∈ {`oracle`, 2, 5, 10, 20, 50, 100, 500}. Sizes 1 and 250 have no QA file and are skipped.
  - `multi_needle`: valid ∈ {5, 10, 20, 50, 100, 500}. No `oracle` or size-2 file.
- **`oracle` is the 1-image condition** — only the positive (needle) image with no distractors. Perception baseline, not a memory/search test.
- **N=500 on custom haystacks (`eval_visual_haystacks.py`) skipped** — only 224 THINGS images available; raises `AssertionError: Haystack size mismatch: expected 500, got 224`. That script is the older internal task (results prefix `results_haystacks_*`) and is **not included in final results**.
- Claude has no VHS run scripts; gemini's VHS scripts use `gemini-3-flash-preview` (not `gemini-2.5-flash`).

### Foil Types (2-AFC)
- **THINGS dataset:** supports `novel`, `exemplar`, `all` (mixed). **`state` not supported** — THINGS images have no within-object state variation; state-foil trials are skipped for this dataset.
- **THINGS N caps by foil type:** `novel` → max N = 927 (needs 2 categories/trial out of 1854). `all` → max N = 1235 (needs 1.5N). `exemplar` → max N = 1854 (1 category/trial). Current `SIZES=(1 2 5 10 50 100 250)` is well under all caps; `AFCRecognitionTask` in `tasks/afc_recognition.py:_get_pairs` raises a descriptive `ValueError` if exceeded.
- **Brady2008 dataset:** supports `novel`, `exemplar`, `state`, `all`.
- Older runs used `foil_type: "accuracy"` for a mixed/undifferentiated foil condition. These have been renamed to `"all"` for consistency.

## Result File Conventions

- **Standard sizes** (in `scripts/run_*.sh`):
  - API models (Claude, Gemini, GPT-4o): `SIZES=(1 2 5 10 50 100 250)`; assoc uses `(2 4 6 10 50 100 250)` (even N only).
  - Local models (Qwen, Molmo2): drop N=50 — `SIZES=(1 2 5 10 100 250)` — due to throughput/VRAM.
  - VHS: `single_needle` uses `(oracle 2 5 10 50 100 250)`; `multi_needle` uses `(5 10 20 50 100 500)`. Note: 250/500 cells often fail per-model limits above.
- **Archived results** live in `results_archive/` with descriptive subdirs/suffixes:
  - Non-standard sizes from earlier runs (3, 6, 20, 200, 300, 400)
  - Failed runs by category: e.g. `2026-05-08_downgrade_redo/`, `2026-05-10_truncation_fix/`, `2026-05-12_qwen_continuous_brady_empty/`
  - Parse-failure files: `*.parse_fail_YYYYMMDD.json`
