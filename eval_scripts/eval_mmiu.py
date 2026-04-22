#!/usr/bin/env python
"""
MMIU (Multimodal Multi-Image Understanding) Evaluation

Dataset: FanqingM/MMIU-Benchmark (HuggingFace, ~25.5 GB)
Format:  Multiple-choice (A-D), 11,698 samples, avg 6.6 images/sample
Tasks:   52 tasks across 7 image-relation types

The dataset images are referenced by relative paths in `input_image_path`.
HuggingFace streams them as PIL Images in the `images` column when loaded
with streaming=False (default). This script handles both:
  - HuggingFace-loaded PIL images (standard path)
  - Local image root via --image-root (if you've downloaded the dataset)

Usage:
  python -m eval_scripts.eval_mmiu --models gpt-4o gemini --max-samples 100
  python -m eval_scripts.eval_mmiu --models gpt-4o --tasks forensic_detection_blink scene_comparison
  python -m eval_scripts.eval_mmiu --models gpt-4o gemini qwen --max-samples 500
  python -m eval_scripts.eval_mmiu --models gpt-4o --image-root datasets/mmiu
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

from PIL import Image

from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator


MMIU_DATASET = "FanqingM/MMIU-Benchmark"

PROMPT_TEMPLATE = """\
Look at the images provided and answer the following multiple-choice question.
Respond with only the letter of the correct answer (A, B, C, or D).

{context}{question}

{options}

Answer:"""


MMIU_PARQUET = (
    Path.home() / ".cache/huggingface/hub"
    / "datasets--FanqingM--MMIU-Benchmark"
    / "snapshots/03bf7d143d920e97a757f606b6b7baee161b019b/all.parquet"
)


def _ensure_image_root(image_root: Path) -> Path:
    """Check that image_root exists and has been extracted."""
    subdirs = [p for p in image_root.iterdir() if p.is_dir()] if image_root.exists() else []
    if not subdirs:
        raise FileNotFoundError(
            f"MMIU image root '{image_root}' is missing or empty.\n"
            f"Run first:  bash download_mmiu.sh"
        )
    return image_root


def load_dataset(max_samples=None, tasks=None, image_root=None):
    import pyarrow.parquet as pq

    if image_root is None:
        raise ValueError("--image-root is required for MMIU. Use --image-root datasets/mmiu")

    root = Path(image_root)
    _ensure_image_root(root)

    print(f"Loading MMIU from parquet + image root '{root}'...")
    table = pq.read_table(str(MMIU_PARQUET))
    rows = table.to_pydict()
    n_rows = len(rows["task"])

    samples = []
    n_skipped = 0
    for i in range(n_rows):
        task = rows["task"][i]
        if tasks and task not in tasks:
            continue

        row = {k: rows[k][i] for k in rows}
        images = _get_images(row, root)
        if not images:
            n_skipped += 1
            continue

        samples.append({
            "task": task,
            "visual_input_component": row.get("visual_input_component") or "",
            "question": row["question"],
            "context": row.get("context") or "",
            "options": row["options"],
            "answer": row["output"],
            "images": images,
            "image_paths": row.get("input_image_path") or [],
        })
        if max_samples and len(samples) >= max_samples:
            break

    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} rows with no loadable images.")
    print(f"  Loaded {len(samples)} samples")
    return samples


def _get_images(row, image_root=None):
    """Return list of PIL Images for a row."""
    # HuggingFace datasets with Image features decode automatically
    if "images" in row and row["images"]:
        imgs = row["images"]
        # Could be list of PIL or list of dicts with 'bytes'/'path'
        result = []
        for img in imgs:
            if isinstance(img, Image.Image):
                result.append(img.convert("RGB"))
            elif isinstance(img, dict) and "bytes" in img and img["bytes"]:
                from io import BytesIO
                result.append(Image.open(BytesIO(img["bytes"])).convert("RGB"))
            elif isinstance(img, dict) and "path" in img and img["path"]:
                result.append(Image.open(img["path"]).convert("RGB"))
        if result:
            return result

    # Fallback: load from local image_root using input_image_path
    if image_root and "input_image_path" in row and row["input_image_path"]:
        root = Path(image_root)
        result = []
        for rel_path in row["input_image_path"]:
            # Paths are like "./Low-level-semantic/task/task_0_0.jpg"
            # but extracted structure is flat: mmiu/task/task_0_0.jpg
            clean = rel_path.lstrip("./")
            full = root / clean
            if not full.exists():
                # Try skipping the category prefix (e.g. "Low-level-semantic/")
                parts = Path(clean).parts
                if len(parts) > 2:
                    full = root / Path(*parts[1:])
            if full.exists():
                result.append(Image.open(full).convert("RGB"))
        return result

    return []


def parse_options_string(options_str):
    """Parse 'A: foo\nB: bar\n...' into list of option strings."""
    lines = [l.strip() for l in options_str.strip().splitlines() if l.strip()]
    # Strip leading "A: ", "B: " etc.
    cleaned = []
    for line in lines:
        m = re.match(r"^[A-Da-d][:.)\s]\s*(.+)$", line)
        cleaned.append(m.group(1) if m else line)
    return cleaned


def format_options(options_str):
    """Re-format options string, ensuring A/B/C/D labels."""
    lines = [l.strip() for l in options_str.strip().splitlines() if l.strip()]
    return "\n".join(lines)


def build_messages(evaluator, sample):
    context_prefix = f"Context: {sample['context']}\n\n" if sample["context"] else ""
    prompt = PROMPT_TEMPLATE.format(
        context=context_prefix,
        question=sample["question"],
        options=format_options(sample["options"]),
    )
    content = [{"type": "text", "text": prompt}]
    for img in sample["images"]:
        content.append(evaluator._encode_image(img))
    return [{"role": "user", "content": content}]


def parse_answer(text):
    """Extract A/B/C/D from response."""
    if text is None:
        return None
    text = text.strip()
    if re.match(r"^[A-Da-d]\.?$", text):
        return text[0].upper()
    m = re.match(r"^([A-Da-d])[.):,\s]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return None


def compute_metrics(trials):
    total = len(trials)
    valid = [t for t in trials if t["parsed_response"] is not None]
    n_valid = len(valid)
    n_correct = sum(1 for t in valid if t["correct"])

    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    by_component = defaultdict(lambda: {"correct": 0, "total": 0})
    for t in trials:
        task = t["metadata"]["task"]
        comp = t["metadata"]["visual_input_component"]
        by_task[task]["total"] += 1
        by_component[comp]["total"] += 1
        if t["correct"]:
            by_task[task]["correct"] += 1
            by_component[comp]["correct"] += 1

    task_acc = {k: v["correct"] / v["total"] for k, v in by_task.items()}
    component_acc = {k: v["correct"] / v["total"] for k, v in by_component.items()}

    return {
        "accuracy": n_correct / total if total > 0 else 0.0,
        "valid_accuracy": n_correct / n_valid if n_valid > 0 else 0.0,
        "compliance": n_valid / total if total > 0 else 0.0,
        "n_trials": total,
        "n_valid": n_valid,
        "n_correct": n_correct,
        "n_parse_failures": total - n_valid,
        "accuracy_by_task": task_acc,
        "accuracy_by_visual_component": component_acc,
    }


def run_evaluation(evaluators, samples):
    all_results = {}

    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        trials = []

        for i, sample in enumerate(samples):
            messages = build_messages(evaluator, sample)
            raw = evaluator._call_api(messages)
            parsed = parse_answer(raw)
            correct = parsed == sample["answer"] if parsed is not None else False

            trials.append({
                "target": sample["answer"],
                "parsed_response": parsed,
                "raw_response": raw,
                "correct": correct,
                "n_images": len(sample["images"]),
                "metadata": {
                    "task": sample["task"],
                    "visual_input_component": sample["visual_input_component"],
                },
            })
            status = "✓" if correct else "✗"
            print(f"  Sample {i + 1}/{len(samples)}: {status}  [{sample['task']}]", end="\r")

        metrics = compute_metrics(trials)
        print(
            f"\n  Accuracy: {metrics['accuracy']:.1%} | "
            f"Valid Acc: {metrics['valid_accuracy']:.1%} | "
            f"Compliance: {metrics['compliance']:.1%}"
        )
        all_results[evaluator.get_name()] = {"trials": trials, **metrics}

    return all_results


def _find_completed_models(results_dir, **match_fields):
    """Return set of model names that already have a completed result matching the given metadata fields."""
    completed = set()
    for path in Path(results_dir).glob("results_mmiu_*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            meta = data.get("_metadata", {})
            if all(str(meta.get(k)) == str(v) for k, v in match_fields.items() if v is not None):
                completed.update(meta.get("models", []))
        except Exception:
            pass
    return completed


def main():
    parser = argparse.ArgumentParser(description="MMIU Multi-Image Evaluation")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o", "gemini"],
        help="Models: gpt-4o, claude, gemini, qwen",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (default: all ~11698)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Filter to specific task names (e.g. forensic_detection_blink)")
    parser.add_argument("--image-root", type=str, default=None,
                        help="Local path to downloaded MMIU images (fallback if HF images unavailable)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename under results/ (default: results_mmiu_<ts>.json)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    done = _find_completed_models(results_dir, task="MMIU", max_samples=args.max_samples)
    if done:
        print(f"Skipping already-completed models: {sorted(done)}")

    evaluators = []
    for m in args.models:
        m = m.strip()
        if m == "gpt-4o":
            ev = OpenAIEvaluator("gpt-4o")
        elif m == "claude":
            ev = AnthropicEvaluator()
        elif m == "gemini":
            ev = GoogleEvaluator()
        elif m == "qwen":
            ev = QwenEvaluator("Qwen/Qwen3-VL-8B-Instruct")
        elif m.startswith("claude"):
            ev = AnthropicEvaluator(m)
        elif m.startswith("gemini"):
            ev = GoogleEvaluator(m)
        elif m.startswith("qwen") or m.startswith("Qwen"):
            ev = QwenEvaluator(m)
        else:
            ev = OpenAIEvaluator(m)
        if ev.get_name() not in done:
            evaluators.append(ev)

    if not evaluators:
        print("No valid models specified (or all already completed).")
        return

    samples = load_dataset(
        max_samples=args.max_samples,
        tasks=args.tasks,
        image_root=args.image_root,
    )

    if not samples:
        print("No samples loaded. Check dataset availability and filters.")
        return

    print(f"\nRunning MMIU evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  Samples: {len(samples)}")
    if args.tasks:
        print(f"  Tasks filter: {args.tasks}")

    results = run_evaluation(evaluators, samples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "MMIU",
            "timestamp": timestamp,
            "dataset": MMIU_DATASET,
            "n_samples": len(samples),
            "tasks_filter": args.tasks,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "valid_accuracy": results[model]["valid_accuracy"],
                    "compliance": results[model]["compliance"],
                    "accuracy_by_task": results[model]["accuracy_by_task"],
                    "accuracy_by_visual_component": results[model]["accuracy_by_visual_component"],
                }
                for model in results
            },
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    fname = args.output or f"results_mmiu_{timestamp}.json"
    output_path = results_dir / fname
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
