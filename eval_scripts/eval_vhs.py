#!/usr/bin/env python
"""
Visual Haystacks (VHs) Evaluation

Usage examples:
  python -m eval_scripts.eval_vhs --models gpt-4o gemini --mode single_needle --split VHs_large --image-count 10
  python -m eval_scripts.eval_vhs --models gpt-4o --mode multi_needle --image-count 10 --max-samples 100
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime

from tasks.visual_haystacks import VisualHaystacksTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator


def parse_yes_no(text):
    """Parse yes/no from response text."""
    if text is None:
        return "unknown"
    lower = text.lower().strip()
    has_yes = re.search(r"\byes\b", lower) is not None
    has_no = re.search(r"\bno\b", lower) is not None
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unknown"


def build_messages(evaluator, prompt, images, max_context_images=None):
    # Anthropic commonly has lower practical image count limits.
    if "Anthropic" in type(evaluator).__name__:
        provider_limit = 100
    elif "OpenAI" in type(evaluator).__name__:
        provider_limit = 500  # GPT-4o: max 500 images per request
    else:
        provider_limit = len(images)
    if max_context_images is not None:
        n_ctx = min(max_context_images, provider_limit)
    else:
        n_ctx = provider_limit
    images = images[:n_ctx]

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(evaluator._encode_image(img))
    return [{"role": "user", "content": content}], n_ctx


def compute_metrics(trials):
    total = len(trials)
    valid = [t for t in trials if t["parsed_response"] in ("yes", "no")]
    n_valid = len(valid)
    n_correct = sum(1 for t in valid if t["correct"] == 1)
    overall_correct = sum(1 for t in trials if t["correct"] == 1)
    accuracy = overall_correct / total if total > 0 else 0.0
    valid_accuracy = n_correct / n_valid if n_valid > 0 else 0.0
    compliance = n_valid / total if total > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "valid_accuracy": float(valid_accuracy),
        "compliance": float(compliance),
        "n_trials": total,
        "n_valid_trials": n_valid,
        "n_correct": int(overall_correct),
        "n_parse_failures": total - n_valid,
    }


def run_evaluation(evaluators, args):
    task = VisualHaystacksTask(
        qa_root=args.qa_root,
        image_root=args.image_root,
        mode=args.mode,
        split=args.split,
        image_count=args.image_count,
        max_samples=args.max_samples,
        shuffle_images=not args.no_shuffle,
        seed=args.seed,
        fetch_missing_coco=args.fetch_missing_coco,
        coco_base_url=args.coco_base_url,
        fetch_timeout_s=args.coco_fetch_timeout,
    )
    trials = task.get_trials()

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        model_trials = []

        for i, trial in enumerate(trials):
            messages, n_ctx = build_messages(
                evaluator,
                trial["prompt"],
                trial["images"],
                max_context_images=args.max_context_images,
            )
            raw_response = evaluator._call_api(messages)
            parsed = parse_yes_no(raw_response)
            correct = 1 if parsed == trial["target"] else 0

            model_trials.append(
                {
                    "trial": i,
                    "target": trial["target"],
                    "response": parsed,
                    "parsed_response": parsed,
                    "raw_response": raw_response,
                    "correct": correct,
                    "n_images_sent": n_ctx,
                    "metadata": trial["metadata"],
                }
            )
            status = "✓" if correct else "✗"
            print(f"  Trial {i + 1}/{len(trials)}: {status}", end="\r")

        metrics = compute_metrics(model_trials)
        print(
            f"\n  Accuracy: {metrics['accuracy']:.1%} | "
            f"Valid Acc: {metrics['valid_accuracy']:.1%} | "
            f"Compliance: {metrics['compliance']:.1%}"
        )
        all_results[evaluator.get_name()] = {"trials": model_trials, **metrics}

    return all_results, len(trials)


def _find_completed_models(results_dir, prefix, **match_fields):
    """Return set of model names that already have a completed result matching the given metadata fields."""
    completed = set()
    for path in Path(results_dir).glob(f"{prefix}*.json"):
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
    parser = argparse.ArgumentParser(description="Visual Haystacks Benchmark Evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o", "claude", "gemini"],
        help="Models to evaluate: gpt-4o, claude, gemini, qwen",
    )
    parser.add_argument(
        "--mode",
        choices=["single_needle", "multi_needle"],
        default="single_needle",
        help="VHs challenge mode",
    )
    parser.add_argument(
        "--split",
        choices=["VHs_large", "VHs_small"],
        default="VHs_large",
        help="Single-needle split; ignored for multi-needle",
    )
    parser.add_argument(
        "--image-count",
        type=str,
        default="10",
        help="Haystack count entry in VHs file name (e.g., oracle, 2, 3, 5, 10, 20, 50, 100)",
    )
    parser.add_argument(
        "--qa-root",
        type=str,
        default="datasets/VHs_qa",
        help="Path to downloaded Visual Haystacks QA files",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="datasets/coco",
        help="Path to COCO root containing val2017/test2017 directories",
    )
    parser.add_argument(
        "--fetch-missing-coco",
        action="store_true",
        help=(
            "If a referenced COCO image is missing locally, download it from "
            "images.cocodataset.org into --image-root (saves disk vs full zip)."
        ),
    )
    parser.add_argument(
        "--coco-base-url",
        type=str,
        default="https://images.cocodataset.org",
        help="Base URL for on-demand COCO fetches (e.g., http://images.cocodataset.org)",
    )
    parser.add_argument(
        "--coco-fetch-timeout",
        type=int,
        default=120,
        help="Per-image HTTP timeout (seconds) when using --fetch-missing-coco",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of VHs items")
    parser.add_argument(
        "--max-context-images",
        type=int,
        default=None,
        help="Maximum number of images to send to model per trial",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image shuffling")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable image order shuffling")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path under results/ (default: results_vhs_<timestamp>.json)",
    )
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    done = _find_completed_models(
        results_dir, prefix="results_vhs_",
        task="Visual Haystacks", mode=args.mode,
        image_count=str(args.image_count), max_samples=args.max_samples,
    )
    if done:
        print(f"Skipping already-completed models: {sorted(done)}")

    evaluators = []
    if "gpt-4o" in args.models and "gpt-4o" not in done:
        evaluators.append(OpenAIEvaluator("gpt-4o"))
    if "claude" in args.models and "claude" not in done:
        evaluators.append(AnthropicEvaluator())
    if "gemini" in args.models and "gemini-2.5-flash" not in done:
        evaluators.append(GoogleEvaluator())
    if "qwen" in args.models and "Qwen/Qwen3-VL-8B-Instruct" not in done:
        evaluators.append(QwenEvaluator("Qwen/Qwen3-VL-8B-Instruct"))
    if not evaluators:
        print("No valid models specified (or all already completed). Use --models gpt-4o claude gemini qwen")
        return

    print("Running Visual Haystacks evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  Mode: {args.mode}")
    print(f"  Split: {args.split}")
    print(f"  Image count: {args.image_count}")
    print(f"  QA root: {args.qa_root}")
    print(f"  Image root: {args.image_root}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Max context images: {args.max_context_images or 'all'}")
    print(f"  Fetch missing COCO images: {args.fetch_missing_coco}")
    if args.fetch_missing_coco:
        print(f"  COCO base URL: {args.coco_base_url}")

    results, n_trials = run_evaluation(evaluators, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "Visual Haystacks",
            "timestamp": timestamp,
            "mode": args.mode,
            "split": args.split,
            "image_count": args.image_count,
            "qa_root": args.qa_root,
            "image_root": args.image_root,
            "n_trials": n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "valid_accuracy": results[model]["valid_accuracy"],
                    "compliance": results[model]["compliance"],
                }
                for model in results
            },
            "max_samples": args.max_samples,
            "max_context_images": args.max_context_images,
            "seed": args.seed,
            "shuffle_images": not args.no_shuffle,
            "fetch_missing_coco": args.fetch_missing_coco,
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / (args.output if args.output else f"results_vhs_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
