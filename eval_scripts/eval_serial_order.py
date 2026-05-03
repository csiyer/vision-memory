#!/usr/bin/env python
"""
Serial Order Memory Evaluation

Two variants:
  free  — study N images, then for each image report its serial position (1-N)
  afc   — study N images, then for each pair of images choose which came first

Usage:
    python -m eval_scripts.eval_serial_order --models gpt-4o --n-images 20 --variant free
    python -m eval_scripts.eval_serial_order --models gpt-4o claude gemini --n-images 50 --variant afc
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime
from tasks.serial_order_memory import SerialOrderMemoryTask, AFCSerialOrderMemoryTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_serial_order_metrics, calculate_afc_serial_order_metrics


def build_messages_free(evaluator, study_sequence, study_prompt, test_image, test_prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_sequence:
        study_content.append(evaluator._encode_image(img))

    test_content = [
        evaluator._encode_image(test_image),
        {"type": "text", "text": test_prompt},
    ]

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content},
    ]


def build_messages_afc(evaluator, study_sequence, study_prompt, test_images, test_prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_sequence:
        study_content.append(evaluator._encode_image(img))

    test_content = [{"type": "text", "text": test_prompt}]
    for img in test_images:
        test_content.append(evaluator._encode_image(img))

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content},
    ]


def parse_position_response(text, n):
    """Parse an integer position 1-N from free-report response.

    Takes the last candidate value since models typically state their final
    answer at the end of a verbose response.
    """
    if text is None:
        return -1

    candidates = []

    # digit ordinals: 1st, 2nd, 3rd, 4th, ...
    for m in re.finditer(r"\b(\d+)(?:st|nd|rd|th)\b", text, re.IGNORECASE):
        val = int(m.group(1))
        if 1 <= val <= n:
            candidates.append((m.start(), val))

    # plain integers
    for m in re.finditer(r"\b(\d+)\b", text):
        val = int(m.group(1))
        if 1 <= val <= n:
            candidates.append((m.start(), val))

    if candidates:
        # take the last match
        return max(candidates, key=lambda x: x[0])[1]

    # ordinal words fallback for small N
    ordinal_words = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    lower = text.lower()
    last_word_match = None
    for m in re.finditer(r"\b(" + "|".join(ordinal_words) + r")\b", lower):
        val = ordinal_words[m.group(1)]
        if 1 <= val <= n:
            last_word_match = val
    if last_word_match is not None:
        return last_word_match

    return -1


def parse_afc_response(text):
    """Parse 1 or 2 from AFC response, taking the first standalone digit since
    models typically lead with their answer (e.g. 'Image 1 appeared first...')."""
    if text is None:
        return -1
    text = text.strip()

    # First standalone 1 or 2 (not part of a larger number)
    m = re.search(r"(?<!\d)[12](?!\d)", text)
    if m:
        return int(m.group())

    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


def run_free_evaluation(evaluators, n_images=20, dataset="things", n_trials=None):
    """Each trial is an independent study+test episode with a fresh sample of n_images."""
    n_trials = n_trials if n_trials is not None else n_images

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        print(f"  Probing capacity for {n_images} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_images + 1):
            print(f"SKIP — model rejected {n_images} images in a single request")
            continue
        print("OK")

        results = []

        for i in range(n_trials):
            task = SerialOrderMemoryTask(dataset_name=dataset, n_images=n_images)
            trial_data = task.get_trials()
            n = len(trial_data["study_sequence"])
            test_trial = trial_data["test_phase"][0]

            messages = build_messages_free(
                evaluator,
                trial_data["study_sequence"],
                trial_data["study_prompt"],
                test_trial["image"],
                test_trial["prompt"],
            )
            response_text = evaluator._call_api(messages)
            reported = parse_position_response(response_text, n)
            correct = 1 if reported == test_trial["target"] else 0

            results.append({
                "trial": i,
                "target": test_trial["target"],
                "reported": reported,
                "correct": correct,
                "raw_response": response_text,
                "metadata": test_trial["metadata"],
            })
            status = "✓" if correct else "✗"
            print(f"  Trial {i+1}/{n_trials}: {status} (target={test_trial['target']}, got={reported})", end="\r")

        reported_positions = [r["reported"] for r in results if r["reported"] != -1]
        actual_positions = [r["target"] for r in results if r["reported"] != -1]
        metrics = calculate_serial_order_metrics(reported_positions, actual_positions)
        metrics["n_parse_failures"] = sum(1 for r in results if r["reported"] == -1)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} | Avg error: {metrics['average_error']:.2f} positions")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def run_afc_evaluation(evaluators, n_images=20, dataset="things", n_trials=None):
    """Each trial is an independent study+test episode with a fresh sample of n_images."""
    if n_images < 2:
        print(f"[SKIP] AFC serial order requires at least 2 images, got n_images={n_images}")
        return {}
    n_trials = n_trials if n_trials is not None else n_images

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        print(f"  Probing capacity for {n_images} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_images + 2):
            print(f"SKIP — model rejected {n_images} images in a single request")
            continue
        print("OK")

        results = []

        for i in range(n_trials):
            task = AFCSerialOrderMemoryTask(dataset_name=dataset, n_images=n_images, n_tests=1)
            trial_data = task.get_trials()
            test_trial = trial_data["test_phase"][0]

            messages = build_messages_afc(
                evaluator,
                trial_data["study_sequence"],
                trial_data["study_prompt"],
                test_trial["images"],
                test_trial["prompt"],
            )
            response_text = evaluator._call_api(messages)
            reported = parse_afc_response(response_text)
            correct = 1 if reported == test_trial["target"] else 0

            results.append({
                "trial": i,
                "target": test_trial["target"],
                "reported": reported,
                "correct": correct,
                "raw_response": response_text,
                "metadata": test_trial["metadata"],
            })
            status = "✓" if correct else "✗"
            print(f"  Trial {i+1}/{n_trials}: {status}", end="\r")

        metrics = calculate_afc_serial_order_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['total']})")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Serial Order Memory Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of images in study sequence")
    parser.add_argument("--variant", choices=["free", "afc"], default="afc",
                        help="free=report position 1-N; afc=which image came first (default: afc)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: n-images; resamples with replacement if larger)")

    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_serial_<variant>_<timestamp>.json)")
    args = parser.parse_args()

    evaluators = []
    for model in args.models:
        m = model.strip()
        if not m:
            continue
        if m == "gpt-4o":
            evaluators.append(OpenAIEvaluator("gpt-4o"))
        elif m == "claude":
            evaluators.append(AnthropicEvaluator())
        elif m == "gemini":
            evaluators.append(GoogleEvaluator())
        elif m == "qwen":
            evaluators.append(QwenEvaluator("Qwen/Qwen3-VL-8B-Instruct"))
        elif m == "molmo2":
            evaluators.append(Molmo2Evaluator("allenai/Molmo2-8B"))
        elif m.startswith("claude"):
            evaluators.append(AnthropicEvaluator(m))
        elif m.startswith("gemini"):
            evaluators.append(GoogleEvaluator(m))
        elif m.startswith("qwen") or m.startswith("Qwen"):
            evaluators.append(QwenEvaluator(m))
        elif m.startswith("molmo") or m.startswith("allenai"):
            evaluators.append(Molmo2Evaluator(m))
        else:
            evaluators.append(OpenAIEvaluator(m))

    if not evaluators:
        print("No valid models specified.")
        return

    print(f"Running Serial Order Memory ({args.variant}) evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images: {args.n_images}")
    print(f"  N trials: {args.n_trials or args.n_images}")
    print(f"  Variant: {args.variant}")

    print(f"  Dataset: {args.dataset}")

    if args.variant == "free":
        results = run_free_evaluation(evaluators, args.n_images, args.dataset, n_trials=args.n_trials)
        task_name = "Serial Order Memory (Free Report)"
    else:
        results = run_afc_evaluation(evaluators, args.n_images, args.dataset, n_trials=args.n_trials)
        task_name = "Serial Order Memory (AFC)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials = len(next(iter(results.values()))["trials"]) if results else 0
    output_data = {
        "_metadata": {
            "task": task_name,
            "variant": args.variant,
            "timestamp": timestamp,
            "dataset": args.dataset,
            "n_images": args.n_images,
            "n_trials": n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {"accuracy": results[model]["accuracy"]}
                for model in results
            },

        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_str = "+".join(e.get_name() for e in evaluators)
    default_name = f"results_serial_{args.variant}_{model_str}_n{args.n_images}_{args.dataset}.json"
    output_path = results_dir / (args.output if args.output else default_name)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
