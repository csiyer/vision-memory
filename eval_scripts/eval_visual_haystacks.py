#!/usr/bin/env python
"""
Visual Haystacks Evaluation

Task: Given a haystack of N images, answer questions about object presence.
- Single-needle: "For the image with [anchor], is there [target]?"
- Multi-needle: "For all/any images with [anchor], do all/any contain [target]?"

Usage:
    python -m eval_scripts.eval_visual_haystacks --models gpt-4o --n-images 10 --n-trials 20
    python -m eval_scripts.eval_visual_haystacks --models gemini --n-images 100 --n-trials 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from tasks.visual_haystacks import VisualHaystacksTaskSimple
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator


def build_messages(evaluator, images, prompt):
    """Build API messages for Visual Haystacks trial."""
    content = [{"type": "text", "text": "Here are a series of images:"}]

    for img in images:
        content.append(evaluator._encode_image(img))

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def parse_response(text):
    """Parse yes/no from response."""
    if text is None:
        return None
    text = text.lower().strip()
    if "yes" in text and "no" not in text:
        return "yes"
    elif "no" in text and "yes" not in text:
        return "no"
    elif text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    return None


def calculate_metrics(results):
    """Calculate accuracy and other metrics."""
    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_valid = sum(1 for r in results if r["response"] is not None)

    # Calculate by target presence
    target_present = [r for r in results if r.get("target_present", True)]
    target_absent = [r for r in results if not r.get("target_present", True)]

    tp_correct = sum(1 for r in target_present if r["correct"])
    ta_correct = sum(1 for r in target_absent if r["correct"])

    return {
        "accuracy": n_correct / n_total if n_total > 0 else 0,
        "valid_accuracy": n_correct / n_valid if n_valid > 0 else 0,
        "n_trials": n_total,
        "n_valid": n_valid,
        "n_correct": n_correct,
        "hit_rate": tp_correct / len(target_present) if target_present else 0,
        "correct_rejection_rate": ta_correct / len(target_absent) if target_absent else 0,
    }


def run_evaluation(evaluators, n_images=10, n_trials=20):
    """Run Visual Haystacks evaluation."""
    task = VisualHaystacksTaskSimple(n_images=n_images, n_trials=n_trials)
    trials = task.get_trials()

    all_results = {}

    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        results = []

        for i, trial in enumerate(trials):
            messages = build_messages(evaluator, trial["images"], trial["prompt"])
            response_text = evaluator._call_api(messages)
            response = parse_response(response_text)
            correct = response == trial["target"]

            results.append({
                "trial": i,
                "target": trial["target"],
                "response": response,
                "correct": correct,
                "target_present": trial["metadata"].get("target_present", True),
                "raw_response": response_text,
            })

            status = "✓" if correct else "✗"
            print(f"  Trial {i+1}/{len(trials)}: {status}", end="\r")

        metrics = calculate_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} | Hit: {metrics['hit_rate']:.1%} | CR: {metrics['correct_rejection_rate']:.1%}")

        all_results[evaluator.get_name()] = {
            "trials": results,
            **metrics,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Visual Haystacks Evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="Models to evaluate: gpt-4o, claude, gemini, qwen",
    )
    parser.add_argument(
        "--n-images", type=int, default=10, help="Number of images in haystack"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of trials to run"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path"
    )
    args = parser.parse_args()

    evaluators = []
    if "gpt-4o" in args.models:
        evaluators.append(OpenAIEvaluator("gpt-4o"))
    if "claude" in args.models:
        evaluators.append(AnthropicEvaluator())
    if "gemini" in args.models:
        evaluators.append(GoogleEvaluator())
    if "qwen" in args.models:
        evaluators.append(QwenEvaluator("Qwen/Qwen3-VL-8B-Instruct"))

    if not evaluators:
        print("No valid models specified.")
        return

    print(f"Running Visual Haystacks evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images (haystack): {args.n_images}")
    print(f"  N trials: {args.n_trials}")

    results = run_evaluation(evaluators, args.n_images, args.n_trials)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "Visual Haystacks",
            "timestamp": timestamp,
            "n_images": args.n_images,
            "n_trials": args.n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "hit_rate": results[model]["hit_rate"],
                    "correct_rejection_rate": results[model]["correct_rejection_rate"],
                }
                for model in results
            },
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        model_str = "+".join(e.get_name() for e in evaluators)
        output_path = results_dir / f"results_haystacks_{model_str}_n{args.n_images}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
