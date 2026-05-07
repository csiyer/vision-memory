#!/usr/bin/env python
"""
Associative Inference Evaluation

Task: Study A-B and B-C image pairs, then given cue image A choose which C
      image is transitively associated (via bridge B). Never saw A-C together.

Usage:
    python -m eval_scripts.eval_associative_inference --models gpt-4o --n-images 20
    python -m eval_scripts.eval_associative_inference --models gpt-4o claude gemini --n-images 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime
from tasks.associative_inference import AssociativeInferenceTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_associative_inference_metrics


def build_messages(evaluator, study_sequence, study_prompt, cue_image, test_images, test_prompt):
    """Build API messages for associative inference trial.

    Study sequence is a list of dicts with keys 'images' (list of 2) and 'pair_type'.
    Each study event shows two images side by side.
    """
    study_content = [{"type": "text", "text": study_prompt}]
    for event in study_sequence:
        study_content.append({"type": "text", "text": f"({event['pair_type']} pair)"})
        for img in event["images"]:
            study_content.append(evaluator._encode_image(img))

    test_content = [
        {"type": "text", "text": "Cue image:"},
        evaluator._encode_image(cue_image),
        {"type": "text", "text": test_prompt},
    ]
    for img in test_images:
        test_content.append(evaluator._encode_image(img))

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the image pairs."},
        {"role": "user", "content": test_content},
    ]


def parse_response(text):
    """Parse 1 or 2 from response, taking the last standalone digit to handle verbose responses."""
    if text is None:
        return -1
    text = text.strip()

    # Find all standalone 1s and 2s (not part of larger numbers), take the last one
    matches = [(m.start(), int(m.group())) for m in re.finditer(r"(?<!\d)[12](?!\d)", text)]
    if matches:
        return max(matches, key=lambda x: x[0])[1]

    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


def run_evaluation(evaluators, n_images=20, dataset="things", n_trials=None):
    """Each trial is an independent study+test episode with a fresh sample of n_images chains."""
    n_trials = n_trials if n_trials is not None else n_images // 2

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        print(f"  Probing capacity for {n_images} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_images * 2 + 3):
            print(f"SKIP — model rejected {n_images} images in a single request")
            continue
        print("OK")

        results = []

        for i in range(n_trials):
            task = AssociativeInferenceTask(dataset_name=dataset, n_trials=n_images)
            trial_data = task.get_trials()
            test_trial = trial_data["test_phase"][0]

            messages = build_messages(
                evaluator,
                trial_data["study_sequence"],
                trial_data["study_prompt"],
                test_trial["cue_image"],
                test_trial["images"],
                test_trial["prompt"],

            )
            response_text = evaluator._call_api(messages)
            reported = parse_response(response_text)
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

        metrics = calculate_associative_inference_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['total']})")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Associative Inference Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of trials (must be even; sets number of A-B and B-C pairs = n/2 each)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: n-images/2 chains; resamples with replacement if larger)")

    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_assoc_<timestamp>.json)")
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

    # n_images must be even for the task
    if args.n_images % 2 != 0:
        args.n_images += 1
        print(f"  Note: n_images rounded up to {args.n_images} (must be even)")

    print(f"Running Associative Inference evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N chains: {args.n_images // 2} ({args.n_images} study pairs)")
    print(f"  N trials: {args.n_trials or args.n_images // 2}")

    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(evaluators, args.n_images, args.dataset, n_trials=args.n_trials)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials = len(next(iter(results.values()))["trials"]) if results else 0
    output_data = {
        "_metadata": {
            "task": "Associative Inference",
            "timestamp": timestamp,
            "dataset": args.dataset,
            "n_images": args.n_images,
            "n_trials": n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "n_correct": results[model]["n_correct"],
                    "total": results[model]["total"],
                }
                for model in results
            },

        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_str = "+".join(e.get_name() for e in evaluators)
    output_path = results_dir / (args.output if args.output else f"results_assoc_{model_str}_n{args.n_images}_{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
