#!/usr/bin/env python
"""
Continuous Recognition Memory Evaluation

Task: Show images sequentially. For each image, ask if it has appeared before.

Usage:
    python -m eval_scripts.eval_continuous --models gpt-4o claude gemini --n-images 50
    python -m eval_scripts.eval_continuous --models gpt-4o --n-images 10  # pilot run
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from tasks.continuous_recognition import ContinuousRecognitionTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from metrics import calculate_metrics


def parse_yes_no(text):
    """Parse yes/no from response."""
    text = text.lower().strip()
    if "yes" in text:
        return 1
    elif "no" in text:
        return 0
    return -1


def run_evaluation(evaluators, n_images=50, dataset='things'):
    """Run continuous recognition evaluation on all evaluators."""
    task = ContinuousRecognitionTask(dataset_name=dataset, n_images=n_images)
    trials = task.get_trials()

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        history = []
        results = []

        for i, trial in enumerate(trials):
            # Build message for this trial
            content = [
                {"type": "text", "text": trial['prompt']},
                evaluator._encode_image(trial['image'])
            ]
            history.append({"role": "user", "content": content})

            # Call API with full history
            response_text = evaluator._call_api(history)
            history.append({"role": "assistant", "content": response_text})

            response = parse_yes_no(response_text)
            results.append({
                'trial': i,
                'target': trial['target'],
                'response': response,
                'delay': trial['metadata'].get('delay'),
                'raw_response': response_text
            })
            print(f"  Trial {i+1}/{len(trials)}", end="\r")

        # Calculate metrics
        metrics = calculate_metrics(
            [r['response'] for r in results],
            [r['target'] for r in results]
        )
        print(f"\n  d'={metrics['d_prime']:.2f}, hit={metrics['hit_rate']:.2f}, fa={metrics['false_alarm_rate']:.2f}")

        all_results[evaluator.get_name()] = {
            'trials': results,
            'metrics': metrics
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Continuous Recognition Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini")
    parser.add_argument("--n-images", type=int, default=50,
                        help="Number of unique images")
    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_continuous_<timestamp>.json)")
    args = parser.parse_args()

    evaluators = []
    if "gpt-4o" in args.models:
        evaluators.append(OpenAIEvaluator("gpt-4o"))
    if "claude" in args.models:
        evaluators.append(AnthropicEvaluator())
    if "gemini" in args.models:
        evaluators.append(GoogleEvaluator())

    if not evaluators:
        print("No valid models specified. Use --models gpt-4o claude gemini")
        return

    print(f"Running continuous recognition evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images: {args.n_images}")
    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(evaluators, args.n_images, args.dataset)

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"results_continuous_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
