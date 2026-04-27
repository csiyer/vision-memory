#!/usr/bin/env python
"""
Continuous Recognition Memory Evaluation

Task: Show images sequentially. For each image, ask if it has appeared before.

Usage:
    python -m eval_scripts.eval_continuous --models gpt-4o claude gemini --n-images 50
    python -m eval_scripts.eval_continuous --models gpt-4o --n-images 100 --n-trials 100
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
from evaluators.qwen_evaluator import QwenEvaluator
from src.metrics import calculate_metrics, calculate_hit_rate_by_delay
from src.plotting import default_plots_dir, plot_continuous_recognition


def parse_yes_no(text):
    """Parse yes/no from response."""
    if text is None:
        return -1
    text = text.lower().strip()
    if "yes" in text:
        return 1
    elif "no" in text:
        return 0
    return -1


def run_evaluation(evaluators, n_images=50, dataset='things'):
    """Run continuous recognition evaluation on all evaluators.

    Returns:
        tuple: (all_results, task_info) where task_info has n_trials and task hyperparameters.
    """
    task = ContinuousRecognitionTask(dataset_name=dataset, n_images=n_images)
    trials = task.get_trials()
    task_info = {
        "n_trials": len(trials),
        "n_images": n_images,
        "p_old": task.p_old,
        "min_delay": task.min_delay,
        "max_delay": task.max_delay,
        "dataset": dataset,
    }

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
                'raw_response': response_text,
                'metadata': trial['metadata'],
            })
            print(f"  Trial {i+1}/{len(trials)}", end="\r")

        # Calculate metrics (top-level keys match 2-AFC / PAM result files for easier comparison)
        metrics = calculate_metrics(
            [r['response'] for r in results],
            [r['target'] for r in results]
        )
        delays = [r['metadata'].get('delay') for r in results]
        hit_by_delay = calculate_hit_rate_by_delay(
            [r['response'] for r in results],
            [r['target'] for r in results],
            delays,
        )
        metrics['hit_rate_by_delay'] = {str(k): v for k, v in hit_by_delay.items()}
        print(f"\n  d'={metrics['d_prime']:.2f}, hit={metrics['hit_rate']:.2f}, fa={metrics['false_alarm_rate']:.2f}")

        all_results[evaluator.get_name()] = {
            'trials': results,
            **metrics,
        }

    return all_results, task_info


def main():
    parser = argparse.ArgumentParser(description="Continuous Recognition Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen")
    parser.add_argument("--n-images", type=int, default=50,
                        help="Number of unique images")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Ignored; sequence length is derived from n-images and task p_old (kept for CLI compatibility)")
    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_continuous_<timestamp>.json)")
    parser.add_argument("--plot", action="store_true",
                        help="Write figures under output/plots (or --plot-dir)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for figures (default: repo output/plots)")
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
        print("No valid models specified. Use --models gpt-4o claude gemini qwen")
        return

    print(f"Running continuous recognition evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images: {args.n_images}")
    print(f"  Dataset: {args.dataset}")

    results, task_info = run_evaluation(evaluators, args.n_images, args.dataset)
    print(f"  N trials (generated): {task_info['n_trials']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "Continuous Recognition",
            "timestamp": timestamp,
            "dataset": task_info["dataset"],
            "n_images": task_info["n_images"],
            "n_trials": task_info["n_trials"],
            "models": [ev.get_name() for ev in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "d_prime": results[model]["d_prime"],
                    "hit_rate": results[model]["hit_rate"],
                    "false_alarm_rate": results[model]["false_alarm_rate"],
                    "weighted_f1": results[model]["weighted_f1"],
                }
                for model in results
            },
            "p_old": task_info["p_old"],
            "min_delay": task_info["min_delay"],
            "max_delay": task_info["max_delay"],
        },
        **results,
    }

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        model_str = "+".join(ev.get_name() for ev in evaluators)
        output_path = results_dir / f"results_continuous_{model_str}_n{args.n_images}_{args.dataset}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        fig_path = plot_continuous_recognition(output_data, output_dir=plot_dir)
        print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
