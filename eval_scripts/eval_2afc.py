#!/usr/bin/env python
"""
2-AFC Recognition Memory Evaluation

Task: Study N images, then for each test trial show 2 images (original + foil)
      and ask which was in the study sequence.

Usage:
    python -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 100 --foil-type novel
    python -m eval_scripts.eval_2afc --models gpt-4o --n-images 5  # pilot run
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime
from tasks.afc_recognition import AFCRecognitionTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from src.metrics import calculate_2afc_metrics
from src.plotting import default_plots_dir, plot_2afc_all


def build_messages(evaluator, study_images, study_prompt, test_images, test_prompt, max_context_images=None):
    """Build API messages for 2-AFC trial with simulated acknowledgment.

    Args:
        evaluator: The evaluator instance
        study_images: List of study phase images
        study_prompt: Prompt for study phase
        test_images: List of test images (usually 2 for 2-AFC)
        test_prompt: Prompt for test phase
        max_context_images: Max images to send in context. If None, uses provider defaults.
    """
    # Determine max study images based on provider limits and user config
    if "Anthropic" in type(evaluator).__name__:
        provider_limit = 100 - len(test_images)  # Reserve space for test images
    else:
        provider_limit = len(study_images)  # No limit for other providers

    if max_context_images is not None:
        max_study = min(max_context_images, provider_limit)
    else:
        max_study = provider_limit

    if len(study_images) > max_study:
        study_images = study_images[:max_study]

    # Study phase
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_images:
        study_content.append(evaluator._encode_image(img))

    # Test phase
    test_content = [{"type": "text", "text": test_prompt}]
    for img in test_images:
        test_content.append(evaluator._encode_image(img))

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content}
    ]


def parse_response(text):
    """Parse 1 or 2 from response (digits or common paraphrases like 'first' / 'second')."""
    if text is None:
        return -1
    text = text.strip()
    if "1" in text and "2" not in text:
        return 1
    elif "2" in text and "1" not in text:
        return 2
    elif text.startswith("1"):
        return 1
    elif text.startswith("2"):
        return 2
    # Models often answer "The first image..." — no digit `1` in "first"
    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


def run_evaluation(evaluators, n_images=20, n_trials=None, foil_type='novel', dataset='things', max_context_images=None):
    """Run 2-AFC evaluation on all evaluators.

    Args:
        evaluators: List of evaluator instances
        n_images: Number of images in study sequence
        n_trials: Number of test trials (defaults to n_images if not specified)
        foil_type: Type of foils ('novel', 'exemplar', 'state', 'all')
        dataset: Dataset to use ('things', 'Brady2008')
        max_context_images: Max images to send in context per trial. If None, sends all study images.
    """
    task = AFCRecognitionTask(dataset_name=dataset, n_images=n_images, n_trials=n_trials, foil_type=foil_type)
    trial_data = task.get_trials()

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        # Warn if images will be truncated
        actual_context = max_context_images if max_context_images else len(trial_data['study_sequence'])
        if "Anthropic" in type(evaluator).__name__:
            actual_context = min(actual_context, 98)
        if actual_context < len(trial_data['study_sequence']):
            print(f"  Note: Sending {actual_context} of {len(trial_data['study_sequence'])} study images to context")

        results = []

        for i, test_trial in enumerate(trial_data['test_phase']):
            messages = build_messages(
                evaluator,
                trial_data['study_sequence'],
                trial_data['study_prompt'],
                test_trial['images'],
                test_trial['prompt'],
                max_context_images=max_context_images
            )
            response_text = evaluator._call_api(messages)
            response = parse_response(response_text)
            correct = 1 if response == test_trial['target'] else 0

            results.append({
                'trial': i,
                'target': test_trial['target'],
                'response': response,
                'correct': correct,
                'foil_type': test_trial['type'],
                'raw_response': response_text
            })
            status = '✓' if correct else '✗'
            print(f"  Trial {i+1}/{len(trial_data['test_phase'])}: {status}", end="\r")

        # Calculate metrics using the new function
        metrics = calculate_2afc_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} | d': {metrics['d_prime']:.2f} | Mem Score: {metrics['mem_score']:.2f}")

        all_results[evaluator.get_name()] = {
            'trials': results,
            **metrics
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="2-AFC Recognition Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen (or provider model IDs)")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of images in study sequence")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: same as n-images)")
    parser.add_argument("--max-context-images", type=int, default=None,
                        help="Max images to send in context per trial (default: all study images)")
    parser.add_argument("--foil-type", choices=["novel", "exemplar", "state", "all"], default="novel",
                        help="Type of foils to use")
    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_2afc_<timestamp>.json)")
    parser.add_argument("--plot", action="store_true",
                        help="Write figures under output/plots (or --plot-dir)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for figures (default: repo output/plots)")
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
        elif m.startswith("claude"):
            evaluators.append(AnthropicEvaluator(m))
        elif m.startswith("gemini"):
            evaluators.append(GoogleEvaluator(m))
        elif m.startswith("qwen") or m.startswith("Qwen"):
            evaluators.append(QwenEvaluator(m))
        else:
            evaluators.append(OpenAIEvaluator(m))

    if not evaluators:
        print("No valid models specified. Use --models gpt-4o claude gemini qwen")
        return

    n_trials = args.n_trials if args.n_trials is not None else args.n_images

    print(f"Running 2-AFC evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images (study): {args.n_images}")
    print(f"  N trials (test): {n_trials}")
    print(f"  Max context images: {args.max_context_images or 'all'}")
    print(f"  Foil type: {args.foil_type}")
    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(
        evaluators,
        args.n_images,
        n_trials,
        args.foil_type,
        args.dataset,
        max_context_images=args.max_context_images
    )

    # Build output with metadata at the top
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "2-AFC Recognition",
            "timestamp": timestamp,
            "dataset": args.dataset,
            "n_images": args.n_images,
            "n_trials": n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "d_prime": results[model]["d_prime"],
                    "mem_score": results[model]["mem_score"],
                }
                for model in results
            },
            "max_context_images": args.max_context_images,
            "foil_type": args.foil_type,
        },
        **results,
    }

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        model_str = "+".join(e.get_name() for e in evaluators)
        output_path = results_dir / f"results_2afc_{model_str}_n{args.n_images}_{args.dataset}_{args.foil_type}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        p_acc, p_met = plot_2afc_all(output_data, output_dir=plot_dir)
        print(f"Plots saved to {p_acc} and {p_met}")


if __name__ == "__main__":
    main()
