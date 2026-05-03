#!/usr/bin/env python
"""
Paired Associate Memory Evaluation

Task: Study N image-word pairs, then for each test trial show an image
      and ask which word was paired with it.

Usage:
    python -m eval_scripts.eval_pam --models gpt-4o --n-images 20
    python -m eval_scripts.eval_pam --models gpt-4o claude gemini --n-images 50
    python -m eval_scripts.eval_pam --models gpt-4o --n-images 5  # pilot run
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from tasks.paired_associate_memory import PairedAssociateMemoryTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_pam_metrics
from src.plotting import default_plots_dir, plot_pam


def build_messages(evaluator, study_sequence, study_prompt, test_image, test_prompt):
    """Build API messages for paired associate memory trial with simulated acknowledgment."""

    # Study phase: alternate images and words
    study_content = [{"type": "text", "text": study_prompt}]
    for img, word in study_sequence:
        study_content.append(evaluator._encode_image(img))
        study_content.append({"type": "text", "text": f"Word: {word}"})

    # Test phase
    test_content = [
        {"type": "text", "text": test_prompt},
        evaluator._encode_image(test_image)
    ]

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the image-word pairs."},
        {"role": "user", "content": test_content}
    ]


def parse_word_response(text, target_word):
    """Parse word from response text.

    Args:
        text: Response text from model
        target_word: The correct target word

    Returns:
        The extracted word (lowercased and stripped)
    """
    if text is None:
        return ""

    text = text.strip()

    # Strip Qwen3 thinking blocks (<think>...</think>)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    words = text.split()

    # If response is a single word, return it
    if len(words) == 1:
        return words[0].lower().strip('.,;:!?"\'')

    # Look for the target word in the response (case insensitive)
    if target_word.lower() in text.lower():
        return target_word.lower()

    # Wordpool words are ALL_CAPS — prefer the first all-caps token in the response
    for word in words:
        cleaned = word.strip('.,;:!?"\'')
        if cleaned and cleaned.isupper() and len(cleaned) > 1:
            return cleaned.lower()

    # Last resort: return the last non-empty token
    for word in reversed(words):
        cleaned = word.strip('.,;:!?"\'')
        if cleaned:
            return cleaned.lower()

    return text.lower()


def run_evaluation(evaluators, n_images=20, dataset='things', n_trials=None):
    """Run paired associate memory evaluation on all evaluators.

    Each trial is an independent study+test episode: a fresh set of n_images
    image-word pairs is sampled, studied, then tested (one test question per episode).
    n_trials controls how many independent episodes are run.
    """
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
            task = PairedAssociateMemoryTask(dataset_name=dataset, n_images=n_images)
            trial_data = task.get_trials()
            # Each episode: study all pairs, test on one randomly chosen pair
            test_trial = trial_data['test_phase'][0]

            messages = build_messages(
                evaluator,
                trial_data['study_sequence'],
                trial_data['study_prompt'],
                test_trial['image'],
                test_trial['prompt'],
            )
            response_text = evaluator._call_api(messages)
            reported_word = parse_word_response(response_text, test_trial['target'])

            correct = 1 if reported_word.lower() == test_trial['target'].lower() else 0

            results.append({
                'trial': i,
                'target': test_trial['target'],
                'reported': reported_word,
                'correct': correct,
                'raw_response': response_text,
                'metadata': test_trial['metadata']
            })
            status = '✓' if correct else '✗'
            print(f"  Trial {i+1}/{n_trials}: {status} (target: {test_trial['target']}, got: {reported_word})", end="\r")

        metrics = calculate_pam_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['total']})")

        all_results[evaluator.get_name()] = {
            'trials': results,
            **metrics
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Paired Associate Memory Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of image-word pairs to study")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: n-images; resamples with replacement if larger)")

    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_pam_<timestamp>.json)")
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
    if "molmo2" in args.models:
        evaluators.append(Molmo2Evaluator("allenai/Molmo2-8B"))

    if not evaluators:
        print("No valid models specified. Use --models gpt-4o claude gemini qwen molmo2")
        return

    print(f"Running Paired Associate Memory evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N pairs: {args.n_images}")
    print(f"  N trials: {args.n_trials or args.n_images}")

    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(
        evaluators,
        args.n_images,
        args.dataset,
        n_trials=args.n_trials,
    )

    # Build output with metadata at the top (base keys align with eval_continuous / eval_2afc)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials = len(next(iter(results.values()))["trials"]) if results else 0
    output_data = {
        "_metadata": {
            "task": "Paired Associate Memory",
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

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        model_str = "+".join(e.get_name() for e in evaluators)
        output_path = results_dir / f"results_pam_{model_str}_n{args.n_images}_{args.dataset}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        fig_path = plot_pam(output_data, output_dir=plot_dir)
        print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
