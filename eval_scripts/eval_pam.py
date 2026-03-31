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
from metrics import calculate_pam_metrics
from plotting import default_plots_dir, plot_pam


def build_messages(evaluator, study_sequence, study_prompt, test_image, test_prompt, max_context_pairs=None):
    """Build API messages for paired associate memory trial with simulated acknowledgment.

    Args:
        evaluator: The evaluator instance
        study_sequence: List of (image, word) tuples
        study_prompt: Prompt for study phase
        test_image: The test image
        test_prompt: Prompt for test phase
        max_context_pairs: Max pairs to send in context. If None, uses provider defaults.
    """
    # Determine max study pairs based on provider limits
    if "Anthropic" in type(evaluator).__name__:
        provider_limit = 99 - 1  # Reserve space for test image
    else:
        provider_limit = len(study_sequence)  # No limit for other providers

    if max_context_pairs is not None:
        max_study = min(max_context_pairs, provider_limit)
    else:
        max_study = provider_limit

    if len(study_sequence) > max_study:
        study_sequence = study_sequence[:max_study]

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

    # Simple extraction: look for the word in the response
    # This handles cases like "The word was: APPLE" or just "APPLE"
    words = text.split()

    # If response is a single word, return it
    if len(words) == 1:
        return words[0].lower().strip('.,;:!?"\'')

    # Look for the target word in the response (case insensitive)
    text_lower = text.lower()
    if target_word.lower() in text_lower:
        return target_word.lower()

    # Try to extract last capitalized word or last word
    for word in reversed(words):
        cleaned = word.strip('.,;:!?"\'')
        if cleaned:
            return cleaned.lower()

    return text.lower()


def run_evaluation(evaluators, n_images=20, dataset='things', max_context_pairs=None):
    """Run paired associate memory evaluation on all evaluators.

    Args:
        evaluators: List of evaluator instances
        n_images: Number of image-word pairs to study
        dataset: Dataset to use ('things', 'Brady2008')
        max_context_pairs: Max pairs to send in context per trial. If None, sends all pairs.
    """
    task = PairedAssociateMemoryTask(dataset_name=dataset, n_images=n_images)
    trial_data = task.get_trials()

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        # Warn if pairs will be truncated
        actual_context = max_context_pairs if max_context_pairs else len(trial_data['study_sequence'])
        if "Anthropic" in type(evaluator).__name__:
            actual_context = min(actual_context, 98)
        if actual_context < len(trial_data['study_sequence']):
            print(f"  Note: Sending {actual_context} of {len(trial_data['study_sequence'])} pairs to context")

        results = []

        for i, test_trial in enumerate(trial_data['test_phase']):
            messages = build_messages(
                evaluator,
                trial_data['study_sequence'],
                trial_data['study_prompt'],
                test_trial['image'],
                test_trial['prompt'],
                max_context_pairs=max_context_pairs
            )
            response_text = evaluator._call_api(messages)
            reported_word = parse_word_response(response_text, test_trial['target'])

            # Check if correct (case-insensitive comparison)
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
            print(f"  Trial {i+1}/{len(trial_data['test_phase'])}: {status} (target: {test_trial['target']}, got: {reported_word})", end="\r")

        # Calculate metrics
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
                        help="Models to evaluate: gpt-4o, claude, gemini")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of image-word pairs to study")
    parser.add_argument("--max-context-pairs", type=int, default=None,
                        help="Max pairs to send in context per trial (default: all pairs)")
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

    if not evaluators:
        print("No valid models specified. Use --models gpt-4o claude gemini")
        return

    print(f"Running Paired Associate Memory evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N pairs: {args.n_images}")
    print(f"  Max context pairs: {args.max_context_pairs or 'all'}")
    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(
        evaluators,
        args.n_images,
        args.dataset,
        max_context_pairs=args.max_context_pairs
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
            "max_context_pairs": args.max_context_pairs,
        },
        **results,
    }

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        output_path = results_dir / f"results_pam_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        fig_path = plot_pam(output_data, output_dir=plot_dir)
        print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
