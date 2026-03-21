#!/usr/bin/env python
"""
2-AFC Recognition Memory Evaluation

Task: Study N images, then for each test trial show 2 images (original + foil)
      and ask which was in the study sequence.

Usage:
    python -m eval_scripts.eval_2afc --models gpt-4o claude gemini --n-images 20 --foil-type novel
    python -m eval_scripts.eval_2afc --models gpt-4o --n-images 5  # pilot run
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from tasks.afc_recognition import AFCRecognitionTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator


def build_messages(evaluator, study_images, study_prompt, test_images, test_prompt):
    """Build API messages for 2-AFC trial with simulated acknowledgment."""
    # Study phase
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_images:
        study_content.append(evaluator._encode_image(img))

    # Test phase (2 images)
    test_content = [{"type": "text", "text": test_prompt}]
    for img in test_images:
        test_content.append(evaluator._encode_image(img))

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content}
    ]


def parse_response(text):
    """Parse 1 or 2 from response."""
    text = text.strip()
    if "1" in text and "2" not in text:
        return 1
    elif "2" in text and "1" not in text:
        return 2
    elif text.startswith("1"):
        return 1
    elif text.startswith("2"):
        return 2
    return -1


def run_evaluation(evaluators, n_images=20, foil_type='novel', dataset='things'):
    """Run 2-AFC evaluation on all evaluators."""
    task = AFCRecognitionTask(dataset_name=dataset, n_images=n_images, foil_type=foil_type)
    trial_data = task.get_trials()

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        results = []

        for i, test_trial in enumerate(trial_data['test_phase']):
            messages = build_messages(
                evaluator,
                trial_data['study_sequence'],
                trial_data['study_prompt'],
                test_trial['images'],
                test_trial['prompt']
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

        accuracy = sum(r['correct'] for r in results) / len(results)
        print(f"\n  Accuracy: {accuracy:.1%}")
        all_results[evaluator.get_name()] = {'trials': results, 'accuracy': accuracy}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="2-AFC Recognition Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of images in study sequence")
    parser.add_argument("--foil-type", choices=["novel", "exemplar", "state", "all"], default="novel",
                        help="Type of foils to use")
    parser.add_argument("--dataset", choices=["things", "Brady2008"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_2afc_<timestamp>.json)")
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

    print(f"Running 2-AFC evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images: {args.n_images}")
    print(f"  Foil type: {args.foil_type}")
    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(evaluators, args.n_images, args.foil_type, args.dataset)

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"results_2afc_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
