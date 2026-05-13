#!/usr/bin/env python
"""
Color Memory Evaluation

Two variants:
  continuous — study N color objects, then for each gray probe report the hue
               angle in degrees on a CIELAB color wheel (0-360)
  named      — study N color objects, then for each gray probe choose one of
               6 named colors (red, orange, yellow, green, blue, purple)

Usage:
    python -m eval_scripts.eval_color_memory --models gpt-4o --n-images 10 --variant continuous
    python -m eval_scripts.eval_color_memory --models gpt-4o claude gemini --n-images 50 --variant named
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime
from tasks.color_memory import ColorMemoryTask, NAMED_COLORS
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_color_metrics, calculate_named_color_metrics
from src.plotting import default_plots_dir, plot_color_memory


VARIANT_TO_MODE = {
    "continuous": "continuous_color_report",
    "named": "named",
}

NAMED_COLOR_NAMES = [name for name, _ in NAMED_COLORS]


def build_messages_continuous(evaluator, study_sequence, study_prompt,
                              test_image, test_prompt, color_wheel):
    """Continuous variant: include the color wheel image alongside the gray probe."""
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_sequence:
        study_content.append(evaluator._encode_image(img))

    test_content = [
        {"type": "text", "text": test_prompt},
        evaluator._encode_image(test_image),
        {"type": "text", "text": "Color wheel:"},
        evaluator._encode_image(color_wheel),
    ]

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the colored items."},
        {"role": "user", "content": test_content},
    ]


def build_messages_named(evaluator, study_sequence, study_prompt,
                         test_image, test_prompt):
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_sequence:
        study_content.append(evaluator._encode_image(img))

    test_content = [
        {"type": "text", "text": test_prompt},
        evaluator._encode_image(test_image),
    ]

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the colored items."},
        {"role": "user", "content": test_content},
    ]


def parse_angle_response(text):
    """Parse a hue angle 0-360 from response. Take the last in-range number."""
    if text is None:
        return -1
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Match floats and ints; keep those in [0, 360]
    candidates = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", text):
        try:
            val = float(m.group())
        except ValueError:
            continue
        if 0 <= val <= 360:
            candidates.append((m.start(), val))

    if candidates:
        return float(max(candidates, key=lambda x: x[0])[1]) % 360.0
    return -1


def parse_named_response(text):
    """Parse one of the 6 named colors from response."""
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lower = text.lower()

    # First named color word that appears
    matches = []
    for name in NAMED_COLOR_NAMES:
        for m in re.finditer(rf"\b{name}\b", lower):
            matches.append((m.start(), name))
    if matches:
        # Take the last reference — verbose responses tend to end with the answer
        return max(matches, key=lambda x: x[0])[1]
    return lower.strip().split()[0] if lower.strip() else ""


def run_continuous_evaluation(evaluators, n_images=10, n_trials=None):
    """Each trial is an independent study+test episode."""
    n_trials = n_trials if n_trials is not None else n_images

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        # +2 for probe + color wheel
        print(f"  Probing capacity for {n_images} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_images + 2):
            print(f"SKIP — model rejected {n_images} images in a single request")
            continue
        print("OK")

        results = []

        for i in range(n_trials):
            task = ColorMemoryTask(n_images=n_images, mode="continuous_color_report")
            trial_data = task.get_trials()
            test_trial = trial_data["test_phase"][0]

            messages = build_messages_continuous(
                evaluator,
                trial_data["study_sequence"],
                trial_data["study_prompt"],
                test_trial["image"],
                test_trial["prompt"],
                test_trial["color_wheel"],
            )
            response_text = evaluator._call_api(messages)
            reported = parse_angle_response(response_text)

            results.append({
                "trial": i,
                "target": test_trial["target"],
                "reported": reported,
                "raw_response": response_text,
                "metadata": test_trial["metadata"],
            })
            err = "?" if reported == -1 else f"{abs(((reported - test_trial['target'] + 180) % 360) - 180):.1f}°"
            print(f"  Trial {i+1}/{n_trials}: target={test_trial['target']:.1f}°, got={reported}, err={err}", end="\r")

        valid = [(r["reported"], r["target"]) for r in results if r["reported"] != -1]
        if valid:
            reported_vals = [v[0] for v in valid]
            target_vals = [v[1] for v in valid]
            metrics = calculate_color_metrics(reported_vals, target_vals)
        else:
            metrics = {"average_abs_error": 0, "guess_rate_heuristic": 0, "precision_heuristic": 0}
        metrics["n_parse_failures"] = sum(1 for r in results if r["reported"] == -1)
        metrics["total"] = len(results)
        print(f"\n  Avg abs error: {metrics['average_abs_error']:.1f}° | guess rate: {metrics['guess_rate_heuristic']:.2f}")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def run_named_evaluation(evaluators, n_images=10, n_trials=None):
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
            task = ColorMemoryTask(n_images=n_images, mode="named")
            trial_data = task.get_trials()
            test_trial = trial_data["test_phase"][0]

            messages = build_messages_named(
                evaluator,
                trial_data["study_sequence"],
                trial_data["study_prompt"],
                test_trial["image"],
                test_trial["prompt"],
            )
            response_text = evaluator._call_api(messages)
            reported = parse_named_response(response_text)
            correct = 1 if reported == str(test_trial["target"]).lower() else 0

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

        reported_vals = [r["reported"] for r in results]
        target_vals = [r["target"] for r in results]
        metrics = calculate_named_color_metrics(reported_vals, target_vals)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['total']})")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Color Memory Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2")
    parser.add_argument("--n-images", type=int, default=10,
                        help="Number of items in study sequence")
    parser.add_argument("--variant", choices=["continuous", "named"], default="continuous",
                        help="continuous=hue angle 0-360; named=one of 6 color names")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: n-images)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_color_<variant>_<...>.json)")
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

    print(f"Running Color Memory ({args.variant}) evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images: {args.n_images}")
    print(f"  N trials: {args.n_trials or args.n_images}")
    print(f"  Variant: {args.variant}")

    if args.variant == "continuous":
        results = run_continuous_evaluation(evaluators, args.n_images, n_trials=args.n_trials)
        task_name = "Color Memory (Continuous)"
    else:
        results = run_named_evaluation(evaluators, args.n_images, n_trials=args.n_trials)
        task_name = "Color Memory (Named)"

    if not results:
        print("No results produced — not writing output file.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials_actual = len(next(iter(results.values()))["trials"]) if results else 0

    if args.variant == "continuous":
        summary = {
            model: {
                "accuracy": results[model].get("accuracy"),
                "average_abs_error": results[model].get("average_abs_error"),
                "guess_rate_heuristic": results[model].get("guess_rate_heuristic"),
                "precision_heuristic": results[model].get("precision_heuristic"),
            }
            for model in results
        }
    else:
        summary = {
            model: {
                "accuracy": results[model]["accuracy"],
                "n_correct": results[model]["n_correct"],
                "total": results[model]["total"],
            }
            for model in results
        }

    output_data = {
        "_metadata": {
            "task": task_name,
            "variant": args.variant,
            "timestamp": timestamp,
            "dataset": "Brady2013ColorObjects",
            "n_images": args.n_images,
            "n_trials": n_trials_actual,
            "models": [e.get_name() for e in evaluators],
            "summary": summary,
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_str = "+".join(e.get_name() for e in evaluators)
    default_name = f"results_color_{args.variant}_{model_str}_n{args.n_images}.json"
    output_path = results_dir / (args.output if args.output else default_name)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot and args.variant == "continuous":
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        fig_path = plot_color_memory(output_data, output_dir=plot_dir)
        print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
