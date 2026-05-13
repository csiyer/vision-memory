#!/usr/bin/env python
"""
Mnemonic Similarity Task (MST) Evaluation

Task: Study N images. Then, for each test probe, classify it as:
      old      — exactly as seen before
      similar  — like something seen but not identical (lure)
      new      — not seen before (foil)

Usage:
    python -m eval_scripts.eval_mst --models gpt-4o --n-study 32
    python -m eval_scripts.eval_mst --models gpt-4o claude gemini --n-study 64 --n-trials 100
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import re
from datetime import datetime
from tasks.mnemonic_similarity import MnemonicSimilarityTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_mst_metrics


def build_messages(evaluator, study_content_cache, study_prompt, study_sequence,
                   test_image, test_prompt):
    """Build messages with a cached study payload (encoded once per session)."""
    if study_content_cache["content"] is None:
        content = [{"type": "text", "text": study_prompt}]
        for img in study_sequence:
            content.append(evaluator._encode_image(img))
        study_content_cache["content"] = content

    test_content = [
        {"type": "text", "text": test_prompt},
        evaluator._encode_image(test_image),
    ]

    return [
        {"role": "user", "content": study_content_cache["content"]},
        {"role": "assistant", "content": "I have studied the images."},
        {"role": "user", "content": test_content},
    ]


def parse_response(text):
    """Parse 'old' / 'similar' / 'new' from response."""
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lower = text.lower().strip()

    matches = []
    for label in ("old", "similar", "new"):
        for m in re.finditer(rf"\b{label}\b", lower):
            matches.append((m.start(), label))
    if matches:
        # Take the last reference — verbose responses tend to end with the answer
        return max(matches, key=lambda x: x[0])[1]
    return lower.split()[0] if lower else ""


def stratified_sample(test_phase, n):
    """Sample n items balanced across target/lure/foil while keeping order random."""
    by_type = {"target": [], "lure": [], "foil": []}
    for item in test_phase:
        by_type.setdefault(item["type"], []).append(item)
    for items in by_type.values():
        random.shuffle(items)

    n_per = n // 3
    selected = []
    for t in ("target", "lure", "foil"):
        selected.extend(by_type[t][:n_per])
    # Fill remainder round-robin
    remaining = n - len(selected)
    if remaining > 0:
        leftover = []
        for t in ("target", "lure", "foil"):
            leftover.extend(by_type[t][n_per:])
        random.shuffle(leftover)
        selected.extend(leftover[:remaining])
    random.shuffle(selected)
    return selected


def run_evaluation(evaluators, n_study=32, n_trials=None, source="local"):
    """Single study session per evaluator; iterate over (subsampled) test items."""
    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        # +1 for probe image
        print(f"  Probing capacity for {n_study} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_study + 1):
            print(f"SKIP — model rejected {n_study} images in a single request")
            continue
        print("OK")

        task = MnemonicSimilarityTask(n_study=n_study, source=source)
        trial_data = task.get_trials()

        test_items = trial_data["test_phase"]
        if n_trials is not None and n_trials < len(test_items):
            test_items = stratified_sample(test_items, n_trials)

        study_content_cache = {"content": None}
        results = []

        for i, test_item in enumerate(test_items):
            messages = build_messages(
                evaluator,
                study_content_cache,
                trial_data["study_prompt"],
                trial_data["study_sequence"],
                test_item["image"],
                test_item["prompt"],
            )
            response_text = evaluator._call_api(messages)
            reported = parse_response(response_text)
            correct = 1 if reported == str(test_item["target"]).lower() else 0

            results.append({
                "trial": i,
                "type": test_item["type"],
                "target": test_item["target"],
                "reported": reported,
                "correct": correct,
                "raw_response": response_text,
                "metadata": test_item["metadata"],
            })
            status = "✓" if correct else "✗"
            print(f"  Trial {i+1}/{len(test_items)}: {status} ({test_item['type']}: target={test_item['target']}, got={reported})", end="\r")

        metrics = calculate_mst_metrics(results)
        print(f"\n  LDI: {metrics['ldi']:.3f} | hit: {metrics['hit_rate']:.2f} | FA: {metrics['false_alarm_rate']:.2f}")
        print(f"  target acc: {metrics['target_accuracy']:.2f} | lure acc: {metrics['lure_accuracy']:.2f} | foil acc: {metrics['foil_accuracy']:.2f}")

        all_results[evaluator.get_name()] = {"trials": results, **metrics}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Mnemonic Similarity Task Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2")
    parser.add_argument("--n-study", type=int, default=32,
                        help="Number of items in study sequence")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Total test items (default: 3*n_study; subsampled balanced if smaller)")
    parser.add_argument("--source", choices=["local", "hf"], default="local",
                        help="Stimulus source: local files or huggingface")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_mst_<...>.json)")
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

    print(f"Running Mnemonic Similarity Task evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N study: {args.n_study}")
    print(f"  N trials: {args.n_trials or 3 * args.n_study}")

    results = run_evaluation(evaluators, args.n_study, n_trials=args.n_trials, source=args.source)

    if not results:
        print("No results produced — not writing output file.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials_actual = len(next(iter(results.values()))["trials"]) if results else 0
    output_data = {
        "_metadata": {
            "task": "Mnemonic Similarity Task",
            "timestamp": timestamp,
            "dataset": "MST",
            "n_study": args.n_study,
            "n_images": args.n_study,
            "n_trials": n_trials_actual,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "ldi": results[model]["ldi"],
                    "hit_rate": results[model]["hit_rate"],
                    "false_alarm_rate": results[model]["false_alarm_rate"],
                    "target_accuracy": results[model]["target_accuracy"],
                    "lure_accuracy": results[model]["lure_accuracy"],
                    "foil_accuracy": results[model]["foil_accuracy"],
                }
                for model in results
            },
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_str = "+".join(e.get_name() for e in evaluators)
    default_name = f"results_mst_{model_str}_n{args.n_study}.json"
    output_path = results_dir / (args.output if args.output else default_name)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
