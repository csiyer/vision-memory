#!/usr/bin/env python
"""
MuirBench Multi-Image Understanding Evaluation

Dataset: MUIRBENCH/MUIRBENCH (HuggingFace)
Format:  Multiple-choice (A-D), 2,600 samples, avg 4.3 images/sample
Tasks:   12 tasks across 10 image-relation categories

Usage:
  python -m eval_scripts.eval_muirbench --models gpt-4o gemini --max-samples 100
  python -m eval_scripts.eval_muirbench --models gpt-4o --tasks difference_spotting ordering
  python -m eval_scripts.eval_muirbench --models gpt-4o gemini qwen
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator


MUIRBENCH_DATASET = "MUIRBENCH/MUIRBENCH"

PROMPT_TEMPLATE = """\
Look at the images provided and answer the following multiple-choice question.
Respond with only the letter of the correct answer (A, B, C, or D).

{question}

{options}

Answer:"""


def load_dataset(max_samples=None, tasks=None, relations=None):
    from datasets import load_dataset as hf_load
    print(f"Loading MuirBench from HuggingFace ({MUIRBENCH_DATASET})...")
    ds = hf_load(MUIRBENCH_DATASET, split="test")

    samples = []
    for row in ds:
        if tasks and row["task"] not in tasks:
            continue
        if relations and row["image_relation"] not in relations:
            continue
        samples.append(row)
        if max_samples and len(samples) >= max_samples:
            break

    print(f"  Loaded {len(samples)} samples")
    return samples


def format_options(options):
    letters = "ABCDEFGH"
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def build_messages(evaluator, question, options, images):
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        options=format_options(options),
    )
    content = [{"type": "text", "text": prompt}]
    for img in images:
        # MuirBench images come as PIL Images from HuggingFace
        content.append(evaluator._encode_image(img))
    return [{"role": "user", "content": content}]


def parse_answer(text):
    """Extract A/B/C/D from response."""
    if text is None:
        return None
    text = text.strip()
    # Direct single letter
    if re.match(r"^[A-Da-d]\.?$", text):
        return text[0].upper()
    # First letter on a line like "A. something"
    m = re.match(r"^([A-Da-d])[.):,\s]", text)
    if m:
        return m.group(1).upper()
    # Letter anywhere near the start
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return None


def compute_metrics(trials):
    total = len(trials)
    valid = [t for t in trials if t["parsed_response"] is not None]
    n_valid = len(valid)
    n_correct = sum(1 for t in valid if t["correct"])

    # Per-task breakdown
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    by_relation = defaultdict(lambda: {"correct": 0, "total": 0})
    for t in trials:
        task = t["metadata"]["task"]
        rel = t["metadata"]["image_relation"]
        by_task[task]["total"] += 1
        by_relation[rel]["total"] += 1
        if t["correct"]:
            by_task[task]["correct"] += 1
            by_relation[rel]["correct"] += 1

    task_acc = {k: v["correct"] / v["total"] for k, v in by_task.items()}
    relation_acc = {k: v["correct"] / v["total"] for k, v in by_relation.items()}

    return {
        "accuracy": n_correct / total if total > 0 else 0.0,
        "valid_accuracy": n_correct / n_valid if n_valid > 0 else 0.0,
        "compliance": n_valid / total if total > 0 else 0.0,
        "n_trials": total,
        "n_valid": n_valid,
        "n_correct": n_correct,
        "n_parse_failures": total - n_valid,
        "accuracy_by_task": task_acc,
        "accuracy_by_relation": relation_acc,
    }


def run_evaluation(evaluators, samples):
    all_results = {}

    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")
        trials = []

        for i, row in enumerate(samples):
            images = list(row["image_list"])  # list of PIL Images
            messages = build_messages(evaluator, row["question"], row["options"], images)
            raw = evaluator._call_api(messages)
            parsed = parse_answer(raw)
            correct = parsed == row["answer"] if parsed is not None else False

            trials.append({
                "idx": row["idx"],
                "target": row["answer"],
                "parsed_response": parsed,
                "raw_response": raw,
                "correct": correct,
                "n_images": len(images),
                "metadata": {
                    "task": row["task"],
                    "image_relation": row["image_relation"],
                    "image_type": row["image_type"],
                },
            })
            status = "✓" if correct else "✗"
            print(f"  Sample {i + 1}/{len(samples)}: {status}  [{row['task']}]", end="\r")

        metrics = compute_metrics(trials)
        print(
            f"\n  Accuracy: {metrics['accuracy']:.1%} | "
            f"Valid Acc: {metrics['valid_accuracy']:.1%} | "
            f"Compliance: {metrics['compliance']:.1%}"
        )
        all_results[evaluator.get_name()] = {"trials": trials, **metrics}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MuirBench Multi-Image Evaluation")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o", "gemini"],
        help="Models: gpt-4o, claude, gemini, qwen",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (default: all 2600)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Filter to specific tasks (e.g. difference_spotting ordering)")
    parser.add_argument("--relations", nargs="+", default=None,
                        help="Filter to specific image relations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename under results/ (default: results_muirbench_<ts>.json)")
    args = parser.parse_args()

    evaluators = []
    for m in args.models:
        m = m.strip()
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
        print("No valid models specified.")
        return

    samples = load_dataset(
        max_samples=args.max_samples,
        tasks=args.tasks,
        relations=args.relations,
    )

    print(f"\nRunning MuirBench evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  Samples: {len(samples)}")
    if args.tasks:
        print(f"  Tasks filter: {args.tasks}")
    if args.relations:
        print(f"  Relations filter: {args.relations}")

    results = run_evaluation(evaluators, samples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "MuirBench",
            "timestamp": timestamp,
            "dataset": MUIRBENCH_DATASET,
            "n_samples": len(samples),
            "tasks_filter": args.tasks,
            "relations_filter": args.relations,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "valid_accuracy": results[model]["valid_accuracy"],
                    "compliance": results[model]["compliance"],
                    "accuracy_by_task": results[model]["accuracy_by_task"],
                    "accuracy_by_relation": results[model]["accuracy_by_relation"],
                }
                for model in results
            },
        },
        **results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    fname = args.output or f"results_muirbench_{timestamp}.json"
    output_path = results_dir / fname
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
