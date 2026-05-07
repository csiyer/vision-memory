#!/usr/bin/env python3
"""
Regenerate results_summary.csv from all result JSON files in results/.

Run from the project root:
    python3 -m src.generate_summary_csv
"""
import csv
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_CSV = Path("results_summary.csv")

COLUMNS = [
    "date", "task", "dataset", "model", "foil_type",
    "n_images_in_context", "n_trials", "n_valid", "n_correct", "n_parse_failures",
    "max_context", "p_old", "min_delay", "max_delay", "mode", "split",
    "accuracy", "d_prime", "mem_score", "hit_rate", "correct_rejection_rate",
    "false_alarm_rate", "weighted_f1", "compliance",
]

# Files to skip — benchmarks, combo models, old unlabelled formats
SKIP_PATTERNS = ["accuracy", "muirbench", "mmiu"]


def should_skip(filename):
    return any(p in filename for p in SKIP_PATTERNS)


def parse_date(timestamp):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestamp, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    return timestamp[:10] if timestamp else ""


def extract_rows(result_file):
    with open(result_file) as f:
        data = json.load(f)

    meta = data.get("_metadata", {})
    if not meta:
        return []

    task = meta.get("task", "")
    date = parse_date(meta.get("timestamp", ""))
    dataset = meta.get("dataset", "")
    n_images = meta.get("n_images") or meta.get("image_count", "")
    n_trials = meta.get("n_trials", "")
    foil_type = meta.get("foil_type", "")
    max_context = meta.get("max_context_pairs") or meta.get("max_context_images", "")
    p_old = meta.get("p_old", "")
    min_delay = meta.get("min_delay", "")
    max_delay = meta.get("max_delay", "")
    mode = meta.get("mode", "")
    split = meta.get("split", "")
    summary = meta.get("summary", {})

    rows = []
    for model, model_sum in summary.items():
        row = {col: "" for col in COLUMNS}
        row["date"] = date
        row["task"] = task
        row["dataset"] = dataset
        row["model"] = model
        row["foil_type"] = foil_type
        row["n_images_in_context"] = n_images
        row["n_trials"] = n_trials
        row["max_context"] = max_context
        row["p_old"] = p_old
        row["min_delay"] = min_delay
        row["max_delay"] = max_delay
        row["mode"] = mode
        row["split"] = split

        row["accuracy"] = model_sum.get("accuracy", "")
        row["d_prime"] = model_sum.get("d_prime", "")
        row["mem_score"] = model_sum.get("mem_score", "")
        row["hit_rate"] = model_sum.get("hit_rate", "")
        row["false_alarm_rate"] = model_sum.get("false_alarm_rate", "")
        fa = model_sum.get("false_alarm_rate")
        row["correct_rejection_rate"] = model_sum.get("correct_rejection_rate",
            round(1 - float(fa), 6) if fa != "" and fa is not None else "")
        row["weighted_f1"] = model_sum.get("weighted_f1", "")
        row["compliance"] = model_sum.get("compliance", "")
        row["n_valid"] = model_sum.get("n_valid_trials", model_sum.get("n_valid", ""))
        row["n_correct"] = model_sum.get("n_correct", "")
        row["n_parse_failures"] = model_sum.get("n_parse_failures", "")

        rows.append(row)
    return rows


def main():
    all_rows = []
    skipped = []

    for result_file in sorted(RESULTS_DIR.glob("results_*.json")):
        if should_skip(result_file.stem):
            skipped.append(result_file.name)
            continue
        try:
            rows = extract_rows(result_file)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [WARN] {result_file.name}: {e}")

    # Sort by task, dataset, model, foil, n_images for readability
    def sort_key(r):
        try:
            n = int(r["n_images_in_context"])
        except (ValueError, TypeError):
            n = 0
        return (r["task"], r["dataset"], r["model"], r["foil_type"], n)

    all_rows.sort(key=sort_key)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows to {OUTPUT_CSV}")
    print(f"Skipped {len(skipped)} files: {skipped}")


if __name__ == "__main__":
    main()
