"""
Plot evaluation results. Accepts either saved JSON (with optional `_metadata`) or the
in-memory dict returned by eval runners (model names -> per-model dicts with `trials` + metrics).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PLOTS_DIR = _REPO_ROOT / "output" / "plots"
TARGET_DATA_PATH = _REPO_ROOT.parent / "datasets" / "target_data.json"


def default_plots_dir() -> Path:
    """Default directory for figures (``output/plots`` under the repo root)."""
    return DEFAULT_PLOTS_DIR


def extract_models_from_results(results: dict[str, Any]) -> dict[str, Any]:
    """Remove ``_metadata``; remaining top-level keys are model name -> result dict."""
    return {k: v for k, v in results.items() if k != "_metadata"}


def load_results_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_target_data() -> dict[str, Any]:
    if TARGET_DATA_PATH.exists():
        with open(TARGET_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _plots_path(output_dir: str | Path | None) -> Path:
    p = Path(output_dir) if output_dir is not None else DEFAULT_PLOTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hit_rate_delay_series(hit_rate_by_delay: dict | None) -> tuple[list[int], list[float]]:
    """Sort delays and return parallel lists of delay (int) and hit rate (float)."""
    if not hit_rate_by_delay:
        return [], []
    delays: list[int] = []
    for k in hit_rate_by_delay:
        try:
            delays.append(int(k))
        except (TypeError, ValueError):
            continue
    delays.sort()
    rates: list[float] = []
    for d in delays:
        raw = hit_rate_by_delay.get(d)
        if raw is None:
            raw = hit_rate_by_delay.get(str(d))
        rates.append(float(raw) if raw is not None else 0.0)
    return delays, rates


def plot_continuous_recognition(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "continuous_recognition.png",
) -> Path:
    """
    d', weighted F1, and hit rate vs delay. Pass full JSON or model-only dict.
    """
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    target_data = load_target_data().get("Brady2008Continuous", {})

    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())
    d_primes = [models_data[m].get("d_prime", 0) for m in models]
    f1s = [models_data[m].get("weighted_f1", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, d_primes, width, label="d-prime")
    plt.bar(x + width / 2, f1s, width, label="weighted F1")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Continuous Recognition Performance")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "hit_rate_delays" in target_data:
        hd = target_data["hit_rate_delays"]
        hr = target_data["hit_rate_by_delay"]
        n = min(len(hd), len(hr))
        plt.plot(hd[:n], hr[:n], "k--", marker="s", label="Human (Brady2008)")

    for m in models:
        delays, rates = _hit_rate_delay_series(models_data[m].get("hit_rate_by_delay"))
        if delays:
            plt.plot(delays, rates, marker="o", label=m)

    plt.xscale("log")
    plt.xlabel("Delay (images)")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate by Delay")
    plt.legend()

    plt.tight_layout()
    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_2afc(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "afc_recognition.png",
) -> Path:
    """Grouped accuracy by foil type (novel / exemplar / state) vs human baseline."""
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    target_data = load_target_data().get("Brady2008AFC", {}).get("accuracy", {})

    plt.figure(figsize=(10, 6))
    foil_types = ["novel", "exemplar", "state"]
    models = list(models_data.keys())

    x = np.arange(len(foil_types))
    width = 0.8 / (len(models) + 1) if models else 0.8

    human_accs = [target_data.get(f, 0) for f in foil_types]
    plt.bar(x - (len(models) * width) / 2, human_accs, width, color="gray", alpha=0.5, label="Human")

    for i, m in enumerate(models):
        accs = [models_data[m].get("accuracy_by_type", {}).get(f, 0) for f in foil_types]
        plt.bar(x - (len(models) * width) / 2 + (i + 1) * width, accs, width, label=m)

    plt.xticks(x, foil_types)
    plt.ylabel("Accuracy")
    plt.title("2-AFC Recognition Accuracy")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_2afc_metrics(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "afc_metrics.png",
) -> Path:
    """Overall accuracy, mem score, d' (scaled) and accuracy by foil type."""
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    models = list(models_data.keys())

    foil_types: set[str] = set()
    for m in models:
        foil_types.update(models_data[m].get("accuracy_by_type", {}).keys())
    foil_types_sorted = sorted(foil_types)

    if foil_types_sorted:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes[0], axes[1]
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = None

    x = np.arange(len(models))
    w = 0.25

    accuracies = [models_data[m].get("accuracy", 0) for m in models]
    d_primes = [models_data[m].get("d_prime", 0) for m in models]
    mem_scores = [models_data[m].get("mem_score", 0) for m in models]

    ax1.bar(x - w, accuracies, w, label="Accuracy", color="steelblue")
    ax1.bar(x, mem_scores, w, label="Mem Score", color="seagreen")
    ax1.bar(x + w, [d / 3 for d in d_primes], w, label="d' (scaled /3)", color="coral")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("2-AFC Overall Performance")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.1)

    if ax2 is not None:
        x2 = np.arange(len(foil_types_sorted))
        width2 = 0.8 / len(models) if models else 0.8
        for i, m in enumerate(models):
            accs = [models_data[m].get("accuracy_by_type", {}).get(ft, 0) for ft in foil_types_sorted]
            ax2.bar(
                x2 + i * width2 - (len(models) - 1) * width2 / 2,
                accs,
                width2,
                label=m,
            )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(foil_types_sorted)
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy by Foil Type")
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.legend()
        ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_2afc_all(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write both 2-AFC figures."""
    p1 = plot_2afc(results, output_dir=output_dir)
    p2 = plot_2afc_metrics(results, output_dir=output_dir)
    return p1, p2


def plot_pam(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "pam_accuracy.png",
) -> Path:
    """Per-model accuracy for paired associate memory."""
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    models = list(models_data.keys())

    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    accs = [models_data[m].get("accuracy", 0) for m in models]
    plt.bar(x, accs, color="steelblue")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Paired Associate Memory")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_source_memory(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "source_memory.png",
) -> Path:
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    plt.figure(figsize=(8, 6))
    models = list(models_data.keys())
    errors = [models_data[m].get("average_error", 0) for m in models]
    plt.bar(models, errors)
    plt.ylabel("Average Position Error")
    plt.title("Source Memory Error")
    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_color_memory(
    results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    filename: str = "color_memory.png",
) -> Path:
    models_data = extract_models_from_results(results)
    out = _plots_path(output_dir)
    target_data = load_target_data().get("Brady2013Color", {}).get("long_term_memory", {})

    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())

    plt.subplot(1, 2, 1)
    guess_rates = [models_data[m].get("guess_rate_heuristic", 0) for m in models]
    plt.bar(models, guess_rates, label="Model")
    if "guess_rate" in target_data:
        plt.axhline(y=target_data["guess_rate"], color="r", linestyle="--", label="Human (LTM)")
    plt.ylabel("Guess Rate")
    plt.title("Color Memory: Guess Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    precisions = [models_data[m].get("precision_heuristic", 0) for m in models]
    plt.bar(models, precisions, label="Model")
    plt.ylabel("Precision (1/std)")
    plt.title("Color Memory: Precision")

    plt.tight_layout()
    path = out / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path
