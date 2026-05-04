import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _load_literature(task_key=None):
    """Load from literature/ directory. Optionally return a sub-key."""
    lit = {}
    lit_dir = Path("literature")
    for path in lit_dir.glob("*.json"):
        with open(path) as f:
            lit.update(json.load(f))
    if task_key:
        return lit.get(task_key, {})
    return lit


def plot_continuous_recognition(models_data, output_dir="plots"):
    """Plots d-prime, weighted F1, and hit rate vs delay for continuous recognition."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("Brady2008Continuous")

    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())
    d_primes = [models_data[m].get("d_prime", 0) for m in models]
    f1s = [models_data[m].get("weighted_f1", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, d_primes, width, label="d-prime")
    plt.bar(x + width / 2, f1s, width, label="weighted F1")
    plt.xticks(x, models, rotation=45)
    plt.ylabel("Score")
    plt.title("Continuous Recognition Performance")
    plt.legend()

    plt.subplot(1, 2, 2)
    if human.get("hit_rate_delays") and human.get("hit_rate_by_delay"):
        plt.plot(
            human["hit_rate_delays"],
            human["hit_rate_by_delay"][:len(human["hit_rate_delays"])],
            "k--", marker="s", label="Human (Brady2008)"
        )

    for m in models:
        hr_delay = models_data[m].get("hit_rate_by_delay", {})
        if hr_delay:
            delays = sorted(int(d) for d in hr_delay.keys())
            rates = [hr_delay[str(d)] if str(d) in hr_delay else hr_delay[d] for d in delays]
            plt.plot(delays, rates, marker="o", label=m)

    plt.xscale("log")
    plt.xlabel("Delay (images)")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate by Delay")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/continuous_recognition.png")
    plt.close()


def plot_2afc(models_data, output_dir="plots"):
    """Plots accuracy by foil type for 2-AFC recognition."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("Brady2008AFC").get("accuracy", {})

    plt.figure(figsize=(10, 6))
    foil_types = ["novel", "exemplar", "state"]
    models = list(models_data.keys())

    x = np.arange(len(foil_types))
    width = 0.8 / (len(models) + 1)

    human_accs = [human.get(f, 0) for f in foil_types]
    plt.bar(x - (len(models) * width) / 2, human_accs, width,
            color="gray", alpha=0.5, label="Human (Brady2008)")

    for i, m in enumerate(models):
        accs = [models_data[m].get("accuracy_by_type", {}).get(f, 0) for f in foil_types]
        plt.bar(x - (len(models) * width) / 2 + (i + 1) * width, accs, width, label=m)

    plt.xticks(x, foil_types)
    plt.ylabel("Accuracy")
    plt.title("2-AFC Recognition Accuracy")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(f"{output_dir}/afc_recognition.png")
    plt.close()


def plot_serial_order_memory(models_data, output_dir="plots"):
    """Plots average position error and Spearman rho for free serial order memory."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("serial_order").get("free", {})  # {"Sherman2025": {...}}

    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())

    # Average position error
    plt.subplot(1, 2, 1)
    errors = [models_data[m].get("average_error", 0) for m in models]
    plt.bar(models, errors)
    plt.ylabel("Average Position Error")
    plt.title("Serial Order Memory: Position Error")
    plt.xticks(rotation=45)

    # Spearman rho vs. human benchmark
    plt.subplot(1, 2, 2)
    rhos = [models_data[m].get("spearman_rho") or 0 for m in models]
    plt.bar(models, rhos, label="Model")

    if "Sherman2025" in human:
        human_rho = human["Sherman2025"].get("rank_correlation_with_truth")
        if human_rho is not None:
            plt.axhline(y=human_rho, color="k", linestyle="--",
                        label=f"Human — Sherman2025 (ρ={human_rho})")

    plt.ylabel("Spearman ρ")
    plt.title("Serial Order Memory: Rank Correlation")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/serial_order_memory.png")
    plt.close()


def plot_afc_serial_order_memory(models_data, output_dir="plots"):
    """Plots 2-AFC serial order memory accuracy by distance with human benchmarks."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("serial_order").get("2afc", {})

    plt.figure(figsize=(9, 6))
    models = list(models_data.keys())

    for model_name in models:
        accuracy_by_distance = models_data[model_name].get("accuracy_by_distance", {})
        if not accuracy_by_distance:
            continue
        distances = sorted(int(d) for d in accuracy_by_distance.keys())
        accuracies = [
            accuracy_by_distance[str(d)] if str(d) in accuracy_by_distance else accuracy_by_distance[d]
            for d in distances
        ]
        plt.plot(distances, accuracies, marker="o", label=model_name)

    # Human data points — Dubrow2014 reports accuracy at specific lags
    # "1 trial apart" = lag 1 → distance 0; "3 trials apart" = lag 3 → distance 2
    if "Dubrow2014" in human:
        d = human["Dubrow2014"]
        dubrow_points = {}
        if "accuracy_1_trial_apart" in d:
            dubrow_points[0] = d["accuracy_1_trial_apart"]
        if "accuracy_3_trials_apart" in d:
            dubrow_points[2] = d["accuracy_3_trials_apart"]
        if dubrow_points:
            plt.scatter(list(dubrow_points.keys()), list(dubrow_points.values()),
                        marker="s", color="gray", zorder=5, label="Human — Dubrow2014")
        if "accuracy" in d:
            plt.axhline(y=d["accuracy"], color="gray", linestyle="--", alpha=0.6,
                        label=f"Human overall — Dubrow2014 ({d['accuracy']})")

    if "Sherman 2025" in human:
        d = human["Sherman 2025"]
        if "accuracy_1_trial_apart" in d:
            plt.scatter([0], [d["accuracy_1_trial_apart"]], marker="^", color="black",
                        zorder=5, label="Human (lag 1) — Sherman2025")
        if "accuracy" in d:
            plt.axhline(y=d["accuracy"], color="black", linestyle="--", alpha=0.6,
                        label=f"Human overall — Sherman2025 ({d['accuracy']})")

    plt.xlabel("Distance (items between probes)")
    plt.ylabel("Accuracy")
    plt.title("2-AFC Serial Order Accuracy by Distance")
    plt.ylim(0.4, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Chance")
    if models or human:
        plt.legend(fontsize=8)

    plt.savefig(f"{output_dir}/afc_serial_order_memory.png")
    plt.close()


def plot_color_memory(models_data, output_dir="plots"):
    """Plots circular error metrics (continuous) or accuracy (named) for color memory."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("Brady2013Color").get("long_term_memory", {})

    # Determine whether data is from continuous or named mode
    first = next(iter(models_data.values()), {})
    is_named = "accuracy_by_color" in first

    if is_named:
        plt.figure(figsize=(8, 6))
        models = list(models_data.keys())
        accuracies = [models_data[m].get("accuracy", 0) for m in models]
        plt.bar(models, accuracies, label="Model")
        plt.ylabel("Accuracy")
        plt.title("Color Memory: Named Color Accuracy")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
    else:
        plt.figure(figsize=(15, 5))
        models = list(models_data.keys())

        plt.subplot(1, 3, 1)
        errors = [models_data[m].get("average_abs_error", 0) for m in models]
        plt.bar(models, errors)
        plt.ylabel("Mean absolute circular error (degrees)")
        plt.title("Color Memory: Circular Error")
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 2)
        guess_rates = [models_data[m].get("guess_rate_heuristic", 0) for m in models]
        plt.bar(models, guess_rates)
        if "guess_rate" in human:
            plt.axhline(y=human["guess_rate"], color="r", linestyle="--",
                        label=f"Human LTM (Brady2013) — {human['guess_rate']}")
        plt.ylabel("Guess Rate")
        plt.title("Color Memory: Guess Rate")
        plt.legend(fontsize=8)
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 3)
        precisions = [models_data[m].get("precision_heuristic", 0) for m in models]
        plt.bar(models, precisions)
        plt.ylabel("Precision (1 / std circular error)")
        plt.title("Color Memory: Precision")
        plt.xticks(rotation=45)

        plt.tight_layout()

    plt.savefig(f"{output_dir}/color_memory.png")
    plt.close()


def plot_paired_associates(models_data, output_dir="plots"):
    """Plots accuracy for paired associate memory with human benchmark."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("Tompary2015")

    plt.figure(figsize=(8, 6))
    models = list(models_data.keys())
    accuracies = [models_data[m].get("accuracy", 0) for m in models]

    plt.bar(models, accuracies)

    if "accuracy" in human:
        plt.axhline(y=human["accuracy"], color="k", linestyle="--",
                    label=f"Human — Tompary2015 ({human['accuracy']})")

    plt.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Chance (2-AFC)")
    plt.ylabel("Accuracy")
    plt.title("Paired Associate Memory Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=8)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/paired_associates.png")
    plt.close()


def plot_mst(models_data, output_dir="plots"):
    """Plots MST LDI (overall bar) and LDI-by-bin tuning curve with human benchmarks."""
    Path(output_dir).mkdir(exist_ok=True)
    lit = _load_literature()
    human_ldi = lit.get("VanderlipStark2024_pooled", {}).get("ldi")
    human_bins_entry = lit.get("Kirwan2007_ldi_by_bin", {})
    human_ldi_by_bin = human_bins_entry.get("ldi_by_bin", {})

    models = list(models_data.keys())

    plt.figure(figsize=(14, 5))

    # --- Left: overall LDI bar chart ---
    plt.subplot(1, 2, 1)
    ldis = [models_data[m].get("ldi", 0) for m in models]
    plt.bar(models, ldis)
    if human_ldi is not None:
        plt.axhline(y=human_ldi, color="k", linestyle="--",
                    label=f"Human — Vanderlip & Stark 2024, pooled N=410 (LDI={human_ldi})")
    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="Chance")
    plt.ylabel("LDI")
    plt.title("MST: Lure Discrimination Index")
    plt.ylim(-0.1, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=8)
    plt.xticks(rotation=45)

    # --- Right: LDI by bin (tuning curve) ---
    plt.subplot(1, 2, 2)
    for model_name in models:
        ldi_by_bin = models_data[model_name].get("ldi_by_bin", {})
        if ldi_by_bin:
            bins = sorted(int(b) for b in ldi_by_bin.keys())
            vals = [ldi_by_bin[b] if b in ldi_by_bin else ldi_by_bin[str(b)] for b in bins]
            plt.plot(bins, vals, marker="o", label=model_name)

    if human_ldi_by_bin:
        hbins = sorted(int(b) for b in human_ldi_by_bin.keys())
        hvals = [human_ldi_by_bin[str(b)] if str(b) in human_ldi_by_bin else human_ldi_by_bin[b]
                 for b in hbins]
        approx = human_bins_entry.get("approximate", False)
        label = f"Human — Kirwan2007{' (approx.)' if approx else ''}"
        plt.plot(hbins, hvals, "k--", marker="s", label=label)

    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Lure Bin (1=most similar, 5=least similar)")
    plt.ylabel("LDI")
    plt.title("MST: LDI by Lure Similarity Bin")
    plt.ylim(-0.1, 0.8)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    if models or human_ldi_by_bin:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mst.png")
    plt.close()


def plot_associative_inference(models_data, output_dir="plots"):
    """Plots associative inference accuracy with human benchmark."""
    Path(output_dir).mkdir(exist_ok=True)
    human = _load_literature("Banino2016")

    plt.figure(figsize=(8, 6))
    models = list(models_data.keys())
    accuracies = [models_data[m].get("accuracy", 0) for m in models]

    plt.bar(models, accuracies)

    if "AC_accuracy" in human:
        plt.axhline(y=human["AC_accuracy"], color="k", linestyle="--",
                    label=f"Human A→C — Banino2016 ({human['AC_accuracy']})")
    if "AB_accuracy" in human:
        plt.axhline(y=human["AB_accuracy"], color="gray", linestyle=":",
                    label=f"Human A→B — Banino2016 ({human['AB_accuracy']})")

    plt.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Chance (2-AFC)")
    plt.ylabel("Accuracy")
    plt.title("Associative Inference Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=8)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/associative_inference.png")
    plt.close()
