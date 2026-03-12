import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_target_data():
    target_path = Path("memory_datasets/target_data.json")
    if target_path.exists():
        with open(target_path, "r") as f:
            return json.load(f)
    return {}

def plot_continuous_recognition(models_data, output_dir="plots"):
    """
    Plots d-prime, weighted F1, and hit rate vs delay for continuous recognition.
    """
    Path(output_dir).mkdir(exist_ok=True)
    target_data = load_target_data().get("Brady2008Continuous", {})
    
    # D-prime and F1
    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())
    d_primes = [models_data[m].get("d_prime", 0) for m in models]
    f1s = [models_data[m].get("weighted_f1", 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, d_primes, width, label='d-prime')
    plt.bar(x + width/2, f1s, width, label='weighted F1')
    plt.xticks(x, models, rotation=45)
    plt.ylabel("Score")
    plt.title("Continuous Recognition Performance")
    plt.legend()
    
    # Hit rate vs delay
    plt.subplot(1, 2, 2)
    if "hit_rate_delays" in target_data:
        plt.plot(target_data["hit_rate_delays"], target_data["hit_rate_by_delay"][:len(target_data["hit_rate_delays"])], 
                 'k--', marker='s', label='Human (Brady2008)')
    
    for m in models:
        hr_delay = models_data[m].get("hit_rate_by_delay", {})
        if hr_delay:
            delays = sorted([int(d) for d in hr_delay.keys()])
            rates = [hr_delay[str(d)] for d in delays]
            plt.plot(delays, rates, marker='o', label=m)
            
    plt.xscale('log')
    plt.xlabel("Delay (images)")
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate by Delay")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/continuous_recognition.png")
    plt.close()

def plot_2afc(models_data, output_dir="plots"):
    """Plots accuracy by foil type for 2-AFC."""
    Path(output_dir).mkdir(exist_ok=True)
    target_data = load_target_data().get("Brady2008AFC", {}).get("accuracy", {})
    
    plt.figure(figsize=(10, 6))
    foil_types = ["novel", "exemplar", "state"]
    models = list(models_data.keys())
    
    x = np.arange(len(foil_types))
    width = 0.8 / (len(models) + 1)
    
    # Human data
    human_accs = [target_data.get(f, 0) for f in foil_types]
    plt.bar(x - (len(models)*width)/2, human_accs, width, color='gray', alpha=0.5, label='Human')
    
    for i, m in enumerate(models):
        accs = [models_data[m].get("accuracy_by_type", {}).get(f, 0) for f in foil_types]
        plt.bar(x - (len(models)*width)/2 + (i+1)*width, accs, width, label=m)
        
    plt.xticks(x, foil_types)
    plt.ylabel("Accuracy")
    plt.title("2-AFC Recognition Accuracy")
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f"{output_dir}/afc_recognition.png")
    plt.close()

def plot_source_memory(models_data, output_dir="plots"):
    """Plots average error for source memory."""
    Path(output_dir).mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    models = list(models_data.keys())
    errors = [models_data[m].get("average_error", 0) for m in models]
    
    plt.bar(models, errors)
    plt.ylabel("Average Position Error")
    plt.title("Source Memory Error")
    plt.savefig(f"{output_dir}/source_memory.png")
    plt.close()

def plot_color_memory(models_data, output_dir="plots"):
    """Plots precision and guess rate for color memory."""
    Path(output_dir).mkdir(exist_ok=True)
    target_data = load_target_data().get("Brady2013Color", {}).get("long_term_memory", {})
    
    plt.figure(figsize=(12, 5))
    models = list(models_data.keys())
    
    # Guess rate
    plt.subplot(1, 2, 1)
    guess_rates = [models_data[m].get("guess_rate_heuristic", 0) for m in models]
    plt.bar(models, guess_rates, label='Model')
    if "guess_rate" in target_data:
        plt.axhline(y=target_data["guess_rate"], color='r', linestyle='--', label='Human (LTM)')
    plt.ylabel("Guess Rate")
    plt.title("Color Memory: Guess Rate")
    plt.legend()
    
    # Precision
    plt.subplot(1, 2, 2)
    precisions = [models_data[m].get("precision_heuristic", 0) for m in models]
    plt.bar(models, precisions, label='Model')
    # Target precision is often in SD degrees, our heuristic is 1/std
    # This is a bit loose but shows the idea.
    plt.ylabel("Precision (1/std)")
    plt.title("Color Memory: Precision")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/color_memory.png")
    plt.close()
