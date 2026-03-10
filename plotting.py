import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_file="benchmark_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
        
    models = list(data.keys())
    colors = plt.cm.get_cmap("tab10", len(models))
    
    # --- Plot 1: D-Prime ---
    plt.figure(figsize=(10, 6))
    d_primes = [data[m]["metrics"]["d_prime"] for m in models]
    
    bars = plt.bar(models, d_primes, color=[colors(i) for i in range(len(models))])
    plt.ylabel("d' (Sensitivity)")
    plt.title("Recognition Memory Sensitivity (d')")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("d_primes.png")
    print("Saved d_primes.png")
    
    # --- Plot 2: Hit Rate by Delay ---
    plt.figure(figsize=(12, 6))
    
    for i, model in enumerate(models):
        delays = np.array(data[model]["delays"])
        hits = np.array(data[model]["hits"])
        
        # Bin the delays for smoother plotting
        # We'll use bins of size 10
        bins = np.arange(0, 110, 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        for j in range(len(bins)-1):
            mask = (delays >= bins[j]) & (delays < bins[j+1])
            if np.any(mask):
                bin_means.append(np.mean(hits[mask]))
            else:
                bin_means.append(np.nan)
        
        bin_means = np.array(bin_means)
        valid_mask = ~np.isnan(bin_means)
        
        plt.plot(bin_centers[valid_mask], bin_means[valid_mask], marker='o', label=model, color=colors(i), linewidth=2)
        
    plt.xlabel("Delay (Number of intermediate images)")
    plt.ylabel("Hit Rate")
    plt.title("Memory Decay: Hit Rate vs. Delay")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.1)
    plt.savefig("hit_rate_delay.png")
    print("Saved hit_rate_delay.png")

if __name__ == "__main__":
    try:
        plot_results()
    except FileNotFoundError:
        print("Error: benchmark_results.json not found. Run benchmark.py first.")
