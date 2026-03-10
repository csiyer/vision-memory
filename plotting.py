import json
import matplotlib.pyplot as plt
import numpy as np

def plot_group(group_name, group_models, data, colors_map):
    if not group_models:
        return

    # --- Plot 1: D-Prime ---
    plt.figure(figsize=(10, 6))
    d_primes = [data[m]["metrics"]["d_prime"] for m in group_models]
    
    plt.bar(group_models, d_primes, color=[colors_map[m] for m in group_models])
    plt.ylabel("d' (Sensitivity)")
    plt.title(f"Recognition Memory Sensitivity ({group_name})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    filename_dp = f"dprimes_{group_name}.png"
    plt.savefig(filename_dp)
    print(f"Saved {filename_dp}")
    plt.close()
    
    # --- Plot 2: Hit Rate by Delay ---
    plt.figure(figsize=(12, 6))
    bins = np.arange(0, 110, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for model in group_models:
        delays = np.array(data[model]["delays"])
        hits = np.array(data[model]["hits"])
        
        bin_means = []
        for j in range(len(bins)-1):
            mask = (delays >= bins[j]) & (delays < bins[j+1])
            if np.any(mask):
                bin_means.append(np.mean(hits[mask]))
            else:
                bin_means.append(np.nan)
        
        bin_means = np.array(bin_means)
        valid_mask = ~np.isnan(bin_means)
        
        plt.plot(bin_centers[valid_mask], bin_means[valid_mask], marker='o', label=model, color=colors_map[model], linewidth=2)
        
    plt.xlabel("Delay (Number of intermediate images)")
    plt.ylabel("Hit Rate")
    plt.title(f"Memory Decay: Hit Rate vs. Delay ({group_name})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.1)
    filename_hr = f"hitrate_delay_{group_name}.png"
    plt.savefig(filename_hr)
    print(f"Saved {filename_hr}")
    plt.close()

def plot_results(results_file="benchmark_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
        
    all_models = list(data.keys())
    # Define groups
    vision_keywords = ["vit", "mamba", "titans", "recurrent"]
    vlm_keywords = ["gemini", "gpt", "qwen", "intern"]
    
    vision_models = []
    vlm_models = []
    
    for m in all_models:
        m_lower = m.lower()
        if any(kw in m_lower for kw in vlm_keywords):
            vlm_models.append(m)
        else:
            vision_models.append(m)
            
    # Stable color mapping
    cmap = plt.cm.get_cmap("tab10")
    colors_map = {m: cmap(i % 10) for i, m in enumerate(all_models)}

    # Generate plots
    plot_group("vision-0shot", vision_models, data, colors_map)
    plot_group("vlm", vlm_models, data, colors_map)

if __name__ == "__main__":
    try:
        plot_results()
    except FileNotFoundError:
        print("Error: benchmark_results.json not found. Run benchmark.py first.")
