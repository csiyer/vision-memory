import numpy as np
import random
from metrics import compare_memorability

def test_memorability_correlation():
    print("Testing Memorability Correlation Logic...")
    
    # 1. Setup Dummy Data
    n_images = 100
    m_runs = 5
    image_names = [f"image_{i:03}.jpg" for i in range(n_images)]
    
    # Generate random human ground truth scores [0, 1]
    gt_scores = {name: random.random() for name in image_names}
    
    # Generate model performance matrix (M runs x N images)
    # Let's make the model's performance correlate with the ground truth
    # Performance is 1 (remembered) or 0 (forgotten)
    perf_matrix = []
    for run in range(m_runs):
        run_perf = []
        for name in image_names:
            score = gt_scores[name]
            # Higher score -> higher probability of being remembered
            prob = score * 0.8 + 0.1 # Probability between 0.1 and 0.9
            run_perf.append(1 if random.random() < prob else 0)
        perf_matrix.append(run_perf)
        
    # 2. Run Correlation
    results = compare_memorability(image_names, perf_matrix, gt_scores)
    
    # 3. Print Results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Correlation Results (n={results['samples'] if 'samples' in results else results['n_samples']}):")
        print(f"  Pearson r: {results['pearson']['r']:.4f} (p={results['pearson']['p']:.4e})")
        print(f"  Spearman r: {results['spearman']['r']:.4f} (p={results['spearman']['p']:.4e})")
        
        # Check if correlation is positive (as expected by our simulation)
        if results['pearson']['r'] > 0:
            print("SUCCESS: Positive correlation detected as expected.")
        else:
            print("WARNING: Correlation was not positive. This might be due to random chance with small n.")

if __name__ == "__main__":
    test_memorability_correlation()
