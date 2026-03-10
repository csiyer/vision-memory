import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

def calculate_metrics(responses, targets):
    """
    Calculate performance metrics for the continuous recognition task.
    
    Args:
        responses: Array-like of binary responses (1 for 'Old', 0 for 'New')
        targets: Array-like of binary ground truth (1 for 'Old', 0 for 'New')
        
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - hit_rate: Correct 'Old' identifications / Total 'Old'
            - fa_rate: Incorrect 'Old' identifications / Total 'New'
            - d_prime: Sensitivity index (Z(H) - Z(FA))
            - criterion: Response bias ( -0.5 * (Z(H) + Z(FA)) )
    """
    responses = np.array(responses)
    targets = np.array(targets)
    
    hits = np.sum((responses == 1) & (targets == 1))
    misses = np.sum((responses == 0) & (targets == 1))
    fas = np.sum((responses == 1) & (targets == 0))
    crs = np.sum((responses == 0) & (targets == 0))
    
    n_old = hits + misses
    n_new = fas + crs
    
    accuracy = (hits + crs) / (n_old + n_new) if (n_old + n_new) > 0 else 0

    hit_rate = hits / n_old if n_old > 0 else 0
    fa_rate = fas / n_new if n_new > 0 else 0
    
    if hit_rate == 1 or fa_rate == 0:
        d_prime = np.inf
        criterion = np.inf
    elif hit_rate == 0 or fa_rate == 1:
        d_prime = -np.inf
        criterion = -np.inf
    else:
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    
    return {
        "accuracy": float(accuracy),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(fa_rate),
        "d_prime": float(d_prime),
        "criterion": float(criterion),
        "counts": {
            "hits": int(hits),
            "misses": int(misses),
            "false_alarms": int(fas),
            "correct_rejections": int(crs)
        }
    }


def compare_memorability(image_ids, performance_matrix, ground_truth_dict):
    """
    Compare model memory performance with human ground truth memorability.
    
    Args:
        image_ids: List of N unique image IDs.
        performance_matrix: Array-like (M runs x N images) where each entry is 0 or 1 
                           (model's response on the 'Old' trial for that image).
        ground_truth_dict: Dictionary mapping image_id -> human memorability score.
        
    Returns:
        Dictionary with correlation coefficients and p-values.
    """
    image_ids = list(image_ids)
    perf_array = np.array(performance_matrix)
    
    # Calculate average performance per image across runs (axis 0 is runs)
    avg_performance = np.mean(perf_array, axis=0) 
    
    # Align with ground truth
    scores = []
    perfs = []
    
    for i, img_id in enumerate(image_ids):
        if img_id in ground_truth_dict:
            scores.append(ground_truth_dict[img_id])
            perfs.append(avg_performance[i])
            
    if len(scores) < 2:
        return {"error": "Insufficient overlapping data for correlation"}
        
    pearson_r, p_pearson = pearsonr(perfs, scores)
    spearman_r, p_spearman = spearmanr(perfs, scores)
    
    return {
        "pearson": {
            "r": float(pearson_r),
            "p": float(p_pearson)
        },
        "spearman": {
            "r": float(spearman_r),
            "p": float(p_spearman)
        },
        "n_samples": len(scores)
    }
#