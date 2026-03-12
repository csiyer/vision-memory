import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import f1_score

def calculate_metrics(responses, targets):
    """
    Calculate performance metrics for the continuous recognition task.
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
    
    # Standard approximations for d-prime to avoid infinity
    adj_hit_rate = (hits + 0.5) / (n_old + 1) if n_old > 0 else 0.5
    adj_fa_rate = (fas + 0.5) / (n_new + 1) if n_new > 0 else 0.5
    d_prime = norm.ppf(adj_hit_rate) - norm.ppf(adj_fa_rate)
    
    # Weighted F1
    wf1 = f1_score(targets, responses, average='weighted', zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(fa_rate),
        "d_prime": float(d_prime),
        "weighted_f1": float(wf1),
        "counts": {"hits": int(hits), "misses": int(misses), "false_alarms": int(fas), "correct_rejections": int(crs)}
    }

def calculate_hit_rate_by_delay(responses, targets, delays):
    """Calculate hit rate for each distinct delay value."""
    responses = np.array(responses)
    targets = np.array(targets)
    delays = np.array(delays)
    
    old_mask = (targets == 1)
    old_responses = responses[old_mask]
    old_delays = delays[old_mask]
    
    unique_delays = np.unique([d for d in old_delays if d is not None])
    hr_by_delay = {}
    
    for d in unique_delays:
        mask = (old_delays == d)
        hr_by_delay[int(d)] = float(np.mean(old_responses[mask]))
        
    return hr_by_delay

def calculate_source_metrics(reported_positions, actual_positions):
    """Metrics for source memory task."""
    reported = np.array(reported_positions)
    actual = np.array(actual_positions)
    
    errors = np.abs(reported - actual)
    avg_error = np.mean(errors)
    correct = np.sum(errors == 0)
    
    return {
        "average_error": float(avg_error),
        "n_correct": int(correct),
        "accuracy": float(correct / len(actual)) if len(actual) > 0 else 0
    }

def calculate_color_metrics(reported_colors, actual_colors, n_colors=36):
    """
    Metrics for color memory task.
    Precision and guess rate are often estimated via mixture models,
    but here we provide basic error metrics.
    """
    reported = np.array(reported_colors)
    actual = np.array(actual_colors)
    
    # Circular error
    diff = (reported - actual + n_colors / 2) % n_colors - n_colors / 2
    abs_error = np.abs(diff)
    
    # Heuristic guess rate: proportion of errors > threshold
    threshold = n_colors / 4
    guess_rate = np.mean(abs_error > threshold)
    
    return {
        "average_abs_error": float(np.mean(abs_error)),
        "guess_rate_heuristic": float(guess_rate),
        "precision_heuristic": float(1.0 / np.std(diff)) if np.std(diff) > 0 else 0
    }
#