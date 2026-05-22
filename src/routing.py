import numpy as np
import pandas as pd

def compute_routing_metrics(probabilities, true_labels, t_low, t_high):
    """
    Computes workload reduction and missed critical fault rate for a given t_low and t_high.
    Parameters:
        probabilities (np.ndarray): calibrated predicted probabilities
        true_labels (np.ndarray): true binary labels (positive = critical fault)
        t_low (float): lower threshold (p <= t_low -> auto-clear)
        t_high (float): upper threshold (p >= t_high -> urgent inspection)
    """
    total_cases = len(probabilities)
    positive_cases = np.sum(true_labels == 1)
    
    if positive_cases == 0:
        mcfr = 0.0
    else:
        # Missed critical faults are those positive cases routed to auto-clear (p <= t_low)
        auto_clear_mask = probabilities <= t_low
        missed_faults = np.sum((true_labels == 1) & auto_clear_mask)
        mcfr = missed_faults / positive_cases
        
    # Workload reduction: auto-clear cases / total cases
    auto_clear_count = np.sum(probabilities <= t_low)
    workload_red = auto_clear_count / total_cases
    
    # Other routing counts
    urgent_count = np.sum(probabilities >= t_high)
    human_count = total_cases - auto_clear_count - urgent_count
    
    return {
        "workload_reduction": workload_red,
        "missed_critical_fault_rate": mcfr,
        "auto_clear_rate": auto_clear_count / total_cases,
        "human_review_rate": human_count / total_cases,
        "urgent_inspection_rate": urgent_count / total_cases
    }

def optimize_thresholds(probabilities, true_labels, alpha=0.05, step=0.01):
    """
    Performs grid search over t_low and t_high to maximize Workload Reduction
    subject to Missed Critical Fault Rate <= alpha on the validation set.
    """
    best_wr = -1.0
    best_mcfr = 2.0
    best_t_low = 0.0
    best_t_high = 1.0
    
    thresholds = np.arange(0.0, 1.0 + step, step)
    
    # Loop over all t_low and t_high combinations where t_low < t_high
    for t_l in thresholds:
        for t_h in thresholds:
            if t_l >= t_h:
                continue
                
            metrics = compute_routing_metrics(probabilities, true_labels, t_l, t_h)
            wr = metrics["workload_reduction"]
            mcfr = metrics["missed_critical_fault_rate"]
            
            # Constraint check
            if mcfr <= alpha:
                # Tie-breaking logic:
                # 1. Maximize workload reduction
                # 2. If tie, minimize missed critical fault rate
                # 3. If tie, choose lower thresholds (more conservative)
                is_better = False
                if wr > best_wr:
                    is_better = True
                elif np.isclose(wr, best_wr):
                    if mcfr < best_mcfr:
                        is_better = True
                    elif np.isclose(mcfr, best_mcfr):
                        if t_l < best_t_low or (np.isclose(t_l, best_t_low) and t_h < best_t_high):
                            is_better = True
                            
                if is_better:
                    best_wr = wr
                    best_mcfr = mcfr
                    best_t_low = t_l
                    best_t_high = t_h
                    
    # In case no threshold pair satisfied the constraint (e.g. extremely strict alpha)
    # fall back to most conservative: t_low = 0.0, t_high = 0.01
    if best_wr < -0.5:
        best_t_low = 0.0
        best_t_high = 0.01
        metrics = compute_routing_metrics(probabilities, true_labels, best_t_low, best_t_high)
        best_wr = metrics["workload_reduction"]
        best_mcfr = metrics["missed_critical_fault_rate"]
        
    return best_t_low, best_t_high, best_wr, best_mcfr

def run_sensitivity_analysis(val_probs, val_labels, test_probs, test_labels, alphas=[0.01, 0.03, 0.05, 0.10]):
    """
    Runs routing optimization on validation probabilities for multiple alpha levels,
    then applies them to the test set to examine workload vs safety trade-off.
    """
    results = []
    
    for alpha in alphas:
        t_low, t_high, val_wr, val_mcfr = optimize_thresholds(val_probs, val_labels, alpha=alpha)
        
        # Apply to test set
        test_metrics = compute_routing_metrics(test_probs, test_labels, t_low, t_high)
        
        results.append({
            "alpha": alpha,
            "t_low": t_low,
            "t_high": t_high,
            "val_workload_reduction": val_wr,
            "val_missed_fault_rate": val_mcfr,
            "test_workload_reduction": test_metrics["workload_reduction"],
            "test_missed_fault_rate": test_metrics["missed_critical_fault_rate"],
            "test_auto_clear_rate": test_metrics["auto_clear_rate"],
            "test_human_review_rate": test_metrics["human_review_rate"],
            "test_urgent_inspection_rate": test_metrics["urgent_inspection_rate"]
        })
        
    return pd.DataFrame(results)
