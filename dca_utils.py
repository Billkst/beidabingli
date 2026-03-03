import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_net_benefit(y_true, y_prob, thresholds):
    """
    Calculate Net Benefit for a range of thresholds.
    
    Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
    """
    net_benefits = []
    n = len(y_true)
    
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Net Benefit calculation
        if pt == 1.0:
            nb = 0 # Undefined specific punishment, treat as 0 benefit
        else:
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            
        net_benefits.append(nb)
        
    return np.array(net_benefits)

def calculate_net_benefit_all(y_true):
    """Calculate Net Benefit for 'Treat All' strategy."""
    # Treat All means y_pred is all 1s
    # TP = number of actual positives
    # FP = number of actual negatives
    tp = np.sum(y_true)
    fp = len(y_true) - tp
    n = len(y_true)
    
    return lambda pt: (tp / n) - (fp / n) * (pt / (1 - pt))

def plot_dca(y_true, y_prob, model_name, ax=None, thresholds=None):
    """
    Plot Decision Curve Analysis (DCA).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Model Net Benefit
    nb_model = calculate_net_benefit(y_true, y_prob, thresholds)
    
    # Treat All
    # NB_all = (Prevalence) - (1-Prevalence) * (pt / (1-pt))
    prevalence = np.mean(y_true)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    
    # Treat None (Net Benefit is always 0)
    nb_none = np.zeros_like(thresholds)
    
    # Plotting
    ax.plot(thresholds, nb_model, label=f'{model_name}', linewidth=2)
    ax.plot(thresholds, nb_all, linestyle=':', label='Treat All', color='gray')
    ax.plot(thresholds, nb_none, linestyle='--', label='Treat None', color='black')
    
    # Limits and Labels
    # Y-axis typically limited to purely positive range slightly extended or just [0, prevalence]
    ymax = max(prevalence * 1.05, np.max(nb_model) * 1.05) if np.max(nb_model) > 0 else prevalence
    ax.set_ylim([-0.05, ymax]) 
    ax.set_xlim([0, 1])
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title(f'Decision Curve Analysis: {model_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return nb_model

def get_youden_metrics(y_true, y_prob):
    """
    Find optimal threshold by Youden Index and return metrics.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]
    
    return {
        'best_threshold': best_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'youden': youden_index[best_idx]
    }
