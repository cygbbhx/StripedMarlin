import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import pandas as pd


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

def get_metric(y_true, y_prob):
    y_true = y_true.detach().cpu().numpy()
    y_prob = y_prob.detach().cpu().numpy()

    f_true, r_true = y_true[:, 0], y_true[:, 1]
    f_prob, r_prob = y_prob[:, 0], y_prob[:, 1]
    
    labelwise_data = [[f_true, f_prob], [r_true, r_prob]]

    auc_scores = []
    for label_true, label_prob in labelwise_data: 
        auc_scores.append(roc_auc_score(label_true, label_prob))
    
    mean_auc = np.mean(auc_scores)

    brier_scores = []
    ece_scores = []
    
    for label_true, label_prob in labelwise_data: 
        # Brier Score
        brier = mean_squared_error(label_true, label_prob)
        brier_scores.append(brier)
        
        # ECE
        ece = expected_calibration_error(label_true, label_prob)
        ece_scores.append(ece)
    
    # Calculate mean Brier Score and mean ECE
    mean_brier = np.mean(brier_scores)
    mean_ece = np.mean(ece_scores)
    
    # Calculate combined score
    combined_score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece


    return round(mean_auc, 4), round(mean_brier, 4), round(mean_ece, 4), round(combined_score, 4)
