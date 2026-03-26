"""
Shared utilities for model evaluation and comparison.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def compute_aurocs(y_true, y_prob):
    """
    Compute per-class and macro AUROC for a given dataset.

    Args:
        y_true: (N, C) binary ground truth
        y_prob: (N, C) predicted probabilities

    Returns:
        Tuple of (macro_auroc, per_class_auroc_dict)
    """
    per_class = {}
    for i, label in enumerate(LABEL_COLS):
        # Handle cases where a class has only one outcome in a bootstrap sample
        if len(np.unique(y_true[:, i])) < 2:
            per_class[label] = float("nan")
            continue
        try:
            per_class[label] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            per_class[label] = float("nan")

    valid_aucs = [v for v in per_class.values() if not np.isnan(v)]
    macro = np.mean(valid_aucs) if valid_aucs else float("nan")
    return macro, per_class


def bootstrap_auroc_ci(y_true, y_prob, n_bootstraps=2000, confidence=0.95, seed=42):
    """
    Bootstrap 95% CI for per-class and macro AUROC.
    """
    rng = np.random.RandomState(seed)
    n_samples = y_true.shape[0]
    alpha = (1 - confidence) / 2

    macro_boots = []
    per_class_boots = {label: [] for label in LABEL_COLS}

    print(f"Computing bootstrap CIs ({n_bootstraps} iterations)...")
    for _ in tqdm(range(n_bootstraps), desc="Bootstrap CI"):
        idx = rng.randint(0, n_samples, size=n_samples)
        macro_b, per_class_b = compute_aurocs(y_true[idx], y_prob[idx])
        macro_boots.append(macro_b)
        for label in LABEL_COLS:
            per_class_boots[label].append(per_class_b[label])

    ci = {}
    # Macro CI
    arr = np.array(macro_boots)[~np.isnan(macro_boots)]
    if len(arr) > 0:
        ci["macro"] = {"lower": float(np.percentile(arr, 100 * alpha)), "upper": float(np.percentile(arr, 100 * (1 - alpha)))}
    else:
        ci["macro"] = {"lower": None, "upper": None}

    # Per-class CI
    for label in LABEL_COLS:
        arr = np.array(per_class_boots[label])[~np.isnan(per_class_boots[label])]
        if len(arr) > 0:
            ci[label] = {"lower": float(np.percentile(arr, 100 * alpha)), "upper": float(np.percentile(arr, 100 * (1 - alpha)))}
        else:
            ci[label] = {"lower": None, "upper": None}
    return ci


def bootstrap_auroc_difference_ci(y_true, y_prob1, y_prob2, n_bootstraps=2000, confidence=0.95, seed=42):
    """
    Bootstrap CI for the difference in macro AUROC between two models.
    """
    rng = np.random.RandomState(seed)
    n_samples = y_true.shape[0]
    alpha = (1 - confidence) / 2

    auroc_diffs = []
    print(f"Computing bootstrap CI for difference ({n_bootstraps} iterations)...")
    for _ in tqdm(range(n_bootstraps), desc="Bootstrap Diff CI"):
        idx = rng.randint(0, n_samples, size=n_samples)
        macro_auc1, _ = compute_aurocs(y_true[idx], y_prob1[idx])
        macro_auc2, _ = compute_aurocs(y_true[idx], y_prob2[idx])
        if not np.isnan(macro_auc1) and not np.isnan(macro_auc2):
            auroc_diffs.append(macro_auc1 - macro_auc2)

    point_estimate_auc1, _ = compute_aurocs(y_true, y_prob1)
    point_estimate_auc2, _ = compute_aurocs(y_true, y_prob2)
    point_estimate_diff = point_estimate_auc1 - point_estimate_auc2

    lower_bound = float(np.percentile(auroc_diffs, 100 * alpha))
    upper_bound = float(np.percentile(auroc_diffs, 100 * (1 - alpha)))

    return {
        "model1_macro_auc": point_estimate_auc1,
        "model2_macro_auc": point_estimate_auc2,
        "difference_macro_auc": point_estimate_diff,
        "confidence_interval": {"lower": lower_bound, "upper": upper_bound},
    }
