import pandas as pd
import numpy as np
import json
import itertools
from pathlib import Path
from eval_utils import LABEL_COLS, compute_aurocs, bootstrap_auroc_ci

def load_data(logit_files, labels_csv):
    """
    Load and align data from multiple logit files and a label file.
    
    Returns:
        y_true: Ground truth labels (N, C)
        all_probs: List of probability arrays [(N, C), (N, C), ...]
        sorted_images: List of image names used
    """
    print(f"Loading data for {len(logit_files)} models...")
    
    # 1. Identify common images
    labels_df_raw = pd.read_csv(labels_csv)
    common_images = set(labels_df_raw["Image_name"])
    
    for f in logit_files:
        try:
            df = pd.read_csv(f, usecols=["Image_name"])
            common_images = common_images.intersection(set(df["Image_name"]))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            raise

    if not common_images:
        raise ValueError("No common images found!")
    
    sorted_images = sorted(list(common_images))
    print(f"Found {len(sorted_images)} common images based on intersection.")

    # 2. Load ground truth
    labels_df = labels_df_raw.set_index("Image_name").loc[sorted_images].reset_index()
    y_true = labels_df[LABEL_COLS].values
    
    # 3. Load probabilities
    all_probs = []
    for f in logit_files:
        print(f"  Loading predictions from {f}...")
        df = pd.read_csv(f).set_index("Image_name").loc[sorted_images].reset_index()
        logits = df[LABEL_COLS].values
        probs = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
        
    return y_true, all_probs, sorted_images

def generate_weights(n_models, step=0.1):
    """
    Generate valid weight combinations summing to 1.0 (approx).
    """
    options = np.arange(0, 1.0 + step/2, step)
    # Generate cartesian product
    for combo in itertools.product(options, repeat=n_models):
        if np.isclose(sum(combo), 1.0):
            yield combo

def optimise_ensemble(logit_files, labels_csv, output_path, step=0.1):
    # Load data
    y_true, all_probs, _ = load_data(logit_files, labels_csv)
    n_models = len(logit_files)
    
    print(f"\nStarting grid search for weights (step={step})...")
    
    best_macro_auc = -1
    best_weights = None
    best_per_class = None
    
    count = 0
    # Convert list of arrays to a single stacked array for faster computation: (M, N, C)
    stacked_probs = np.stack(all_probs, axis=0)
    
    weight_gen = generate_weights(n_models, step)
    
    for weights in weight_gen:
        count += 1
        # Compute weighted average
        # weights shape: (M,)
        # stacked_probs shape: (M, N, C)
        # We need sum(w_m * P_mnc) over m
        
        # Reshape weights to (M, 1, 1) for broadcasting
        # w_array = np.array(weights).reshape(-1, 1, 1) # OLD
        # ensemble_probs = np.sum(w_array * stacked_probs, axis=0) # OLD
        
        # Optimized: tensordot over the model dimension (axis 0 of stacked_probs, axis 0 of weights)
        ensemble_probs = np.tensordot(weights, stacked_probs, axes=([0], [0]))
        
        # Compute metric (only Macro AUC for speed during search)
        macro_auc, per_class = compute_aurocs(y_true, ensemble_probs)
        
        # Check if better
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_weights = weights
            best_per_class = per_class
            print(f"  New best: {best_macro_auc:.5f} with weights {weights}")

    print(f"\nSearch complete. Checked {count} combinations.")
    print(f"Best Macro AUROC: {best_macro_auc:.5f}")
    print(f"Optimal Weights: {best_weights}")
    
    # Calculate intervals for the best model
    print("Computing bootstrap confidence intervals for the optimal ensemble...")
    # best_w_array = np.array(best_weights).reshape(-1, 1, 1) # OLD
    # best_ensemble_probs = np.sum(best_w_array * stacked_probs, axis=0) # OLD
    best_ensemble_probs = np.tensordot(best_weights, stacked_probs, axes=([0], [0]))

    ci = bootstrap_auroc_ci(y_true, best_ensemble_probs, n_bootstraps=2000)
    
    results = {
        "optimal_weights": list(best_weights),
        "macro_auc": best_macro_auc,
        "per_class_auc": best_per_class,
        "confidence_intervals": ci
    }

    # Save results
    save_results(results, output_path)

def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _safe(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        return round(v, 6)
    
    serialisable = {
        "optimal_weights": results["optimal_weights"],
        "macro_auc": _safe(results["macro_auc"]),
        "per_class_auc": {k: _safe(v) for k, v in results["per_class_auc"].items()},
        "confidence_intervals": {} 
    }
    
    # Add CI formatting similar to evaluation.py
    ci = results["confidence_intervals"]
    macro_ci = ci.get("macro", {})
    serialisable["macro_auc_95ci"] = {
        "lower": _safe(macro_ci.get("lower")),
        "upper": _safe(macro_ci.get("upper"))
    }
    
    for label in LABEL_COLS:
        label_ci = ci.get(label, {})
        if label not in serialisable["confidence_intervals"]:
             serialisable["confidence_intervals"][label] = {}
        
        serialisable["confidence_intervals"][label] = {
             "lower": _safe(label_ci.get("lower")), 
             "upper": _safe(label_ci.get("upper"))
        }

    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)
        
    print(f"\nOptimisation results saved to {output_path}")       

if __name__ == "__main__":
    def main():
        # ========================== CONFIGURATION ==========================
        # List of logit files to optimise
        logit_files = [
            "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f5_fullset_chexfound_logits_list.csv",
            "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f6_fullset_chexfound_logits_list.csv",
            "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f5_fullset_evax_logits_list.csv",
            "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f6_fullset_evax_logits_list.csv",
        ]
        
        # Path to ground truth labels
        data_path = './MLP Project/xray-slam-data/grand-xray-slam-division-b'
        labels_csv = f'{data_path}/val2.csv'
        
        # Output path
        output_path = "experiments_eidf/ensemble/outputs/optimised_ensemble.json"
        
        step_size = 0.1
        # ===================================================================

        if not logit_files:
            print("Please add paths to logit CSV files in the 'logit_files' list within the script.")
            return

        optimise_ensemble(
            logit_files=logit_files,
            labels_csv=labels_csv,
            output_path=output_path,
            step=step_size
        )

    main()
