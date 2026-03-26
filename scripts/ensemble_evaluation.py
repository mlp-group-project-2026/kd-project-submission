import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from eval_utils import LABEL_COLS, compute_aurocs, bootstrap_auroc_ci

def save_results(results, output_path):
    """Save evaluation results (with bootstrap CIs) to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(v):
        """Convert NaN / non-finite to None for JSON."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(v, 6)

    ci = results.get("confidence_intervals", {})

    macro_ci = ci.get("macro", {})
    serialisable = {
        "macro_auc": _safe(results["macro_auc"]),
        "macro_auc_95ci": {
            "lower": _safe(macro_ci.get("lower")),
            "upper": _safe(macro_ci.get("upper")),
        },
        "per_class_auc": {},
    }

    for label in LABEL_COLS:
        auc_val = results["per_class_auc"].get(label, float("nan"))
        label_ci = ci.get(label, {})
        serialisable["per_class_auc"][label] = {
            "auc": _safe(auc_val),
            "95ci_lower": _safe(label_ci.get("lower")),
            "95ci_upper": _safe(label_ci.get("upper")),
        }

    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  {'Label':<30} {'AUC':>8}  {'95% CI':>20}")
    print(f"{'='*65}")
    for label in LABEL_COLS:
        entry = serialisable["per_class_auc"][label]
        auc_str = f"{entry['auc']:.4f}" if entry["auc"] is not None else "N/A"
        lo = entry["95ci_lower"]
        hi = entry["95ci_upper"]
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else "N/A"
        print(f"  {label:<30} {auc_str:>8}  {ci_str:>20}")
    print(f"{'-'*65}")
    macro_lo = serialisable["macro_auc_95ci"]["lower"]
    macro_hi = serialisable["macro_auc_95ci"]["upper"]
    macro_str = f"{serialisable['macro_auc']:.4f}" if serialisable["macro_auc"] is not None else "N/A"
    macro_ci_str = f"[{macro_lo:.4f}, {macro_hi:.4f}]" if macro_lo is not None and macro_hi is not None else "N/A"
    print(f"  {'Macro AUROC':<30} {macro_str:>8}  {macro_ci_str:>20}")
    print(f"{'='*65}\n")

    print(f"Results saved to {output_path}")

def evaluate_ensemble(logit_files, labels_csv, output_path, n_bootstraps=2000):
    print(f"Initializing ensemble evaluation with {len(logit_files)} models...")
    
    # 1. Identify common images across all logit files and the label file
    print("Reading files to find common images...")
    
    # Start with labels
    labels_df_raw = pd.read_csv(labels_csv)
    common_images = set(labels_df_raw["Image_name"])
    
    # Intersect with each logit file
    for f in logit_files:
        try:
            df = pd.read_csv(f, usecols=["Image_name"])
            common_images = common_images.intersection(set(df["Image_name"]))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            raise

    if not common_images:
        raise ValueError("No common images found across all provided files!")
    
    sorted_images = sorted(list(common_images))
    print(f"Found {len(sorted_images)} common images for evaluation.")

    # 2. Load ground truth for common images
    labels_df = labels_df_raw.set_index("Image_name").loc[sorted_images].reset_index()
    y_true = labels_df[LABEL_COLS].values
    
    # 3. Load probabilities for each model
    all_probs = []
    for f in logit_files:
        print(f"Loading predictions from {f}...")
        df = pd.read_csv(f).set_index("Image_name").loc[sorted_images].reset_index()
        
        # Check for missing columns
        missing_cols = [c for c in LABEL_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"File {f} is missing columns: {missing_cols}")
            
        logits = df[LABEL_COLS].values
        # Convert to probabilities (sigmoid)
        probs = 1 / (1 + np.exp(-logits))
        all_probs.append(probs)
    
    # 4. Average probabilities
    print("Averaging probabilities...")
    avg_probs = np.mean(all_probs, axis=0)
    
    # 5. Compute metrics
    print("Computing AUROC metrics...")
    macro_auc, per_class_auc = compute_aurocs(y_true, avg_probs)
    
    print(f"Computing bootstrap confidence intervals ({n_bootstraps} iterations)...")
    ci = bootstrap_auroc_ci(y_true, avg_probs, n_bootstraps=n_bootstraps)
    
    results = {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "confidence_intervals": ci,
    }
    
    save_results(results, output_path)


def main():
    # ========================== CONFIGURATION ==========================
    # List of logit files to ensemble
    logit_files = [
        "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f5_fullset_chexfound_logits_list.csv",
        "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f6_fullset_chexfound_logits_list.csv",
        "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f5_fullset_evax_logits_list.csv",
        "experiments_eidf/teacher_inference/outputs/full_val_set/ema_model_f6_fullset_evax_logits_list.csv",
    ]
    
    # Path to ground truth labels
    data_path = './MLP Project/xray-slam-data/grand-xray-slam-division-b'
    labels_csv = f'{data_path}/val2.csv'
    
    # Output path for the ensemble evaluation
    output_path = "experiments_eidf/teacher_inference/outputs/full_val_set/ensemble_eval.json"
    
    n_bootstraps = 2000
    # ===================================================================

    if not logit_files:
        print("Please add paths to logit CSV files in the 'logit_files' list within the script.")
        return

    evaluate_ensemble(
        logit_files=logit_files,
        labels_csv=labels_csv,
        output_path=output_path,
        n_bootstraps=n_bootstraps
    )

if __name__ == "__main__":
    main()
