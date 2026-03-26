import pandas as pd
import numpy as np
import json
from pathlib import Path
from eval_utils import LABEL_COLS, compute_aurocs, bootstrap_auroc_ci
import argparse


def evaluate(logits_csv, labels_csv, n_bootstraps=2000):
    """
    Evaluate model predictions against ground truth labels.

    Args:
        logits_csv:    Path to CSV with columns [Image_name, <label_cols>...] containing raw logits
        labels_csv:    Path to CSV with columns [Image_name, ..., <label_cols>...] containing binary labels
        n_bootstraps:  Number of bootstrap iterations for confidence intervals

    Returns:
        Dict with macro AUROC, per-class AUROC scores, and 95% bootstrap CIs
    """
    logits_df = pd.read_csv(logits_csv)
    labels_df = pd.read_csv(labels_csv)

    # Merge on Image_name to ensure alignment
    merged = labels_df[["Image_name"] + LABEL_COLS].merge(
        logits_df[["Image_name"] + LABEL_COLS],
        on="Image_name",
        suffixes=("_true", "_pred"),
    )
    
    if len(merged) == 0:
        print("WARNING: Merged dataframe is empty! Check Image_name overlap.")
        return {
            "macro_auc": float("nan"),
            "per_class_auc": {c: float("nan") for c in LABEL_COLS},
            "confidence_intervals": {},
        }
    
    print(f"Evaluated on {len(merged)} images.")

    y_true = merged[[f"{c}_true" for c in LABEL_COLS]].values
    y_score = merged[[f"{c}_pred" for c in LABEL_COLS]].values

    # Convert logits to probabilities
    y_prob = 1 / (1 + np.exp(-y_score))  # sigmoid

    # Point estimates
    macro_auc, per_class_auc = compute_aurocs(y_true, y_prob)

    # Bootstrap 95% confidence intervals
    ci = bootstrap_auroc_ci(y_true, y_prob, n_bootstraps=n_bootstraps)

    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "confidence_intervals": ci,
    }


def save_results(results, output_path):
    """Save evaluation results (with bootstrap CIs) to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--expt_folder", type=str, required=True, help="Experiment folder relative to project root")
    parser.add_argument("--data_path", type=str, default='/Users/s1807328/Desktop/MLP Project/xray-slam-data/grand-xray-slam-division-b', help="Base path for data")
    parser.add_argument("--csv_file", type=str, default='val2.csv', help="CSV file name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ========================== CONFIGURATION ==========================
    # Specify the experiment path (relative to project root)
    model_name = args.model_name
    expt_folder = args.expt_folder

    # Data paths - keep or override as needed
    data_path = args.data_path
    labels_csv = f'{data_path}/{args.csv_file}'

    # Auto-infer paths based on expt_folder
    project_root = Path(__file__).parent.parent

    if expt_folder.endswith("teacher_inference"):
        output_dir = project_root / expt_folder / "outputs/full_val_set"
        # Input/Output paths
        logits_csv = output_dir / f"{model_name}_logits_list.csv"
        output_path = output_dir / f"{model_name}_eval.json"
    else:
        output_dir = project_root / expt_folder / "outputs"
        logits_csv = output_dir / f"{model_name}_logits.csv"
        output_path = output_dir / f"{model_name}_eval.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===================================================================

    print(f"Reading logits from: {logits_csv}")
    print(f"Reading labels from: {labels_csv}")
    
    if not logits_csv.exists():
        raise FileNotFoundError(f"Logits file not found: {logits_csv}")

    results = evaluate(logits_csv, labels_csv)
    save_results(results, output_path)

if __name__ == "__main__":
    main()
