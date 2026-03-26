import pandas as pd
import numpy as np
import json
from pathlib import Path
from eval_utils import LABEL_COLS, bootstrap_auroc_difference_ci


def main():
    # ========================== CONFIGURATION ==========================
    project_root = Path(__file__).parent.parent
    data_path = '/Users/s1807328/Desktop/MLP Project/xray-slam-data/grand-xray-slam-division-b'
    labels_csv = f'{data_path}/val2.csv'

    # --- Model 1 ---
    model1_name = "mobilevit_v2_050_tuning"
    expt1_folder = "experiments_eidf/student_baselines/mobilevit"
    
    # --- Model 2 ---
    model2_name = "mobilevit_v2_050_gamma1"
    expt2_folder = "experiments_eidf/student_focal_loss/mobilevit_experiments"

    # --- Output ---
    output_json_path = project_root / "output" / "comparison_mobilenet_vs_mobilevit.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    # ===================================================================

    # Resolve paths
    logits1_csv = project_root / expt1_folder / "outputs" / f"{model1_name}_logits.csv"
    logits2_csv = project_root / expt2_folder / "outputs" / f"{model2_name}_logits.csv"

    print(f"Comparing:")
    print(f"  Model 1: {model1_name} (from {expt1_folder})")
    print(f"  Model 2: {model2_name} (from {expt2_folder})")
    print(f"  Ground Truth: {labels_csv}")

    # Load data
    logits1_df = pd.read_csv(logits1_csv)
    logits2_df = pd.read_csv(logits2_csv)
    labels_df = pd.read_csv(labels_csv)

    # Check for common images
    common_images = set(labels_df["Image_name"]).intersection(
        set(logits1_df["Image_name"])
    ).intersection(
        set(logits2_df["Image_name"])
    )
    
    if len(common_images) == 0:
        print("CRITICAL ERROR: No common images found between Ground Truth, Model 1, and Model 2.")
        print(f"  Labels images: {len(labels_df)}")
        print(f"  Model 1 images: {len(logits1_df)}")
        print(f"  Model 2 images: {len(logits2_df)}")
        print("Please check that both models were evaluated on the same dataset (val2.csv).")
        return

    print(f"Found {len(common_images)} common images across all files.")

    # Merge all three dataframes on Image_name
    merged = labels_df[["Image_name"] + LABEL_COLS].merge(
        logits1_df[["Image_name"] + LABEL_COLS], on="Image_name", suffixes=("_true", "_m1")
    ).merge(
        logits2_df[["Image_name"] + LABEL_COLS], on="Image_name", suffixes=("", "_m2")
    )

    # Prepare arrays
    y_true = merged[[f"{c}_true" for c in LABEL_COLS]].values
    y_score1 = merged[[f"{c}_m1" for c in LABEL_COLS]].values
    y_score2 = merged[[c for c in LABEL_COLS]].values # No suffix from second merge

    # Convert logits to probabilities (sigmoid)
    y_prob1 = 1 / (1 + np.exp(-y_score1))
    y_prob2 = 1 / (1 + np.exp(-y_score2))

    # Compute results
    results = bootstrap_auroc_difference_ci(y_true, y_prob1, y_prob2)

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nComparison results saved to: {output_json_path}")

    # Print summary
    print("\n" + "="*60)
    print("Paired Bootstrap Comparison Results")
    print("="*60)
    print(f"Model 1 ({model1_name}):")
    print(f"  Macro AUROC: {results['model1_macro_auc']:.4f}")
    print(f"\nModel 2 ({model2_name}):")
    print(f"  Macro AUROC: {results['model2_macro_auc']:.4f}")
    print("\n" + "-"*60)
    print("Difference (Model 1 - Model 2):")
    print(f"  Point Estimate: {results['difference_macro_auc']:+.4f}")
    print(f"  95% CI:         [{results['confidence_interval']['lower']:+.4f}, {results['confidence_interval']['upper']:+.4f}]")
    print("="*60)

    if results['confidence_interval']['lower'] * results['confidence_interval']['upper'] > 0:
        winner = model1_name if results['difference_macro_auc'] > 0 else model2_name
        print(f"\nConclusion: The difference is statistically significant.")
        print(f"'{winner}' performs better.")
    else:
        print("\nConclusion: The difference is not statistically significant (CI includes 0).")

if __name__ == "__main__":
    main()