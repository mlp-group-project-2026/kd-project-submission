import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

kd_best_teacher = pd.read_csv("./results/kd_best_teacher.csv")
kd_ensemble = pd.read_csv("./results/kd_ensemble.csv")
mobile_vit = pd.read_csv("./results/mobilevit_focal_loss.csv")

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x="epoch", y="val_auroc", data=kd_best_teacher, label="Student CheXFound-KD", linestyle='--')
sns.lineplot(x="epoch", y="val_auroc", data=kd_ensemble, label="Student Ensemble-KD", linestyle='-.')
sns.lineplot(x="epoch", y="val_auroc", data=mobile_vit, label="Student Baseline", linestyle=':')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()
plt.savefig(os.path.join("results", "model_comparison.png"))


# Grouped per-class AUC bar chart from test_stats_compiled.csv
test_stats_compiled = pd.read_csv("./results/test_stats_compiled.csv")

per_class_auc_columns = [
    col for col in test_stats_compiled.columns
    if col.startswith("per_class_auc.") and col.endswith(".auc")
 ]
class_order = [col.replace("per_class_auc.", "").replace(".auc", "") for col in per_class_auc_columns]

model_map = {
    "student_focal_loss.json": "Student Baseline",
    "kd_best_teacher.json": "Student CheXFound-KD",
    "kd_ensemble.json": "Student Ensemble-KD",
}
filtered = test_stats_compiled[test_stats_compiled["relative_path"].isin(model_map)]

long_df = filtered.melt(
    id_vars=["relative_path"],
    value_vars=per_class_auc_columns,
    var_name="class",
    value_name="auc",
 )
long_df["class"] = long_df["class"].str.replace("per_class_auc.", "", regex=False).str.replace(".auc", "", regex=False)
long_df["model"] = long_df["relative_path"].map(model_map)
long_df["display_class"] = long_df["class"].str.replace(" ", "\n", regex=False)
display_order = [label.replace(" ", "\n") for label in class_order]

plt.figure(figsize=(14, 6))
palette = ["#1b9e77", "#d95f02", "#7570b3"]
sns.barplot(
    data=long_df,
    x="display_class",
    y="auc",
    hue="model",
    order=display_order,
    palette=palette,
 )
plt.xlabel("Class")
plt.ylabel("AUC")
plt.title("Per-class AUC (Test Set)")
plt.xticks(rotation=0, ha="center", fontsize=9)
plt.ylim(0.8, 1.0)
plt.legend(title="Model")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join("results", "per_class_auc_comparison.png"))

# Bland-Altman style plot: Student Focal Loss vs KD Ensemble (per-class AUC)
focal_row = test_stats_compiled.loc[
    test_stats_compiled["relative_path"] == "student_focal_loss.json",
    per_class_auc_columns,
].iloc[0]
kd_row = test_stats_compiled.loc[
    test_stats_compiled["relative_path"] == "kd_ensemble.json",
    per_class_auc_columns,
].iloc[0]

ba_df = pd.DataFrame({
    "class": class_order,
    "auc_focal": focal_row.values,
    "auc_kd_ensemble": kd_row.values,
})
ba_df["mean_auc"] = (ba_df["auc_focal"] + ba_df["auc_kd_ensemble"]) / 2
ba_df["diff_auc"] = ba_df["auc_kd_ensemble"] - ba_df["auc_focal"]

plt.figure(figsize=(7.5, 7.5))
palette = plt.cm.tab20.colors
for idx, row in ba_df.iterrows():
    plt.scatter(
        row["mean_auc"],
        row["diff_auc"],
        color=palette[idx % len(palette)],
        label=row["class"],
    )
plt.axhline(0, color="#666666", linestyle="--", linewidth=1)
plt.xlabel("Mean AUC (Student Baseline, Student Ensemble-KD)")
plt.ylabel("AUC Difference (Student Ensemble-KD - Student Baseline)")
plt.title("Bland-Altman Plot: Student Ensemble-KD vs Student Baseline")
plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join("results", "bland_altman_plot.png"))

print("All graphs created and saved in the results directory.")