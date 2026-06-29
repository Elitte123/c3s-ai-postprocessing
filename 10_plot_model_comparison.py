from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = Path("model_comparison_t2m_metrics.csv")
LEAD = Path("model_comparison_t2m_lead_metrics.csv")
IMP = Path("model_comparison_t2m_feature_importance.csv")
PRED = Path("model_comparison_t2m_predictions_sample.parquet")

OUTDIR = Path("figures_t2m_model_comparison")
OUTDIR.mkdir(exist_ok=True)

print("Loading data...")
metrics = pd.read_csv(METRICS)
lead = pd.read_csv(LEAD)
imp = pd.read_csv(IMP)
pred = pd.read_parquet(PRED)

print(metrics)
print(lead.head())
print(imp.head())
print(pred.shape)

# nicer names
name_map = {
    "C3S ensemble mean": "C3S baseline",
    "ERA5 monthly local climatology": "ERA5 climatology",
    "Random Forest residual": "Random Forest",
    "LightGBM residual": "LightGBM",
    "XGBoost residual": "XGBoost",
}

metrics["model_short"] = metrics["model"].map(name_map).fillna(metrics["model"])
lead["model_short"] = lead["model"].map(name_map).fillna(lead["model"])
imp["model_short"] = imp["model"].map(name_map).fillna(imp["model"])

order = [
    "C3S baseline",
    "ERA5 climatology",
    "Random Forest",
    "LightGBM",
    "XGBoost",
]

metrics["model_short"] = pd.Categorical(metrics["model_short"], categories=order, ordered=True)
metrics = metrics.sort_values("model_short")

# ---------------------------------------------------------
# 1. Global metric bar plots
# ---------------------------------------------------------
for metric, ylabel, title in [
    ("MAE_C", "MAE (°C)", "Mean Absolute Error"),
    ("RMSE_C", "RMSE (°C)", "Root Mean Square Error"),
    ("Bias_C", "Bias (°C)", "Mean Bias"),
    ("R2", "R²", "Coefficient of Determination"),
]:
    plt.figure(figsize=(9, 5))
    plt.bar(metrics["model_short"].astype(str), metrics[metric])
    plt.ylabel(ylabel)
    plt.title(f"T2M model comparison: {title}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_model_comparison_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 2. Combined MAE/RMSE
# ---------------------------------------------------------
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, metrics["MAE_C"], width, label="MAE")
plt.bar(x + width / 2, metrics["RMSE_C"], width, label="RMSE")
plt.xticks(x, metrics["model_short"].astype(str), rotation=20, ha="right")
plt.ylabel("Error (°C)")
plt.title("T2M model comparison: MAE and RMSE")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_model_comparison_mae_rmse_combined.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 3. Improvement relative to C3S baseline
# ---------------------------------------------------------
base = metrics.loc[metrics["model_short"] == "C3S baseline"].iloc[0]

impr = metrics.copy()
impr["MAE_reduction_percent"] = 100 * (base["MAE_C"] - impr["MAE_C"]) / base["MAE_C"]
impr["RMSE_reduction_percent"] = 100 * (base["RMSE_C"] - impr["RMSE_C"]) / base["RMSE_C"]

plt.figure(figsize=(10, 5))
plt.bar(impr["model_short"].astype(str), impr["RMSE_reduction_percent"])
plt.ylabel("RMSE reduction vs C3S baseline (%)")
plt.title("Relative improvement over raw C3S baseline")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_rmse_reduction_vs_c3s_baseline.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

impr.to_csv(OUTDIR / "model_comparison_with_improvement.csv", index=False)
print("Saved:", OUTDIR / "model_comparison_with_improvement.csv")

# ---------------------------------------------------------
# 4. Lead-time RMSE comparison
# ---------------------------------------------------------
lead_plot = lead.copy()
lead_plot["model_short"] = pd.Categorical(
    lead_plot["model_short"],
    categories=order,
    ordered=True
)

for metric, ylabel in [
    ("MAE_C", "MAE (°C)"),
    ("RMSE_C", "RMSE (°C)"),
    ("Bias_C", "Bias (°C)"),
    ("R2", "R²"),
]:
    plt.figure(figsize=(9, 5))

    for model in order:
        g = lead_plot[lead_plot["model_short"] == model].sort_values("lead_days")
        if len(g) == 0:
            continue
        plt.plot(g["lead_days"], g[metric], marker="o", label=model)

    plt.xlabel("Lead time (days)")
    plt.ylabel(ylabel)
    plt.title(f"T2M model comparison by lead time: {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_lead_model_comparison_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 5. Feature importance, one figure per ML model
# ---------------------------------------------------------
if not imp.empty:
    for model in ["Random Forest", "LightGBM", "XGBoost"]:
        g = imp[imp["model_short"] == model].copy()
        if g.empty:
            continue

        g = g.sort_values("importance_normalized", ascending=False).head(15)
        g = g.iloc[::-1]

        plt.figure(figsize=(8, 6))
        plt.barh(g["feature"], g["importance_normalized"])
        plt.xlabel("Normalized importance")
        plt.title(f"Feature importance: {model}")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        safe = model.lower().replace(" ", "_")
        out = OUTDIR / f"fig_feature_importance_{safe}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)

# ---------------------------------------------------------
# 6. Error histograms for best models
# ---------------------------------------------------------
error_cols = {
    "C3S baseline": "baseline_error_c",
    "ERA5 climatology": "climatology_error_c",
    "Random Forest": "random_forest_residual_error_c",
    "LightGBM": "lightgbm_residual_error_c",
    "XGBoost": "xgboost_residual_error_c",
}

plt.figure(figsize=(10, 6))
for label, col in error_cols.items():
    if col in pred.columns:
        plt.hist(pred[col], bins=80, alpha=0.35, label=label)

plt.xlabel("Prediction error (°C)")
plt.ylabel("Frequency")
plt.title("Prediction error distributions")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_error_histograms_all_models.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 7. Scatter ERA5 vs best model
# ---------------------------------------------------------
best_model = metrics.sort_values("RMSE_C").iloc[0]["model_short"]
best_col_map = {
    "Random Forest": "random_forest_residual_pred_c",
    "LightGBM": "lightgbm_residual_pred_c",
    "XGBoost": "xgboost_residual_pred_c",
    "ERA5 climatology": "climatology_pred_c",
    "C3S baseline": "baseline_pred_c",
}

best_col = best_col_map.get(str(best_model), None)

if best_col in pred.columns:
    sample = pred.sample(min(50000, len(pred)), random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(sample["t2m_era5_c"], sample[best_col], s=3, alpha=0.25)

    mn = min(sample["t2m_era5_c"].min(), sample[best_col].min())
    mx = max(sample["t2m_era5_c"].max(), sample[best_col].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xlabel("ERA5-Land T2M (°C)")
    plt.ylabel(f"{best_model} predicted T2M (°C)")
    plt.title(f"ERA5-Land vs {best_model} prediction")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / "fig_scatter_era5_vs_best_model.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

print("\nDone. Figures saved in:", OUTDIR)
