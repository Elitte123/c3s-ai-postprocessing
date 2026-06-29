from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = Path("metrics_t2m_baseline_vs_rf.csv")
PRED = Path("predictions_t2m_rf_test_2021_2024.parquet")
IMP = Path("feature_importances_t2m_rf.csv")
OUTDIR = Path("figures_t2m")
OUTDIR.mkdir(exist_ok=True)

print("Loading data...")
metrics = pd.read_csv(METRICS)
pred = pd.read_parquet(PRED)
imp = pd.read_csv(IMP)

print("metrics:")
print(metrics)

print("pred shape:", pred.shape)
print("importance:")
print(imp.head())

# ---------------------------------------------------------
# 1. Metrics: MAE / RMSE / Bias
# ---------------------------------------------------------
m = metrics.copy()
m["short_model"] = ["C3S baseline", "RF corrected"]

for metric in ["MAE_C", "RMSE_C", "Bias_C", "R2"]:
    plt.figure(figsize=(7, 5))
    plt.bar(m["short_model"], m[metric])
    plt.ylabel(metric)
    plt.title(f"{metric}: C3S baseline vs Random Forest correction")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUTDIR / f"fig_metric_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 2. Feature importance
# ---------------------------------------------------------
top = imp.head(15).iloc[::-1]

plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance")
plt.title("Random Forest feature importance for T2M residual correction")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
out = OUTDIR / "fig_feature_importance_rf_t2m.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 3. Error histograms
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(pred["baseline_error_c"], bins=80, alpha=0.6, label="C3S baseline")
plt.hist(pred["rf_error_c"], bins=80, alpha=0.6, label="RF corrected")
plt.xlabel("Prediction error (°C)")
plt.ylabel("Frequency")
plt.title("Error distribution: baseline vs RF correction")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = OUTDIR / "fig_error_histogram_baseline_vs_rf.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 4. ERA5 vs predictions scatter
# ---------------------------------------------------------
sample = pred.sample(min(50000, len(pred)), random_state=42)

plt.figure(figsize=(6, 6))
plt.scatter(sample["t2m_era5_c"], sample["baseline_pred_c"], s=3, alpha=0.25, label="C3S baseline")
plt.scatter(sample["t2m_era5_c"], sample["rf_corrected_pred_c"], s=3, alpha=0.25, label="RF corrected")

mn = min(sample["t2m_era5_c"].min(), sample["baseline_pred_c"].min(), sample["rf_corrected_pred_c"].min())
mx = max(sample["t2m_era5_c"].max(), sample["baseline_pred_c"].max(), sample["rf_corrected_pred_c"].max())
plt.plot([mn, mx], [mn, mx], linestyle="--")

plt.xlabel("ERA5-Land T2M (°C)")
plt.ylabel("Predicted T2M (°C)")
plt.title("ERA5-Land vs predicted T2M")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out = OUTDIR / "fig_scatter_era5_vs_predictions.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 5. Monthly RMSE
# ---------------------------------------------------------
def rmse(x):
    return np.sqrt(np.mean(x**2))

monthly = (
    pred
    .groupby("valid_month")
    .agg(
        baseline_rmse=("baseline_error_c", rmse),
        rf_rmse=("rf_error_c", rmse),
        baseline_mae=("baseline_error_c", lambda x: np.mean(np.abs(x))),
        rf_mae=("rf_error_c", lambda x: np.mean(np.abs(x))),
    )
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.plot(monthly["valid_month"], monthly["baseline_rmse"], marker="o", label="C3S baseline")
plt.plot(monthly["valid_month"], monthly["rf_rmse"], marker="o", label="RF corrected")
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("RMSE (°C)")
plt.title("Monthly RMSE: baseline vs RF correction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out = OUTDIR / "fig_monthly_rmse_baseline_vs_rf.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

monthly.to_csv(OUTDIR / "monthly_metrics_t2m.csv", index=False)
print("Saved:", OUTDIR / "monthly_metrics_t2m.csv")

# ---------------------------------------------------------
# 6. Spatial bias maps, approximate scatter maps
# ---------------------------------------------------------
spatial = (
    pred
    .groupby(["latitude", "longitude"])
    .agg(
        baseline_bias=("baseline_error_c", "mean"),
        rf_bias=("rf_error_c", "mean"),
        baseline_rmse=("baseline_error_c", rmse),
        rf_rmse=("rf_error_c", rmse),
    )
    .reset_index()
)

for col, title in [
    ("baseline_bias", "Spatial mean error: C3S baseline"),
    ("rf_bias", "Spatial mean error: RF corrected"),
    ("baseline_rmse", "Spatial RMSE: C3S baseline"),
    ("rf_rmse", "Spatial RMSE: RF corrected"),
]:
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        spatial["longitude"],
        spatial["latitude"],
        c=spatial[col],
        s=12
    )
    plt.colorbar(sc, label="°C")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out = OUTDIR / f"fig_spatial_{col}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

spatial.to_csv(OUTDIR / "spatial_metrics_t2m.csv", index=False)
print("Saved:", OUTDIR / "spatial_metrics_t2m.csv")

print("\nDone. Figures saved in:", OUTDIR)
