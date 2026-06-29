from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = Path("t2m_multivar_model_metrics.csv")
LEAD = Path("t2m_multivar_model_lead_metrics.csv")
IMP = Path("t2m_multivar_feature_importance.csv")

OUTDIR = Path("figures_t2m_multivar")
OUTDIR.mkdir(exist_ok=True)

metrics = pd.read_csv(METRICS)
lead = pd.read_csv(LEAD)
imp = pd.read_csv(IMP)

print("Metrics:")
print(metrics.to_string(index=False))

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
def label(row):
    if row["model"] == "C3S ensemble mean":
        return "C3S baseline"
    if row["model"] == "ERA5 monthly local climatology":
        return "ERA5 climatology"
    return f"{row['model'].replace(' residual','')} ({row['feature_set']})"

metrics["label"] = metrics.apply(label, axis=1)
lead["label"] = lead.apply(label, axis=1)

order = [
    "C3S baseline",
    "ERA5 climatology",
    "LightGBM (T2M-only)",
    "XGBoost (T2M-only)",
    "LightGBM (Multivar)",
    "XGBoost (Multivar)",
]

metrics["label"] = pd.Categorical(metrics["label"], categories=order, ordered=True)
metrics = metrics.sort_values("label")

# ---------------------------------------------------------
# 1. Global metric plots
# ---------------------------------------------------------
for metric, ylabel in [
    ("MAE_C", "MAE (°C)"),
    ("RMSE_C", "RMSE (°C)"),
    ("Bias_C", "Bias (°C)"),
    ("R2", "R²"),
]:
    plt.figure(figsize=(10, 5))
    plt.bar(metrics["label"].astype(str), metrics[metric])
    plt.ylabel(ylabel)
    plt.title(f"T2M multivariate model comparison: {metric}")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_t2m_multivar_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 2. Combined MAE/RMSE
# ---------------------------------------------------------
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 5))
plt.bar(x - width / 2, metrics["MAE_C"], width, label="MAE")
plt.bar(x + width / 2, metrics["RMSE_C"], width, label="RMSE")
plt.xticks(x, metrics["label"].astype(str), rotation=25, ha="right")
plt.ylabel("Error (°C)")
plt.title("T2M multivariate model comparison: MAE and RMSE")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_t2m_multivar_mae_rmse_combined.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 3. Improvement vs C3S baseline
# ---------------------------------------------------------
base = metrics.loc[metrics["label"] == "C3S baseline"].iloc[0]

impr = metrics.copy()
impr["MAE_reduction_percent"] = 100 * (base["MAE_C"] - impr["MAE_C"]) / base["MAE_C"]
impr["RMSE_reduction_percent"] = 100 * (base["RMSE_C"] - impr["RMSE_C"]) / base["RMSE_C"]

plt.figure(figsize=(11, 5))
plt.bar(impr["label"].astype(str), impr["RMSE_reduction_percent"])
plt.ylabel("RMSE reduction vs C3S baseline (%)")
plt.title("Relative RMSE improvement over raw C3S")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_t2m_multivar_rmse_reduction_vs_c3s.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

impr.to_csv(OUTDIR / "t2m_multivar_metrics_with_improvement.csv", index=False)
print("Saved:", OUTDIR / "t2m_multivar_metrics_with_improvement.csv")

# ---------------------------------------------------------
# 4. Lead-time RMSE / MAE / Bias
# ---------------------------------------------------------
lead["label"] = pd.Categorical(lead["label"], categories=order, ordered=True)

for metric, ylabel in [
    ("MAE_C", "MAE (°C)"),
    ("RMSE_C", "RMSE (°C)"),
    ("Bias_C", "Bias (°C)"),
    ("R2", "R²"),
]:
    plt.figure(figsize=(10, 5))

    for model_label in order:
        g = lead[lead["label"] == model_label].sort_values("lead_days")
        if g.empty:
            continue
        plt.plot(g["lead_days"], g[metric], marker="o", label=model_label)

    plt.xlabel("Lead time (days)")
    plt.ylabel(ylabel)
    plt.title(f"T2M multivariate comparison by lead time: {metric}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_t2m_multivar_lead_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 5. Feature importance for multivar models
# ---------------------------------------------------------
for model_name in ["LightGBM residual", "XGBoost residual"]:
    g = imp[(imp["model"] == model_name) & (imp["feature_set"] == "Multivar")].copy()
    if g.empty:
        continue

    g = g.sort_values("importance_normalized", ascending=False).head(20)
    g = g.iloc[::-1]

    plt.figure(figsize=(8, 7))
    plt.barh(g["feature"], g["importance_normalized"])
    plt.xlabel("Normalized importance")
    plt.title(f"Top multivariate predictors: {model_name.replace(' residual','')}")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe = model_name.lower().replace(" ", "_").replace("residual", "multivar")
    out = OUTDIR / f"fig_feature_importance_{safe}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 6. Grouped importance by physical variable family
# ---------------------------------------------------------
def family(feat):
    if feat.startswith("t2m_"):
        return "T2M"
    if feat.startswith("d2m_"):
        return "Dewpoint"
    if feat.startswith("msl_"):
        return "MSLP"
    if feat.startswith("stl1_"):
        return "Soil temperature"
    if feat.startswith("swvl1_"):
        return "Soil moisture"
    if feat.startswith("e_"):
        return "Evaporation"
    if feat in ["lead_days"]:
        return "Lead"
    if feat in ["latitude", "longitude"]:
        return "Location"
    if feat in ["valid_month", "month_sin", "month_cos"]:
        return "Seasonality"
    return "Other"

imp["family"] = imp["feature"].apply(family)

fam = (
    imp[imp["feature_set"] == "Multivar"]
    .groupby(["model", "family"], as_index=False)["importance_normalized"]
    .sum()
)

fam.to_csv(OUTDIR / "t2m_multivar_importance_by_family.csv", index=False)
print("Saved:", OUTDIR / "t2m_multivar_importance_by_family.csv")

for model_name in ["LightGBM residual", "XGBoost residual"]:
    g = fam[fam["model"] == model_name].sort_values("importance_normalized", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(g["family"], g["importance_normalized"])
    plt.xlabel("Total normalized importance")
    plt.title(f"Predictor-family importance: {model_name.replace(' residual','')}")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe = model_name.lower().replace(" ", "_").replace("residual", "families")
    out = OUTDIR / f"fig_importance_families_{safe}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

print("\nDone. Figures saved in:", OUTDIR)
