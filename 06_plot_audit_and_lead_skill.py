from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AUDIT = Path("audit_t2m_benchmarks.csv")
PRED = Path("predictions_t2m_rf_test_2021_2024.parquet")
TRAINING = Path("training_t2m_c3s_era5_land.parquet")

OUTDIR = Path("figures_t2m_audit")
OUTDIR.mkdir(exist_ok=True)

print("Loading data...")
audit = pd.read_csv(AUDIT)
pred = pd.read_parquet(PRED)
df = pd.read_parquet(TRAINING)

print("Audit:")
print(audit)
print("Pred:", pred.shape)
print("Training:", df.shape)

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def rmse(x):
    return np.sqrt(np.mean(x ** 2))

def calc_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE_C": np.mean(np.abs(err)),
        "RMSE_C": np.sqrt(np.mean(err ** 2)),
        "Bias_C": np.mean(err),
    }

# ---------------------------------------------------------
# 1. Audit bar plots: MAE, RMSE, Bias, R2
# ---------------------------------------------------------
audit_plot = audit.copy()
audit_plot["model_short"] = [
    "C3S baseline",
    "ERA5 climatology",
    "RF geo-seasonal",
    "RF full"
]

for metric in ["MAE_C", "RMSE_C", "Bias_C", "R2"]:
    plt.figure(figsize=(9, 5))
    plt.bar(audit_plot["model_short"], audit_plot[metric])
    plt.ylabel(metric)
    plt.title(f"Audit benchmark comparison: {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_audit_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# Combined MAE/RMSE plot
x = np.arange(len(audit_plot))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, audit_plot["MAE_C"], width, label="MAE")
plt.bar(x + width/2, audit_plot["RMSE_C"], width, label="RMSE")
plt.xticks(x, audit_plot["model_short"], rotation=20, ha="right")
plt.ylabel("Error (°C)")
plt.title("Audit benchmark comparison: MAE and RMSE")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out = OUTDIR / "fig_audit_mae_rmse_combined.png"
plt.savefig(out, dpi=300)
plt.close()
print("Saved:", out)

# ---------------------------------------------------------
# 2. Lead-time skill for baseline and RF residual model
# ---------------------------------------------------------
lead_metrics = []

for lead, g in pred.groupby("lead_days"):
    m_base = calc_metrics(g["t2m_era5_c"], g["baseline_pred_c"])
    m_rf = calc_metrics(g["t2m_era5_c"], g["rf_corrected_pred_c"])

    lead_metrics.append({
        "lead_days": lead,
        "model": "C3S baseline",
        **m_base
    })
    lead_metrics.append({
        "lead_days": lead,
        "model": "RF residual correction",
        **m_rf
    })

lead_df = pd.DataFrame(lead_metrics)
lead_df.to_csv(OUTDIR / "lead_skill_baseline_rf.csv", index=False)
print("Saved:", OUTDIR / "lead_skill_baseline_rf.csv")
print(lead_df)

for metric in ["MAE_C", "RMSE_C", "Bias_C"]:
    plt.figure(figsize=(8, 5))

    for model, g in lead_df.groupby("model"):
        g = g.sort_values("lead_days")
        plt.plot(g["lead_days"], g[metric], marker="o", label=model)

    plt.xlabel("Lead time (days)")
    plt.ylabel(metric)
    plt.title(f"Skill by lead time: {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_lead_skill_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 3. Yearly performance 2021-2024
# ---------------------------------------------------------
year_metrics = []

for year, g in pred.groupby("valid_year"):
    m_base = calc_metrics(g["t2m_era5_c"], g["baseline_pred_c"])
    m_rf = calc_metrics(g["t2m_era5_c"], g["rf_corrected_pred_c"])

    year_metrics.append({
        "valid_year": year,
        "model": "C3S baseline",
        **m_base
    })
    year_metrics.append({
        "valid_year": year,
        "model": "RF residual correction",
        **m_rf
    })

year_df = pd.DataFrame(year_metrics)
year_df.to_csv(OUTDIR / "yearly_skill_baseline_rf.csv", index=False)
print("Saved:", OUTDIR / "yearly_skill_baseline_rf.csv")
print(year_df)

for metric in ["MAE_C", "RMSE_C", "Bias_C"]:
    plt.figure(figsize=(8, 5))

    for model, g in year_df.groupby("model"):
        g = g.sort_values("valid_year")
        plt.plot(g["valid_year"], g[metric], marker="o", label=model)

    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.title(f"Skill by test year: {metric}")
    plt.xticks(sorted(year_df["valid_year"].unique()))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_yearly_skill_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

# ---------------------------------------------------------
# 4. ERA5 climatology by lead/time on full test set
# ---------------------------------------------------------
train = df[df["valid_year"] <= 2020].copy()
test = df[df["valid_year"] >= 2021].copy()

clim = (
    train
    .groupby(["valid_month", "latitude", "longitude"], as_index=False)["t2m_era5_c"]
    .mean()
    .rename(columns={"t2m_era5_c": "era5_climatology_c"})
)

test_clim = test.merge(
    clim,
    on=["valid_month", "latitude", "longitude"],
    how="left"
).dropna(subset=["era5_climatology_c"])

test_clim["clim_error_c"] = test_clim["era5_climatology_c"] - test_clim["t2m_era5_c"]
test_clim["baseline_error_c"] = test_clim["t2m_ens_mean"] - test_clim["t2m_era5_c"]

clim_lead_metrics = []

for lead, g in test_clim.groupby("lead_days"):
    clim_lead_metrics.append({
        "lead_days": lead,
        "model": "ERA5 climatology",
        "MAE_C": np.mean(np.abs(g["clim_error_c"])),
        "RMSE_C": rmse(g["clim_error_c"]),
        "Bias_C": np.mean(g["clim_error_c"]),
    })

    clim_lead_metrics.append({
        "lead_days": lead,
        "model": "C3S baseline full test",
        "MAE_C": np.mean(np.abs(g["baseline_error_c"])),
        "RMSE_C": rmse(g["baseline_error_c"]),
        "Bias_C": np.mean(g["baseline_error_c"]),
    })

clim_lead_df = pd.DataFrame(clim_lead_metrics)
clim_lead_df.to_csv(OUTDIR / "lead_skill_climatology_baseline_fulltest.csv", index=False)
print("Saved:", OUTDIR / "lead_skill_climatology_baseline_fulltest.csv")
print(clim_lead_df)

for metric in ["MAE_C", "RMSE_C", "Bias_C"]:
    plt.figure(figsize=(8, 5))

    for model, g in clim_lead_df.groupby("model"):
        g = g.sort_values("lead_days")
        plt.plot(g["lead_days"], g[metric], marker="o", label=model)

    plt.xlabel("Lead time (days)")
    plt.ylabel(metric)
    plt.title(f"Climatology benchmark by lead time: {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / f"fig_climatology_lead_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

print("\nDone. Audit figures saved in:", OUTDIR)
