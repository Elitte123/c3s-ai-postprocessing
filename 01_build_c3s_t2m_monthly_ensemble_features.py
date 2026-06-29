from pathlib import Path
import pandas as pd
import numpy as np

INV = Path("inventory_c3s_t2m_monthly_leads.csv")
OUT = Path("c3s_t2m_monthly_ensemble_features.parquet")

inv = pd.read_csv(INV)

all_parts = []

for i, row in inv.iterrows():
    f = row["file"]
    fp = row["forecast_period"]

    print(f"[{i+1}/{len(inv)}] {fp} | {f}", flush=True)

    df = pd.read_parquet(f)

    # Kelvin -> Celsius
    df["t2m_c"] = df["variable_value"].astype("float32") - 273.15

    # one file should contain one forecast_reference_time and one valid_time
    group_cols = [
        "forecast_reference_time",
        "valid_time",
        "latitude",
        "longitude",
    ]

    g = df.groupby(group_cols)["t2m_c"]

    feat = g.agg(
        t2m_ens_mean="mean",
        t2m_ens_std="std",
        t2m_ens_min="min",
        t2m_ens_max="max",
        t2m_ens_median="median",
    ).reset_index()

    q = g.quantile([0.10, 0.25, 0.75, 0.90]).unstack()
    q.columns = ["t2m_ens_p10", "t2m_ens_p25", "t2m_ens_p75", "t2m_ens_p90"]
    q = q.reset_index()

    feat = feat.merge(q, on=group_cols, how="left")

    feat["init_year"] = int(row["year"])
    feat["init_month"] = int(row["month"])
    feat["forecast_period"] = fp

    # forecast lead in days
    feat["lead_days"] = float(fp.split(" days")[0])

    all_parts.append(feat)

out = pd.concat(all_parts, ignore_index=True)

out["valid_year"] = pd.to_datetime(out["valid_time"]).dt.year
out["valid_month"] = pd.to_datetime(out["valid_time"]).dt.month
out["month_sin"] = np.sin(2 * np.pi * out["valid_month"] / 12)
out["month_cos"] = np.cos(2 * np.pi * out["valid_month"] / 12)

print("\nFinal shape:", out.shape)
print(out.head())
print(out.dtypes)

out.to_parquet(OUT, index=False)
print(f"\nSaved: {OUT}")
