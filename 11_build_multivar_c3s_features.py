from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/rocky/t1_data")
OUT = Path("c3s_multivar_monthly_ensemble_features.parquet")

TARGET_LEADS = [
    "30 days 00:00:00",
    "60 days 00:00:00",
    "90 days 00:00:00",
    "120 days 00:00:00",
    "150 days 00:00:00",
    "180 days 00:00:00",
]

VARIABLES = {
    "2m_dewpoint_temperature": "d2m",
    "mean_sea_level_pressure": "msl",
    "sea_surface_temperature": "sst",
    "soil_temperature_level_1": "stl1",
    "volumetric_soil_moisture": "swvl1",
    "evaporation": "e",
}

def lead_to_days(s):
    return float(s.split(" days")[0])

all_parts = []

for var_dir_name, short in VARIABLES.items():
    print(f"\n=== VARIABLE: {var_dir_name} -> {short} ===", flush=True)

    var_parts = []

    for year in range(1993, 2025):
        for month in range(1, 13):
            d = ROOT / f"seasonal-original-single-levels_{year}_{month:02d}_{var_dir_name}"

            if not d.exists():
                print("Missing:", d, flush=True)
                continue

            for fp in TARGET_LEADS:
                f = d / f"forecast_period={fp}" / "part.0.parquet"

                if not f.exists():
                    print("Missing file:", f, flush=True)
                    continue

                df = pd.read_parquet(f)

                # units
                # Kelvin variables -> Celsius
                if short in ["d2m", "sst", "stl1"]:
                    values = df["variable_value"].astype("float32") - 273.15
                # pressure Pa -> hPa
                elif short == "msl":
                    values = df["variable_value"].astype("float32") / 100.0
                # evaporation often m accumulated, keep mm equivalent
                elif short == "e":
                    values = df["variable_value"].astype("float32") * 1000.0
                # soil moisture is volumetric fraction, keep as is
                elif short == "swvl1":
                    values = df["variable_value"].astype("float32")
                else:
                    values = df["variable_value"].astype("float32")

                df[f"{short}_value"] = values
                df["lead_days"] = lead_to_days(fp)

                group_cols = [
                    "forecast_reference_time",
                    "valid_time",
                    "lead_days",
                    "latitude",
                    "longitude",
                ]

                g = df.groupby(group_cols)[f"{short}_value"]

                feat = g.agg(
                    **{
                        f"{short}_ens_mean": "mean",
                        f"{short}_ens_std": "std",
                        f"{short}_ens_min": "min",
                        f"{short}_ens_max": "max",
                        f"{short}_ens_median": "median",
                    }
                ).reset_index()

                q = g.quantile([0.10, 0.25, 0.75, 0.90]).unstack()
                q.columns = [
                    f"{short}_ens_p10",
                    f"{short}_ens_p25",
                    f"{short}_ens_p75",
                    f"{short}_ens_p90",
                ]
                q = q.reset_index()

                feat = feat.merge(q, on=group_cols, how="left")
                feat["init_year"] = year
                feat["init_month"] = month

                var_parts.append(feat)

    if not var_parts:
        print("No data for", var_dir_name)
        continue

    var_out = pd.concat(var_parts, ignore_index=True)

    var_out["target_time"] = pd.to_datetime(var_out["valid_time"]).dt.to_period("M").dt.to_timestamp()
    var_out["valid_year"] = var_out["target_time"].dt.year
    var_out["valid_month"] = var_out["target_time"].dt.month

    print(short, "shape:", var_out.shape)
    print(var_out.head())

    all_parts.append((short, var_out))

# merge all variables
print("\nMerging variables...")

base_short, merged = all_parts[0]

merge_keys = [
    "forecast_reference_time",
    "valid_time",
    "lead_days",
    "latitude",
    "longitude",
    "init_year",
    "init_month",
    "target_time",
    "valid_year",
    "valid_month",
]

for short, df in all_parts[1:]:
    cols_to_use = merge_keys + [c for c in df.columns if c.startswith(f"{short}_")]
    merged = merged.merge(df[cols_to_use], on=merge_keys, how="outer")
    print("After merge", short, merged.shape)

merged["month_sin"] = np.sin(2 * np.pi * merged["valid_month"] / 12)
merged["month_cos"] = np.cos(2 * np.pi * merged["valid_month"] / 12)

print("\nFinal shape:", merged.shape)
print(merged.head())
print("\nMissing values:")
print(merged.isna().sum().sort_values(ascending=False).head(30))

merged.to_parquet(OUT, index=False)
print("\nSaved:", OUT)
