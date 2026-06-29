import pandas as pd
import xarray as xr
from pathlib import Path

C3S_FILE = Path("c3s_t2m_monthly_ensemble_features.parquet")
ERA5_FILE = Path("/era5_land_romania/era5_land_t2m_monthly_romania_1993_2024.nc")
OUT = Path("training_t2m_c3s_era5_land.parquet")

print("Loading C3S...")
c3s = pd.read_parquet(C3S_FILE)

# Keep only months covered by ERA5-Land
c3s["target_time"] = pd.to_datetime(c3s["valid_time"]).dt.to_period("M").dt.to_timestamp()
c3s["target_year"] = c3s["target_time"].dt.year
c3s = c3s[c3s["target_year"] <= 2024].copy()

print("C3S shape after target_year <= 2024:", c3s.shape)

feature_cols = [
    "t2m_ens_mean",
    "t2m_ens_std",
    "t2m_ens_min",
    "t2m_ens_max",
    "t2m_ens_median",
    "t2m_ens_p10",
    "t2m_ens_p25",
    "t2m_ens_p75",
    "t2m_ens_p90",
]

print("Checking duplicates...")
dup = c3s.duplicated(
    subset=["target_time", "lead_days", "latitude", "longitude"],
    keep=False
).sum()
print("duplicate rows:", dup)

if dup > 0:
    print("Averaging duplicated target_time/lead/lat/lon rows...")
    c3s = (
        c3s
        .groupby(["target_time", "lead_days", "latitude", "longitude"], as_index=False)[feature_cols]
        .mean()
    )

print("Converting C3S to xarray...")
c3s_ds = c3s.set_index(
    ["target_time", "lead_days", "latitude", "longitude"]
)[feature_cols].to_xarray()

c3s_ds = c3s_ds.rename({"target_time": "valid_time"})

print(c3s_ds)

print("Loading ERA5-Land...")
era = xr.open_dataset(ERA5_FILE)

# Keep only true monthly timestamps
era = era.sortby("valid_time")

# Kelvin -> Celsius
era_t2m = (era["t2m"] - 273.15).rename("t2m_era5_c")

print("ERA5 target:")
print(era_t2m)

print("Interpolating C3S to ERA5-Land grid...")
c3s_interp = c3s_ds.interp(
    latitude=era.latitude,
    longitude=era.longitude,
    method="linear"
)

# Preserve float32 where possible to reduce memory
for v in feature_cols:
    c3s_interp[v] = c3s_interp[v].astype("float32")

print("Merging C3S features with ERA5 target...")
merged = xr.merge([c3s_interp, era_t2m], join="inner")

print(merged)

print("Converting to dataframe...")
df = merged.to_dataframe().reset_index()

df["valid_year"] = pd.to_datetime(df["valid_time"]).dt.year
df["valid_month"] = pd.to_datetime(df["valid_time"]).dt.month

# Drop interpolation margins and any missing target values
df = df.dropna(subset=feature_cols + ["t2m_era5_c"]).copy()

# Optional baseline residual
df["residual_era5_minus_c3s_mean"] = df["t2m_era5_c"] - df["t2m_ens_mean"]

# Remove auxiliary scalar coords if present
for col in ["number", "expver"]:
    if col in df.columns:
        df = df.drop(columns=[col])

print("Final training shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isna().sum())

print("\nDate range:")
print(df["valid_time"].min(), df["valid_time"].max())

print("\nLead days:")
print(sorted(df["lead_days"].unique()))

print("\nGrid:")
print("lat:", df["latitude"].nunique())
print("lon:", df["longitude"].nunique())

df.to_parquet(OUT, index=False)
print(f"\nSaved: {OUT}")
