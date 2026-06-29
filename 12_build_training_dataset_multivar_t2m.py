from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

BASE_TRAINING = Path("training_t2m_c3s_era5_land.parquet")
MULTIVAR = Path("c3s_multivar_monthly_ensemble_features.parquet")
OUT = Path("training_t2m_multivar_c3s_era5_land.parquet")

print("Loading base T2M training dataset...")
base = pd.read_parquet(BASE_TRAINING)

print("Base shape:", base.shape)
print("Base columns:", len(base.columns))

print("Loading multivar C3S features...")
mv = pd.read_parquet(MULTIVAR)

print("Multivar coarse shape:", mv.shape)

# Use target_time for monthly alignment, not raw valid_time day.
mv["target_time"] = pd.to_datetime(mv["target_time"])

# Exclude SST for now because it is mostly missing over land.
drop_prefixes = ["sst_"]

feature_cols = [
    c for c in mv.columns
    if (
        c.endswith("_ens_mean")
        or c.endswith("_ens_std")
        or c.endswith("_ens_min")
        or c.endswith("_ens_max")
        or c.endswith("_ens_median")
        or c.endswith("_ens_p10")
        or c.endswith("_ens_p25")
        or c.endswith("_ens_p75")
        or c.endswith("_ens_p90")
    )
    and not any(c.startswith(p) for p in drop_prefixes)
]

print("Selected multivar feature columns:", len(feature_cols))
print(feature_cols)

# Average duplicates just in case
mv = (
    mv
    .groupby(["target_time", "lead_days", "latitude", "longitude"], as_index=False)[feature_cols]
    .mean()
)

print("Multivar after grouping:", mv.shape)

print("Converting multivar to xarray...")
mv_ds = mv.set_index(
    ["target_time", "lead_days", "latitude", "longitude"]
)[feature_cols].to_xarray()

mv_ds = mv_ds.rename({"target_time": "valid_time"})

# ERA5-Land grid is available from base dataframe
era_lats = np.sort(base["latitude"].unique())[::-1]
era_lons = np.sort(base["longitude"].unique())

print("ERA grid:")
print("lat:", len(era_lats), era_lats.min(), era_lats.max())
print("lon:", len(era_lons), era_lons.min(), era_lons.max())

print("Interpolating multivar C3S features to ERA5-Land grid...")
mv_interp = mv_ds.interp(
    latitude=era_lats,
    longitude=era_lons,
    method="linear"
)

# cast to float32 to reduce memory
for c in feature_cols:
    mv_interp[c] = mv_interp[c].astype("float32")

print(mv_interp)

print("Converting interpolated multivar to dataframe...")
mv_hi = mv_interp.to_dataframe().reset_index()

# Drop rows where all multivar features missing after interpolation
mv_hi = mv_hi.dropna(subset=feature_cols, how="all").copy()

print("Multivar high-res shape:", mv_hi.shape)

# Align monthly
mv_hi["valid_time"] = pd.to_datetime(mv_hi["valid_time"])

base["valid_time"] = pd.to_datetime(base["valid_time"])

merge_keys = [
    "valid_time",
    "lead_days",
    "latitude",
    "longitude",
]

print("Merging base T2M training with multivar predictors...")
out = base.merge(
    mv_hi[merge_keys + feature_cols],
    on=merge_keys,
    how="left",
    validate="many_to_one"
)

print("Merged shape:", out.shape)

print("\nMissing values in multivar features:")
missing = out[feature_cols].isna().sum().sort_values(ascending=False)
print(missing.head(50))

# Drop rows where any selected multivar feature is missing.
# This should mostly remove border/interpolation-margin rows if any.
before = len(out)
out = out.dropna(subset=feature_cols).copy()
after = len(out)

print(f"Dropped rows due to missing multivar features: {before - after}")
print("Final shape:", out.shape)

print("\nFinal columns:", len(out.columns))
print(out.head())

out.to_parquet(OUT, index=False)
print("\nSaved:", OUT)
