from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("training_t2m_c3s_era5_land.parquet")
OUT = Path("audit_t2m_benchmarks.csv")

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 700_000
MAX_TEST_ROWS = 300_000

print("Loading data...")
df = pd.read_parquet(DATA)

df["month_sin"] = np.sin(2 * np.pi * df["valid_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["valid_month"] / 12)

train = df[df["valid_year"] <= 2020].copy()
test = df[df["valid_year"] >= 2021].copy()

print("Train:", train.shape)
print("Test:", test.shape)

overlap = set(train["valid_time"].unique()) & set(test["valid_time"].unique())
print("Temporal overlap train/test:", len(overlap))

# ---------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------
def metrics(y_true, y_pred, name):
    return {
        "model": name,
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": mean_squared_error(y_true, y_pred) ** 0.5,
        "Bias_C": float(np.mean(y_pred - y_true)),
        "R2": r2_score(y_true, y_pred),
    }

results = []

# ---------------------------------------------------------
# 1. C3S baseline
# ---------------------------------------------------------
results.append(
    metrics(
        test["t2m_era5_c"],
        test["t2m_ens_mean"],
        "C3S ensemble mean interpolated"
    )
)

# ---------------------------------------------------------
# 2. ERA5 climatology benchmark
# climatology calculated only on train years
# keyed by month, lat, lon
# ---------------------------------------------------------
print("Building ERA5 climatology benchmark...")

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
)

test_clim = test_clim.dropna(subset=["era5_climatology_c"])

results.append(
    metrics(
        test_clim["t2m_era5_c"],
        test_clim["era5_climatology_c"],
        "ERA5 monthly local climatology"
    )
)

# ---------------------------------------------------------
# 3. RF geo-seasonal only
# no C3S thermal predictors
# ---------------------------------------------------------
geo_features = [
    "lead_days",
    "latitude",
    "longitude",
    "valid_month",
    "month_sin",
    "month_cos",
]

train_s = train.sample(min(MAX_TRAIN_ROWS, len(train)), random_state=RANDOM_STATE)
test_s = test.sample(min(MAX_TEST_ROWS, len(test)), random_state=RANDOM_STATE)

print("Training RF geo-seasonal only...")
rf_geo = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

rf_geo.fit(train_s[geo_features], train_s["t2m_era5_c"])
pred_geo = rf_geo.predict(test_s[geo_features])

results.append(
    metrics(
        test_s["t2m_era5_c"],
        pred_geo,
        "RF geo-seasonal only"
    )
)

# ---------------------------------------------------------
# 4. RF full, direct prediction for comparison
# not residual, predicts ERA5 directly
# ---------------------------------------------------------
full_features = [
    "lead_days",
    "latitude",
    "longitude",
    "valid_month",
    "month_sin",
    "month_cos",
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

print("Training RF full direct...")
rf_full = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

rf_full.fit(train_s[full_features], train_s["t2m_era5_c"])
pred_full = rf_full.predict(test_s[full_features])

results.append(
    metrics(
        test_s["t2m_era5_c"],
        pred_full,
        "RF full direct T2M"
    )
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
res = pd.DataFrame(results)
print("\n=== AUDIT RESULTS ===")
print(res)

res.to_csv(OUT, index=False)
print(f"Saved: {OUT}")

print("\nRF geo feature importances:")
print(pd.DataFrame({
    "feature": geo_features,
    "importance": rf_geo.feature_importances_
}).sort_values("importance", ascending=False))

print("\nRF full feature importances:")
print(pd.DataFrame({
    "feature": full_features,
    "importance": rf_full.feature_importances_
}).sort_values("importance", ascending=False))
