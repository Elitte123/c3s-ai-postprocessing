from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("training_t2m_c3s_era5_land.parquet")
OUT = Path("lead_specific_rf_results.csv")

RANDOM_STATE = 42
MAX_TRAIN_ROWS_PER_LEAD = 300_000
MAX_TEST_ROWS_PER_LEAD = 150_000

print("Loading data...")
df = pd.read_parquet(DATA)

df["month_sin"] = np.sin(2 * np.pi * df["valid_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["valid_month"] / 12)
df["target_residual"] = df["t2m_era5_c"] - df["t2m_ens_mean"]

feature_cols = [
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

def metrics(y_true, y_pred, name, lead):
    return {
        "lead_days": lead,
        "model": name,
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": mean_squared_error(y_true, y_pred) ** 0.5,
        "Bias_C": float(np.mean(y_pred - y_true)),
        "R2": r2_score(y_true, y_pred),
    }

results = []

for lead in sorted(df["lead_days"].unique()):
    print(f"\n=== Lead {lead} days ===")

    d = df[df["lead_days"] == lead].copy()

    train = d[d["valid_year"] <= 2020].copy()
    test = d[d["valid_year"] >= 2021].copy()

    print("Train before sample:", train.shape)
    print("Test before sample:", test.shape)

    if len(train) > MAX_TRAIN_ROWS_PER_LEAD:
        train = train.sample(MAX_TRAIN_ROWS_PER_LEAD, random_state=RANDOM_STATE)

    if len(test) > MAX_TEST_ROWS_PER_LEAD:
        test = test.sample(MAX_TEST_ROWS_PER_LEAD, random_state=RANDOM_STATE)

    print("Train used:", train.shape)
    print("Test used:", test.shape)

    baseline_pred = test["t2m_ens_mean"].values

    results.append(
        metrics(
            test["t2m_era5_c"],
            baseline_pred,
            "C3S baseline",
            lead
        )
    )

    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=22,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print("Training lead-specific residual RF...")
    rf.fit(train[feature_cols], train["target_residual"])

    residual_pred = rf.predict(test[feature_cols])
    corrected_pred = baseline_pred + residual_pred

    results.append(
        metrics(
            test["t2m_era5_c"],
            corrected_pred,
            "Lead-specific RF residual correction",
            lead
        )
    )

res = pd.DataFrame(results)
print("\n=== RESULTS ===")
print(res)

res.to_csv(OUT, index=False)
print(f"Saved: {OUT}")
