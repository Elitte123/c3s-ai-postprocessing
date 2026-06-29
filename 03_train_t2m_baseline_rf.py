from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA = Path("training_t2m_c3s_era5_land.parquet")
MODEL_OUT = Path("rf_t2m_residual_model.joblib")
PRED_OUT = Path("predictions_t2m_rf_test_2021_2024.parquet")
METRICS_OUT = Path("metrics_t2m_baseline_vs_rf.csv")

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 700_000
MAX_TEST_ROWS = 300_000

print("Loading training dataset...")
df = pd.read_parquet(DATA)

print("Full shape:", df.shape)
print("Years:", df["valid_year"].min(), df["valid_year"].max())

# Optional spatial/seasonal features
df["month_sin"] = np.sin(2 * np.pi * df["valid_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["valid_month"] / 12)

# Target: residual correction
df["target_residual"] = df["t2m_era5_c"] - df["t2m_ens_mean"]

feature_cols = [
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

train = df[df["valid_year"] <= 2020].copy()
test = df[df["valid_year"] >= 2021].copy()

print("Train shape before sample:", train.shape)
print("Test shape before sample:", test.shape)

if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_STATE)

if len(test) > MAX_TEST_ROWS:
    test = test.sample(MAX_TEST_ROWS, random_state=RANDOM_STATE)

print("Train shape used:", train.shape)
print("Test shape used:", test.shape)

X_train = train[feature_cols]
y_train = train["target_residual"]

X_test = test[feature_cols]
y_test_true = test["t2m_era5_c"]

baseline_pred = test["t2m_ens_mean"].values

print("Training Random Forest residual model...")
rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

rf.fit(X_train, y_train)

print("Predicting residuals...")
residual_pred = rf.predict(X_test)
rf_corrected_pred = baseline_pred + residual_pred

def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    bias = float(np.mean(y_pred - y_true))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": name,
        "MAE_C": mae,
        "RMSE_C": rmse,
        "Bias_C": bias,
        "R2": r2,
    }

results = [
    metrics(y_test_true, baseline_pred, "C3S ensemble mean interpolated"),
    metrics(y_test_true, rf_corrected_pred, "Random Forest residual correction"),
]

metrics_df = pd.DataFrame(results)
print("\n=== Metrics ===")
print(metrics_df)

metrics_df.to_csv(METRICS_OUT, index=False)
print(f"Saved metrics: {METRICS_OUT}")

pred = test[
    [
        "valid_time",
        "valid_year",
        "valid_month",
        "lead_days",
        "latitude",
        "longitude",
        "t2m_era5_c",
        "t2m_ens_mean",
    ]
].copy()

pred["baseline_pred_c"] = baseline_pred
pred["rf_residual_pred_c"] = residual_pred
pred["rf_corrected_pred_c"] = rf_corrected_pred
pred["baseline_error_c"] = pred["baseline_pred_c"] - pred["t2m_era5_c"]
pred["rf_error_c"] = pred["rf_corrected_pred_c"] - pred["t2m_era5_c"]

pred.to_parquet(PRED_OUT, index=False)
print(f"Saved predictions: {PRED_OUT}")

joblib.dump(rf, MODEL_OUT)
print(f"Saved model: {MODEL_OUT}")

print("\nFeature importances:")
imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print(imp)
imp.to_csv("feature_importances_t2m_rf.csv", index=False)
print("Saved: feature_importances_t2m_rf.csv")
