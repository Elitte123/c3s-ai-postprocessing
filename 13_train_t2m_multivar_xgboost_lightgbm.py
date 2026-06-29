from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATA = Path("training_t2m_multivar_c3s_era5_land.parquet")

OUT_METRICS = Path("t2m_multivar_model_metrics.csv")
OUT_LEAD = Path("t2m_multivar_model_lead_metrics.csv")
OUT_IMPORTANCE = Path("t2m_multivar_feature_importance.csv")
OUT_PRED = Path("t2m_multivar_predictions_sample.parquet")

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 700_000
MAX_TEST_ROWS = 300_000

print("Loading multivar dataset...")
df = pd.read_parquet(DATA)

print("Shape:", df.shape)
print("Years:", df["valid_year"].min(), df["valid_year"].max())

df["month_sin"] = np.sin(2 * np.pi * df["valid_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["valid_month"] / 12)
df["target_residual"] = df["t2m_era5_c"] - df["t2m_ens_mean"]

# Base T2M features
t2m_features = [
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

# Multivar extra features
extra_features = [
    c for c in df.columns
    if (
        c.startswith("d2m_")
        or c.startswith("msl_")
        or c.startswith("stl1_")
        or c.startswith("swvl1_")
        or c.startswith("e_")
    )
]

feature_sets = {
    "T2M-only": t2m_features,
    "Multivar": t2m_features + extra_features,
}

print("T2M features:", len(t2m_features))
print("Extra multivar features:", len(extra_features))
print("Total multivar features:", len(feature_sets["Multivar"]))

train_full = df[df["valid_year"] <= 2020].copy()
test_full = df[df["valid_year"] >= 2021].copy()

train = train_full.sample(min(MAX_TRAIN_ROWS, len(train_full)), random_state=RANDOM_STATE)
test = test_full.sample(min(MAX_TEST_ROWS, len(test_full)), random_state=RANDOM_STATE)

print("Train used:", train.shape)
print("Test used:", test.shape)

y_train = train["target_residual"]
y_test = test["t2m_era5_c"]
baseline_pred = test["t2m_ens_mean"].values

metrics_rows = []
lead_rows = []
importance_rows = []

pred_out = test[
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

pred_out["baseline_pred_c"] = baseline_pred
pred_out["baseline_error_c"] = pred_out["baseline_pred_c"] - pred_out["t2m_era5_c"]


def calc_metrics(y_true, y_pred, model_name, feature_set):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "model": model_name,
        "feature_set": feature_set,
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": mean_squared_error(y_true, y_pred) ** 0.5,
        "Bias_C": float(np.mean(err)),
        "R2": r2_score(y_true, y_pred),
    }


def calc_lead_metrics(test_df, pred, model_name, feature_set):
    tmp = test_df[["lead_days", "t2m_era5_c"]].copy()
    tmp["pred"] = pred
    tmp["err"] = tmp["pred"] - tmp["t2m_era5_c"]

    rows = []
    for lead, g in tmp.groupby("lead_days"):
        rows.append({
            "model": model_name,
            "feature_set": feature_set,
            "lead_days": lead,
            "MAE_C": np.mean(np.abs(g["err"])),
            "RMSE_C": np.sqrt(np.mean(g["err"] ** 2)),
            "Bias_C": np.mean(g["err"]),
            "R2": r2_score(g["t2m_era5_c"], g["pred"]),
        })
    return rows


# baseline
metrics_rows.append(
    calc_metrics(y_test, baseline_pred, "C3S ensemble mean", "baseline")
)
lead_rows.extend(
    calc_lead_metrics(test, baseline_pred, "C3S ensemble mean", "baseline")
)

# climatology
print("Building climatology benchmark...")
clim = (
    train_full
    .groupby(["valid_month", "latitude", "longitude"], as_index=False)["t2m_era5_c"]
    .mean()
    .rename(columns={"t2m_era5_c": "era5_climatology_c"})
)

test_clim = test.merge(
    clim,
    on=["valid_month", "latitude", "longitude"],
    how="left"
)

clim_pred = test_clim["era5_climatology_c"].values

metrics_rows.append(
    calc_metrics(y_test, clim_pred, "ERA5 monthly local climatology", "benchmark")
)
lead_rows.extend(
    calc_lead_metrics(test, clim_pred, "ERA5 monthly local climatology", "benchmark")
)

pred_out["climatology_pred_c"] = clim_pred
pred_out["climatology_error_c"] = clim_pred - pred_out["t2m_era5_c"]

# models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

models = {
    "LightGBM residual": LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=96,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_alpha=0.1,
        reg_lambda=0.3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="regression",
    ),
    "XGBoost residual": XGBRegressor(
        n_estimators=900,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

for feature_set_name, features in feature_sets.items():
    print(f"\n=== Feature set: {feature_set_name} | {len(features)} features ===")

    X_train = train[features]
    X_test = test[features]

    for model_name, model_template in models.items():
        print(f"\nTraining {model_name} with {feature_set_name}...")
        t0 = time.time()

        # recreate model each time
        if model_name.startswith("LightGBM"):
            model = LGBMRegressor(**model_template.get_params())
        else:
            model = XGBRegressor(**model_template.get_params())

        model.fit(X_train, y_train)

        train_time = time.time() - t0
        print(f"Training time: {train_time:.1f} s")

        residual_pred = model.predict(X_test)
        corrected_pred = baseline_pred + residual_pred

        row = calc_metrics(y_test, corrected_pred, model_name, feature_set_name)
        row["training_time_s"] = train_time
        metrics_rows.append(row)

        lead_rows.extend(
            calc_lead_metrics(test, corrected_pred, model_name, feature_set_name)
        )

        safe = f"{model_name}_{feature_set_name}".lower().replace(" ", "_").replace("-", "_")
        pred_out[f"{safe}_pred_c"] = corrected_pred
        pred_out[f"{safe}_error_c"] = corrected_pred - pred_out["t2m_era5_c"]

        if hasattr(model, "feature_importances_"):
            total = np.sum(model.feature_importances_)
            for feat, imp in zip(features, model.feature_importances_):
                importance_rows.append({
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "feature": feat,
                    "importance": imp,
                    "importance_normalized": imp / total if total != 0 else np.nan,
                })

metrics = pd.DataFrame(metrics_rows)
lead_metrics = pd.DataFrame(lead_rows)
importance = pd.DataFrame(importance_rows)

print("\n=== METRICS ===")
print(metrics)

print("\n=== LEAD METRICS HEAD ===")
print(lead_metrics.head(20))

print("\n=== IMPORTANCE TOP ===")
if not importance.empty:
    print(
        importance
        .sort_values(["feature_set", "model", "importance"], ascending=[True, True, False])
        .groupby(["feature_set", "model"])
        .head(15)
    )

metrics.to_csv(OUT_METRICS, index=False)
lead_metrics.to_csv(OUT_LEAD, index=False)
importance.to_csv(OUT_IMPORTANCE, index=False)
pred_out.to_parquet(OUT_PRED, index=False)

print("\nSaved:")
print(OUT_METRICS)
print(OUT_LEAD)
print(OUT_IMPORTANCE)
print(OUT_PRED)
