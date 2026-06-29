from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATA = Path("training_t2m_c3s_era5_land.parquet")

OUT_METRICS = Path("model_comparison_t2m_metrics.csv")
OUT_LEAD = Path("model_comparison_t2m_lead_metrics.csv")
OUT_IMPORTANCE = Path("model_comparison_t2m_feature_importance.csv")
OUT_PRED = Path("model_comparison_t2m_predictions_sample.parquet")
OUTDIR_MODELS = Path("models_t2m")
OUTDIR_MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_TRAIN_ROWS = 700_000
MAX_TEST_ROWS = 300_000

print("Loading dataset...")
df = pd.read_parquet(DATA)

df["month_sin"] = np.sin(2 * np.pi * df["valid_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["valid_month"] / 12)

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

print("Full dataset:", df.shape)
print("Years:", df["valid_year"].min(), df["valid_year"].max())

train_full = df[df["valid_year"] <= 2020].copy()
test_full = df[df["valid_year"] >= 2021].copy()

print("Train full:", train_full.shape)
print("Test full:", test_full.shape)

train = train_full.sample(
    min(MAX_TRAIN_ROWS, len(train_full)),
    random_state=RANDOM_STATE
)

test = test_full.sample(
    min(MAX_TEST_ROWS, len(test_full)),
    random_state=RANDOM_STATE
)

print("Train used:", train.shape)
print("Test used:", test.shape)

X_train = train[feature_cols]
y_train_residual = train["target_residual"]

X_test = test[feature_cols]
y_test = test["t2m_era5_c"]
baseline_pred = test["t2m_ens_mean"].values

all_metrics = []
all_lead_metrics = []
all_importances = []

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


def metric_dict(y_true, y_pred, model_name):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "model": model_name,
        "MAE_C": mean_absolute_error(y_true, y_pred),
        "RMSE_C": mean_squared_error(y_true, y_pred) ** 0.5,
        "Bias_C": float(np.mean(err)),
        "R2": r2_score(y_true, y_pred),
    }


def add_lead_metrics(df_test, pred, model_name):
    tmp = df_test[["lead_days", "t2m_era5_c"]].copy()
    tmp["pred"] = pred
    tmp["err"] = tmp["pred"] - tmp["t2m_era5_c"]

    rows = []
    for lead, g in tmp.groupby("lead_days"):
        rows.append({
            "model": model_name,
            "lead_days": lead,
            "MAE_C": np.mean(np.abs(g["err"])),
            "RMSE_C": np.sqrt(np.mean(g["err"] ** 2)),
            "Bias_C": np.mean(g["err"]),
            "R2": r2_score(g["t2m_era5_c"], g["pred"]),
        })
    return rows


# ---------------------------------------------------------
# Baseline: C3S ensemble mean
# ---------------------------------------------------------
print("\n=== Baseline: C3S ensemble mean ===")
all_metrics.append(
    metric_dict(y_test, baseline_pred, "C3S ensemble mean")
)
all_lead_metrics.extend(
    add_lead_metrics(test, baseline_pred, "C3S ensemble mean")
)

# ---------------------------------------------------------
# Climatology benchmark
# ---------------------------------------------------------
print("\n=== Benchmark: ERA5 monthly local climatology ===")
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

all_metrics.append(
    metric_dict(y_test, clim_pred, "ERA5 monthly local climatology")
)
all_lead_metrics.extend(
    add_lead_metrics(test, clim_pred, "ERA5 monthly local climatology")
)

pred_out["climatology_pred_c"] = clim_pred
pred_out["climatology_error_c"] = pred_out["climatology_pred_c"] - pred_out["t2m_era5_c"]


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
models = []

models.append((
    "Random Forest residual",
    RandomForestRegressor(
        n_estimators=120,
        max_depth=22,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
))

try:
    from lightgbm import LGBMRegressor

    models.append((
        "LightGBM residual",
        LGBMRegressor(
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
        )
    ))
except Exception as e:
    print("LightGBM unavailable:", e)

try:
    from xgboost import XGBRegressor

    models.append((
        "XGBoost residual",
        XGBRegressor(
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
        )
    ))
except Exception as e:
    print("XGBoost unavailable:", e)


for model_name, model in models:
    print(f"\n=== Training {model_name} ===")
    t0 = time.time()

    model.fit(X_train, y_train_residual)

    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    print(f"Predicting {model_name}...")
    residual_pred = model.predict(X_test)
    corrected_pred = baseline_pred + residual_pred

    all_metrics.append(
        {
            **metric_dict(y_test, corrected_pred, model_name),
            "training_time_s": train_time,
        }
    )

    all_lead_metrics.extend(
        add_lead_metrics(test, corrected_pred, model_name)
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    pred_out[f"{safe_name}_pred_c"] = corrected_pred
    pred_out[f"{safe_name}_residual_pred_c"] = residual_pred
    pred_out[f"{safe_name}_error_c"] = corrected_pred - pred_out["t2m_era5_c"]

    model_path = OUTDIR_MODELS / f"{safe_name}.joblib"
    joblib.dump(model, model_path)
    print("Saved model:", model_path)

    if hasattr(model, "feature_importances_"):
        for feat, imp in zip(feature_cols, model.feature_importances_):
            all_importances.append({
                "model": model_name,
                "feature": feat,
                "importance": imp,
            })


metrics_df = pd.DataFrame(all_metrics)
lead_df = pd.DataFrame(all_lead_metrics)
imp_df = pd.DataFrame(all_importances)

# Normalize importance per model for easier comparison
if not imp_df.empty:
    imp_df["importance_normalized"] = imp_df.groupby("model")["importance"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

print("\n=== Global metrics ===")
print(metrics_df)

print("\n=== Lead metrics ===")
print(lead_df)

print("\n=== Feature importances ===")
print(imp_df.sort_values(["model", "importance"], ascending=[True, False]).head(50))

metrics_df.to_csv(OUT_METRICS, index=False)
lead_df.to_csv(OUT_LEAD, index=False)
imp_df.to_csv(OUT_IMPORTANCE, index=False)
pred_out.to_parquet(OUT_PRED, index=False)

print("\nSaved:")
print(OUT_METRICS)
print(OUT_LEAD)
print(OUT_IMPORTANCE)
print(OUT_PRED)
print("Models:", OUTDIR_MODELS)
