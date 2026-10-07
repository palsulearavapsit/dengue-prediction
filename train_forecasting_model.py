from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "forecasting_dataset.csv"
MODEL_PATH = ROOT / "dengue_forecast_model.joblib"
RESULTS_PATH = ROOT / "model_results.csv"
METADATA_PATH = ROOT / "model_metadata.json"

TARGET = "target_cases"
LOG_TARGET = "target_cases_log1p"

# Exact feature list from build_forecasting_dataset.py
NUMERIC_FEATURES = [
    "current_cases",
    "previous_cases",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "previous_case_change",
    "previous_case_growth",
    "previous_cases_rolling_mean_4",
    "previous_cases_rolling_median_4",
    "previous_cases_rolling_min_4",
    "previous_cases_rolling_max_4",
    "previous_cases_rolling_std_4",
    "days_since_previous_report",
    "previous_reporting_gap_mean_4",
    "previous_reporting_gap_median_4",
    "previous_reporting_gap_std_4",
    "observation_year",
    "observation_month",
    "observation_week_of_year",
    "observation_day_of_year",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "season",
    "Latitude",
    "Longitude",
    "Temp",
    "preci",
    "LAI",
]

CATEGORICAL_FEATURES = [
    "state_ut",
    "district",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ============================================================
# METRICS
# ============================================================

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)

    values = np.where(
        denominator == 0,
        0,
        2 * np.abs(y_pred - y_true) / denominator,
    )

    return float(np.mean(values) * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


# ============================================================
# BASELINES
# ============================================================

def persistence_prediction(df):
    return df["current_cases"].fillna(0).to_numpy()


def recent3_prediction(df):
    values = df[
        [
            "current_cases",
            "previous_cases",
            "cases_lag_2",
        ]
    ]

    return values.mean(axis=1, skipna=True).fillna(
        df["current_cases"]
    ).to_numpy()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DENGUE FORECASTING MODEL TRAINING")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Forecasting dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

# Sort chronologically
df["date"] = pd.to_datetime(df["date"])
df["next_observation_date"] = pd.to_datetime(
    df["next_observation_date"]
)

df = df.sort_values("date").reset_index(drop=True)

# Verify features
missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

# ============================================================
# TEMPORAL HOLDOUT
# ============================================================

# Final 20% is reserved as a completely unseen chronological test set.
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(f"Training rows: {len(train_df)}")
print(f"Test rows:     {len(test_df)}")

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].astype(float)
y_test = test_df[TARGET].astype(float)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Ridge_log1p": Ridge(
        alpha=10.0
    ),

    "RandomForest_log1p": RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "ExtraTrees_log1p": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "HistGradientBoosting_log1p": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    ),
}


# ============================================================
# LOG-TRANSFORMED TRAINING
# ============================================================

y_train_log = np.log1p(y_train)


# ============================================================
# TEMPORAL TEST EVALUATION
# ============================================================

results = []

print("\n" + "=" * 80)
print("BASELINES")
print("=" * 80)

# Persistence
pred = persistence_prediction(test_df)

metrics = evaluate(y_test, pred)

results.append({
    "model": "Persistence",
    **metrics,
})

print(
    f"Persistence       "
    f"MAE={metrics['MAE']:.4f} "
    f"RMSE={metrics['RMSE']:.4f} "
    f"SMAPE={metrics['SMAPE']:.2f}"
)

# Recent 3
pred = recent3_prediction(test_df)

metrics = evaluate(y_test, pred)

results.append({
    "model": "Recent3_Mean",
    **metrics,
})

print(
    f"Recent-3 Mean     "
    f"MAE={metrics['MAE']:.4f} "
    f"RMSE={metrics['RMSE']:.4f} "
    f"SMAPE={metrics['SMAPE']:.2f}"
)


# ============================================================
# ML MODELS
# ============================================================

print("\n" + "=" * 80)
print("MACHINE LEARNING MODELS")
print("=" * 80)

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train_log,
    )

    pred_log = pipeline.predict(X_test)

    pred = np.expm1(pred_log)

    # Case counts cannot be negative.
    pred = np.maximum(pred, 0)

    metrics = evaluate(
        y_test,
        pred,
    )

    results.append({
        "model": name,
        **metrics,
    })

    trained_models[name] = pipeline

    print(
        f"{name:<28}"
        f"MAE={metrics['MAE']:.4f} "
        f"RMSE={metrics['RMSE']:.4f} "
        f"SMAPE={metrics['SMAPE']:.2f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE",
    ascending=True,
).reset_index(drop=True)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


# ============================================================
# SELECT MODEL
# ============================================================

ml_results = results_df[
    results_df["model"].str.contains(
        "_log1p",
        regex=False,
    )
].copy()

best_model_name = ml_results.iloc[0]["model"]

print("\n" + "=" * 80)
print("SELECTED MODEL")
print("=" * 80)

print(
    f"Selected model: {best_model_name}"
)

print(
    f"Test MAE: "
    f"{ml_results.iloc[0]['MAE']:.4f}"
)

print(
    "\nNOTE:"
    "\nThe model is selected using chronological holdout MAE."
    "\nThis is NOT classification accuracy."
)


# ============================================================
# RETRAIN SELECTED MODEL ON ALL AVAILABLE DATA
# ============================================================

print("\nRetraining selected model on full forecasting dataset...")

X_all = df[FEATURES].copy()
y_all = df[TARGET].astype(float)

best_estimator = models[
    best_model_name
]

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            best_estimator,
        ),
    ]
)

final_pipeline.fit(
    X_all,
    np.log1p(y_all),
)

joblib.dump(
    final_pipeline,
    MODEL_PATH,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "project": "Dengue Outbreak Prediction System",
    "task": "next_observed_report_regression",
    "target": TARGET,
    "target_transformation": "log1p",
    "selected_model": best_model_name,
    "feature_count": len(FEATURES),
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "dataset_rows": int(len(df)),
    "training_rows_before_final_retrain": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "selection_metric": "MAE",
    "validation": "chronological 80/20 holdout",
    "model_file": MODEL_PATH.name,
}

METADATA_PATH.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)

print(
    f"\nResults saved to:\n{RESULTS_PATH}"
)

print(
    f"\nMetadata saved to:\n{METADATA_PATH}"
)

print(
    "\nNext step:"
    "\nConnect the saved model to the n8n ML Prediction component."
)

print("=" * 80)