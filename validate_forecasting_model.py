from pathlib import Path
import warnings

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "forecasting_dataset.csv"

TARGET = "target_cases"

CATEGORICAL_FEATURES = [
    "state_ut",
    "district",
]

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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)

    values = np.where(
        denominator == 0,
        0,
        2 * np.abs(y_pred - y_true) / denominator,
    )

    return np.mean(values) * 100


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
    }


def make_preprocessor():
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_models():

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),

        "Ridge": Ridge(
            alpha=10.0
        ),
    }


print("=" * 80)
print("DENGUE FORECASTING — ROLLING TEMPORAL VALIDATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"\nDataset rows: {len(df)}")
print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

# ------------------------------------------------------------
# 4 expanding-window chronological folds
# ------------------------------------------------------------

n = len(df)

folds = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
]

all_results = []

models = make_models()

for fold_number, (train_end, test_end) in enumerate(
    folds,
    start=1
):

    train_end_idx = int(n * train_end)
    test_end_idx = int(n * test_end)

    train_df = df.iloc[:train_end_idx].copy()
    test_df = df.iloc[train_end_idx:test_end_idx].copy()

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df[TARGET].astype(float)
    y_test = test_df[TARGET].astype(float)

    print("\n" + "=" * 80)
    print(f"FOLD {fold_number}")
    print("=" * 80)

    print(
        f"Train: {len(train_df)} rows | "
        f"{train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Test:  {len(test_df)} rows | "
        f"{test_df['date'].min().date()} → "
        f"{test_df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Persistence baseline
    # --------------------------------------------------------

    persistence_pred = (
        test_df["current_cases"]
        .fillna(0)
        .to_numpy()
    )

    result = metrics(
        y_test,
        persistence_pred,
    )

    all_results.append({
        "fold": fold_number,
        "model": "Persistence",
        **result,
    })

    print(
        f"Persistence            "
        f"MAE={result['MAE']:.2f}"
    )

    # --------------------------------------------------------
    # Recent-3 baseline
    # --------------------------------------------------------

    recent3 = test_df[
        [
            "current_cases",
            "previous_cases",
            "cases_lag_2",
        ]
    ]

    recent3_pred = (
        recent3
        .mean(axis=1, skipna=True)
        .fillna(test_df["current_cases"])
        .to_numpy()
    )

    result = metrics(
        y_test,
        recent3_pred,
    )

    all_results.append({
        "fold": fold_number,
        "model": "Recent3_Mean",
        **result,
    })

    print(
        f"Recent-3 Mean          "
        f"MAE={result['MAE']:.2f}"
    )

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            np.log1p(y_train),
        )

        prediction_log = pipeline.predict(X_test)

        prediction = np.expm1(
            prediction_log
        )

        prediction = np.maximum(
            prediction,
            0,
        )

        result = metrics(
            y_test,
            prediction,
        )

        all_results.append({
            "fold": fold_number,
            "model": name,
            **result,
        })

        print(
            f"{name:<23}"
            f"MAE={result['MAE']:.2f}"
        )


# ============================================================
# SUMMARY
# ============================================================

results = pd.DataFrame(
    all_results
)

print("\n" + "=" * 80)
print("ROLLING VALIDATION RESULTS")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


summary = (
    results
    .groupby("model")
    .agg(
        mean_MAE=("MAE", "mean"),
        mean_RMSE=("RMSE", "mean"),
        mean_SMAPE=("SMAPE", "mean"),
        std_MAE=("MAE", "std"),
    )
    .sort_values("mean_MAE")
)

print("\n" + "=" * 80)
print("AVERAGE ACROSS ALL TEMPORAL FOLDS")
print("=" * 80)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

best_model = summary.index[0]

print("\n" + "=" * 80)
print("FINAL VALIDATION RESULT")
print("=" * 80)

print(
    f"Best model by mean rolling MAE: {best_model}"
)

print(
    f"Mean rolling MAE: "
    f"{summary.loc[best_model, 'mean_MAE']:.2f}"
)

print(
    f"MAE standard deviation: "
    f"{summary.loc[best_model, 'std_MAE']:.2f}"
)

print("\nValidation complete.")