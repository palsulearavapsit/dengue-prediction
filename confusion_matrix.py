
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "model_data.csv"

CATEGORICAL = ["state_ut", "district"]

NUMERIC = [
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

FEATURES = CATEGORICAL + NUMERIC
LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

def risk_category(values):
    values = np.asarray(values, dtype=float)
    return np.select(
        [
            values < 10,
            values < 25,
            values < 50,
        ],
        [
            "LOW",
            "MODERATE",
            "HIGH",
        ],
        default="VERY HIGH",
    )

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    target = next(
        (
            col for col in
            ["target_cases", "next_observed_report_cases"]
            if col in df.columns
        ),
        None,
    )

    if target is None:
        raise ValueError(
            "Could not find target_cases or "
            "next_observed_report_cases in model_data.csv. "
            f"Available columns: {list(df.columns)}"
        )

    if "date" not in df.columns:
        raise ValueError(
            "The dataset has no 'date' column. "
            f"Available columns: {list(df.columns)}"
        )

    missing = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required feature columns are missing: "
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )
    df[target] = pd.to_numeric(
        df[target], errors="coerce"
    )

    df = (
        df.dropna(subset=["date", target])
        .sort_values("date")
        .reset_index(drop=True)
    )

    n = len(df)
    folds = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
    ]

    actual_all = []
    predicted_all = []

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
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
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    for fold, (train_fraction, test_fraction) in enumerate(
        folds, start=1
    ):
        train_end = int(n * train_fraction)
        test_end = int(n * test_fraction)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        if train_df.empty or test_df.empty:
            continue

        model = Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ])

        model.fit(
            train_df[FEATURES],
            np.log1p(
                train_df[target].clip(lower=0)
            ),
        )

        prediction = np.expm1(
            model.predict(test_df[FEATURES])
        )
        prediction = np.maximum(prediction, 0)

        actual_all.extend(
            test_df[target].clip(lower=0).to_numpy()
        )
        predicted_all.extend(prediction)

        print(
            f"Fold {fold}: "
            f"{len(train_df)} training rows, "
            f"{len(test_df)} test rows"
        )

    if not actual_all:
        raise ValueError("No holdout predictions were generated.")

    actual_risk = risk_category(actual_all)
    predicted_risk = risk_category(predicted_all)

    cm = confusion_matrix(
        actual_risk,
        predicted_risk,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS,
    )
    cm_df.index.name = "Actual / Predicted"
    cm_df.to_csv(ROOT / "confusion_matrix.csv")

    report = classification_report(
        actual_risk,
        predicted_risk,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        ROOT / "risk_classification_report.csv"
    )

    accuracy = accuracy_score(
        actual_risk,
        predicted_risk,
    )

    print("\nRISK CATEGORY CONFUSION MATRIX")
    print("Rows = actual; columns = predicted")
    print(cm_df)
    print(f"\nRisk-category accuracy: {accuracy:.2%}")
    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            actual_risk,
            predicted_risk,
            labels=LABELS,
            zero_division=0,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted risk category",
        ylabel="Actual risk category",
        title="Dengue Risk Category Confusion Matrix",
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=25,
        ha="right",
    )

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        ROOT / "confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nSaved files:")
    print("  confusion_matrix.png")
    print("  confusion_matrix.csv")
    print("  risk_classification_report.csv")
    print(
        "\nThese metrics evaluate risk categories, "
        "not the original regression model's accuracy."
    )

if __name__ == "__main__":
    main()
