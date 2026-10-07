import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Final_data(3)(1).csv"

OUTPUT_DIR = Path("prepared_data")
OUTPUT_DIR.mkdir(exist_ok=True)

CLEAN_FILE = OUTPUT_DIR / "dengue_clean.csv"
BALANCED_FILE = OUTPUT_DIR / "dengue_balanced.csv"
SYNTHETIC_FILE = OUTPUT_DIR / "dengue_synthetic.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("DENGUE DATA PREPARATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal shape: {df.shape}")


# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = [c.strip() for c in df.columns]


# ============================================================
# 3. KEEP DENGUE ONLY
# ============================================================

df["Disease"] = df["Disease"].astype(str).str.strip()

df = df[
    df["Disease"].str.lower() == "dengue"
].copy()

print(f"Dengue rows: {len(df)}")


# ============================================================
# 4. REMOVE UNNECESSARY INDEX COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


# ============================================================
# 5. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Cases",
    "Deaths",
    "day",
    "mon",
    "year",
    "Latitude",
    "Longitude",
    "preci",
    "LAI",
    "Temp"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 6. CREATE DATE
# ============================================================

df["date"] = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["mon"],
        day=df["day"]
    ),
    errors="coerce"
)


# ============================================================
# 7. REMOVE INVALID ROWS
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "state_ut",
        "district",
        "Cases",
        "day",
        "mon",
        "year",
        "Latitude",
        "Longitude",
        "preci",
        "LAI",
        "date"
    ]
)

print(f"Removed invalid rows: {before - len(df)}")


# ============================================================
# 8. CLEAN CATEGORICAL VALUES
# ============================================================

for col in ["state_ut", "district"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 9. HANDLE DEATHS
# ============================================================

# Missing deaths are NOT automatically treated as actual zero.
# Keep a missingness indicator.

df["Deaths_missing"] = df["Deaths"].isna().astype(int)

# For ML purposes, fill missing deaths with median.
if df["Deaths"].isna().any():
    df["Deaths"] = df["Deaths"].fillna(
        df["Deaths"].median()
    )


# ============================================================
# 10. TEMPERATURE
# ============================================================

if "Temp" in df.columns:

    missing_temp = df["Temp"].isna().sum()

    print(f"Missing temperature values: {missing_temp}")

    # The source dataset has no usable Temp values for the
    # cleaned Dengue dataset, so do NOT fabricate temperature.
    if missing_temp == len(df):
        df.drop(columns=["Temp"], inplace=True)
        print("Temp removed because it is completely missing.")


# ============================================================
# 11. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["district", "date"]
).reset_index(drop=True)


# ============================================================
# 12. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "state_ut",
        "district",
        "Disease",
        "date"
    ],
    keep="first"
)

print(f"Duplicate rows removed: {before - len(df)}")


# ============================================================
# 13. SAVE CLEAN DATASET
# ============================================================

df.to_csv(
    CLEAN_FILE,
    index=False
)

print(f"\nClean dataset saved:")
print(CLEAN_FILE)

print(f"Clean shape: {df.shape}")


# ============================================================
# 14. CREATE OUTBREAK TARGET
# ============================================================

# IMPORTANT:
# This is a temporary classification target for experimentation.
#
# We calculate the threshold INSIDE each district to avoid using
# one global threshold across all districts.

district_threshold = (
    df.groupby("district")["Cases"]
    .transform(lambda x: x.quantile(0.90))
)

df["outbreak_proxy"] = (
    df["Cases"] >= district_threshold
).astype(int)


print("\nClass distribution:")
print(
    df["outbreak_proxy"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. SELECT ML FEATURES
# ============================================================

feature_columns = [
    "Latitude",
    "Longitude",
    "year",
    "mon",
    "day",
    "preci",
    "LAI",
    "Deaths",
    "Deaths_missing"
]

feature_columns = [
    c for c in feature_columns
    if c in df.columns
]

X = df[feature_columns].copy()
y = df["outbreak_proxy"].copy()


# ============================================================
# 16. HANDLE MISSING FEATURES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)


# ============================================================
# 17. TRAIN/TEST SPLIT BY TIME
# ============================================================

# NEVER synthesize validation/test data.

sorted_indices = df["date"].sort_values().index

split_position = int(
    len(sorted_indices) * 0.80
)

train_indices = sorted_indices[:split_position]
test_indices = sorted_indices[split_position:]

X_train = X.loc[train_indices]
y_train = y.loc[train_indices]

X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())


# ============================================================
# 18. SYNTHETIC BALANCING WITH SMOTENC
# ============================================================

# All current selected features are numeric, therefore regular
# SMOTE is sufficient.
#
# We use ONLY the training data.

from imblearn.over_sampling import SMOTE

minority_count = y_train.value_counts().min()

# SMOTE requires at least 2 minority samples.
if minority_count >= 2:

    k_neighbors = min(
        5,
        minority_count - 1
    )

    smote = SMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=k_neighbors
    )

    X_balanced, y_balanced = smote.fit_resample(
        X_train,
        y_train
    )

else:
    raise ValueError(
        "Not enough minority samples for SMOTE."
    )


# ============================================================
# 19. IDENTIFY SYNTHETIC ROWS
# ============================================================

original_train_count = len(X_train)

balanced_count = len(X_balanced)

synthetic_count = (
    balanced_count -
    original_train_count
)

print("\nSynthetic samples generated:", synthetic_count)

print("\nBalanced distribution:")
print(
    pd.Series(y_balanced)
    .value_counts()
    .sort_index()
)


# ============================================================
# 20. SAVE SYNTHETIC-ONLY DATA
# ============================================================

synthetic_X = X_balanced.iloc[
    original_train_count:
].copy()

synthetic_y = y_balanced.iloc[
    original_train_count:
].copy()

synthetic_df = synthetic_X.copy()

synthetic_df["outbreak_proxy"] = (
    synthetic_y.values
)

synthetic_df["synthetic"] = 1

synthetic_df.to_csv(
    SYNTHETIC_FILE,
    index=False
)

print(
    f"\nSynthetic dataset saved:\n{SYNTHETIC_FILE}"
)


# ============================================================
# 21. SAVE COMPLETE BALANCED TRAINING DATA
# ============================================================

balanced_df = X_balanced.copy()

balanced_df["outbreak_proxy"] = (
    y_balanced.values
)

balanced_df["synthetic"] = 0

balanced_df.iloc[
    original_train_count:,
    balanced_df.columns.get_loc("synthetic")
] = 1

balanced_df.to_csv(
    BALANCED_FILE,
    index=False
)

print(
    f"\nBalanced dataset saved:\n{BALANCED_FILE}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Original rows       : {len(pd.read_csv(INPUT_FILE))}")
print(f"Clean Dengue rows   : {len(df)}")
print(f"Training rows       : {len(X_train)}")
print(f"Test rows           : {len(X_test)}")
print(f"Synthetic rows      : {synthetic_count}")
print(f"Balanced train rows : {len(X_balanced)}")

print("\nFiles created:")
print(f"1. {CLEAN_FILE}")
print(f"2. {SYNTHETIC_FILE}")
print(f"3. {BALANCED_FILE}")

print("\nIMPORTANT:")
print("Synthetic data was generated ONLY from training data.")
print("The test set remains completely untouched.")