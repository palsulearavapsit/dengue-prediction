
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_PATH = "model_data.csv"
TARGET = "outbreak_proxy"

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

# Remove rows without target labels
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# Exclude the target and any synthetic-data indicator from predictors
exclude = [TARGET, "synthetic"]
X = df.drop(columns=[c for c in exclude if c in df.columns])
y = df[TARGET]

# Convert categorical columns to strings and identify feature types
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

# Do not use a pre-existing row index as a predictor
for col in list(X.columns):
    if col.lower().startswith("unnamed:"):
        X = X.drop(columns=[col])
        if col in numeric:
            numeric.remove(col)
        if col in categorical:
            categorical.remove(col)

# Chronological split if a usable date column exists; otherwise
# preserve row order as stored in model_data.csv.
date_col = next(
    (c for c in X.columns if c.lower() in ["date", "observation_date"]),
    None
)

if date_col:
    dates = pd.to_datetime(X[date_col], errors="coerce")
    order = np.argsort(dates.fillna(pd.Timestamp.min).values)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)

split = int(len(X) * 0.8)
if split <= 0 or split >= len(X):
    raise ValueError("Not enough rows for an 80/20 train/test split.")

X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Drop raw date strings after sorting; keep numeric date-derived features
for col in X_train.columns:
    if col.lower() in ["date", "observation_date"]:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()

transformers = []

if numeric:
    transformers.append((
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric
    ))

if categorical:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", __import__("sklearn").preprocessing.OneHotEncoder(
                handle_unknown="ignore"
            ))
        ]),
        categorical
    ))

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

labels = [0, 1]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\n=== Outbreak Proxy Confusion Matrix ===")
print("Rows = actual; columns = predicted")
print(pd.DataFrame(
    cm,
    index=["Actual 0 (non-outbreak)", "Actual 1 (outbreak)"],
    columns=["Predicted 0", "Predicted 1"]
))

print("\n=== Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\n=== Classification Report ===")
print(classification_report(
    y_test, y_pred,
    labels=labels,
    target_names=["Non-outbreak", "Outbreak"],
    zero_division=0
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-outbreak (0)", "Outbreak (1)"]
)
disp.plot(values_format="d")
plt.title("Outbreak Proxy Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
).to_csv("confusion_matrix.csv")

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_pred
}).to_csv("outbreak_test_predictions.csv", index=False)

print("\nSaved:")
print(" - confusion_matrix.png")
print(" - confusion_matrix.csv")
print(" - outbreak_test_predictions.csv")
print("\nNote: results evaluate the outbreak_proxy label, not future case-count prediction.")

