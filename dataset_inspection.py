import pandas as pd

df = pd.read_csv("dengue_cleaned.csv")

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== SAMPLE ==========")
print(df.head(10).to_string())

print("\n========== NUMERIC SUMMARY ==========")
print(df.describe().T)

print("\n========== DISEASE ==========")
if "Disease" in df.columns:
    print(df["Disease"].value_counts())

print("\n========== CASES ==========")
print(df["Cases"].describe())

print("\n========== RAINFALL ==========")
if "preci" in df.columns:
    print(df["preci"].describe())

print("\n========== LAI ==========")
if "LAI" in df.columns:
    print(df["LAI"].describe())

print("\n========== TEMPERATURE ==========")
if "Temp" in df.columns:
    print(df["Temp"].describe())