# data_cleaning.py

import pandas as pd
import numpy as np

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("NBA_Data.csv")

# Drop unnamed index column if it exists
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# -----------------------------
# Normalize text fields
# -----------------------------
text_columns = ["Player", "Colleges", "Pos", "Tm", "Team"]

for col in text_columns:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

# -----------------------------
# Convert height to numeric (feet-inches -> inches)
# -----------------------------
def height_to_inches(h):
    try:
        feet, inches = h.split("-")
        return int(feet) * 12 + int(inches)
    except:
        return np.nan

if "Ht" in df.columns:
    df["Ht"] = df["Ht"].apply(height_to_inches)

# -----------------------------
# Convert year to datetime
# -----------------------------
if "Year" in df.columns:
    df["Year"] = df["Year"].astype(int)
    df["season_date"] = pd.to_datetime(df["Year"], format="%Y")

# -----------------------------
# Handle missing values
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# Fix obvious data entry errors / outliers
# -----------------------------
df = df[
    (df["Age"] >= 18) & (df["Age"] <= 45) &
    (df["MP"] >= 0) & (df["MP"] <= 48) &
    (df["G"] <= 82) &
    (df["FG%"] >= 0) & (df["FG%"] <= 1) &
    (df["3P%"] >= 0) & (df["3P%"] <= 1) &
    (df["FT%"] >= 0) & (df["FT%"] <= 1)
]

# -----------------------------
# Encode categorical variables
# -----------------------------
categorical_features = ["Pos", "Tm", "Team"]
existing_features = [c for c in categorical_features if c in df.columns]

df = pd.get_dummies(df, columns=existing_features, drop_first=True)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv("NBA_Data_Cleaned.csv", index=False)

print("Data cleaning completed. Saved as NBA_Data_Cleaned.csv")

