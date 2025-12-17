import pandas as pd

# Load full cleaned dataset
df = pd.read_csv("NBA_Data_Cleaned.csv", parse_dates=["season_date"])

# Select only shooting stats + target + date
shooting_cols = [
    "Player", "FG%","3P%",
 "2P%", "eFG%", "FT%",
    "PTS", "Year", "season_date"
]

df_shooting = df[shooting_cols]   # <-- defines df_shooting

# Sort by date
df_shooting = df_shooting.sort_values("season_date").reset_index(drop=True)

# Split chronologically
n = len(df_shooting)
train_end = int(n * 0.35)
validate_end = int(n * 0.70)

train_df = df_shooting.iloc[:train_end]
validate_df = df_shooting.iloc[train_end:validate_end]
test_df = df_shooting.iloc[validate_end:]

# Save splits
train_df.to_csv("train_shooting.csv", index=False)
validate_df.to_csv("validate_shooting.csv", index=False)
test_df.to_csv("test_shooting.csv", index=False)

print("Time-based split complete:")
print(f"Train rows: {len(train_df)}")
print(f"Validate rows: {len(validate_df)}")
print(f"Test rows: {len(test_df)}")
