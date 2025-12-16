import pandas as pd

# Load full cleaned dataset
df = pd.read_csv("NBA_Data_Cleaned.csv", parse_dates=["season_date"])

# Select only shooting stats + target + date (PTS is target)
shooting_cols = [
    "Player", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%",
    "PTS", "Year", "Pts Won", "season_date"
]

df_shooting = df[shooting_cols]

# Save new CSV
df_shooting.to_csv("NBA_ShootingOnly.csv", index=False)
print("Saved shooting-only dataset to NBA_ShootingOnly.csv")
