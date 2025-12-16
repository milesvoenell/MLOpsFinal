import h2o
from h2o.automl import H2OAutoML

# Start H2O
h2o.init(max_mem_size="4G")

# Load shooting-only splits
train = h2o.import_file("train_shooting.csv")
validate = h2o.import_file("validate_shooting.csv")
test = h2o.import_file("test_shooting.csv")

# Define target and features
target = "PTS"
features = [
    "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%"
]

# Run AutoML
aml = H2OAutoML(
    max_runtime_secs=600,
    sort_metric="RMSE",
    seed=42,
    exclude_algos=["DeepLearning"]
)

aml.train(
    x=features,
    y=target,
    training_frame=train,
    validation_frame=validate
)

# Leaderboard
leaderboard = aml.leaderboard
print("Top 10 models:")
print(leaderboard.head(rows=10))

# Evaluate top model on test set
leader = aml.leader
perf = leader.model_performance(test)
print("\nTest RMSE:", perf.rmse())
print("Test MAE:", perf.mae())
print("Test R²:", perf.r2())

# Identify top 3 model types
lb_df = leaderboard.as_data_frame()
lb_df["model_type"] = lb_df["model_id"].apply(lambda x: x.split("_")[0])
top_3_models = lb_df.groupby("model_type")["rmse"].min().sort_values().head(3)
print("\nTop 3 model types by RMSE:")
print(top_3_models)
