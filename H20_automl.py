import h2o
from h2o.automl import H2OAutoML

# Start H2O
h2o.init(max_mem_size="4G")

# Load shooting splits
train = h2o.import_file("train_shooting.csv")
validate = h2o.import_file("validate_shooting.csv")
test = h2o.import_file("test_shooting.csv")

# Define target and features
target = "PTS"
features = ["FG%", "3P%", "2P%", "eFG%", "FT%"]

# Run AutoML (exclude DeepLearning for simplicity)
aml = H2OAutoML(
    max_runtime_secs=600,
    sort_metric="RMSE",
    seed=42,
    exclude_algos=["DeepLearning"]
)
aml.train(x=features, y=target, training_frame=train, validation_frame=validate)

# Leaderboard
leaderboard = aml.leaderboard
print("Top 10 models (Validation RMSE):")
print(leaderboard.head(rows=10))

# Evaluate the best model on test set
leader = aml.leader
perf = leader.model_performance(test)
print("\nTop model performance on test set:")
print(f"RMSE: {perf.rmse():.4f}")
print(f"MAE: {perf.mae():.4f}")
print(f"R²: {perf.r2():.4f}")

# Identify top 3 model types based on best RMSE per type
lb_df = leaderboard.as_data_frame()
lb_df["model_type"] = lb_df["model_id"].apply(lambda x: x.split("_")[0])

# Get minimum RMSE per model type
top_3_models = lb_df.groupby("model_type")["rmse"].min().sort_values().head(3)
print("\nTop 3 model types by RMSE on validation set:")
print(top_3_models)
