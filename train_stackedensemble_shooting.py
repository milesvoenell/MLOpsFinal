import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o
import pandas as pd
from datetime import datetime

# Function for timestamped logs
def log(msg):
    print(f"{datetime.now()} - {msg}")

# MLflow setup
mlflow.set_tracking_uri("http://18.232.31.184:2000")
mlflow.set_experiment("Remote_H2O_Shooting_Models")

# Load data
train = pd.read_csv("train_shooting.csv")
valid = pd.read_csv("valid_shooting.csv")
test  = pd.read_csv("test_shooting.csv")

features = ["FG%", "3P%",  "2P%", "eFG%", "FT%"]
target = "PTS"

# Start H2O cluster with progress printing
log("Initializing H2O cluster...")
h2o.init(max_mem_size="4G", nthreads=-1, strict_version_check=False)

train_hf = h2o.H2OFrame(train)
valid_hf = h2o.H2OFrame(valid)
test_hf  = h2o.H2OFrame(test)

log("Starting H2O AutoML (Stacked Ensemble)...")

with mlflow.start_run(run_name="StackedEnsemble_Shooting") as run:
    aml = H2OAutoML(
        max_runtime_secs=600,
        seed=42,
        verbosity="info",  # <-- ensures H2O logs progress
        project_name="Shooting_AutoML"
    )

    aml.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

    # Print leaderboard
    log("AutoML Leaderboard:")
    leaderboard_df = aml.leaderboard.as_data_frame()
    print(leaderboard_df)

    # Leader model
    stacked_model = aml.leader

    # Evaluate performance on test set
    perf = stacked_model.model_performance(test_hf)
    rmse = perf.rmse()
    mae = perf.mae()
    r2 = perf.r2()

    log(f"Stacked Ensemble complete. Test set performance -> RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "StackedEnsemble")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log the stacked ensemble model
    mlflow.h2o.log_model(stacked_model, artifact_path="StackedEnsemble_Shooting")
    log("Stacked Ensemble model logged to MLflow.")

    # Save and log leaderboard as CSV
    leaderboard_csv = "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    mlflow.log_artifact(leaderboard_csv)
    log("Leaderboard CSV logged to MLflow.")

    # Register model
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/StackedEnsemble_Shooting"
    mlflow.register_model(model_uri=model_uri, name="StackedEnsemble_Shooting")
    log(f"Stacked Ensemble registered. Run ID: {run_id}")

# Shutdown H2O
h2o.cluster().shutdown(prompt=False)
log("H2O cluster shutdown. Training and registration complete.")
