import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.h2o
import pandas as pd
from datetime import datetime

# MLflow setup
mlflow.set_tracking_uri("http://localhost:2000")
mlflow.set_experiment("Remote_H2O_Shooting_Models")

# Load data
train = pd.read_csv("train_shooting.csv")
valid = pd.read_csv("valid_shooting.csv")
test  = pd.read_csv("test_shooting.csv")

features = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%"]
target = "PTS"

# Start H2O
h2o.init(max_mem_size="4G")
train_hf = h2o.H2OFrame(train)
valid_hf = h2o.H2OFrame(valid)
test_hf  = h2o.H2OFrame(test)

print(f"{datetime.now()} - Starting Stacked Ensemble training...")

with mlflow.start_run(run_name="StackedEnsemble_Shooting") as run:
    # Train AutoML
    aml = H2OAutoML(max_runtime_secs=600, seed=42)
    aml.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

    # Log leaderboard (all base models)
    lb = aml.leaderboard.as_data_frame()
    lb_file = "leaderboard.csv"
    lb.to_csv(lb_file, index=False)
    mlflow.log_artifact(lb_file)
    print(f"{datetime.now()} - Leaderboard logged to MLflow.")

    # Log each base model metrics
    for model_id in aml.leaderboard['model_id']:
        base_model = h2o.get_model(model_id)
        perf = base_model.model_performance(test_hf)
        mlflow.log_param(f"{model_id}_type", base_model.algo)
        mlflow.log_metric(f"{model_id}_rmse", perf.rmse())
        mlflow.log_metric(f"{model_id}_mae", perf.mae())
        mlflow.log_metric(f"{model_id}_r2", perf.r2())

    # Log final stacked ensemble (leader)
    leader = aml.leader
    leader_perf = leader.model_performance(test_hf)
    mlflow.log_param("model_type", "StackedEnsemble")
    mlflow.log_metric("rmse", leader_perf.rmse())
    mlflow.log_metric("mae", leader_perf.mae())
    mlflow.log_metric("r2", leader_perf.r2())
    mlflow.h2o.log_model(leader, artifact_path="StackedEnsemble_Shooting")

    # Register model
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/StackedEnsemble_Shooting"
    mlflow.register_model(model_uri=model_uri, name="StackedEnsemble_Shooting")

print(f"{datetime.now()} - StackedEnsemble_Shooting training and registration complete.")
h2o.cluster().shutdown(prompt=False)
