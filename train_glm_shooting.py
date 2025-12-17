import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import mlflow
import mlflow.h2o
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIG: Remote MLflow server
# ----------------------------
# Use the REST API endpoint (without #/)
MLFLOW_TRACKING_URI = "http://ec2-18-232-31-184.compute-1.amazonaws.com:5000"  # replace port if needed
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Remote_H2O_Shooting_Models")

# ----------------------------
# DATA
# ----------------------------
train = pd.read_csv("train_shooting.csv")
valid = pd.read_csv("valid_shooting.csv")
test  = pd.read_csv("test_shooting.csv")

# sanitize column names
train = train.rename(columns=lambda x: x.replace('%','_pct'))
valid = valid.rename(columns=lambda x: x.replace('%','_pct'))
test  = test.rename(columns=lambda x: x.replace('%','_pct'))

features = ["FG_pct","3P_pct","2P_pct","eFG_pct","FT_pct"]
target = "PTS"

# ----------------------------
# H2O INIT
# ----------------------------
h2o.init(max_mem_size="4G")
train_hf = h2o.H2OFrame(train)
valid_hf = h2o.H2OFrame(valid)
test_hf  = h2o.H2OFrame(test)

# ----------------------------
# TRAIN, LOG, REGISTER GLM
# ----------------------------
with mlflow.start_run(run_name="GLM") as run:
    try:
        # Train H2O GLM with progress
        model = H2OGeneralizedLinearEstimator(
            lambda_search=True,
            max_iterations=50,
            alpha=0.5,
            compute_p_values=True
        )
        print(f"{datetime.now()} - Starting GLM training...")
        model.train(
            x=features,
            y=target,
            training_frame=train_hf,
            validation_frame=valid_hf
        )
        print(f"{datetime.now()} - GLM training completed!")

        # Evaluate performance
        perf_train = model.model_performance(train_hf)
        perf_valid = model.model_performance(valid_hf)
        perf_test  = model.model_performance(test_hf)

        metrics = {
            "train_rmse": perf_train.rmse(),
            "train_mae": perf_train.mae(),
            "train_r2": perf_train.r2(),
            "valid_rmse": perf_valid.rmse(),
            "valid_mae": perf_valid.mae(),
            "valid_r2": perf_valid.r2(),
            "test_rmse": perf_test.rmse(),
            "test_mae": perf_test.mae(),
            "test_r2": perf_test.r2()
        }

        print(f"{datetime.now()} - GLM performance metrics: {metrics}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", "GLM")
        mlflow.log_param("features", features)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model artifact to MLflow
        mlflow.h2o.log_model(model, artifact_path="GLM")
        print(f"{datetime.now()} - Model logged to MLflow artifacts.")

        # Register model safely
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/GLM"
        try:
            mlflow.register_model(model_uri=model_uri, name="GLM_Shooting")
            print(f"{datetime.now()} - GLM registered at {model_uri}")
        except Exception as e:
            print(f"{datetime.now()} - Model registration error: {e}")

    except Exception as e:
        print(f"{datetime.now()} - GLM failed: {e}")
        mlflow.log_param("error", str(e))

# Shutdown H2O cluster
h2o.cluster().shutdown(prompt=False)
