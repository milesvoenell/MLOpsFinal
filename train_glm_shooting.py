import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import mlflow
import mlflow.h2o
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIG: Remote MLflow server
# ----------------------------
mlflow.set_tracking_uri("http://18.232.31.184:2000")
mlflow.set_experiment("Remote_H2O_Shooting_Models")

# ----------------------------
# DATA
# ----------------------------
train = pd.read_csv("train_shooting.csv")
valid = pd.read_csv("valid_shooting.csv")
test  = pd.read_csv("test_shooting.csv")

train = train.rename(columns=lambda x: x.replace('%','_pct'))
valid = valid.rename(columns=lambda x: x.replace('%','_pct'))
test  = test.rename(columns=lambda x: x.replace('%','_pct'))

features = ["FG","FGA","FG_pct","3P","3PA","3P_pct","2P","2PA","2P_pct",
            "eFG_pct","FT","FTA","FT_pct"]
target = "PTS"

# ----------------------------
# H2O INIT
# ----------------------------
h2o.init(max_mem_size="4G")
train_hf = h2o.H2OFrame(train)
valid_hf = h2o.H2OFrame(valid)
test_hf  = h2o.H2OFrame(test)

# ----------------------------
# TRAIN, LOG, REGISTER
# ----------------------------
with mlflow.start_run(run_name="GLM") as run:
    try:
        model = H2OGeneralizedLinearEstimator()
        model.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)
        
        perf = model.model_performance(test_hf)
        rmse = perf.rmse()
        mae  = perf.mae()
        r2   = perf.r2()
        print(f"{datetime.now()} - GLM performance: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        
        mlflow.log_param("model_type", "GLM")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.h2o.log_model(model, artifact_path="GLM")
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/GLM"
        mlflow.register_model(model_uri=model_uri, name="GLM_Shooting")
    except Exception as e:
        print("GLM failed:", e)
        mlflow.log_param("error", str(e))

h2o.cluster().shutdown(prompt=False)
