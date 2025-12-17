import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import mlflow
import mlflow.h2o
import pandas as pd

# ----------------------------
# CONFIG: MLflow
# ----------------------------
mlflow.set_tracking_uri("http://18.232.31.184:2000")
mlflow.set_experiment("Test_H2O_S3_Artifacts")

# ----------------------------
# CREATE DUMMY DATA
# ----------------------------
data = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5],
    "x2": [5, 4, 3, 2, 1],
    "y": [1, 0, 1, 0, 1]
})

h2o.init(max_mem_size="2G")
hf = h2o.H2OFrame(data)

# ----------------------------
# TRAIN AND LOG MODEL
# ----------------------------
with mlflow.start_run(run_name="GLM_Test") as run:
    model = H2OGeneralizedLinearEstimator()
    model.train(x=["x1", "x2"], y="y", training_frame=hf)
    
    # log model to MLflow (artifact_path ensures it's stored in S3 if server is configured)
    mlflow.h2o.log_model(model, artifact_path="GLM_Test")
    
    # Print the artifact URI
    artifact_uri = mlflow.get_artifact_uri("GLM_Test")
    print("Artifact should be in S3 at:", artifact_uri)

h2o.cluster().shutdown(prompt=False)
