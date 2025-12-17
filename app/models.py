# app/models.py
import os
import boto3
import mlflow
import pandas as pd
from datetime import datetime

# --- S3 paths ---
S3_BUCKET = "mlflow-artifacts-finalproject"
GLM_ARTIFACT_KEY = "1/1baec3a10579487a9778a0d6c8b90002/artifacts/model.h2o"
GBM_ARTIFACT_KEY = "1/1baec3a10579487a9778a0d6c8b90002/artifacts/model.h2o"

# --- Local paths ---
LOCAL_DIR = "./models"
os.makedirs(LOCAL_DIR, exist_ok=True)

GLM_LOCAL_PATH = os.path.join(LOCAL_DIR, "GLM_Shooting.h2o")
GBM_LOCAL_PATH = os.path.join(LOCAL_DIR, "GBM_Shooting.h2o")

# --- Download from S3 if missing ---
def download_from_s3(key, local_path):
    if not os.path.exists(local_path):
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, key, local_path)
    return local_path

download_from_s3(GLM_ARTIFACT_KEY, GLM_LOCAL_PATH)
download_from_s3(GBM_ARTIFACT_KEY, GBM_LOCAL_PATH)

# --- Load models ---
glm_model = mlflow.h2o.load_model(GLM_LOCAL_PATH)
gbm_model = mlflow.h2o.load_model(GBM_LOCAL_PATH)

# --- Prediction helper ---
def predict(model, input_data):
    df = pd.DataFrame(input_data)
    preds = model.predict(df)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    return {
        "predictions": preds,
        "model_name": model._model_impl_name,
        "timestamp": datetime.utcnow().isoformat()
    }
