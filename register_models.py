import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://ec2-18-232-31-184.compute-1.amazonaws.com:5000")
mlflow.set_experiment("Remote_H2O_Shooting_Models")

# ------------------------------
# Replace these run_ids with actual printed run_ids from training
# ------------------------------
models_info = [
    {"name": "GLM", "artifact_path": "glm_model", "run_id": "<GLM_RUN_ID>"},
    {"name": "GBM", "artifact_path": "gbm_model", "run_id": "<GBM_RUN_ID>"},
    {"name": "StackedEnsemble", "artifact_path": "stackedensemble_model", "run_id": "<STACKED_RUN_ID>"}
]

for m in models_info:
    mlflow.register_model(
        model_uri=f"runs:/{m['run_id']}/{m['artifact_path']}",
        name=f"Shooting_{m['name']}"
    )
    print(f"Registered model Shooting_{m['name']}")

# ------------------------------
# Optional: Comparison table (fill metrics if desired)
# ------------------------------
comparison = pd.DataFrame({
    "Model": ["GLM", "GBM", "StackedEnsemble"],
    "RMSE": [2.35, 1.90, 1.75],  # replace with real metrics
    "MAE": [1.80, 1.45, 1.35],
    "R2":  [0.82, 0.88, 0.90]
})

print("\n=== Model Comparison ===")
print(comparison)
