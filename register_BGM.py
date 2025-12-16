import mlflow
import mlflow.h2o

# Set your MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:2000")
experiment_name = "Remote_H2O_Shooting_Models"
mlflow.set_experiment(experiment_name)

# Dictionary of run_ids and corresponding model names
runs_to_register = {
    "fac9bcf4344e49e7b888a78878c49776": "GLM_Shooting",
    "abd1a3369ef8459e9ec810f979626df2": "GBM_Shooting",
    "36fbd3628c034b47aee2ee230f2e2f5e": "StackedEnsemble_Shooting"
}

for run_id, model_name in runs_to_register.items():
    model_uri = f"runs:/{run_id}/model"  # adjust artifact path if different
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    print(f"Registered {model_name} from run {run_id} at version {registered_model.version}")
