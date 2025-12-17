import mlflow.pyfunc
from datetime import datetime

# Load only model1 from the run
RUN_ID = "aae449f6509347f28b751eb85ea9304e"
MODEL_URI = f"runs:/{RUN_ID}/model1"

# Load the GLM model
model1 = mlflow.pyfunc.load_model(MODEL_URI)

def predict_model1(input_data):
    predictions = model1.predict(input_data)
    return {
        "model_name": "model1",
        "model_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions
    }
