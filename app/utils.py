# app/utils.py
import pandas as pd
from datetime import datetime

def predict_with_metadata(model, model_name, input_records):
    df = pd.DataFrame(input_records)
    preds = model.predict(df)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    
    return {
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "predictions": preds,
        "n_records": len(input_records)
    }
