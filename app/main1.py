from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models import predict_model1

app = FastAPI(title="MLflow FastAPI Deployment (GLM Only)")

class Features(BaseModel):
    features: List[Dict[str, Any]]  # single record or batch

@app.post("/predict_model1")
async def predict_model1_endpoint(data: Features):
    try:
        return predict_model1(data.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_model2")
async def predict_model2_endpoint(data: Features):
    raise HTTPException(status_code=400, detail="Model 'model2' is not available in this deployment.")

@app.post("/predict_model3")
async def predict_model3_endpoint(data: Features):
    raise HTTPException(status_code=400, detail="Model 'model3' is not available in this deployment.")
