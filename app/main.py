# app/main.py
from fastapi import FastAPI
from app.models import glm_model, gbm_model, predict
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Shooting Models API")

# --- Pydantic schema for validation ---
class InputRecord(BaseModel):
    FG_pct: float
    TP_pct: float
    TWOP_pct: float
    eFG_pct: float
    FT_pct: float

@app.post("/predict_glm")
def predict_glm(data: List[InputRecord]):
    return predict(glm_model, [d.dict() for d in data])

@app.post("/predict_gbm")
def predict_gbm(data: List[InputRecord]):
    return predict(gbm_model, [d.dict() for d in data])
