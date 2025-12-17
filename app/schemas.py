# app/schemas.py
from pydantic import BaseModel
from typing import List

class ShootingStats(BaseModel):
    FG_pct: float
    TP_pct: float
    TWOP_pct: float
    eFG_pct: float
    FT_pct: float

class PredictionRequest(BaseModel):
    records: List[ShootingStats]
