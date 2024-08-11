from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[str]

class PredictionResponse(BaseModel):
    predictions: List[List[str]]