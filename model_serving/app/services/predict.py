import pickle
from typing import List
import sys
from app.services.lib.classifiers import TopDownClassifier
import app.services.lib as lib


class ModelService:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model: TopDownClassifier = pickle.load(f)

    async def predict(self, features: List[str]) -> List[List[str]]:
        predictions = self.model.predict(features).tolist()
        return predictions

sys.modules['lib'] = lib
model_service = ModelService("app/models/model.pkl")