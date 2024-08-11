from fastapi import FastAPI, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predict import model_service

app = FastAPI(title="Sklearn Model Serving API")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        predictions = await model_service.predict(request.features)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "send POST to /predict to get predictions"}