from fastapi import FastAPI

from pydantic import BaseModel


import logging

logger = logging.getLogger("uvicorn.error")

import joblib

app = FastAPI(title="House Price Prediction API")

model = joblib.load("house_price_model.pkl")

class HouseInput(BaseModel):
  MedInc: float
  HouseAge: float
  AveRooms: float
  AveBedrms: float
  Population: float
  AveOccup: float
  Latitude: float
  Longitude: float


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}
  

@app.post("/predict")
def predict(data: HouseInput):

    logger.info(f"Prediction request received: {data}")

    features = [[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]]

    prediction = model.predict(features)

    logger.info(f"Prediction generated: {prediction[0]}")

    return {
        "predicted_house_price": float(prediction[0])
    }
  