# app/main.py

from fastapi import FastAPI
import pandas as pd

from src.predict import (
    predict_house_price,
    predict_batch
)

from src.schemas import (
    HouseData,
   
)

app = FastAPI(
    title="House Price Prediction API",
    description="Predict California House Prices using Machine Learning",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is Running!"
    }


@app.post("/predict")
def predict(data: HouseData):

    prediction = predict_house_price(
        longitude=data.longitude,
        latitude=data.latitude,
        housing_median_age=data.housing_median_age,
        total_rooms=data.total_rooms,
        total_bedrooms=data.total_bedrooms,
        population=data.population,
        households=data.households,
        median_income=data.median_income,
        ocean_proximity=data.ocean_proximity
    )

    return {
        "Predicted House Price": prediction
    }


