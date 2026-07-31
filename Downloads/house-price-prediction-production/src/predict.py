# src/predict.py

import joblib
import pandas as pd

# Load trained model once when the application starts
MODEL_PATH = "model/model.pkl"
model = joblib.load(MODEL_PATH)


def predict_house_price(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str
):
    """
    Predict the house price for a single record.
    """

    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    prediction = model.predict(input_data)

    return float(prediction[0])


def predict_batch(df: pd.DataFrame):
    """
    Predict prices for multiple houses.
    Input: DataFrame
    Output: List of predictions
    """

    predictions = model.predict(df)

    return predictions.tolist()