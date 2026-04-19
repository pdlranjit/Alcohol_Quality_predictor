from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
import pandas as pd
import numpy as np


app=FastAPI()

model=joblib.load('my_trained_model.pkl')
scaler=joblib.load('scaler.pkl')

class WineInput(BaseModel):
    fixed_acidity: float = Field(...,example=3.4,description="Fixed Accidity level",gt=0,lt=15),
    volatile_acidity: float= Field(...,example=3.2,description="Volatile acidity",gt=0,lt=5),
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def home():
    return {"message": "Wine Quality API is running 🍷"}


@app.post("/predict")
def predict_quality(data: WineInput):

    # Convert input to array
    input_data = np.array([[
        data.fixed_acidity,
        data.volatile_acidit,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)

    return {
        "THe predicted_quality  is ": int(prediction[0])
    }