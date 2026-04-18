from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app=FastAPI()

model=joblib.load('my_trained_model.pkl')

class QualityData(BaseModel):



app.post