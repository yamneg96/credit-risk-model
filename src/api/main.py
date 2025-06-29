from fastapi import FastAPI
from src.api.pydantic_models import CustomerData
from joblib import load
import pandas as pd

app = FastAPI()
model = load("model.joblib")

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    return {"risk_probability": prob}