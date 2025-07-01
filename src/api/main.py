from fastapi import FastAPI
from src.api.pydantic_models import RiskRequest
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/credit_model.pkl")

@app.post("/predict")
def predict_risk(data: RiskRequest):
    input_data = np.array([list(data.dict().values())])
    prediction = model.predict_proba(input_data)[0][1]
    return {"risk_probability": prediction}
