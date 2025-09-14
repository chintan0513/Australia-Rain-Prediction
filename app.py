from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

PORT = int(os.getenv("PORT", 8080))

app = FastAPI(title="Australia Rain Predictor")

# Define input schema (must match your dataset features)
class PredictRequest(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustSpeed: float
    WindSpeed9am: float
    Humidity9am: float
    Pressure9am: float

# Load ANN model
MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert input into numpy array in same order as training features
    input_data = np.array([[req.MinTemp, req.MaxTemp, req.Rainfall, req.Evaporation,
                            req.Sunshine, req.WindGustSpeed, req.WindSpeed9am,
                            req.Humidity9am, req.Pressure9am]])
    prediction = model.predict(input_data)
    prob = float(prediction[0][0])  # Assuming binary classification
    return {"rain_probability": prob, "prediction": int(prob >= 0.5)}
