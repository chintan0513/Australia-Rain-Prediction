from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

PORT = int(os.getenv("PORT", 8080))
app = FastAPI(title="Australia Rain Predictor")

# Schema (only 9 raw inputs for simplicity)
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

# Load model and preprocessing objects
MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_columns.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)  # list of 26 feature names

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert request into dict
    input_dict = {
        "MinTemp": req.MinTemp,
        "MaxTemp": req.MaxTemp,
        "Rainfall": req.Rainfall,
        "Evaporation": req.Evaporation,
        "Sunshine": req.Sunshine,
        "WindGustSpeed": req.WindGustSpeed,
        "WindSpeed9am": req.WindSpeed9am,
        "Humidity9am": req.Humidity9am,
        "Pressure9am": req.Pressure9am,
    }

    # Build row with all expected columns (missing cols → 0)
    row = [input_dict.get(col, 0) for col in feature_columns]

    # Apply scaler
    row_scaled = scaler.transform([row])

    # Predict
    prediction = model.predict(row_scaled)
    prob = float(prediction[0][0])

    return {"rain_probability": prob, "prediction": int(prob >= 0.5)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
