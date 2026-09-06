"""
app.py — API de prédiction de churn pour Google Cloud Run.
Version sans base de données : le conteneur charge le modèle et sert les prédictions.
"""
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")

with open("feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

with open("best_model_name.txt") as f:
    MODEL_NAME = f.read().strip()


class ClientInput(BaseModel):
    gender: float
    senior_citizen: float
    partner: float
    dependents: float
    tenure: float
    phone_service: float
    online_security: float
    online_backup: float
    device_protection: float
    tech_support: float
    streaming_tv: float
    streaming_movies: float
    paperless_billing: float
    monthly_charges: float
    total_charges: float
    multiple_lines_no_phone_service: float
    multiple_lines_yes: float
    internet_service_fiber_optic: float
    internet_service_no: float
    contract_one_year: float
    contract_two_year: float
    payment_method_credit_card_automatic: float
    payment_method_electronic_check: float
    payment_method_mailed_check: float
    charge_moyenne: float
    client_recent: float


@app.get("/")
def home():
    return {"status": "ok", "model": MODEL_NAME, "features": FEATURE_NAMES}


@app.post("/predict")
def predict(client: ClientInput):
    donnees_client = client.model_dump()
    client_df = pd.DataFrame([donnees_client])[FEATURE_NAMES]

    features_scaled = scaler.transform(client_df)
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    return {
        "prediction": "churn" if pred == 1 else "reste",
        "proba_churn": round(float(proba[1]), 4),
        "model_used": MODEL_NAME,
    }