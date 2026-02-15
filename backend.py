from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load scaler and all models
scaler = joblib.load('scaler.joblib')

models = {
    "logistic_regression": joblib.load('logistic_regression_model.joblib'),
    "decision_tree":       joblib.load('decision_tree_model.joblib'),
    "random_forest":       joblib.load('random_forest_model.joblib'),
    "svm":                 joblib.load('svm_model.joblib')
}

class PatientData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_induced_angina: int
    st_depression: float
    st_slope: int
    num_major_vessels: int
    thalassemia: int

@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running",
        "available_models": list(models.keys()),
        "default_model": "logistic_regression"
    }

@app.post("/predict")
def predict(data: PatientData, model_name: str = Query("logistic_regression")):
    if model_name not in models:
        raise HTTPException(400, f"Invalid model. Use: {list(models.keys())}")

    try:
        input_data = np.array([[data.age, data.sex, data.chest_pain_type,
                                data.resting_blood_pressure, data.cholesterol,
                                data.fasting_blood_sugar, data.resting_ecg,
                                data.max_heart_rate, data.exercise_induced_angina,
                                data.st_depression, data.st_slope,
                                data.num_major_vessels, data.thalassemia]])

        input_scaled = scaler.transform(input_data)
        model = models[model_name]
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return {
            "model_used": model_name,
            "prediction": int(pred),
            "probability": float(prob),
            "interpretation": "Heart disease likely" if pred == 1 else "No heart disease"
        }
    except Exception as e:
        raise HTTPException(500, str(e))