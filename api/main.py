import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from .schema import PredictionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for diabetes prediction using XGBoost",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
model = None
scaler = None
feature_selector = None
RAW_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
DERIVED_FEATURES = [
    "BMI_category", "Age_group", "metabolic_risk", "lifestyle_risk",
    "healthy_habits", "age_bmi_interaction"
]
FEATURE_NAMES = RAW_FEATURES + DERIVED_FEATURES
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def load_model_and_dependencies() -> None:
    global model, scaler, feature_selector
    
    try:
        model_path = Path("models/xgboost_model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found")
        scaler_path = Path("models/scaler.joblib")
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        selector_path = Path("models/feature_selector.joblib")
        if selector_path.exists():
            feature_selector = joblib.load(selector_path)
            logger.info("Feature selector loaded successfully")      
    except Exception as e:
        logger.error(f"Error loading model dependencies: {e}")
@app.on_event("startup")
async def startup_event():
    load_model_and_dependencies()
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        version="1.0.0")
@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded")
    params = model.get_params()
    hyperparameters = {
        'max_depth': params.get('max_depth'),
        'learning_rate': params.get('learning_rate'),
        'n_estimators': params.get('n_estimators'),
        'gamma': params.get('gamma'),
        'min_child_weight': params.get('min_child_weight'),
        'subsample': params.get('subsample'),
        'colsample_bytree': params.get('colsample_bytree'),
        'reg_alpha': params.get('reg_alpha'),
        'reg_lambda': params.get('reg_lambda')
    }
    features = FEATURE_NAMES
    if feature_selector is not None and getattr(feature_selector, "selected_features", None) is not None:
        features = feature_selector.selected_features
    return ModelInfoResponse(
        model_type="XGBoost Classifier",
        features=features,
        hyperparameters=hyperparameters,
        performance_metrics={
            "note": "Performance metrics should be loaded from evaluation results"})
def compute_derived_features(payload: dict[str, float | int]) -> dict[str, float | int]:
    bmi = float(payload["BMI"])
    age = float(payload["Age"])
    bmi_category = int(np.digitize([bmi], [18.5, 24.9, 29.9])[0])
    age_group = int(np.digitize([age], [30, 50, 65])[0])
    metabolic_risk = int(
        payload["HighBP"] + payload["HighChol"] + payload["HeartDiseaseorAttack"] + payload["Stroke"]
    )
    lifestyle_risk = int(
        payload["Smoker"] + payload["HvyAlcoholConsump"] + payload["DiffWalk"]
    )
    healthy_habits = int(
        payload["PhysActivity"] + payload["Fruits"] + payload["Veggies"]
    )
    return {
        "BMI_category": bmi_category,
        "Age_group": age_group,
        "metabolic_risk": metabolic_risk,
        "lifestyle_risk": lifestyle_risk,
        "healthy_habits": healthy_habits,
        "age_bmi_interaction": float(age * bmi),
    }


def build_feature_vector(payload: dict[str, float | int]) -> np.ndarray:
    derived = compute_derived_features(payload)
    feature_vector = [payload[name] for name in RAW_FEATURES] + [derived[name] for name in DERIVED_FEATURES]
    return np.array([feature_vector])


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained and saved.")
    try:
        payload = {
            "HighBP": request.HighBP,
            "HighChol": request.HighChol,
            "CholCheck": request.CholCheck,
            "BMI": request.BMI,
            "Smoker": request.Smoker,
            "Stroke": request.Stroke,
            "HeartDiseaseorAttack": request.HeartDiseaseorAttack,
            "PhysActivity": request.PhysActivity,
            "Fruits": request.Fruits,
            "Veggies": request.Veggies,
            "HvyAlcoholConsump": request.HvyAlcoholConsump,
            "AnyHealthcare": request.AnyHealthcare,
            "NoDocbcCost": request.NoDocbcCost,
            "GenHlth": request.GenHlth,
            "MentHlth": request.MentHlth,
            "PhysHlth": request.PhysHlth,
            "DiffWalk": request.DiffWalk,
            "Sex": request.Sex,
            "Age": request.Age,
            "Education": request.Education,
            "Income": request.Income,
        }
        features = build_feature_vector(payload)
        if scaler is not None:
            features = scaler.transform(features)
        if feature_selector is not None:
            selected = getattr(feature_selector, "selected_features", None)
            if selected:
                indices = [FEATURE_NAMES.index(name) for name in selected]
                features = features[:, indices]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        if probability >= 0.8:
            confidence = "High"
        elif probability >= 0.6:
            confidence = "Medium"
        elif probability >= 0.4:
            confidence = "Low"
        else:
            confidence = "Very Low"
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")
@app.get("/")
async def root():
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/model-info": "Model information",
            "/predict": "Make prediction (POST)",
            "/docs": "API documentation"}}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
