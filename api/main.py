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
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
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
    return ModelInfoResponse(
        model_type="XGBoost Classifier",
        features=feature_names,
        hyperparameters=hyperparameters,
        performance_metrics={
            "note": "Performance metrics should be loaded from evaluation results"})
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model is trained and saved.")
    try:
        features = np.array([[
            request.pregnancies,
            request.glucose,
            request.blood_pressure,
            request.skin_thickness,
            request.insulin,
            request.bmi,
            request.diabetes_pedigree_function,
            request.age
        ]])
        if scaler is not None:
            features = scaler.transform(features)
        if feature_selector is not None:
            selected = getattr(feature_selector, "selected_features", None)
            if selected:
                indices = [feature_names.index(name) for name in selected]
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Prediction failed: {str(e)}")
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
