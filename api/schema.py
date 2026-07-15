from pydantic import BaseModel, Field
from typing import List
class PredictionRequest(BaseModel):
    pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    blood_pressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, description="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree_function: float = Field(..., ge=0, description="Diabetes pedigree function")
    age: int = Field(..., ge=0, description="Age (years)")
    model_config = {"json_schema_extra": {"example": {"pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35, "insulin": 0, "bmi": 33.6, "diabetes_pedigree_function": 0.627, "age": 50}}}
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0 = Non-diabetic, 1 = Diabetic)")
    probability: float = Field(..., ge=0, le=1, description="Probability of being diabetic")
    confidence: str = Field(..., description="Confidence level of the prediction")
    model_config = {"json_schema_extra": {"example": {"prediction": 1, "probability": 0.85, "confidence": "High"}}}
class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")
class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Type of model")
    features: List[str] = Field(..., description="List of features used")
    hyperparameters: dict = Field(..., description="Model hyperparameters")
    performance_metrics: dict = Field(..., description="Model performance metrics")
