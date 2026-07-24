from pydantic import BaseModel, Field
from typing import List
class PredictionRequest(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="High blood pressure indicator")
    HighChol: int = Field(..., ge=0, le=1, description="High cholesterol indicator")
    CholCheck: int = Field(..., ge=0, le=1, description="Cholesterol check completed")
    BMI: float = Field(..., ge=0, description="Body mass index")
    Smoker: int = Field(..., ge=0, le=1, description="Smoker indicator")
    Stroke: int = Field(..., ge=0, le=1, description="Stroke history indicator")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Heart disease or attack indicator")
    PhysActivity: int = Field(..., ge=0, le=1, description="Physical activity indicator")
    Fruits: int = Field(..., ge=0, le=1, description="Fruit consumption indicator")
    Veggies: int = Field(..., ge=0, le=1, description="Vegetable consumption indicator")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Heavy alcohol consumption indicator")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Has any healthcare coverage")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Cannot see doctor because of cost")
    GenHlth: int = Field(..., ge=0, description="General health rating")
    MentHlth: int = Field(..., ge=0, description="Number of days mental health was not good")
    PhysHlth: int = Field(..., ge=0, description="Number of days physical health was not good")
    DiffWalk: int = Field(..., ge=0, le=1, description="Difficulty walking indicator")
    Sex: int = Field(..., ge=0, le=1, description="Sex indicator (0 or 1)")
    Age: int = Field(..., ge=0, description="Age in years")
    Education: int = Field(..., ge=0, description="Education level")
    Income: int = Field(..., ge=0, description="Income level")
    model_config = {"json_schema_extra": {"example": {"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 28.5, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 5, "PhysHlth": 2, "DiffWalk": 0, "Sex": 1, "Age": 45, "Education": 4, "Income": 3}}}
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
