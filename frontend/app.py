import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
API_URL = "http://localhost:8000"
def load_model_locally():
    try:
        model_path = PROJECT_ROOT / "models/xgboost_model.joblib"
        scaler_path = PROJECT_ROOT / "models/scaler.joblib"
        selector_path = PROJECT_ROOT / "models/feature_selector.joblib"
        
        if model_path.exists() and scaler_path.exists() and selector_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selector = joblib.load(selector_path)
            return model, scaler, selector
    except Exception as e:
        st.error(f"Error loading local model: {e}")
    return None, None, None
def predict_via_api(features: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=5)
        if response.status_code == 200:
            return response.json(), None
        else:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return None, f"API error ({response.status_code}): {detail}"
    except requests.RequestException as e:
        return None, f"Could not reach the API: {e}"
def predict_locally(features: list, model, scaler, selector):
    try:
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        all_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        selected = selector.selected_features
        features_selected = features_scaled[:, [all_features.index(name) for name in selected]]
        prediction = model.predict(features_selected)[0]
        probability = model.predict_proba(features_selected)[0, 1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": "High" if probability >= 0.8 else "Medium" if probability >= 0.6 else "Low"
        }
    except Exception as e:
        return None
def main():
    st.markdown('<h1 class="main-header">🩺 Diabetes Prediction System</h1>', 
                unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Prediction", "About Model", "Feature Information"]
    )
    model, scaler, selector = load_model_locally()
    use_api = st.sidebar.checkbox("Use API (if available)", value=True)
    
    if use_api:
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=2)
            api_available = health_response.status_code == 200 and health_response.json().get("model_loaded", False)
        except requests.RequestException:
            api_available = False
            use_api = False
    else:
        api_available = False
    if not api_available and model is None:
        st.warning("⚠️ Neither API nor local model is available. Please train the model first.")
        return
    if page == "Prediction":
        st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input(
                    "Pregnancies", 
                    min_value=0, 
                    max_value=20, 
                    value=1, 
                    help="Number of times pregnant"
                )
                glucose = st.number_input(
                    "Glucose (mg/dL)", 
                    min_value=0, 
                    max_value=300, 
                    value=120,
                    help="Plasma glucose concentration"
                )
                blood_pressure = st.number_input(
                    "Blood Pressure (mm Hg)", 
                    min_value=0, 
                    max_value=200, 
                    value=70,
                    help="Diastolic blood pressure"
                )
                skin_thickness = st.number_input(
                    "Skin Thickness (mm)", 
                    min_value=0, 
                    max_value=100, 
                    value=20,
                    help="Triceps skin fold thickness"
                )
            
            with col2:
                insulin = st.number_input(
                    "Insulin (mu U/ml)", 
                    min_value=0, 
                    max_value=900, 
                    value=80,
                    help="2-Hour serum insulin"
                )
                bmi = st.number_input(
                    "BMI", 
                    min_value=0.0, 
                    max_value=70.0, 
                    value=25.0,
                    help="Body mass index"
                )
                dpf = st.number_input(
                    "Diabetes Pedigree Function", 
                    min_value=0.0, 
                    max_value=3.0, 
                    value=0.5,
                    help="Diabetes pedigree function"
                )
                age = st.number_input(
                    "Age (years)", 
                    min_value=0, 
                    max_value=120, 
                    value=35,
                    help="Age in years"
                )
            submitted = st.form_submit_button("Predict Diabetes Risk", use_container_width=True)
        if submitted:
            st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
            if use_api and api_available:
                features = {
                    "pregnancies": int(pregnancies),
                    "glucose": float(glucose),
                    "blood_pressure": float(blood_pressure),
                    "skin_thickness": float(skin_thickness),
                    "insulin": float(insulin),
                    "bmi": float(bmi),
                    "diabetes_pedigree_function": float(dpf),
                    "age": int(age)
                }
                result, error = predict_via_api(features)
            else:
                features_list = [
                    pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, dpf, age
                ]
                result = predict_locally(features_list, model, scaler, selector)
                error = "Local prediction failed; ensure the model and preprocessing artifacts were created together."
            if result:
                prediction = result["prediction"]
                probability = result["probability"]
                confidence = result["confidence"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                        st.metric("Prediction", "Diabetic", delta="High Risk")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.metric("Prediction", "Non-Diabetic", delta="Low Risk")
                        st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.metric("Probability", f"{probability:.2%}")
                with col3:
                    st.metric("Confidence", confidence)
                st.markdown('<h2 class="sub-header">Probability Visualization</h2>', 
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 2))
                colors = ['#e74c3c' if probability > 0.5 else '#27ae60']
                ax.barh(['Diabetes Probability'], [probability], color=colors, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Diabetes Risk Probability')
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
                ax.text(0.5, 0, 'Threshold', ha='center', va='bottom', fontsize=8)
                st.pyplot(fig)
                st.markdown('<h2 class="sub-header">Risk Assessment</h2>',unsafe_allow_html=True)
                if probability < 0.3:
                    st.markdown('<div class="success-box">✅ Low Risk: Your diabetes risk is low. Continue maintaining a healthy lifestyle.</div>', 
                                unsafe_allow_html=True)
                elif probability < 0.5:
                    st.markdown('<div class="warning-box">⚠️ Moderate Risk: Consider consulting a healthcare provider for a check-up.</div>', 
                                unsafe_allow_html=True)
                elif probability < 0.7:
                    st.markdown('<div class="warning-box">⚠️ Elevated Risk: Please consult a healthcare provider for proper evaluation.</div>', 
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="danger-box">🚨 High Risk: Immediate medical consultation is recommended.</div>', 
                                unsafe_allow_html=True)
            else:
                st.error(error)
    elif page == "About Model":
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        st.markdown("""
        ### Diffusion-Enhanced TSO-HBA Optimized XGBoost Framework
        
        This diabetes prediction system uses an advanced machine learning framework combining:
        
        - **Tabular Diffusion Model**: Generates synthetic patient data for data augmentation
        - **Hybrid TSO-HBA Optimization**: Custom metaheuristic algorithm for hyperparameter tuning
        - **XGBoost Classifier**: State-of-the-art gradient boosting algorithm
        - **Multi-Method Feature Selection**: Correlation filtering, Mutual Information, RFE, and XGBoost importance
        
        ### Model Performance
        
        The model is evaluated using multiple metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - ROC-AUC
        
        ### Features Used
        
        The model uses the following patient features:
        - Pregnancies
        - Glucose level
        - Blood pressure
        - Skin thickness
        - Insulin level
        - BMI
        - Diabetes pedigree function
        - Age
        """)
    elif page == "Feature Information":
        st.markdown('<h2 class="sub-header">Feature Descriptions</h2>', unsafe_allow_html=True)
        feature_info = {
            "Pregnancies": "Number of times the patient has been pregnant",
            "Glucose": "Plasma glucose concentration after a 2-hour oral glucose tolerance test (mg/dL)",
            "Blood Pressure": "Diastolic blood pressure (mm Hg)",
            "Skin Thickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg/(height in m)^2)",
            "Diabetes Pedigree Function": "A function that scores likelihood of diabetes based on family history",
            "Age": "Age of the patient (years)"
        }
        for feature, description in feature_info.items():
            with st.expander(feature):
                st.write(description)
        st.markdown('<h2 class="sub-header">Normal Ranges</h2>', unsafe_allow_html=True)
        normal_ranges = {"Glucose": "70-140 mg/dL (fasting: 70-99 mg/dL)","Blood Pressure": "60-80 mm Hg (diastolic)","BMI": "18.5-24.9 (normal weight)","Insulin": "16-166 mIU/L (fasting: 2-20 mIU/L)"}
        for feature, range_val in normal_ranges.items():
            st.write(f"**{feature}**: {range_val}")
if __name__ == "__main__":
    main()
