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
    raw_features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
    ]
    derived = compute_derived_features(payload)
    feature_vector = [payload[name] for name in raw_features] + [derived[name] for name in [
        "BMI_category", "Age_group", "metabolic_risk", "lifestyle_risk",
        "healthy_habits", "age_bmi_interaction"
    ]]
    return np.array([feature_vector])

def predict_locally(features: dict, model, scaler, selector):
    try:
        features_array = build_feature_vector(features)
        features_scaled = scaler.transform(features_array)
        selected = selector.selected_features
        if selected:
            indices = [
                ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
                 'BMI_category', 'Age_group', 'metabolic_risk', 'lifestyle_risk',
                 'healthy_habits', 'age_bmi_interaction'].index(name)
                for name in selected
            ]
            features_scaled = features_scaled[:, indices]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, 1]
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
            col1, col2, col3 = st.columns(3)
            with col1:
                high_bp = st.selectbox("High Blood Pressure", [0, 1], index=1, help="1 if the patient has high blood pressure")
                high_chol = st.selectbox("High Cholesterol", [0, 1], index=1, help="1 if the patient has high cholesterol")
                chol_check = st.selectbox("Cholesterol Check", [0, 1], index=1, help="1 if the patient has had a cholesterol check")
                bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, help="Body mass index")
                smoker = st.selectbox("Smoker", [0, 1], index=0, help="1 if the patient smokes")
                stroke = st.selectbox("Stroke History", [0, 1], index=0, help="1 if the patient has had a stroke")
            with col2:
                heart_disease_or_attack = st.selectbox("Heart Disease or Attack", [0, 1], index=0, help="1 if the patient has heart disease or attack history")
                phys_activity = st.selectbox("Physical Activity", [0, 1], index=1, help="1 if the patient is physically active")
                fruits = st.selectbox("Fruits", [0, 1], index=1, help="1 if the patient eats fruits regularly")
                veggies = st.selectbox("Veggies", [0, 1], index=1, help="1 if the patient eats vegetables regularly")
                hvy_alcohol_consump = st.selectbox("Heavy Alcohol Consumption", [0, 1], index=0, help="1 if the patient consumes alcohol heavily")
                any_healthcare = st.selectbox("Has Healthcare", [0, 1], index=1, help="1 if the patient has any healthcare coverage")
            with col3:
                no_doc_bc_cost = st.selectbox("No Doctor Because of Cost", [0, 1], index=0, help="1 if the patient cannot see a doctor because of cost")
                gen_hlth = st.number_input("General Health (1-5)", min_value=1, max_value=5, value=3, help="General health rating")
                ment_hlth = st.number_input("Poor Mental Health Days", min_value=0, max_value=30, value=5, help="Number of days mental health was not good")
                phys_hlth = st.number_input("Poor Physical Health Days", min_value=0, max_value=30, value=2, help="Number of days physical health was not good")
                diff_walk = st.selectbox("Difficulty Walking", [0, 1], index=0, help="1 if the patient has difficulty walking")
                sex = st.selectbox("Sex", [0, 1], index=1, help="0 or 1 gender encoding")
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, help="Age in years")
                education = st.number_input("Education Level", min_value=0, max_value=6, value=4, help="Education level encoding")
                income = st.number_input("Income Level", min_value=0, max_value=8, value=4, help="Income level encoding")
            submitted = st.form_submit_button("Predict Diabetes Risk", use_container_width=True)
        if submitted:
            st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
            features = {
                "HighBP": int(high_bp),
                "HighChol": int(high_chol),
                "CholCheck": int(chol_check),
                "BMI": float(bmi),
                "Smoker": int(smoker),
                "Stroke": int(stroke),
                "HeartDiseaseorAttack": int(heart_disease_or_attack),
                "PhysActivity": int(phys_activity),
                "Fruits": int(fruits),
                "Veggies": int(veggies),
                "HvyAlcoholConsump": int(hvy_alcohol_consump),
                "AnyHealthcare": int(any_healthcare),
                "NoDocbcCost": int(no_doc_bc_cost),
                "GenHlth": int(gen_hlth),
                "MentHlth": int(ment_hlth),
                "PhysHlth": int(phys_hlth),
                "DiffWalk": int(diff_walk),
                "Sex": int(sex),
                "Age": int(age),
                "Education": int(education),
                "Income": int(income),
            }
            if use_api and api_available:
                result, error = predict_via_api(features)
            else:
                result = predict_locally(features, model, scaler, selector)
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
