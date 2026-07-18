# Diffusion-Enhanced TSO-HBA Optimized XGBoost Framework for Early Diabetes Prediction

A research-oriented machine learning framework that combines tabular diffusion-based data augmentation, hybrid Tuna Swarm Optimization–Honey Badger Algorithm (TSO-HBA), feature selection, and XGBoost classification for early diabetes prediction.

## Features

- **Tabular Diffusion Model**: Generates synthetic patient data to augment training sets
- **Hybrid TSO-HBA Optimization**: Custom metaheuristic algorithm for XGBoost hyperparameter tuning
- **Multi-Method Feature Selection**: Correlation filtering, Mutual Information, RFE, and XGBoost importance
- **Comprehensive Evaluation**: ROC-AUC, Precision, Recall, F1-score, Confusion Matrix
- **SHAP Explainability**: Global and local model interpretation
- **Baseline Comparisons**: Compare against Logistic Regression, Random Forest, SVM, and standard XGBoost
- **FastAPI Backend**: RESTful API for predictions
- **Streamlit Frontend**: Interactive web interface for predictions and explanations
- **Docker Support**: Containerized deployment

## 📂 Project Structure

```
diabetes_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── outputs/
│   ├── plots/
│   ├── models/
│   └── reports/
│
├── requirements.txt
├── config.yaml
└── README.md
```


## Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd diabetes-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the Pima Indians Diabetes dataset as `data/raw/diabetes.csv`

## Usage

### Running the Complete Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Generate synthetic data using diffusion model
4. Select optimal features
5. Optimize XGBoost hyperparameters using TSO-HBA
6. Train the model
7. Evaluate performance
8. Generate SHAP explanations
9. Compare with baseline models

### Running the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /predict` - Make predictions
- `GET /health` - Health check
- `GET /model-info` - Model information

### Running the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

### Docker Deployment

```bash
docker-compose up --build
```

## Dataset

The project uses the Pima Indians Diabetes Dataset with the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target: Outcome (0 = Non-diabetic, 1 = Diabetic)

## Research Comparisons

The framework supports experimental comparison between:

1. Baseline XGBoost
2. TSO-HBA Optimized XGBoost
3. Diffusion-Augmented XGBoost
4. Proposed Diffusion + TSO-HBA Optimized XGBoost

## Configuration

Edit `config.yaml` to customize:
- Model hyperparameters
- Optimization settings
- Data paths
- Logging configuration

## Testing

Run unit tests:
```bash
pytest tests/
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{diabetes_predictor,
  title={Diffusion-Enhanced TSO-HBA Optimized XGBoost Framework for Early Diabetes Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diabetes-predictor}
}
```

## License

MIT License

## Acknowledgments

- Pima Indians Diabetes Dataset from National Institute of Diabetes and Digestive and Kidney Diseases
- XGBoost library
- SHAP library for model explainability
