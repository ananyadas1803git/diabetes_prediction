import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.config import load_config
@pytest.fixture
def config():
    return load_config("config.yaml")
@pytest.fixture
def sample_data():
    np.random.seed(42)
    data = {'Pregnancies': np.random.randint(0, 10, 100),'Glucose': np.random.randint(70, 200, 100),'BloodPressure': np.random.randint(50, 100, 100),'SkinThickness': np.random.randint(0, 50, 100),'Insulin': np.random.randint(0, 300, 100),'BMI': np.random.uniform(18, 40, 100),'DiabetesPedigreeFunction': np.random.uniform(0, 2, 100),'Age': np.random.randint(20, 70, 100),'Outcome': np.random.randint(0, 2, 100)}
    return pd.DataFrame(data)
@pytest.fixture
def preprocessor(config):
    from src.utils.logger import setup_logger
    logger = setup_logger("test", level="ERROR")
    return DataPreprocessor(config, logger)
def test_data_preprocessor_initialization(preprocessor):
    assert preprocessor is not None
    assert preprocessor.config is not None
    assert preprocessor.target_column == "Outcome"
def test_handle_missing_values(preprocessor, sample_data):
    sample_data.loc[0:5, 'Glucose'] = 0
    sample_data.loc[10:15, 'BloodPressure'] = 0
    result = preprocessor.handle_missing_values(sample_data)
    assert result['Glucose'].iloc[0] != 0
    assert result['BloodPressure'].iloc[10] != 0
def test_detect_outliers(preprocessor, sample_data):
    result = preprocessor.detect_outliers(sample_data, method="iqr")
    assert result is not None
    assert len(result) == len(sample_data)
def test_split_data(preprocessor, sample_data):
    train_df, test_df = preprocessor.split_data(sample_data, test_size=0.2)
    assert len(train_df) + len(test_df) == len(sample_data)
    assert len(train_df) > len(test_df)
def test_scale_features(preprocessor, sample_data):
    preprocessor.config['preprocessing']['scale_features'] = True
    result = preprocessor.scale_features(sample_data, fit=True)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Outcome']
    for col in numeric_cols:
        assert abs(result[col].mean()) < 1.0
        assert abs(result[col].std() - 1.0) < 0.5