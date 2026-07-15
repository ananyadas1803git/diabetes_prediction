import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_selection.feature_selector import FeatureSelector
from src.utils.config import load_config
@pytest.fixture
def config():
    return load_config("config.yaml")
@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = np.random.randint(0, 2, 100)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    return X, y, feature_names
@pytest.fixture
def feature_selector(config):
    from src.utils.logger import setup_logger
    logger = setup_logger("test", level="ERROR")
    return FeatureSelector(config, logger)
def test_feature_selector_initialization(feature_selector):
    assert feature_selector is not None
    assert feature_selector.config is not None
def test_correlation_filtering(feature_selector, sample_data):
    X, y, feature_names = sample_data
    selected = feature_selector.correlation_filtering(X, feature_names, threshold=0.95)
    assert isinstance(selected, list)
    assert len(selected) <= len(feature_names)
def test_mutual_information_selection(feature_selector, sample_data):
    X, y, feature_names = sample_data
    selected = feature_selector.mutual_information_selection(X, y, feature_names, n_features=5)
    assert isinstance(selected, list)
    assert len(selected) == 5
def test_rfe_selection(feature_selector, sample_data):
    X, y, feature_names = sample_data
    selected = feature_selector.rfe_selection(X, y, feature_names, n_features=5)
    assert isinstance(selected, list)
    assert len(selected) == 5
def test_xgboost_importance_selection(feature_selector, sample_data):
    X, y, feature_names = sample_data
    selected = feature_selector.xgboost_importance_selection(X, y, feature_names, n_features=5)
    assert isinstance(selected, list)
    assert len(selected) == 5
def test_ensemble_selection(feature_selector, sample_data):
    X, y, feature_names = sample_data
    selected, X_selected = feature_selector.select_features(X, y, feature_names, method='ensemble')  
    assert isinstance(selected, list)
    assert X_selected.shape[1] <= X.shape[1]
    assert X_selected.shape[0] == X.shape[0]
def test_generate_feature_report(feature_selector, sample_data):
    X, y, feature_names = sample_data
    feature_selector.select_features(X, y, feature_names, method='ensemble')
    report = feature_selector.generate_feature_report()
    assert isinstance(report, pd.DataFrame)
    assert 'feature' in report.columns
    assert 'selected' in report.columns