"""Integration smoke test for the stacking ensemble."""

from pathlib import Path

from src.feature_selection.feature_selector import FeatureSelector
from src.models.ensemble import EnsembleModel
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.config import load_config
from src.utils.logger import setup_logger


def test_ensemble_training_evaluation_and_save(tmp_path: Path):
    config = load_config("config.yaml")
    logger = setup_logger("ensemble_test", level="ERROR")
    preprocessor = DataPreprocessor(config, logger)
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess(
        config["data"]["raw_path"], config["data"]["processed_path"]
    )
    selector = FeatureSelector(config, logger)
    selected, X_train_selected = selector.select_features(X_train, y_train, preprocessor.get_feature_names())
    indices = [preprocessor.get_feature_names().index(feature) for feature in selected]
    ensemble = EnsembleModel(config, logger)
    ensemble.fit(X_train_selected, y_train)
    metrics = ensemble.evaluate(X_test[:, indices], y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    output = tmp_path / "ensemble_model.joblib"
    ensemble.save(output)
    assert output.is_file()
