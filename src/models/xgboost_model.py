from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(self, config: dict[str, Any], logger: Any):
        self.config, self.logger = config, logger
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] | None = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any] | None = None,
              feature_names: list[str] | None = None, eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
              early_stopping_rounds: int | None = None) -> XGBClassifier:
        self.feature_names = feature_names
        defaults = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "random_state": 42,
            "gamma": 0,
            "subsample": 0.8,
            "min_child_weight": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        if early_stopping_rounds is not None:
            defaults["early_stopping_rounds"] = early_stopping_rounds
        defaults.update(params or {})
        self.model = XGBClassifier(**defaults)
        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        self.model.fit(X_train, y_train, **fit_params)
        return self.model

    def predict(self, X):
        if self.model is None: raise ValueError("Model has not been trained")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None: raise ValueError("Model has not been trained")
        return self.model.predict_proba(X)

    def save_model(self, save_path: str) -> None:
        if self.model is None: raise ValueError("Model has not been trained")
        path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info("XGBoost model saved to %s", path)

    def load_model(self, load_path: str) -> XGBClassifier:
        self.model = joblib.load(load_path)
        return self.model

    def get_feature_importance(self, importance_type: str = "weight"):
        if self.model is None: raise ValueError("Model has not been trained")
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        if not self.feature_names: return importance
        return {name: float(importance.get(f"f{i}", 0)) for i, name in enumerate(self.feature_names)}

    def get_model_params(self):
        if self.model is None: raise ValueError("Model has not been trained")
        return self.model.get_params()
