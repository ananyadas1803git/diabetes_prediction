"""Baseline estimators used for optional pipeline comparisons."""
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


class BaselineComparison:
    def __init__(self, config: dict[str, Any], logger: Any): self.config, self.logger, self.results = config, logger, {}

    def _models(self):
        seed = self.config["data"]["random_state"]
        return {"logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
                "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                "svm": SVC(probability=True, random_state=seed),
                "default_xgboost": XGBClassifier(random_state=seed, eval_metric="logloss")}

    def run_comparison(self, X_train, y_train, X_test, y_test):
        candidates = self._models()
        for name in self.config["baseline"]["models"]:
            if name not in candidates:
                self.logger.warning("Unsupported baseline model: %s", name); continue
            model = candidates[name].fit(X_train, y_train); predictions = model.predict(X_test); probabilities = model.predict_proba(X_test)
            metrics = {"accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, zero_division=0), "recall": recall_score(y_test, predictions, zero_division=0), "f1": f1_score(y_test, predictions, zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities[:, 1])}
            self.results[name] = {"model": model, "metrics": metrics, "predictions": predictions, "probabilities": probabilities}
        return {name: result["metrics"] for name, result in self.results.items()}
