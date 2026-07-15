import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from typing import Dict, Any, List, Tuple
import logging
import joblib
from pathlib import Path
class EnsembleModel:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
    def create_base_models(self) -> Dict[str, Any]:
        self.logger.info("Creating base models")
        models = {
            'xgboost': xgb.XGBClassifier(max_depth=8,learning_rate=0.29,n_estimators=300,gamma=4.80,min_child_weight=7,subsample=0.98,colsample_bytree=1.0,reg_alpha=1.82,reg_lambda=4.18, scale_pos_weight=1.44,random_state=42,eval_metric='logloss', n_jobs=1),
            'random_forest': RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,min_samples_split=5,min_samples_leaf=2,random_state=42)}
        self.base_models = models
        return models
    def train_base_models(self,X_train: np.ndarray,y_train: np.ndarray) -> Dict[str, np.ndarray]:
        self.logger.info("Training base models")
        predictions = {}
        for name, model in self.base_models.items():
            self.logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            cv_preds = cross_val_predict(model, X_train, y_train,cv=5,method='predict_proba',n_jobs=1
            )[:, 1]
            predictions[name] = cv_preds
            self.logger.info(f"{name} CV AUC: {np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')):.4f}")
        return predictions
    def train_meta_model(self,predictions: Dict[str, np.ndarray],y_train: np.ndarray) -> None:
        self.logger.info("Training meta-model")
        meta_features = np.column_stack(list(predictions.values()))
        self.meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        self.meta_model.fit(meta_features, y_train)
        self.logger.info("Meta-model trained")
    def fit(self,X_train: np.ndarray,y_train: np.ndarray) -> None:
        self.logger.info("Training ensemble model")
        self.create_base_models()
        predictions = self.train_base_models(X_train, y_train)
        self.train_meta_model(predictions, y_train)
        for name, model in self.base_models.items():
            self.logger.info(f"Retraining {name} on full data")
            model.fit(X_train, y_train)
        self.logger.info("Ensemble model training completed")
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.meta_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        base_preds = []
        for name, model in self.base_models.items():
            preds = model.predict_proba(X)[:, 1]
            base_preds.append(preds)
        meta_features = np.column_stack(base_preds)
        ensemble_preds = self.meta_model.predict_proba(meta_features)[:, 1]
        return ensemble_preds
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    def evaluate(self,X_test: np.ndarray,y_test: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        metrics = {'accuracy': accuracy_score(y_test, y_pred),'precision': precision_score(y_test, y_pred),'recall': recall_score(y_test, y_pred),'f1': f1_score(y_test, y_pred),'roc_auc': roc_auc_score(y_test, y_proba)}
        return metrics
    def save(self, save_path: str) -> None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        model_data = {'base_models': self.base_models,'meta_model': self.meta_model}
        joblib.dump(model_data, save_path)
        self.logger.info(f"Ensemble model saved to {save_path}")
    def load(self, load_path: str) -> None:
        model_data = joblib.load(load_path)
        self.base_models = model_data['base_models']
        self.meta_model = model_data['meta_model']
        self.logger.info(f"Ensemble model loaded from {load_path}")
