import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGB


class BaselineComparison:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = {}
    def logistic_regression(self,X_train:np.ndarray,y_train:np.ndarray) -> Any:
        self.logger.info("Training Logistic Regression")
        model = LogisticRegression(random_state=42,max_iter=1000)
        model.fit(X_train, y_train)
        return model
    def random_forest(self,X_train:np.ndarray,y_train:np.ndarray) -> Any:
        self.logger.info("Training Random Forest")
        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X_train, y_train)
        return model
    def svm(self,X_train:np.ndarray,y_train:np.ndarray) -> Any:
        self.logger.info("Training Support Vector Machine")
        model = SVC(probability=True,random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def default_xgboost(self,X_train:np.ndarray,y_train:np.ndarray) -> Any:
        self.logger.info("Training Default XGBoost Model")
        model = XGB(random_state=42,eval_metric="logloss",use_label_encoder=False)
        model.fit(X_train, y_train)
        return model
    def train_grid_search_xgboost(self,X_train: np.ndarray,y_train: np.ndarray) -> Any:
        self.logger.info("Training Grid Search XGBoost")
        param_grid = {'max_depth': [3, 5, 7],'learning_rate': [0.01, 0.1, 0.2],'n_estimators': [100, 200, 300],'min_child_weight': [1, 3, 5]}
        model = xgb.XGBClassifier(random_state=42,use_label_encoder=False,eval_metric='logloss')
        grid_search = GridSearchCV(model,param_grid,cv=self.config['baseline']['grid_search_cv'],scoring='roc_auc',n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.logger.info(f"Best Grid Search params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    def train_random_search_xgboost(self,X_train: np.ndarray,y_train: np.ndarray) -> Any:
        self.logger.info("Training Random Search XGBoost")
        param_distributions = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}    
        model = xgb.XGBClassifier(random_state=42,use_label_encoder=False,eval_metric='logloss')
        random_search = RandomizedSearchCV(model,param_distributions,n_iter=self.config['baseline']['random_search_iterations'],cv=5,scoring='roc_auc',n_jobs=-1,random_state=42)
        random_search.fit(X_train, y_train)
        self.logger.info(f"Best Random Search params: {random_search.best_params_}")
        return random_search.best_estimator_
    def train_optuna_xgboost(self,X_train: np.ndarray,y_train: np.ndarray) -> Any:
        self.logger.info("Training Optuna XGBoost")
        def objective(trial):
            params = {'max_depth': trial.suggest_int('max_depth', 3, 10),'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),'n_estimators': trial.suggest_int('n_estimators', 50, 500),'gamma': trial.suggest_float('gamma', 0, 5),'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),'subsample': trial.suggest_float('subsample', 0.5, 1.0),'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),'random_state': 42,'use_label_encoder': False,'eval_metric': 'logloss'}
            model = xgb.XGBClassifier(**params)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train,cv=5, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['baseline']['optuna_trials'])
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        self.logger.info(f"Best Optuna params: {best_params}")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        return model
    def run_comparison(self,X_train: np.ndarray,y_train: np.ndarray,X_test: np.ndarray,y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        models_to_test = self.config['baseline']['models']
        for model_name in models_to_test:
            try:
                if model_name == 'logistic_regression':
                    model = self.train_logistic_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    model = self.train_random_forest(X_train, y_train)
                elif model_name == 'svm':
                    model = self.train_svm(X_train, y_train)
                elif model_name == 'default_xgboost':
                    model = self.train_default_xgboost(X_train, y_train)
                elif model_name == 'grid_search_xgboost':
                    model = self.train_grid_search_xgboost(X_train, y_train)
                elif model_name == 'random_search_xgboost':
                    model = self.train_random_search_xgboost(X_train, y_train)
                elif model_name == 'optuna_xgboost':
                    model = self.train_optuna_xgboost(X_train, y_train)
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics = {'accuracy': accuracy_score(y_test, y_pred),'precision': precision_score(y_test, y_pred),'recall': recall_score(y_test, y_pred),'f1': f1_score(y_test, y_pred),'roc_auc': roc_auc_score(y_test, y_proba[:, 1])}
                self.results[model_name] = {'model': model,'metrics': metrics,'predictions': y_pred,'probabilities': y_proba}
                self.logger.info(f"{model_name} - ROC-AUC: {metrics['roc_auc']:.4f}")
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
        self.logger.info("Baseline comparison completed")
        return self.results
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        if not self.results:
            raise ValueError("No results available. Run comparison first.")
        metrics_only = {}
        for model_name, result in self.results.items():
            metrics_only[model_name] = result['metrics']
        return metrics_only
    