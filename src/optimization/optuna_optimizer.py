import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from typing import Dict, Any
import logging


class OptunaXGBoostOptimizer:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cv_folds = config['optimization'].get('cv_folds', 5)
        self.cv_repeats = config['optimization'].get('repeated_cv_repeats', 1)
        self.optuna_trials = config['optimization'].get('optuna_trials', 20)
        self.search_space = config['optimization']['search_space']
        self.objective_metric = config['optimization'].get('objective_metric', 'roc_auc')
        self.random_state = config['data']['random_state']

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            if spec['type'] == 'int':
                params[name] = trial.suggest_int(name, spec['min'], spec['max'])
            elif spec['type'] == 'float':
                params[name] = trial.suggest_float(name, spec['min'], spec['max'])
            else:
                raise ValueError(f"Unsupported parameter type for {name}: {spec['type']}")
        params['objective'] = 'binary:logistic'
        params['random_state'] = self.random_state
        params['eval_metric'] = ['logloss', 'auc']
        params['verbosity'] = 0
        params['n_jobs'] = -1
        return params

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        params = self._suggest_parameters(trial)
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        if pos_count > 0:
            params['scale_pos_weight'] = float(neg_count) / pos_count
        else:
            params['scale_pos_weight'] = 1.0

        model = xgb.XGBClassifier(**params)
        cv = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=self.cv_repeats, random_state=self.random_state) if self.cv_repeats > 1 else StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=self.objective_metric,
            n_jobs=-1,
        )
        return float(np.mean(scores))

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Starting Optuna hyperparameter optimization")

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=self.optuna_trials)

        best_params = study.best_params
        best_params['objective'] = 'binary:logistic'
        best_params['random_state'] = self.random_state
        if self.objective_metric == 'average_precision':
            best_params['eval_metric'] = ['logloss', 'aucpr']
        else:
            best_params['eval_metric'] = ['logloss', 'auc']
        best_params['verbosity'] = 0
        best_params['n_jobs'] = -1
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        best_params['scale_pos_weight'] = float(neg_count) / pos_count if pos_count > 0 else 1.0

        self.logger.info(f"Optuna completed. Best {self.objective_metric}: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        return best_params
