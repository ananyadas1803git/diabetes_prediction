"""Feature-selection methods and an ensemble vote."""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif


class FeatureSelector:
    def __init__(self, config: dict[str, Any], logger: Any):
        self.config, self.logger = config, logger
        self.feature_scores: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.selected_features: list[str] | None = None

    def correlation_filtering(self, X, feature_names, threshold=0.9):
        corr = pd.DataFrame(X, columns=feature_names).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        removed = [name for name in upper.columns if (upper[name] > threshold).any()]
        selected = [name for name in feature_names if name not in removed]
        self.feature_scores["correlation"] = {"removed": removed, "selected": selected}
        return selected

    def _count(self, n_features: int | None, feature_names: list[str]) -> int:
        count = n_features if n_features is not None else self.config["feature_selection"]["n_features_to_select"]
        if count is None or count == "all":
            return len(feature_names)
        if not isinstance(count, int):
            raise ValueError("n_features_to_select must be int, None, or 'all'")
        return min(count, len(feature_names))

    def mutual_information_selection(self, X, y, feature_names, n_features=None):
        scores = mutual_info_classif(X, y, random_state=self.config["data"]["random_state"])
        table = pd.DataFrame({"feature": feature_names, "mi_score": scores}).sort_values("mi_score", ascending=False)
        self.feature_scores["mutual_information"] = table.to_dict("records")
        return table.head(self._count(n_features, feature_names))["feature"].tolist()

    def rfe_selection(self, X, y, feature_names, n_features=None):
        selector = RFE(RandomForestClassifier(n_estimators=100, random_state=self.config["data"]["random_state"], n_jobs=1),
                       n_features_to_select=self._count(n_features, feature_names)).fit(X, y)
        table = pd.DataFrame({"feature": feature_names, "ranking": selector.ranking_}).sort_values("ranking")
        self.feature_scores["rfe"] = table.to_dict("records")
        return [feature_names[i] for i, chosen in enumerate(selector.support_) if chosen]

    def xgboost_importance_selection(self, X, y, feature_names, n_features=None):
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  random_state=self.config["data"]["random_state"], eval_metric="logloss", n_jobs=1)
        model.fit(X, y)
        table = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        self.feature_scores["xgboost_importance"] = table.to_dict("records")
        return table.head(self._count(n_features, feature_names))["feature"].tolist()

    def ensemble_selection(self, X, y, feature_names):
        methods = self.config["feature_selection"]["methods"]
        votes = dict.fromkeys(feature_names, 0)
        for method in methods:
            selected = getattr(self, f"{method}_selection", None)
            if method == "correlation":
                chosen = self.correlation_filtering(X, feature_names, self.config["feature_selection"]["correlation_threshold"])
            elif selected:
                chosen = selected(X, y, feature_names)
            else:
                self.logger.warning("Skipping unknown feature-selection method: %s", method)
                continue
            for feature in chosen:
                votes[feature] += 1
        cutoff = max(1, (len(methods) + 1) // 2)
        chosen = [feature for feature, votes_for_feature in votes.items() if votes_for_feature >= cutoff]
        self.feature_scores["ensemble_votes"] = votes
        return chosen or sorted(votes, key=votes.get, reverse=True)[:self._count(None, feature_names)]

    def select_features(self, X, y, feature_names, method="ensemble"):
        self.feature_names = list(feature_names)
        if method == "ensemble":
            chosen = self.ensemble_selection(X, y, self.feature_names)
        elif method == "correlation":
            chosen = self.correlation_filtering(X, self.feature_names, self.config["feature_selection"]["correlation_threshold"])
        else:
            chosen = getattr(self, f"{method}_selection")(X, y, self.feature_names)
        self.selected_features = chosen
        return chosen, X[:, [self.feature_names.index(feature) for feature in chosen]]

    def generate_feature_report(self):
        if self.selected_features is None:
            raise ValueError("Run select_features before generating a report")
        report = pd.DataFrame({"feature": self.feature_names})
        report["selected"] = report.feature.isin(self.selected_features)
        report["ensemble_votes"] = report.feature.map(self.feature_scores.get("ensemble_votes", {})).fillna(0).astype(int)
        return report

    def get_selected_feature_indices(self, feature_names):
        if self.selected_features is None:
            raise ValueError("Feature selection has not been performed")
        return [feature_names.index(feature) for feature in self.selected_features]


# Backward-compatible spelling used by earlier imports.
Feature_Selector = FeatureSelector
