import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
from pathlib import Path


class ModelEvaluator:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.plots_path = Path("results")  # Changed to save to results folder
        self.plots_path.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    @staticmethod
    def _extract_positive_proba(y_proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(y_proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        positive_probability = self._extract_positive_proba(y_proba)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, positive_probability),
        }

    def find_best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, method: str = "youden", beta: float = 1.0) -> float:
        probabilities = y_proba[:, 1]
        if method == "youden":
            fpr, tpr, thresholds = roc_curve(y_true, probabilities)
            j_scores = tpr - fpr
            best_index = int(np.argmax(j_scores))
            return float(thresholds[best_index])

        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        if method == "f1":
            f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-8, None)
            best_index = int(np.nanargmax(f1_scores[:-1]))
            return float(thresholds[best_index])

        if method == "precision_recall":
            f1_scores = (1 + beta**2) * (precision * recall) / np.clip((beta**2 * precision + recall), 1e-8, None)
            best_index = int(np.nanargmax(f1_scores[:-1]))
            return float(thresholds[best_index])

        raise ValueError(f"Unsupported threshold method: {method}")

    def metrics_at_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        return self.compute_metrics(y_true, y_pred, y_proba)

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        for name, value in metrics.items():
            self.logger.info("%s: %.4f", name, value)
        if self.config["evaluation"].get("save_plots", True):
            self.plot_confusion_matrix(y_true, y_pred, model_name)
            self.plot_roc_curve(y_true, y_proba, model_name)
            self.plot_precision_recall_curve(y_true, y_proba, model_name)
        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{model_name} Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.plots_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model") -> None:
        proba = self._extract_positive_proba(y_proba)
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_score = roc_auc_score(y_true, proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC Curve - {model_name}")
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_path / f"roc_curve_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model") -> None:
        proba = self._extract_positive_proba(y_proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        avg_precision = average_precision_score(y_true, proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP = {avg_precision:.4f})")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curve - {model_name}")
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_path / f"precision_recall_curve_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def compare_model(self, results: Dict[str, dict[str, float]], save_path: Optional[str] = None) -> pd.DataFrame:
        comparison_df = pd.DataFrame(results).T
        sort_metric = "roc_auc" if "roc_auc" in comparison_df.columns else "ROC-AUC" if "ROC-AUC" in comparison_df.columns else comparison_df.columns[0]
        comparison_df = comparison_df.sort_values(sort_metric, ascending=False)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path)
        return comparison_df

    def compare_models(self, results: Dict[str, dict[str, float]], save_path: Optional[str] = None) -> pd.DataFrame:
        return self.compare_model(results, save_path)

    def plot_comparison(self, results: Dict[str, dict[str, float]], metric: str = "roc_auc") -> None:
        values = {name: item.get("metrics", item).get(metric) for name, item in results.items()}
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(values.keys(), values.values(), color="steelblue", alpha=0.8)
        ax.set(xlabel="Models", ylabel=metric.replace('_', ' ').title(), title="Model Comparison")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(self.plots_path / f"comparison_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_multiple_roc_curves(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, (y_true, y_proba) in results.items():
            proba = self._extract_positive_proba(y_proba)
            fpr, tpr, _ = roc_curve(y_true, proba)
            ax.plot(fpr, tpr, linewidth=2, label=name)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plots_path / "roc_curve_multiple_models.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


Evaluator = ModelEvaluator
