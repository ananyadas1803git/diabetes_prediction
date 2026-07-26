import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any
import logging
class SHAPExplainer:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.save_path = Path("results")  # Changed to save to results folder
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.explainer = None
        self.shap_values = None
        self.feature_names = None 
    def fit(self,model,X_train: np.ndarray,feature_names: Optional[list] = None) -> None:
        self.logger.info("Fitting SHAP explainer")
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        num_samples = self.config['explainability']['num_samples']
        if len(X_train) > num_samples:
            X_sample = X_train[:num_samples]
        else:
            X_sample = X_train
        self._sampled_data = X_sample
        values = self.explainer.shap_values(X_sample)
        # Newer SHAP versions may return one array per class or a 3-D array.
        if isinstance(values, list):
            values = values[-1]
        if np.ndim(values) == 3:
            values = values[:, :, -1]
        self.shap_values = np.asarray(values)
        self.logger.info("SHAP explainer fitted")
    def plot_summary(self, save_name: str = "shap_summary.png") -> None:
        if self.shap_values is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        self.logger.info("Generating SHAP summary plot")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, features=self._sampled_data, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        save_file = self.save_path / save_name
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved SHAP summary plot to {save_file}")
    def plot_feature_importance(self, save_name: str = "shap_feature_importance.png") -> None:
        if self.shap_values is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        self.logger.info("Generating SHAP feature importance plot")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, features=self._sampled_data, feature_names=self.feature_names, plot_type="bar",show=False)
        plt.tight_layout()
        save_file = self.save_path / save_name
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved SHAP feature importance plot to {save_file}")
    def plot_waterfall(self,instance_idx: int = 0,save_name: Optional[str] = None) -> None:
        if self.shap_values is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        if instance_idx >= len(self.shap_values):
            raise ValueError(f"Instance index {instance_idx} out of range")
        self.logger.info(f"Generating SHAP waterfall plot for instance {instance_idx}")
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(values=self.shap_values[instance_idx],base_values=self.explainer.expected_value,feature_names=self.feature_names),show=False)
        plt.tight_layout()
        if save_name is None:
            save_name = f"shap_waterfall_{instance_idx}.png"
        save_file = self.save_path / save_name
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved SHAP waterfall plot to {save_file}")
    def plot_force(self,instance_idx: int = 0,save_name: Optional[str] = None) -> None:
        if self.shap_values is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        if instance_idx >= len(self.shap_values):
            raise ValueError(f"Instance index {instance_idx} out of range") 
        self.logger.info(f"Generating SHAP force plot for instance {instance_idx}")
        if save_name is None:
            save_name = f"shap_force_{instance_idx}.html"
        save_file = self.save_path / save_name
        shap.force_plot(self.explainer.expected_value,self.shap_values[instance_idx],feature_names=self.feature_names,show=False,matplotlib=False)
        shap.save_html(str(save_file), plt.gcf())
        plt.close()
        self.logger.info(f"Saved SHAP force plot to {save_file}")
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        if self.shap_values is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        self.logger.info("Computing feature importance ranking")
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({'feature': self.feature_names if self.feature_names else [f'f{i}' for i in range(len(mean_shap))],'mean_shap_value': mean_shap}).sort_values('mean_shap_value', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        return importance_df
    def explain_instance(self,X_instance: np.ndarray,instance_idx: int = 0) -> Dict[str, Any]:
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        self.logger.info(f"Explaining instance {instance_idx}")
        shap_values = self.explainer.shap_values(X_instance.reshape(1, -1))
        explanation = {'base_value': float(self.explainer.expected_value),'shap_values': shap_values[0].tolist(),'feature_names': self.feature_names,'prediction': float(self.explainer.expected_value + shap_values[0].sum())}
        return explanation
    def generate_all_plots(self) -> None:
        self.logger.info("Generating all SHAP plots")
        self.plot_summary()
        self.plot_feature_importance()
        for i in range(min(3, len(self.shap_values))):
            self.plot_waterfall(instance_idx=i)
        self.logger.info("All SHAP plots generated")
