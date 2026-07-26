"""Exploratory plots for the training split."""
from pathlib import Path
from typing import Any

import matplotlib

# The pipeline runs without a display server (for example in CI or from a terminal).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAAnalyzer:
    def __init__(self, config: dict[str, Any], logger: Any):
        self.logger = logger
        self.target_column = config["data"].get("target_column", "Outcome")
        self.plots_path = Path("results")  # Changed to save to results folder
        self.plots_path.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str):
        plt.tight_layout(); plt.savefig(self.plots_path / name, dpi=150, bbox_inches="tight"); plt.close()

    def correlation_heatmap(self, df):
        plt.figure(figsize=(10, 8)); sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0); self._save("correlation_heatmap.png")

    def class_distribution(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); counts = df[self.target_column].value_counts().sort_index()
        sns.countplot(data=df, x=self.target_column, ax=axes[0]); axes[0].set_title("Class distribution")
        axes[1].pie(counts, labels=["Non-diabetic", "Diabetic"], autopct="%1.1f%%"); self._save("class_distribution.png")

    def histogram(self, df):
        numeric = [c for c in df.select_dtypes("number") if c != self.target_column]; df[numeric].hist(figsize=(15, 10), bins=25); self._save("histograms.png")

    def pairplot(self, df):
        columns = [c for c in df.select_dtypes("number") if c != self.target_column][:5] + [self.target_column]
        grid = sns.pairplot(df.sample(min(len(df), 500), random_state=42)[columns], hue=self.target_column); grid.savefig(self.plots_path / "pairplot.png", dpi=150); plt.close("all")

    def generate_missing_value_analysis(self, df):
        missing = df.isna().sum(); plt.figure(figsize=(10, 5)); missing.plot.bar(); plt.ylabel("Missing values"); self._save("missing_values.png")

    def boxplot(self, df):
        numeric = [c for c in df.select_dtypes("number") if c != self.target_column]; df[numeric].plot.box(figsize=(15, 7), rot=45); self._save("boxplot.png")

    def statistical_summary(self, df):
        summary = df.describe(); summary.to_csv(self.plots_path / "statistical_summary.csv"); return summary

    def run_full_eda(self, df):
        for function in (self.correlation_heatmap, self.class_distribution, self.histogram, self.pairplot, self.generate_missing_value_analysis, self.boxplot): function(df)
        self.statistical_summary(df)


EdaAnalyser = EDAAnalyzer
