"""Data loading and preprocessing for the diabetes dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, config: dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None
        self.target_column: str | None = config["data"].get("target_column")
        self.zero_as_missing_columns: tuple[str, ...] = tuple(config["preprocessing"].get("zero_as_missing_columns", []))
        if not self.zero_as_missing_columns:
            self.zero_as_missing_columns = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

    def resolve_target_column(self, df: pd.DataFrame) -> str:
        if self.target_column:
            return self.target_column
        if "Outcome" in df.columns:
            return "Outcome"
        if "Diabetes_binary" in df.columns:
            return "Diabetes_binary"
        raise ValueError("Could not determine target column. Set 'data.target_column' in the config.")

    def load_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        self.target_column = self.resolve_target_column(df)
        if not self.zero_as_missing_columns:
            default_columns = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
            self.zero_as_missing_columns = tuple(c for c in default_columns if c in df.columns)
        else:
            self.zero_as_missing_columns = tuple(c for c in self.zero_as_missing_columns if c in df.columns)

        required = {self.target_column}
        if self.config["preprocessing"].get("required_features"):
            required |= set(self.config["preprocessing"]["required_features"])
        elif self.target_column == "Outcome":
            required |= set(self.zero_as_missing_columns) | {"Pregnancies", "DiabetesPedigreeFunction", "Age"}

        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        self.logger.info("Loaded data with shape %s", df.shape)
        return df.copy()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in self.zero_as_missing_columns:
            result[column] = result[column].replace(0, np.nan)
            result[column] = result[column].fillna(result[column].median())
        for column in result.columns:
            if result[column].isna().any():
                result[column] = result[column].fillna(result[column].median())
        self.logger.info("Imputed missing values with column medians")
        return result

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config["preprocessing"].get("feature_engineering", False):
            return df

        result = df.copy()
        if "BMI" in result.columns:
            result["BMI_category"] = pd.cut(
                result["BMI"],
                bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
                labels=[0, 1, 2, 3],
            ).astype(int)
        if "Age" in result.columns:
            result["Age_group"] = pd.cut(
                result["Age"],
                bins=[-np.inf, 30, 50, 65, np.inf],
                labels=[0, 1, 2, 3],
            ).astype(int)
        if set(["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"]).issubset(result.columns):
            result["metabolic_risk"] = (
                result[["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"]].sum(axis=1)
            )
        if set(["Smoker", "HvyAlcoholConsump", "DiffWalk"]).issubset(result.columns):
            result["lifestyle_risk"] = (
                result[["Smoker", "HvyAlcoholConsump", "DiffWalk"]].sum(axis=1)
            )
        if set(["PhysActivity", "Fruits", "Veggies"]).issubset(result.columns):
            result["healthy_habits"] = (
                result[["PhysActivity", "Fruits", "Veggies"]].sum(axis=1)
            )
        if set(["Age", "BMI"]).issubset(result.columns):
            result["age_bmi_interaction"] = result["Age"] * result["BMI"]

        self.logger.info("Applied feature engineering and created derived features")
        return result

    def rebalance_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strategy = self.config["data"].get("rebalance_strategy")
        if strategy != "oversample_minority":
            return X, y

        target_ratio = float(self.config["data"].get("oversample_target_ratio", 1.0))
        if target_ratio <= 0:
            return X, y

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return X, y

        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        majority_count = int(counts.max())
        minority_count = int(counts.min())
        desired_minority = int(np.ceil(majority_count * target_ratio))

        if minority_count >= desired_minority:
            self.logger.info(
                "Minority class already meets target ratio (%s). No oversampling applied.",
                target_ratio,
            )
            return X, y

        minority_indices = np.where(y == minority_class)[0]
        rng = np.random.default_rng(self.config["data"]["random_state"])
        additional_indices = rng.choice(minority_indices, desired_minority - minority_count, replace=True)
        X_resampled = np.vstack([X, X[additional_indices]])
        y_resampled = np.hstack([y, y[additional_indices]])
        self.logger.info(
            "Oversampled minority class from %d to %d samples to achieve ratio %.2f",
            minority_count,
            desired_minority,
            target_ratio,
        )
        return X_resampled, y_resampled

    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        if method not in {"iqr", "zscore"}:
            raise ValueError("method must be 'iqr' or 'zscore'")
        numeric_columns = [column for column in df.select_dtypes(include=np.number).columns if column != self.target_column]
        counts: dict[str, int] = {}
        for column in numeric_columns:
            series = df[column]
            if method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            else:
                std = series.std()
                mask = pd.Series(False, index=df.index) if std == 0 else ((series - series.mean()).abs() / std > 3)
            counts[column] = int(mask.sum())
        self.logger.info("Detected outliers (%s): %s", method, counts)
        return df.copy()

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        result = df.copy()
        if not self.config["preprocessing"]["scale_features"]:
            return result
        columns = [column for column in result.select_dtypes(include=np.number).columns if column != self.target_column]
        if fit:
            self.scaler = StandardScaler().fit(result[columns].to_numpy())
        if self.scaler is None:
            raise ValueError("Scaler is not fitted; call scale_features(..., fit=True) first")
        result[columns] = self.scaler.transform(result[columns].to_numpy())
        return result

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[self.target_column])

    def preprocess(self, input_path: str, output_path: str | None = None):
        df = self.load_data(input_path)
        if self.config["preprocessing"]["handle_missing"]:
            df = self.handle_missing_values(df)
        if self.config["preprocessing"]["detect_outliers"]:
            self.detect_outliers(df, self.config["preprocessing"]["outlier_method"])
        df = self.apply_feature_engineering(df)
        self.feature_columns = [column for column in df.columns if column != self.target_column]
        train_df, test_df = self.split_data(df, 1 - self.config["data"]["train_test_split"], self.config["data"]["random_state"])
        train_df = self.scale_features(train_df, fit=True)
        test_df = self.scale_features(test_df, fit=False)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([train_df, test_df]).to_csv(path, index=False)
        return (train_df[self.feature_columns].to_numpy(), test_df[self.feature_columns].to_numpy(),
                train_df[self.target_column].to_numpy(), test_df[self.target_column].to_numpy(), train_df, test_df)

    def get_feature_names(self) -> list[str]:
        if self.feature_columns is None:
            raise ValueError("Data has not been preprocessed yet")
        return self.feature_columns.copy()
