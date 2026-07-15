"""Data loading and preprocessing for the Pima diabetes dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    target_column = "Outcome"
    zero_as_missing_columns = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

    def __init__(self, config: dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        required = set(self.zero_as_missing_columns) | {"Pregnancies", "DiabetesPedigreeFunction", "Age", self.target_column}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        self.logger.info("Loaded data with shape %s", df.shape)
        return df.copy()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        # In this dataset zero represents an unavailable measurement for these fields.
        for column in self.zero_as_missing_columns:
            result[column] = result[column].replace(0, np.nan)
            result[column] = result[column].fillna(result[column].median())
        for column in result.columns:
            if result[column].isna().any():
                result[column] = result[column].fillna(result[column].median())
        self.logger.info("Imputed missing clinical measurements with column medians")
        return result

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
