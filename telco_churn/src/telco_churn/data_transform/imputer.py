# src/data_transform/imputer.py
import pandas as pd

from .base_transformer import BaseTransformer


class MissingValueImputer(BaseTransformer):
    def __init__(self, strategy="median", columns=None):
        """
        Initialize the imputer.
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'mode').
            columns (list or None): Columns to impute.
        """
        self.strategy = strategy
        self.columns = columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value imputation.
        Args:
            data (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: Data with imputed values.
        """
        if self.columns is None:
            self.columns = data.columns

        for column in self.columns:
            # Convert column to numeric, coercing errors to NaN
            data[column] = pd.to_numeric(data[column], errors="coerce")

            # Impute missing values based on the chosen strategy
            if self.strategy == "mean":
                value = data[column].mean()
            elif self.strategy == "median":
                value = data[column].median()
            elif self.strategy == "mode":
                value = data[column].mode()[0]
            else:
                raise ValueError(f"Invalid imputation strategy: {self.strategy}")

            # Fill missing values
            data[column] = data[column].fillna(value)

        return data
