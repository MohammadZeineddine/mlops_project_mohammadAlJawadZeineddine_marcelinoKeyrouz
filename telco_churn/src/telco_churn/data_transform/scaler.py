# src/data_transform/scaler.py
from sklearn.preprocessing import StandardScaler
import pandas as pd
from .base_transformer import BaseTransformer


class NumericalScaler(BaseTransformer):
    def __init__(self, columns=None):
        """
        Initialize the scaler.
        Args:
            columns (list or None): Columns to scale. If None, all numerical columns are scaled.
        """
        self.columns = columns
        self.scaler = StandardScaler()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        Args:
            data (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: Data with scaled numerical features.
        """
        if self.columns is None:
            self.columns = data.select_dtypes(include=["number"]).columns

        # Fit and transform the selected columns
        scaled_values = self.scaler.fit_transform(data[self.columns])

        # Assign scaled values back to the original DataFrame column by column
        for i, column in enumerate(self.columns):
            data[column] = scaled_values[:, i]

        return data
