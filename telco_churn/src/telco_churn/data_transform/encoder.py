import pandas as pd

from .base_transformer import BaseTransformer


class CategoricalEncoder(BaseTransformer):
    def __init__(self, columns=None):
        """
        Initialize the encoder.
        Args:
            columns (list or None): Columns to encode. If None, all categorical columns are encoded.
        """
        self.columns = columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            self.columns = data.select_dtypes(include=["object"]).columns

        for column in self.columns:
            if data[column].nunique() == 2:  # Binary encoding
                data[column] = data[column].map(
                    {value: idx for idx, value in enumerate(data[column].unique())}
                )
            else:  # One-hot encoding
                dummies = pd.get_dummies(data[column], prefix=column, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                data.drop(column, axis=1, inplace=True)

        return data
