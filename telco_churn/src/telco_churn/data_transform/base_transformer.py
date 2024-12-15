from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """
    Abstract base class for all transformers.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the transformer to the data.
        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series, optional): Target data. Default is None.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data.
        Args:
            X (pd.DataFrame): Feature data to transform.
        Returns:
            pd.DataFrame: Transformed data.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits the transformer and then transforms the data.
        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series, optional): Target data. Default is None.
        Returns:
            pd.DataFrame: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)
