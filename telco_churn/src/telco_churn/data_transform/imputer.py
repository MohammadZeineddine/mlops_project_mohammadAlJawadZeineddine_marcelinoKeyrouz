from sklearn.impute import SimpleImputer
from .base_transformer import BaseTransformer
import pandas as pd


class DataImputer(BaseTransformer):
    """
    Handles missing data imputation.
    """

    def __init__(self, strategy="mean"):
        """
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the imputer to the data.
        Args:
            X (pd.DataFrame): Feature data.
        """
        # Convert invalid values to NaN
        X = X.apply(pd.to_numeric, errors="coerce")
        self.imputer.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by imputing missing values.
        Args:
            X (pd.DataFrame): Feature data to transform.
        Returns:
            pd.DataFrame: Data with imputed values.
        """
        # Convert invalid values to NaN
        X = X.apply(pd.to_numeric, errors="coerce")
        transformed = self.imputer.transform(X)
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)
