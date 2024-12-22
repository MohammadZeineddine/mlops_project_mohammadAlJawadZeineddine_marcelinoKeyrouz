import pandas as pd
from sklearn.impute import SimpleImputer

from .base_transformer import BaseTransformer


class DataImputer(BaseTransformer):
    """
    **Data Imputer Transformer**

    This transformer class handles missing data imputation using various strategies.
    It leverages the scikit-learn SimpleImputer class to fill missing values
    based on the chosen strategy.

    **Attributes:**

    * `strategy (str)`: The imputation strategy to be used ('mean', 'median', 'most_frequent', or 'constant').

    **Methods:**

    * **__init__(self, strategy="mean")**
        - Initializes the imputer with the specified imputation strategy.

        Args:
            strategy (str): The strategy for imputing missing values. Defaults to 'mean'.

    * **fit(self, X: pd.DataFrame, y: pd.Series = None):**
        - Fits the imputer to the data to learn the statistics
            needed for imputation based on the chosen strategy.

        Args:
            X (pd.DataFrame): The feature data containing missing values.
            y (pd.Series, optional): The target data (ignored). Default is None.

    * **transform(self, X: pd.DataFrame) -> pd.DataFrame:**
        - Imputes missing values in the data using the fitted imputer.

        Args:
            X (pd.DataFrame): The feature data to be transformed (containing missing values).

        Returns:
            pd.DataFrame: The transformed DataFrame with imputed missing values.
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
        X = X.apply(pd.to_numeric, errors="coerce")
        transformed = self.imputer.transform(X)
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)
